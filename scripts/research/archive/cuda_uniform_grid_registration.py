"""Standalone original-double uniform-grid retrieval; no production selection.

NearestICP's CPU estimator, updates and convergence stay inherited unchanged.
GPU first/second minima retrieve candidates; CPU KDTreeFlann resolves ties,
metric boundaries, unsupported inputs and, by default, every complete miss.
Full CPU shadow audits invalidate timing attribution. Direct missing results
are a separately labelled research policy requiring a named proof artifact.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import hashlib
import math
import time
from collections import OrderedDict

import numpy as np
import open3d as o3d
from scanner_server.cuda_nn_registration import NearestICP
from scripts.research.archive.validate_uniform_grid_proof import GridProofAuthority

CUDA_SOURCE = Path(__file__).with_name("research_uniform_grid_nn.cu")
DOMAIN = {"version": "dyadic27-original-double-v1", "radius_m": "2^-20 <= radius <= 1",
    "cell_width": "smallest power of two >=radius; reversible finite ldexp scaling",
    "cells": "floor(scaled coordinate) in signed21-bit range [-1048576,1048575]",
    "packing": "bias each axis by1048576; x<<42|y<<21|z, no collisions in declared domain",
    "enclosure": "CPU strict squared-distance hit implies every absolute coordinate difference<cell_width; at most one floor cell apart",
    "distance": "original float64 points; explicit RN subtraction, multiply and left-associated add; --fmad=false; CuPy injects -ftz=true for single precision only",
    "uncertainty": "64epsilon relative metric margin for nearest/second/radius; CPU ties and boundary/unsupported fallback",
    "missing_policy": "original CPU tree unless explicitly research-certified and CPU hit/miss proof bound to helper/source/kernel/input"}


def dyadic_shift(radius):
    if not np.isfinite(radius) or not 2.**-20 <= radius <= 1.:
        return None
    mantissa, exponent = math.frexp(float(radius))
    power = exponent-1 if mantissa == .5 else exponent
    return -power


def host_cells(points, shift):
    if shift is None or not np.isfinite(points).all():
        return None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled = np.ldexp(points, shift)
        if not np.isfinite(scaled).all() or not np.array_equal(np.ldexp(scaled, -shift), points):
            return None
        cells = np.floor(scaled)
    if np.any(cells < -1048576) or np.any(cells > 1048575):
        return None
    return cells.astype(np.int64)


def packed_keys(cells):
    if cells.ndim != 2 or cells.shape[1] != 3 or np.any(cells < -1048576) or np.any(cells > 1048575):
        raise ValueError("Check signed21-bit cells before packing")
    biased = (cells + 1048576).astype(np.uint64)
    return (biased[:, 0] << np.uint64(42)) | (biased[:, 1] << np.uint64(21)) | biased[:, 2]


class UniformGridICP(NearestICP):
    def __init__(self, device="CUDA:0", max_clouds=64, max_cache_bytes=256*1024**2,
                 audit_nearest=False, audit_misses=False, miss_policy="cpu-fallback", proof_authority=None):
        self.device = o3d.core.Device(str(device))
        if not str(self.device).startswith("CUDA:") or max_clouds < 1 or max_cache_bytes < 1:
            raise ValueError("Require a CUDA device and positive bounded cache")
        if miss_policy not in ("cpu-fallback", "direct-miss-research-v1"):
            raise ValueError("Unknown research miss policy")
        # An unaudited proof is never self-certified by choosing a CLI flag.
        if proof_authority is not None and not isinstance(proof_authority, GridProofAuthority):
            raise ValueError("Proof authority must come from the separate artifact validator")
        if miss_policy != "cpu-fallback" and not (audit_nearest and audit_misses) and proof_authority is None:
            raise ValueError("Direct research misses require full CPU hit/miss audits or a separately verified proof artifact")
        self.proof_authority = proof_authority
        if proof_authority is not None:
            expected = dict(proof_authority.artifact_sha256)
            for path in (Path(__file__), CUDA_SOURCE, Path(__file__).parents[3]/"scanner_server/cuda_nn_registration.py",
                         Path(__file__).with_name("validate_uniform_grid_proof.py"),
                         ROOT/"scripts/tool_paths.py", ROOT/"scripts/tool-catalog.json"):
                relative = str(path.relative_to(Path(__file__).parents[3]))
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(relative):
                    raise ValueError("Proof adapter/kernel/original CPU helper/validator bytes no longer match")
        self.device_id = int(str(self.device).split(":")[1])
        self.max_clouds, self.max_cache_bytes = max_clouds, max_cache_bytes
        self.audit_nearest, self.audit_misses, self.miss_policy = audit_nearest, audit_misses, miss_policy
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.target_membership_sha256 = set()
        self.statistics = {name: 0 for name in ("cloud_uploads", "index_builds", "cache_hits", "searches", "pose_iterations",
            "exact_cpu_queries", "query_rows", "candidate_visits", "direct_gpu_hits", "declared_gpu_misses",
            "audited_hits", "audited_misses", "audit_index_mismatches", "audit_false_misses", "uncertain_queries",
            "unsupported_queries", "cache_evictions", "cache_budget_bypasses", "peak_retained_gpu_bytes", "proof_membership_cpu_clouds")}
        self.statistics.update({name: 0. for name in ("dataset_hash_s", "index_build_s", "gpu_search_transfer_s",
            "cpu_fallback_s", "cpu_audit_s", "dataset_wall_s", "correspondence_wall_s", "icp_wall_s", "cpu_tree_build_s")})
        self.statistics["icp_calls"] = 0
        import cupy as cp
        self.cp = cp
        self.kernel_sha256 = hashlib.sha256(CUDA_SOURCE.read_bytes()).hexdigest()
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            self.kernel = cp.RawKernel(CUDA_SOURCE.read_text(), "grid_nearest_two",
                                      options=("--std=c++11", "--fmad=false"))
            self.kernel.compile()

    def _release(self, item):
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            self.cp.cuda.Stream.null.synchronize()
        self.cache_bytes -= item["gpu_bytes"]
        item["grids"].clear()
        item["gpu_bytes"] = 0

    def clear_cache(self):
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            self.cp.cuda.Stream.null.synchronize()
            for item in self.cache.values():
                self._release(item)
            self.cache.clear()

    def close(self):
        self.clear_cache()

    def match(self, source, target, initial):
        started = time.perf_counter()
        try:
            return super().match(source, target, initial)
        finally:
            self.statistics["icp_calls"] += 1
            self.statistics["icp_wall_s"] += time.perf_counter()-started

    def _dataset(self, target, radius):
        started = time.perf_counter()
        try:
            return self._dataset_impl(target, radius)
        finally:
            self.statistics["dataset_wall_s"] += time.perf_counter()-started

    def _dataset_impl(self, target, radius):
        points = np.asarray(target.points)
        if len(points) > 1000000:
            raise ValueError("Standalone grid bounds targets at one million points")
        started = time.perf_counter()
        digest = hashlib.sha256(memoryview(points).cast("B")).digest()
        self.target_membership_sha256.add(digest.hex())
        self.statistics["dataset_hash_s"] += time.perf_counter()-started
        item = self.cache.pop(id(target), None)
        if item is None or item["target"] is not target or item["digest"] != digest:
            if item is not None:
                self._release(item)
            cpu_started = time.perf_counter()
            cpu = o3d.geometry.KDTreeFlann(target)
            self.statistics["cpu_tree_build_s"] += time.perf_counter()-cpu_started
            item = {"target": target, "digest": digest, "cpu": cpu,
                    "grids": {}, "gpu_bytes": 0, "cpu_only": False}
            if self.proof_authority is not None and digest.hex() not in self.proof_authority.target_digests:
                item["cpu_only"] = True
                self.statistics["proof_membership_cpu_clouds"] += 1
        else:
            self.statistics["cache_hits"] += 1
        self.cache[id(target)] = item
        shift = dyadic_shift(radius)
        if shift not in item["grids"]:
            build_started = time.perf_counter()
            cells = host_cells(points, shift)
            grid = None
            if cells is not None and len(points) and not item["cpu_only"]:
                keys = packed_keys(cells)
                order = np.argsort(keys, kind="stable")
                unique, starts = np.unique(keys[order], return_index=True)
                host = (np.ascontiguousarray(points[order]), order.astype(np.int32), unique,
                        np.r_[starts, len(points)].astype(np.uint32))
                needed = sum(array.nbytes for array in host)
                while (len(self.cache) > self.max_clouds or self.cache_bytes+needed > self.max_cache_bytes) and len(self.cache) > 1:
                    _, previous = self.cache.popitem(last=False)
                    self._release(previous)
                    self.statistics["cache_evictions"] += 1
                if item["gpu_bytes"]+needed <= self.max_cache_bytes:
                    with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
                        grid = tuple(self.cp.asarray(array) for array in host)
                    item["gpu_bytes"] += needed
                    self.cache_bytes += needed
                    self.statistics["cloud_uploads"] += 1
                    self.statistics["index_builds"] += 1
                    self.statistics["peak_retained_gpu_bytes"] = max(self.statistics["peak_retained_gpu_bytes"], self.cache_bytes)
                else:
                    self._release(item)
                    item["cpu_only"] = True
                    self.statistics["cache_budget_bypasses"] += 1
            item["grids"][shift] = grid
            self.statistics["index_build_s"] += time.perf_counter()-build_started
        while len(self.cache) > self.max_clouds:
            _, previous = self.cache.popitem(last=False)
            self._release(previous)
            self.statistics["cache_evictions"] += 1
        return item

    def _raw_device(self, queries, item, radius):
        cp = self.cp
        if queries.ndim != 2 or queries.shape[1] != 3 or queries.dtype != cp.float64 or not queries.flags.c_contiguous:
            raise ValueError("Resident research queries must be contiguous original float64 (N,3)")
        if queries.device.id != self.device_id or len(queries) > 1000000:
            raise ValueError("Query device or bounded count mismatch")
        shift = dyadic_shift(radius)
        grid = item["grids"].get(shift)
        output = cp.empty((len(queries), 5), cp.float64)
        if grid is not None and len(queries):
            points, ids, keys, offsets = grid
            self.kernel((len(queries),), (128,), (points, ids, keys, offsets, np.uint32(len(keys)),
                queries, np.uint32(len(queries)), np.int32(shift), output))
        elif len(queries):
            output[:] = cp.asarray([-3., -1., cp.inf, cp.inf, 0.])
        return output

    def _resolve(self, points, item, radius, first, a, b):
        """Settle ambiguous/missing inputs with the original complete CPU tree."""
        r2 = radius*radius
        error = 64*np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(a), abs(r2)), np.finfo(np.float64).tiny)
        finite = np.isfinite(a)
        with np.errstate(invalid="ignore"):
            boundary = finite & (np.abs(a-r2) <= error)
            tie = finite & np.isfinite(b) & (b-a <= 64*np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(a), np.abs(b)), abs(r2)))
        unsupported = first == -3
        uncertain = boundary | tie
        nearest = first.copy()
        missing = ~unsupported & (~finite | (a >= r2))
        nearest[missing] = -1
        fallback = unsupported | uncertain | missing
        if self.miss_policy == "direct-miss-research-v1":
            if not (self.audit_nearest and self.audit_misses) and self.proof_authority is None:
                raise RuntimeError("Missing results require complete CPU audit or validated scoped proof authority")
            fallback = unsupported | uncertain
            self.statistics["declared_gpu_misses"] += int(np.count_nonzero(missing & ~uncertain))
        self.statistics["unsupported_queries"] += int(np.count_nonzero(unsupported))
        self.statistics["uncertain_queries"] += int(np.count_nonzero(uncertain))
        direct_hits = (~fallback) & (nearest >= 0)
        self.statistics["direct_gpu_hits"] += int(np.count_nonzero(direct_hits))
        started = time.perf_counter()
        for row in np.flatnonzero(fallback):
            count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
            nearest[row] = ids[0] if count else -1
        self.statistics["exact_cpu_queries"] += int(np.count_nonzero(fallback))
        self.statistics["cpu_fallback_s"] += time.perf_counter()-started
        audit = (direct_hits if self.audit_nearest else np.zeros(len(points), bool)) | (
            (missing & ~fallback) if self.audit_misses else np.zeros(len(points), bool))
        started = time.perf_counter()
        mismatches = false_misses = 0
        for row in np.flatnonzero(audit):
            count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
            expected = ids[0] if count else -1
            mismatches += int(nearest[row] != expected)
            false_misses += int(nearest[row] < 0 and expected >= 0)
        self.statistics["audited_hits"] += int(np.count_nonzero(audit & direct_hits))
        self.statistics["audited_misses"] += int(np.count_nonzero(audit & missing))
        self.statistics["audit_index_mismatches"] += mismatches
        self.statistics["audit_false_misses"] += false_misses
        self.statistics["cpu_audit_s"] += time.perf_counter()-started
        if mismatches:
            raise RuntimeError(f"Uniform grid changed {mismatches} original CPU nearest IDs, including{false_misses} false misses")
        return nearest

    def nearest_indices(self, points, item, radius):
        if (not isinstance(points, np.ndarray) or points.dtype != np.float64 or points.ndim != 2
                or points.shape[1] != 3 or len(points) > 1000000):
            raise ValueError("Host research queries must be original float64 (N,3) within bounded count")
        started = time.perf_counter()
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            query = self.cp.asarray(np.ascontiguousarray(points, dtype=np.float64))
            output = self.cp.asnumpy(self._raw_device(query, item, radius))
            first, a, b, visits = output[:, 0].astype(np.int32), output[:, 2], output[:, 3], output[:, 4]
        self.statistics["gpu_search_transfer_s"] += time.perf_counter()-started
        self.statistics["query_rows"] += len(points)
        self.statistics["candidate_visits"] += int(visits.sum(dtype=np.float64))
        return self._resolve(points, item, radius, first, a, b)

    def nearest_device(self, queries, item, radius):
        # Adapter correctness first: this prototype downloads query coordinates
        # for CPU tie/miss/support resolution. It makes no residency speed claim.
        if (not isinstance(queries, self.cp.ndarray) or queries.ndim != 2 or queries.shape[1] != 3
                or queries.dtype != self.cp.float64 or not queries.flags.c_contiguous
                or queries.device.id != self.device_id or len(queries) > 1000000):
            raise ValueError("Device adapter requires contiguous original float64 (N,3) on its selected CUDA device")
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            host = self.cp.asnumpy(queries)
            ids = self.nearest_indices(host, item, radius)
            squared = np.full(len(ids), np.inf)
            selected = np.flatnonzero(ids >= 0)
            difference = host[selected]-np.asarray(item["target"].points)[ids[selected]]
            squared[selected] = np.sum(difference*difference, axis=1)
            return self.cp.asarray(ids), self.cp.asarray(squared)

    def _correspondences(self, source, target, search, radius):
        started = time.perf_counter()
        try:
            points = np.asarray(source.points)
            ids = self.nearest_indices(points, search, radius)
            selected = np.flatnonzero(ids >= 0)
            difference = points[selected]-np.asarray(target.points)[ids[selected]]
            squared = np.sum(difference*difference, axis=1)
            inside = squared < radius*radius
            selected, squared = selected[inside], squared[inside]
            pairs = np.column_stack((selected, ids[selected])).astype(np.int32)
            self.statistics["searches"] += 1
            rmse = float(np.sqrt(np.sum(squared)/len(selected))) if len(selected) else 0.
            return pairs, len(selected)/len(points), rmse
        finally:
            self.statistics["correspondence_wall_s"] += time.perf_counter()-started

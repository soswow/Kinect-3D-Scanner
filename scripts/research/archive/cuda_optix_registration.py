"""Standalone RTX retrieval research, preserving original CPU pose estimation.

Float32 conservative AABBs only retrieve candidates. Original float64 points
and distance accumulation select candidates; the original legacy CPU tree
settles ties, strict-radius uncertainty, invalid floats and missing candidates.
A separately labelled research policy can certify complete-radius empty results;
its optional CPU shadow audit is mandatory before performance attribution.
Never selected by a scanner option or imported by the production backend.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import ctypes
import hashlib
import time
from collections import OrderedDict

import numpy as np
import open3d as o3d

from scanner_server.cuda_nn_registration import NearestICP

ARTIFACTS = ROOT / "benchmark-output/cuda-pipeline/optix-nearest"

# This is an experimental numerical domain, not a claim of universal RTX search
# completeness. Source/PTX/helper hashes also bind all proofs to this enclosure.
CERTIFIED_DOMAIN = {
    "version": "outward-fp32-aabb-fp64-radius-v1",
    "coordinates": "finite float64 target and query; absolute coordinate <= 1000000",
    "minimum_index_radius_m": 1e-6,
    "bounds": "float64 p +/- radius outward nextafter, float32 conversion, eight outward float32 ULPs",
    "traversal": "positive x axis ray; t in [0, 1e-20]; enumerate without reporting an intersection",
    "distance": "original float64 coordinates; (dx*dx + dy*dy) + dz*dz; fmad disabled",
    "complete_radius_bin": 1.0,
    "uncertainty": "CPU complete-radius search for tied/near-boundary hits, unsupported coordinates or absent indexes",
}


def conservative_aabbs(points, radius):
    """Outward round the double bounds, then pad float bounds by eight ULPs.

    Round-to-nearest is monotone: an original query inside the double cube
    rounds inside its outward float enclosure. Padding gives strict interior
    room for the RTX slab tests and the positive tiny ray segment. Extremely
    large/small/nonfinite coordinate regimes are rejected and retain CPU search.
    """
    if (not np.isfinite(points).all() or not np.isfinite(radius) or radius <= 0
            or np.max(np.abs(points), initial=0) > 1e6 or radius < 1e-6):
        return None
    low = np.nextafter(points - radius, -np.inf).astype(np.float32)
    high = np.nextafter(points + radius, np.inf).astype(np.float32)
    for _ in range(8):
        low = np.nextafter(low, np.float32(-np.inf))
        high = np.nextafter(high, np.float32(np.inf))
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return None
    return np.ascontiguousarray(np.concatenate((low, high), axis=1))


class OptixICP(NearestICP):
    def __init__(self, device="CUDA:0", max_clouds=64, radius_bins=(1.,),
                 max_cache_bytes=512 * 1024 * 1024, audit_nearest=False,
                 miss_policy="cpu-fallback", audit_misses=False):
        self.device = o3d.core.Device(str(device))
        if not str(self.device).startswith("CUDA:"):
            raise ValueError("RTX research requires a CUDA device")
        self.device_id = int(str(self.device).split(":")[1])
        if not isinstance(max_clouds, int) or max_clouds < 1 or max_cache_bytes < 1:
            raise ValueError("Require positive cloud and retained GPU memory limits")
        self.max_clouds, self.max_cache_bytes = max_clouds, max_cache_bytes
        self.audit_nearest = bool(audit_nearest)
        if miss_policy not in ("cpu-fallback", "direct-miss-research-v1"):
            raise ValueError("Unknown experimental missing-query policy")
        self.miss_policy, self.audit_misses = miss_policy, bool(audit_misses)
        self.radius_bins = tuple(sorted(set(radius_bins)))
        if not self.radius_bins or self.radius_bins[-1] != 1. or any(not np.isfinite(x) or x <= 0 or x > 1 for x in self.radius_bins):
            raise ValueError("Radius bins must increase to the complete original radius")
        import cupy as cp

        self.cache = OrderedDict()
        self.statistics = {"cloud_uploads": 0, "index_builds": 0, "cache_hits": 0,
            "searches": 0, "pose_iterations": 0, "exact_cpu_queries": 0,
            "rt_queries": 0, "query_rows": 0, "candidate_visits": 0, "bvh_bytes": 0,
            "full_radius_rt_misses": 0, "rt_miss_cpu_hits": 0, "rt_uncertain_queries": 0,
            "direct_rt_hit_queries": 0,
            "audited_rt_hit_queries": 0, "audit_index_mismatches": 0, "cpu_audit_s": 0.,
            "declared_rt_miss_queries": 0, "audited_rt_miss_queries": 0,
            "audit_false_misses": 0, "cpu_miss_audit_s": 0.,
            "peak_bvh_bytes": 0, "cache_gpu_bytes": 0, "peak_allocated_cache_gpu_bytes": 0,
            "cache_evictions": 0, "cache_budget_bypasses": 0,
            "dataset_hash_s": 0., "index_build_s": 0.,
            "cpu_fallback_tree_build_s": 0., "rt_search_s": 0., "cpu_fallback_s": 0.,
            "icp_calls": 0, "icp_wall_s": 0., "dataset_wall_s": 0., "correspondence_wall_s": 0.}
        self.lib = ctypes.CDLL(str(ARTIFACTS / "nearest.dll"))
        pointer, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        self.lib.nn_error.restype = ctypes.c_char_p
        self.lib.nn_create.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
        self.lib.nn_create.restype = pointer
        self.lib.nn_destroy.argtypes = (pointer,)
        self.lib.nn_scene_destroy.argtypes = (pointer,)
        self.lib.nn_scene_bytes.argtypes = (pointer,)
        self.lib.nn_scene_bytes.restype = ctypes.c_size_t
        self.lib.nn_build.argtypes = (pointer, u64, u32)
        self.lib.nn_build.restype = pointer
        self.lib.nn_launch.argtypes = (pointer, pointer, u64, u64, u64, u64, u64, u32, ctypes.c_double)
        self.lib.nn_launch.restype = ctypes.c_int
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            cp.zeros(1, cp.uint8)  # create the current CuPy/CUDA context
            ptx = (ARTIFACTS / "nearest.ptx").read_bytes()
            self.context = self.lib.nn_create(ptx, len(ptx))
        if not self.context:
            raise RuntimeError(self.lib.nn_error().decode())

    def _release(self, item):
        for scene, _, _ in item["scenes"].values():
            if scene:
                self.statistics["bvh_bytes"] -= self.lib.nn_scene_bytes(scene)
                self.lib.nn_scene_destroy(scene)
        self.statistics["cache_gpu_bytes"] -= item["gpu_bytes"]
        item["scenes"].clear()
        item["data"] = None
        item["gpu_bytes"] = 0

    def clear_cache(self):
        import cupy as cp

        with cp.cuda.Device(self.device_id):
            cp.cuda.Stream.null.synchronize()
            for item in self.cache.values():
                self._release(item)
            self.cache.clear()

    def close(self):
        import cupy as cp

        with cp.cuda.Device(self.device_id):
            self.clear_cache()
            if self.context:
                self.lib.nn_destroy(self.context)
                self.context = None

    def match(self, source, target, initial):
        # Inherit every original CPU estimator, pose update and convergence line.
        started = time.perf_counter()
        try:
            return super().match(source, target, initial)
        finally:
            self.statistics["icp_calls"] += 1
            self.statistics["icp_wall_s"] += time.perf_counter() - started

    def _dataset(self, target, radius):
        started = time.perf_counter()
        try:
            return self._dataset_impl(target, radius)
        finally:
            self.statistics["dataset_wall_s"] += time.perf_counter() - started

    def _dataset_impl(self, target, radius):
        import cupy as cp

        points = np.asarray(target.points)
        if not len(points) or len(points) > np.iinfo(np.int32).max:
            raise ValueError("RTX dataset must have a nonempty int32-addressable point set")
        hash_started = time.perf_counter()
        digest = hashlib.sha256(memoryview(points).cast("B")).digest()
        self.statistics["dataset_hash_s"] += time.perf_counter() - hash_started
        item = self.cache.pop(id(target), None)
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            if item is None or item["target"] is not target or item["digest"] != digest:
                if item is not None:
                    self._release(item)
                cpu_started = time.perf_counter()
                cpu = o3d.geometry.KDTreeFlann(target)
                self.statistics["cpu_fallback_tree_build_s"] += time.perf_counter() - cpu_started
                item = {"target": target, "digest": digest, "data": cp.asarray(points),
                        "scenes": {}, "cpu": cpu, "gpu_bytes": points.nbytes,
                        "cpu_only": False}
                self.statistics["cache_gpu_bytes"] += item["gpu_bytes"]
                self.statistics["cloud_uploads"] += 1
            else:
                self.statistics["cache_hits"] += 1
            # Install ownership before any scene build can raise.
            self.cache[id(target)] = item
            for fraction in self.radius_bins:
                search_radius = radius * fraction
                if search_radius not in item["scenes"]:
                    build_started = time.perf_counter()
                    boxes = conservative_aabbs(points, search_radius)
                    if boxes is None or item["cpu_only"]:
                        item["scenes"][search_radius] = (None, None, search_radius)
                        continue
                    uploaded = cp.asarray(boxes)
                    scene = self.lib.nn_build(self.context, uploaded.data.ptr, len(points))
                    if not scene:
                        raise RuntimeError(self.lib.nn_error().decode())
                    item["scenes"][search_radius] = (scene, uploaded, search_radius)
                    self.statistics["index_builds"] += 1
                    self.statistics["bvh_bytes"] += self.lib.nn_scene_bytes(scene)
                    owned = uploaded.nbytes + self.lib.nn_scene_bytes(scene)
                    item["gpu_bytes"] += owned
                    self.statistics["cache_gpu_bytes"] += owned
                    self.statistics["peak_bvh_bytes"] = max(self.statistics["peak_bvh_bytes"], self.statistics["bvh_bytes"])
                    self.statistics["index_build_s"] += time.perf_counter() - build_started
            self.statistics["peak_allocated_cache_gpu_bytes"] = max(
                self.statistics["peak_allocated_cache_gpu_bytes"], self.statistics["cache_gpu_bytes"])
            # The cap bounds retained arrays/index buffers. Temporary OptiX build
            # workspace and one newly built scene can transiently exceed it.
            while len(self.cache) > self.max_clouds or (
                    self.statistics["cache_gpu_bytes"] > self.max_cache_bytes and len(self.cache) > 1):
                _, old = self.cache.popitem(last=False)
                self._release(old)
                self.statistics["cache_evictions"] += 1
            if item["gpu_bytes"] > self.max_cache_bytes:
                self._release(item)
                item["cpu_only"] = True
                self.statistics["cache_budget_bypasses"] += 1
                for fraction in self.radius_bins:
                    item["scenes"][radius * fraction] = (None, None, radius * fraction)
        return item

    def _audit_declared_misses(self, queries, item, radius):
        """CPU shadow every research-certified miss; fail before consuming it."""
        started = time.perf_counter()
        false_misses = 0
        for query in queries:
            count, _, _ = item["cpu"].search_hybrid_vector_3d(query, radius, 1)
            false_misses += int(bool(count))
        self.statistics["audited_rt_miss_queries"] += len(queries)
        self.statistics["audit_false_misses"] += false_misses
        self.statistics["rt_miss_cpu_hits"] += false_misses
        self.statistics["cpu_miss_audit_s"] += time.perf_counter() - started
        if false_misses:
            raise RuntimeError(f"Research RTX misses omitted {false_misses} original CPU neighbours")

    def nearest_indices(self, points, item, radius):
        import cupy as cp

        nearest = np.full(len(points), -3, dtype=np.int32)
        full_radius_miss = np.zeros(len(points), dtype=bool)
        self.statistics["query_rows"] += len(points)
        eligible = np.isfinite(points).all(axis=1) & (np.max(np.abs(points), axis=1) <= 1e6)
        pending = np.flatnonzero(eligible)
        rt_started = time.perf_counter()
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            for fraction in self.radius_bins:
                scene, _, search_radius = item["scenes"][radius * fraction]
                if not len(pending) or scene is None:
                    continue
                queries = cp.asarray(np.ascontiguousarray(points[pending]))
                output = cp.empty(len(pending), cp.int32)
                distances = cp.empty(len(pending), cp.float64)
                visits = cp.empty(len(pending), cp.uint32)
                status = self.lib.nn_launch(self.context, scene, item["data"].data.ptr,
                    queries.data.ptr, output.data.ptr, distances.data.ptr, visits.data.ptr,
                    len(pending), search_radius * search_radius)
                if status:
                    raise RuntimeError(self.lib.nn_error().decode())
                values = cp.asnumpy(output)
                self.statistics["rt_queries"] += len(pending)
                self.statistics["candidate_visits"] += int(cp.asnumpy(visits).sum())
                # An ambiguous result is settled by the original complete-radius CPU tree.
                done = values >= 0
                uncertain = values == -2
                if fraction == 1.:
                    full_radius_miss[pending[values == -1]] = True
                self.statistics["rt_uncertain_queries"] += int(np.count_nonzero(uncertain))
                nearest[pending[done]] = values[done]
                nearest[pending[uncertain]] = -2
                pending = pending[~(done | uncertain)]
        self.statistics["rt_search_s"] += time.perf_counter() - rt_started
        self.statistics["direct_rt_hit_queries"] += int(np.count_nonzero(nearest >= 0))
        if self.miss_policy == "direct-miss-research-v1":
            declared = np.flatnonzero(full_radius_miss)
            self.statistics["declared_rt_miss_queries"] += len(declared)
            if self.audit_misses:
                self._audit_declared_misses(points[declared], item, radius)
            nearest[declared] = -1
            fallback = np.flatnonzero((nearest < 0) & ~full_radius_miss)
        else:
            fallback = np.flatnonzero(nearest < 0)
        fallback_started = time.perf_counter()
        for row in fallback:
            count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
            if full_radius_miss[row] and count:
                self.statistics["rt_miss_cpu_hits"] += 1
            nearest[row] = ids[0] if count else -1
        self.statistics["full_radius_rt_misses"] += int(np.count_nonzero(full_radius_miss))
        self.statistics["exact_cpu_queries"] += len(fallback)
        self.statistics["cpu_fallback_s"] += time.perf_counter() - fallback_started
        if self.audit_nearest:
            # Missing/uncertain rows were already resolved above; shadow every
            # direct RTX hit with the original CPU tree for an exact ID audit.
            # This optional correctness pass invalidates performance attribution.
            audit_started = time.perf_counter()
            rows = np.flatnonzero(nearest >= 0)
            rows = rows[~np.isin(rows, fallback)]
            mismatches = 0
            for row in rows:
                count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
                mismatches += int(nearest[row] != (ids[0] if count else -1))
            self.statistics["audited_rt_hit_queries"] += len(rows)
            self.statistics["audit_index_mismatches"] += mismatches
            self.statistics["cpu_audit_s"] += time.perf_counter() - audit_started
            if mismatches:
                raise RuntimeError(f"RTX direct hits changed {mismatches} original CPU nearest indices")
        return nearest

    def nearest_device(self, queries, item, radius):
        """Resident research adapter; resolve every exceptional row on CPU.

        Inputs: contiguous float64 N×3 CuPy queries and an owned _dataset item.
        Outputs: int32 N indices (>=0/-1) and float64 N squared distances.
        All work uses the selected device/default stream. No query/index array
        is copied to host on the unambiguous hit path; diagnostics copy scalar
        counters. CPU fallback/audit rows explicitly copy their exact queries.
        This separate adapter is untested until its own GPU parity proof passes.
        """
        import cupy as cp

        if not isinstance(queries, cp.ndarray) or queries.dtype != cp.float64 \
                or queries.device.id != self.device_id:
            raise ValueError("Resident queries must be float64 CuPy arrays on the selected device")
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            queries = cp.ascontiguousarray(queries)
            if queries.ndim != 2 or queries.shape[1] != 3:
                raise ValueError("Resident queries must have shape N×3")
            nearest = cp.full(len(queries), -3, cp.int32)
            distances = cp.full(len(queries), cp.inf, cp.float64)
            full_miss = cp.zeros(len(queries), cp.bool_)
            self.statistics["query_rows"] += len(queries)
            eligible = cp.isfinite(queries).all(axis=1) & (cp.max(cp.abs(queries), axis=1) <= 1e6)
            pending = cp.flatnonzero(eligible)
            rt_started = time.perf_counter()
            for fraction in self.radius_bins:
                scene, _, search_radius = item["scenes"][radius * fraction]
                if not len(pending) or scene is None:
                    continue
                selected = cp.ascontiguousarray(queries[pending])
                output = cp.empty(len(pending), cp.int32)
                squared = cp.empty(len(pending), cp.float64)
                visits = cp.empty(len(pending), cp.uint32)
                status = self.lib.nn_launch(self.context, scene, item["data"].data.ptr,
                    selected.data.ptr, output.data.ptr, squared.data.ptr, visits.data.ptr,
                    len(pending), search_radius * search_radius)
                if status:
                    raise RuntimeError(self.lib.nn_error().decode())
                self.statistics["rt_queries"] += len(pending)
                self.statistics["candidate_visits"] += int(visits.sum().item())
                done, uncertain = output >= 0, output == -2
                if fraction == 1.:
                    full_miss[pending[output == -1]] = True
                self.statistics["rt_uncertain_queries"] += int(cp.count_nonzero(uncertain).item())
                nearest[pending[done]] = output[done]
                distances[pending[done]] = squared[done]
                nearest[pending[uncertain]] = -2
                pending = pending[~(done | uncertain)]
            self.statistics["rt_search_s"] += time.perf_counter() - rt_started
            self.statistics["direct_rt_hit_queries"] += int(cp.count_nonzero(nearest >= 0).item())
            direct_hits = cp.flatnonzero(nearest >= 0) if self.audit_nearest else None
            if self.miss_policy == "direct-miss-research-v1":
                declared = cp.flatnonzero(full_miss)
                self.statistics["declared_rt_miss_queries"] += len(declared)
                if self.audit_misses and len(declared):
                    self._audit_declared_misses(cp.asnumpy(queries[declared]), item, radius)
                nearest[declared] = -1
                distances[declared] = cp.inf
                fallback_mask = (nearest < 0) & ~full_miss
            else:
                fallback_mask = nearest < 0
            fallback_started = time.perf_counter()
            fallback = cp.asnumpy(cp.flatnonzero(fallback_mask))
            if len(fallback):
                host_queries = cp.asnumpy(queries[fallback])
                missed = cp.asnumpy(full_miss[fallback])
                resolved = np.full(len(fallback), -1, np.int32)
                cpu_squared = np.full(len(fallback), np.inf, np.float64)
                for row, query in enumerate(host_queries):
                    count, ids, squared = item["cpu"].search_hybrid_vector_3d(query, radius, 1)
                    if count:
                        resolved[row], cpu_squared[row] = ids[0], squared[0]
                        self.statistics["rt_miss_cpu_hits"] += int(missed[row])
                nearest[fallback] = cp.asarray(resolved)
                distances[fallback] = cp.asarray(cpu_squared)
            self.statistics["full_radius_rt_misses"] += int(cp.count_nonzero(full_miss).item())
            self.statistics["exact_cpu_queries"] += len(fallback)
            self.statistics["cpu_fallback_s"] += time.perf_counter() - fallback_started
            if self.audit_nearest and len(direct_hits):
                audit_started = time.perf_counter()
                host_queries, actual = cp.asnumpy(queries[direct_hits]), cp.asnumpy(nearest[direct_hits])
                mismatches = 0
                for row, query in enumerate(host_queries):
                    count, ids, _ = item["cpu"].search_hybrid_vector_3d(query, radius, 1)
                    mismatches += int(actual[row] != (ids[0] if count else -1))
                self.statistics["audited_rt_hit_queries"] += len(actual)
                self.statistics["audit_index_mismatches"] += mismatches
                self.statistics["cpu_audit_s"] += time.perf_counter() - audit_started
                if mismatches:
                    raise RuntimeError(f"Resident RTX hits changed {mismatches} original CPU nearest indices")
            return nearest, distances

    def _correspondences(self, source, target, search, radius):
        started = time.perf_counter()
        try:
            return self._correspondences_impl(source, target, search, radius)
        finally:
            self.statistics["correspondence_wall_s"] += time.perf_counter() - started

    def _correspondences_impl(self, source, target, search, radius):
        points = np.asarray(source.points)
        nearest = self.nearest_indices(points, search, radius)
        selected = np.flatnonzero(nearest >= 0)
        delta = points[selected] - np.asarray(target.points)[nearest[selected]]
        squared = np.sum(delta * delta, axis=1)
        inside = squared < radius * radius
        selected, squared = selected[inside], squared[inside]
        pairs = np.column_stack((selected, nearest[selected])).astype(np.int32)
        self.statistics["searches"] += 1
        rmse = float(np.sqrt(np.sum(squared) / len(selected))) if len(selected) else 0.
        return pairs, len(selected) / len(points), rmse

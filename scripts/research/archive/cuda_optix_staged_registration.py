"""Unmeasured single-launch staged RTX retrieval; production never imports it."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import ctypes
from collections import OrderedDict
import time

import numpy as np
import open3d as o3d

from scripts.research.archive.cuda_optix_registration import OptixICP, ARTIFACTS as DEPENDENCIES, CERTIFIED_DOMAIN

ARTIFACTS = DEPENDENCIES / "staged-v1"


class StagedOptixICP(OptixICP):
    """Inherit original CPU math/cache/fallback; replace only staged retrieval."""

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
        self.audit_nearest, self.audit_misses = bool(audit_nearest), bool(audit_misses)
        if miss_policy not in ("cpu-fallback", "direct-miss-research-v1"):
            raise ValueError("Unknown experimental missing-query policy")
        self.miss_policy = miss_policy
        self.radius_bins = tuple(sorted(set(radius_bins)))
        if not self.radius_bins or len(self.radius_bins) > 4 or self.radius_bins[-1] != 1. \
                or any(not np.isfinite(x) or x <= 0 or x > 1 for x in self.radius_bins):
            raise ValueError("Require at most four positive radius bins ending at the complete radius")
        self.cache = OrderedDict()
        self.statistics = {"cloud_uploads": 0, "index_builds": 0, "cache_hits": 0,
            "searches": 0, "pose_iterations": 0, "exact_cpu_queries": 0,
            "rt_queries": 0, "query_rows": 0, "candidate_visits": 0, "bvh_bytes": 0,
            "full_radius_rt_misses": 0, "rt_miss_cpu_hits": 0, "rt_uncertain_queries": 0,
            "direct_rt_hit_queries": 0, "audited_rt_hit_queries": 0,
            "audit_index_mismatches": 0, "cpu_audit_s": 0.,
            "declared_rt_miss_queries": 0, "audited_rt_miss_queries": 0,
            "audit_false_misses": 0, "cpu_miss_audit_s": 0.,
            "peak_bvh_bytes": 0, "cache_gpu_bytes": 0, "peak_allocated_cache_gpu_bytes": 0,
            "cache_evictions": 0, "cache_budget_bypasses": 0,
            "dataset_hash_s": 0., "index_build_s": 0., "cpu_fallback_tree_build_s": 0.,
            "rt_search_s": 0., "cpu_fallback_s": 0., "icp_calls": 0, "icp_wall_s": 0.,
            "dataset_wall_s": 0., "correspondence_wall_s": 0., "staged_launches": 0}
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
        self.lib.nn_launch_staged.argtypes = (pointer, ctypes.POINTER(pointer),
            ctypes.POINTER(ctypes.c_double), u32, u64, u64, u64, u64, u64, u64, u32)
        self.lib.nn_launch_staged.restype = ctypes.c_int
        import cupy as cp
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            cp.zeros(1, cp.uint8)
            ptx = (ARTIFACTS / "nearest.ptx").read_bytes()
            self.context = self.lib.nn_create(ptx, len(ptx))
        if not self.context:
            raise RuntimeError(self.lib.nn_error().decode())

    def nearest_indices(self, points, item, radius):
        import cupy as cp
        nearest = np.full(len(points), -3, np.int32)
        full_miss = np.zeros(len(points), bool)
        self.statistics["query_rows"] += len(points)
        started = time.perf_counter()
        supported = [item["scenes"][radius * fraction] for fraction in self.radius_bins
                     if item["scenes"][radius * fraction][0] is not None]
        complete = bool(supported) and supported[-1][2] == radius and item["data"] is not None
        eligible = np.isfinite(points).all(axis=1) & (np.max(np.abs(points), axis=1) <= 1e6)
        pending = np.flatnonzero(eligible) if complete else np.empty(0, np.int64)
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            if len(pending):
                selected = cp.asarray(np.ascontiguousarray(points[pending]))
                output = cp.empty(len(pending), cp.int32)
                squared = cp.empty(len(pending), cp.float64)
                visits = cp.empty(len(pending), cp.uint64)
                traces = cp.empty(len(pending), cp.uint32)
                scene_array = (ctypes.c_void_p * len(supported))(*(row[0] for row in supported))
                radius_array = (ctypes.c_double * len(supported))(*(row[2] * row[2] for row in supported))
                status = self.lib.nn_launch_staged(self.context, scene_array, radius_array, len(supported),
                    item["data"].data.ptr, selected.data.ptr, output.data.ptr, squared.data.ptr,
                    visits.data.ptr, traces.data.ptr, len(pending))
                if status:
                    raise RuntimeError(self.lib.nn_error().decode())
                values = cp.asnumpy(output)
                nearest[pending] = values
                full_miss[pending[values == -1]] = True
                self.statistics["candidate_visits"] += int(cp.asnumpy(visits).sum(dtype=np.uint64))
                self.statistics["rt_queries"] += int(cp.asnumpy(traces).sum(dtype=np.uint64))
                self.statistics["rt_uncertain_queries"] += int(np.count_nonzero(values == -2))
                self.statistics["staged_launches"] += 1
        self.statistics["rt_search_s"] += time.perf_counter() - started
        direct_hits = np.flatnonzero(nearest >= 0)
        self.statistics["direct_rt_hit_queries"] += len(direct_hits)
        if self.miss_policy == "direct-miss-research-v1":
            declared = np.flatnonzero(full_miss)
            self.statistics["declared_rt_miss_queries"] += len(declared)
            if self.audit_misses:
                self._audit_declared_misses(points[declared], item, radius)
            fallback = np.flatnonzero((nearest < 0) & ~full_miss)
        else:
            fallback = np.flatnonzero(nearest < 0)
        started = time.perf_counter()
        for row in fallback:
            count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
            if full_miss[row] and count:
                self.statistics["rt_miss_cpu_hits"] += 1
            nearest[row] = ids[0] if count else -1
        self.statistics["full_radius_rt_misses"] += int(np.count_nonzero(full_miss))
        self.statistics["exact_cpu_queries"] += len(fallback)
        self.statistics["cpu_fallback_s"] += time.perf_counter() - started
        if self.audit_nearest:
            started = time.perf_counter()
            mismatches = 0
            for row in direct_hits:
                count, ids, _ = item["cpu"].search_hybrid_vector_3d(points[row], radius, 1)
                mismatches += int(nearest[row] != (ids[0] if count else -1))
            self.statistics["audited_rt_hit_queries"] += len(direct_hits)
            self.statistics["audit_index_mismatches"] += mismatches
            self.statistics["cpu_audit_s"] += time.perf_counter() - started
            if mismatches:
                raise RuntimeError(f"Staged RTX hits changed {mismatches} original CPU nearest indices")
        return nearest

    def nearest_device(self, queries, item, radius):
        """One original-query upload/launch; all radius-bin decisions stay in RTX."""
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
            started = time.perf_counter()
            supported = [item["scenes"][radius * fraction] for fraction in self.radius_bins
                         if item["scenes"][radius * fraction][0] is not None]
            # The last scene must cover the complete original radius. Otherwise
            # neither a hit nor a miss can use the experimental fast path.
            complete = bool(supported) and supported[-1][2] == radius and item["data"] is not None
            eligible = cp.isfinite(queries).all(axis=1) & (cp.max(cp.abs(queries), axis=1) <= 1e6)
            pending = cp.flatnonzero(eligible) if complete else cp.empty(0, cp.int64)
            if len(pending):
                selected = cp.ascontiguousarray(queries[pending])
                output = cp.empty(len(pending), cp.int32)
                squared = cp.empty(len(pending), cp.float64)
                visits = cp.empty(len(pending), cp.uint64)
                traces = cp.empty(len(pending), cp.uint32)
                scene_array = (ctypes.c_void_p * len(supported))(*(row[0] for row in supported))
                radius_array = (ctypes.c_double * len(supported))(*(row[2] * row[2] for row in supported))
                status = self.lib.nn_launch_staged(self.context, scene_array, radius_array, len(supported),
                    item["data"].data.ptr, selected.data.ptr, output.data.ptr, squared.data.ptr,
                    visits.data.ptr, traces.data.ptr, len(pending))
                if status:
                    raise RuntimeError(self.lib.nn_error().decode())
                nearest[pending] = output
                distances[pending] = squared
                full_miss[pending[output == -1]] = True
                # Scalar diagnostics share one host transfer. No per-bin mask,
                # pending compaction, host query upload or synchronization.
                counts = cp.asnumpy(cp.stack((visits.sum(dtype=cp.uint64), traces.sum(dtype=cp.uint64),
                    cp.count_nonzero(output == -2).astype(cp.uint64),
                    cp.count_nonzero(output >= 0).astype(cp.uint64))))
                self.statistics["candidate_visits"] += int(counts[0])
                self.statistics["rt_queries"] += int(counts[1])
                self.statistics["rt_uncertain_queries"] += int(counts[2])
                self.statistics["direct_rt_hit_queries"] += int(counts[3])
                self.statistics["staged_launches"] += 1
            self.statistics["rt_search_s"] += time.perf_counter() - started
            direct_hits = cp.flatnonzero(nearest >= 0) if self.audit_nearest else None
            if self.miss_policy == "direct-miss-research-v1":
                declared = cp.flatnonzero(full_miss)
                self.statistics["declared_rt_miss_queries"] += len(declared)
                if self.audit_misses and len(declared):
                    self._audit_declared_misses(cp.asnumpy(queries[declared]), item, radius)
                distances[declared] = cp.inf
                fallback_mask = (nearest < 0) & ~full_miss
            else:
                fallback_mask = nearest < 0
            started = time.perf_counter()
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
                nearest[fallback], distances[fallback] = cp.asarray(resolved), cp.asarray(cpu_squared)
            self.statistics["full_radius_rt_misses"] += int(cp.count_nonzero(full_miss).item())
            self.statistics["exact_cpu_queries"] += len(fallback)
            self.statistics["cpu_fallback_s"] += time.perf_counter() - started
            if self.audit_nearest and len(direct_hits):
                started = time.perf_counter()
                host_queries, actual = cp.asnumpy(queries[direct_hits]), cp.asnumpy(nearest[direct_hits])
                mismatches = 0
                for row, query in enumerate(host_queries):
                    count, ids, _ = item["cpu"].search_hybrid_vector_3d(query, radius, 1)
                    mismatches += int(actual[row] != (ids[0] if count else -1))
                self.statistics["audited_rt_hit_queries"] += len(actual)
                self.statistics["audit_index_mismatches"] += mismatches
                self.statistics["cpu_audit_s"] += time.perf_counter() - started
                if mismatches:
                    raise RuntimeError(f"Staged RTX hits changed {mismatches} original CPU nearest indices")
            return nearest, distances

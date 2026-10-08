"""Bounded exact-double CUDA minimum for offline RGB-corner/depth association.

No native imports at import time. This supplies proposal pixel identities only;
it never authorizes a pose, changes ICP, or infers an unmeasured depth value.
"""

from __future__ import annotations

import hashlib
import time

MAX_TARGETS = 640 * 480
MAX_QUERIES = 250 * 4
MAX_COORDINATE = 2048.
# CuPy adds -ftz=true itself. NVIDIA defines FTZ for single precision;
# this kernel performs its subtraction, multiply and sum entirely in FP64.
# https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options
OPTIONS = ("--fmad=false",)
SOURCE = r'''
#define EMPTY_DISTANCE 1.7976931348623157e308
#define EMPTY_ID 2147483647
__device__ __forceinline__ bool better(double d, int i, double b, int j) {
    return d < b || (d == b && i < j);
}
extern "C" __global__ void corner_partial(const double* q, const double* t,
    int n, int tiles, double* ds, int* ids) {
    __shared__ double values[256];
    __shared__ int rows[256];
    int query = blockIdx.x, tile = blockIdx.y, tid = threadIdx.x;
    double x = q[2*query], y = q[2*query+1];
    double best = EMPTY_DISTANCE;
    int index = EMPTY_ID;
    int stop = (tile+1)*1024;
    if (stop > n) stop = n;
    for (int row=tile*1024+tid; row<stop; row+=256) {
        double dx = t[2*row]-x, dy = t[2*row+1]-y;
        double distance = dx*dx + dy*dy;
        if (better(distance,row,best,index)) {best=distance; index=row;}
    }
    values[tid]=best; rows[tid]=index; __syncthreads();
    for (int offset=128; offset; offset/=2) {
        if (tid<offset && better(values[tid+offset],rows[tid+offset],values[tid],rows[tid])) {
            values[tid]=values[tid+offset]; rows[tid]=rows[tid+offset];
        }
        __syncthreads();
    }
    if (!tid) {ds[query*tiles+tile]=values[0]; ids[query*tiles+tile]=rows[0];}
}
extern "C" __global__ void corner_minimum(const double* ds, const int* ids,
    int tiles, double* output_d, int* output_i) {
    __shared__ double values[256];
    __shared__ int rows[256];
    int query=blockIdx.x, tid=threadIdx.x;
    double best=EMPTY_DISTANCE; int index=EMPTY_ID;
    for (int tile=tid; tile<tiles; tile+=256) {
        int k=query*tiles+tile;
        if (better(ds[k],ids[k],best,index)) {best=ds[k]; index=ids[k];}
    }
    values[tid]=best; rows[tid]=index; __syncthreads();
    for (int offset=128; offset; offset/=2) {
        if (tid<offset && better(values[tid+offset],rows[tid+offset],values[tid],rows[tid])) {
            values[tid]=values[tid+offset]; rows[tid]=rows[tid+offset];
        }
        __syncthreads();
    }
    if (!tid) {output_d[query]=values[0]; output_i[query]=rows[0];}
}
'''


class NativeCornerCuda:
    def __init__(self, device=0):
        started = time.perf_counter()
        import cupy as cp
        import numpy as np
        self.cp, self.np, self.device = cp, np, device
        with cp.cuda.Device(device), cp.cuda.Stream.null:
            self.partial = cp.RawKernel(SOURCE, "corner_partial", options=OPTIONS)
            self.minimum = cp.RawKernel(SOURCE, "corner_minimum", options=OPTIONS)
            self.partial.compile()
            self.minimum.compile()
            cp.cuda.Stream.null.synchronize()
        props = cp.cuda.runtime.getDeviceProperties(device)
        self.provenance = {"kernel_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "options": list(OPTIONS), "cupy": cp.__version__, "numpy": np.__version__,
            "driver_version": cp.cuda.runtime.driverGetVersion(), "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "device": device, "device_name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "setup_and_compile_s": time.perf_counter() - started,
            "tie_policy": "Original numpy.argmin: lowest input row for equal original double squared distance",
            "scope": "All original visible depth-grid projected pixels; no target subsampling, approximate search, float32 arithmetic or pose authority. New bounded proposal component; every actual query must be CPU-shadowed in this diagnostic."}

    def query(self, queries, targets):
        cp, np = self.cp, self.np
        started = time.perf_counter()
        bindings = []
        for name, array, bound in (("queries", queries, MAX_QUERIES), ("targets", targets, MAX_TARGETS)):
            if (not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.ndim != 2 or array.shape[1] != 2
                    or not 0 < len(array) <= bound or not np.isfinite(array).all() or np.any(np.abs(array) > MAX_COORDINATE)):
                raise ValueError(f"Require original finite FP64 {name}(N,2) within declared corner domain")
            bindings.append({"shape": list(array.shape), "dtype": array.dtype.str,
                "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()})
        count, tiles = len(queries), (len(targets)+1023)//1024
        with cp.cuda.Device(self.device), cp.cuda.Stream.null:
            q, t = cp.asarray(np.ascontiguousarray(queries)), cp.asarray(np.ascontiguousarray(targets))
            ds, indices = cp.empty((count, tiles), cp.float64), cp.empty((count, tiles), cp.int32)
            output_d, output_i = cp.empty(count, cp.float64), cp.empty(count, cp.int32)
            self.partial((count, tiles), (256,), (q, t, np.int32(len(targets)), np.int32(tiles), ds, indices))
            self.minimum((count,), (256,), (ds, indices, np.int32(tiles), output_d, output_i))
            result_i, result_d = cp.asnumpy(output_i), cp.asnumpy(output_d)
            cp.cuda.Stream.null.synchronize()
        gpu_s = time.perf_counter() - started
        if np.any(result_i < 0) or np.any(result_i >= len(targets)) or not np.isfinite(result_d).all() or np.any(result_d < 0):
            raise RuntimeError("Malformed CUDA corner minimum")
        started = time.perf_counter()
        cpu_i, cpu_d = np.empty(count, np.int32), np.empty(count, np.float64)
        for i, query in enumerate(queries):
            distance = np.sum((targets-query)**2, axis=1)
            index = int(np.argmin(distance))
            cpu_i[i], cpu_d[i] = index, distance[index]
        cpu_s = time.perf_counter() - started
        mismatches = int(np.count_nonzero(cpu_i != result_i))
        bit_mismatches = int(np.count_nonzero(cpu_d.view(np.uint64) != result_d.view(np.uint64)))
        if mismatches or bit_mismatches:
            raise RuntimeError(f"CPU/CUDA corner mismatch: IDs={mismatches}, squared-distance bits={bit_mismatches}")
        for array, binding in zip((queries, targets), bindings):
            if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != binding["sha256"]:
                raise RuntimeError("Corner association input mutated")
        return result_i, result_d, {"query_rows": count, "target_rows": len(targets),
            "query_binding": bindings[0], "target_binding": bindings[1], "gpu_upload_allocate_lookup_download_s": gpu_s,
            "cpu_brute_shadow_s": cpu_s, "all_queries_shadowed": True, "id_mismatches": mismatches,
            "squared_distance_bit_mismatches": bit_mismatches,
            "visible_device_arrays_bytes": q.nbytes+t.nbytes+ds.nbytes+indices.nbytes+output_d.nbytes+output_i.nbytes,
            "memory_scope": "Visible arrays plus separately reported compiler/allocator pools; buffers bounded by307200targets/1000corners/300tiles."}

    def self_check(self):
        np = self.np
        reports = []
        cases = [
            (np.asarray([[0., -0.], [-0., 0.], [1., 0.], [-1., 0.]], np.float64),
             np.asarray([[0., 0.], [.5, 0.], [np.nextafter(.5, 1.), 0.], [2048., 2048.]], np.float64)),
            (np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], np.float64),
             np.asarray([[0., 0.], [np.nextafter(0., 1.), 0.], [np.nextafter(0., -1.), 0.]], np.float64)),
            (np.column_stack((np.arange(307200) % 640, np.arange(307200) // 640)).astype(np.float64),
             np.asarray([[1., 1.], [1.5, 1.5], [639.5, 479.5]], np.float64)),
        ]
        for targets, queries in cases:
            for order in (np.arange(len(targets)), np.arange(len(targets))[::-1]):
                _, _, report = self.query(queries, np.ascontiguousarray(targets[order]))
                reports.append(report)
        return {"cases": len(reports), "queries": sum(row["query_rows"] for row in reports),
            "all_ids_and_squared_bits_exact": True, "records": reports}

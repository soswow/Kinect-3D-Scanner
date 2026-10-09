"""Fresh, bounded warp-per-query NN experiment; never enables scanner registration.

Historical trace arrays are input data only. Fresh exhaustive CUDA first/two
minima and original Open3D CPU nearest queries supply the reference. Timers
cover preuploaded raw lookup and synchronization, not whole ICP or Finish.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "scripts/research/archive/research_flat_grid_nn.cu"
NEW = Path(__file__).with_name("microbatch_warp_flat_grid_nn.cu")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, report, save):
    import cupy as cp
    import numpy as np
    import open3d as o3d
    from scripts.profile_session import source_hash
    from scripts.research.archive.cuda_uniform_grid_registration import UniformGridICP, dyadic_shift

    o3d.utility.set_max_threads(20)
    report.update(core_sha256=source_hash(), runtime={"cupy": cp.__version__,
        "numpy": np.__version__, "open3d": o3d.__version__,
        "driver": cp.cuda.runtime.driverGetVersion(), "cuda": cp.cuda.runtime.runtimeGetVersion(),
        "device": "CUDA:0", "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()})
    paths = [Path(__file__), OLD, NEW,
        ROOT / "scripts/research/archive/cuda_uniform_grid_registration.py",
        ROOT / "scripts/profile_session.py", ROOT / "scripts/process_metrics.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    report["artifacts_sha256"] = hashes
    cases = []
    if args.trace:
        if args.trace.stat().st_size > 128 * 1024**2 or args.manifest.stat().st_size > 8 * 1024**2:
            raise ValueError("Bounded local trace required")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if sha(args.trace) != manifest["trace_sha256"]:
            raise ValueError("Trace payload does not match its input manifest")
        report["trace_input"] = {"npz_sha256": sha(args.trace), "manifest_sha256": sha(args.manifest),
            "authority": "Private historical sampled arrays only; original proof not consumed"}
        with np.load(args.trace, allow_pickle=False) as data:
            for row in manifest["batches"]:
                arrays = {}
                for name in ("target_points", "queries"):
                    value = data[f"batch{row['index']}_{name}"]
                    bound = row["arrays"][name]
                    if (value.dtype.str != bound["dtype"] or list(value.shape) != bound["shape"]
                            or value.nbytes != bound["bytes"]
                            or hashlib.sha256(value.tobytes()).hexdigest() != bound["sha256"]):
                        raise ValueError("Trace array binding changed")
                    arrays[name] = value
                cases.append((f"trace-{row['index']}", arrays["target_points"], arrays["queries"], row["radius_m"]))
        if not 1 <= len(cases) <= 12:
            raise ValueError("At most twelve trace cases")
    rng = np.random.default_rng(1234)
    for radius in (.12, .06, .03):
        points = rng.uniform(-.2, .2, (4096, 3))
        queries = points[rng.integers(0, len(points), 2048)] + rng.normal(0, .002, (2048, 3))
        cases.append((f"random-{radius}", points, queries, radius))
    tiny = np.nextafter(0., 1.)
    focused = [
        ("duplicate-tie-boundary", [[0,0,0],[0,0,0],[.03,0,0]], [[0,0,0],[.015,0,0],[.06,0,0]], .03),
        ("empty-centre", [[.0313,0,0],[.0314,0,0],[.05,0,0]], [[.0312,0,0],[0,0,0]], .03),
        ("subnormal", [[-tiny,0,0],[tiny,0,0]], [[0,0,0],[tiny,0,0]], 2.**-20),
        ("cell-boundaries", [[.0625,0,0],[.0625,.0625,0],[.0625,0,.0625],[-.0625,0,0]],
            [[np.nextafter(.0625,0.),0,0],[np.nextafter(.0625,1.),0,0],[0,0,0]], .06),
        ("unsupported", [[1e7,0,0]], [[1e7,0,0],[0,0,0]], .12),
        ("edge-cell", [[1048575.,0,0],[-1048576.,0,0]], [[1048575.5,0,0],[-1048575.5,0,0]], 1.),
    ]
    for label, points, queries, radius in focused:
        cases.append((label, np.array(points, dtype=np.float64), np.array(queries, dtype=np.float64), radius))

    adapter = None
    try:
        started = time.perf_counter()
        adapter = UniformGridICP(max_clouds=1, max_cache_bytes=64*1024**2,
            audit_nearest=True, audit_misses=True, miss_policy="direct-miss-research-v1")
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            kernels = {}
            for name, path in (("exhaustive", OLD), ("warp-exhaustive", NEW)):
                compile_start = time.perf_counter()
                kernel = cp.RawKernel(path.read_text(), "flat_grid_nearest_two", options=("--std=c++11", "--fmad=false"))
                kernel.compile()
                if name == "warp-exhaustive":
                    class WarpLaunch:
                        def __call__(self, grid, block, arguments):
                            return self.kernel(((grid[0]+3)//4,), block, arguments)
                    wrapped = WarpLaunch()
                    wrapped.kernel = kernel
                    kernel = wrapped
                kernels[name] = kernel
                report.setdefault("kernel_setup", []).append({"variant": name, "compile_s": time.perf_counter()-compile_start})
            cp.cuda.Stream.null.synchronize()
        report["setup_s"] = time.perf_counter()-started
        for label, points, queries, radius in cases:
            if (points.dtype != np.float64 or queries.dtype != np.float64 or points.shape[1:] != (3,)
                    or queries.shape[1:] != (3,) or not 0 < len(points) <= 1_000_000
                    or not 0 < len(queries) <= 100_000 or points.nbytes+queries.nbytes > 64*1024**2):
                raise ValueError("Bounded FP64 case required")
            inputs_before = hashlib.sha256(points.tobytes()+queries.tobytes()).hexdigest()
            target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            build_start = time.perf_counter()
            item = adapter._dataset(target, radius)
            with cp.cuda.Device(0), cp.cuda.Stream.null:
                gpu_queries = cp.asarray(queries)
                output = cp.empty((len(queries), 5), dtype=cp.float64)
                cp.cuda.Stream.null.synchronize()
            row = {"label": label, "source_rows": len(queries), "target_rows": len(points),
                "radius_m": radius, "grid_build_and_query_upload_s": time.perf_counter()-build_start,
                "input_arrays_sha256": inputs_before, "samples": []}
            report.setdefault("cases", []).append(row)
            reference = None
            for repeat in range(args.repeats+1):
                order = ["exhaustive", "warp-exhaustive"] if repeat % 2 == 0 else ["warp-exhaustive", "exhaustive"]
                results = {}
                for name in order:
                    with cp.cuda.Device(0), cp.cuda.Stream.null:
                        adapter.kernel = kernels[name]
                        start = time.perf_counter()
                        value = adapter._raw_device(gpu_queries, item, radius)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter()-start
                        result = cp.asnumpy(value)
                    results[name] = result
                    if reference is None and name == "exhaustive":
                        reference = result.copy()
                    row["samples"].append({"variant": name, "repeat": repeat, "warmup": repeat == 0,
                        "lookup_alloc_enqueue_sync_s": elapsed, "candidate_visits": int(result[:,4].sum())})
                for name, value in results.items():
                    if not np.array_equal(value[:,:4].copy().view(np.uint64), reference[:,:4].copy().view(np.uint64)):
                        raise RuntimeError(f"{label}: {name} changed exhaustive first/two IDs or metric bits")
                if not np.array_equal(results["warp-exhaustive"][:,4], results["exhaustive"][:,4]):
                    raise RuntimeError("Warp traversal visits differ from exhaustive traversal")
            audit_start = time.perf_counter()
            # Fresh full original CPU audit of both hit/miss decisions. Flagged
            # ties, unsupported and radius boundaries use original CPU resolution.
            adapter._resolve(queries, item, radius, reference[:,0].astype(np.int32), reference[:,2], reference[:,3])
            row.update(raw_first_four_uint64_exact=True, original_cpu_hit_miss_audit=True,
                cpu_audit_s=time.perf_counter()-audit_start, input_arrays_unchanged=
                hashlib.sha256(points.tobytes()+queries.tobytes()).hexdigest() == inputs_before)
            if not row["input_arrays_unchanged"]:
                raise RuntimeError("Input arrays changed")
            save()
            print(label+": exact raw minima and fresh original CPU IDs passed", flush=True)
        report["cpu_audit_statistics"] = dict(adapter.statistics)
    finally:
        if adapter is not None:
            adapter.close()
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            cp.cuda.Stream.null.synchronize()
        report["device_cleanup_passed"] = True
    report["core_sha256_after"] = source_hash()
    report["artifacts_sha256_after"] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if hashes != report["artifacts_sha256_after"] or report["core_sha256_after"] != report["core_sha256"]:
        raise RuntimeError("Measured source changed")
    if args.trace and (sha(args.trace) != report["trace_input"]["npz_sha256"]
            or sha(args.manifest) != report["trace_input"]["manifest_sha256"]):
        raise RuntimeError("Trace changed")
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    if not args.run_allocated or not 3 <= args.repeats <= 20 or bool(args.trace) != bool(args.manifest):
        parser.error("Require exclusive hardware, 3-20 repeats and both trace/manifest if supplied")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep private outputs in benchmark-output")
    if args.output.exists():
        parser.error("Fresh output required; preserve previous result")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1",8000)) == 0:
            parser.error("Stop idle field server before exclusive benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "warp-per-query-raw-neighbour-experiment-v1", "status": "running",
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authority": "Fresh raw-lookup pilot only. No trajectory, proposal gates, full Finish, or production authority.",
        "timer_scope": "Preuploaded raw lookup, allocation, enqueue and selected-stream synchronization. CPU audits/copies outside sample timer; setup/build separately reported.",
        "repeats": args.repeats}
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    code = 0
    try:
        save()
        run(args, report, save)
    except BaseException as error:
        import traceback
        code = 1
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
    report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save()
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    if code == 0:
        from scripts.process_metrics import finish_cuda_worker
        finish_cuda_worker()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

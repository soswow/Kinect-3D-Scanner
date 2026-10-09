"""Standalone exact RTX nearest-neighbour/real bridge correctness experiment.

GPU EXECUTION: run only in an explicitly allocated hardware benchmark slot.
First build offline with scripts/research/archive/build_optix_staged.ps1. The official NVIDIA
optix-dev v9.0.0 headers/license live in the ignored experiment vendor folder.
Original CPU solve, thresholds, independent witnesses and heldout checks remain.
Fixture local poses only reproduce component inputs; no live tracking is seeded.

The NN check compares individual indices with original Open3D KDTreeFlann.
Timing compares complete original _verify_bridge calls, including target hashing,
uploads, index construction, exact CPU fallbacks, and original CPU pose solves.
Warm repeats separately measure persistent index reuse. NN diagnostics do not
use slow Python query loops as a CPU performance baseline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.tool_paths import tool_path


import argparse
import hashlib
import json
import os
import pickle
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_NATIVE", "on")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_memory_snapshot(device_id=0):
    """Outside measured calls; observed device-wide usage includes other apps."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {"driver_free_bytes": free, "driver_total_bytes": total,
                "driver_used_bytes": total - free,
                "cupy_pool_used_bytes": pool.used_bytes(), "cupy_pool_total_bytes": pool.total_bytes(),
                "scope": "Post-stage snapshot, not a peak; driver usage includes desktop/other processes and temporary build workspace has already been freed."}


def check_indices(solver, device_adapter=False):
    import numpy as np
    import open3d as o3d

    rng = np.random.default_rng(72)
    rows = []
    def check(name, targets, queries, radius):
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(targets))
        item = solver._dataset(target, radius)
        search_started = time.perf_counter()
        actual = solver.nearest_indices(queries, item, radius)
        search_s = time.perf_counter() - search_started
        tree = o3d.geometry.KDTreeFlann(target)
        expected, expected_squared = [], []
        for query in queries:
            count, ids, squared = tree.search_hybrid_vector_3d(query, radius, 1)
            expected.append(ids[0] if count else -1)
            expected_squared.append(squared[0] if count else np.inf)
        mismatch = int(np.count_nonzero(actual != expected))
        rows.append({"case": name, "queries": len(queries), "radius": radius,
                     "index_mismatches": mismatch, "rtx_query_s": search_s})
        if mismatch:
            raise RuntimeError(f"RTX retrieval changed {mismatch} original CPU indices: {name}")
        if device_adapter:
            import cupy as cp

            with cp.cuda.Device(solver.device_id), cp.cuda.Stream.null:
                ids, squared = solver.nearest_device(cp.asarray(queries), item, radius)
                ids, squared = cp.asnumpy(ids), cp.asnumpy(squared)
            mismatch = int(np.count_nonzero(ids != expected))
            finite = np.isfinite(expected_squared)
            distance_delta = float(np.max(np.abs(squared[finite] - np.asarray(expected_squared)[finite]), initial=0))
            rows[-1].update({"device_adapter_index_mismatches": mismatch,
                             "device_adapter_max_squared_distance_delta": distance_delta})
            if mismatch or not np.allclose(squared, expected_squared, atol=1e-12, rtol=1e-12):
                raise RuntimeError(f"Resident RTX adapter changed original nearest evidence: {name}")
    for size in (1, 17, 257, 4099):
        targets = rng.normal(size=(size, 3))
        if size > 17:
            targets[2] = targets[1]
        queries = np.concatenate((targets[:10], rng.normal(size=(250, 3))))
        for radius in (.03, .12, 3.):
            check(f"random-duplicates-{size}", targets, queries, radius)
    boundary = .03
    check("strict-radius-and-adjacent-doubles", np.array([[0., 0., 0.]]),
          np.array([[boundary, 0., 0.], [np.nextafter(boundary, 0.), 0., 0.],
                    [np.nextafter(boundary, np.inf), 0., 0.]]), boundary)
    check("equidistant-distinct-and-duplicate", np.array([[-.01, 0., 0.], [.01, 0., 0.], [.01, 0., 0.]]),
          np.array([[0., 0., 0.], [1e-16, 0., 0.], [-1e-16, 0., 0.]]), .03)
    base = np.float64(np.float32(1.234567))
    queries = np.array([[np.nextafter(base + boundary, direction), 0., 0.]
                        for direction in (-np.inf, base, np.inf)])
    check("float-rounding-radius-boundary", np.array([[base, 0., 0.]]), queries, boundary)
    check("extreme-coordinate-CPU-fallback", np.array([[1e7, 0., 0.], [1e7+.02, 0., 0.]]),
          np.array([[1e7+.01, 0., 0.], [1e7+.03, 0., 0.]]), .03)
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.normal(size=(32, 3))))
    solver._dataset(target, .03)
    uploads = solver.statistics["cloud_uploads"]
    np.asarray(target.points)[0, 0] += .1
    solver._dataset(target, .03)
    if solver.statistics["cloud_uploads"] != uploads + 1 or len(solver.cache) > solver.max_clouds:
        raise RuntimeError("Content mutation or bounded index retention failed")
    return rows


def canonical_evidence(result):
    from scripts.research.benchmark_parallel_fragments import json_value

    return None if result is None else json_value(result)


def evidence_agreement(reference, actual):
    """Preserve decisions/support fields exactly; tightly compare numerical evidence."""
    import numpy as np

    if reference is None or actual is None:
        return {"passed": reference is None and actual is None, "same_decision": reference is None and actual is None,
                "max_translation_delta_m": None, "max_information_delta": None}
    a, b = canonical_evidence(reference), canonical_evidence(actual)
    exact_keys = set(a) | set(b)
    numerical = {"transform", "information", "geometry", "visual", "validation"}
    differences = [key for key in exact_keys - numerical if a.get(key) != b.get(key)]
    transform_delta = float(np.max(np.abs(np.asarray(a["transform"]) - np.asarray(b["transform"]))))
    info_delta = float(np.max(np.abs(np.asarray(a["information"]) - np.asarray(b["information"])))) if "information" in a and "information" in b else 0.
    # Recursively check every numeric and nonnumeric support/gate value.
    def equal(x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(equal(x[key], y[key]) for key in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(equal(xx, yy) for xx, yy in zip(x, y))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            return bool(np.isclose(x, y, rtol=1e-8, atol=1e-8))
        return x == y
    passed = not differences and transform_delta <= 1e-8 and info_delta <= 1e-5 and equal(a, b)
    return {"passed": passed, "same_decision": True, "exact_field_differences": differences,
            "max_transform_entry_delta": transform_delta, "max_information_delta": info_delta}


def pair_verdict(proposals):
    """Consume every competing proposal in original order, then apply ambiguity."""
    from scanner_server.fragments import _disagrees

    verified = [value for value in proposals if value is not None]
    ambiguous = bool(verified) and any(_disagrees(verified[0]["transform"], value["transform"])
                                       for value in verified[1:])
    return {"accepted": bool(verified) and not ambiguous, "ambiguous": ambiguous,
            "verified_proposals": len(verified),
            "best": canonical_evidence(verified[0]) if verified and not ambiguous else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--bins", type=float, nargs="+", default=(1.,))
    parser.add_argument("--device", default="CUDA:0")
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--max-clouds", type=int, default=64)
    parser.add_argument("--audit-nearest", action="store_true",
                        help="Shadow every direct RTX hit with the original CPU tree; correctness only, timing confounded")
    parser.add_argument("--miss-policy", choices=("cpu-fallback", "direct-miss-research-v1"),
                        default="cpu-fallback", help="Direct missing-query certification remains research-only")
    parser.add_argument("--audit-misses", action="store_true",
                        help="Shadow every research-declared complete-radius miss with the original CPU tree")
    parser.add_argument("--check-device-adapter", action="store_true",
                        help="Also verify the future resident-query adapter in synthetic correctness checks")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/optix-nearest/staged-v1/proof.json")
    args = parser.parse_args()
    if args.pairs < 1 or args.repeats < 1 or args.cache_mib < 1 or args.max_clouds < 1:
        parser.error("Require positive pairs/repeats/cache limits")
    if tuple(args.bins) != tuple(sorted(set(args.bins))):
        parser.error("Radius bins must be unique and strictly increasing")
    import cupy as cp
    import cv2
    import numpy as np
    import open3d as o3d
    import scanner_server.fragments as fragments_module
    from scripts.research.benchmark_parallel_fragments import numerical_source_hash, unpack_fragment
    from scripts.research.archive.cuda_optix_staged_registration import StagedOptixICP as OptixICP, ARTIFACTS, CERTIFIED_DOMAIN
    from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
    from scripts.profile_session import source_hash

    o3d.utility.set_max_threads(20)
    cv2.setNumThreads(20)
    source_before, math_before = source_hash(), numerical_source_hash()
    artifact_paths = {name: tool_path(name) for name in
        ("benchmark_optix_staged.py", "cuda_optix_staged_registration.py", "research_optix_staged.cu",
         "research_optix_staged_host.cpp", "build_optix_staged.ps1", "benchmark_parallel_fragments.py",
         "cuda_optix_registration.py")}
    artifact_paths.update({name: ROOT/name for name in ("scripts/tool_paths.py", "scripts/tool-catalog.json")})
    artifact_paths["original_cpu_icp_adapter"] = ROOT / "scanner_server/cuda_nn_registration.py"
    artifact_paths.update({name: ARTIFACTS / name for name in ("nearest.ptx", "nearest.dll")})
    artifact_paths["vendor/optix-dev-v9.0.0.zip"] = ARTIFACTS.parent / "vendor/optix-dev-v9.0.0.zip"
    artifacts = {name: hash_file(path) for name, path in artifact_paths.items()}
    report = {"kind": "standalone-exact-rtx-fused-staged-neighbour-search", "source_sha256": source_before,
        "retrieval_execution": "one OptiX launch, register payload, all radius bins inside raygen; FP64 no-FMA unchanged",
        "verification_dependencies_sha256": math_before, "artifacts_sha256": artifacts,
        "gpu": gpu_info(), "versions": {"numpy": np.__version__, "opencv": cv2.__version__,
            "open3d": o3d.__version__, "cupy": cp.__version__},
        "cache_policy": {"max_clouds": args.max_clouds, "retained_gpu_bytes": args.cache_mib * 1024**2,
            "scope": "Points, AABBs and BVH buffers; one new scene/build workspace and CuPy allocator pools can transiently exceed retained cap."},
        "radius_bins": args.bins, "thread_policy": {"omp": os.environ["OMP_NUM_THREADS"], "open3d": 20, "opencv": 20},
        "timing_state": "Context/module setup is separate. Synthetic checks warm programs; cache clears before real tasks. First real pass includes newly needed target indexes, warm repeats may reuse them.",
        "cpu_index_audit": args.audit_nearest, "cpu_miss_audit": args.audit_misses,
        "miss_policy": args.miss_policy, "certified_domain": CERTIFIED_DOMAIN,
        "performance_attribution_valid": not (args.audit_nearest or args.audit_misses),
        "synthetic_device_adapter_checked": args.check_device_adapter,
        "authority": "Fixed component fixture and original CPU gates only; no live seeds, adaptive graph frontier, fresh fusion or mesh speed/quality claim."}
    original, solver, failure = fragments_module._match, None, None
    report["real_pairs"] = []
    setup_started = time.perf_counter()
    try:
        solver = OptixICP(device=args.device, radius_bins=args.bins, max_clouds=args.max_clouds,
                          max_cache_bytes=args.cache_mib * 1024**2, audit_nearest=args.audit_nearest,
                          miss_policy=args.miss_policy, audit_misses=args.audit_misses)
        report["device"] = str(solver.device)
        report["radius_bins"] = list(solver.radius_bins)
        report["setup_s"] = time.perf_counter() - setup_started
        report["setup_gpu_memory"] = gpu_memory_snapshot(solver.device_id)
        synthetic_started = time.perf_counter()
        report["synthetic_indices"] = check_indices(solver, args.check_device_adapter)
        report["synthetic_checks_wall_s"] = time.perf_counter() - synthetic_started
        report["synthetic_statistics"] = dict(solver.statistics)
        report["synthetic_gpu_memory"] = gpu_memory_snapshot(solver.device_id)
        # Synthetic clouds never contribute retained indexes to real timing.
        solver.clear_cache()
        print(f"Synthetic index parity passed: {sum(row['queries'] for row in report['synthetic_indices'])} queries", flush=True)
        if not args.synthetic_only:
            fixture_started = time.perf_counter()
            fixture_sha = hash_file(args.fixture)
            with args.fixture.open("rb") as handle:
                fixture = pickle.load(handle)
            if fixture["metadata"]["component_source_sha256"] != math_before:
                raise RuntimeError("Fixture verification mathematics changed")
            report["fixture_sha256"] = fixture_sha
            report["fixture_provenance"] = fixture["metadata"]
            report["raw_input_sha256"] = hash_file(fixture["metadata"]["session"])
            if report["raw_input_sha256"] != fixture["metadata"]["input_sha256"]:
                raise RuntimeError("Original raw archive changed since fixture preparation")
            report["fixture_load_and_hash_s"] = time.perf_counter() - fixture_started
            fixture_started = time.perf_counter()
            # Include one accepted and one rejected task from the original fixture order.
            tasks = fixture["tasks"][::max(1, len(fixture["tasks"]) // args.pairs)][:args.pairs]
            packed = {index: unpack_fragment(data) for index, data in fixture["fragments"].items()}
            # Both timing modes begin with the fixture's same exact descriptor cache.
            from scanner_server.fragments import _cache_matches
            for index, data in fixture["fragments"].items():
                for original_view, view in zip(data["keys"], packed[index].keys):
                    for other_fragment in packed.values():
                        for other in other_fragment.keys:
                            matches = original_view["matches"].get(other.index)
                            if matches is not None:
                                _cache_matches(view, other, matches.copy())
            report["fixture_unpack_and_cache_s"] = time.perf_counter() - fixture_started
            for task in tasks:
                a, b = task["pair"]
                source, target = packed[a], packed[b]
                baseline, cpu_s = [], 0.
                for proposal_index, proposal in enumerate(task["proposals"]):
                    started = time.perf_counter()
                    baseline.append(fragments_module._verify_bridge(source, target, proposal, fixture["camera"]))
                    elapsed = time.perf_counter() - started
                    cpu_s += elapsed
                    print(f"CPU pair {a}/{b} proposal {proposal_index}: {elapsed:.3f}s", flush=True)
                cpu_verdict = pair_verdict(baseline)
                fragments_module._match = solver.match
                gpu_runs = []
                pair_row = {"pair": task["pair"], "proposals": len(baseline),
                    "accepted_proposals": sum(value is not None for value in baseline),
                    "original_proposal_evidence": [canonical_evidence(value) for value in baseline],
                    "original_pair_verdict": cpu_verdict,
                    "cpu_original_s": cpu_s, "rtx_runs": gpu_runs}
                report["real_pairs"].append(pair_row)
                try:
                    for repeat in range(args.repeats):
                        before_statistics = dict(solver.statistics)
                        started = time.perf_counter()
                        actual, proposal_times = [], []
                        for proposal_index, proposal in enumerate(task["proposals"]):
                            proposal_started = time.perf_counter()
                            actual.append(fragments_module._verify_bridge(source, target, proposal, fixture["camera"]))
                            proposal_times.append(time.perf_counter() - proposal_started)
                            print(f"RTX repeat {repeat} pair {a}/{b} proposal {proposal_index}: {proposal_times[-1]:.3f}s", flush=True)
                        pair_wall_s = time.perf_counter() - started
                        elapsed = sum(proposal_times)
                        checks = [evidence_agreement(old, new) for old, new in zip(baseline, actual)]
                        verdict = pair_verdict(actual)
                        verdict_agrees = all(verdict[key] == cpu_verdict[key]
                            for key in ("accepted", "ambiguous", "verified_proposals"))
                        gpu_runs.append({"repeat": repeat, "elapsed_s": elapsed, "agreement": checks,
                            "pair_wall_s": pair_wall_s, "proposal_times_s": proposal_times,
                            "gpu_memory_after": gpu_memory_snapshot(solver.device_id),
                            "proposal_evidence": [canonical_evidence(value) for value in actual],
                            "pair_verdict": verdict, "pair_verdict_agrees": verdict_agrees,
                            "statistics": dict(solver.statistics),
                            "statistics_delta": {key: value - before_statistics[key]
                                                 for key, value in solver.statistics.items()}})
                        delta = gpu_runs[-1]["statistics_delta"]
                        gpu_runs[-1]["native_hit_audit_complete"] = None if not args.audit_nearest else (
                            delta["direct_rt_hit_queries"] == delta["audited_rt_hit_queries"]
                            and delta["audit_index_mismatches"] == 0)
                        gpu_runs[-1]["native_miss_audit_complete"] = None if not args.audit_misses else (
                            delta["declared_rt_miss_queries"] == delta["audited_rt_miss_queries"]
                            and delta["audit_false_misses"] == 0)
                        if not verdict_agrees or not all(check["passed"] for check in checks):
                            raise RuntimeError("RTX proposal evidence changed; retain CPU authority")
                finally:
                    fragments_module._match = original
                print(json.dumps({"pair": task["pair"], "cpu_s": cpu_s,
                    "rtx_s": [row["elapsed_s"] for row in gpu_runs],
                    "agrees": all(row["pair_verdict_agrees"] for row in gpu_runs)}), flush=True)
    except Exception as error:
        failure = error
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        fragments_module._match = original
        report.setdefault("setup_s", time.perf_counter() - setup_started)
        report["final_statistics"] = dict(solver.statistics) if solver else None
        report["retrieval_complete_on_tested_queries"] = bool(solver) and solver.statistics["rt_miss_cpu_hits"] == 0
        report["source_unchanged"] = source_hash() == source_before
        report["math_unchanged"] = numerical_source_hash() == math_before
        report["artifacts_unchanged"] = all(hash_file(path) == artifacts[name]
            for name, path in artifact_paths.items())
        report["fixture_unchanged"] = args.synthetic_only or (
            "fixture_sha256" in report and hash_file(args.fixture) == report["fixture_sha256"])
        report["raw_input_unchanged"] = args.synthetic_only or (
            "raw_input_sha256" in report and hash_file(report["fixture_provenance"]["session"])
                == report["raw_input_sha256"])
        report["peak_process_rss_bytes"] = peak_rss_bytes()
        report["passed"] = failure is None and report["source_unchanged"] and report["math_unchanged"] and report["artifacts_unchanged"] \
            and report["fixture_unchanged"] and report["raw_input_unchanged"] \
            and (args.synthetic_only or bool(report["real_pairs"])) and all(run["pair_verdict_agrees"]
                for pair in report["real_pairs"] for run in pair["rtx_runs"]) and all(check["passed"]
            for pair in report["real_pairs"] for run in pair["rtx_runs"] for check in run["agreement"])
        if args.audit_nearest:
            report["passed"] = report["passed"] and all(run["native_hit_audit_complete"]
                for pair in report["real_pairs"] for run in pair["rtx_runs"])
        if args.audit_misses:
            report["passed"] = report["passed"] and all(run["native_miss_audit_complete"]
                for pair in report["real_pairs"] for run in pair["rtx_runs"])
        try:
            if solver:
                solver.close()
            cp.cuda.Stream.null.synchronize()
        except Exception as cleanup_error:
            report["cleanup_error"] = str(cleanup_error)
            report["passed"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report {args.output}; passed={report['passed']}", flush=True)
    if not report["passed"]:
        if report.get("error"):
            print(report["error"]["traceback"], file=sys.stderr, flush=True)
        # Avoid the known Windows preview-wheel finalizer crash, retaining a
        # failing process status. The parent must still read report.passed.
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel.TerminateProcess.restype = wintypes.BOOL
            kernel.TerminateProcess(kernel.GetCurrentProcess(), 1)
        raise RuntimeError(f"RTX proof failed; inspect {args.output}") from failure
    finish_cuda_worker()


if __name__ == "__main__":
    main()

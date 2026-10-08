"""Separate true-device flat NN and unchanged resident ICP proof/benchmark.

Only new component research; no production backend or complete Finish claim.
Run in an explicitly allocated exclusive slot. Every new resident trajectory
must pass fresh original CPU nearest hits/misses and all nine original gates.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"
EXPECTED_SOURCE = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
from scripts.benchmark_uniform_grid_nn import (file_hash, host_enclosure_checks, extra_device_cases,
    restore_matches, load_fixture_scope, current_runtime_binding as original_runtime_binding)
from scripts.benchmark_flat_grid_nn import cache_ownership_cases


def current_runtime_binding(report, cp, np, o3d):
    from scripts.research_resident_icp import GPU_SOURCE, CPU_BRIDGE_SOURCE
    binding = original_runtime_binding(report, cp, np, o3d)
    binding["resident_configuration"] = dict(report["resident_configuration"])
    library = Path(report["solver_library_path"])
    binding["resident_math"] = {
        "script_sha256": file_hash(ROOT/"scripts/research_resident_icp.py"),
        "gpu_source_sha256": hashlib.sha256(GPU_SOURCE.encode()).hexdigest(),
        "cpu_bridge_source_sha256": hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest(),
        "solve_library_sha256": file_hash(library),
        "solve_manifest_sha256": file_hash(library.with_suffix(".build.json"))}
    return binding


def bridge_class(solver_library, scratch_bytes, query_bytes):
    from scripts.cuda_device_flat_grid_registration import DeviceFlatGridICP
    from scripts.research_device_grid_resident_icp import DeviceGridResidentICP
    from scripts.research_resident_icp import EigenSolve

    class DeviceResidentBridge(DeviceFlatGridICP):
        def __init__(self, **kwargs):
            super().__init__(max_query_bytes=query_bytes, **kwargs)
            try:
                self.solve = EigenSolve(solver_library)
                self.resident = DeviceGridResidentICP(self, self.solve, max_points=1000000,
                    max_scratch_bytes=scratch_bytes, gpu_timing=False, owns_retrieval=False)
            except BaseException as error:
                try:
                    super().close()
                except BaseException as cleanup_error:
                    error.add_note(f"Partial resident construction cleanup also failed: {cleanup_error}")
                raise

        def match(self, source, target, initial):
            started = time.perf_counter()
            try:
                return self.resident.match(source, target, initial)
            finally:
                self.statistics["icp_calls"] += 1
                self.statistics["icp_wall_s"] += time.perf_counter()-started

        def close(self):
            primary = sys.exception()
            failures = []
            for action in (self.resident.close, super().close):
                try:
                    action()
                except BaseException as error:
                    failures.append(error)
            if failures:
                if primary is not None:
                    primary.add_note(f"Resident bridge cleanup also failed: {failures}")
                else:
                    for error in failures[1:]:
                        failures[0].add_note(f"Additional resident cleanup failure: {error}")
                    raise failures[0]

    return DeviceResidentBridge


def actual_resident_configuration(solver):
    return {"device": str(solver.device), "max_points": solver.resident.max_points,
        "max_scratch_bytes": solver.resident.max_scratch_bytes,
        "max_query_bytes": solver.max_query_bytes, "gpu_timing": solver.resident.gpu_timing}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--scratch-mib", type=int, default=256)
    parser.add_argument("--query-mib", type=int, default=136)
    parser.add_argument("--fixture", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    parser.add_argument("--reference", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--audit-nearest", action="store_true")
    parser.add_argument("--audit-misses", action="store_true")
    parser.add_argument("--miss-policy", choices=("cpu-fallback", "direct-miss-research-v1"), default="cpu-fallback")
    parser.add_argument("--check-device-adapter", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--max-clouds", type=int, default=64)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE)
    parser.add_argument("--proof-synthetic", type=Path)
    parser.add_argument("--proof-bridge", type=Path)
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.repeats <= 4 or not 1 <= args.cache_mib <= 256 or not 1 <= args.max_clouds <= 64:
        parser.error("Require an allocated slot and bounded repeat/cache settings")
    if not 1 <= args.scratch_mib <= 256 or not 1 <= args.query_mib <= 136:
        parser.error("Require bounded resident and query scratch budgets")
    try:
        args.solver_dll.resolve().relative_to(ROOT)
    except ValueError:
        parser.error("Require pinned solve DLL inside the workspace for explicit artifact binding")
    if bool(args.proof_synthetic) != bool(args.proof_bridge):
        parser.error("Both separate proof artifacts are required")
    if args.proof_synthetic and (args.synthetic_only or args.miss_policy != "direct-miss-research-v1"):
        parser.error("Proof authority applies to the real fixed-bridge direct-miss experiment")
    if args.miss_policy != "cpu-fallback" and not (args.audit_nearest and args.audit_misses) and not args.proof_synthetic:
        parser.error("Direct misses require both full audits or separately validated current proof artifacts")
    if args.output.exists():
        parser.error("Require a fresh output path; preserve all previous measured reports")
    if args.expected_source_sha256 != EXPECTED_SOURCE:
        parser.error("Flat experiment requires the frozen9331 production source")
    if os.environ["OMP_NUM_THREADS"] != "8":
        parser.error("Require original OMP8/Open3D20/OpenCV20 policy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.profile_session import source_hash
    from scripts import benchmark_parallel_fragments as baseline
    source_before, component_before = source_hash(), baseline.numerical_source_hash()
    if source_before != args.expected_source_sha256:
        raise RuntimeError("Require frozen production source before numerical imports")
    artifact_paths = [Path(__file__), ROOT/"scripts/benchmark_flat_grid_nn.py",
        ROOT/"scripts/validate_flat_grid_proof.py", ROOT/"scripts/FLAT_GRID_RESEARCH.md", ROOT/"scripts/benchmark_flat_grid_timing.py",
        ROOT/"scripts/cuda_device_flat_grid_registration.py", ROOT/"scripts/research_device_flat_grid_nn.cu",
        ROOT/"scripts/device_flat_grid_synthetic.py", ROOT/"scripts/research_device_grid_resident_icp.py",
        ROOT/"scripts/research_resident_icp.py", args.solver_dll.resolve(), args.solver_dll.resolve().with_suffix(".build.json"),
        args.solver_dll.resolve().with_suffix(".cpp"), ROOT/"scripts/cuda_flat_grid_registration.py", ROOT/"scripts/research_flat_grid_nn.cu",
        ROOT/"scripts/validate_device_flat_grid_proof.py", ROOT/"scripts/benchmark_device_grid_resident_timing.py", ROOT/"scripts/DEVICE_FLAT_GRID_RESEARCH.md",
        ROOT/"scripts/benchmark_uniform_grid_nn.py", ROOT/"scripts/cuda_uniform_grid_registration.py", ROOT/"scripts/research_uniform_grid_nn.cu",
        ROOT/"scripts/UNIFORM_GRID_RESEARCH.md", ROOT/"scripts/benchmark_optix_nn.py", ROOT/"scripts/benchmark_parallel_fragments.py",
        ROOT/"scripts/benchmark_selective_fragment_threads.py", ROOT/"scripts/profile_fragment_verification.py",
        ROOT/"scanner_server/cuda_nn_registration.py", ROOT/"scripts/validate_uniform_grid_proof.py"]
    artifacts = {str(path.relative_to(ROOT)): file_hash(path) for path in artifact_paths}
    import cupy as cp
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server import fragments as module
    from scripts.cuda_device_flat_grid_registration import DOMAIN
    from scripts.device_flat_grid_synthetic import run_device_checks
    UniformGridICP = bridge_class(args.solver_dll, args.scratch_mib*1024**2, args.query_mib*1024**2)
    from scripts.benchmark_optix_nn import check_indices, evidence_agreement, canonical_evidence, pair_verdict, gpu_memory_snapshot
    from scripts.benchmark_selective_fragment_threads import fresh_fragments
    from scripts.profile_fragment_verification import array_digest, cloud_digest
    from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
    from scripts.validate_device_flat_grid_proof import validate_grid_proof, canonical_hash
    o3d.utility.set_max_threads(20)
    cv2.setNumThreads(20)
    report = {"kind": "standalone-original-double-device-flat-resident", "traversal": "device-flat-resident-v1", "status": "running", "source_sha256": source_before,
        "expected_source_sha256": args.expected_source_sha256, "component_source_sha256": component_before,
        "artifacts_sha256": artifacts, "domain": DOMAIN, "gpu": gpu_info(),
        "performance_attribution_valid": not (args.audit_nearest or args.audit_misses), "miss_policy": args.miss_policy,
        "cpu_hit_audit": args.audit_nearest, "cpu_miss_audit": args.audit_misses,
        "synthetic_only": args.synthetic_only, "synthetic_device_adapter_checked": args.check_device_adapter,
        "thread_policy": {"open3d": 20, "opencv": 20, "omp": os.environ["OMP_NUM_THREADS"]},
        "cache_policy": {"max_clouds": args.max_clouds, "retained_gpu_bytes": args.cache_mib*1024**2,
            "limit": "Retained sorted grid arrays plus original-order XYZ share this cap; transient launch/transfer buffers and CuPy pools separately observed"},
        "versions": {"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
        "authority": "Original raw fixed component with measured local poses only; all nine original proposals/order/gates, no archived live seeds or full Finish/mesh throughput claim",
        "real_runs": []}
    report["solver_library_path"] = str(args.solver_dll.resolve())
    report["resident_configuration"] = {"device": "CUDA:0", "max_points": 1000000,
        "max_scratch_bytes": args.scratch_mib*1024**2, "max_query_bytes": args.query_mib*1024**2, "gpu_timing": False}
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    original_match, solver, failure, fixture, raw_path = module._match, None, None, None, None
    save()
    try:
        report["host_enclosure"] = host_enclosure_checks()
        bindings = current_runtime_binding(report, cp, np, o3d)
        report["proof_bindings"], report["proof_bindings_sha256"] = bindings, canonical_hash(bindings)
        report["resident_math"] = dict(bindings["resident_math"])
        proof_authority = None
        if not args.synthetic_only:
            fixture, reference, raw_path, tasks, fixture_binding = load_fixture_scope(args, component_before)
            report.update(fixture_sha256=fixture_binding["fixture_sha256"], reference_sha256=fixture_binding["reference_sha256"],
                original_raw_input_sha256=fixture_binding["original_raw_input_sha256"], fixture_provenance=fixture["metadata"],
                fixture_binding=fixture_binding, fixture_binding_sha256=canonical_hash(fixture_binding))
            if args.proof_synthetic:
                proof_authority = validate_grid_proof(args.proof_synthetic, args.proof_bridge, bindings, fixture_binding)
                report["validated_proof_authority"] = {"bindings_sha256": proof_authority.bindings_sha256,
                    "fixture_binding_sha256": proof_authority.fixture_binding_sha256,
                    "synthetic_report_sha256": proof_authority.synthetic_report_sha256,
                    "bridge_report_sha256": proof_authority.bridge_report_sha256,
                    "target_membership_count": len(proof_authority.target_digests)}
        started = time.perf_counter()
        solver = UniformGridICP(max_clouds=args.max_clouds, max_cache_bytes=args.cache_mib*1024**2,
            audit_nearest=args.audit_nearest, audit_misses=args.audit_misses, miss_policy=args.miss_policy,
            proof_authority=proof_authority)
        report["setup_s"] = time.perf_counter()-started
        report["device"] = str(solver.device)
        report["solve_metadata"] = dict(solver.solve.metadata)
        if actual_resident_configuration(solver) != report["resident_configuration"]:
            raise RuntimeError("Actual resident/device query/point/timing configuration differs from its proof binding")
        if report["device"] != "CUDA:0":
            raise RuntimeError("Require actual CUDA0 used by the bound runtime/GPU identity")
        report["setup_partition"] = {"inherited_serial_adapter_setup_s": solver.inherited_adapter_setup_s,
            "flat_shader_compile_setup_s": solver.flat_shader_setup_s,
            "device_classifier_resident_eigen_setup_s": report["setup_s"]-solver.inherited_adapter_setup_s-solver.flat_shader_setup_s,
            "scope": "Includes original/flat construction, device classifier compile, unchanged resident kernels and pinned Eigen load; imports are outside solver setup"}
        report["synthetic_indices"] = check_indices(solver, args.check_device_adapter)
        report["synthetic_grid_boundaries"] = extra_device_cases(solver)
        report["synthetic_statistics"] = dict(solver.statistics)
        report["flat_cache_ownership"] = cache_ownership_cases(solver)
        report["device_flat_synthetic"] = run_device_checks(solver)
        if proof_authority is None and report["device_flat_synthetic"].get("passed") is not True:
            raise RuntimeError("Focused device/classifier/ownership synthetic proof did not pass")
        report["synthetic_gpu_memory"] = gpu_memory_snapshot(solver.device_id)
        solver.clear_cache()
        solver.target_membership_sha256.clear()
        save()
        print("Original CPU synthetic host/device/cell boundary gates passed", flush=True)
        if not args.synthetic_only:
            fixture_arrays_before = array_digest(fixture)
            cpu = fresh_fragments(fixture, baseline, module)
            gpu = fresh_fragments(fixture, baseline, module)
            cloud_before = (cloud_digest(cpu), cloud_digest(gpu))
            reference_proposals = {}
            reference_verdicts = {}
            for mode, repeat in [("native_cpu", 0)]+[("grid", i) for i in range(args.repeats)]:
                packed = cpu if mode == "native_cpu" else gpu
                restore_matches(fixture, packed, module)
                module._match = original_match if mode == "native_cpu" else solver.match
                before = dict(solver.statistics)
                resident_before = dict(solver.resident.statistics)
                run = {"mode": mode, "repeat": repeat, "pairs": [], "complete": False}
                report["real_runs"].append(run)
                started = time.perf_counter()
                for task in tasks:
                    a,b = task["pair"]
                    values, proposals = [], []
                    pair_started = time.perf_counter()
                    for index, pose in enumerate(task["proposals"]):
                        proposal_started = time.perf_counter()
                        value = module._verify_bridge(packed[a], packed[b], pose, fixture["camera"])
                        values.append(value)
                        quality = None if mode == "native_cpu" else evidence_agreement(reference_proposals[(task["position"],index)],value)
                        proposals.append({"proposal_index": index, "input_sha256": hashlib.sha256(pose.tobytes()).hexdigest(),
                            "elapsed_s": time.perf_counter()-proposal_started, "evidence": canonical_evidence(value), "quality": quality})
                        if mode == "native_cpu":
                            reference_proposals[(task["position"],index)] = value
                        print(f"{mode} repeat{repeat} pair{a}/{b} proposal{index}: {proposals[-1]['elapsed_s']:.3f}s", flush=True)
                    verdict = pair_verdict(values)
                    if mode == "native_cpu":
                        reference_verdicts[task["position"]] = verdict
                    same_verdict = all(verdict[key] == reference_verdicts[task["position"]][key] for key in
                        ("accepted", "ambiguous", "verified_proposals"))
                    run["pairs"].append({"position": task["position"], "pair": task["pair"], "proposal_results": proposals,
                                         "verdict": verdict, "pair_verdict_same": same_verdict, "elapsed_s": time.perf_counter()-pair_started})
                    save()
                run.update(complete=True, elapsed_s=time.perf_counter()-started,
                    statistics_delta={key: solver.statistics[key]-before[key] for key in solver.statistics},
                    gpu_memory_after=gpu_memory_snapshot(solver.device_id),
                    resident_statistics_delta={key: solver.resident.statistics[key]-resident_before[key] for key in solver.resident.statistics},
                    device_resident=solver.resident.report())
                if not solver.resident.gpu_timing:
                    run["resident_uncollected_statistics"] = ["gpu_transform_ms", "gpu_equations_ms"]
                    for key in run["resident_uncollected_statistics"]:
                        run["resident_statistics_delta"].pop(key, None)
                save()
            report["fixture_arrays_unchanged"] = fixture_arrays_before == array_digest(fixture)
            report["cloud_arrays_unchanged"] = cloud_before == (cloud_digest(cpu), cloud_digest(gpu))
            original_rows = [row for result in reference["results"] if result["configuration"] == "1x20"
                             for row in result["rows"] if row["position"] in (0,4)]
            native_rows = [{"pair": pair["pair"], "proposal_count": len(pair["proposal_results"]),
                **pair["verdict"]} for pair in report["real_runs"][0]["pairs"]]
            report["original_fixture_authority"] = baseline.compare_rows(original_rows, native_rows)
            report["real_target_membership_sha256"] = sorted(solver.target_membership_sha256)
            report["fixture_sha256_after"], report["reference_sha256_after"] = file_hash(args.fixture), file_hash(args.reference)
            report["original_raw_input_sha256_after"] = file_hash(raw_path)
            if (report["fixture_sha256_after"] != report["fixture_sha256"] or report["reference_sha256_after"] != report["reference_sha256"]
                    or report["original_raw_input_sha256_after"] != report["original_raw_input_sha256"]):
                raise RuntimeError("Fixture/reference bytes changed during measurement")
    except BaseException as exc:
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        module._match = original_match
        cleanup_failures = []
        def cleanup(name, action):
            try:
                return action()
            except BaseException as exc:
                cleanup_failures.append({"action": name, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                return None
        report["source_sha256_after"] = cleanup("core source hash", source_hash)
        report["component_source_sha256_after"] = cleanup("math source hash", baseline.numerical_source_hash)
        report["artifacts_sha256_after"] = cleanup("helper artifact hashes", lambda: {str(path.relative_to(ROOT)):file_hash(path) for path in artifact_paths})
        report["gpu_after"] = cleanup("GPU identity", gpu_info)
        report["proof_bindings_after"] = cleanup("installed CPU/CUDA runtime binding", lambda: current_runtime_binding(report, cp, np, o3d))
        report["resident_math_after"] = (dict(report["proof_bindings_after"]["resident_math"])
            if report["proof_bindings_after"] is not None else None)
        if solver is not None:
            report["statistics"] = dict(solver.statistics)
            report["device_resident"] = cleanup("resident metadata/source guard", solver.resident.report)
            report["resident_configuration_after"] = cleanup("actual resident configuration", lambda: actual_resident_configuration(solver))
            report["gpu_memory_before_close"] = cleanup("GPU memory snapshot", lambda:gpu_memory_snapshot(solver.device_id))
            cleanup("grid owned cache cleanup", solver.close)
        report["peak_process_rss_bytes"] = cleanup("process RSS", peak_rss_bytes)
        report["cleanup_failures"], report["cleanup_passed"] = cleanup_failures, not cleanup_failures
        passed = (failure is None and not cleanup_failures and report["proof_bindings_after"] == report.get("proof_bindings")
            and report.get("resident_configuration_after") == report["resident_configuration"]
            and report["gpu"] is not None and report["gpu_after"] == report["gpu"]
            and report["source_sha256_after"] == source_before and
            report["component_source_sha256_after"] == component_before and report["artifacts_sha256_after"] == artifacts and
            (args.synthetic_only or (report["fixture_arrays_unchanged"] and report["cloud_arrays_unchanged"] and
                report["original_fixture_authority"]["same_decisions_and_support"] and
                all(run["complete"] and run["resident_statistics_delta"]["calls"] > 0
                    and run["resident_statistics_delta"]["pose_iterations"] > 0
                    and run["resident_statistics_delta"]["cpu_fallback_calls"] == 0
                    and run["statistics_delta"]["device_calls"] > 0
                    and run["statistics_delta"]["device_malformed_results"] == 0
                    and run["device_resident"]["provenance"]["actual_retrieval_unchanged"] is True
                    and run["device_resident"]["provenance"]["original_resident_math_unchanged"] is True
                    and run["device_resident"]["provenance"]["source_unchanged"] is True
                    and all(pair["pair_verdict_same"] for pair in run["pairs"])
                    and all(p["quality"]["passed"] for pair in run["pairs"] for p in pair["proposal_results"])
                    for run in report["real_runs"] if run["mode"] == "grid"))))
        report["status"] = "passed" if passed else "failed"
        try:
            save()
        except BaseException as write_error:
            if failure is not None:
                failure.add_note(f"Final flat report write also failed: {write_error}")
                raise failure.with_traceback(failure.__traceback__) from write_error
            raise
    print(json.dumps({"output": str(args.output), "status": report["status"],
        "runs": [{key: run.get(key) for key in ("mode", "repeat", "elapsed_s")} for run in report["real_runs"]]}), flush=True)
    if not passed:
        raise RuntimeError("Uniform-grid exactness/provenance gates failed; diagnostic report preserved") from failure
    finish_cuda_worker()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.output.exists():
        parser.error("Require a fresh output path; preserve previous measured artifacts")
    try:
        run()
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        failed = {"kind": "standalone-original-double-device-flat-resident", "status": "failed",
            "traversal": "device-flat-resident-v1"}
        secondary = []
        if args.output.exists():
            try:
                failed = json.loads(args.output.read_text(encoding="utf-8"))
                if not isinstance(failed, dict):
                    raise ValueError("Current partial report is not an object")
            except BaseException as read_error:
                secondary.append({"action": "partial report read", "type": type(read_error).__name__, "message": str(read_error)})
                failed = {"kind": "flat-grid-early-failure"}
                backup = args.output.with_name(args.output.name+".partial-"+str(time.time_ns()))
                try:
                    backup.write_bytes(args.output.read_bytes())
                    failed["partial_report_preserved_at"] = str(backup)
                except BaseException as backup_error:
                    secondary.append({"action": "partial preservation", "type": type(backup_error).__name__, "message": str(backup_error)})
        failed.update(status="failed", producer_current_failure={"type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()}, producer_secondary_errors=secondary)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failed, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        except BaseException as write_error:
            print(json.dumps({"primary": failed["producer_current_failure"], "secondary": secondary,
                "report_write_error": str(write_error)}), file=sys.stderr, flush=True)
            raise error.with_traceback(error.__traceback__) from write_error
        raise


if __name__ == "__main__":
    main()

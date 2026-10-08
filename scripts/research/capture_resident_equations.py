"""Allocated output-only all-nine capture of unchanged original Eigen equations.

Reuses the frozen DeviceGridResident producer and validated component authority.
Only copies inputs/results and attaches observational proposal/call contexts.
No new solver, GPU math, correspondence, seed, convergence or graph authority.
The private NPZ is a micro-solver input fixture, never a live pose seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FROZEN = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
KIND = "observed-original-resident-eigen-equations-v1"
ARTIFACTS = ("scripts/research/capture_resident_equations.py", "scripts/research/resident_equation_recorder.py",
    "scripts/research/check_resident_equation_recorder.py", "scripts/profile_session.py", "scripts/process_metrics.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-synthetic", type=Path, required=True,
        help="Current helper-path/hash-bound Device synthetic proof; historical reports cannot authorize relocated helpers")
    parser.add_argument("--proof-bridge", type=Path, required=True,
        help="Matching current all-nine Device trajectory proof")
    parser.add_argument("--fixture", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    parser.add_argument("--reference", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json")
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--maximum-records", type=int, default=20000)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.maximum_records <= 20000:
        parser.error("Require an exclusively allocated native/GPU slot and bounded equation copies")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
        args.solver_dll.resolve(strict=True).relative_to(ROOT)
    except (OSError, ValueError):
        parser.error("Keep private fresh outputs and pinned solve library in this workspace")
    args.component_output = args.output.with_suffix(".component.json")
    args.array_output = args.output.with_suffix(".equations.npz")
    for path in (args.output, args.component_output, args.array_output):
        if path.exists():
            parser.error("Require fresh outputs; preserve earlier "+str(path))
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before exclusively allocated capture")
    if os.environ.get("OMP_NUM_THREADS", "8") != "8":
        parser.error("Require the original OMP8/OpenCV20/Open3D20 component policy")
    return args


def run(args):
    from scripts.profile_session import source_hash
    from scripts.research import benchmark_parallel_fragments as baseline
    from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
    from scripts.research.resident_equation_recorder import EquationRecorder, digest, restore_owned_hooks

    source_before, component_before = source_hash(), baseline.numerical_source_hash()
    if source_before != FROZEN:
        raise RuntimeError("Require frozen9331 production source before native imports")
    bridge_report = json.loads(args.proof_bridge.read_text(encoding="utf-8"))
    authority = validate_grid_proof(args.proof_synthetic, args.proof_bridge,
        bridge_report["proof_bindings"], bridge_report["fixture_binding"])
    bindings = authority.runtime_binding
    fixture_binding = bridge_report["fixture_binding"]
    actual_artifacts = {path: sha(ROOT/path) for path in ARTIFACTS}
    fixed_files = {str(path.resolve()): sha(path) for path in (args.proof_synthetic, args.proof_bridge,
        args.fixture, args.reference, args.solver_dll, args.solver_dll.with_suffix(".build.json"), args.solver_dll.with_suffix(".cpp"))}
    if (sha(args.fixture) != fixture_binding["fixture_sha256"] or sha(args.reference) != fixture_binding["reference_sha256"]
            or sha(args.solver_dll) != bindings["resident_math"]["solve_library_sha256"]):
        raise RuntimeError("Actual capture fixture/reference/Eigen DLL differ from the complete component proof")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": KIND, "status": "running", "start_utc": now(), "source_sha256": source_before,
        "component_source_sha256": component_before, "artifacts_sha256": actual_artifacts,
        "fixed_files_sha256": fixed_files, "proof_bindings": bindings, "fixture_binding": fixture_binding,
        "fixture_binding_sha256": authority.fixture_binding_sha256,
        "proof_synthetic_sha256": authority.synthetic_report_sha256, "proof_bridge_sha256": authority.bridge_report_sha256,
        "component_proofs": {"synthetic": {"path": str(args.proof_synthetic.resolve()), "sha256": authority.synthetic_report_sha256},
            "bridge": {"path": str(args.proof_bridge.resolve()), "sha256": authority.bridge_report_sha256}},
        "scope": "Output-only original Eigen solve observation during one separately authorized fixed all-nine resident component pass. Both native reference and unchanged resident gates run. Copy/hash/context overhead makes all timings diagnostic; this is not a full-session or CUDA speed result.",
        "performance_attribution_valid": False, "new_solve_authority": False, "previous_pose_is_actual": False,
        "proposal_scopes": [], "match_scopes": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    recorder = None
    hooks = []
    saved_argv = sys.argv[:]
    finalized = False
    terminal_finalized = False

    def replace(owner, name, replacement):
        original = getattr(owner, name)
        hooks.append((owner, name, original))
        setattr(owner, name, replacement)
        return original

    def restore(*, clear=True):
        failures = restore_owned_hooks(hooks, saved_argv, clear=clear)
        report.setdefault("hook_cleanup_failures", []).extend(
            {"type": type(error).__name__, "message": str(error)} for error in failures)
        report["hook_cleanup_passed"] = not report["hook_cleanup_failures"]
        report["observer_hooks_restored"] = not failures and report["hook_cleanup_passed"]
        return failures

    def finalize(primary=None):
        nonlocal finalized
        if finalized:
            return
        secondary = []
        if primary is not None:
            report["failure"] = {"type": type(primary).__name__, "message": str(primary), "traceback": traceback.format_exc()}
        try:
            if recorder is not None:
                report["equation_artifact"] = recorder.export(args.array_output)
                report["recorder"] = recorder.metadata()
            if args.component_output.exists():
                component = json.loads(args.component_output.read_text(encoding="utf-8"))
                # This is a NEW output of the inherited producer; no historical
                # report is changed. Observation overhead disallows attribution.
                component["performance_attribution_valid"] = False
                component["equation_observation"] = {"kind": KIND, "output": str(args.output),
                    "scope": "Original numerical outputs retained; observation copies/context hashes are included, so wall timers are diagnostic."}
                args.component_output.write_text(json.dumps(component, indent=2, allow_nan=False)+"\n", encoding="utf-8")
                report["component_report"] = {"path": str(args.component_output), "sha256": sha(args.component_output),
                    "status": component.get("status")}
                report["actual_runtime_binding"] = component.get("proof_bindings")
                report["actual_runtime_binding_after"] = component.get("proof_bindings_after")
                report["solve_metadata"] = component.get("solve_metadata")
                grid = [row for row in component.get("real_runs", []) if row.get("mode") == "grid"]
                native = [row for row in component.get("real_runs", []) if row.get("mode") == "native_cpu"]
                if (component.get("status") != "passed" or len(grid) != 1 or len(native) != 1
                        or component.get("cleanup_passed") is not True or component.get("fixture_binding") != fixture_binding
                        or component.get("proof_bindings") != bindings or component.get("proof_bindings_after") != bindings):
                    raise RuntimeError("Original all-nine component output/runtime/fixture/cleanup did not close")
                if recorder is None or not 0 < len(recorder.records) <= grid[0]["resident_statistics_delta"]["pose_iterations"]:
                    raise RuntimeError("Actual positive original solve count is inconsistent with resident iteration count")
                recorder.check()
                if recorder.delegates != 1 or any(row["status"] != 0 for row in recorder.records):
                    raise RuntimeError("Require one original solve instance and no rejected original calls for a passed microfixture")
                report["actual_component_counts"] = {"resident_pose_iterations": grid[0]["resident_statistics_delta"]["pose_iterations"],
                    "resident_calls": grid[0]["resident_statistics_delta"]["calls"], "actual_solve_calls": len(recorder.records),
                    "query_rows": grid[0]["statistics_delta"]["query_rows"], "nn_device_calls": grid[0]["statistics_delta"]["device_calls"]}
                expected = [(task["pair"], index, value) for task in fixture_binding["tasks"]
                    for index, value in enumerate(task["proposal_sha256"])]
                for mode in ("native_cpu", "grid"):
                    scopes = [row for row in report["proposal_scopes"] if row["mode"] == mode]
                    if [(row["pair"], row["proposal_index"], row["input_sha256"]) for row in scopes] != expected:
                        raise RuntimeError("Original ordered proposal contexts were incomplete or changed")
                    if not all(row.get("completed") is True for row in scopes):
                        raise RuntimeError("A proposal observation did not complete")
            else:
                raise RuntimeError("Original component report was not preserved")
        except BaseException as error:
            secondary.append(error)
        try:
            report["source_sha256_after"] = source_hash()
            report["component_source_sha256_after"] = baseline.numerical_source_hash()
            report["artifacts_sha256_after"] = {path: sha(ROOT/path) for path in ARTIFACTS}
            report["fixed_files_sha256_after"] = {path: sha(path) for path in fixed_files}
            if (report["source_sha256_after"] != source_before or report["component_source_sha256_after"] != component_before
                    or report["artifacts_sha256_after"] != actual_artifacts or report["fixed_files_sha256_after"] != fixed_files):
                raise RuntimeError("Capture source/core/component/proof/fixture/solve artifacts changed")
        except BaseException as error:
            secondary.append(error)
        report["closure_failures"] = [{"type": type(error).__name__, "message": str(error)} for error in secondary]
        report["status"] = "passed" if primary is None and not secondary else "failed"
        report["end_utc"] = now()
        try:
            save()
        except BaseException as write_error:
            if primary is not None:
                primary.add_note(f"Equation capture report write also failed: {write_error}")
                raise primary.with_traceback(primary.__traceback__) from write_error
            if secondary:
                secondary[0].add_note(f"Equation capture report write also failed: {write_error}")
                raise secondary[0] from write_error
            raise
        finalized = True
        if secondary:
            if primary is not None:
                primary.add_note(f"Equation capture closure also failed: {secondary}")
            else:
                raise secondary[0]

    try:
        import numpy as np
        from scanner_server import fragments
        from scripts import process_metrics
        from scripts.research import benchmark_device_grid_resident as producer
        from scripts.research import benchmark_device_grid_resident_timing as timing
        from scripts.research.archive import research_resident_icp as original_math

        recorder = EquationRecorder(np, max_records=args.maximum_records)
        original_eigen = original_math.EigenSolve
        original_bridge = producer.bridge_class
        original_verify = fragments._verify_bridge
        original_match = fragments._match
        original_finish = process_metrics.finish_cuda_worker
        proposal_counts = {"native_cpu": 0, "grid": 0}

        def observed_eigen(*values, **kwargs):
            return recorder.wrap(original_eigen(*values, **kwargs), original_math.ResidentFallback)

        def observed_bridge(*values, **kwargs):
            base = original_bridge(*values, **kwargs)

            class ObservedBridge(base):
                def match(self, source, target, initial):
                    recorder.check()
                    cloud = lambda value: {"points_sha256": digest(np.asarray(value.points)),
                        "normals_sha256": digest(np.asarray(value.normals)), "point_count": len(value.points)}
                    record = {"match_index": len(report["match_scopes"]), "source": cloud(source), "target": cloud(target),
                        "initial_sha256": digest(np.asarray(initial)), "first_record": len(recorder.records), "completed": False}
                    report["match_scopes"].append(record)
                    with recorder.scope(match_index=record["match_index"], source=record["source"], target=record["target"], initial_sha256=record["initial_sha256"]):
                        result = super().match(source, target, initial)
                        if cloud(source) != record["source"] or cloud(target) != record["target"] or digest(np.asarray(initial)) != record["initial_sha256"]:
                            raise RuntimeError("Original match inputs changed during observational solve capture")
                        record.update(completed=True, end_record=len(recorder.records), transformation_sha256=digest(np.asarray(result.transformation)))
                        return result

            return ObservedBridge

        def observed_verify(source, target, initial, *values, **kwargs):
            mode = "native_cpu" if fragments._match is original_match else "grid"
            sequence = proposal_counts[mode]
            proposal_counts[mode] += 1
            expected = [(task["pair"], index) for task in fixture_binding["tasks"] for index in range(len(task["proposal_sha256"]))]
            if sequence >= len(expected) or [source.index, target.index] != expected[sequence][0]:
                raise RuntimeError("Original proposal membership/order changed before equation observation")
            record = {"mode": mode, "sequence": sequence, "pair": [source.index, target.index],
                "proposal_index": expected[sequence][1], "input_sha256": digest(np.asarray(initial)),
                "first_record": len(recorder.records), "completed": False}
            report["proposal_scopes"].append(record)
            with recorder.scope(mode=mode, sequence=sequence, pair=record["pair"], proposal_index=record["proposal_index"], input_sha256=record["input_sha256"]):
                result = original_verify(source, target, initial, *values, **kwargs)
                record.update(completed=True, end_record=len(recorder.records))
                return result

        def finish_capture():
            nonlocal terminal_finalized
            # The old worker exits only after numerical cleanup and complete
            # inherited gates. Close private copies and restore hooks first.
            primary = None
            try:
                finalize()
                terminal_finalized = True
            except BaseException as error:
                primary = error
                raise
            finally:
                # On non-Windows the inherited timing finally restores its
                # saved finish callback; retain our list for the outer restore.
                failures = restore(clear=False)
                if failures:
                    report["status"] = "failed"
                try:
                    save()
                except BaseException as write_error:
                    failures.append(write_error)
                if failures:
                    if primary is not None:
                        primary.add_note(f"Equation observer hook/report cleanup also failed: {failures}")
                    else:
                        raise failures[0]
            original_finish()

        replace(original_math, "EigenSolve", observed_eigen)
        replace(producer, "bridge_class", observed_bridge)
        replace(fragments, "_verify_bridge", observed_verify)
        replace(process_metrics, "finish_cuda_worker", finish_capture)
        sys.argv = [str(ROOT/"scripts/research/benchmark_device_grid_resident_timing.py"), "--run-allocated",
            "--output", str(args.component_output), "--proof-synthetic", str(args.proof_synthetic),
            "--proof-bridge", str(args.proof_bridge), "--fixture", str(args.fixture), "--reference", str(args.reference),
            "--solver-dll", str(args.solver_dll), "--repeats", "1", "--miss-policy", "direct-miss-research-v1"]
        timing.main()
        if not terminal_finalized:
            raise RuntimeError("Original closed CUDA worker terminal finalization was not reached")
    except BaseException as primary:
        try:
            finalize(primary)
        except BaseException as secondary:
            if secondary is not primary:
                primary.add_note(f"Equation capture finalization also failed: {type(secondary).__name__}: {secondary}")
        raise
    finally:
        primary = sys.exception()
        failures = restore()
        if failures:
            report["status"] = "failed"
            try:
                save()
            except BaseException as write_error:
                failures.append(write_error)
            if primary is not None:
                primary.add_note(f"Final equation observer restoration also failed: {failures}")
            else:
                raise failures[0]


def main():
    args = parse()
    try:
        run(args)
    except BaseException as primary:
        # Preserve failures that occur before native imports or hook setup.
        partial = {}
        if args.output.exists():
            try:
                partial = json.loads(args.output.read_text(encoding="utf-8"))
            except BaseException as read_error:
                primary.add_note(f"Partial capture report read also failed: {read_error}")
        partial.update(kind=KIND, status="failed", outer_failure={"type": type(primary).__name__,
            "message": str(primary), "traceback": traceback.format_exc()}, end_utc=now())
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(partial, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        except BaseException as write_error:
            primary.add_note(f"Early capture failure report write also failed: {write_error}")
        raise


if __name__ == "__main__":
    main()

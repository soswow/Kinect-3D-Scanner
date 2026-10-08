"""Proof-authorized timing wrapper for the unchanged uniform-grid harness.

The producer's synthetic check insists mutations upload an index, which is
inappropriate for proof-excluded CPU-only synthetic targets in timing. This
driver omits only that redundant synthetic setup after separate proofs validate.
It leaves the original native/grid nine-proposal loop and every final gate intact.
No production module imports this driver.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-synthetic", type=Path, required=True)
    parser.add_argument("--proof-bridge", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.output.exists():
        parser.error("Require a fresh output path; an old result must not be annotated as this run")
    from scripts.validate_parallel_staged_grid_proof import validate_grid_proof

    driver_path = Path(__file__)
    harness_path = driver_path.with_name("benchmark_parallel_staged_grid_nn.py")
    driver_before, harness_before = file_hash(driver_path), file_hash(harness_path)
    bridge = json.loads(args.proof_bridge.read_text())
    authority = validate_grid_proof(args.proof_synthetic, args.proof_bridge,
                                   bridge["proof_bindings"], bridge["fixture_binding"])
    artifact_key = str(harness_path.relative_to(ROOT))
    if dict(authority.artifact_sha256).get(artifact_key) != harness_before:
        raise RuntimeError("Current proof-producing harness differs from the successful proofs")

    from scripts import benchmark_parallel_staged_grid_nn as harness
    from scripts import benchmark_optix_nn as helpers
    from scripts import process_metrics
    original_indices = helpers.check_indices
    original_boundaries = harness.extra_device_cases
    original_finish = process_metrics.finish_cuda_worker
    skipped = []

    def separate_proof_setup(solver, *_):
        actual = solver.proof_authority
        if actual != authority or solver.audit_nearest or solver.audit_misses:
            raise RuntimeError("Timing setup omission requires current validated non-audited proof authority")
        skipped.append("host/device nearest synthetic setup")
        return []

    def separate_boundary_setup(solver):
        if solver.proof_authority != authority:
            raise RuntimeError("Boundary setup omission requires current validated proof authority")
        skipped.append("signed-cell/min-radius/subnormal synthetic setup")
        return []

    def finalize_report():
        report = json.loads(args.output.read_text())
        driver_after, harness_after = file_hash(driver_path), file_hash(harness_path)
        unchanged = driver_before == driver_after and harness_before == harness_after
        report["timing_driver"] = {
            "path": str(driver_path.relative_to(ROOT)), "sha256": driver_before,
            "sha256_after": driver_after, "unchanged": unchanged,
            "proof_producing_harness_path": str(harness_path.relative_to(ROOT)),
            "proof_producing_harness_sha256": harness_before,
            "proof_producing_harness_sha256_after": harness_after,
            "changed_scope": "Omit redundant synthetic setup; optional source-bound HOST stage-decoding toggle only. Original native/grid all-nine proposal loop, comparisons and source/input/cleanup gates execute unchanged",
        }
        report["synthetic_setup_omitted"] = {
            "scope": "Separate unchanged GPU/CPU-audited synthetic proof supplies coverage; this timing run claims no fresh synthetic GPU coverage",
            "omitted_checks": skipped+["restricted interval and unsupported-inner-stage guards", "budget/full-radius CPU fallback guards", "grid/table/global-cache ownership and absent-final-pointer guards", "HOST-only diagnostic on/off GPU output equivalence"],
            "synthetic_report_sha256": authority.synthetic_report_sha256,
            "bridge_report_sha256": authority.bridge_report_sha256,
        }
        if not unchanged:
            report["status"] = "failed"
            report["timing_driver_guard_failure"] = "Timing driver or unchanged producer harness changed during execution"
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return unchanged

    def guarded_finish():
        if not finalize_report():
            raise RuntimeError("Timing driver provenance gate failed; diagnostics preserved")
        original_finish()

    helpers.check_indices = separate_proof_setup
    harness.extra_device_cases = separate_boundary_setup
    process_metrics.finish_cuda_worker = guarded_finish
    try:
        print("Separate synthetic and bridge proofs validated; redundant synthetic timing setup omitted", flush=True)
        harness.main()
    except BaseException as error:
        import traceback
        import time
        primary={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        secondary=[]
        failed={"kind":"staged-grid-timing-early-failure","timing_driver_sha256":driver_before}
        if args.output.exists():
            try:
                finalize_report()
            except BaseException as cleanup_error:
                secondary.append({"action":"driver report finalization","type":type(cleanup_error).__name__,
                    "message":str(cleanup_error),"traceback":traceback.format_exc()})
            try:
                existing=json.loads(args.output.read_text())
                if not isinstance(existing,dict):
                    raise ValueError("Current report is not a JSON object")
                failed=existing
            except BaseException as read_error:
                secondary.append({"action":"current report read","type":type(read_error).__name__,
                    "message":str(read_error),"traceback":traceback.format_exc()})
                try:
                    backup=args.output.with_name(args.output.name+".partial-"+str(time.time_ns()))
                    backup.write_bytes(args.output.read_bytes())
                    failed["partial_report_preserved_at"]=str(backup)
                except BaseException as backup_error:
                    secondary.append({"action":"partial report preservation","type":type(backup_error).__name__,
                        "message":str(backup_error),"traceback":traceback.format_exc()})
        failed["status"]="failed"
        failed["timing_driver_current_failure"]=primary
        failed["timing_driver_secondary_errors"]=secondary
        try:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(failed,indent=2,allow_nan=False)+"\n")
        except BaseException as write_error:
            print(json.dumps({"primary":primary,"secondary":secondary,"report_write_error":str(write_error)}),file=sys.stderr,flush=True)
            raise error.with_traceback(error.__traceback__) from write_error
        raise
    finally:
        helpers.check_indices = original_indices
        harness.extra_device_cases = original_boundaries
        process_metrics.finish_cuda_worker = original_finish


if __name__ == "__main__":
    main()

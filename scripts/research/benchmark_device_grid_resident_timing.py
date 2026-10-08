"""Proof-authorized separate device-flat resident all-nine timing driver.

The producer's synthetic check insists mutations upload an index, which is
inappropriate for proof-excluded CPU-only synthetic targets in timing. This
driver omits only that redundant synthetic setup after separate proofs validate.
It leaves the original native/grid nine-proposal loop and every final gate intact.
No production module imports this driver.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import hashlib
import json
import time
import traceback



def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    from scripts.research.validate_device_flat_grid_proof import validate_grid_proof

    driver_path = Path(__file__)
    harness_path = driver_path.with_name("benchmark_device_grid_resident.py")
    driver_before, harness_before = file_hash(driver_path), file_hash(harness_path)
    bridge = json.loads(args.proof_bridge.read_text(encoding="utf-8"))
    authority = validate_grid_proof(args.proof_synthetic, args.proof_bridge,
                                   bridge["proof_bindings"], bridge["fixture_binding"])
    artifact_key = str(harness_path.relative_to(ROOT))
    if dict(authority.artifact_sha256).get(artifact_key) != harness_before:
        raise RuntimeError("Current proof-producing harness differs from the successful proofs")

    if dict(authority.artifact_sha256).get(str(driver_path.relative_to(ROOT))) != driver_before:
        raise RuntimeError("Current timing driver differs from the successful flat proofs")
    from scripts.research import benchmark_device_grid_resident as harness
    from scripts.research.archive import benchmark_optix_nn as helpers
    from scripts import process_metrics
    from scripts.research import device_flat_grid_synthetic as focused
    original_indices = helpers.check_indices
    original_boundaries = harness.extra_device_cases
    original_owned = harness.cache_ownership_cases
    original_device = focused.run_device_checks
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

    def separate_owned_setup(solver):
        if solver.proof_authority != authority or solver.audit_nearest or solver.audit_misses:
            raise RuntimeError("Cache setup omission requires separately validated unaudited flat authority")
        skipped.append("inherited cache cloud eviction/clear and one-byte-budget CPU fallback")
        return {"omitted": True, "separate_synthetic_report_sha256": authority.synthetic_report_sha256}

    def separate_device_setup(solver):
        if solver.proof_authority != authority or solver.audit_nearest or solver.audit_misses:
            raise RuntimeError("Device-focused setup omission requires separately validated device-resident trajectory authority")
        skipped.append("device classifier/packet/scatter/copy-only-flagged/policy/budget/XYZ ownership synthetic checks")
        return {"omitted": True, "separate_synthetic_report_sha256": authority.synthetic_report_sha256}

    def finalize_report():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        driver_after, harness_after = file_hash(driver_path), file_hash(harness_path)
        unchanged = driver_before == driver_after and harness_before == harness_after
        report["timing_driver"] = {
            "path": str(driver_path.relative_to(ROOT)), "sha256": driver_before,
            "sha256_after": driver_after, "unchanged": unchanged,
            "proof_producing_harness_path": str(harness_path.relative_to(ROOT)),
            "proof_producing_harness_sha256": harness_before,
            "proof_producing_harness_sha256_after": harness_after,
            "changed_scope": "Omit redundant synthetic setup only; original native/grid all-nine proposal loop, comparisons and source/input/cleanup gates execute unchanged",
        }
        report["synthetic_setup_omitted"] = {
            "scope": "Separate unchanged GPU/CPU-audited synthetic proof supplies coverage; this timing run claims no fresh synthetic GPU coverage",
            "omitted_checks": skipped,
            "synthetic_report_sha256": authority.synthetic_report_sha256,
            "bridge_report_sha256": authority.bridge_report_sha256,
        }
        if not unchanged:
            report["status"] = "failed"
            report["timing_driver_guard_failure"] = "Timing driver or unchanged producer harness changed during execution"
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return unchanged

    def guarded_finish():
        if not finalize_report():
            raise RuntimeError("Timing driver provenance gate failed; diagnostics preserved")
        original_finish()

    helpers.check_indices = separate_proof_setup
    harness.extra_device_cases = separate_boundary_setup
    harness.cache_ownership_cases = separate_owned_setup
    focused.run_device_checks = separate_device_setup
    process_metrics.finish_cuda_worker = guarded_finish
    try:
        print("Separate synthetic and bridge proofs validated; redundant synthetic timing setup omitted", flush=True)
        harness.main()
    except BaseException as error:
        if args.output.exists():
            try:
                finalize_report()
            except BaseException as cleanup_error:
                error.add_note(f"Timing report finalization also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        raise
    finally:
        helpers.check_indices = original_indices
        harness.extra_device_cases = original_boundaries
        harness.cache_ownership_cases = original_owned
        focused.run_device_checks = original_device
        process_metrics.finish_cuda_worker = original_finish


def preserve_failure(args, error):
    primary = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    secondary = []
    failed = {"kind": "device-flat-resident-timing-early-failure", "traversal": "device-flat-resident-v1"}
    try:
        failed["timing_driver_sha256"] = file_hash(Path(__file__))
    except BaseException as hash_error:
        secondary.append({"action": "timing driver hash", "type": type(hash_error).__name__, "message": str(hash_error)})
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("Current report is not a JSON object")
            failed = existing
        except BaseException as read_error:
            secondary.append({"action": "current report read", "type": type(read_error).__name__, "message": str(read_error)})
            try:
                backup = args.output.with_name(args.output.name+".partial-"+str(time.time_ns()))
                backup.write_bytes(args.output.read_bytes())
                failed["partial_report_preserved_at"] = str(backup)
            except BaseException as backup_error:
                secondary.append({"action": "partial report preservation", "type": type(backup_error).__name__, "message": str(backup_error)})
    failed.update(status="failed", timing_driver_current_failure=primary, timing_driver_secondary_errors=secondary)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failed, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except BaseException as write_error:
        print(json.dumps({"primary": primary, "secondary": secondary, "report_write_error": str(write_error)}), file=sys.stderr, flush=True)
        raise error.with_traceback(error.__traceback__) from write_error


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Additional fixture/cache/audit options are accepted from the full producer; run "
               "python scripts/research/benchmark_device_grid_resident.py --help for those options.")
    parser.add_argument("--run-allocated", action="store_true",
        help="Confirm an exclusively allocated hardware slot; required for measurements.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-synthetic", type=Path, required=True)
    parser.add_argument("--proof-bridge", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if "--run-allocated" not in sys.argv:
        parser.error("Require parent-allocated exclusive hardware slot")
    if args.output.exists():
        parser.error("Require a fresh output path; an old result must not be annotated as this run")
    try:
        run(args)
    except BaseException as error:
        preserve_failure(args, error)
        raise


if __name__ == "__main__":
    main()

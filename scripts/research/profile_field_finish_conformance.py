"""New field-conformance supervisor for the frozen checkpoint Finish runner.

Run with --proposal-policy original (default) or canonical-fresh-1um, then --
and the complete allocated native/audit/timing checkpoint-runner arguments.
Timing requires a NEW conformance audit envelope and quality proof; old failed
strict Finish quality and old component tokens alone cannot authorize it.

Checkpoint materialization, original gate bodies, scalar/bulk CPU shadows and
resident numerical math are delegated unchanged. The optional micrometre
policy changes only private coarse proposal inputs and is declared separately.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KIND = "offline-field-finish-actual-input-conformance-v2"
NEW_ARTIFACTS = (
    "scripts/research/profile_field_finish_conformance.py",
    "scripts/research/field_finish_conformance_scope.py",
    "scripts/research/validate_field_finish_conformance.py",
    "scripts/research/compare_field_finish_conformance.py",
    "scripts/research/canonical_fpfh_proposals.py",
    "scripts/process_metrics.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option(arguments, name):
    if arguments.count(name) != 1:
        raise ValueError(f"Require exactly one {name} argument")
    index = arguments.index(name)
    if index + 1 == len(arguments):
        raise ValueError(f"Require value for {name}")
    return arguments[index + 1]


@contextmanager
def owned_patches(patches, failures):
    """Transactional installation; attempt every restore after partial entry."""
    originals = []
    primary = None
    try:
        for owner, name, value in patches:
            previous = getattr(owner, name)
            originals.append((owner, name, previous))
            setattr(owner, name, value)
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        secondary = []
        for owner, name, previous in reversed(originals):
            try:
                setattr(owner, name, previous)
                if getattr(owner, name) is not previous:
                    raise RuntimeError(f"Owned {name} hook identity was not restored")
            except BaseException as error:
                failures.append({"type": type(error).__name__, "message": str(error), "hook": name})
                secondary.append(error)
        if secondary:
            if primary is not None:
                for error in secondary:
                    primary.add_note(f"Supervisor restoration also failed: {error}")
            else:
                raise RuntimeError("Field supervisor hook restoration failed") from secondary[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-policy", choices=("original", "canonical-fresh-1um"), default="original")
    parser.add_argument("finish_arguments", nargs=argparse.REMAINDER,
                        help="After --, original checkpoint runner arguments; --finish-audit references a NEW .conformance.json")
    args = parser.parse_args()
    arguments = args.finish_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if "--help" in arguments or "-h" in arguments:
        from scripts.research.profile_checkpoint_resident_finish import main as base_main
        with owned_patches([(sys, "argv", ["profile_checkpoint_resident_finish.py", "--help"])], []):
            base_main()
        return
    try:
        mode, output = option(arguments, "--mode"), Path(option(arguments, "--output")).resolve()
        if mode not in ("native", "audit", "timing") or "--run-allocated" not in arguments:
            raise ValueError("Require explicit allocation and native/audit/timing only; use original capture runner")
        output.relative_to(ROOT / "benchmark-output")
    except ValueError as error:
        parser.error(str(error))
    envelope_path = output.with_suffix(".conformance.json")
    delegated_path = output.with_suffix(".resident.json")
    if envelope_path.exists() or delegated_path.exists() or output.exists():
        parser.error("Preserve all previous artifacts; require fresh output paths")

    # All these imports are stdlib-only. Numerical imports remain behind the
    # delegated runner's allocation/preflight and original measurement boundary.
    from scripts.research import profile_checkpoint_resident_finish as base
    from scripts.research import validate_checkpoint_finish_proof as checkpoint_guard
    from scripts.research import validate_field_finish_conformance as field_guard
    from scripts.research.field_finish_conformance_scope import FieldConformanceScope
    from scripts import process_metrics

    delegated_artifacts = {name: sha(ROOT / name) for name in checkpoint_guard.FINISH_ARTIFACTS}
    pins = {name: sha(ROOT / name) for name in (*checkpoint_guard.FINISH_ARTIFACTS, *NEW_ARTIFACTS)}
    report = {"kind": KIND, "status": "running", "mode": mode,
        "proposal_policy": args.proposal_policy, "artifacts_sha256": pins,
        "delegated_artifacts_sha256": delegated_artifacts,
        "supervisor_restored": False, "cleanup_failures": [], "failure": None,
        "performance_attribution_valid": mode in ("native", "timing"),
        "native_history_equivalence_claimed": False,
        "scope": "New actual-input field-conformance protocol. Every audited actual GPU query and complete ICP result is shadowed by original CPU on identical inputs/seeds. Original acceptance/witness/information bodies/settings execute unchanged. Native intermediate history differences are retained as diagnostics; final discrete reconstruction graph/witness/coverage, pose bounds and fixed-coordinate physical surfaces require independent NEW quality closure. Timing must reproduce the exact audited GPU call inputs/order and its own gate/result trajectory. Checkpoint materialization, source/proof scans after Finish and output writing are outside finish_s; all optional canonical preparation and field timing authority validation are inside finish_s."}
    scopes, worker_requests, minted = [], [], []
    original_worker = process_metrics.finish_cuda_worker
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        envelope_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def deferred_worker():
        worker_requests.append(True)

    class SupervisedScope(FieldConformanceScope):
        def __init__(self, *positional, **keywords):
            from scanner_server import refinement
            from scanner_server.engine import ScanEngine
            super().__init__(*positional, proposal_policy=args.proposal_policy,
                             graph_module=refinement.REG, final_engine_type=ScanEngine, **keywords)
            scopes.append(self)

    def new_authority(audit_path, component_authority, expected_scope_binding, expected_artifacts, **keywords):
        if expected_artifacts != delegated_artifacts:
            raise ValueError("Delegated checkpoint dependency scope differs from frozen original fifteen pins")
        token = field_guard.validate_field_finish_conformance(audit_path, component_authority,
            expected_scope_binding, pins, proposal_policy=args.proposal_policy, **keywords)
        if type(token) is not field_guard.FieldFinishConformanceAuthority:
            raise ValueError("Require distinct NEW field conformance authority, not old strict Finish authority")
        minted.append(token)
        return SimpleNamespace(registration_authority=token)

    save()
    primary = None
    try:
        with owned_patches([
            (base, "KIND", KIND), (base, "FinishRegistrationScope", SupervisedScope),
            (checkpoint_guard, "validate_checkpoint_finish_proof", new_authority),
            (process_metrics, "finish_cuda_worker", deferred_worker),
            (sys, "argv", [str(Path(base.__file__).resolve()), *arguments]),
        ], report["cleanup_failures"]):
            base.main()
        report["supervisor_restored"] = True
        if len(scopes) != 1 or not scopes[0].complete or not scopes[0].restored or scopes[0].failure is not None:
            raise ValueError("Full original Finish context did not close healthy and restored")
        if not scopes[0].policy_restored or report["cleanup_failures"]:
            raise ValueError("Field proposal/supervisor restoration failed")
        if len(minted) != int(mode == "timing"):
            raise ValueError("Unexpected field authority validation path")
        if sys.platform == "win32" and worker_requests != [True]:
            raise ValueError("Expected one deferred successful terminal worker request")
        delegated = json.loads(delegated_path.read_text(encoding="utf-8"))
        if (delegated["kind"] != KIND or delegated["mode"] != mode or delegated["status"] != "complete"
                or delegated.get("failure") is not None or delegated.get("cleanup_failures")
                or not delegated["runner_hooks_restored"]):
            raise ValueError("Delegated Finish did not close under distinct conformance kind")
        if not (delegated["artifacts_sha256"] == delegated["artifacts_sha256_after"] == delegated_artifacts):
            raise ValueError("Delegated source closure differs")
        for name in ("checkpoint", "scope_binding", "scope_binding_sha256", "runtime_binding",
                     "runtime_binding_after", "component_proof", "component_proof_after", "profile"):
            report[name] = delegated[name]
        report["actual_trace"] = delegated["trace"]
        report["delegated_sidecar"] = {"path": str(delegated_path), "sha256": sha(delegated_path)}
        report["registration"] = scopes[0].report()
        report["final_pose_inventory"] = scopes[0].final_poses
        report["finish_s"] = delegated["finish_s"]
        if minted:
            report["timing_authority"] = {"audit_sha256": minted[0].conformance_audit_report_sha256,
                                           "quality_sha256": minted[0].quality_proof_sha256}
        report["artifacts_sha256_after"] = {name: sha(ROOT / name) for name in pins}
        if report["artifacts_sha256_after"] != pins:
            raise ValueError("Field conformance sources changed during measurement")
        report["status"] = "complete"
    except BaseException as error:
        primary = error
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["deferred_terminal_worker_calls"] = len(worker_requests)
        if scopes:
            try:
                report["registration"] = scopes[0].report()
                report["final_pose_inventory"] = scopes[0].final_poses
            except BaseException as error:
                report["cleanup_failures"].append({"type": type(error).__name__, "message": str(error)})
                report["status"] = "failed"
                if primary is not None:
                    primary.add_note(f"Final field context reporting also failed: {error}")
                else:
                    primary = error
        try:
            save()
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"Final conformance status write also failed: {error}")
            else:
                raise
    if primary is not None:
        raise primary
    if worker_requests:
        original_worker()


if __name__ == "__main__":
    main()

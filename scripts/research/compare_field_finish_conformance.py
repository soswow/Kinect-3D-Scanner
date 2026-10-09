"""Close NEW field final-output conformance from separate native/audit Finish.

Require the same bit-verified fresh Live checkpoint, runtime and declared
proposal policy. Compare actual retained graph/witness/coverage/stage decisions
and unrounded Final poses; intermediate native histories remain diagnostic.
Surface comparison uses 30,000 fixed-coordinate samples and no alignment.
Only an allocated invocation imports NumPy/Open3D for mesh comparison.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json

from scripts.research import validate_field_finish_conformance as guard
from scripts.research import validate_checkpoint_finish_proof as checkpoint
from scripts.research import validate_finish_resident_proof as original
from scripts.research.validate_device_flat_grid_proof import validate_grid_proof, normalized_artifacts
from scripts.research.validate_bulk_resident_audit import validate_bulk_resident_audit
from scripts.research.archive.validate_uniform_grid_proof import require, file_hash


def closed_inputs(native_path, audit_path):
    first = original.read_json(audit_path)
    artifacts = normalized_artifacts(first["artifacts_sha256"])
    require(set(artifacts) == set(guard.FINISH_ARTIFACTS), "Separate old15/new6 field source scope absent")
    require(all(file_hash(ROOT/name) == value for name,value in artifacts.items()), "Actual field producer source changed")
    proof = first["component_proof"]
    runtime = first["runtime_binding"]
    component_report = original.read_json(proof["bridge"]["path"])
    component = validate_grid_proof(proof["synthetic"]["path"],proof["bridge"]["path"],runtime,component_report["fixture_binding"])
    old_pins = {name:artifacts[name] for name in checkpoint.FINISH_ARTIFACTS}
    authority = checkpoint.validate_checkpoint_manifest(first["checkpoint"]["path"],runtime,old_pins)
    tracked = {}
    policy = first["proposal_policy"]
    require(policy in guard.PROPOSAL_POLICIES, "Unknown declared proposal policy")
    native,_,_,native_gates,native_graphs,native_rows = guard.closed_field_envelope(native_path,"native",authority,artifacts,proof,tracked,policy)
    audit,delegated,_,audit_gates,audit_graphs,audit_rows = guard.closed_field_envelope(audit_path,"audit",authority,artifacts,proof,tracked,policy)
    require(native["scope_binding"] == audit["scope_binding"], "Native/audit did not use the same exact measured Live scope")
    bulk = None
    if delegated.get("cpu_query_auditor",{}).get("mode") == "bulk-shadow":
        path = original.closed_file(delegated["cpu_query_auditor"]["proof"],tracked)
        report = original.read_json(path)
        bulk = validate_bulk_resident_audit(path,delegated["bulk_runtime_binding"],report["fixture_binding"])
    matches = [row for row in audit_rows if row["event"] == "match"]
    checkpoint.validate_field_cpu_auditor(delegated,matches,bulk,runtime,artifacts,tracked)
    require(component.synthetic_report_sha256 == proof["synthetic"]["sha256"]
            and component.bridge_report_sha256 == proof["bridge"]["sha256"], "Actual component proof record differs")
    return native,audit,native_gates,audit_gates,native_graphs,audit_graphs,authority,artifacts,tracked


def history_diagnostic(native, audit):
    """Count retained raw history differences without turning them into gates."""
    left = [(row["gate_index"],row["function"],row["result"]) for row in native]
    right = [(row["gate_index"],row["function"],row["result"]) for row in audit]
    return {"native_gate_calls":len(left),"audit_gate_calls":len(right),
        "identical_raw_records_in_common_prefix":sum(a == b for a,b in zip(left,right)),
        "raw_history_equal":left == right,"native_history_equivalence_claimed":False,
        "scope":"Diagnostic only; new authority requires final graph/witness/coverage/stage/pose and physical surface criteria."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_envelope",type=Path)
    parser.add_argument("audit_envelope",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-allocated",action="store_true")
    args = parser.parse_args()
    if not args.run_allocated or args.output.exists():
        parser.error("Require an exclusive allocated surface-comparison slot and fresh output")
    paths = [path.resolve(strict=True) for path in (args.native_envelope,args.audit_envelope)]
    require(paths[0] != paths[1] and args.output.resolve() not in paths, "Cannot compare one reference with itself or overwrite inputs")
    script_hash = file_hash(Path(__file__))
    result = {"kind":guard.QUALITY_KIND,"status":"failed","failure":None,
        "native_envelope":{"path":str(paths[0]),"sha256":file_hash(paths[0])},
        "audit":{"path":str(paths[1]),"sha256":file_hash(paths[1])},
        "comparison_artifact_sha256":script_hash,"native_history_equivalence_claimed":False,
        "declared_output_criteria":dict(guard.OUTPUT_CRITERIA),
        "note":"Independent original native Finish is a reference, not ground truth. No fitting, rounded seed or history-equivalence claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save()
    try:
        native,audit,native_gates,audit_gates,native_graphs,audit_graphs,authority,artifacts,tracked = closed_inputs(*paths)
        evidence = guard.final_output_conformance(native,audit,native_graphs,audit_graphs)
        result.update(checkpoint=authority.record,proposal_policy=audit["proposal_policy"],
            scope_binding_sha256=audit["scope_binding_sha256"],candidate=dict(audit["profile"]),native=dict(native["profile"]),
            candidate_trace=dict(audit["actual_trace"]),native_trace=dict(native["actual_trace"]),
            candidate_sidecar=dict(audit["delegated_sidecar"]),native_sidecar=dict(native["delegated_sidecar"]),
            final_output_conformance=evidence,intermediate_history_diagnostic=history_diagnostic(native_gates,audit_gates))
        require(evidence["passed"], "Actual final graph/witness/stage/view or unrounded pose criteria failed")
        # Numerical runtime is loaded only after complete source/actual-input,
        # scalar-or-bulk audit, materialization and final discrete proofs close.
        import numpy as np
        import open3d as o3d
        from scripts.profile_session import compare_quality
        from shared.surface_metrics import surface_metrics
        profile = original.read_json(audit["profile"]["path"])
        reference = original.read_json(native["profile"]["path"])
        comparison = compare_quality(profile,Path(native["profile"]["path"]),np)
        comparison.update(candidate_report_sha256=audit["profile"]["sha256"],baseline_report_sha256=native["profile"]["sha256"])

        def mesh(record):
            with np.load(record["geometry"]["artifact"],allow_pickle=False) as data:
                value = o3d.geometry.TriangleMesh()
                value.vertices = o3d.utility.Vector3dVector(data["points"])
                value.triangles = o3d.utility.Vector3iVector(data["faces"])
            return value

        comparison["triangle_surface_metrics_vs_cpu"] = surface_metrics(mesh(profile),mesh(reference),threshold_m=.005,samples=30000)
        result["comparison"] = comparison
        guard.validate_surface_comparison(comparison,audit,native)
        require(file_hash(Path(__file__)) == script_hash and all(file_hash(ROOT/name) == value for name,value in artifacts.items())
                and all(file_hash(path) == value for path,value in tracked.items()), "Comparator/input/current source bytes changed during comparison")
        result["inputs_sha256"] = {str(path):value for path,value in tracked.items()}
        result["status"] = "passed"
        save()
    except BaseException as error:
        result.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        try:
            save()
        except BaseException as write_error:
            error.add_note(f"Final conformance quality write also failed: {write_error}")
            raise error from write_error
        raise


if __name__ == "__main__":
    main()

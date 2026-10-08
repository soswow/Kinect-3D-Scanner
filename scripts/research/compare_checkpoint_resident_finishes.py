"""Compare independent original/resident Finish from one exact Live checkpoint.

No alignment, seed rounding or gate changes. Native imports occur only after
closed scopes pass and an exclusive triangle-comparison slot is requested.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
from scripts.research.compare_resident_finishes import semantic_agreement
from scripts.research.validate_checkpoint_finish_proof import (
    QUALITY_KIND, FINISH_ARTIFACTS, validate_checkpoint_manifest, closed_sidecar)
from scripts.research.validate_finish_resident_proof import read_json
from scripts.research.archive.validate_uniform_grid_proof import file_hash, require


def compare_gates(native, audit):
    deltas = {"numeric_max_abs_delta":0.,"information_max_abs_delta":0.,"transform_max_abs_delta":0.}
    same_order = [(r["gate_index"],r["function"]) for r in native] == [(r["gate_index"],r["function"]) for r in audit]
    passed = bool(native) and same_order and all(semantic_agreement(a["result"],b["result"],
        a["function"]+"/result",deltas) for a,b in zip(native,audit))
    return passed,{"native_calls":len(native),"audit_calls":len(audit),"same_invocation_order":same_order,**deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_sidecar",type=Path)
    parser.add_argument("audit_sidecar",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-allocated",action="store_true")
    args = parser.parse_args()
    if not args.run_allocated or args.output.exists():
        parser.error("Require an exclusive allocated comparison slot and fresh output")
    paths = [path.resolve(strict=True) for path in (args.native_sidecar,args.audit_sidecar)]
    require(paths[0] != paths[1] and args.output.resolve() not in paths,"Quality cannot overwrite or compare one reference with itself")
    tracked = {path:file_hash(path) for path in paths}
    result = {"kind":QUALITY_KIND,"status":"failed","audit_report_sha256":tracked[paths[1]],
        "native_sidecar":{"path":str(paths[0]),"sha256":tracked[paths[0]]},
        "audit":{"path":str(paths[1]),"sha256":tracked[paths[1]]},
        "comparison_artifact_sha256":file_hash(Path(__file__)),
        "note":"Same privately captured raw Live state; native mesh is a reference, not ground truth. Fixed input coordinates; no fitting."}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save()
    try:
        first = read_json(paths[1])
        artifacts = first["artifacts_sha256"]
        require(set(FINISH_ARTIFACTS).issubset(artifacts),"Incomplete current quality producer pins")
        authority = validate_checkpoint_manifest(first["checkpoint"]["path"],first["runtime_binding"],
            artifacts,first["checkpoint_scope_base"])
        native,_,native_gates = closed_sidecar(paths[0],"native",authority,artifacts,first["component_proof"],tracked)
        audit,_,audit_gates = closed_sidecar(paths[1],"audit",authority,artifacts,first["component_proof"],tracked)
        require(native["scope_binding"] == audit["scope_binding"],"Exact captured Live/settings/seed/RNG scope changed")
        passed,deltas = compare_gates(native_gates,audit_gates)
        result.update(checkpoint=authority.record,candidate=dict(audit["profile"]),native=dict(native["profile"]),
            candidate_trace=dict(audit["trace"]),native_trace=dict(native["trace"]),
            scope_binding_sha256=audit["scope_binding_sha256"],live_pose_signatures_equal=True,
            live_decision_signatures_equal=True,same_ordered_gate_decisions=passed,
            ordered_gate_evidence_passed=passed,gate_comparison=deltas)
        require(passed,"Original ordered graph/proposal/witness/pose/information outcomes changed")
        import numpy as np
        import open3d as o3d
        from scripts.profile_session import compare_quality
        from shared.surface_metrics import surface_metrics
        profile,reference = (read_json(side["profile"]["path"]) for side in (audit,native))
        comparison = compare_quality(profile,Path(native["profile"]["path"]),np)
        comparison.update(candidate_report_sha256=audit["profile"]["sha256"],baseline_report_sha256=native["profile"]["sha256"])

        def mesh(record):
            with np.load(record["geometry"]["artifact"],allow_pickle=False) as data:
                value = o3d.geometry.TriangleMesh()
                value.vertices = o3d.utility.Vector3dVector(data["points"])
                value.triangles = o3d.utility.Vector3iVector(data["faces"])
            return value

        surface = surface_metrics(mesh(profile),mesh(reference),threshold_m=.005,samples=30000)
        comparison["triangle_surface_metrics_vs_cpu"] = surface
        result["comparison"] = comparison
        require(comparison["same_accepted_indices"] and comparison["same_mesh_success"]
                and surface["surface_p95_m"] <= .0005 and surface["precision"] >= .999 and surface["completeness"] >= .999,
                "Fixed whole-surface quality failed")
        require(file_hash(Path(__file__)) == result["comparison_artifact_sha256"] == artifacts[str(Path(__file__).relative_to(ROOT)).replace("\\","/")]
                and all(file_hash(path) == expected for path,expected in tracked.items()),"Closed comparator/input bytes changed")
        result["inputs_sha256"] = {str(path):value for path,value in tracked.items()}
        result["status"] = "passed"
        save()
    except BaseException as error:
        result["status"] = "failed"
        result["failure"] = {"type":type(error).__name__,"message":str(error)}
        try:
            save()
        except BaseException as write_error:
            error.add_note(f"Failure report write also failed: {write_error}")
            raise error from write_error
        raise


if __name__ == "__main__":
    main()

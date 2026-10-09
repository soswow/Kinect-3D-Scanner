"""Close native/audited full-Finish quality separately from measured artifacts.

Ordered discrete graph decisions/witnesses and exact source-ID correspondence
mappings must agree. Small original transforms use absolute1e-8, information
matrices absolute1e-5 and other original numeric evidence rtol/atol1e-8, matching
the earlier component gate comparison. Raw array bytes remain diagnostic.
Triangle surfaces are compared in the original metric frame without alignment.
Imports are lazy: --help and importing semantic guards need no native runtime.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, compare_quality
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

KIND = "offline-full-finish-device-flat-quality-v1"


def semantic_agreement(left, right, path="", deltas=None):
    deltas = deltas if deltas is not None else {"numeric_max_abs_delta": 0., "information_max_abs_delta": 0., "transform_max_abs_delta": 0.}
    if type(left) is not type(right):
        # JSON floats/integers can only interoperate for actual numeric metrics.
        if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False
    if isinstance(left, dict):
        if set(left) != set(right):
            return False
        keys = set(left)
        if path == "bundle_adjustment.propose_bundle_poses/result":
            # Output-only observer retains this diagnostic in the raw trace.
            # Budget limit, exhaustion/reason and solver result remain gates.
            keys.discard("elapsed_ms")
        if "correspondence_mapping" in left:
            keys.discard("correspondence_set")  # original raw output order retained elsewhere
        if {"dtype", "shape", "sha256"} <= keys and "values" in left:
            keys.discard("sha256")  # bounded numeric arrays use original gate tolerances
        return all(semantic_agreement(left[key], right[key], path+"/"+key, deltas) for key in sorted(keys))
    if isinstance(left, list):
        return len(left) == len(right) and all(semantic_agreement(a, b, path, deltas) for a, b in zip(left, right))
    if isinstance(left, bool) or left is None or isinstance(left, str):
        return left == right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        delta = abs(left-right)
        if "information" in path:
            deltas["information_max_abs_delta"] = max(deltas["information_max_abs_delta"], delta)
            return delta <= 1e-5
        if "transform" in path or "/pose" in path or "propose_poses" in path:
            deltas["transform_max_abs_delta"] = max(deltas["transform_max_abs_delta"], delta)
            return delta <= 1e-8
        deltas["numeric_max_abs_delta"] = max(deltas["numeric_max_abs_delta"], delta)
        return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8)
    return False


def gate_rows(sidecar):
    rows = []
    for line in Path(sidecar["trace"]["path"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "gate":
            if not row.get("complete"):
                raise ValueError("Original gate did not complete")
            rows.append({key: row[key] for key in ("gate_index", "function", "result")})
    rows.sort(key=lambda row: row["gate_index"])
    if [row["gate_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Original ordered gate evidence is incomplete or duplicated")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_sidecar", type=Path)
    parser.add_argument("audit_sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true", help="Triangle comparison needs an exclusive CPU hardware slot")
    args = parser.parse_args()
    if not args.run_allocated or args.output.exists():
        parser.error("Require allocated hardware and fresh output; preserve measured artifacts")
    paths = (args.native_sidecar.resolve(strict=True), args.audit_sidecar.resolve(strict=True))
    originals = {str(path): file_hash(path) for path in paths}
    native, audit = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = {"kind": KIND, "status": "failed", "audit_report_sha256": originals[str(paths[1])],
              "native_sidecar": {"path": str(paths[0]), "sha256": originals[str(paths[0])]},
              "audit": {"path": str(paths[1]), "sha256": originals[str(paths[1])]},
              "comparison_artifact_sha256": file_hash(Path(__file__)),
              "note": "Matched native reconstruction is a reference, not ground truth; no alignment/scale fitting"}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    try:
        for side, mode in ((native, "native"), (audit, "audit")):
            if (side.get("status") != "complete" or side.get("mode") != mode
                    or not side["registration"]["complete"] or not side["registration"]["restored"]
                    or side["registration"]["failure"] is not None or side["registration"]["cleanup_failures"]
                    or not side.get("runner_hooks_restored") or side.get("failure") is not None):
                raise ValueError("Native/audit closure or restoration failed")
            for prefix in ("source_sha256", "archive_sha256", "artifacts_sha256", "runtime_binding", "component_proof"):
                if side[prefix] != side[prefix+"_after"]:
                    raise ValueError("Measured scope changed before closure")
            records = [side["profile"], side["trace"]]
            for item in records:
                path = Path(item["path"])
                originals[str(path)] = file_hash(path)
                if originals[str(path)] != item["sha256"]:
                    raise ValueError("Measured profile/trace bytes changed")
            path = Path(side["profile"]["geometry_path"])
            originals[str(path)] = file_hash(path)
            if originals[str(path)] != side["profile"]["geometry_sha256"]:
                raise ValueError("Measured geometry changed")
        # Live numeric hashes may differ between independent raw runs. Never
        # reuse that difference for timing: timing binds the audited exact hash.
        a, b = native["scope_binding"], audit["scope_binding"]
        for name in ("archive", "selected_indices", "seed", "settings", "pipeline_options", "thread_policy", "runtime_binding"):
            if a[name] != b[name]:
                raise ValueError(f"Matched Finish scope differs: {name}")
        for name in ("accepted_indices", "decisions_sha256", "unprocessed_count", "pose_source"):
            if a["live"][name] != b["live"][name]:
                raise ValueError("Fresh Live accepted decisions differ")
        if native["artifacts_sha256"] != audit["artifacts_sha256"]:
            raise ValueError("Native and resident helper versions differ")
        result.update(candidate=dict(audit["profile"]), native=dict(native["profile"]),
            native_trace=dict(native["trace"]), candidate_trace=dict(audit["trace"]),
            scope_binding_sha256=audit["scope_binding_sha256"],
            live_pose_signatures_equal=a["live"]["poses_sha256"] == b["live"]["poses_sha256"],
            live_decision_signatures_equal=a["live"]["decisions_sha256"] == b["live"]["decisions_sha256"])
        rows_a, rows_b = gate_rows(native), gate_rows(audit)
        deltas = {"numeric_max_abs_delta": 0., "information_max_abs_delta": 0., "transform_max_abs_delta": 0.}
        same_order = [(r["gate_index"], r["function"]) for r in rows_a] == [(r["gate_index"], r["function"]) for r in rows_b]
        same = same_order and all(semantic_agreement(x, y, x["function"], deltas) for x, y in zip(rows_a, rows_b))
        result.update(same_ordered_gate_decisions=same, ordered_gate_evidence_passed=same,
                      gate_comparison={"native_calls": len(rows_a), "audit_calls": len(rows_b),
                                       "same_invocation_order": same_order, **deltas})
        if not same or not rows_a:
            raise ValueError("Original graph gate, proposal, witness, pose or information evidence changed")
        import numpy as np
        import open3d as o3d
        from shared.surface_metrics import surface_metrics
        profile = json.loads(Path(audit["profile"]["path"]).read_text(encoding="utf-8"))
        reference = json.loads(Path(native["profile"]["path"]).read_text(encoding="utf-8"))
        if not profile["mesh_built"] or not reference["mesh_built"]:
            raise ValueError("Both matched controls must complete Final mesh")
        comparison = compare_quality(profile, Path(native["profile"]["path"]), np)
        comparison.update(candidate_report_sha256=audit["profile"]["sha256"],
                          baseline_report_sha256=native["profile"]["sha256"])

        def mesh(item):
            with np.load(item["geometry"]["artifact"], allow_pickle=False) as data:
                value = o3d.geometry.TriangleMesh()
                value.vertices = o3d.utility.Vector3dVector(data["points"])
                value.triangles = o3d.utility.Vector3iVector(data["faces"])
            return value

        comparison["triangle_surface_metrics_vs_cpu"] = surface_metrics(mesh(profile), mesh(reference), threshold_m=.005, samples=30000)
        result["comparison"] = comparison
        metrics = comparison["triangle_surface_metrics_vs_cpu"]
        # Fixed conservative whole-surface gate; per-call/gate pose tolerances
        # above are much tighter and remain independent of these mesh limits.
        if (not comparison["same_accepted_indices"] or not comparison["same_mesh_success"]
                or metrics["surface_p95_m"] > .0005 or metrics["precision"] < .999
                or metrics["completeness"] < .999):
            raise ValueError("Matched native triangle surface quality gate failed")
        if any(file_hash(Path(path)) != digest for path, digest in originals.items()):
            raise ValueError("Measured input artifacts changed during quality comparison")
        result["inputs_sha256"] = originals
        result["status"] = "passed"
        save()
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


if __name__ == "__main__":
    main()

"""Stdlib diagnostic of closed Finish traces; creates no quality authority.

Visits every gate value rather than stopping at the first failed comparison.
The frozen comparator's thresholds and exact diagnostic exclusions are retained
only to describe where its strict proof failed. No mesh/native/GPU imports.
"""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(matrix):
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


def eigenvalues(matrix):
    """Small symmetric Jacobi diagnostic, not a registration decision."""
    a = [list(row) for row in matrix]
    n = len(a)
    for _ in range(400):
        p, q = max(((p, q) for p in range(n) for q in range(p + 1, n)),
                   key=lambda pair: abs(a[pair[0]][pair[1]]))
        if abs(a[p][q]) <= max(abs(a[i][i]) for i in range(n)) * 1e-15:
            break
        angle = .5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq
        a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.
    return sorted(a[i][i] for i in range(n))


def walk(left, right, path, deltas, mismatches):
    if type(left) is not type(right) and not (type(left) in (int, float) and type(right) in (int, float)):
        mismatches.append({"path": path, "kind": "type"})
        return
    if isinstance(left, dict):
        if set(left) != set(right):
            mismatches.append({"path": path, "kind": "keys"})
            return
        keys = set(left)
        if path in ("bundle_adjustment.propose_bundle_poses/result",
                    "bundle_adjustment.propose_bundle_poses/result[1]"):
            keys.discard("elapsed_ms")
        if "correspondence_mapping" in left:
            keys.discard("correspondence_set")
        if {"dtype", "shape", "sha256", "values"} <= keys:
            keys.discard("sha256")
        for key in sorted(keys):
            walk(left[key], right[key], path + "/" + key, deltas, mismatches)
    elif isinstance(left, list):
        if len(left) != len(right):
            mismatches.append({"path": path, "kind": "length"})
        for i, (a, b) in enumerate(zip(left, right)):
            walk(a, b, path + f"[{i}]", deltas, mismatches)
    elif type(left) in (int, float) and type(right) in (int, float) and not (type(left) is int and type(right) is int):
        delta = abs(left - right)
        category = "information" if "information" in path else "transform" if any(x in path for x in ("transform", "/pose", "propose_poses")) else "numeric"
        deltas[category] = max(deltas[category], delta)
        passed = delta <= (1e-5 if category == "information" else 1e-8) if category != "numeric" else math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8)
        if not math.isfinite(left) or not math.isfinite(right) or not passed:
            mismatches.append({"path": path, "kind": category, "native": left, "audit": right, "absolute_delta": delta})
    elif left != right:
        mismatches.append({"path": path, "kind": "discrete", "native": left, "audit": right})


def contains_hash(value, fingerprint):
    if isinstance(value, dict):
        return value.get("sha256") == fingerprint or any(contains_hash(v, fingerprint) for v in value.values())
    return isinstance(value, list) and any(contains_hash(v, fingerprint) for v in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-profile", type=Path)
    parser.add_argument("--audit-profile", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require fresh output; preserve measured and derived reports")
    if bool(args.native_profile) != bool(args.audit_profile):
        parser.error("Both matched profiles are required")
    files = {str(p.resolve()): sha(p) for p in (args.native, args.audit, Path(__file__))}
    traces = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] for path in (args.native, args.audit)]
    gates = [sorted((row for row in trace if row["event"] == "gate"), key=lambda row: row["gate_index"]) for trace in traces]
    if any([r["gate_index"] for r in rows] != list(range(len(rows))) or not all(r.get("complete") for r in rows) for rows in gates):
        raise ValueError("Incomplete ordered gate evidence")
    same_order = [(r["gate_index"], r["function"]) for r in gates[0]] == [(r["gate_index"], r["function"]) for r in gates[1]]
    if not same_order:
        raise ValueError("Gate sequences differ; this diagnostic requires aligned invocations")
    deltas = {key: 0. for key in ("numeric", "information", "transform")}
    differences, matrices = [], []
    for a, b in zip(*gates):
        diff = []
        walk(a["result"], b["result"], a["function"] + "/result", deltas, diff)
        if diff:
            differences.append({"gate_index": a["gate_index"], "function": a["function"], "mismatches": diff})
        if a["function"] == "original_information_matrix":
            x, y = a["result"]["values"], b["result"]["values"]
            d = [[v - u for u, v in zip(xrow, yrow)] for xrow, yrow in zip(x, y)]
            if a["result"]["sha256"] != b["result"]["sha256"]:
                maximum = max((abs(d[i][j]), i, j) for i in range(6) for j in range(6))
                eig = eigenvalues(x)
                owners = []
                for row in gates[0]:
                    if row["gate_index"] < a["gate_index"] and contains_hash(row["result"], a["result"]["sha256"]):
                        result = row["result"]
                        owners.append({"gate_index": row["gate_index"], "function": row["function"],
                            "source": result.get("source") if isinstance(result, dict) else None,
                            "target": result.get("target") if isinstance(result, dict) else None,
                            "support": result.get("support") if isinstance(result, dict) else None,
                            "validation_scope": result.get("validation_scope") if isinstance(result, dict) else None})
                matrices.append({"gate_index": a["gate_index"], "maximum_absolute_delta": maximum[0],
                    "maximum_position": list(maximum[1:]), "native_frobenius": norm(x),
                    "difference_frobenius": norm(d), "relative_frobenius": norm(d) / norm(x),
                    "native_eigenvalues_jacobi_diagnostic": eig,
                    "native_spectral_condition_diagnostic": eig[-1] / eig[0] if eig[0] > 0 else None,
                    "native_translation_diagonal": [x[i][i] for i in (3, 4, 5)],
                    "audit_translation_diagonal": [y[i][i] for i in (3, 4, 5)],
                    "copied_into_parent_results": owners})
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError("Closed trace or diagnostic source changed")
    result = {"kind": "closed-finish-complete-trace-diagnostic-v1", "status": "complete",
        "quality_authority": False, "inputs_sha256": files, "same_invocation_order": same_order,
        "gate_count": len(gates[0]), "strict_threshold_mismatched_gate_count": len(differences),
        "full_trace_maximum_absolute_deltas": deltas,
        "mismatch_categories": dict(Counter(item["kind"] for row in differences for item in row["mismatches"])),
        "information_matrices_with_different_raw_bits": matrices,
        "gate_differences": differences,
        "note": "Complete diagnostic includes all later failures; original failed strict proof and thresholds remain unchanged. Relative norms/eigenvalues do not authorize altered information matrices. Original information correspondence IDs were not captured."}
    if args.native_profile:
        profiles = []
        for path in (args.native_profile, args.audit_profile):
            files[str(path.resolve())] = sha(path)
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        a, b = profiles
        def graph(profile):
            report = profile["fragment_reconnection"]
            return {"accepted_indices": profile["accepted_indices"], "mesh_built": profile["mesh_built"],
                "fragments": [{key: f[key] for key in ("id", "connected", "frame_indices", "context_frame_indices")} for f in report["fragments"]],
                "bridges": [{key: edge[key] for key in ("source", "target", "support", "validation_scope", "connected_to_scan")} for edge in report["verified_bridges"]],
                "report": {key: report[key] for key in ("applied", "reason", "ambiguous_pairs", "budget_limited", "invalid_indices", "unassigned_indices", "recovered_frames", "corrected_frames", "excluded_frames", "connected_fragments", "unconnected_fragments", "recent_reference_links", "loop_closures")}}
        native_graph, audit_graph = graph(a), graph(b)
        if [p["index"] for p in a["poses"]] != [p["index"] for p in b["poses"]]:
            raise ValueError("Final pose view sequence differs")
        pose_rows = []
        for x, y in zip(a["poses"], b["poses"]):
            u, v = x["camera_to_world"], y["camera_to_world"]
            rotation_frobenius = norm([[v[i][j] - u[i][j] for j in range(3)] for i in range(3)])
            pose_rows.append({"index": x["index"],
                "translation_delta_m": math.sqrt(math.fsum((v[i][3] - u[i][3]) ** 2 for i in range(3))),
                "rotation_frobenius_delta": rotation_frobenius,
                "rotation_delta_deg_orthogonal_formula": math.degrees(2 * math.asin(min(1., rotation_frobenius / (2 * math.sqrt(2.)))))})
        result["final_reconstruction_diagnostic"] = {"same_discrete_graph_and_witness_evidence": native_graph == audit_graph,
            "native_graph": native_graph, "audit_graph": audit_graph, "pose_comparison": pose_rows,
            "maximum_translation_delta_m": max(r["translation_delta_m"] for r in pose_rows),
            "maximum_rotation_delta_deg_orthogonal_formula": max(r["rotation_delta_deg_orthogonal_formula"] for r in pose_rows),
            "note": "No triangle surface comparison was run by this stdlib diagnostic."}
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError("Closed inputs changed during diagnostic")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("gate_count", "strict_threshold_mismatched_gate_count", "full_trace_maximum_absolute_deltas", "mismatch_categories")}))


if __name__ == "__main__":
    main()

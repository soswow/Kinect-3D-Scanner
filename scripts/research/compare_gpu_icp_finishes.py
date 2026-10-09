"""Independent whole-Finish GPU quality from one current raw-Live checkpoint.

Pure graph/gate/pose checks are reusable by the new timing validator. Surface
metrics are physically measured in an allocated comparison process, in the
original coordinate frame, then bound to both complete geometry artifacts.
No old pair, resident-Finish or near-seed token authorizes this new method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.compare_resident_finishes import semantic_agreement

KIND = "gpu-icp-whole-finish-quality-v1"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Finite numeric evidence required")
    return float(value)


def matrix(value):
    record = value if isinstance(value, dict) else None
    values = record["values"] if record is not None else value
    require(isinstance(values, list) and len(values) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in values), "Complete 4x4 pose required")
    result = [[number(item) for item in row] for row in values]
    require(result[3] == [0., 0., 0., 1.], "Original homogeneous pose convention required")
    if record is not None:
        require(record.get("dtype") == "<f8" and record.get("shape") == [4, 4]
            and record.get("sha256") == hashlib.sha256(struct.pack("<16d", *(x for row in result for x in row))).hexdigest(),
            "Unrounded pose values must match their original byte evidence")
    return result


def closed(report, mode):
    registration = report.get("registration", {})
    expected_kind = "gpu-icp-whole-finish-"+mode+"-v1"
    require(mode in ("native", "audit") and report.get("kind") == expected_kind
        and report.get("status") == "passed" and report.get("mode") == mode
        and report.get("failure") is None and report.get("cleanup_passed") is True
        and report.get("cleanup_failures") == [] and report.get("binding") == report.get("binding_after")
        and registration.get("complete") is True and registration.get("restored") is True
        and registration.get("failure") is None and registration.get("cleanup_failures") == []
        and registration.get("successful_builds") == 1,
        "Only fully closed original/native or new GPU Finish reports may be compared")
    require(report["binding"].get("source_sha256") == CURRENT, "Current measured production core required")
    profile = report["profile"]
    require(profile.get("finish_requested") is True and profile.get("pose_seeds_used") is False
        and profile.get("mesh_built") is True and profile.get("input_changed_during_profile") is False
        and profile.get("source_changed_during_profile") is False, "Successful unchanged-source unseeded Finish required")
    return profile


def ordered_gates(report):
    rows = report["registration"]["gates"]
    require(rows and [r.get("gate_index") for r in rows] == list(range(len(rows)))
        and all(r.get("complete") is True and isinstance(r.get("function"), str) for r in rows),
        "Complete original ordered gate evidence required")
    return rows


def pose_values(report, profile):
    inventory = report["final_pose_inventory"]
    require(inventory.get("captured_after_successful_build") is True
        and inventory.get("pose_convention") == "camera_to_world" and inventory.get("length_unit") == "metres",
        "Original Final pose inventory must be captured after successful build")
    rows = inventory["rows"]
    require(inventory.get("pose_inventory_sha256") == hashlib.sha256(canonical(rows).encode()).hexdigest(),
        "Complete actual Final pose inventory must match its recorded digest")
    require(0 < len(rows) <= 4096 and [row["index"] for row in rows] == profile["accepted_indices"]
        and len(set(profile["accepted_indices"])) == len(rows)
        and all(type(row["index"]) is int and row["index"] >= 0 for row in rows), "Complete unique retained view membership required")
    poses = {row["index"]: matrix(row["camera_to_world"]) for row in rows}
    require([row["index"] for row in profile["poses"]] == profile["accepted_indices"]
        and all(matrix(row["camera_to_world"]) == poses[row["index"]] for row in profile["poses"]),
        "Profile poses must equal every actual unrounded Final inventory pose")
    return poses


def pose_deltas(left, right):
    require(left.keys() == right.keys(), "Final pose memberships changed")
    translations, rotations = [], []
    for index in left:
        a, b = left[index], right[index]
        for pose in (a, b):
            require(all(abs(sum(pose[k][i]*pose[k][j] for k in range(3)) - (i == j)) <= 1e-8
                for i in range(3) for j in range(3)), "Rigid unrounded Final rotation required")
            determinant = sum(pose[0][i]*(pose[1][(i+1)%3]*pose[2][(i+2)%3]
                -pose[1][(i+2)%3]*pose[2][(i+1)%3]) for i in range(3))
            require(abs(determinant-1.) <= 1e-8, "Proper Final rotation required; reflections are invalid")
        translations.append(math.sqrt(sum((a[i][3]-b[i][3])**2 for i in range(3))))
        trace = sum(a[k][i]*b[k][i] for k in range(3) for i in range(3))
        rotations.append(math.degrees(math.acos(max(-1., min(1., (trace-1.)/2.)))))
    return max(translations), max(rotations)


def shadow_coverage(report):
    rows, hits, misses, query_rows = report["calls"], 0, 0, 0
    require(len(rows) <= 4096 and [r.get("call_index") for r in rows] == list(range(len(rows))),
        "Actual global GPU call order must be complete and bounded")
    for row in rows:
        shadow = row["native_shadow"]
        require(row.get("complete") is True and row.get("input_bytes_unchanged") is True
            and shadow.get("passed") is True and shadow.get("correspondence_ids_equal") is True,
            "Each actual GPU result needs a successful same-input original CPU shadow")
        original, candidate = shadow["native"], shadow["candidate"]
        left, right = matrix(original["transformation"]), matrix(candidate["transformation"])
        deltas = {"transform_max_abs_delta": max(abs(a-b) for ra, rb in zip(left, right) for a, b in zip(ra, rb)),
            "fitness_abs_delta": abs(number(original["fitness"])-number(candidate["fitness"])),
            "rmse_abs_delta": abs(number(original["inlier_rmse"])-number(candidate["inlier_rmse"]))}
        require(all(shadow.get(key) == value and value <= 1e-8 for key, value in deltas.items())
            and original["correspondence_mapping"] == candidate["correspondence_mapping"],
            "Actual original CPU result differences and canonical IDs must independently agree")
        loop = row["loop_report"]
        require(loop.get("closed") is True and loop.get("failure") is None, "Actual lane must close successfully")
        stats = loop["statistics"]
        require(type(stats["queries"]) is int and stats["queries"] == len(row["query_trace"]) > 0
            and all(type(stats[k]) is int and stats[k] >= 0 for k in ("query_rows", "audited_hits", "audited_misses", "cpu_ambiguity_rows")),
            "Actual nearest-query CPU audit coverage required")
        hits += stats["audited_hits"]; misses += stats["audited_misses"]; query_rows += stats["query_rows"]
    require(not rows or (hits > 0 and misses > 0 and query_rows > 0), "Nonempty GPU scope needs positive actual hit and miss audit coverage")
    return len(rows), query_rows


def bounded_values(value):
    """Verify nested raw gate descriptors before numeric small-array comparison.

    Large arrays and canonical correspondence descriptors keep exact hashes.
    Raw reports retain both formats; only this independent quality comparison
    uses the declared existing small numeric gate tolerances.
    """
    if isinstance(value, list):
        return [bounded_values(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"array", "values"}:
        descriptor = value["array"]
        shape, dtype = descriptor["shape"], descriptor["dtype"]
        require(isinstance(shape, list) and all(type(i) is int and 0 <= i <= 64 for i in shape)
            and math.prod(shape) <= 64, "Only bounded complete small gate values may use numeric comparison")
        formats = {"<f8":"d", "<f4":"f", "<i8":"q", "<i4":"i", "<i2":"h", "|i1":"b",
            "<u8":"Q", "<u4":"I", "<u2":"H", "|u1":"B", "|b1":"?"}
        require(dtype in formats, "Known original gate array dtype required")
        def flatten(items, dimensions):
            if not dimensions:
                if dtype.startswith("<f"):
                    number(items)
                else:
                    require(type(items) is (bool if dtype == "|b1" else int), "Discrete gate array values required")
                return [items]
            require(isinstance(items, list) and len(items) == dimensions[0], "Complete gate array shape required")
            return [item for child in items for item in flatten(child, dimensions[1:])]
        flat = flatten(value["values"], shape)
        raw = struct.pack("<"+str(len(flat))+formats[dtype], *flat)
        require(descriptor["nbytes"] == len(raw) and descriptor["sha256"] == hashlib.sha256(raw).hexdigest(),
            "Actual small gate values must match their unchanged raw descriptor")
        return {"dtype":dtype, "shape":shape, "values":value["values"]}
    return {key:bounded_values(item) for key,item in value.items()}


def retained_graphs(report):
    """Exact ordered endpoints/components using the original .25 prune rule."""
    result = []
    for row in report["registration"]["graphs"]:
        context, graph = row["context"], row["after"]
        key = {"refinement.propose_poses": "node_raw_view_indices",
            "fragments.propose_fragment_poses": "node_fragment_indices"}.get(context.get("owner"))
        ids = context.get(key) if key else None
        require(row.get("complete") is True and isinstance(ids, list) and 0 < len(ids) <= 64
            and len(set(ids)) == len(ids) and len(graph["nodes"]) == len(ids)
            and all(type(i) is int and i >= 0 for i in ids), "Actual optimizer node identities required")
        require(row["options"]["edge_prune_threshold"] == .25, "Original graph prune rule required")
        groups, retained = list(range(len(ids))), []
        def root(i):
            while groups[i] != i:
                i = groups[i]
            return i
        for edge in graph["edges"]:
            a, b = edge["source_node"], edge["target_node"]
            require(type(a) is int and type(b) is int and a != b and 0 <= a < len(ids)
                and 0 <= b < len(ids) and type(edge["uncertain"]) is bool
                and 0 <= number(edge["confidence"]) <= 1, "Actual retained edge evidence required")
            if not edge["uncertain"] or edge["confidence"] >= .25:
                retained.append([ids[a], ids[b], edge["uncertain"]])
                groups[root(b)] = root(a)
        components = sorted(sorted(ids[i] for i in range(len(ids)) if root(i) == representative)
            for representative in {root(i) for i in range(len(ids))})
        result.append({"context": context, "retained_edges": retained, "retained_components": components})
    return result


def quality_record(native_report, audit_report):
    """Recompute discrete, original gate and unrounded pose quality with stdlib.

    Surface dispatch is deliberately separate. This function creates no token
    and grants no general domain or production permission.
    """
    native, audit = closed(native_report, "native"), closed(audit_report, "audit")
    require(native_report["binding"] == audit_report["binding"], "Independent controls must use the same declared method/resources")
    same_scope = native_report["scope_binding"] == audit_report["scope_binding"]
    require(same_scope, "Controls require exactly the same current raw-Live checkpoint")
    keys = ("input_sha256", "selected_indices", "seed", "settings", "finish_requested", "pose_seeds_used")
    require(all(native[key] == audit[key] for key in keys), "Raw/settings/phases differ between controls")
    a, b = ordered_gates(native_report), ordered_gates(audit_report)
    deltas = {"numeric_max_abs_delta": 0., "information_max_abs_delta": 0., "transform_max_abs_delta": 0.}
    ordered = [(r["gate_index"], r["function"]) for r in a] == [(r["gate_index"], r["function"]) for r in b]
    gates = ordered and all(semantic_agreement(bounded_values(x["result"]), bounded_values(y["result"]),
        x["function"]+"/result", deltas)
        and ("consumed_problem" in x) == ("consumed_problem" in y)
        and ("consumed_problem" not in x or semantic_agreement(bounded_values(x["consumed_problem"]),
            bounded_values(y["consumed_problem"]), x["function"]+"/consumed_problem", deltas))
        for x, y in zip(a, b))
    graphs = (retained_graphs(native_report) == retained_graphs(audit_report)
        and semantic_agreement(native_report["registration"]["graphs"], audit_report["registration"]["graphs"], "graph_calls", deltas))
    # ScanEngine appends this one duration after the original proposal output.
    # Retain it in raw reports; budgets, limits and all outcomes remain gates.
    summaries = [dict(p["fragment_reconnection"]) for p in (native, audit)]
    for summary in summaries:
        summary.pop("elapsed_ms", None)
    graph_summary = semantic_agreement(*summaries, "fragment_reconnection", deltas)
    translation, rotation = pose_deltas(pose_values(native_report, native), pose_values(audit_report, audit))
    calls, query_rows = shadow_coverage(audit_report)
    checks = {"same_checkpoint": same_scope, "same_accepted_indices": native["accepted_indices"] == audit["accepted_indices"],
        "same_retained_graph": graphs and graph_summary, "same_ordered_gate_memberships": gates,
        "same_mesh_success": native["mesh_built"] is True and audit["mesh_built"] is True,
        "pose_bounds": translation <= .0005 and rotation <= .1,
        "actual_gpu_calls_shadowed": True, "actual_gpu_query_rows_shadowed": True}
    return {"checks": checks, "metrics": {"pose_translation_max_m": translation, "pose_rotation_max_deg": rotation,
        "actual_gpu_calls": calls, "actual_gpu_query_rows": query_rows, "original_gate_calls": len(a), **deltas}}


def check_sources(binding):
    from scripts.profile_session import source_hash
    require(source_hash() == binding["source_sha256"] == CURRENT, "Current production bytes changed")
    sources = binding["artifacts_sha256"]
    own = str(Path(__file__).relative_to(ROOT)).replace("\\", "/")
    require(own in sources and sources[own] == sha(__file__), "Actual comparator source must belong to measured family")
    for name, digest in sources.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == digest, "Measured helper source changed: "+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("audit_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    require(args.run_allocated and output.is_relative_to(ROOT/"benchmark-output") and not output.exists(),
        "Allocated idle-server comparison and fresh private output required")
    with socket.socket() as sock:
        sock.settimeout(.25)
        require(sock.connect_ex(("127.0.0.1", 8000)) != 0, "Field server must be stopped for numerical quality comparison")
    paths = [p.resolve(strict=True) for p in (args.native_report, args.audit_report)]
    require(paths[0] != paths[1] and output not in paths, "Independent reports and fresh output required")
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    digests = [sha(p) for p in paths]
    result = {"kind": KIND, "status": "failed", "failure": None,
        "native_report": {"path": str(paths[0]), "sha256": digests[0]},
        "audit_report": {"path": str(paths[1]), "sha256": digests[1]},
        "binding": reports[1]["binding"], "scope_binding": reports[1]["scope_binding"],
        "comparator_source_sha256": sha(__file__), "production_authority": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        check_sources(result["binding"])
        result.update(quality_record(*reports))
        require(all(result["checks"].values()), "Whole-Finish original gate/graph/pose quality changed")
        records = []
        for report in reports:
            geometry = report["profile"]["geometry"]
            path = Path(geometry["artifact"]).resolve(strict=True)
            require(path.is_relative_to(ROOT/"benchmark-output") and sha(path) == geometry["sha256"], "Measured geometry artifact changed")
            records.append({"path": str(path), "sha256": geometry["sha256"]})
        import numpy as np
        import open3d as o3d
        from shared.surface_metrics import surface_metrics
        def mesh(record):
            with np.load(record["path"], allow_pickle=False) as data:
                value = o3d.geometry.TriangleMesh()
                value.vertices = o3d.utility.Vector3dVector(data["points"])
                value.triangles = o3d.utility.Vector3iVector(data["faces"])
            return value
        surface = surface_metrics(mesh(records[1]), mesh(records[0]), threshold_m=.005, samples=30000)
        surface.update(samples=30000, candidate_geometry_sha256=records[1]["sha256"], reference_geometry_sha256=records[0]["sha256"])
        result["surface"] = surface
        result["checks"]["surface_bounds"] = surface["surface_p95_m"] <= .0005 and surface["precision"] >= .999 and surface["completeness"] >= .999
        result["metrics"].update({k: surface[k] for k in ("surface_p95_m", "precision", "completeness", "threshold_m", "samples")})
        for record in records:
            record["sha256_after"] = sha(record["path"])
            require(record["sha256_after"] == record["sha256"], "Geometry changed during allocated comparison")
        result["reference_geometry"], result["candidate_geometry"] = records
        check_sources(result["binding"])
        require(all(sha(path) == digest for path, digest in zip(paths, digests)) and all(result["checks"].values()),
            "Bound quality inputs changed or physical surface quality failed")
        result["status"] = "passed"
        result["limits"] = "Independent original/native reference, not ground truth. Fixed coordinates; no alignment. Recorded surface certificate; only this exact whole-Finish trajectory is qualified. No production or general-domain authority."
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Whole-Finish original gates, graph, poses and fixed-coordinate surfaces passed")


if __name__ == "__main__":
    main()

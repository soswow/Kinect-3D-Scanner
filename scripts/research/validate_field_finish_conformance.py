"""Separate actual-input field authority; old strict Finish proofs stay separate.

Every audited resident query and complete ICP result retains its original CPU
shadow. Independent native comparison certifies declared final graph/coverage,
0.5 mm / 0.1 degree poses and fixed-coordinate surfaces, not intermediate
native history or bit equivalence. Timing must reproduce its own audited GPU
inputs, result/gate/optimization events and complete suffix. Stdlib only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ast
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import struct
import textwrap

from scripts.research import validate_checkpoint_finish_proof as checkpoint
from scripts.research import validate_finish_resident_proof as original
from scripts.research.compare_resident_finishes import semantic_agreement
from scripts.research.validate_device_flat_grid_proof import (
    DeviceFlatGridProofAuthority, normalized_artifacts, digest,
    validate_configuration, validate_current_artifacts)
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofError, require, canonical_hash, file_hash)

KIND = "offline-field-finish-actual-input-conformance-v2"
QUALITY_KIND = "offline-field-finish-conformance-quality-v2"
PROPOSAL_POLICIES = ("original", "canonical-fresh-1um")
NEW_ARTIFACTS = ("scripts/research/profile_field_finish_conformance.py",
    "scripts/research/field_finish_conformance_scope.py",
    "scripts/research/validate_field_finish_conformance.py",
    "scripts/research/canonical_fpfh_proposals.py", "scripts/process_metrics.py",
    "scripts/research/compare_field_finish_conformance.py")
FINISH_ARTIFACTS = checkpoint.FINISH_ARTIFACTS + NEW_ARTIFACTS
COMPARATOR = "scripts/research/compare_field_finish_conformance.py"
POSE_LIMITS = {"translation_m": .0005, "rotation_deg": .1}
OUTPUT_CRITERIA = {"pose_limits":dict(POSE_LIMITS),"surface_p95_m":.0005,
    "surface_threshold_m":.005,"precision_min":.999,"completeness_min":.999,"samples_per_surface":30000}
MAX_TRACE_EVENTS = 300000
MAX_TOKEN_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class FieldFinishConformanceAuthority(DeviceFlatGridProofAuthority):
    field_scope_json: str
    expected_call_signatures: tuple
    expected_match_results_json: str
    expected_gate_records_json: str
    expected_graph_records_json: str
    expected_event_order_json: str
    expected_final_poses_json: str
    conformance_audit_report_sha256: str
    quality_proof_sha256: str
    conformance_artifact_sha256: tuple
    proposal_policy: str


# A caller-constructed dataclass is not authority. Only successful closed proof
# validation registers a token. Bounded decoded state also avoids reparsing a
# full trajectory on every call and makes duplicate/reordered events fail.
_ACTIVE = OrderedDict()


def strict_json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def adapted_sidecar_reader():
    """Reuse unchanged closure/shadow guards with only new kind/event domain.

    This creates a private function/global namespace. It never patches the old
    validator, modifies a raw report or creates an old Finish authority.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(checkpoint.closed_sidecar)))
    baseline = ast.dump(tree, include_attributes=False)
    seam = [node for node in ast.walk(tree) if isinstance(node, ast.Tuple)
            and len(node.elts) == 2 and all(isinstance(item, ast.Constant) for item in node.elts)
            and [item.value for item in node.elts] == ["match", "gate"]]
    # One domain tuple and one extraction tuple; replace domain only.
    domains = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
               and any(isinstance(operator, ast.In) for operator in node.ops)
               and any(item is candidate for item in node.comparators for candidate in seam)]
    require(len(domains) == 1, "Original sidecar event-domain seam changed")
    domain = domains[0].comparators[0]
    domain.elts.append(ast.Constant(value="graph_optimization"))
    ast.fix_missing_locations(tree)
    namespace = dict(checkpoint.__dict__, KIND=KIND)
    exec(compile(tree, str(Path(__file__)), "exec", dont_inherit=True), namespace)
    result = namespace["closed_sidecar"]
    domain.elts.pop()
    require(ast.dump(tree, include_attributes=False) == baseline,
            "Field closure changed original query/shadow/input/profile guards")
    return result


_CLOSED_SIDECAR = adapted_sidecar_reader()


def matrix_evidence(record, size):
    require(isinstance(record, dict) and set(record) == {"dtype", "shape", "sha256", "values"}
            and record["dtype"] == "<f8" and record["shape"] == [size, size]
            and digest(record["sha256"]), "Missing unrounded bounded FP64 matrix evidence")
    values = record["values"]
    require(isinstance(values, list) and len(values) == size and all(isinstance(row, list)
            and len(row) == size for row in values), "Matrix shape differs from bounded values")
    flat = [item for row in values for item in row]
    require(all(type(item) in (int, float) and math.isfinite(item) for item in flat), "Nonfinite matrix")
    require(hashlib.sha256(struct.pack("<"+"d"*len(flat), *flat)).hexdigest() == record["sha256"],
            "Matrix values do not bind original FP64 bytes")
    return values


def rigid_pose(record):
    value = matrix_evidence(record, 4)
    require(all(abs(value[3][i]-expected) <= 1e-10 for i, expected in enumerate((0,0,0,1))),
            "Malformed homogeneous pose")
    rotation = [row[:3] for row in value[:3]]
    require(all(abs(sum(rotation[k][i]*rotation[k][j] for k in range(3))-(i == j)) <= 1e-4
            for i in range(3) for j in range(3)), "Nonrigid rotation")
    determinant = (rotation[0][0]*(rotation[1][1]*rotation[2][2]-rotation[1][2]*rotation[2][1])
        - rotation[0][1]*(rotation[1][0]*rotation[2][2]-rotation[1][2]*rotation[2][0])
        + rotation[0][2]*(rotation[1][0]*rotation[2][1]-rotation[1][1]*rotation[2][0]))
    require(abs(determinant-1) <= 1e-4, "Reflection or singular pose")
    return value


def pose_inventory(record, profile):
    require(isinstance(record, dict) and set(record) == {"pose_convention","length_unit","captured_after_successful_build","rows","pose_inventory_sha256"}
            and record["pose_convention"] == "camera_to_world" and record["length_unit"] == "metres"
            and record["captured_after_successful_build"] is True
            and isinstance(record["rows"], list) and 0 < len(record["rows"]) <= 4096,
            "Final unrounded native engine pose inventory absent")
    poses = record["rows"]
    require(all(isinstance(row, dict) and set(row) == {"index", "camera_to_world"}
            and type(row["index"]) is int and row["index"] >= 0 for row in poses), "Malformed final pose identity")
    indices = [row["index"] for row in poses]
    require(len(set(indices)) == len(indices) and indices == profile.get("accepted_indices")
            and record["pose_inventory_sha256"] == canonical_hash(poses), "Final pose inventory/order differs from successful mesh views")
    return [(row["index"], rigid_pose(row["camera_to_world"])) for row in poses]


def graph_record(row):
    require(row.get("event") == "graph_optimization" and row.get("complete") is True
            and type(row.get("graph_index")) is int and row["graph_index"] >= 0
            and isinstance(row.get("gate_context"), list), "Incomplete original graph observation")
    context = row["context"]
    owner = context.get("owner")
    key = {"refinement.propose_poses":"node_raw_view_indices",
           "fragments.propose_fragment_poses":"node_fragment_indices"}.get(owner)
    require(key is not None and set(context) == {"owner", key}, "Unknown graph owner")
    ids = context[key]
    require(isinstance(ids, list) and 0 < len(ids) <= 64 and len(set(ids)) == len(ids)
            and all(type(index) is int and index >= 0 for index in ids), "Missing graph node identity")
    options = row["options"]
    require(set(options) == {"max_correspondence_distance","edge_prune_threshold","reference_node"}
            and options["max_correspondence_distance"] == .03 and options["edge_prune_threshold"] == .25
            and type(options["reference_node"]) is int and 0 <= options["reference_node"] < len(ids),
            "Original optimization settings changed")
    for phase in ("before", "after"):
        graph = row[phase]
        require(set(graph) == {"nodes", "edges"} and len(graph["nodes"]) == len(ids)
                and isinstance(graph["edges"], list) and len(graph["edges"]) <= 1024,
                "Graph node/edge inventory incomplete")
        for pose in graph["nodes"]:
            rigid_pose(pose)
        for edge in graph["edges"]:
            require(set(edge) == {"source_node","target_node","uncertain","confidence","transformation","information"}
                    and type(edge["uncertain"]) is bool and all(type(edge[k]) is int and 0 <= edge[k] < len(ids)
                        for k in ("source_node","target_node")) and edge["source_node"] != edge["target_node"]
                    and type(edge["confidence"]) in (int,float) and math.isfinite(edge["confidence"])
                    and 0 <= edge["confidence"] <= 1, "Malformed actual optimizer edge")
            rigid_pose(edge["transformation"])
            matrix_evidence(edge["information"], 6)
    return row


def graph_membership(row):
    graph_record(row)
    result = {"context":row["context"], "options":row["options"]}
    for phase in ("before", "after"):
        edges = row[phase]["edges"]
        # Creation-time JSON schema: a closed quality file must compare exactly
        # with a fresh computation, including nonempty optimizer edge triples.
        membership = sorted([edge["source_node"], edge["target_node"], edge["uncertain"]] for edge in edges)
        def components(edges):
            groups = list(range(len(row[phase]["nodes"])))
            def root(index):
                while groups[index] != index:
                    index = groups[index]
                return index
            for a,b,_ in edges:
                groups[root(b)] = root(a)
            return sorted(sorted(i for i in range(len(groups)) if root(i) == representative)
                for representative in {root(i) for i in range(len(groups))})
        retained = sorted([edge["source_node"],edge["target_node"],edge["uncertain"]]
            for edge in edges if not edge["uncertain"] or edge["confidence"] >= .25)
        result[phase] = {"edges":membership,"constructed_components":components(membership),
            "retained_edges":retained,"retained_components":components(retained)}
    return result


def trace_rows(record, tracked):
    path = original.closed_file(record, tracked)
    rows = [original.strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(0 < len(rows) <= MAX_TRACE_EVENTS and len(rows) == record.get("rows"), "Trace row inventory differs")
    return rows


def closed_field_envelope(path, mode, checkpoint_authority, artifacts, component_proof, tracked, proposal_policy):
    path = Path(path).resolve(strict=True)
    tracked[path] = file_hash(path)
    report = original.read_json(path)
    delegated_pins = normalized_artifacts(checkpoint_authority.manifest["producer_artifacts_sha256"])
    require(set(artifacts) == set(FINISH_ARTIFACTS) and set(delegated_pins) == set(checkpoint.FINISH_ARTIFACTS),
            "Distinct field/delegated source inventory differs")
    require(report.get("kind") == KIND and report.get("mode") == mode and report.get("status") == "complete"
            and report.get("proposal_policy") == proposal_policy and report.get("performance_attribution_valid") is (mode != "audit")
            and report.get("native_history_equivalence_claimed") is False and report.get("supervisor_restored") is True
            and not report.get("failure") and not report.get("cleanup_failures"), "New field supervisor did not close cleanly")
    require(normalized_artifacts(report["artifacts_sha256"]) == normalized_artifacts(report["artifacts_sha256_after"]) == artifacts
            and normalized_artifacts(report["delegated_artifacts_sha256"]) == delegated_pins,
            "Original/new field dependency closure differs")
    delegated_path = original.closed_file(report["delegated_sidecar"], tracked)
    delegated,matches,gates = _CLOSED_SIDECAR(delegated_path, mode, checkpoint_authority,
                                            delegated_pins, component_proof, tracked)
    require(report["registration"] == delegated["registration"] and report["actual_trace"] == delegated["trace"],
            "Supervisor registration/trace differs from actual delegated result")
    for name in ("checkpoint","scope_binding","scope_binding_sha256","runtime_binding",
                 "runtime_binding_after","component_proof","component_proof_after","profile"):
        require(report.get(name) == delegated.get(name), "Supervisor/actual delegated binding differs: "+name)
    require(original.finite_nonnegative(report.get("finish_s")) and report["finish_s"] == delegated.get("finish_s"),
            "Field Finish timing differs from actual measurement")
    rows = trace_rows(report["actual_trace"], tracked)
    graphs = [graph_record(row) for row in rows if row["event"] == "graph_optimization"]
    require([row["graph_index"] for row in graphs] == list(range(len(graphs))), "Optimization graph order/suffix missing")
    scope = report["registration"].get("field_conformance", {})
    from scripts.research.finish_resident_registration import FinishRegistrationScope
    dispatch_ast = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(FinishRegistrationScope.dispatch))),include_attributes=False)
    dispatch_hash = hashlib.sha256(dispatch_ast.encode()).hexdigest()
    require(scope.get("proposal_policy") == proposal_policy and scope.get("policy_restored") is True
            and scope.get("original_dispatch_math_and_cpu_shadows_unchanged") is True
            and scope.get("original_dispatch_ast_sha256") == dispatch_hash and scope.get("graph_observer_enabled") is True
            and scope.get("event_match_count") == len(matches) and scope.get("event_gate_count") == len(gates)
            and scope.get("event_graph_count") == scope.get("graph_calls") == len(graphs)
            and type(scope.get("successful_builds")) is int and scope["successful_builds"] == 1
            and scope.get("final_pose_inventory") == report.get("final_pose_inventory")
            and scope.get("native_history_equivalence_claimed") is False,
            "Actual input, output gate or optimization evidence incomplete")
    canonical = scope.get("canonical_proposals")
    if proposal_policy == "original":
        require(canonical is None, "Original policy silently used canonical proposal inputs")
    else:
        require(isinstance(canonical,dict) and canonical.get("hooks_restored") is True
                and not canonical.get("failure") and not canonical.get("cleanup_failures"), "Canonical proposal scope did not close")
    profile = original.read_json(report["profile"]["path"])
    pose_inventory(report["final_pose_inventory"], profile)
    require(profile.get("bundle_adjustment", {}).get("applied") is False,
            "Applied bundle adjustment needs separate tracked landmark/witness inventory before authority")
    return report, delegated, matches, gates, graphs, rows


def final_discrete_inventory(profile):
    """Only actual final graph/coverage/stage decisions, not discarded history."""
    accepted = profile["accepted_indices"]
    require(isinstance(accepted,list) and accepted and len(accepted) == len(set(accepted))
            and all(type(index) is int and index >= 0 for index in accepted) and profile.get("mesh_built") is True,
            "Incomplete successful Final view inventory")
    stages = {}
    for name in ("fragment_reconnection","refinement","bundle_adjustment","final_reconstruction"):
        stage = profile[name]
        require(type(stage.get("applied")) is bool and type(stage.get("failed",False)) is bool,
                "Stage acceptance state absent")
        stages[name] = {"applied":stage["applied"], "failed":stage.get("failed",False)}
    require(stages["final_reconstruction"] == {"applied":True,"failed":False}
            and stages["bundle_adjustment"]["applied"] is False, "Unproved Final/landmark state")
    fragment = profile["fragment_reconnection"]
    inventory = []
    frames = {}
    for row in fragment.get("fragments", []):
        index = row["id"]
        require(type(index) is int and index == len(inventory) and type(row["connected"]) is bool,
                "Fragment identity/order malformed")
        view_sets = []
        for name in ("frame_indices","context_frame_indices"):
            values = row[name]
            require(isinstance(values,list) and len(values) == len(set(values))
                    and all(type(value) is int and value >= 0 for value in values), "Fragment view identity malformed")
            view_sets.append(values)
        frames[index] = set(view_sets[0]+view_sets[1])
        inventory.append({"id":index,"connected":row["connected"],"frame_indices":view_sets[0],"context_frame_indices":view_sets[1]})
    require(len(inventory) <= 32, "Unbounded final fragments")
    if stages["fragment_reconnection"]["applied"]:
        require(inventory and isinstance(fragment.get("connected_fragments"),list)
                and isinstance(fragment.get("unconnected_fragments"),list), "Applied fragment graph lacks final membership")
        connected = fragment["connected_fragments"]
        disconnected = fragment["unconnected_fragments"]
        require(connected == [row["id"] for row in inventory if row["connected"]]
                and disconnected == [row["id"] for row in inventory if not row["connected"]],
                "Final connected fragment labels disagree")
    all_views = [i for item in inventory for i in item["frame_indices"]]
    require(len(all_views) == len(set(all_views)), "View duplicated between actual fragments")
    if inventory:
        require(set(accepted) <= set(all_views), "Final accepted views absent from fragment graph")
        require(type(fragment.get("budget_limited")) is bool and fragment.get("pair_budget") == 256
                and fragment.get("fragment_limit") == 32 and isinstance(fragment.get("ambiguous_pairs"),list),
                "Final fragment search/ambiguity limits absent")
    retained = []
    for edge in fragment.get("verified_bridges", []):
        require(type(edge.get("connected_to_scan")) is bool, "Actual retained bridge membership absent")
        if not edge["connected_to_scan"]:
            continue
        a,b = edge["source"],edge["target"]
        require(type(a) is int and type(b) is int and a != b and a in frames and b in frames,
                "Retained bridge endpoint absent")
        support = edge["support"]
        require(isinstance(support,list) and support and all(isinstance(pair,(list,tuple)) and len(pair) == 2
                and type(pair[0]) is int and type(pair[1]) is int and pair[0] in frames[a] and pair[1] in frames[b]
                for pair in support) and len({tuple(pair) for pair in support}) == len(support),
                "Retained original independent camera witness membership malformed")
        retained.append({"source":a,"target":b,"support":sorted([list(pair) for pair in support]),
            "validation_scope":edge["validation_scope"],"visual_constraint":edge.get("visual_constraint",False)})
    retained.sort(key=strict_json_text)
    return {"accepted_indices":accepted,"stage_acceptance":stages,"fragments":inventory,"retained_bridges":retained,
        "connected_fragments":fragment.get("connected_fragments",[]),
        "unconnected_fragments":fragment.get("unconnected_fragments",[]),
        "excluded_frames":fragment.get("excluded_frames",[]),
        "invalid_indices":fragment.get("invalid_indices",[]),"unassigned_indices":fragment.get("unassigned_indices",[]),
        "loop_closures":fragment.get("loop_closures",[]),"ambiguous_pairs":fragment.get("ambiguous_pairs",[]),
        "budget_limited":fragment.get("budget_limited",False),"pair_budget":fragment.get("pair_budget"),
        "fragment_limit":fragment.get("fragment_limit"),"view_coverage_count":len(all_views)}


def pose_deltas(left, right):
    require([i for i,_ in left] == [i for i,_ in right], "Unrounded final pose identities differ")
    deltas = []
    for (index,a),(_,b) in zip(left,right):
        translation = math.sqrt(sum((a[i][3]-b[i][3])**2 for i in range(3)))
        cosine = (sum(a[i][j]*b[i][j] for i in range(3) for j in range(3))-1)/2
        rotation = math.degrees(math.acos(max(-1.,min(1.,cosine))))
        deltas.append({"index":index,"translation_m":translation,"rotation_deg":rotation})
    return {"limits":dict(POSE_LIMITS),"rows":deltas,
        "max_translation_m":max(row["translation_m"] for row in deltas),
        "max_rotation_deg":max(row["rotation_deg"] for row in deltas),
        "passed":all(row["translation_m"] <= POSE_LIMITS["translation_m"]
            and row["rotation_deg"] <= POSE_LIMITS["rotation_deg"] for row in deltas)}


def final_output_conformance(native, audit, native_graphs, audit_graphs):
    left = original.read_json(native["profile"]["path"])
    right = original.read_json(audit["profile"]["path"])
    a,b = final_discrete_inventory(left), final_discrete_inventory(right)
    poses = pose_deltas(pose_inventory(native["final_pose_inventory"],left),pose_inventory(audit["final_pose_inventory"],right))
    def retained_graphs(profile, graphs):
        require(not profile["refinement"]["applied"] or any(row["context"]["owner"] == "refinement.propose_poses" for row in graphs),
                "Applied refinement lacks its actual optimizer graph")
        fragment = profile["fragment_reconnection"]
        require(not (fragment["applied"] and len(fragment.get("connected_fragments",[])) > 1)
                or any(row["context"]["owner"] == "fragments.propose_fragment_poses" for row in graphs),
                "Connected multi-fragment Final lacks its actual optimizer graph")
        return [graph_membership(row) for row in graphs if profile[
            "refinement" if row["context"]["owner"] == "refinement.propose_poses" else "fragment_reconnection"]["applied"]]
    graphs_a,graphs_b = retained_graphs(left,native_graphs),retained_graphs(right,audit_graphs)
    graph_equal = graphs_a == graphs_b
    discrete_equal = a == b
    return {"passed":discrete_equal and graph_equal and poses["passed"],
        "final_discrete_equal":discrete_equal,"retained_optimizer_graphs_equal":graph_equal,
        "native_final_discrete":a,"candidate_final_discrete":b,
        "native_retained_optimizer_graphs":graphs_a,"candidate_retained_optimizer_graphs":graphs_b,
        "unrounded_final_poses":poses,"native_history_equivalence_claimed":False}


def validate_surface_comparison(comparison, audit, native):
    require(comparison.get("candidate_report_sha256") == audit["profile"]["sha256"]
            and comparison.get("baseline_report_sha256") == native["profile"]["sha256"]
            and comparison.get("same_mesh_success") is True and comparison.get("same_accepted_indices") is True,
            "Physical quality references changed reports or accepted views")
    surface = comparison["triangle_surface_metrics_vs_cpu"]
    require(surface.get("threshold_m") == .005 and surface.get("samples_per_surface") == 30000
            and surface.get("alignment") == "fixed input coordinate frame; no scale or trajectory fitting"
            and original.finite_nonnegative(surface.get("surface_p95_m")) and surface["surface_p95_m"] <= .0005
            and all(original.finite_nonnegative(surface.get(key)) and .999 <= surface[key] <= 1
                for key in ("precision","completeness")), "Declared fixed 0.5mm p95/99.9% surface criteria failed")


def validate_quality(path, audit_path, audit, graphs, authority, artifacts, component_proof, tracked, policy):
    path = Path(path).resolve(strict=True)
    tracked[path] = file_hash(path)
    value = original.read_json(path)
    audit_record = {"path":str(Path(audit_path).resolve()),"sha256":tracked[Path(audit_path).resolve()]}
    require(value.get("kind") == QUALITY_KIND and value.get("status") == "passed" and not value.get("failure")
            and value.get("audit") == audit_record and value.get("checkpoint") == authority.record
            and value.get("proposal_policy") == policy and value.get("scope_binding_sha256") == audit["scope_binding_sha256"]
            and value.get("native_history_equivalence_claimed") is False and value.get("declared_output_criteria") == OUTPUT_CRITERIA,
            "New field quality absent/stale or relabelled strict history proof")
    native_path = original.closed_file(value["native_envelope"],tracked)
    require(native_path != Path(audit_path).resolve(), "Candidate cannot be its own native reference")
    native,_,_,_,native_graphs,_ = closed_field_envelope(native_path,"native",authority,artifacts,component_proof,tracked,policy)
    require(native["scope_binding"] == audit["scope_binding"] and value.get("candidate") == audit["profile"]
            and value.get("native") == native["profile"] and value.get("candidate_trace") == audit["actual_trace"]
            and value.get("native_trace") == native["actual_trace"] and value.get("candidate_sidecar") == audit["delegated_sidecar"]
            and value.get("native_sidecar") == native["delegated_sidecar"], "Quality native/candidate closure differs")
    actual = final_output_conformance(native,audit,native_graphs,graphs)
    require(actual["passed"] is True and value.get("final_output_conformance") == actual,
            "Actual final graph/witness/stage/pose conformance failed")
    validate_surface_comparison(value["comparison"],audit,native)
    require(value.get("comparison_artifact_sha256") == artifacts[COMPARATOR] == file_hash(ROOT/COMPARATOR), "Field quality comparator changed")
    tracked[ROOT/COMPARATOR] = value["comparison_artifact_sha256"]
    return tracked[path]


def _register(token, matches, gates, graphs, rows):
    total = sum(len(getattr(token,name).encode()) for name in ("expected_match_results_json","expected_gate_records_json",
        "expected_graph_records_json","expected_event_order_json","expected_final_poses_json","field_scope_json"))
    require(total <= MAX_TOKEN_BYTES, "Field timing authority exceeds bounded trajectory storage")
    while len(_ACTIVE) >= 4:
        _ACTIVE.popitem(last=False)
    _ACTIVE[id(token)] = {"token":token,"matches":matches,"gates":gates,"graphs":graphs,
        "events":[(row["event"],row[{"match":"call_index","gate":"gate_index","graph_optimization":"graph_index"}[row["event"]]]) for row in rows],
        "entry":0,"event":0,"final_poses":False}
    return token


def _state(token):
    state = _ACTIVE.get(id(token))
    require(type(token) is FieldFinishConformanceAuthority and state is not None and state["token"] is token,
            "Only a freshly validated distinct field authority can dispatch")
    return state


def validate_field_finish_conformance(audit_path, component_authority, expected_scope_binding, expected_artifacts,
        *, quality_path, checkpoint_authority, bulk_authority=None, proposal_policy="original"):
    try:
        require(proposal_policy in PROPOSAL_POLICIES and type(component_authority) is DeviceFlatGridProofAuthority
                and type(checkpoint_authority) is checkpoint.CheckpointAuthority, "Require current separate component/checkpoint authorities")
        artifacts = normalized_artifacts(expected_artifacts)
        require(set(artifacts) == set(FINISH_ARTIFACTS), "Exact old15/new6 dependency pins required")
        runtime = component_authority.runtime_binding
        old_pins = {name:artifacts[name] for name in checkpoint.FINISH_ARTIFACTS}
        require(checkpoint_authority.manifest["runtime_binding"] == runtime
                and normalized_artifacts(checkpoint_authority.manifest["producer_artifacts_sha256"]) == old_pins,
                "Measured Live checkpoint/runtime sources differ")
        require(checkpoint.validate_checkpoint_manifest(checkpoint_authority.path,runtime,old_pins,
            checkpoint_authority.manifest["scope_base"]) == checkpoint_authority, "Checkpoint authority stale")
        component_artifacts = validate_configuration(runtime)
        require(component_authority.bindings_sha256 == canonical_hash(runtime)
                and normalized_artifacts(dict(component_authority.artifact_sha256)) == normalized_artifacts(runtime["artifacts_sha256"]),
                "Component actual configuration/source authority differs")
        validate_current_artifacts(runtime,component_artifacts)
        audit_path = Path(audit_path).resolve(strict=True)
        first = original.read_json(audit_path)
        proof = first["component_proof"]
        require(proof["synthetic"]["sha256"] == component_authority.synthetic_report_sha256
                and proof["bridge"]["sha256"] == component_authority.bridge_report_sha256
                and Path(proof["synthetic"]["path"]).resolve() != Path(proof["bridge"]["path"]).resolve(),
                "Separate current component audit prerequisites differ")
        tracked = {}
        audit,delegated,matches,gates,graphs,rows = closed_field_envelope(audit_path,"audit",checkpoint_authority,
            artifacts,proof,tracked,proposal_policy)
        require(audit["scope_binding"] == expected_scope_binding, "Requested timing raw/checkpoint/settings/thread scope differs")
        checkpoint.validate_field_cpu_auditor(delegated,matches,bulk_authority,runtime,artifacts,tracked)
        quality_hash = validate_quality(quality_path,audit_path,audit,graphs,checkpoint_authority,artifacts,proof,tracked,proposal_policy)
        validate_current_artifacts(runtime,component_artifacts)
        require(all(file_hash(ROOT/name) == value for name,value in artifacts.items())
                and all(file_hash(path) == value for path,value in tracked.items()), "Field proof/current sources/raw outputs changed during validation")
        order = [(row["event"],row[{"match":"call_index","gate":"gate_index","graph_optimization":"graph_index"}[row["event"]]]) for row in rows]
        values = dict(component_authority.__dict__,target_digests=frozenset(audit["registration"]["target_digests"]))
        token = FieldFinishConformanceAuthority(**values,field_scope_json=strict_json_text(expected_scope_binding),
            expected_call_signatures=tuple(row["call_signature"] for row in matches),
            expected_match_results_json=strict_json_text([row["result"] for row in matches]),
            expected_gate_records_json=strict_json_text(gates),expected_graph_records_json=strict_json_text(graphs),
            expected_event_order_json=strict_json_text(order),expected_final_poses_json=strict_json_text(audit["final_pose_inventory"]),
            conformance_audit_report_sha256=tracked[audit_path],
            quality_proof_sha256=quality_hash,conformance_artifact_sha256=tuple(sorted(artifacts.items())),proposal_policy=proposal_policy)
        return _register(token,matches,gates,graphs,rows)
    except GridProofError:
        raise
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale actual-input field conformance proof: {error}") from error


def validate_expected_field_call(token, index, payload):
    state = _state(token)
    require(type(index) is int and index == state["entry"] and 0 <= index < len(token.expected_call_signatures),
            "Field call order/prefix/repeated entry differs")
    signature = original.call_signature(payload)
    require(signature == token.expected_call_signatures[index] and payload["target_points"]["sha256"] in token.target_digests,
            "Actual field points/normals/unrounded seed/site/gate/thread/stage inputs differ before GPU work")
    state["entry"] += 1
    return signature


def match_result_equal(left,right):
    keys = {"transformation","fitness","rmse","correspondence_count","correspondence_sha256","correspondence_mapping_sha256"}
    require(isinstance(left,dict) and isinstance(right,dict) and set(left) == set(right) == keys,
            "Complete original registration result summary absent")
    require(all(digest(value[name]) for value in (left,right) for name in ("correspondence_sha256","correspondence_mapping_sha256")),
            "Complete raw/canonical correspondence diagnostics absent")
    require(all(type(value["correspondence_count"]) is int and value["correspondence_count"] >= 0 for value in (left,right)),
            "Canonical correspondence count malformed")
    # Raw vector row order is diagnostic. Canonical source-ID mapping and its
    # exact count remain authority; original returned order is never changed.
    a,b = ({key:value for key,value in result.items() if key != "correspondence_sha256"} for result in (left,right))
    return semantic_agreement(a,b,"match/result")


def validate_expected_field_event(token, row):
    state = _state(token)
    event = row.get("event")
    index_key = {"match":"call_index","gate":"gate_index","graph_optimization":"graph_index"}.get(event)
    require(index_key is not None and row.get("complete") is True and type(row.get(index_key)) is int,
            "Incomplete/unknown field timing event")
    index = row[index_key]
    require(state["event"] < len(state["events"]) and state["events"][state["event"]] == (event,index),
            "Actual audited event completion order/prefix differs")
    if event == "match":
        require(index < state["entry"] and original.call_signature(row["call_inputs"]) == token.expected_call_signatures[index]
                == row["call_signature"] == row["inputs_after_resident"]
                and type(row.get("resident_full_call_fallbacks")) is int and row["resident_full_call_fallbacks"] == 0
                and match_result_equal(state["matches"][index]["result"],row["result"]),
                "Timing registration result/input/NN mapping differs from its own complete audited GPU call")
    elif event == "gate":
        expected = state["gates"][index]
        require(row["function"] == expected["function"] and semantic_agreement(expected["result"],row["result"],row["function"]+"/result"),
                "Timing original gate/witness/information result differs from its audited GPU trajectory")
    else:
        graph_record(row)
        require(graph_membership(state["graphs"][index]) == graph_membership(row)
                and semantic_agreement(state["graphs"][index],row,"graph_optimization"),
                "Timing actual optimization graph/identity/result differs from audited GPU trajectory")
    state["event"] += 1


def validate_complete_field_calls(token, call_count, gate_count, graph_count=0):
    state = _state(token)
    require(all(type(value) is int for value in (call_count,gate_count,graph_count))
            and call_count == state["entry"] == len(token.expected_call_signatures)
            and gate_count == len(state["gates"]) and graph_count == len(state["graphs"])
            and state["event"] == len(state["events"]) and state["final_poses"] is True,
            "Field timing omitted audited call/gate/optimization/final-pose suffix")


def validate_expected_field_final_poses(token, inventory):
    state = _state(token)
    expected = original.strict_json(token.expected_final_poses_json)
    ids = [row["index"] for row in expected["rows"]]
    pose_inventory(expected,{"accepted_indices":ids})
    pose_inventory(inventory,{"accepted_indices":ids})
    require(not state["final_poses"] and semantic_agreement(expected["rows"],inventory["rows"],"final/poses"),
            "Timing final unrounded poses/order differ from its own audited GPU trajectory")
    state["final_poses"] = True

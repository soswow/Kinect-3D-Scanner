"""New actual-input field conformance scope; old strict authorities stay distinct.

Original capture, CPU shadows, hard fault latch and rollback are inherited.
Only the timing validator names in the unchanged dispatch are substituted.
Native history equivalence is not asserted by this scope. Timing nevertheless
must reproduce its own audited GPU trajectory, gate outputs and terminal count.
"""

from __future__ import annotations
import ast
import hashlib
import inspect
import json
import math
import operator
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import finish_resident_registration as original

PROPOSAL_POLICIES = ("original", "canonical-fresh-1um")


def expected_call(authority, index, payload):
    from scripts.research.validate_field_finish_conformance import validate_expected_field_call
    return validate_expected_field_call(authority, index, payload)


def expected_event(authority, row):
    from scripts.research.validate_field_finish_conformance import validate_expected_field_event
    return validate_expected_field_event(authority, row)


def complete_calls(authority, calls, gates, graphs):
    from scripts.research.validate_field_finish_conformance import validate_complete_field_calls
    return validate_complete_field_calls(authority, calls, gates, graphs)


def expected_final_poses(authority, inventory):
    from scripts.research.validate_field_finish_conformance import validate_expected_field_final_poses
    return validate_expected_field_final_poses(authority, inventory)


def matrix_evidence(value, shape, copier=original.evidence):
    item = copier(value)
    if (not isinstance(item, dict) or item.get("dtype") != "<f8" or item.get("shape") != list(shape)
            or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64
            or not isinstance(item.get("values"), list) or len(item["values"]) != shape[0]):
        raise ValueError("Require bounded original FP64 matrix evidence with values/hash")
    for row in item["values"]:
        if (not isinstance(row, list) or len(row) != shape[1]
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
            raise ValueError("Original matrix evidence is malformed or nonfinite")
    if any(x not in "0123456789abcdef" for x in item["sha256"]):
        raise ValueError("Original matrix evidence hash is malformed")
    return item


def graph_snapshot(graph, copier=original.evidence):
    nodes, edges = list(graph.nodes), list(graph.edges)
    if not 0 < len(nodes) <= 64 or len(edges) > 1024:
        raise ValueError("Original optimization graph exceeds the explicit observation bound")
    result = {"nodes": [matrix_evidence(node.pose, (4, 4), copier) for node in nodes], "edges": []}
    for edge in edges:
        source, target = operator.index(edge.source_node_id), operator.index(edge.target_node_id)
        if not (0 <= source < len(nodes) and 0 <= target < len(nodes)):
            raise ValueError("Original graph edge endpoint is invalid")
        confidence = float(edge.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Original graph edge confidence is invalid")
        result["edges"].append({"source_node": source, "target_node": target,
            "uncertain": bool(edge.uncertain), "confidence": confidence,
            "transformation": matrix_evidence(edge.transformation, (4, 4), copier),
            "information": matrix_evidence(edge.information, (6, 6), copier)})
    return result


def final_pose_inventory(engine, copier=original.evidence):
    rows, ids = [], set()
    if not 0 < len(engine.poses) <= 4096:
        raise ValueError("Final pose inventory exceeds the explicit observation bound")
    for index, pose in engine.poses:
        index = operator.index(index)
        if index < 0 or index in ids:
            raise ValueError("Final raw view identities are invalid or duplicated")
        ids.add(index)
        rows.append({"index": index, "camera_to_world": matrix_evidence(pose, (4, 4), copier)})
    return {"pose_convention": "camera_to_world", "length_unit": "metres",
            "captured_after_successful_build": True, "rows": rows,
            "pose_inventory_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def validate_graph_context(context, nodes):
    keys = {"refinement.propose_poses": "node_raw_view_indices",
            "fragments.propose_fragment_poses": "node_fragment_indices"}
    if not isinstance(context, dict) or context.get("owner") not in keys:
        raise ValueError("Unknown original optimization graph ownership")
    key = keys[context["owner"]]
    ids = context.get(key)
    if (set(context) != {"owner", key} or not isinstance(ids, list) or len(ids) != nodes
            or any(type(index) is not int or index < 0 for index in ids) or len(set(ids)) != nodes):
        raise ValueError("Original optimization node ownership is malformed")


def graph_context():
    """Observe bounded raw-view/fragment identities from the original caller."""
    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            path = Path(frame.f_code.co_filename).resolve()
            if path == ROOT / "scanner_server/refinement.py" and frame.f_code.co_name == "propose_poses":
                values = frame.f_locals
                ids = [int(values["poses"][int(i)][0]) for i in values["chosen"]]
                return {"owner": "refinement.propose_poses", "node_raw_view_indices": ids}
            if path == ROOT / "scanner_server/fragments.py" and frame.f_code.co_name == "propose_fragment_poses":
                ids = [int(i) for i in frame.f_locals["node_ids"]]
                return {"owner": "fragments.propose_fragment_poses", "node_fragment_indices": ids}
            frame = frame.f_back
        raise ValueError("Unknown original optimization caller; cannot claim graph ownership")
    finally:
        del frame


def derived_dispatch():
    """Only replace the old exact-type validator; recover original AST exactly."""
    source = textwrap.dedent(inspect.getsource(original.FinishRegistrationScope.dispatch))
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
             and node.id == "validate_expected_call"]
    if len(nodes) != 1:
        raise ValueError("Original dispatch validator seam is ambiguous")
    before = ast.dump(tree, include_attributes=False)
    nodes[0].id = "field_expected_call"
    namespace = dict(original.__dict__, field_expected_call=expected_call)
    exec(compile(tree, __file__, "exec"), namespace)
    derived = namespace["dispatch"]
    nodes[0].id = "validate_expected_call"
    if ast.dump(tree, include_attributes=False) != before:
        raise ValueError("Field timing changed original capture/shadow/registration dispatch")
    return derived, hashlib.sha256(before.encode()).hexdigest()


_DISPATCH, ORIGINAL_DISPATCH_AST_SHA256 = derived_dispatch()


class FieldConformanceScope(original.FinishRegistrationScope):
    dispatch = _DISPATCH

    def __init__(self, *args, proposal_policy="original", canonical_factory=None,
                 graph_module=None, graph_context_capture=graph_context, graph_copier=original.evidence,
                 final_engine_type=None, final_pose_capture=final_pose_inventory, **kwargs):
        if proposal_policy not in PROPOSAL_POLICIES:
            raise ValueError("Unknown explicit field proposal policy")
        self.proposal_policy = proposal_policy
        self.canonical_factory = canonical_factory
        self.canonical_scope = None
        self.canonical_report = None
        self.policy_restored = proposal_policy == "original"
        self.event_match_count = self.event_gate_count = 0
        self.event_graph_count = 0
        self.graph_module, self.graph_context_capture, self.graph_copier = graph_module, graph_context_capture, graph_copier
        self.graph_calls = []
        self.final_engine_type, self.final_pose_capture = final_engine_type, final_pose_capture
        self.final_poses, self.successful_builds = None, 0
        self.exit_attempted = False
        self.original_trace = kwargs.pop("trace", None) or (lambda row: None)
        super().__init__(*args, trace=self.checked_trace, **kwargs)
        if self.mode != "timing" and self.authority is not None:
            raise ValueError("Native/audit field scope cannot consume any timing token")

    def checked_trace(self, row):
        try:
            if self.mode == "timing" and row.get("complete"):
                expected_event(self.authority, row)
            if row.get("complete"):
                if row["event"] == "match":
                    self.event_match_count += 1
                elif row["event"] == "gate":
                    self.event_gate_count += 1
                elif row["event"] == "graph_optimization":
                    self.event_graph_count += 1
                else:
                    raise ValueError("Unknown field trace event")
            self.original_trace(row)
        except BaseException as error:
            self.fail(error)

    def _make_canonical_scope(self):
        if self.canonical_factory is not None:
            return self.canonical_factory()
        import numpy as np
        import open3d as o3d
        from scanner_server import fragments
        from scripts.research.canonical_fpfh_proposals import FreshCanonicalFeatures, CanonicalGlobalSeedScope
        return CanonicalGlobalSeedScope(fragments, FreshCanonicalFeatures(np, o3d, normal_policy="fresh"))

    def __enter__(self):
        try:
            # The private proposal wrapper captures the original function before
            # the frozen Finish gate observers and RANSAC thread scope enter.
            if self.proposal_policy == "canonical-fresh-1um":
                self.canonical_scope = self._make_canonical_scope()
                self.canonical_scope.__enter__()
            super().__enter__()
            if self.graph_module is not None:
                self._observe_graph_optimizer()
            if self.final_engine_type is not None:
                self._observe_final_build()
            return self
        except BaseException as primary:
            if self.entered and not self.exit_attempted:
                # A hook added after the original registration entry can fail.
                # Python will not call __exit__ for that failed entry.
                self.__exit__(type(primary), primary, primary.__traceback__)
            elif self.canonical_scope is not None:
                try:
                    self._restore_policy(type(primary), primary, primary.__traceback__)
                except BaseException as cleanup:
                    primary.add_note(f"Canonical partial-entry cleanup also failed: {cleanup}")
            raise

    def _observe_graph_optimizer(self):
        optimizer = self.graph_module.global_optimization
        def observed(*args, **kwargs):
            self.healthy()
            row = {"event": "graph_optimization", "graph_index": len(self.graph_calls),
                   "gate_context": list(self.gate_context.get()), "complete": False}
            self.graph_calls.append(row["graph_index"])
            try:
                if len(args) != 4 or kwargs:
                    raise ValueError("Require original graph optimizer call with all four original positional arguments")
                graph, _, _, option = args
                row["context"] = self.graph_context_capture()
                row["options"] = {"max_correspondence_distance": float(option.max_correspondence_distance),
                    "edge_prune_threshold": float(option.edge_prune_threshold), "reference_node": int(option.reference_node)}
                row["before"] = graph_snapshot(graph, self.graph_copier)
                validate_graph_context(row["context"], len(row["before"]["nodes"]))
                if (row["options"]["max_correspondence_distance"] != .03
                        or row["options"]["edge_prune_threshold"] != .25
                        or not 0 <= row["options"]["reference_node"] < len(row["before"]["nodes"])):
                    raise ValueError("Original optimization options differ from declared gate policy")
            except BaseException as error:
                self.fail(error)
            try:
                result = optimizer(*args, **kwargs)
            except BaseException as error:
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                self.checked_trace(row)
                raise
            try:
                row.update(after=graph_snapshot(graph, self.graph_copier), complete=True)
                self.checked_trace(row)
                return result
            except BaseException as error:
                self.fail(error)
        self.patches.append((self.graph_module, "global_optimization", optimizer))
        self.graph_module.global_optimization = observed

    def _observe_final_build(self):
        build = self.final_engine_type.build_mesh
        def observed(engine, *args, **kwargs):
            self.healthy()
            result = build(engine, *args, **kwargs)
            try:
                self.healthy()
                if not isinstance(result, tuple) or len(result) != 2 or result[0] is not True:
                    raise ValueError("Conformance requires an original successful transactional final build")
                if self.successful_builds:
                    raise ValueError("Conformance cannot accept more than one original final build")
                inventory = self.final_pose_capture(engine)
                if self.mode == "timing":
                    expected_final_poses(self.authority, inventory)
                self.final_poses, self.successful_builds = inventory, 1
                return result
            except BaseException as error:
                self.fail(error)
        self.patches.append((self.final_engine_type, "build_mesh", build))
        self.final_engine_type.build_mesh = observed

    def _restore_policy(self, kind, error, traceback):
        scope, self.canonical_scope = self.canonical_scope, None
        if scope is not None:
            primary = None
            try:
                scope.__exit__(kind, error, traceback)
            except BaseException as failure:
                primary = failure
            try:
                self.canonical_report = scope.report()
                self.policy_restored = self.canonical_report["hooks_restored"]
            except BaseException as reporting_error:
                if primary is None:
                    primary = reporting_error
                else:
                    primary.add_note(f"Canonical restoration report also failed: {reporting_error}")
            if primary is not None:
                raise primary

    def __exit__(self, kind, error, traceback):
        self.exit_attempted = True
        primary = None
        try:
            super().__exit__(kind, error, traceback)
        except BaseException as failure:
            primary = failure
        try:
            self._restore_policy(type(primary) if primary else kind,
                                 primary if primary else error,
                                 primary.__traceback__ if primary else traceback)
        except BaseException as cleanup:
            self.cleanup_failures.append(f"Restore canonical proposal scope: {cleanup}")
            if primary is None:
                primary = cleanup
            else:
                primary.add_note(f"Canonical proposal restoration also failed: {cleanup}")
        if primary is not None:
            self.fail(primary)
        return False

    def finish(self):
        self.healthy()
        try:
            if self.canonical_scope is not None and self.canonical_scope.failure is not None:
                raise self.canonical_scope.failure
            if self.mode == "timing":
                complete_calls(self.authority, len(self.calls), len(self.gates), len(self.graph_calls))
            if (not self.calls or self.event_match_count != len(self.calls)
                    or self.event_gate_count != len(self.gates) or self.event_graph_count != len(self.graph_calls)):
                raise ValueError("Field scope omitted or failed to complete call/gate evidence")
            if self.final_engine_type is not None and (self.successful_builds != 1 or self.final_poses is None):
                raise ValueError("Field scope omitted the successful original final pose inventory")
            self.complete = True
        except BaseException as error:
            self.fail(error)

    def report(self):
        value = super().report()
        value.update(field_conformance={"proposal_policy": self.proposal_policy,
            "original_dispatch_ast_sha256": ORIGINAL_DISPATCH_AST_SHA256,
            "original_dispatch_math_and_cpu_shadows_unchanged": True,
            "event_match_count": self.event_match_count, "event_gate_count": self.event_gate_count,
            "event_graph_count": self.event_graph_count, "graph_calls": len(self.graph_calls),
            "graph_observer_enabled": self.graph_module is not None,
            "successful_builds": self.successful_builds, "final_pose_inventory": self.final_poses,
            "policy_restored": self.policy_restored, "canonical_proposals": self.canonical_report,
            "native_history_equivalence_claimed": False})
        return value

"""Artificial stdlib contracts, not numerical/geometry performance evidence."""

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import validate_field_finish_conformance as guard
from scripts.research import compare_field_finish_conformance as quality
from tests.test_finish_resident_contract import payload, component, authority as old_authority


def identity(size=4):
    return [[float(i == j) for j in range(size)] for i in range(size)]


def evidence(values):
    flat = [item for row in values for item in row]
    return {"dtype":"<f8","shape":[len(values),len(values[0])],
        "sha256":hashlib.sha256(struct.pack("<"+"d"*len(flat),*flat)).hexdigest(),"values":copy.deepcopy(values)}


def poses(value=None):
    rows = [{"index":0,"camera_to_world":evidence(value or identity())}]
    return {"pose_convention":"camera_to_world","length_unit":"metres","captured_after_successful_build":True,
        "rows":rows,"pose_inventory_sha256":guard.canonical_hash(rows)}


def result():
    return {"transformation":identity(),"fitness":.8,"rmse":.01,"correspondence_count":4,
        "correspondence_sha256":"0"*64,"correspondence_mapping_sha256":"1"*64}


def match():
    inputs = payload()
    signature = guard.original.call_signature(inputs)
    return {"event":"match","complete":True,"call_index":0,"call_inputs":inputs,
        "call_signature":signature,"inputs_after_resident":signature,"resident_full_call_fallbacks":0,"result":result()}


def gate():
    return {"event":"gate","complete":True,"gate_index":0,"function":"bundle_adjustment.propose_bundle_poses",
        "result":{"elapsed_ms":1.,"time_budget_s":45.,"applied":False,"reason":"unchanged"}}


def graph(confidence=.5):
    edge = {"source_node":0,"target_node":1,"uncertain":True,"confidence":confidence,
        "transformation":evidence(identity()),"information":evidence(identity(6))}
    snapshot = {"nodes":[evidence(identity()),evidence(identity())],"edges":[edge]}
    return {"event":"graph_optimization","complete":True,"graph_index":0,"gate_context":[],
        "context":{"owner":"refinement.propose_poses","node_raw_view_indices":[0,1]},
        "options":{"max_correspondence_distance":.03,"edge_prune_threshold":.25,"reference_node":0},
        "before":copy.deepcopy(snapshot),"after":copy.deepcopy(snapshot)}


def token(rows=None, register=True):
    rows = rows or [match(),gate(),graph()]
    matches = [row for row in rows if row["event"] == "match"]
    gates = sorted([row for row in rows if row["event"] == "gate"],key=lambda row:row["gate_index"])
    graphs = [row for row in rows if row["event"] == "graph_optimization"]
    base = dict(component().__dict__,target_digests=frozenset(row["call_inputs"]["target_points"]["sha256"] for row in matches))
    value = guard.FieldFinishConformanceAuthority(**base,field_scope_json="{}",
        expected_call_signatures=tuple(row["call_signature"] for row in matches),
        expected_match_results_json=guard.strict_json_text([row["result"] for row in matches]),
        expected_gate_records_json=guard.strict_json_text(gates),expected_graph_records_json=guard.strict_json_text(graphs),
        expected_event_order_json=guard.strict_json_text([(row["event"],row.get("call_index",row.get("gate_index",row.get("graph_index")))) for row in rows]),
        expected_final_poses_json=guard.strict_json_text(poses()),conformance_audit_report_sha256="2"*64,
        quality_proof_sha256="3"*64,conformance_artifact_sha256=(),proposal_policy="original")
    return guard._register(value,matches,gates,graphs,copy.deepcopy(rows)) if register else value


def profile():
    return {"accepted_indices":[0],"mesh_built":True,"fragment_reconnection":{"applied":True,
        "fragments":[{"id":0,"connected":True,"frame_indices":[0],"context_frame_indices":[]}],
        "connected_fragments":[0],"unconnected_fragments":[],"verified_bridges":[],"loop_closures":[],
        "budget_limited":False,"pair_budget":256,"fragment_limit":32,"ambiguous_pairs":[]},
        "refinement":{"applied":False},"bundle_adjustment":{"applied":False},"final_reconstruction":{"applied":True}}


class ActualInputContracts(unittest.TestCase):
    def setUp(self):
        guard._ACTIVE.clear()

    def tearDown(self):
        guard._ACTIVE.clear()

    def test_private_common_reader_keeps_original_shadow_and_profile_guards(self):
        self.assertEqual(guard._CLOSED_SIDECAR.__globals__["KIND"],guard.KIND)
        self.assertNotEqual(guard.checkpoint.KIND,guard.KIND)
        self.assertIs(guard._CLOSED_SIDECAR.__globals__["original"],guard.checkpoint.original)
        self.assertIs(guard._CLOSED_SIDECAR.__globals__["device_audit_coverage"],guard.checkpoint.device_audit_coverage)

    def test_old_constructed_or_copied_tokens_are_not_field_authority(self):
        for value in (old_authority([payload()]),token(register=False),replace(token())):
            with self.subTest(kind=type(value).__name__),self.assertRaises(guard.GridProofError):
                guard.validate_expected_field_call(value,0,payload())

    def test_exact_input_recipe_and_entry_order_are_required_before_gpu(self):
        mutations = (lambda p:p["seed"].update(sha256="f"*64),lambda p:p["target_points"].update(sha256="b"*64),
            lambda p:p.update(huber_m=.02),lambda p:p["site"].update(line=25),lambda p:p.update(gate_context=["another"]))
        for mutate in mutations:
            value = token(); changed = payload(); mutate(changed)
            with self.subTest(changed=changed),self.assertRaises(guard.GridProofError):
                guard.validate_expected_field_call(value,0,changed)
            self.assertEqual(guard._state(value)["entry"],0)
        value = token()
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_call(value,1,payload())
        guard.validate_expected_field_call(value,0,payload())
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_call(value,0,payload())

    def test_own_gpu_complete_events_and_final_pose_suffix_close(self):
        value = token()
        guard.validate_expected_field_call(value,0,payload())
        for row in (match(),gate(),graph()): guard.validate_expected_field_event(value,row)
        with self.assertRaises(guard.GridProofError): guard.validate_complete_field_calls(value,1,1,1)
        guard.validate_expected_field_final_poses(value,poses())
        guard.validate_complete_field_calls(value,1,1,1)

    def test_missing_and_reordered_events_and_duplicate_final_inventory_fail(self):
        value = token(); guard.validate_expected_field_call(value,0,payload())
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_event(value,gate())
        guard.validate_expected_field_event(value,match())
        with self.assertRaises(guard.GridProofError): guard.validate_complete_field_calls(value,1,0,0)
        guard.validate_expected_field_final_poses(value,poses())
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_final_poses(value,poses())

    def test_raw_pair_order_diagnostic_passes_but_canonical_mapping_is_exact(self):
        row = match(); row["result"]["correspondence_sha256"] = "4"*64
        value = token(); guard.validate_expected_field_call(value,0,payload())
        guard.validate_expected_field_event(value,row)
        row = match(); row["result"]["correspondence_mapping_sha256"] = "4"*64
        value = token(); guard.validate_expected_field_call(value,0,payload())
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_event(value,row)

    def test_own_result_transform_and_typed_counts_cannot_be_approximate(self):
        mutations = (lambda r:r["transformation"][0].__setitem__(3,1e-6),
            lambda r:r.update(correspondence_count=4.),lambda r:r.update(correspondence_count=True))
        for mutate in mutations:
            changed = result(); mutate(changed)
            with self.subTest(changed=changed):
                try: passed = guard.match_result_equal(result(),changed)
                except guard.GridProofError: passed = False
                self.assertFalse(passed)

    def test_only_named_bundle_elapsed_is_diagnostic(self):
        value = token(); guard.validate_expected_field_call(value,0,payload()); guard.validate_expected_field_event(value,match())
        changed = gate(); changed["result"]["elapsed_ms"] = 1000.
        guard.validate_expected_field_event(value,changed)
        for mutation in (lambda row:row["result"].update(time_budget_s=46.),lambda row:row["result"].update(applied=True)):
            value = token(); guard.validate_expected_field_call(value,0,payload()); guard.validate_expected_field_event(value,match())
            changed = gate(); mutation(changed)
            with self.assertRaises(guard.GridProofError): guard.validate_expected_field_event(value,changed)

    def test_graph_confidence_threshold_is_discrete_even_inside_numeric_tolerance(self):
        baseline = graph(.25+1e-10)
        value = token([match(),gate(),baseline]); guard.validate_expected_field_call(value,0,payload())
        guard.validate_expected_field_event(value,match()); guard.validate_expected_field_event(value,gate())
        changed = graph(.25-1e-10)
        self.assertTrue(guard.semantic_agreement(baseline,changed,"graph_optimization"))
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_event(value,changed)
        selected = guard.graph_membership(changed)["after"]
        self.assertEqual(selected["retained_edges"],[])
        self.assertEqual(selected["retained_components"],[[0],[1]])

    def test_c6_nonempty_optimizer_graph_conformance_survives_exact_json_roundtrip(self):
        # C6 exercises both applied fragment and refinement optimizer graphs;
        # C5's empty graph inventory could not expose v1's tuple/list defect.
        profile_value = profile()
        profile_value["accepted_indices"] = [0,1,2]
        profile_value["refinement"]["applied"] = True
        fragment = profile_value["fragment_reconnection"]
        fragment["fragments"] = [{"id":i,"connected":True,"frame_indices":[i],
            "context_frame_indices":[]} for i in range(3)]
        fragment["connected_fragments"] = [0,1,2]
        graphs = []
        for index,owner in enumerate(("fragments.propose_fragment_poses","refinement.propose_poses")):
            row = graph()
            row["graph_index"] = index
            row["context"] = {"owner":owner,
                "node_fragment_indices" if index == 0 else "node_raw_view_indices":[0,1,2]}
            def edge(a,b,uncertain,confidence):
                value = copy.deepcopy(row["before"]["edges"][0])
                value.update(source_node=a,target_node=b,uncertain=uncertain,confidence=confidence)
                return value
            row["before"] = {"nodes":[evidence(identity()) for _ in range(3)],
                "edges":[edge(0,1,False,1.),edge(0,2,True,.9),edge(1,2,False,1.)]}
            row["after"] = {"nodes":copy.deepcopy(row["before"]["nodes"]),
                "edges":[edge(0,1,False,1.),edge(0,2,True,.1),edge(1,2,False,1.)]}
            graphs.append(row)
        inventory = poses()
        inventory["rows"] = [{"index":i,"camera_to_world":evidence(identity())} for i in range(3)]
        inventory["pose_inventory_sha256"] = guard.canonical_hash(inventory["rows"])
        native = {"profile":{"path":"native.json"},"final_pose_inventory":inventory}
        audit = {"profile":{"path":"audit.json"},"final_pose_inventory":copy.deepcopy(inventory)}
        with patch.object(guard.original,"read_json",return_value=profile_value):
            actual = guard.final_output_conformance(native,audit,graphs,copy.deepcopy(graphs))
        self.assertTrue(actual["passed"])
        self.assertEqual(len(actual["native_retained_optimizer_graphs"]),2)
        for record in actual["native_retained_optimizer_graphs"]:
            for phase in ("before","after"):
                self.assertTrue(record[phase]["edges"] and record[phase]["retained_edges"])
                self.assertTrue(all(type(item) is list for item in record[phase]["edges"]
                    +record[phase]["retained_edges"]))
        self.assertEqual(json.loads(json.dumps(actual,allow_nan=False)),actual)

    def test_json_graph_creation_preserves_exact_retained_edge_membership(self):
        baseline = guard.graph_membership(graph(.25))
        restored = json.loads(json.dumps(baseline,allow_nan=False))
        self.assertEqual(restored,baseline)
        changed = graph(.25-1e-10)
        self.assertNotEqual(guard.graph_membership(changed),restored)
        changed = graph(.25)
        changed["after"]["edges"][0]["source_node"] = 1
        changed["after"]["edges"][0]["target_node"] = 0
        self.assertNotEqual(guard.graph_membership(changed),restored)

    def test_graph_identity_information_and_unknown_owner_fail(self):
        mutations = (lambda row:row["context"].update(owner="opaque"),
            lambda row:row["context"].update(node_raw_view_indices=[0,0]),
            lambda row:row["after"]["edges"][0].update(source_node=2),
            lambda row:row["after"]["nodes"][0]["values"][0].__setitem__(3,.01))
        for mutate in mutations:
            row = graph(); mutate(row)
            with self.subTest(row=row),self.assertRaises(guard.GridProofError): guard.graph_record(row)

    def test_final_pose_inventory_hash_and_success_and_unique_identity_are_required(self):
        mutations = (lambda r:r.update(captured_after_successful_build=False),
            lambda r:r.update(pose_inventory_sha256="f"*64),lambda r:r["rows"][0].update(index=True),
            lambda r:r["rows"][0]["camera_to_world"]["values"][0].__setitem__(3,1e-7))
        for mutate in mutations:
            value = poses(); mutate(value)
            with self.subTest(value=value),self.assertRaises(guard.GridProofError): guard.pose_inventory(value,{"accepted_indices":[0]})

    def test_new_practical_pose_limits_do_not_relax_own_audited_pose_checks(self):
        changed = identity(); changed[0][3] = .0004
        a = guard.pose_inventory(poses(),{"accepted_indices":[0]})
        b = guard.pose_inventory(poses(changed),{"accepted_indices":[0]})
        self.assertTrue(guard.pose_deltas(a,b)["passed"])
        changed[0][3] = .0005001
        self.assertFalse(guard.pose_deltas(a,guard.pose_inventory(poses(changed),{"accepted_indices":[0]}))["passed"])
        value = token()
        with self.assertRaises(guard.GridProofError): guard.validate_expected_field_final_poses(value,poses(changed))

    def test_fixed_rotation_limit_on_unrounded_pose(self):
        a = guard.pose_inventory(poses(),{"accepted_indices":[0]})
        for angle,passed in ((.09,True),(.1001,False)):
            radians = math.radians(angle); value = identity()
            value[0][:2] = [math.cos(radians),-math.sin(radians)]
            value[1][:2] = [math.sin(radians),math.cos(radians)]
            self.assertEqual(guard.pose_deltas(a,guard.pose_inventory(poses(value),{"accepted_indices":[0]}))["passed"],passed)

    def test_final_discrete_stage_coverage_ambiguity_and_budget_are_evidence(self):
        baseline = guard.final_discrete_inventory(profile())
        for mutate in (lambda p:p["fragment_reconnection"].update(budget_limited=True),
            lambda p:p["fragment_reconnection"].update(ambiguous_pairs=[[0,1]]),
            lambda p:p["refinement"].update(applied=True)):
            changed = profile(); mutate(changed)
            self.assertNotEqual(guard.final_discrete_inventory(changed),baseline)
        for mutate in (lambda p:p["bundle_adjustment"].update(applied=True),
            lambda p:p["fragment_reconnection"]["fragments"][0].update(connected=False),
            lambda p:p.update(accepted_indices=[1])):
            changed = profile(); mutate(changed)
            with self.assertRaises(guard.GridProofError): guard.final_discrete_inventory(changed)

    def test_retained_independent_witness_change_is_not_a_history_diagnostic(self):
        left = profile(); left["accepted_indices"] = [0,1]
        left["fragment_reconnection"]["fragments"].append({"id":1,"connected":True,"frame_indices":[1],"context_frame_indices":[2]})
        left["fragment_reconnection"]["connected_fragments"] = [0,1]
        edge = {"source":0,"target":1,"connected_to_scan":True,"support":[[0,1]],"validation_scope":"independent camera pairs"}
        left["fragment_reconnection"]["verified_bridges"] = [edge]
        right = copy.deepcopy(left); right["fragment_reconnection"]["verified_bridges"][0]["support"] = [[0,2]]
        self.assertNotEqual(guard.final_discrete_inventory(left),guard.final_discrete_inventory(right))

    def test_quality_surface_gate_is_fixed_and_stale_profile_fails(self):
        audit = {"profile":{"sha256":"a"*64}}; native = {"profile":{"sha256":"b"*64}}
        comparison = {"candidate_report_sha256":"a"*64,"baseline_report_sha256":"b"*64,
            "same_mesh_success":True,"same_accepted_indices":True,"triangle_surface_metrics_vs_cpu":{
                "threshold_m":.005,"samples_per_surface":30000,"alignment":"fixed input coordinate frame; no scale or trajectory fitting",
                "surface_p95_m":.0005,"precision":.999,"completeness":.999}}
        guard.validate_surface_comparison(comparison,audit,native)
        for mutate in (lambda p:p.update(candidate_report_sha256="c"*64),
            lambda p:p["triangle_surface_metrics_vs_cpu"].update(surface_p95_m=.0005001),
            lambda p:p["triangle_surface_metrics_vs_cpu"].update(precision=.99899),
            lambda p:p["triangle_surface_metrics_vs_cpu"].update(samples_per_surface=1000),
            lambda p:p["triangle_surface_metrics_vs_cpu"].update(alignment="fitted")):
            changed = copy.deepcopy(comparison); mutate(changed)
            with self.assertRaises(guard.GridProofError): guard.validate_surface_comparison(changed,audit,native)

    def test_native_history_drift_is_only_diagnostic_in_new_protocol(self):
        left = [gate()]; right = [gate()]; right[0]["result"]["reason"] = "different discarded history"
        result = quality.history_diagnostic(left,right)
        self.assertFalse(result["raw_history_equal"])
        self.assertFalse(result["native_history_equivalence_claimed"])
        self.assertEqual(guard.QUALITY_KIND,"offline-field-finish-conformance-quality-v2")
        self.assertNotEqual(guard.QUALITY_KIND,guard.checkpoint.QUALITY_KIND)

    def test_cli_help_from_unrelated_folder_has_no_native_imports_or_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            script = guard.ROOT/guard.COMPARATOR
            completed = subprocess.run([sys.executable,"-S",str(script),"--help"],cwd=folder,capture_output=True,text=True,timeout=10)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertIn("usage:",completed.stdout)
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__ == "__main__":
    unittest.main()

"""Artificial quality/fault contracts only; no numerical field authority."""
import copy
import hashlib
import math
import struct
import unittest

from scripts.research import compare_gpu_icp_finishes as quality


def identity():
    return [[float(i == j) for j in range(4)] for i in range(4)]


def evidence(values):
    return {"dtype": "<f8", "shape": [4, 4], "values": copy.deepcopy(values),
        "sha256": hashlib.sha256(struct.pack("<16d", *(x for row in values for x in row))).hexdigest()}


def report(mode="native"):
    pose = identity()
    result = {"transformation": pose, "fitness": .8, "inlier_rmse": .01,
        "correspondence_mapping": {"sha256": "a"*64, "dtype": "<i4", "shape": [4, 2]}}
    gates = [{"gate_index": 0, "function": "fragments._verify_bridge", "complete": True,
        "result": {"accepted": True, "camera_ids": [1, 2], "transformation": evidence(pose)}}]
    row = {"call_index": 0, "complete": True, "input_bytes_unchanged": True,
        "native_shadow": {"native": copy.deepcopy(result), "candidate": copy.deepcopy(result),
            "passed": True, "correspondence_ids_equal": True, "transform_max_abs_delta": 0., "fitness_abs_delta": 0., "rmse_abs_delta": 0.},
        "query_trace": [{"query_index": 0}],
        "loop_report": {"closed": True, "failure": None, "statistics": {"queries": 1,
            "query_rows": 5, "audited_hits": 4, "audited_misses": 1, "cpu_ambiguity_rows": 0}}}
    inventory = {"captured_after_successful_build": True, "pose_convention": "camera_to_world", "length_unit": "metres",
        "rows": [{"index": 1, "camera_to_world": evidence(pose)}]}
    inventory["pose_inventory_sha256"] = hashlib.sha256(quality.canonical(inventory["rows"]).encode()).hexdigest()
    return {"kind": "gpu-icp-whole-finish-"+mode+"-v1", "status": "passed", "mode": mode, "failure": None, "cleanup_passed": True,
        "cleanup_failures": [], "binding": {"source_sha256": quality.CURRENT}, "binding_after": {"source_sha256": quality.CURRENT},
        "scope_binding": {"checkpoint_sha256": "b"*64},
        "registration": {"complete": True, "restored": True, "failure": None, "cleanup_failures": [], "successful_builds": 1,
            "gates": gates, "graphs": [{"complete": True,
                "context": {"owner": "fragments.propose_fragment_poses", "node_fragment_indices": [0, 2]},
                "options": {"edge_prune_threshold": .25},
                "after": {"nodes": [evidence(pose), evidence(pose)],
                    "edges": [{"source_node": 0, "target_node": 1, "uncertain": False, "confidence": 1.}]}}]},
        "profile": {"finish_requested": True, "pose_seeds_used": False, "mesh_built": True,
            "input_changed_during_profile": False, "source_changed_during_profile": False, "input_sha256": "c"*64,
            "selected_indices": [0, 1], "seed": 0, "settings": {"voxel_m": .005}, "accepted_indices": [1],
            "poses": [{"index": 1, "camera_to_world": pose}], "fragment_reconnection": {"retained_edges": [[0, 1]]}},
        "final_pose_inventory": inventory, "calls": [row] if mode == "audit" else []}


class GPUFinishQualityTests(unittest.TestCase):
    def test_closed_matched_scope_and_zero_bridge_negative(self):
        native, audit = report(), report("audit")
        value = quality.quality_record(native, audit)
        self.assertTrue(all(value["checks"].values()))
        self.assertEqual(1, value["metrics"]["actual_gpu_calls"])
        audit["calls"] = []
        value = quality.quality_record(native, audit)
        self.assertEqual(0, value["metrics"]["actual_gpu_calls"])
        self.assertEqual(0, value["metrics"]["actual_gpu_query_rows"])

    def test_memberships_and_graph_endpoints_are_discrete(self):
        for field in ("witness", "edge"):
            native, audit = report(), report("audit")
            if field == "witness":
                audit["registration"]["gates"][0]["result"]["camera_ids"][0] = 3
            else:
                audit["registration"]["graphs"][0]["context"]["node_fragment_indices"][1] = 3
            value = quality.quality_record(native, audit)
            self.assertFalse(value["checks"]["same_ordered_gate_memberships"] if field == "witness" else value["checks"]["same_retained_graph"])

    def test_old_source_changed_checkpoint_and_unclosed_owner_fail(self):
        for mutate in (lambda r: r["binding"].update(source_sha256="d"*64),
                       lambda r: r["scope_binding"].update(checkpoint_sha256="e"*64),
                       lambda r: r["registration"].update(restored=False),
                       lambda r: r.update(cleanup_failures=["stream incomplete"])):
            native, audit = report(), report("audit")
            mutate(audit)
            with self.assertRaises(ValueError):
                quality.quality_record(native, audit)

    def test_incomplete_reordered_duplicate_gates_fail(self):
        for mutate in (lambda g: g[0].update(complete=False), lambda g: g[0].update(gate_index=1), lambda g: g.append(copy.deepcopy(g[0]))):
            native, audit = report(), report("audit")
            mutate(audit["registration"]["gates"])
            with self.assertRaises(ValueError):
                quality.quality_record(native, audit)

    def test_actual_shadow_numeric_and_canonical_ids_recomputed(self):
        for mutate in (lambda s: s["candidate"].update(fitness=.7),
                       lambda s: s["candidate"]["correspondence_mapping"].update(sha256="f"*64),
                       lambda s: s.update(transform_max_abs_delta=1e-16),
                       lambda s: s.update(passed=False)):
            native, audit = report(), report("audit")
            mutate(audit["calls"][0]["native_shadow"])
            with self.assertRaises(ValueError):
                quality.quality_record(native, audit)

    def test_missing_shadow_suffix_or_query_completion_fail(self):
        for mutate in (lambda r: r["calls"][0].update(call_index=1),
                       lambda r: r["calls"][0]["loop_report"].update(closed=False),
                       lambda r: r["calls"][0].update(query_trace=[]),
                       lambda r: r["calls"][0]["loop_report"]["statistics"].update(audited_misses=0)):
            native, audit = report(), report("audit")
            mutate(audit)
            with self.assertRaises(ValueError):
                quality.quality_record(native, audit)

    def test_pose_values_are_unrounded_and_full_inventory_required(self):
        native, audit = report(), report("audit")
        pose = identity(); pose[0][3] = .0006
        audit["profile"]["poses"][0]["camera_to_world"] = pose
        audit["final_pose_inventory"]["rows"][0]["camera_to_world"] = evidence(pose)
        inventory = audit["final_pose_inventory"]
        inventory["pose_inventory_sha256"] = hashlib.sha256(quality.canonical(inventory["rows"]).encode()).hexdigest()
        value = quality.quality_record(native, audit)
        self.assertFalse(value["checks"]["pose_bounds"])
        audit["final_pose_inventory"]["rows"][0]["camera_to_world"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            quality.quality_record(native, audit)

    def test_nonfinite_nonrigid_wrong_homogeneous_pose_fail(self):
        for value in (math.nan, math.inf, True):
            pose = identity(); pose[0][0] = value
            with self.assertRaises(ValueError):
                quality.matrix(pose)
        pose = identity(); pose[3][0] = 1e-20
        with self.assertRaises(ValueError):
            quality.matrix(pose)
        left, right = {0: identity()}, {0: identity()}
        right[0][0][0] = 2.
        with self.assertRaises(ValueError):
            quality.pose_deltas(left, right)

    def test_reflections_old_kinds_and_wrong_build_count_fail(self):
        left, right = {0: identity()}, {0: identity()}
        right[0][0][0] = -1.
        with self.assertRaises(ValueError):
            quality.pose_deltas(left, right)
        for mutate in (lambda r: r.update(kind="old-pair-audit"), lambda r: r["registration"].update(successful_builds=2)):
            native, audit = report(), report("audit")
            mutate(audit)
            with self.assertRaises(ValueError):
                quality.quality_record(native, audit)

    def test_only_declared_post_proposal_duration_is_excluded(self):
        native, audit = report(), report("audit")
        native["profile"]["fragment_reconnection"].update(elapsed_ms=10., budget_ms=20.)
        audit["profile"]["fragment_reconnection"].update(elapsed_ms=100., budget_ms=20.)
        self.assertTrue(quality.quality_record(native, audit)["checks"]["same_retained_graph"])
        audit["profile"]["fragment_reconnection"]["budget_ms"] = 21.
        self.assertFalse(quality.quality_record(native, audit)["checks"]["same_retained_graph"])

    def test_tiny_confidence_change_crossing_prune_threshold_is_discrete(self):
        native, audit = report(), report("audit")
        for value, confidence in ((native, .25+1e-10), (audit, .25-1e-10)):
            edge = value["registration"]["graphs"][0]["after"]["edges"][0]
            edge.update(uncertain=True, confidence=confidence)
        self.assertEqual([[0, 2, True]], quality.retained_graphs(native)[0]["retained_edges"])
        self.assertEqual([[0], [2]], quality.retained_graphs(audit)[0]["retained_components"])
        self.assertFalse(quality.quality_record(native, audit)["checks"]["same_retained_graph"])

    def test_nested_small_gate_values_verify_bytes_before_numeric_comparison(self):
        native, audit = report(), report("audit")
        for value, delta in ((native, 0.), (audit, 1e-15)):
            pose = identity(); pose[0][3] = delta
            record = evidence(pose); values = record.pop("values"); record["nbytes"] = 128
            value["registration"]["gates"][0]["result"]["transformation"] = {"array":record, "values":values}
        self.assertTrue(quality.quality_record(native, audit)["checks"]["same_ordered_gate_memberships"])
        audit["registration"]["gates"][0]["result"]["transformation"]["array"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            quality.quality_record(native, audit)

    def test_solver_consumed_constraint_ids_are_compared(self):
        native, audit = report(), report("audit")
        for value in (native, audit):
            value["registration"]["gates"][0]["consumed_problem"] = {"constraint_camera_ids": [0, 1, 2]}
        audit["registration"]["gates"][0]["consumed_problem"]["constraint_camera_ids"][2] = 1
        self.assertFalse(quality.quality_record(native, audit)["checks"]["same_ordered_gate_memberships"])


if __name__ == "__main__":
    unittest.main()

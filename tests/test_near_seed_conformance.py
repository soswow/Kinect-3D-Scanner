"""Validation-only fixed-pair native-noise contracts; no numerical imports."""
import copy
import hashlib
import json
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.research import benchmark_near_seed_conformance as driver
from tests.test_near_seed_reuse import closed_audit


def rows(x=0.):
    return [[1., 0., 0., x], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def result(x=0.):
    return {"transformation": rows(x), "pose": driver.bit_descriptor(driver.matrix_bits(rows(x))),
        "fitness": .8, "inlier_rmse": .002, "correspondence_mapping": {"dtype": "<i4", "shape": [2, 2], "nbytes": 16, "sha256": "ids"},
        "raw_correspondences": {"dtype": "<i4", "shape": [2, 2], "nbytes": 16, "sha256": "raw"}}


def call(index=0, kind="miss", seed=0., representative=0., pose=0., class_id=0):
    descriptor = {"dtype": "<f8", "shape": [2, 3], "nbytes": 48, "sha256": "cloud"}
    clouds = {role: {k: dict(descriptor) for k in ("points", "normals", "colors")} for role in ("source", "target")}
    cloudsha = hashlib.sha256(driver.scope.canonical(clouds).encode()).hexdigest()
    value = result(pose); rep_values = rows(representative); seed_values = rows(seed)
    return {"context": {"capture_index": 0, "task": 0, "pair": [0, 1], "proposal_index": 0}, "call_index": index,
        "caller": "_pair", "input_binding": dict(clouds, seed=driver.bit_descriptor(driver.matrix_bits(seed_values))),
        "seed_values": seed_values, "result": value,
        "method_receipt": {"kind": kind, "class_id": class_id, "cloud_value_sha256": cloudsha,
            "representative_seed_sha256": driver.bit_descriptor(driver.matrix_bits(rep_values))["sha256"],
            "seed_max_abs_delta": abs(seed-representative), "threshold": 1e-10, "evictions": 0},
        "representative_evidence": {"seed_values": rep_values, "seed": driver.bit_descriptor(driver.matrix_bits(rep_values)),
            "result": copy.deepcopy(value), "cloud_value_sha256": cloudsha}}


def refs(calls):
    return driver.audit_references({"pairs": [{"proposals": [{"calls": calls}]}]})


def complete(trajectory, value):
    trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], value["call_index"], value["caller"])
    trajectory.authorize(value["input_binding"])
    trajectory.finish(value)


class ConformanceTests(unittest.TestCase):
    def test_validation_only_derivation_and_old_sources_exact(self):
        self.assertTrue(driver.derivation_contract()["numerical_execution_unchanged"])

    def test_loaded_source_guard_cold_constructor_and_hot_check(self):
        guard = driver.LoadedMethodGuard()
        guard.check()

    def test_hashes_change_but_actual_unrounded_noise_is_bounded(self):
        a = call(); b = call(seed=1e-15, representative=1e-15, pose=2e-15)
        self.assertNotEqual(a["input_binding"]["seed"], b["input_binding"]["seed"])
        trajectory = driver.TimingTrajectory(refs([a])); complete(trajectory, b)
        self.assertEqual(trajectory.index, 1); self.assertIsNone(trajectory.pending)
        self.assertFalse(b["native_noise_conformance"]["native_history_bitwise_equivalent"])

    def test_current_exact_first_representative_and_no_chaining(self):
        audit = [call(), call(1, "hit", .75e-10)]
        actual = [call(seed=1e-15, representative=1e-15, pose=1e-15),
            call(1, "hit", .75e-10+1e-15, 1e-15, 1e-15)]
        trajectory = driver.TimingTrajectory(refs(audit))
        for value in actual: complete(trajectory, value)
        self.assertEqual(trajectory.index, 2)
        changed = copy.deepcopy(actual[1]); changed["representative_evidence"]["result"] = result(2e-15)
        changed["result"] = result(2e-15)
        broken = driver.TimingTrajectory(refs(audit)); complete(broken, copy.deepcopy(actual[0]))
        with self.assertRaises(driver.scope.BridgeFailure): complete(broken, changed)

    def test_seed_bound_does_not_authorize_entire_threshold_ball(self):
        trajectory = driver.TimingTrajectory(refs([call()]))
        with self.assertRaises(driver.scope.BridgeFailure): complete(trajectory, call(seed=2e-12, representative=2e-12))

    def test_inclusive_noise_boundary(self):
        trajectory = driver.TimingTrajectory(refs([call()]))
        complete(trajectory, call(seed=1e-12, representative=1e-12, pose=1e-12))

    def test_geometry_normals_colors_direction_context_and_caller_exact(self):
        original = call()
        for role, component in (("source", "points"), ("source", "normals"), ("target", "colors")):
            changed = copy.deepcopy(original); changed["input_binding"][role][component]["sha256"] = "different"
            with self.assertRaises(driver.scope.BridgeFailure): complete(driver.TimingTrajectory(refs([original])), changed)
        for field, replacement in (("caller", "_match"), ("call_index", 1), ("context", {"task": 1})):
            changed = copy.deepcopy(original); changed[field] = replacement
            with self.assertRaises(driver.scope.BridgeFailure): complete(driver.TimingTrajectory(refs([original])), changed)

    def test_actual_seed_hash_is_self_bound_before_authorization(self):
        value = call(); value["seed_values"] = rows(1e-15)
        with self.assertRaises(driver.scope.BridgeFailure): complete(driver.TimingTrajectory(refs([call()])), value)

    def test_exact_prepared_payload_and_single_authorization(self):
        value = call(); trajectory = driver.TimingTrajectory(refs([value]))
        trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], 0, value["caller"])
        altered = copy.deepcopy(value["input_binding"]); altered["seed"]["sha256"] = "wrong"
        with self.assertRaises(driver.scope.BridgeFailure): trajectory.authorize(altered)
        trajectory.authorize(value["input_binding"])
        with self.assertRaises(driver.scope.BridgeFailure): trajectory.authorize(value["input_binding"])
        trajectory.finish(value)

    def test_result_values_pose_hash_metrics_and_canonical_ids_checked(self):
        original = call()
        for mutation in ("pose", "metrics", "ids", "posehash", "nan"):
            changed = copy.deepcopy(original)
            if mutation == "pose": changed["result"] = result(2e-12)
            elif mutation == "metrics": changed["result"]["fitness"] += 2e-12
            elif mutation == "ids": changed["result"]["correspondence_mapping"]["sha256"] = "wrong"
            elif mutation == "posehash": changed["result"]["pose"]["sha256"] = "wrong"
            else: changed["result"]["fitness"] = float("nan")
            changed["representative_evidence"]["result"] = copy.deepcopy(changed["result"])
            with self.assertRaises(driver.scope.BridgeFailure): complete(driver.TimingTrajectory(refs([original])), changed)

    def test_kind_class_eviction_threshold_are_exact(self):
        original = call()
        for name, value in (("kind", "hit"), ("class_id", 2), ("evictions", 1), ("threshold", 1e-8)):
            changed = copy.deepcopy(original); changed["method_receipt"][name] = value
            with self.assertRaises(driver.scope.BridgeFailure): complete(driver.TimingTrajectory(refs([original])), changed)

    def test_representative_sha_and_actual_membership_never_approximated(self):
        audit = [call(), call(1, "hit", .5e-10)]
        for mutate in ("hash", "delta", "chain"):
            second = copy.deepcopy(audit[1])
            if mutate == "hash": second["method_receipt"]["representative_seed_sha256"] = "wrong"
            elif mutate == "delta": second["method_receipt"]["seed_max_abs_delta"] = 0.
            else:
                second = call(1, "hit", .5e-10, 1e-15)
            trajectory = driver.TimingTrajectory(refs(audit)); complete(trajectory, copy.deepcopy(audit[0]))
            with self.assertRaises(driver.scope.BridgeFailure): complete(trajectory, second)

    def test_audit_cannot_forge_first_representative_or_seed_descriptor(self):
        for mutation in ("hash", "replace", "missing"):
            values = [call(), call(1, "hit", .5e-10)]
            if mutation == "hash": values[0]["input_binding"]["seed"]["sha256"] = "wrong"
            elif mutation == "replace": values[1] = call(1, "hit", .5e-10, 1e-15)
            else: values = values[1:]
            with self.assertRaises(driver.scope.BridgeFailure): refs(values)

    def test_immutable_references_and_pending_order(self):
        with self.assertRaises(driver.scope.BridgeFailure): driver.TimingTrajectory([])
        value = call(); trajectory = driver.TimingTrajectory(refs([value]))
        with self.assertRaises(driver.scope.BridgeFailure): trajectory.finish(value)
        trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], 0, value["caller"])
        with self.assertRaises(driver.scope.BridgeFailure): trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], 0, value["caller"])

    def test_v1_or_foreign_policy_cannot_mint_v2_references(self):
        with self.assertRaises(driver.scope.BridgeFailure): driver.load_audit(None, {"configuration": {"threshold": 1e-10}}, [])

    def test_policy_immutable_and_replacement_refused_before_callback(self):
        with self.assertRaises(TypeError): driver.CONFORMANCE_POLICY["actual_seed_max_abs_delta"] = 1.
        value = call(); trajectory = driver.TimingTrajectory(refs([value]))
        changed = dict(driver.CONFORMANCE_POLICY, actual_seed_max_abs_delta=1.)
        with patch.object(driver, "CONFORMANCE_POLICY", changed):
            with self.assertRaises(driver.scope.BridgeFailure): trajectory.prepare(value["input_binding"], value["seed_values"], value["context"], 0, value["caller"])
            with self.assertRaises(driver.scope.BridgeFailure): refs([value])
        self.assertIsNone(trajectory.pending)


def own_closed_audit():
    report, binding, captures = closed_audit()
    binding["conformance_policy"] = dict(driver.CONFORMANCE_POLICY)
    report["binding_after"] = copy.deepcopy(binding)
    report.update(kind=driver.KIND, conformance_policy=dict(driver.CONFORMANCE_POLICY),
        native_history_bitwise_equivalent=False, neighborhood_authority=False, performance_authority=False, whole_finish_authority=False)
    values = driver.phase_calls(report["rounds"][0]["cache"])
    for i, value in enumerate(values):
        value["seed_values"] = rows(0. if i == 0 else 1e-12)
        value["representative_evidence"] = {"seed_values": rows(), "seed": driver.bit_descriptor(driver.matrix_bits(rows())),
            "result": copy.deepcopy(value["result"]), "cloud_value_sha256": value["method_receipt"]["cloud_value_sha256"]}
    return report, binding, captures


class ClosedReceiptTests(unittest.TestCase):
    def load(self, report, binding, captures):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"proof.json"; path.write_text(json.dumps(report), encoding="utf-8")
            return driver.load_audit(path, binding, captures)

    def test_new_closed_scope_hit_shadow_and_first_representative_pass(self):
        report, binding, captures = own_closed_audit()
        digest, references, phase = self.load(report, binding, captures)
        self.assertEqual(len(digest), 64); self.assertIs(type(references), tuple); self.assertEqual(len(references), 2)
        self.assertEqual(references, driver.audit_references(phase))

    def test_old_kind_or_contradictory_authority_rejected(self):
        for key, value in (("kind", "complete-original-proposals-near-seed-cpu-method-audit-v1"),
            ("native_history_bitwise_equivalent", True), ("neighborhood_authority", True),
            ("performance_authority", True), ("whole_finish_authority", True)):
            report, binding, captures = own_closed_audit(); report[key] = value
            with self.assertRaises(driver.scope.BridgeFailure): self.load(report, binding, captures)

    def test_fresh_hit_shadow_full_gates_topseeds_and_cleanup_stay_required(self):
        for mutate in ("shadow", "gates", "topseed", "cleanup"):
            report, binding, captures = own_closed_audit(); phase = report["rounds"][0]["cache"]
            if mutate == "shadow": driver.phase_calls(phase)[1]["method_receipt"]["shadow"]["collected"] = False
            elif mutate == "gates": phase["pairs"][0]["proposals"][1]["gates"][0]["result"] = False
            elif mutate == "topseed": phase["pairs"][0]["original_seed_descriptors"][0]["sha256"] = "wrong"
            else: phase["cleanup_passed"] = False
            with self.assertRaises(driver.scope.BridgeFailure): self.load(report, binding, captures)


if __name__ == "__main__": unittest.main()

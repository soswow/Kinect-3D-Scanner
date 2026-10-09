"""Original-call observation and nontransitive census contracts; stdlib only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import benchmark_icp_seed_reuse_census as census


def matrix(x=0.):
    return [[1., 0., 0., x], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def descriptor(values):
    bits = struct.pack("<16d", *(v for row in values for v in row))
    return {"dtype": "<f8", "shape": [4, 4], "nbytes": 128, "sha256": hashlib.sha256(bits).hexdigest()}


def row(x=0., bucket="a", result_x=0., mapping="same"):
    seed = matrix(x)
    cloud = {name: {"dtype": "<f8", "shape": [2, 3], "nbytes": 48, "sha256": bucket+name}
        for name in ("points", "normals", "colors")}
    return {"complete": True, "input_bytes_unchanged": True, "seed_values": seed,
        "input_binding": {"source": copy.deepcopy(cloud), "target": copy.deepcopy(cloud), "seed": descriptor(seed)},
        "result": {"transformation": matrix(result_x), "fitness": .8, "inlier_rmse": .002,
            "correspondence_mapping": {"sha256": mapping}, "raw_correspondences": {"sha256": "raw"}}}


class ClusteringTests(unittest.TestCase):
    def test_exact_bytes_and_all16_entries(self):
        rows = [row(), row(), row()]
        rows[2]["seed_values"][2][1] = 1e-15
        rows[2]["input_binding"]["seed"] = descriptor(rows[2]["seed_values"])
        exact = census.cluster_rows(rows, 0.)
        self.assertEqual(exact["greedy_classes"], 2)
        near = census.cluster_rows(rows, 1e-12)
        self.assertEqual(near["greedy_classes"], 1)
        self.assertEqual(near["groups"][0]["members"], [0, 1, 2])

    def test_first_representative_does_not_form_transitive_chain(self):
        result = census.cluster_rows([row(), row(.75e-12), row(1.5e-12)], 1e-12)
        self.assertEqual(result["greedy_classes"], 2)
        self.assertEqual(result["groups"][0]["members"], [0, 1])
        self.assertFalse(result["transitive_clustering"])

    def test_first_matching_representative_wins_in_original_order(self):
        result = census.cluster_rows([row(), row(1.75e-12), row(.9e-12)], 1e-12)
        self.assertEqual(result["groups"][0]["members"], [0, 2])

    def test_directed_cloud_buckets_include_every_points_normals_colors_descriptor(self):
        for role in ("source", "target"):
            for field in ("points", "normals", "colors"):
                for metadata in ("sha256", "dtype", "shape", "nbytes"):
                    with self.subTest(role=role, field=field, metadata=metadata):
                        a, b = row(), row()
                        b["input_binding"][role][field][metadata] = "changed"
                        self.assertEqual(census.cluster_rows([a, b], 1e-6)["greedy_classes"], 2)
        a, b = row(), row()
        a["input_binding"]["source"]["points"]["sha256"] = "source"
        a["input_binding"]["target"]["points"]["sha256"] = "target"
        b["input_binding"]["source"] = copy.deepcopy(a["input_binding"]["target"])
        b["input_binding"]["target"] = copy.deepcopy(a["input_binding"]["source"])
        self.assertEqual(census.cluster_rows([a, b], 1e-6)["greedy_classes"], 2)

    def test_signed_zero_is_different_exact_bytes_but_near_numeric_match(self):
        a, b = row(), row(-0.)
        self.assertNotEqual(a["input_binding"]["seed"], b["input_binding"]["seed"])
        self.assertEqual(census.cluster_rows([a, b], 0.)["greedy_classes"], 2)
        self.assertEqual(census.cluster_rows([a, b], 1e-12)["greedy_classes"], 1)

    def test_threshold_is_inclusive_and_genuine_value_above_bound_refused(self):
        result = census.cluster_rows([row(), row(1e-12), row(1.0000000000000002e-12)], 1e-12)
        self.assertEqual(result["greedy_classes"], 2)

    def test_returned_result_changes_are_diagnostics_not_seed_exclusion(self):
        a, b = row(), row(1e-15, result_x=1e-5, mapping="different")
        b["result"]["fitness"] += .1; b["result"]["inlier_rmse"] += .001
        result = census.cluster_rows([a, b], 1e-12)
        self.assertEqual(result["entries_after_first_representative"], 1)
        self.assertEqual(result["canonical_id_changed_comparisons"], 1)
        self.assertEqual(result["max_result_deltas"]["transform_max_abs_delta"], 1e-5)
        self.assertGreater(result["max_result_deltas"]["fitness_abs_delta"], .09)
        self.assertFalse(result["safe_reuse_established"])
        self.assertFalse(result["speed_authority"])
        self.assertEqual(result["skipped_calls"], 0)

    def test_raw_pair_order_is_separate_from_canonical_id_membership(self):
        a, b = row(), row()
        b["result"]["raw_correspondences"] = {"sha256": "other order"}
        delta = census.result_delta(a["result"], b["result"])
        self.assertTrue(delta["canonical_ids_equal"]); self.assertFalse(delta["raw_pair_order_equal"])

    def test_partial_or_mutated_calls_are_refused(self):
        for field in ("complete", "input_bytes_unchanged"):
            value = row(); value[field] = False
            with self.assertRaises(census.scope.BridgeFailure): census.cluster_rows([value], 1e-12)

    def test_invalid_threshold_shape_and_nonfinite_values_refused(self):
        for threshold in (0, True, -1., 1e-9, float("nan")):
            with self.assertRaises(census.scope.BridgeFailure): census.cluster_rows([row()], threshold)
        for matrix_value in ([[1.]], matrix(float("nan")), matrix(float("inf")), [[1]*4 for _ in range(4)]):
            with self.assertRaises(census.scope.BridgeFailure): census.matrix_values(matrix_value)

    def test_clustering_is_read_only_and_json_lossless_for_signed_zero(self):
        rows = [row(), row(-0.)]; before = copy.deepcopy(rows)
        value = census.cluster_rows(rows, 1e-12)
        self.assertEqual(rows, before)
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.assertEqual(struct.pack("<d", rows[1]["seed_values"][0][3]), struct.pack("<d", -0.))


class ObservationTests(unittest.TestCase):
    def setup_observer(self):
        self.source = SimpleNamespace(points=[1, 2]); self.target = SimpleNamespace(points=[3, 4])
        self.seed = matrix(); self.result = object(); self.invocations = []
        def match(source, target, seed):
            self.invocations.append((source, target, seed)); return self.result
        self.guard = SimpleNamespace(check=lambda: None, cpu_match=match)
        self.row = {"calls": []}
        self.np = SimpleNamespace(asarray=lambda value: SimpleNamespace(tolist=lambda: copy.deepcopy(value)))
        self.observer = census.CpuCallObserver(self.np, self.guard, self.row, {"proposal": 1}, lambda: None)
        payload = row()["input_binding"]
        patches = (patch.object(census.scope, "input_binding", return_value=payload),
            patch.object(census.scope, "result_evidence", return_value=row()["result"]))
        for mock in patches: mock.start(); self.addCleanup(mock.stop)
        return self.observer

    def test_original_objects_and_exact_result_identity_returned_once(self):
        observer = self.setup_observer()
        self.assertIs(observer(self.source, self.target, self.seed), self.result)
        self.assertEqual(len(self.invocations), 1)
        for actual, original in zip(self.invocations[0], (self.source, self.target, self.seed)): self.assertIs(actual, original)
        self.assertTrue(self.row["calls"][0]["complete"])

    def test_original_error_is_latched_and_never_retried(self):
        observer = self.setup_observer(); failure = RuntimeError("original CPU error")
        def broken(*args): self.invocations.append(args); raise failure
        self.guard.cpu_match = broken
        with self.assertRaises(census.scope.BridgeFailure) as caught: observer(self.source, self.target, self.seed)
        self.assertIs(caught.exception.__cause__, failure)
        with self.assertRaises(census.scope.BridgeFailure): observer(self.source, self.target, self.seed)
        self.assertEqual(len(self.invocations), 1); self.assertFalse(self.row["calls"][0]["complete"])

    def test_recording_failure_after_original_match_is_not_swallowed(self):
        observer = self.setup_observer()
        with patch.object(census.scope, "result_evidence", side_effect=ValueError("evidence")):
            with self.assertRaises(census.scope.BridgeFailure): observer(self.source, self.target, self.seed)
        self.assertEqual(len(self.invocations), 1); self.assertFalse(self.row["calls"][0]["complete"])
        self.assertIsNotNone(observer.failure)

    def test_input_replacement_before_after_is_hard_failure(self):
        observer = self.setup_observer(); before = row()["input_binding"]; after = copy.deepcopy(before)
        after["target"]["normals"]["sha256"] = "changed"
        with patch.object(census.scope, "input_binding", side_effect=(before, after)):
            with self.assertRaises(census.scope.BridgeFailure): observer(self.source, self.target, self.seed)
        self.assertFalse(self.row["calls"][0]["complete"])

    def test_route_failure_stops_before_original_math_and_latches(self):
        observer = self.setup_observer(); observer.route = lambda: (_ for _ in ()).throw(ValueError("GPU enabled"))
        with self.assertRaises(census.scope.BridgeFailure): observer(self.source, self.target, self.seed)
        self.assertFalse(self.invocations); self.assertIsNotNone(observer.failure)


class ImportTests(unittest.TestCase):
    def test_original_cpu_route_context_policy_loaded_function_changes_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"route.py"
            path.write_text("from contextvars import ContextVar\n_device=ContextVar('fake_route',default=None)\ndef match(source,target,initial):\n return None\n", encoding="utf-8")
            module = ModuleType("census_test_route"); module.__file__ = str(path)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True), module.__dict__)
            with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"KINECT_CUDA_REGISTRATION": "cpu"}):
                guard = census.OriginalCpuRoute(module); guard.check(source=True)
                token = module._device.set("CUDA:0")
                try:
                    with self.assertRaises(census.scope.BridgeFailure): guard.check()
                finally: module._device.reset(token)
                with patch.dict(os.environ, {"KINECT_CUDA_REGISTRATION": "tensor"}):
                    with self.assertRaises(census.scope.BridgeFailure): guard.check()
                with patch.object(module, "match", lambda *args: None):
                    with self.assertRaises(census.scope.BridgeFailure): guard.check()
                fn = module.match; previous = fn.__defaults__
                try:
                    fn.__defaults__ = ("changed",)
                    with self.assertRaises(census.scope.BridgeFailure): guard.check()
                finally: fn.__defaults__ = previous
                path.write_text(path.read_text(encoding="utf-8")+"# changed\n", encoding="utf-8")
                with self.assertRaises(census.scope.BridgeFailure): guard.check(source=True)

    def test_loaded_observer_cluster_code_defaults_and_alias_mutation_refused(self):
        guard = census.LoadedCensusGuard(); guard.check(source=True)
        method = census.CpuCallObserver.__call__; code = method.__code__
        try:
            method.__code__ = (lambda self, *args: None).__code__
            with self.assertRaises(census.scope.BridgeFailure): guard.check()
        finally: method.__code__ = code
        function = census.cluster_rows; defaults = function.__defaults__
        try:
            function.__defaults__ = ("changed",)
            with self.assertRaises(census.scope.BridgeFailure): guard.check()
        finally: function.__defaults__ = defaults
        with patch.object(census, "CpuCallObserver", object):
            with self.assertRaises(census.scope.BridgeFailure): guard.check()
        with patch.object(census, "THRESHOLDS", (0., 1e-3)):
            with self.assertRaises(census.scope.BridgeFailure): guard.check()
        guard.check()

    def test_fresh_import_and_othercwd_help_no_numerical_modules(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            code = "import sys;sys.path.insert(0,"+repr(str(census.ROOT))+ ");import scripts.research.benchmark_icp_seed_reuse_census;assert not any(k in sys.modules for k in ('numpy','cupy','open3d','cv2'))"
            value = subprocess.run([sys.executable, "-S", "-c", code], cwd=folder, capture_output=True, text=True)
            self.assertEqual(value.returncode, 0, value.stderr)
            help_result = subprocess.run([sys.executable, "-S", str(census.ROOT/census.OWN_FILES[0]), "--help"], cwd=folder, capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--captures", help_result.stdout)


if __name__ == "__main__": unittest.main()

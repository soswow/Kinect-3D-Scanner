"""Stdlib supervisor restoration/physical prerequisite/failure contracts."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import profile_final_missing_activation as producer
from scripts.research.final_missing_activation import MissingKeyFinalScope, KIND as allocator_kind
from scripts.research.final_missing_activation import WeightedActivationSourceContract, MissingActivationContractError
from scripts.research.final_allocation_scope import ObservedOriginalFinalScope
from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS
from tests.final_source_fixture import baseline_sources


class Harness:
    def __init__(self, args, metrics):
        self.args, self.metrics = args, metrics
        self.KIND = "original-kind"
        self.EXPERIMENT_ARTIFACTS = (*FINISH_ARTIFACTS, "scripts/research/final_allocation_scope.py",
                                    "scripts/research/profile_final_allocation.py")
        self.RightSizedFinalScope, self.ObservedOriginalFinalScope = object(), ObservedOriginalFinalScope
        self.original_allocator = self.RightSizedFinalScope
        self.worker_requests, self.armed, self.fail_restore = 1, False, False
        self.failure = self.modify = None

    def __setattr__(self, name, value):
        if (name == "RightSizedFinalScope" and getattr(self, "armed", False)
                and getattr(self, "fail_restore", False) and value is self.original_allocator):
            raise RuntimeError("Injected hook restoration failure")
        object.__setattr__(self, name, value)

    def main(self):
        self.received = list(sys.argv)
        chosen = self.RightSizedFinalScope if self.args.mode == "exact-missing-key" else self.ObservedOriginalFinalScope
        engine = SimpleNamespace(settings=SimpleNamespace(confidence_fusion=True), device="CUDA:0")
        scope = chosen(engine, original_final=lambda *args: None)
        assert isinstance(scope, MissingKeyFinalScope if self.args.mode == "exact-missing-key" else ObservedOriginalFinalScope)
        if self.failure is not None:
            raise self.failure
        pins = {name: producer.sha(producer.ROOT/name) for name in self.EXPERIMENT_ARTIFACTS}
        allocation = {"kind": allocator_kind if self.args.mode == "exact-missing-key" else "offline-exact-final-allocation-v1",
            "restored": True, "cleanup_failures": [], "weighted_only": True, "native_grid_returned": True,
            "final_key_proofs": [{"exact_key_union": True, "initial_capacity": 3328, "final_capacity": 3328}],
            "preparation_observations": [{"data_owners_and_configuration_unchanged": True}],
            "activation": [{"fault": None, "calls": [{"complete": True}]}]}
        runtime = {"versions": {"numpy": "bound", "open3d": "bound", "cupy": "bound"}, "gpu": ["bound"],
            "cuda_driver_version": 12040, "cuda_runtime_version": 12040,
            "numpy_core_binary_sha256": "bound", "open3d_backend_binary_sha256": "bound"}
        data = {"kind": self.KIND, "status": "complete", "artifacts_sha256": pins, "artifacts_sha256_after": pins,
                "allocation": allocation, "runtime_binding": runtime, "runtime_binding_after": copy.deepcopy(runtime),
                "outcome": "expected-early-capacity-failure" if "--expect-capacity-failure" in sys.argv else "mesh-built-quality-unproven"}
        if self.modify is not None:
            self.modify(data)
        self.args.output.write_text("{}", encoding="utf-8")
        self.args.output.with_suffix(".allocation.json").write_text(json.dumps(data), encoding="utf-8")
        for _ in range(self.worker_requests):
            self.metrics.finish_cuda_worker()
        self.armed = True


class SupervisorContracts(unittest.TestCase):
    def setUp(self):
        # Harness is explicitly artificial. Restore its reviewed source seam so
        # faults exercise the supervisor rather than an unrelated current-core
        # refusal. This read-only fixture cannot authorize numerical execution.
        source_fixture = baseline_sources()
        source_fixture.__enter__()
        self.addCleanup(source_fixture.__exit__, None, None, None)
        self.fake_proof = {"path": "closed-physical.json", "sha256": "bound",
            "runtime_binding": {"versions": {"numpy": "bound", "open3d": "bound", "cupy": "bound"}, "gpu": ["bound"],
                "binaries": {"fixture-library": {"sha256": "bound"}},
                "cuda": {"driver_version": 12040, "runtime_version": 12040}}}
        self.proof_guard = patch.object(producer, "activation_proof", return_value=self.fake_proof)
        self.proof_guard.start()
        self.addCleanup(self.proof_guard.stop)

    def fixture(self, folder, mode="exact-missing-key"):
        args = SimpleNamespace(mode=mode, output=Path(folder)/"report.json", activation_proof=Path(folder)/"physical.json")
        metrics = SimpleNamespace(finish_cuda_worker=None)
        base = Harness(args, metrics)
        calls, saved_argv = [], sys.argv
        def worker():
            closed = json.loads(args.output.with_suffix(".missing-activation.json").read_text(encoding="utf-8"))
            self.assertEqual(closed["status"], "complete")
            self.assertTrue(closed["supervisor_restored"])
            self.assertEqual(base.KIND, "original-kind")
            self.assertIs(sys.argv, saved_argv)
            self.assertIs(metrics.finish_cuda_worker, worker)
            calls.append(True)
        metrics.finish_cuda_worker = worker
        return args, base, metrics, calls

    def test_both_modes_forward_unchanged_inputs_and_defer_worker_until_closure(self):
        for mode in ("native-original", "exact-missing-key"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                args, base, metrics, calls = self.fixture(folder, mode)
                forwarded = ["raw.zip", "--checkpoint", "state.json", "--final-block-count", "20000"]
                producer.run(args, forwarded, base=base, metrics=metrics)
                self.assertEqual(base.received[-len(forwarded):], forwarded)
                expected = "exact-rightsized" if mode == "exact-missing-key" else mode
                self.assertEqual(base.received[base.received.index("--mode")+1], expected)
                self.assertEqual(calls, [True])
                envelope = json.loads(args.output.with_suffix(".missing-activation.json").read_text(encoding="utf-8"))
                self.assertFalse(envelope["geometry_quality_proven"])
                self.assertFalse(envelope["performance_authority"])
                self.assertEqual(len(envelope["artifacts_sha256"]), 21)
                self.assertEqual(envelope["profile"], envelope["profile_after"])

    def test_logical_early_failure_remains_inherited_typed_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            args, base, metrics, calls = self.fixture(folder)
            producer.run(args, ["raw.zip", "--expect-capacity-failure"], base=base, metrics=metrics)
            envelope = json.loads(args.output.with_suffix(".missing-activation.json").read_text(encoding="utf-8"))
            self.assertEqual(envelope["outcome"], "expected-early-capacity-failure")
            self.assertEqual(calls, [True])

    def test_each_incomplete_scope_or_runtime_rejected_before_terminal(self):
        mutations = [lambda d: d.update(status="failed"),
            lambda d: d["allocation"].update(preparation_observations=[]),
            lambda d: d["allocation"].update(activation=[]),
            lambda d: d["allocation"]["activation"][0].update(fault={"message": "partial failure"}),
            lambda d: d["allocation"]["final_key_proofs"][0].update(final_capacity=6656),
            lambda d: d["runtime_binding"]["versions"].update(open3d="different"),
            lambda d: d["runtime_binding"].update(cuda_driver_version=99999),
            lambda d: d["runtime_binding"].update(open3d_backend_binary_sha256="changed")]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as folder:
                args, base, metrics, calls = self.fixture(folder)
                base.modify = mutate
                with self.assertRaises(RuntimeError):
                    producer.run(args, [], base=base, metrics=metrics)
                self.assertEqual(calls, [])
                self.assertEqual(json.loads(args.output.with_suffix(".missing-activation.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_missing_duplicate_terminal_requests_never_terminate(self):
        for number in (0, 2):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as folder:
                args, base, metrics, calls = self.fixture(folder)
                base.worker_requests = number
                with self.assertRaises(RuntimeError):
                    producer.run(args, [], base=base, metrics=metrics)
                self.assertEqual(calls, [])

    def test_original_failure_restores_all_and_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            args, base, metrics, calls = self.fixture(folder)
            primary = RuntimeError("Original native failure")
            base.failure = primary
            saved = (base.KIND, base.EXPERIMENT_ARTIFACTS, base.RightSizedFinalScope, base.ObservedOriginalFinalScope, sys.argv)
            with self.assertRaises(RuntimeError) as caught:
                producer.run(args, [], base=base, metrics=metrics)
            self.assertIs(caught.exception, primary)
            self.assertEqual((base.KIND, base.EXPERIMENT_ARTIFACTS, base.RightSizedFinalScope, base.ObservedOriginalFinalScope, sys.argv), saved)
            self.assertEqual(calls, [])

    def test_restore_failure_attempts_remaining_hooks_and_argv(self):
        with tempfile.TemporaryDirectory() as folder:
            args, base, metrics, calls = self.fixture(folder)
            base.fail_restore = True
            argv, worker = sys.argv, metrics.finish_cuda_worker
            with self.assertRaises(RuntimeError):
                producer.run(args, [], base=base, metrics=metrics)
            self.assertIs(sys.argv, argv)
            self.assertIs(metrics.finish_cuda_worker, worker)
            self.assertEqual(base.KIND, "original-kind")
            self.assertEqual(calls, [])

    def test_late_report_write_failure_keeps_original_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            args, base, metrics, calls = self.fixture(folder)
            primary = RuntimeError("Original Finish failed")
            base.failure = primary
            write, count = Path.write_text, [0]
            def injected(path, *values, **kwargs):
                if path == args.output.with_suffix(".missing-activation.json"):
                    count[0] += 1
                    if count[0] > 1:
                        raise OSError("Final write failed")
                return write(path, *values, **kwargs)
            with patch.object(Path, "write_text", injected), self.assertRaises(RuntimeError) as caught:
                producer.run(args, [], base=base, metrics=metrics)
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertEqual(calls, [])

    def test_existing_report_preserved_before_any_hook_or_proof_read(self):
        with tempfile.TemporaryDirectory() as folder:
            args, base, metrics, calls = self.fixture(folder)
            args.output.write_bytes(b"preserved")
            with self.assertRaises(RuntimeError):
                producer.run(args, [], base=base, metrics=metrics)
            self.assertEqual(args.output.read_bytes(), b"preserved")
            self.assertEqual(base.KIND, "original-kind")
            self.assertEqual(calls, [])

    def test_cli_help_outside_workspace_without_numerical_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-S", str(producer.ROOT/"scripts/research/profile_final_missing_activation.py"), "--help"],
                                    cwd=folder, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--activation-proof", result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])


class CurrentSourceGuardContracts(unittest.TestCase):
    def test_historical_activation_guard_rejects_real_production_callers(self):
        # No source fixture or guard mock: installed missing-key callers are a
        # different authority from the frozen historical proxy experiment.
        with self.assertRaisesRegex(MissingActivationContractError, "Original weighted activation caller changed"):
            WeightedActivationSourceContract()


class PhysicalPrerequisiteContracts(unittest.TestCase):
    """Use a complete JSON fixture plus real source/binary file hashes, no native imports."""
    def fixture(self, folder):
        from scripts.research import benchmark_missing_activation as physical
        binary = Path(folder)/"fixture-open3d.pyd"
        binary.write_bytes(b"bounded-original-native-binary-fixture")
        runtime = {"open3d_threads": 20,
            "versions": {"numpy": "fixture-numpy", "open3d": "fixture-open3d", "cupy": "fixture-cupy"},
            "environment": dict(physical.THREAD_ENV),
            "cuda": {"device": 0, "name": "fixture-selected-device", "driver_version": 12040, "runtime_version": 12040},
            "gpu": [{"name": "fixture-selected-device", "driver": "bound"}],
            "binaries": {str(binary): {"sha256": producer.sha(binary), "bytes": binary.stat().st_size}}}
        pins = {name: producer.sha(producer.ROOT/name) for name in physical.ARTIFACTS}
        pairs = []
        for backend in ("cpu-original", "cuda-tensor", "cuda-fused"):
            inputs = {"depth": {"shape": [24, 32], "dtype": "<u2", "bytes": 1536, "sha256": "0"*64}}
            pairs.append({"backend": backend, "complete": True, "inputs_unchanged": True,
                "inputs": inputs, "inputs_after": copy.deepcopy(inputs), "exact_key_mapped_attribute_bits": True,
                "nonzero_weight_voxels": 16,
                "comparison": {name: {"bit_mismatches": 0, "elements": 16384}
                               for name in ("tsdf", "weight", "color")},
                "original-full": {"initial_capacity": 4, "final_capacity": 8, "final_blocks": 4},
                "missing-only": {"initial_capacity": 4, "final_capacity": 4, "final_blocks": 4,
                    "key_proof": {"exact_key_union": True, "required_blocks": 4,
                        "initial_capacity": 4, "final_capacity": 4,
                        "expected_union_sha256": "1"*64, "actual_union_sha256": "1"*64}}})
        value = {"kind": physical.KIND, "status": "passed", "failure": None, "cleanup_failures": [],
            "environment_restored": True, "source_unchanged": True,
            "source_sha256": physical.FROZEN_CORE, "source_sha256_after": physical.FROZEN_CORE,
            "artifacts_sha256": pins, "artifacts_sha256_after": copy.deepcopy(pins),
            "runtime_binding": runtime, "runtime_binding_after": copy.deepcopy(runtime), "pairs": pairs}
        path = Path(folder)/"proof.json"
        return path, value, binary

    def validate(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return producer.activation_proof(path)

    def test_complete_current_source_and_actual_binary_proof_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value, _ = self.fixture(folder)
            token = self.validate(path, value)
            self.assertEqual(token["sha256"], producer.sha(path))
            self.assertEqual(token["runtime_binding"], value["runtime_binding"])

    def test_failed_partial_and_source_drift_reports_refused(self):
        changes = [lambda d: d.update(kind="weighted-final-missing-activation-causal-synthetic-v1"),
            lambda d: d.update(status="failed"), lambda d: d.update(failure={"message": "native fault"}),
            lambda d: d.update(cleanup_failures=[{"message": "sync fault"}]),
            lambda d: d.update(environment_restored=False),
            lambda d: d["artifacts_sha256_after"].update({producer.NEW_ARTIFACTS[0]: "changed"})]
        self.assert_rejected(changes)

    def test_incomplete_native_settings_and_identity_refused(self):
        def both(d, key, value):
            for field in ("runtime_binding", "runtime_binding_after"):
                d[field][key] = copy.deepcopy(value)
        changes = [lambda d: both(d, "versions", {}), lambda d: both(d, "environment", {"OMP_NUM_THREADS": "20"}),
            lambda d: both(d, "binaries", {}), lambda d: both(d, "gpu", None),
            lambda d: both(d, "cuda", {"device": 0, "name": "fixture", "driver_version": 0, "runtime_version": 12040}),
            lambda d: d["runtime_binding_after"].update(open3d_threads=8)]
        self.assert_rejected(changes)

    def test_native_file_changed_after_closed_proof_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value, binary = self.fixture(folder)
            binary.write_bytes(b"mutated-native-library")
            with self.assertRaises(RuntimeError):
                self.validate(path, value)

    def test_each_backend_and_nonvacuous_attribute_coverage_required(self):
        changes = [lambda d: d["pairs"].pop(), lambda d: d["pairs"].reverse(),
            lambda d: d["pairs"][1].update(nonzero_weight_voxels=0),
            lambda d: d["pairs"][2]["comparison"]["tsdf"].update(elements=0),
            lambda d: d["pairs"][0]["comparison"]["color"].update(bit_mismatches=1)]
        self.assert_rejected(changes)

    def test_inputs_and_capacity_key_union_are_actual_gates(self):
        changes = [lambda d: d["pairs"][0]["inputs_after"]["depth"].update(sha256="changed"),
            lambda d: d["pairs"][1]["original-full"].update(final_capacity=4),
            lambda d: d["pairs"][2]["missing-only"].update(final_capacity=8),
            lambda d: d["pairs"][0]["missing-only"]["key_proof"].update(actual_union_sha256="2"*64),
            lambda d: d["pairs"][0]["missing-only"]["key_proof"].update(expected_union_sha256=None, actual_union_sha256=None)]
        self.assert_rejected(changes)

    def assert_rejected(self, changes):
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path, value, _ = self.fixture(folder)
                change(value)
                with self.assertRaises(RuntimeError):
                    self.validate(path, value)


if __name__ == "__main__":
    unittest.main()

"""Source, accounting and failure contracts for the unobserved Finish control."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research import profile_gpu_icp_finish_control as control


class Clock:
    def __init__(self): self.value = 0
    def __call__(self): return self.value
    def add(self, value): self.value += value


def engine():
    return SimpleNamespace(unprocessed_count=0,
        settings=SimpleNamespace(confidence_fusion=True, final_voxel_m=.005, final_block_count=10000),
        mesh=object(), _final_vbg=object(), final_reconstruction={
            "applied": True, "voxel_m": .005, "allocation_strategy": "exact missing-key activation",
            "allocated_blocks": 3328, "initial_block_capacity": 3328,
            "requested_block_capacity": 3328, "required_blocks": 3328, "blocks": 3328})


class ControlContracts(unittest.TestCase):
    def call(self, owner=None, build=None, **kwargs):
        def forbidden(*args, **options): raise AssertionError("scope/permit must remain uncalled")
        return control.original_build_control(owner or engine(), mode="native",
            scope_factory=forbidden, protocol=SimpleNamespace(validate_finish_audit=forbidden),
            original_build=build or (lambda value, **options: (True, {"success": True})),
            binding={"source_sha256": control.original.CURRENT}, scope_binding={"checkpoint": "same"},
            completion=kwargs.pop("completion", lambda: None), **kwargs)

    def test_exact_one_call_derivation_and_original_module_unchanged(self):
        held_main = control.original.main
        held_finish = control.original.execute_finish
        changed = control.derivation_contract()
        record = control.derivation_contract(changed)
        self.assertEqual(record["inverse_final_call_count"], 1)
        derived = control.derive_main()
        self.assertEqual(derived.__globals__["__name__"], control.original.__name__)
        self.assertIs(control.original.main, held_main)
        self.assertIs(control.original.execute_finish, held_finish)
        self.assertIs(derived.__globals__["CHECKPOINT_FILES"], control.original.CHECKPOINT_FILES)
        self.assertIs(derived.__globals__["FAMILY_FILES"], control.original.FAMILY_FILES)

    def test_derivation_refuses_other_body_change(self):
        changed = control.derivation_contract()
        changed.body.insert(0, ast.Pass())
        with self.assertRaisesRegex(control.original.FinishProfileFailure, "body changed"):
            control.derivation_contract(changed)

    def test_late_derived_code_and_global_mutation_saved_as_failed(self):
        for damage in ("code", "globals", "defaults"):
            derived = control.derive_main()
            namespace = derived.__globals__
            save = namespace["save_report"]
            if damage == "code": derived.__code__ = (lambda argv=None: None).__code__
            elif damage == "globals": namespace["KINDS"] = {"native": "foreign"}
            else: derived.__defaults__ = ("foreign",)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"failed.json"
                with self.assertRaisesRegex(control.original.FinishProfileFailure, "Derived control"):
                    save(path, {"status": "passed", "failure": None})
                self.assertIn('"status": "failed"', path.read_text(encoding="utf-8"))

    def test_unobserved_bundle_slots_keep_original_hot_owner_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bundle.py"
            path.write_text("def propose_bundle_poses():\n    return 1\n", encoding="utf-8")
            module = ModuleType("control_fake_bundle"); module.__file__ = str(path)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True), module.__dict__)
            with patch.dict(sys.modules, {module.__name__: module}):
                guard = control.unobserved_loaded_owners((module,), (module,),
                    observed_slots=((module, "propose_bundle_poses"),))
                module.propose_bundle_poses = lambda: 2
                with self.assertRaisesRegex(control.original.FinishProfileFailure, "code/default changed"):
                    guard.check()

    def test_derivation_refuses_missing_or_multiple_substitution(self):
        changed = control.derivation_contract()
        for node in ast.walk(changed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "original_build_control":
                node.func.id = "execute_finish"
        with self.assertRaisesRegex(control.original.FinishProfileFailure, "exactly one"):
            control.derivation_contract(changed)

    def test_direct_build_once_and_completion_charged(self):
        clock = Clock(); owner = engine(); calls = []
        def build(value, **options):
            self.assertIs(value, owner); calls.append(options); clock.add(5)
            return True, {"success": True}
        result = self.call(owner, build, clock=clock, completion=lambda: clock.add(3))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["finish_s"], 8)
        self.assertTrue(result["registration"]["complete"])
        self.assertFalse(result["registration"]["scope_installed"])
        self.assertEqual(result["registration"]["calls"], [])
        self.assertEqual(result["registration"]["events"], [])
        self.assertFalse(result["registration"]["quality_authority"])
        self.assertFalse(result["owned_proof"]["required"])

    def test_preflight_rejects_timing_and_audit_before_any_build(self):
        owner = engine(); work = []
        for changes in ({"mode": "timing"}, {"audit_path": Path("audit")}, {"quality_path": Path("quality")}):
            values = dict(mode="native", audit_path=None, quality_path=None); values.update(changes)
            with self.assertRaises(control.original.FinishProfileFailure):
                control.original_build_control(owner, scope_factory=None, original_build=lambda *a, **k: work.append(1),
                    protocol=None, binding={"source_sha256": control.original.CURRENT}, scope_binding={},
                    completion=lambda: work.append(2), **values)
        self.assertEqual(work, [])

    def test_failed_build_return_has_closed_failed_diagnostics(self):
        with self.assertRaises(control.original.FinishProfileFailure) as caught:
            self.call(build=lambda *args, **kw: (False, {"message": "original failure"}))
        record = caught.exception.finish_profile_record
        self.assertFalse(record["registration"]["complete"])
        self.assertEqual(record["built"][1]["message"], "original failure")

    def test_native_fault_remains_primary_if_completion_also_fails(self):
        fault = ValueError("original native build")
        def build(*args, **kw): raise fault
        def complete(): raise RuntimeError("selected completion")
        with self.assertRaises(ValueError) as caught:
            self.call(build=build, completion=complete)
        self.assertIs(caught.exception, fault)
        self.assertEqual(fault.finish_profile_record["finish_cleanup_failures"][0]["action"],
                         "control selected-device completion")

    def test_completion_fault_cannot_publish_success(self):
        def complete(): raise RuntimeError("completion")
        with self.assertRaises(control.original.FinishProfileFailure) as caught:
            self.call(completion=complete)
        self.assertFalse(caught.exception.finish_profile_record["registration"]["closed"])

    def test_final_capacity_and_commit_checked(self):
        for changed in ({"allocated_blocks": 6656}, {"applied": False}, {"blocks": 3327},
                        {"allocation_strategy": "configured native activation"}):
            owner = engine(); owner.final_reconstruction.update(changed)
            with self.assertRaises(control.original.FinishProfileFailure): self.call(owner)
        owner = engine(); owner.mesh = None
        with self.assertRaises(control.original.FinishProfileFailure): self.call(owner)

    def test_pending_live_or_wrong_source_rejected_before_work(self):
        owner = engine(); owner.unprocessed_count = 1
        with self.assertRaises(control.original.FinishProfileFailure): self.call(owner)
        with self.assertRaises(control.original.FinishProfileFailure):
            control.original_build_control(engine(), mode="native", scope_factory=None, original_build=None,
                protocol=None, binding={"source_sha256": "old"}, scope_binding={}, completion=lambda: None)

    def test_source_contract_keeps_captured_family_and_separate_own_pins(self):
        contract = control.source_contract()
        self.assertEqual(contract["checkpoint_artifact_names"], list(control.original.CHECKPOINT_FILES))
        self.assertEqual(set(contract["artifacts_sha256"]), set(control.CONTROL_FILES))
        self.assertFalse(contract["gpu_or_quality_authority"])
        self.assertFalse(contract["registration_dispatch_override"])

    def test_parser_has_no_mode_seed_or_permit_options(self):
        tree = ast.parse(Path(control.__file__).read_text(encoding="utf-8"))
        options = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
            and n.args and isinstance(n.args[0], ast.Constant)}
        self.assertFalse(options & {"--mode", "--finish-audit", "--quality-proof", "--use-pose-seeds", "--limit"})
        self.assertTrue({"--checkpoint", "--output", "--run-allocated"} <= options)

    def test_original_terminal_worker_is_retained_after_final_save(self):
        derived = control.derivation_contract()
        last = derived.body[-1]
        self.assertIsInstance(last, ast.Expr)
        self.assertEqual(last.value.func.id, "finish_cuda_worker")
        self.assertEqual(control.original.CHECKPOINT_FILES[0], "scripts/research/profile_gpu_icp_finish.py")


if __name__ == "__main__":
    unittest.main()

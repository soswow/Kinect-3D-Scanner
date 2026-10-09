"""Stdlib derivation and ownership-only v2 producer contracts."""
from __future__ import annotations
import ast
import copy
import subprocess
import sys
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import microbatch_bridge_owned_driver as driver


class OwnedDriverTests(unittest.TestCase):
    def test_exact_inverse_retains_original_numerical_and_control_ast(self):
        _, contract = driver.derive()
        self.assertTrue(contract["inverse_ast_exact"])
        self.assertTrue(contract["original_math_and_gates_unchanged"])
        self.assertFalse(contract["new_cuda_math"])
        self.assertEqual(contract["original_sha256"], driver.ORIGINAL_SHA256)

    def test_original_source_change_refused_before_execution(self):
        with patch.object(driver.original.scope, "sha", return_value="f"*64):
            with self.assertRaises(RuntimeError): driver.derive()

    def test_derived_loaded_code_and_private_aliases_are_checked(self):
        driver._check_owned_loaded()
        with patch.dict(driver._NAMESPACE, {"all_train_clouds": lambda *args: None}):
            with self.assertRaises(RuntimeError): driver._check_owned_loaded()
        with patch.dict(driver._NAMESPACE, {"SharedBridgeCache": object()}):
            with self.assertRaises(RuntimeError): driver._check_owned_loaded()

    def test_in_place_derived_method_mutation_refused(self):
        function = driver.ProposalMatcher.__call__
        code = function.__code__
        try:
            function.__code__ = (lambda self, *args: None).__code__
            with self.assertRaises(RuntimeError): driver._check_owned_loaded()
        finally: function.__code__ = code
        driver._check_owned_loaded()

    def test_derived_default_mutation_refused(self):
        function = driver.ProposalMatcher.__init__
        keywords = function.__kwdefaults__
        try:
            function.__kwdefaults__ = dict(keywords, permit=True)
            with self.assertRaises(RuntimeError): driver._check_owned_loaded()
        finally: function.__kwdefaults__ = keywords

    def test_fresh_version_and_six_file_family(self):
        self.assertNotEqual(driver.KIND, driver.original.KIND)
        self.assertNotEqual(driver.TIMING_KIND, driver.original.TIMING_KIND)
        self.assertEqual(len(set(driver.NEW_FILES)), 6)
        self.assertEqual(len(set(driver.OLD_FILES)), 8)
        self.assertEqual(driver._NAMESPACE["OWN_FILES"], driver.NEW_FILES+driver.OLD_FILES)

    def cleanup(self, *, completion_error=None, proof_error=None, primary=None):
        trees = []
        def capture(tree, *args, **kwargs):
            if isinstance(tree, ast.Module): trees.append(copy.deepcopy(tree))
            return compile(tree, *args, **kwargs)
        namespace = dict(driver.derive.__globals__, compile=capture)
        FunctionType(driver.derive.__code__, namespace)()
        run = next(n for n in trees[0].body if isinstance(n, ast.FunctionDef) and n.name == "run")
        outer = next(n for n in run.body if isinstance(n, ast.Try))
        events, proof = [], {"closed": False, "failure": None}
        workspace = SimpleNamespace(closed=False)
        def close_workspace(**kwargs):
            events.append("workspace")
            if completion_error: raise completion_error
            workspace.closed = True
        workspace.close = close_workspace
        workspace.report = lambda: {"closed": workspace.closed, "failure": None}
        shared = SimpleNamespace(close=lambda: events.append("cache"),
            report=lambda: {"closed": "cache" in events})
        def close_proof(token, **kwargs):
            events.append("proof")
            self.assertIs(kwargs["primary"], primary)
            if proof_error:
                proof["failure"] = repr(proof_error)
                raise proof_error
            proof["closed"] = True
        report = {key: True for key in ("pair_quality_passed", "input_bytes_unchanged",
            "original_seed_bytes_unchanged", "loaded_owners_unchanged")}
        context = {"primary": primary, "workspace": workspace, "shared": shared,
            "retrieval": None, "guard": None, "threads": None, "old_env": {}, "permit": object(),
            "report": report, "producer_begin": 0., "args": SimpleNamespace(mode="timing"), "save": lambda: None,
            "time": SimpleNamespace(perf_counter=lambda: 1.), "os": SimpleNamespace(environ={}),
            "dt": driver.original.dt, "protocol": SimpleNamespace(close_permit=close_proof,
                permit_report=lambda token: dict(proof))}
        tree = ast.fix_missing_locations(ast.Module(body=outer.finalbody, type_ignores=[]))
        exec(compile(tree, "owned-cleanup-contract", "exec"), context)
        return events, report

    def test_owned_close_follows_numeric_owner_completion_before_success(self):
        events, report = self.cleanup()
        self.assertEqual(events, ["workspace", "cache", "proof"])
        self.assertEqual(report["status"], "passed")

    def test_proof_close_error_or_numeric_failure_cannot_publish_success(self):
        primary = ValueError("original body failure")
        events, report = self.cleanup(proof_error=RuntimeError("end hash failure"), primary=primary)
        self.assertEqual(events, ["workspace", "cache", "proof"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cleanup_failures"][0]["action"], "owned proof read-lock closure")
        events, report = self.cleanup(completion_error=RuntimeError("stream completion"))
        self.assertEqual(events, ["workspace", "proof"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["shared_cache"]["closed"])

    def test_old_global_module_is_never_patched(self):
        before = driver.original.run
        driver.derive()
        self.assertIs(driver.original.run, before)
        self.assertIsNot(driver.run, before)

    def test_fresh_import_and_help_do_not_load_numerical_modules(self):
        code = "from scripts.research import microbatch_bridge_owned_driver; import sys; assert not {'numpy','cupy','open3d','cv2'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=driver.ROOT,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", str(driver.ROOT/"scripts/research/microbatch_bridge_owned_driver.py"), "--help"],
            cwd=driver.ROOT.parent, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("immutable audit", result.stdout)


if __name__ == "__main__": unittest.main()

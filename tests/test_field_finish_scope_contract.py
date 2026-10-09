"""Stdlib scoped dispatch/rollback contracts; no native or device work."""

from types import SimpleNamespace
import copy
import unittest
from unittest.mock import patch

from scripts.research import field_finish_conformance_scope as field
from scripts.research.profile_field_finish_conformance import owned_patches
from tests.test_finish_resident_contract import FakeSolver, fake_capture


class ScopeContracts(unittest.TestCase):
    @staticmethod
    def matrix(size, value=1.):
        return {"dtype": "<f8", "shape": [size, size], "sha256": "a" * 64,
                "values": [[value if i == j else 0. for j in range(size)] for i in range(size)]}

    def graph(self):
        return SimpleNamespace(nodes=[SimpleNamespace(pose=self.matrix(4)) for _ in range(2)],
            edges=[SimpleNamespace(source_node_id=0, target_node_id=1, uncertain=True, confidence=1.,
                transformation=self.matrix(4), information=self.matrix(6))])

    def graph_scope(self, optimizer, **keywords):
        module = SimpleNamespace(global_optimization=optimizer)
        scope, match_module, result = self.scope(graph_module=module,
            graph_context_capture=lambda: {"owner": "refinement.propose_poses", "node_raw_view_indices": [3, 7]},
            graph_copier=copy.deepcopy, **keywords)
        return scope, match_module, result, module

    def scope(self, *, mode="native", solver=None, authority=None, **keywords):
        module = SimpleNamespace(match=lambda *args: "original route")
        result = object()
        if solver is not None:
            result = solver.result
        scope = field.FieldConformanceScope(module, lambda *args: result, solver,
            mode=mode, authority=authority, capture_inputs=fake_capture,
            result_summary=lambda *args: {"test": "same original result"}, **keywords)
        return scope, module, result

    def test_original_native_path_closes_without_timing_authority(self):
        rows = []
        scope, module, result = self.scope(trace=rows.append)
        original = module.match
        with scope:
            self.assertIs(module.match("a", "target", "seed"), result)
            scope.finish()
        self.assertIs(module.match, original)
        self.assertTrue(scope.report()["field_conformance"]["policy_restored"])
        self.assertEqual(scope.event_match_count, 1)
        self.assertEqual(rows[0]["event"], "match")

    def test_dispatch_retains_cpu_shadow_validation_and_exact_input_checks(self):
        solver = FakeSolver()
        scope, module, result = self.scope(mode="audit", solver=solver,
            compare_results=lambda *args: {"passed": True, "transformation_max_abs_delta": 0.,
                "fitness_abs_delta": 0., "rmse_abs_delta": 0., "correspondence_ids_equal": True})
        with scope:
            self.assertIs(module.match("a", "target", "seed"), result)
            scope.finish()
        self.assertEqual(scope.cpu_shadow_calls, 1)
        self.assertEqual(solver.match_calls, 1)
        self.assertTrue(solver.closed)

    def test_new_token_call_check_runs_before_gpu_match(self):
        solver = FakeSolver()
        scope, module, _ = self.scope(mode="timing", solver=solver, authority=object())
        def changed(*args): raise ValueError("changed actual GPU seed")
        with patch.dict(field.FieldConformanceScope.dispatch.__globals__, field_expected_call=changed):
            with self.assertRaisesRegex(field.original.FinishResearchFailure, "changed actual GPU seed"):
                with scope:
                    module.match("a", "target", "seed")
        self.assertEqual(solver.match_calls, 0)
        self.assertIsNotNone(scope.failure)

    def test_new_terminal_guard_detects_omitted_gate_suffix(self):
        solver = FakeSolver()
        scope, module, _ = self.scope(mode="timing", solver=solver, authority=object())
        with patch.dict(field.FieldConformanceScope.dispatch.__globals__, field_expected_call=lambda *args: "checked"), \
             patch.object(field, "expected_event", return_value=None), \
             patch.object(field, "complete_calls", side_effect=ValueError("omitted audited gate")):
            with self.assertRaisesRegex(field.original.FinishResearchFailure, "omitted audited gate"):
                with scope:
                    module.match("a", "target", "seed")
                    scope.finish()
        self.assertFalse(scope.complete)

    def test_output_trace_fault_is_hard_latched(self):
        scope, module, _ = self.scope(trace=lambda row: (_ for _ in ()).throw(ValueError("trace loss")))
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "trace loss"):
            with scope:
                module.match("a", "target", "seed")
        self.assertFalse(scope.complete)

    def test_canonical_scope_enters_before_original_observers_and_restores_last(self):
        events = []
        class Canonical:
            failure = None
            def __enter__(self): events.append("canonical enter")
            def __exit__(self, *args): events.append("canonical exit")
            def report(self): return {"hooks_restored": True}
        scope, module, _ = self.scope(proposal_policy="canonical-fresh-1um", canonical_factory=Canonical)
        with scope:
            events.append("original observers")
            module.match("a", "target", "seed")
            scope.finish()
        self.assertEqual(events, ["canonical enter", "original observers", "canonical exit"])
        self.assertTrue(scope.policy_restored)

    def test_canonical_cleanup_preserves_primary_fault_when_report_also_fails(self):
        class Canonical:
            failure = None
            def __enter__(self): pass
            def __exit__(self, *args): raise ValueError("primary canonical restore")
            def report(self): raise RuntimeError("secondary reporting")
        scope, module, _ = self.scope(proposal_policy="canonical-fresh-1um", canonical_factory=Canonical)
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "primary canonical restore") as raised:
            with scope:
                module.match("a", "target", "seed")
                scope.finish()
        self.assertIn("secondary reporting", str(raised.exception.__cause__.__notes__))

    def test_canonical_partial_entry_rollback_still_attempts_reporting(self):
        events = []
        class Canonical:
            failure = None
            def __enter__(self): raise ValueError("failed canonical entry")
            def __exit__(self, *args): events.append("restore")
            def report(self): events.append("report"); return {"hooks_restored": True}
        scope, module, _ = self.scope(proposal_policy="canonical-fresh-1um", canonical_factory=Canonical)
        original = module.match
        with self.assertRaisesRegex(ValueError, "failed canonical entry"):
            scope.__enter__()
        self.assertIs(module.match, original)
        self.assertEqual(events, ["restore", "report"])

    def test_graph_observation_calls_same_optimizer_once_and_restores_identity(self):
        calls, rows = [], []
        graph, method, criteria, result = self.graph(), object(), object(), object()
        option = SimpleNamespace(max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0)
        def optimizer(*args):
            calls.append(args)
            args[0].nodes[1].pose = self.matrix(4, 2.)
            args[0].edges[0].confidence = .1
            return result
        scope, match_module, _, module = self.graph_scope(optimizer, trace=rows.append)
        with scope:
            self.assertIs(module.global_optimization(graph, method, criteria, option), result)
            match_module.match("a", "target", "seed")
            scope.finish()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (graph, method, criteria, option))
        self.assertIs(module.global_optimization, optimizer)
        row = rows[0]
        self.assertEqual(row["context"]["node_raw_view_indices"], [3, 7])
        self.assertEqual(row["before"]["nodes"][1]["values"][0][0], 1.)
        self.assertEqual(row["after"]["nodes"][1]["values"][0][0], 2.)
        self.assertEqual(row["before"]["edges"][0]["confidence"], 1.)
        self.assertEqual(row["after"]["edges"][0]["confidence"], .1)
        self.assertEqual(scope.event_graph_count, 1)

    def test_graph_partial_entry_failure_restores_registration_and_graph_hooks(self):
        optimizer = lambda *args: None
        class FailedHook:
            global_optimization = staticmethod(optimizer)
            def __setattr__(self, name, value):
                if value is not optimizer:
                    raise ValueError("graph hook installation failed")
                object.__setattr__(self, name, value)
        module = FailedHook()
        scope, match_module, _ = self.scope(graph_module=module)
        original_match = match_module.match
        with self.assertRaisesRegex(ValueError, "graph hook installation failed"):
            scope.__enter__()
        self.assertIs(match_module.match, original_match)
        self.assertIs(module.global_optimization, optimizer)
        self.assertTrue(scope.restored)

    def test_original_partial_entry_cleanup_is_not_retried_after_restore_failure(self):
        class Owner:
            def __init__(self):
                self.match = lambda *args: None
                self.previous = self.match
                self.restore_attempts = 0
            def __setattr__(self, name, value):
                if name == "match" and hasattr(self, "previous"):
                    if value is self.previous:
                        self.restore_attempts += 1
                        raise RuntimeError("original restore failure")
                    raise ValueError("original installation failure")
                object.__setattr__(self, name, value)
        owner = Owner()
        solver = FakeSolver()
        closes = []
        solver.close = lambda: closes.append(True)
        scope = field.FieldConformanceScope(owner, lambda *args: solver.result, solver,
            mode="audit", capture_inputs=fake_capture, result_summary=lambda *args: {})
        with self.assertRaisesRegex(ValueError, "original installation failure"):
            scope.__enter__()
        self.assertEqual(owner.restore_attempts, 1)
        self.assertEqual(closes, [True])
        self.assertFalse(scope.restored)

    def test_bad_graph_node_ownership_cannot_reach_optimizer(self):
        calls = []
        scope, _, _, module = self.graph_scope(lambda *args: calls.append(args))
        scope.graph_context_capture = lambda: {"owner": "refinement.propose_poses", "node_raw_view_indices": [3, 3]}
        option = SimpleNamespace(max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0)
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "ownership is malformed"):
            with scope:
                module.global_optimization(self.graph(), object(), object(), option)
        self.assertEqual(calls, [])

    def test_bad_graph_endpoint_cannot_reach_optimizer(self):
        calls = []
        scope, _, _, module = self.graph_scope(lambda *args: calls.append(args))
        graph = self.graph()
        graph.edges[0].target_node_id = 2
        option = SimpleNamespace(max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0)
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "endpoint is invalid"):
            with scope:
                module.global_optimization(graph, object(), object(), option)
        self.assertEqual(calls, [])

    def test_caught_original_optimizer_error_cannot_close_complete_scope(self):
        def optimizer(*args): raise RuntimeError("ordinary native optimizer fault")
        scope, match_module, _, module = self.graph_scope(optimizer)
        option = SimpleNamespace(max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0)
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "failed to complete"):
            with scope:
                with self.assertRaisesRegex(RuntimeError, "ordinary native optimizer fault"):
                    module.global_optimization(self.graph(), object(), object(), option)
                match_module.match("a", "target", "seed")
                scope.finish()

    def test_final_pose_inventory_preserves_original_order_and_unrounded_values(self):
        value = self.matrix(4)
        value["values"][0][3] = 1.0000000000000002
        engine = SimpleNamespace(poses=[(9, value), (2, self.matrix(4))])
        result = field.final_pose_inventory(engine, copy.deepcopy)
        self.assertEqual([row["index"] for row in result["rows"]], [9, 2])
        self.assertEqual(result["rows"][0]["camera_to_world"]["values"][0][3], 1.0000000000000002)
        self.assertEqual(value["values"][0][3], 1.0000000000000002)
        self.assertEqual(len(result["pose_inventory_sha256"]), 64)

    def test_final_pose_inventory_rejects_duplicate_or_nonfinite_values(self):
        value = self.matrix(4)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            field.final_pose_inventory(SimpleNamespace(poses=[(2, value), (2, value)]), copy.deepcopy)
        value["values"][0][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            field.final_pose_inventory(SimpleNamespace(poses=[(2, value)]), copy.deepcopy)

    def test_original_successful_build_pose_observer_returns_same_object_and_restores(self):
        calls, result = [], (True, {"same": "original"})
        class Engine:
            poses = [(4, self.matrix(4))]
            def build_mesh(self, *args, **keywords): calls.append((args, keywords)); return result
        original_build = Engine.build_mesh
        scope, match_module, _ = self.scope(final_engine_type=Engine,
            final_pose_capture=lambda engine: field.final_pose_inventory(engine, copy.deepcopy))
        with scope:
            engine = Engine()
            self.assertIs(engine.build_mesh("progress", unchanged=True), result)
            match_module.match("a", "target", "seed")
            scope.finish()
        self.assertIs(Engine.build_mesh, original_build)
        self.assertEqual(calls, [(("progress",), {"unchanged": True})])
        self.assertEqual(scope.successful_builds, 1)
        self.assertEqual(scope.final_poses["rows"][0]["index"], 4)

    def test_missing_successful_build_cannot_close_scope(self):
        class Engine:
            def build_mesh(self): return False, {"reason": "original bounded rejection"}
        scope, match_module, _ = self.scope(final_engine_type=Engine)
        with self.assertRaisesRegex(field.original.FinishResearchFailure, "successful original final pose"):
            with scope:
                match_module.match("a", "target", "seed")
                scope.finish()

    def test_timing_final_pose_guard_runs_before_build_returns(self):
        result = True, {}
        class Engine:
            def build_mesh(self): return result
        scope, _, _ = self.scope(mode="timing", solver=FakeSolver(), authority=object(),
            final_engine_type=Engine, final_pose_capture=lambda engine: {"test": "inventory"})
        with patch.object(field, "expected_final_poses", side_effect=ValueError("changed audited final poses")):
            with self.assertRaisesRegex(field.original.FinishResearchFailure, "changed audited final poses"):
                with scope:
                    Engine().build_mesh()
        self.assertIsNone(scope.final_poses)

    def test_supervisor_partial_installation_restores_every_owned_hook(self):
        events = []
        class FailingOwner:
            first = object()
            second = object()
            def __setattr__(self, name, value):
                events.append(name)
                if name == "second" and value == "new": raise ValueError("installation fault")
                object.__setattr__(self, name, value)
        owner, failures = FailingOwner(), []
        first, second = owner.first, owner.second
        with self.assertRaisesRegex(ValueError, "installation fault"):
            with owned_patches([(owner, "first", "new"), (owner, "second", "new")], failures):
                self.fail("Body should not execute")
        self.assertIs(owner.first, first)
        self.assertIs(owner.second, second)
        self.assertEqual(events[-2:], ["second", "first"])
        self.assertEqual(failures, [])

    def test_supervisor_cleanup_fault_cannot_mask_body_error(self):
        previous = object()
        class Owner:
            value = previous
            def __setattr__(self, name, value):
                if value is previous: raise RuntimeError("restore fault")
                object.__setattr__(self, name, value)
        failures = []
        with self.assertRaisesRegex(ValueError, "original body error") as raised:
            with owned_patches([(Owner(), "value", object())], failures):
                raise ValueError("original body error")
        self.assertIn("restore fault", str(raised.exception.__notes__))
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()

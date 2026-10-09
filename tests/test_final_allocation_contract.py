"""Exact Final-allocation contracts; artificial state and stdlib only.

The fake reconstruction exercises transaction boundaries, not numerical parity.
The real source/code guards are checked without importing the numerical engine.
"""

import ast
import copy
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from tests.final_source_fixture import baseline_sources
from unittest.mock import patch

from scripts.research import final_allocation_scope as allocation


@dataclass(frozen=True)
class Settings:
    voxel_m: float = .01
    final_voxel_m: float | None = .005
    final_block_count: int = 10000


class Pose:
    def __init__(self, value):
        self.values = [value]

    def copy(self):
        return copy.deepcopy(self)


class Volume:
    def __init__(self, capacity):
        self.allocated = capacity
        self.writes = 0

    def hashmap(self):
        return self

    def capacity(self):
        return self.allocated

    def size(self):
        return self.writes


def prepare_metric_depth(depth, settings):
    return depth


class Engine:
    """Instrumented artificial model of the unchanged Final call structure."""
    def __init__(self, required=3328):
        self.settings = Settings()
        self.voxel_size = .01
        self.sdf_trunc = .08
        self._fusion_block_limit = 10000
        self.required = required
        self.raw_frames = [("rgb0", "depth0"), ("rgb1", "depth1")]
        self.poses = [(0, Pose("first exact seed")), (1, Pose("second exact seed"))]
        self.vbg, self._final_vbg = Volume(10000), None
        self.mesh, self.pointcloud, self.model = object(), object(), object()
        self.helper = SimpleNamespace(calls=0)
        self.backend = {"integrations": 0, "planner_visits": 0}
        self.stage_totals_ms = {"final_reintegration": 0}
        self.events = []
        self.final_reconstruction = {"original": True}
        self.create_fault = self.plan_fault = self.prepare_fault = self.integrate_fault = None
        self.capacity_offset = 0
        self.integration_capacity_growth = 0

    def _create_vbg(self, block_count=None):
        self.events.append(("allocation", block_count, self.voxel_size, self.sdf_trunc))
        if block_count == 1 and self.plan_fault is not None:
            raise self.plan_fault
        if block_count != 1 and self.create_fault is not None:
            raise self.create_fault
        return Volume(block_count + (self.capacity_offset if block_count != 1 else 0))

    def _required_fusion_blocks(self, proposals, progress_cb=None, stage="unused"):
        self._create_vbg(block_count=1)
        for index, pose in proposals:
            depth = prepare_metric_depth(self.raw_frames[index][1], self.settings)
            self.events.append(("plan", index, depth, self.settings.voxel_m, stage))
            self.backend["planner_visits"] += 1
            # Deliberate artificial writes prove detached Python/pose state.
            pose.values.append("planner-only")
            self.raw_frames.append(("private", "private"))
            self.stage_totals_ms["planner"] = 1
        return self.required

    def _prepare_input(self, rgb, depth, settings):
        self.events.append(("prepare", rgb, depth, settings.voxel_m))
        if self.prepare_fault is not None:
            raise self.prepare_fault
        return rgb, depth

    def _integrate_vbg(self, rgb, depth, pose):
        self.events.append(("integrate", rgb, depth, self.voxel_size,
                            self.sdf_trunc, self._fusion_block_limit, pose.values[0]))
        self.helper.calls += 1
        self.backend["integrations"] += 1
        self.vbg.writes += 1
        self.vbg.allocated += self.integration_capacity_growth
        if self.integrate_fault is not None:
            raise self.integrate_fault

    def _final_volume(self, progress_cb=None):
        if self.settings.final_voxel_m is None:
            self.final_reconstruction = {"live": True}
            return self.vbg
        if self._final_vbg is not None:
            return self._final_vbg
        candidate = copy.copy(self)
        candidate.settings = replace(self.settings, voxel_m=self.settings.final_voxel_m)
        candidate.voxel_size = candidate.settings.voxel_m
        candidate._fusion_block_limit = self.settings.final_block_count
        candidate.vbg = candidate._create_vbg(block_count=candidate._fusion_block_limit)
        for index, pose in self.poses:
            rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
            candidate._integrate_vbg(rgb, depth, pose)
            if progress_cb is not None:
                progress_cb(index)
        self.stage_totals_ms["final_reintegration"] += 1
        self.final_reconstruction = {"applied": False, "block_limit": candidate._fusion_block_limit}
        return candidate.vbg


class ArtificialAllocationContracts(unittest.TestCase):
    def setUp(self):
        # Artificial bodies do not claim the actual engine's source authority.
        self.guard = patch.object(allocation, "validate_original_functions", return_value={})
        self.depth = patch.object(allocation, "private_depth_preparation", side_effect=lambda fn: fn)
        self.guard.start()
        self.depth.start()
        self.addCleanup(self.guard.stop)
        self.addCleanup(self.depth.stop)

    def scope(self, engine, **kwargs):
        return allocation.RightSizedFinalScope(engine, original_final=Engine._final_volume, **kwargs)

    def assert_uncommitted(self, engine, owners):
        self.assertIs(engine.vbg, owners[0])
        self.assertIs(engine.mesh, owners[1])
        self.assertIs(engine.pointcloud, owners[2])
        self.assertIs(engine.model, owners[3])
        self.assertIsNone(engine._final_vbg)
        self.assertEqual(engine.vbg.writes, 0)
        self.assertEqual(engine.voxel_size, .01)
        self.assertEqual(engine.sdf_trunc, .08)

    def owners(self, engine):
        return engine.vbg, engine.mesh, engine.pointcloud, engine.model

    def test_full_union_overflow_fails_before_final_allocation_and_live_mutation(self):
        engine = Engine(required=13302)
        owners = self.owners(engine)
        before = copy.deepcopy((engine.raw_frames, engine.backend, engine.stage_totals_ms, engine.events))
        reports = []
        with self.scope(engine, on_plan=reports.append) as scope:
            with self.assertRaises(allocation.FinalCapacityError) as error:
                engine._final_volume()
        self.assertIn("13302", str(error.exception))
        self.assertIn("10000", str(error.exception))
        self.assertEqual(error.exception.plan.allocated_blocks, 0)
        self.assertFalse(error.exception.plan.capacity_fits)
        self.assertEqual(scope.allocations, [])
        self.assertEqual(len(reports), 1)
        self.assertEqual((engine.raw_frames, engine.backend, engine.stage_totals_ms, engine.events), before)
        self.assertEqual(engine.helper.calls, 0)
        self.assertEqual([pose.values for _, pose in engine.poses],
                         [["first exact seed"], ["second exact seed"]])
        self.assertEqual(engine.final_reconstruction, {"original": True})
        self.assert_uncommitted(engine, owners)
        self.assertTrue(scope.restored)

    def test_fit_changes_only_candidate_capacity_and_keeps_original_integration_policy(self):
        engine = Engine()
        owners = self.owners(engine)
        callbacks = []
        with self.scope(engine) as scope:
            final = engine._final_volume(callbacks.append)
        self.assertEqual(final.capacity(), 3328)
        self.assertEqual(final.writes, 2)
        self.assertEqual(scope.allocations, [{"requested_blocks": 10000,
                                            "allocated_blocks": 3328, "actual_capacity": 3328,
                                            "final_capacity":3328}])
        self.assertEqual(engine.events[0], ("allocation", 3328, .005, .08))
        self.assertEqual([row[-1] for row in engine.events if row[0] == "prepare"], [.01, .01])
        integration = [row for row in engine.events if row[0] == "integrate"]
        self.assertEqual([(row[3], row[4], row[5]) for row in integration], [(.005, .08, 10000)] * 2)
        self.assertEqual([row[6] for row in integration], ["first exact seed", "second exact seed"])
        self.assertEqual(callbacks, [0, 1])
        self.assertEqual(engine.helper.calls, 2)
        self.assertEqual(engine.backend, {"integrations": 2, "planner_visits": 0})
        self.assertEqual(engine.stage_totals_ms, {"final_reintegration": 1})
        self.assertEqual(engine.final_reconstruction["block_limit"], 10000)
        self.assertEqual(engine.final_reconstruction["configured_block_limit"], 10000)
        self.assertEqual(engine.final_reconstruction["attribute_budget_mib"], 260)
        self.assertEqual(engine.final_reconstruction["configured_attribute_budget_mib"], 781.25)
        self.assert_uncommitted(engine, owners)

    def test_capacity_boundary_and_empty_union(self):
        for required, expected in ((10000, 10000), (0, 1)):
            with self.subTest(required=required):
                engine = Engine(required)
                with self.scope(engine) as scope:
                    final = engine._final_volume()
                self.assertEqual(final.capacity(), expected)
                self.assertTrue(scope.plans[0].capacity_fits)

    def test_malformed_policy_or_union_is_rejected_without_candidate_writes(self):
        for limit, voxel, required in ((True, .005, 1), (0, .005, 1), (50001, .005, 1),
                                      (10000, float("nan"), 1), (10000, -.005, 1),
                                      (10000, True, 1), (10000, .005, True), (10000, .005, -1)):
            with self.subTest(limit=limit, voxel=voxel, required=required):
                engine = Engine(required)
                engine.settings = replace(engine.settings, final_block_count=limit, final_voxel_m=voxel)
                with self.scope(engine):
                    with self.assertRaises(allocation.FinalAllocationContractError):
                        engine._final_volume()
                self.assertEqual(engine.events, [])
                self.assertEqual(engine.helper.calls, 0)

    def test_each_original_failure_is_preserved_and_never_commits_partial_final(self):
        for attribute, integrations in (("plan_fault", 0), ("create_fault", 0),
                                        ("prepare_fault", 0), ("integrate_fault", 1)):
            with self.subTest(attribute=attribute):
                engine = Engine()
                owners = self.owners(engine)
                primary = RuntimeError(attribute)
                setattr(engine, attribute, primary)
                with self.scope(engine) as scope:
                    with self.assertRaises(RuntimeError) as error:
                        engine._final_volume()
                self.assertIs(error.exception, primary)
                self.assertEqual(engine.helper.calls, integrations)
                self.assertEqual(engine.backend["integrations"], integrations)
                self.assertEqual(engine.final_reconstruction, {"original": True})
                self.assert_uncommitted(engine, owners)
                self.assertTrue(scope.restored)

    def test_callback_error_keeps_original_candidate_only_transaction(self):
        engine = Engine()
        primary = RuntimeError("original callback")
        def callback(_):
            raise primary
        with self.scope(engine):
            with self.assertRaises(RuntimeError) as error:
                engine._final_volume(callback)
        self.assertIs(error.exception, primary)
        self.assertEqual(engine.helper.calls, 1)
        self.assertEqual(engine.vbg.writes, 0)
        self.assertIsNone(engine._final_vbg)
        self.assertEqual(engine.final_reconstruction, {"original": True})

    def test_unexpected_allocator_capacity_is_not_reported_as_success(self):
        engine = Engine()
        engine.capacity_offset = 1
        with self.scope(engine) as scope:
            with self.assertRaisesRegex(allocation.FinalAllocationContractError, "capacity differs"):
                engine._final_volume()
        self.assertEqual(scope.allocations[0]["actual_capacity"], 3329)
        self.assertEqual(engine.helper.calls, 0)
        self.assertEqual(engine.final_reconstruction, {"original": True})

    def test_backend_growth_cannot_silently_claim_the_planned_memory_reduction(self):
        engine = Engine()
        engine.integration_capacity_growth = 1
        with self.scope(engine) as scope:
            with self.assertRaisesRegex(allocation.FinalAllocationContractError, "resized beyond"):
                engine._final_volume()
        self.assertEqual(scope.allocations[0]["final_capacity"], 3330)
        self.assertEqual(scope.report()["actual_allocated_attribute_bytes"], [3330 * 81920])
        self.assertEqual(engine.helper.calls, 2)
        self.assertEqual(engine.final_reconstruction, {"original": True})
        self.assertIsNone(engine._final_vbg)
        self.assertEqual(engine.vbg.writes, 0)

    def test_default_and_cached_final_keep_exact_original_route_without_plan(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                engine = Engine()
                if cached:
                    engine._final_vbg = Volume(12)
                    expected = engine._final_vbg
                else:
                    engine.settings = replace(engine.settings, final_voxel_m=None)
                    expected = engine.vbg
                with self.scope(engine) as scope:
                    self.assertIs(engine._final_volume(), expected)
                self.assertEqual(scope.plans, [])
                self.assertEqual(scope.allocations, [])
                self.assertEqual(engine.events, [])

    def test_scope_is_instance_only_restores_existing_binding_and_rejects_reuse(self):
        engine, other = Engine(), Engine()
        previous = engine._final_volume
        engine._final_volume = previous
        original_class_method = Engine._final_volume
        scope = self.scope(engine)
        with scope:
            self.assertIs(Engine._final_volume, original_class_method)
            self.assertIs(other._final_volume.__func__, original_class_method)
            with self.assertRaisesRegex(allocation.FinalAllocationContractError, "another engine"):
                scope.final(other)
        self.assertIs(engine._final_volume, previous)
        with self.assertRaisesRegex(allocation.FinalAllocationContractError, "cannot be reused"):
            scope.__enter__()

    def test_cleanup_failure_does_not_replace_primary(self):
        class RestoreFailureEngine(Engine):
            def __setattr__(self, name, value):
                if name == "_final_volume" and getattr(self, "fail_restore", False):
                    raise RuntimeError("restore failure")
                super().__setattr__(name, value)
        engine = RestoreFailureEngine()
        engine._final_volume = engine._final_volume
        primary = RuntimeError("body failure")
        scope = self.scope(engine)
        with self.assertRaises(RuntimeError) as error:
            with scope:
                engine.fail_restore = True
                raise primary
        self.assertIs(error.exception, primary)
        self.assertTrue(any("restore failure" in note for note in primary.__notes__))
        self.assertEqual(scope.cleanup_failures, ["restore failure"])

    def test_report_explicitly_has_no_measurement_or_geometry_authority(self):
        engine = Engine()
        with self.scope(engine) as scope:
            engine._final_volume()
        report = scope.report()
        self.assertFalse(report["performance_measured"])
        self.assertFalse(report["geometry_proven"])
        self.assertIn("Existing v1/v2", report["note"])
        self.assertEqual(report["plans"][0]["allocated_attribute_bytes"], 3328 * 81920)

    def test_observed_original_control_retains_configured_allocation_and_delegate(self):
        engine = Engine()
        with allocation.ObservedOriginalFinalScope(engine, original_final=Engine._final_volume) as scope:
            final = engine._final_volume()
        self.assertEqual(final.capacity(), 10000)
        self.assertEqual(final.writes, 2)
        self.assertEqual(scope.plans[0].required_blocks, 3328)
        self.assertEqual(scope.allocations[0]["allocated_blocks"], 10000)
        self.assertNotIn("allocator_scope", engine.final_reconstruction)
        self.assertIn("unchanged original", scope.report()["control"])

    def test_observed_overflow_never_substitutes_early_error_for_original_error(self):
        engine = Engine(required=13302)
        primary = RuntimeError("Original allocation/integration error")
        engine.integrate_fault = primary
        with allocation.ObservedOriginalFinalScope(engine, original_final=Engine._final_volume) as scope:
            with self.assertRaises(RuntimeError) as error:
                engine._final_volume()
        self.assertIs(error.exception, primary)
        self.assertFalse(scope.plans[0].capacity_fits)
        self.assertEqual(engine.events[0][1], 10000)
        self.assertEqual(engine.helper.calls, 1)


class PrivateSourceContracts(unittest.TestCase):
    def test_private_container_copy_preserves_aliases_and_opaque_read_only_owners(self):
        opaque = object()
        shared = [opaque, {"frame": 1}]
        original = {"a": shared, "b": shared, "poses": (1, 2), "set": {3, 4}}
        detached = allocation.private_containers(original)
        self.assertIs(detached["a"], detached["b"])
        self.assertIs(detached["a"][0], opaque)
        detached["a"][1]["frame"] = 2
        detached["set"].add(5)
        self.assertEqual(original["a"][1], {"frame": 1})
        self.assertEqual(original["set"], {3, 4})

    def test_private_calibration_lru_keeps_original_cache_and_globals_unchanged(self):
        namespace = {"__name__": "shared.calibration", "lru_cache": lru_cache}
        exec("@lru_cache(maxsize=8)\ndef lut(camera):\n return [camera]\n"
             "def prepare(depth, settings):\n return lut(settings.camera)[0] + depth\n", namespace)
        original = namespace["prepare"]
        cache = namespace["lut"]
        cache(3)
        before = cache.cache_info()
        private = allocation.private_depth_preparation(original)
        self.assertEqual(private(4, SimpleNamespace(camera=9)), 13)
        self.assertEqual(private(5, SimpleNamespace(camera=9)), 14)
        self.assertEqual(cache.cache_info(), before)
        self.assertIs(namespace["lut"], cache)
        private_cache = private.__globals__["lut"]
        self.assertIsNot(private_cache, cache)
        self.assertEqual((private_cache.cache_info().hits, private_cache.cache_info().misses), (1, 1))

    def test_private_native_cache_uses_only_already_loaded_extension(self):
        calls = []
        namespace = {"__name__": "shared.native", "lru_cache": lru_cache,
                     "importlib": SimpleNamespace(import_module=lambda name: calls.append(name))}
        exec("@lru_cache(maxsize=1)\ndef extension():\n return importlib.import_module('_kinect_native')\n"
             "def kernels():\n return extension()\n", namespace)
        original_cache = namespace["extension"]
        loaded = SimpleNamespace(API_VERSION=2)
        with patch.dict(sys.modules, {"_kinect_native": loaded}):
            private = allocation.private_depth_preparation(namespace["kernels"])
            self.assertIs(private(), loaded)
            self.assertIs(private(), loaded)
        self.assertEqual(calls, [])
        self.assertEqual(original_cache.cache_info().currsize, 0)
        with patch.dict(sys.modules):
            sys.modules.pop("_kinect_native", None)
            cold = allocation.private_depth_preparation(namespace["kernels"])
            with self.assertRaisesRegex(ImportError, "will not load"):
                cold()

    def test_unknown_function_or_closure_cannot_enter_private_preparation(self):
        with self.assertRaises(allocation.FinalAllocationContractError):
            allocation.private_depth_preparation(lambda x: x)
        namespace = {"__name__": "shared.calibration"}
        exec("def factory():\n value = 1\n def closure():\n  return value\n return closure\n", namespace)
        with self.assertRaises(allocation.FinalAllocationContractError):
            allocation.private_depth_preparation(namespace["factory"]())

    def actual_class_without_importing_engine(self):
        source = allocation.ROOT / "scanner_server/engine.py"
        # Compilation does not execute imports. Preserve full-module compiler
        # context: Python can place NULL-call flags differently for imported
        # globals in isolated AST fragments despite equivalent source bodies.
        compiled = compile(source.read_text(encoding="utf-8"), str(source), "exec", dont_inherit=True)
        cls = next(value for value in compiled.co_consts
                   if hasattr(value, "co_name") and value.co_name == "ScanEngine")
        namespace = {"__name__": "scanner_server.engine"}
        methods = {value.co_name: FunctionType(value, namespace, value.co_name)
                   for value in cls.co_consts
                   if hasattr(value, "co_name") and value.co_name in allocation.ORIGINAL_AST}
        return type("ScanEngine", (), methods)

    def test_real_source_and_loaded_function_code_are_bound_without_native_import(self):
        cls = self.actual_class_without_importing_engine()
        engine = cls.__new__(cls)
        try:
            current = allocation.validate_original_functions(engine, cls._final_volume)
        except allocation.FinalAllocationContractError as error:
            self.assertIn("source changed", str(error))
        else:
            self.assertEqual(current, allocation.ORIGINAL_AST)
        self.assertNotIn("scanner_server.engine", sys.modules)
        with baseline_sources():
            preserved = self.actual_class_without_importing_engine()
            self.assertEqual(allocation.validate_original_functions(preserved.__new__(preserved), preserved._final_volume), allocation.ORIGINAL_AST)
        with patch.dict(allocation.ORIGINAL_AST, {"_final_volume": "0" * 64}):
            with self.assertRaisesRegex(allocation.FinalAllocationContractError, "source changed"):
                allocation.validate_original_functions(engine, cls._final_volume)

    def test_loaded_replacement_is_rejected_despite_same_module_and_disk(self):
        with baseline_sources():
            cls = self.actual_class_without_importing_engine()
            engine = cls.__new__(cls)
            replacement = FunctionType(cls._create_vbg.__code__, {"__name__": "scanner_server.engine"}, "_final_volume")
            with self.assertRaisesRegex(allocation.FinalAllocationContractError, "Loaded Final"):
                allocation.validate_original_functions(engine, replacement)

    def test_helper_imports_under_no_site_packages_without_numerical_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            code = ("import sys,json;sys.path.insert(0," + repr(str(allocation.ROOT)) + ");"
                    "from scripts.research import final_allocation_scope;"
                    "print(json.dumps([x for x in sys.modules if x.split('.')[0] in "
                    "('numpy','open3d','cupy') or x=='scanner_server.engine']))")
            result = subprocess.run([sys.executable, "-S", "-c", code], cwd=folder,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [])
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

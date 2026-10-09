"""No numerical imports: private bridge dispatch, ownership and failure checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import microbatch_bridge_scope as scope
from scripts.research import microbatch_bridge_driver as driver


class FakeNP:
    generic = type("Generic", (), {})
    ndarray = type("Array", (), {})


@dataclass
class View:
    index: int
    train: object
    heldout: object
    features: object
    pose: object
    match_cache: dict = field(default_factory=dict)


@dataclass
class Fragment:
    index: int
    train: object
    heldout: object
    keys: list
    views: list = field(default_factory=list)
    context: list = field(default_factory=list)


class Streams:
    def __init__(self, ptr=1, failure=None):
        self.ptr, self.device_id, self.failure, self.calls = ptr, 0, failure, 0
    def synchronize(self):
        self.calls += 1
        if self.failure is not None: raise self.failure
    def __enter__(self): return self
    def __exit__(self, *a): pass


class MicrobatchBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"original.py"
        self.source = "\n".join(f"def {name}(*args, **kwargs): return True" for name in scope.HELPERS
            if name not in ("_verify_bridge", "_strong", "_disagrees"))
        self.source += "\ndef _strong(result, target, minimum=.5): return bool(result)\n"
        self.source += "def _disagrees(a, b, translation_limit=.05, angle_limit=5): return a != b\n"
        self.source += "def _match(source, target, initial): return initial\n"
        self.source += "def motion(value): return value\n"
        self.source += "def correspondences(*args): return None\n"
        self.source += "def feature_agreement(*args): return True\n"
        self.source += "def _verify_bridge(source, target, initial, camera=None, *, visual_first=False):\n"
        self.source += "    result = _match(source, target, initial)\n    return result if _strong(result, target) else None\n"
        self.path.write_text(self.source, encoding="utf-8")
        self.module = ModuleType("_bridge_test_original")
        self.module.__file__ = str(self.path)
        self.module.np = FakeNP
        self.module.MAX_MATCH_CACHE = 16
        self.module.REG = SimpleNamespace(**{n: lambda *a, **k: True for n in
            ("evaluate_registration", "get_information_matrix_from_point_clouds", "registration_icp",
             "TransformationEstimationPointToPlane", "HuberLoss", "ICPConvergenceCriteria")})
        sys.modules[self.module.__name__] = self.module
        self.addCleanup(sys.modules.pop, self.module.__name__, None)
        exec(compile(self.source, str(self.path), "exec", dont_inherit=True), self.module.__dict__)

    def test_clone_unchanged_code_private_globals_and_original_return(self):
        original = self.module._match
        returned = 17
        private = scope.PrivateBridgeScope(self.module, lambda *a: returned)
        self.assertIs(private.originals["_verify_bridge"].__code__, self.module._verify_bridge.__code__)
        self.assertIsNot(private.originals["_verify_bridge"].__globals__, self.module.__dict__)
        self.assertEqual(private.verify(1, 2, 3, None), returned)
        self.assertIs(self.module._match, original)
        self.assertEqual(self.module._verify_bridge(1, 2, 3), 3)
        self.assertEqual([r["name"] for r in private.trace], ["_verify_bridge", "_strong"])
        self.assertTrue(all(r["complete"] for r in private.trace))

    def test_two_private_jobs_do_not_patch_each_other(self):
        a = scope.PrivateBridgeScope(self.module, lambda *a: 7)
        b = scope.PrivateBridgeScope(self.module, lambda *a: 9)
        self.assertEqual((a.verify(0, 1, 1, None), b.verify(0, 1, 1, None)), (7, 9))
        self.assertIsNot(a.namespace, b.namespace)
        self.assertIsNot(a.trace, b.trace)

    def test_actual_shadow_function_cannot_be_prepatched(self):
        self.module._match = lambda *a: 17
        with self.assertRaises(scope.BridgeFailure): scope.OriginalVerifierGuard(self.module)

    def test_original_imported_function_code_cannot_be_prepatched(self):
        self.module.motion.__code__ = (lambda value: 99).__code__
        with self.assertRaises(scope.BridgeFailure): scope.OriginalVerifierGuard(self.module)

    def test_original_cpu_code_mutation_detected_after_capture(self):
        guard = scope.OriginalVerifierGuard(self.module)
        self.module._match.__code__ = (lambda *a: 7).__code__
        with self.assertRaises(scope.BridgeFailure): guard.check()

    def test_original_function_alias_mutation_detected(self):
        guard = scope.OriginalVerifierGuard(self.module)
        self.module._strong = lambda *a: True
        with self.assertRaises(scope.BridgeFailure): guard.check()

    def test_original_native_function_alias_mutation_detected(self):
        guard = scope.OriginalVerifierGuard(self.module)
        self.module.REG.registration_icp = lambda *a: True
        with self.assertRaises(scope.BridgeFailure): guard.check()

    def test_original_dependency_defaults_mutation_detected(self):
        guard = scope.OriginalVerifierGuard(self.module)
        self.module.motion.__defaults__ = (3,)
        with self.assertRaises(scope.BridgeFailure): guard.check()

    def test_original_source_mutation_detected_at_proposal_boundary(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: 7)
        self.path.write_text(self.source+"\nchanged = True\n", encoding="utf-8")
        with self.assertRaises(scope.BridgeFailure): private.verify(0, 1, 1, None)

    def test_private_namespace_mutation_detected(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: 7)
        private.namespace["_pair"] = lambda *a: None
        with self.assertRaises(scope.BridgeFailure): private.healthy()

    def test_in_place_private_gate_clone_code_mutation_detected(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: 7)
        private.originals["_strong"].__code__ = (lambda *a: True).__code__
        with self.assertRaises(scope.BridgeFailure): private.healthy()

    def test_private_observer_defaults_mutation_detected(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: 7)
        private.namespace["_strong"].__defaults__ = (True,)
        with self.assertRaises(scope.BridgeFailure): private.healthy()

    def test_match_failure_is_latched_even_if_body_swallowed_it(self):
        original = RuntimeError("GPU fault")
        def broken(*a): raise original
        private = scope.PrivateBridgeScope(self.module, broken)
        with self.assertRaises(scope.BridgeFailure) as caught: private.verify(0, 1, 1, None)
        self.assertIs(caught.exception.__cause__, original)
        with self.assertRaises(scope.BridgeFailure) as again: private.healthy()
        self.assertIs(again.exception, caught.exception)

    def test_observer_evidence_failure_is_not_ordinary_rejection(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: object())
        with self.assertRaises(scope.BridgeFailure): private.verify(0, 1, 1, None)
        self.assertIsNotNone(private.failure)
        self.assertFalse(private.trace[0]["complete"])

    def test_observer_returns_exact_original_object(self):
        private = scope.PrivateBridgeScope(self.module, lambda *a: 1)
        result = {"passed": True}
        observed = private.observe("original", lambda: result)
        self.assertIs(observed(), result)

    def test_private_views_preserve_aliases_and_numeric_owners(self):
        train, heldout, features, pose = object(), object(), object(), object()
        view = View(1, train, heldout, features, pose, {2: (features, features, object())})
        a = Fragment(0, train, heldout, [view], [view], [view])
        b = Fragment(1, train, heldout, [view])
        aa, bb = scope.private_fragments(a, b)
        self.assertIs(aa.keys[0], bb.keys[0])
        self.assertIs(aa.keys[0], aa.views[0])
        self.assertIs(aa.keys[0], aa.context[0])
        for name in ("train", "heldout", "features", "pose"):
            self.assertIs(getattr(aa.keys[0], name), getattr(view, name))
        self.assertIsNot(aa.keys[0].match_cache, view.match_cache)
        aa.keys[0].match_cache.clear()
        self.assertEqual(len(view.match_cache), 1)

    def test_proposals_exhausted_before_original_ambiguity(self):
        results = [None, {"transform": 1}, {"transform": 2}, None]
        verdict = scope.original_pair_verdict(self.module, results)
        self.assertEqual(verdict, {"proposal_count": 4, "verified_proposals": 2,
            "ambiguous": True, "accepted": False, "chosen_proposal": None})

    def test_ordered_first_verified_proposal_is_original_choice(self):
        verdict = scope.original_pair_verdict(self.module, [None, {"transform": 1}, {"transform": 1}])
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["chosen_proposal"], 1)

    def test_discrete_gates_never_use_numeric_tolerance(self):
        self.assertTrue(scope.compare_evidence(True, 1))
        self.assertTrue(scope.compare_evidence({"support": [[1, 2], [3, 4]]}, {"support": [[3, 4], [1, 2]]}))
        self.assertTrue(scope.compare_evidence({"scope": "visual"}, {"scope": "geometry"}))

    def test_original_featureless_descriptor_is_preserved_exact_none(self):
        cloud = SimpleNamespace(points=object(), normals=object(), colors=object())
        features = SimpleNamespace(pixels=object(), points=object(), descriptors=None)
        view = SimpleNamespace(index=7, train=cloud, heldout=cloud, pose=object(), features=features)
        a = SimpleNamespace(index=0, train=cloud, heldout=cloud, keys=[view])
        b = SimpleNamespace(index=1, train=cloud, heldout=cloud, keys=[])
        with patch.object(scope, "descriptor", side_effect=lambda np, value: {"identity": id(value)}):
            inventory = driver.train_inventory(FakeNP, a, b)
        self.assertIsNone(inventory["0/key0/raw7/features"]["descriptors"])

    def test_none_feature_geometry_rejected_even_when_descriptors_absent(self):
        cloud = SimpleNamespace(points=object(), normals=object(), colors=object())
        features = SimpleNamespace(pixels=None, points=object(), descriptors=None)
        view = SimpleNamespace(index=7, train=cloud, heldout=cloud, pose=object(), features=features)
        a = SimpleNamespace(index=0, train=cloud, heldout=cloud, keys=[view])
        b = SimpleNamespace(index=1, train=cloud, heldout=cloud, keys=[])
        with patch.object(scope, "descriptor", side_effect=lambda np, value: {"identity": id(value)}):
            with self.assertRaises(scope.BridgeFailure): driver.train_inventory(FakeNP, a, b)

    def test_small_numeric_and_information_limits_are_separate(self):
        self.assertFalse(scope.compare_evidence({"metric": 1.0}, {"metric": 1.0+1e-9}))
        self.assertTrue(scope.compare_evidence({"metric": 1.0}, {"metric": 1.0+1e-7}))
        self.assertFalse(scope.compare_evidence({"information": [1.0]}, {"information": [1.0+1e-6]}))
        self.assertTrue(scope.compare_evidence({"information": [1.0]}, {"information": [1.0+1e-4]}))

    def test_gate_nonfinite_evidence_cannot_pass(self):
        with self.assertRaises(scope.BridgeFailure): scope.semantic(FakeNP, float("nan"))
        self.assertTrue(scope.compare_evidence(float("inf"), float("inf")))

    def make_cache(self):
        cache = driver.SharedBridgeCache.__new__(driver.SharedBridgeCache)
        null = Streams(0)
        cache.cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda *a: Streams(99), Stream=SimpleNamespace(null=null)))
        cache.streams, cache.leases, cache.lease_streams = {}, [], {}
        cache.items, cache.normals = {}, {"owned": object()}
        cache.closed, cache.failure, cache.peak_active_streams = False, None, 0
        cache.retrieval = SimpleNamespace(close=lambda: None)
        return cache

    def lease(self, cache):
        value = {"owner": cache, "closed": False}
        cache.leases.append(value)
        return value

    def test_stream_cap_is_active_not_total_history(self):
        cache = self.make_cache()
        for i in range(12):
            lease, stream = self.lease(cache), Streams(i+1)
            cache.register_lease_stream(lease, stream)
            cache.release_lease(lease)
            self.assertEqual(stream.calls, 1)
            self.assertEqual(cache.streams, {})
        self.assertEqual(cache.peak_active_streams, 1)

    def test_more_than_four_active_streams_rejected(self):
        cache = self.make_cache()
        for i in range(4): cache.register_lease_stream(self.lease(cache), Streams(i+1))
        with self.assertRaises(scope.BridgeFailure): cache.register_lease_stream(self.lease(cache), Streams(5))

    def test_same_active_stream_multiple_leases_retained_until_last_join(self):
        cache, stream = self.make_cache(), Streams(1)
        a, b = self.lease(cache), self.lease(cache)
        cache.register_lease_stream(a, stream); cache.register_lease_stream(b, stream)
        cache.release_lease(a)
        self.assertEqual(len(cache.streams), 1)
        cache.release_lease(b)
        self.assertFalse(cache.streams)

    def test_failed_lease_completion_retains_all_owners(self):
        cache, failure = self.make_cache(), RuntimeError("async fault")
        lease = self.lease(cache)
        cache.register_lease_stream(lease, Streams(1, failure))
        with self.assertRaises(RuntimeError) as caught: cache.release_lease(lease)
        self.assertIs(caught.exception, failure)
        self.assertIs(cache.failure, failure)
        self.assertFalse(lease["closed"])
        self.assertTrue(cache.streams)
        self.assertTrue(cache.normals)

    def test_shared_close_never_releases_cache_after_failed_stream_completion(self):
        cache = self.make_cache()
        cache.streams[1] = Streams(1, RuntimeError("async fault"))
        released = []
        cache.retrieval.close = lambda: released.append(True)
        with self.assertRaises(RuntimeError): cache.close()
        self.assertFalse(released)
        self.assertFalse(cache.closed)
        self.assertTrue(cache.normals)

    def test_successful_shared_close_follows_all_selected_and_setup_streams(self):
        cache = self.make_cache()
        stream = Streams(1)
        lease = self.lease(cache)
        cache.register_lease_stream(lease, stream)
        cache.items[1] = object()
        cache.close()
        self.assertEqual(stream.calls, 1)
        self.assertTrue(cache.closed)
        self.assertTrue(lease["closed"])
        self.assertFalse(cache.items)
        self.assertFalse(cache.normals)

    def test_fresh_import_uses_no_numerical_modules(self):
        code = "from scripts.research import microbatch_bridge_driver; import sys; assert not {'numpy','cupy','open3d','cv2'} & set(sys.modules)"
        completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=scope.ROOT,
            text=True, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

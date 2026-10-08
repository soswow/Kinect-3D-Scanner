"""Pure-stdlib dispatch/latch contracts; no image/Open3D/CUDA execution."""

from types import SimpleNamespace
import unittest

from scripts.research.marker_proposal_scope import MarkerSeedScope, MarkerProposalFailure


class Provider:
    def __init__(self, seed="marker-pnp"):
        self.seed, self.proposals = seed, 0

    def features_hash(self, features):
        return dict(features)

    def array_hash(self, value):
        return value

    def proposal(self, source, target, camera):
        self.proposals += 1
        return self.seed

    def report(self):
        return {"proposal_calls": self.proposals}


class MarkerSeedContracts(unittest.TestCase):
    def setUp(self):
        self.source = SimpleNamespace(index=2, features={"value": "original-sift-a"})
        self.target = SimpleNamespace(index=1, features={"value": "original-sift-b"})
        self.provider, self.calls = Provider(), []
        self.module = SimpleNamespace(propose_transform=lambda *args: None,
            _matches=lambda *args: "original-sift-matches")

        def original(source, target, camera, settings, initial=None, *, measured_first=False):
            self.module.propose_transform(source.features, target.features, camera, "original-sift-matches")
            self.calls.append((source, target, initial, measured_first))
            return "original-reciprocal-icp-pose"

        self.module._local_match = original
        self.original = original

    def call(self):
        return self.module._local_match(self.source, self.target, "camera", "settings",
            initial="original-live-seed", measured_first=True)

    def test_marker_is_only_original_icp_initial_not_returned_pose(self):
        scope = MarkerSeedScope(self.module, self.provider)
        with scope:
            result = self.call()
        self.assertEqual(result, "original-reciprocal-icp-pose")
        self.assertEqual(self.calls, [(self.source, self.target, "marker-pnp", True)])
        self.assertEqual(scope.injected, 1)
        self.assertEqual(scope.accepted, 1)
        self.assertEqual(scope.records[0]["delegated_sift_checks"], 1)
        self.assertTrue(scope.restored)
        self.assertIs(self.module._local_match, self.original)

    def test_original_sift_proposal_prevents_marker_work(self):
        self.module.propose_transform = lambda *args: "original-sift-proposal"
        scope = MarkerSeedScope(self.module, self.provider)
        with scope:
            self.call()
        self.assertEqual(self.provider.proposals, 0)
        self.assertEqual(self.calls[0][2], "original-live-seed")
        self.assertEqual(scope.injected, 0)

    def test_absent_marker_preserves_original_initial(self):
        self.provider.seed = None
        scope = MarkerSeedScope(self.module, self.provider)
        with scope:
            self.call()
        self.assertEqual(self.calls[0][2], "original-live-seed")
        self.assertEqual(scope.absent, 1)
        self.assertEqual(scope.injected, 0)

    def test_marker_does_not_override_original_rejection(self):
        original = self.original
        self.module._local_match = lambda *args, **kwargs: (original(*args, **kwargs), None)[1]
        scope = MarkerSeedScope(self.module, self.provider)
        with scope:
            self.assertIsNone(self.call())
        self.assertEqual(scope.accepted, 0)

    def test_sift_absence_drift_is_hard_latched(self):
        results = iter((None, "changed-sift-proposal"))
        self.module.propose_transform = lambda *args: next(results)
        scope = MarkerSeedScope(self.module, self.provider)
        with self.assertRaisesRegex(MarkerProposalFailure, "SIFT absence changed"):
            with scope:
                self.call()
        self.assertTrue(scope.restored)

    def test_core_swallowed_fault_and_cleanup_mask_cannot_authorize_mesh(self):
        def invalid(*args):
            raise ValueError("malformed GPU lookup")

        self.provider.proposal = invalid
        scope = MarkerSeedScope(self.module, self.provider)
        with self.assertRaisesRegex(MarkerProposalFailure, "preflight failed"):
            with scope:
                try:
                    try:
                        self.call()
                    finally:
                        raise RuntimeError("CUDA cleanup replaced primary")
                except Exception:
                    pass
        self.assertTrue(scope.restored)
        self.assertEqual(self.calls, [])

    def test_trace_error_latched_after_valid_original_call(self):
        def trace(row):
            raise OSError("disk full")

        scope = MarkerSeedScope(self.module, self.provider, trace=trace)
        with self.assertRaisesRegex(MarkerProposalFailure, "trace failed"):
            with scope:
                self.call()
        self.assertTrue(scope.restored)

    def test_original_feature_mutation_rejected(self):
        original = self.original

        def mutating(*args, **kwargs):
            result = original(*args, **kwargs)
            self.source.features["value"] = "changed"
            return result

        self.module._local_match = mutating
        scope = MarkerSeedScope(self.module, self.provider)
        with self.assertRaisesRegex(MarkerProposalFailure, "SIFT Features changed"):
            with scope:
                self.call()

    def test_missing_original_second_absence_check_rejected(self):
        self.module._local_match = lambda *args, **kwargs: "fake-accepted-seed"
        scope = MarkerSeedScope(self.module, self.provider)
        with self.assertRaisesRegex(MarkerProposalFailure, "exactly once"):
            with scope:
                self.call()

    def test_original_gate_error_keeps_original_rejection_semantics(self):
        original = self.original

        def rejecting(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("original gate failed")

        self.module._local_match = rejecting
        scope = MarkerSeedScope(self.module, self.provider)
        with scope:
            with self.assertRaisesRegex(RuntimeError, "original gate failed"):
                self.call()
        self.assertIsNone(scope.failure)
        self.assertEqual(scope.records[0]["original_gate_exception"]["type"], "RuntimeError")
        self.assertFalse(scope.records[0]["complete"])

    def test_transactional_enter_restores_first_patch_when_second_fails(self):
        module = self.module

        class Owner:
            def __init__(self):
                object.__setattr__(self, "fail", False)
                self._local_match, self.propose_transform, self._matches = module._local_match, module.propose_transform, module._matches

            def __setattr__(self, name, value):
                if name == "propose_transform" and self.fail:
                    object.__setattr__(self, "fail", False)
                    raise OSError("cannot patch")
                object.__setattr__(self, name, value)

        owner = Owner()
        scope = MarkerSeedScope(owner, self.provider)
        owner.fail = True
        with self.assertRaisesRegex(OSError, "cannot patch"):
            scope.__enter__()
        self.assertIs(owner._local_match, self.original)
        self.assertIs(owner.propose_transform, module.propose_transform)

    def test_entry_cleanup_attempts_both_restores_even_if_first_restore_fails(self):
        module, history = self.module, []

        class Owner:
            def __init__(self):
                object.__setattr__(self, "armed", False)
                self._local_match, self.propose_transform, self._matches = module._local_match, module.propose_transform, module._matches

            def __setattr__(self, name, value):
                if self.armed:
                    history.append((name, value))
                    if name == "propose_transform" and value is not module.propose_transform:
                        raise OSError("primary installation failure")
                    if name == "_local_match" and value is module._local_match:
                        raise RuntimeError("secondary first-restore failure")
                object.__setattr__(self, name, value)

        owner = Owner()
        scope = MarkerSeedScope(owner, self.provider)
        owner.armed = True
        with self.assertRaisesRegex(OSError, "primary installation failure") as failure:
            scope.__enter__()
        self.assertEqual(history[-1], ("propose_transform", module.propose_transform))
        self.assertIs(owner.propose_transform, module.propose_transform)
        self.assertTrue(any("secondary first-restore failure" in note for note in failure.exception.__notes__))


if __name__ == "__main__":
    unittest.main()

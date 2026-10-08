"""Pure-stdlib scoped original RANSAC thread/restoration contracts."""

from types import SimpleNamespace
import unittest

from scripts.research.profile_checkpoint_resident_finish import GlobalSeedThreadScope, canonical_json_scope
from scripts.research.finish_resident_registration import FinishResearchFailure


class Utility:
    def __init__(self):
        self.threads, self.history = 20, []

    def get_max_threads(self):
        return self.threads

    def set_max_threads(self, value):
        self.history.append(value)
        self.threads = value


class RansacProtocolContracts(unittest.TestCase):
    def setUp(self):
        self.utility = Utility()
        self.source, self.target = SimpleNamespace(index=3), SimpleNamespace(index=4)
        self.received = []

        def original(source, target, seed):
            self.received.append((source, target, seed, self.utility.threads))
            return "unchanged-original-proposal"

        self.original = original
        self.module = SimpleNamespace(_global_seed=original)

    def test_only_original_global_seed_uses_requested_threads_and_restores(self):
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with scope:
            self.assertEqual(self.utility.threads, 20)
            result = self.module._global_seed(self.source, self.target, 304)
            self.assertEqual(result, "unchanged-original-proposal")
            self.assertEqual(self.utility.threads, 20)
            scope.healthy()
        self.assertEqual(self.received, [(self.source, self.target, 304, 1)])
        self.assertIs(self.module._global_seed, self.original)
        self.assertTrue(scope.restored)
        self.assertEqual(scope.report()["calls"][0]["restored_threads"], 20)

    def test_parallel_control_keeps_original_twenty_thread_call(self):
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=20)
        with scope:
            self.module._global_seed(self.source, self.target, 10304)
        self.assertEqual(self.received[0][2:], (10304, 20))
        self.assertEqual(self.utility.threads, 20)

    def test_original_rejection_unchanged(self):
        self.module._global_seed = lambda *args: None
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with scope:
            self.assertIsNone(self.module._global_seed(self.source, self.target, 304))
        self.assertFalse(scope.report()["calls"][0]["proposal_present"])

    def test_original_exception_restores_twenty_and_keeps_original_behavior(self):
        def failing(*args):
            raise RuntimeError("original RANSAC failure")

        self.module._global_seed = failing
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with scope:
            with self.assertRaisesRegex(RuntimeError, "original RANSAC failure"):
                self.module._global_seed(self.source, self.target, 304)
            self.assertEqual(self.utility.threads, 20)
        self.assertIsNone(scope.failure)
        self.assertEqual(scope.calls[0]["original_exception"]["message"], "original RANSAC failure")

    def test_thread_application_failure_hard_latches_before_original(self):
        original_set = self.utility.set_max_threads

        def setting(value):
            if value == 1:
                raise OSError("cannot set requested threads")
            original_set(value)

        self.utility.set_max_threads = setting
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with self.assertRaisesRegex(FinishResearchFailure, "setup failed"):
            with scope:
                self.module._global_seed(self.source, self.target, 304)
        self.assertEqual(self.received, [])
        self.assertEqual(self.utility.threads, 20)
        self.assertTrue(scope.restored)

    def test_restore_failure_cannot_be_swallowed_by_original_finish(self):
        original_set = self.utility.set_max_threads
        restoring = False

        def setting(value):
            nonlocal restoring
            if value == 20 and restoring:
                restoring = False
                raise OSError("one failed restore")
            original_set(value)
            if value == 1:
                restoring = True

        self.utility.set_max_threads = setting
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with self.assertRaisesRegex(FinishResearchFailure, "restoration failed"):
            with scope:
                try:
                    try:
                        self.module._global_seed(self.source, self.target, 304)
                    finally:
                        raise RuntimeError("ordinary cleanup masked primary")
                except Exception:
                    pass
        self.assertTrue(scope.restored)
        self.assertEqual(self.utility.threads, 20)

    def test_original_primary_and_restore_secondary_both_preserved(self):
        def original(*args):
            raise ValueError("primary proposal failure")

        self.module._global_seed = original
        setter = self.utility.set_max_threads
        failures = iter((True, False))

        def setting(value):
            if value == 20 and next(failures):
                raise OSError("secondary restore failure")
            setter(value)

        self.utility.set_max_threads = setting
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with self.assertRaises(FinishResearchFailure):
            with scope:
                self.module._global_seed(self.source, self.target, 304)
        self.assertEqual(scope.report()["failure"]["cause"]["message"], "primary proposal failure")
        self.assertEqual(scope.calls[0]["thread_restore_exception"]["message"], "secondary restore failure")

    def test_wrong_existing_thread_state_rejected_before_patch(self):
        self.utility.threads = 8
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with self.assertRaisesRegex(FinishResearchFailure, "twenty-thread"):
            scope.__enter__()
        self.assertIs(self.module._global_seed, self.original)
        self.assertEqual(self.received, [])

    def test_thread_drift_inside_scope_is_fatal_and_restored(self):
        scope = GlobalSeedThreadScope(self.module, self.utility, threads=1)
        with self.assertRaisesRegex(FinishResearchFailure, "drifted"):
            with scope:
                self.utility.threads = 8
                self.module._global_seed(self.source, self.target, 304)
        self.assertEqual(self.utility.threads, 20)
        self.assertEqual(self.received, [])

    def test_invalid_policy_types_and_values_rejected(self):
        for value in (True, 0, 2, 8, 20., "1", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GlobalSeedThreadScope(self.module, self.utility, threads=value)

    def test_explicit_policy_is_a_distinct_json_scope_value(self):
        scopes = [canonical_json_scope({"pipeline_options": {"research_ransac_threads": value}})
            for value in (1, 20)]
        self.assertNotEqual(scopes[0], scopes[1])
        self.assertIs(type(scopes[0]["pipeline_options"]["research_ransac_threads"]), int)


if __name__ == "__main__":
    unittest.main()

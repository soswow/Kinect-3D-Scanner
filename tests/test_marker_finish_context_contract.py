"""Stdlib-only integration ordering/restoration for explicit marker policies."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import profile_marker_finish as runner


class Scope:
    def __init__(self, name, history, *, entry_error=None, exit_error=None):
        self.name, self.history = name, history
        self.entry_error, self.exit_error = entry_error, exit_error

    def __enter__(self):
        self.history.append(("enter", self.name))
        if self.entry_error is not None:
            raise self.entry_error
        return self

    def __exit__(self, *arguments):
        self.history.append(("exit", self.name))
        if self.exit_error is not None:
            raise self.exit_error

    def observe(self, owner, name, label):
        self.history.append(("observe", label))


class MarkerFinishContracts(unittest.TestCase):
    def setUp(self):
        self.history = []
        self.original = Scope("original", self.history)
        self.global_scope = Scope("global", self.history)
        self.local_scope = Scope("local", self.history)
        self.ransac = Scope("ransac", self.history)
        self.provider, self.fragments, self.utility = object(), object(), object()
        self.global_factory = patch.object(runner, "GlobalMarkerProposalScope", return_value=self.global_scope)
        self.local_factory = patch.object(runner, "MarkerSeedScope", return_value=self.local_scope)
        self.ransac_factory = patch.object(runner, "GlobalSeedThreadScope", return_value=self.ransac)
        self.provider_factory = patch.object(runner, "OriginalMarkerGlobalProvider", return_value=SimpleNamespace(original=self.provider))
        for value in (self.global_factory, self.local_factory, self.ransac_factory, self.provider_factory):
            value.start()
            self.addCleanup(value.stop)

    def enter(self, stack, mode="marker", seed_scope="global"):
        return runner.enter_finish_contexts(stack, mode=mode, seed_scope=seed_scope,
            original_scope=self.original, fragments=self.fragments, utility=self.utility,
            ransac_threads=1, provider=self.provider, marker_trace=None,
            observations=[(self.fragments, "propose_fragment_poses", "original_proposals")])

    def test_global_original_body_is_captured_before_observers_and_threads(self):
        with ExitStack() as stack:
            marker, ransac = self.enter(stack)
            self.assertIs(marker, self.global_scope)
            self.assertIs(ransac, self.ransac)
            self.assertEqual(self.history, [("enter", "global"), ("enter", "original"),
                                           ("observe", "original_proposals"), ("enter", "ransac")])
        self.assertEqual(self.history[-3:], [("exit", "ransac"), ("exit", "original"), ("exit", "global")])

    def test_local_wraps_original_observer_seam_after_ransac(self):
        with ExitStack() as stack:
            marker, _ = self.enter(stack, seed_scope="local")
            self.assertIs(marker, self.local_scope)
        self.assertEqual(self.history, [("enter", "original"), ("observe", "original_proposals"),
            ("enter", "ransac"), ("enter", "local"), ("exit", "local"), ("exit", "ransac"), ("exit", "original")])

    def test_original_control_never_constructs_or_enters_marker_scope(self):
        with ExitStack() as stack:
            marker, _ = self.enter(stack, mode="original")
            self.assertIsNone(marker)
        self.assertNotIn(("enter", "global"), self.history)
        self.assertNotIn(("enter", "local"), self.history)

    def test_later_entry_fault_restores_original_and_global(self):
        self.ransac.entry_error = RuntimeError("thread entry failed")
        with self.assertRaisesRegex(RuntimeError, "thread entry failed"):
            with ExitStack() as stack:
                self.enter(stack)
        self.assertEqual(self.history[-2:], [("exit", "original"), ("exit", "global")])

    def test_cleanup_fault_still_attempts_outer_global_restore(self):
        self.original.exit_error = RuntimeError("original restore failed")
        with self.assertRaisesRegex(RuntimeError, "original restore failed"):
            with ExitStack() as stack:
                self.enter(stack)
        self.assertEqual(self.history[-3:], [("exit", "ransac"), ("exit", "original"), ("exit", "global")])

    def test_explicit_policy_domains_fail_before_installation(self):
        for mode, seed_scope in (("resident", "global"), ("marker", "both"), ("marker", None)):
            with self.subTest(mode=mode, seed_scope=seed_scope):
                with self.assertRaises(ValueError):
                    with ExitStack() as stack:
                        self.enter(stack, mode=mode, seed_scope=seed_scope)
        self.assertEqual(self.history, [])


if __name__ == "__main__":
    unittest.main()

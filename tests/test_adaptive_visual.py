"""Fallback state must survive rejection, exceptions, eviction and scan reset."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from scanner_server.adaptive_visual import register, select_policy
from scanner_server.engine import ScanEngine
from shared.settings import ScanSettings


class AdaptiveVisualTests(unittest.TestCase):
    def engine(self):
        return SimpleNamespace(_visual_cache={1: (object(), object())},
            _sift_visual_cache={99: (object(), object())},
            _visual_evidence=None, poses=[(1, None)],
            settings=SimpleNamespace(color_recovery=True),
            backend={"visual_features": "orb", "sift_fallback": {"attempts": 0, "verified": 0}})

    def test_registration_error_restores_primary_bank_and_feature_selection(self):
        engine = self.engine()
        original = engine._visual_cache
        def authorize(source, rgbd):
            if engine.backend["visual_features"] == "orb":
                return None
            engine._visual_cache[1] = (object(), object())
            raise RuntimeError("registration failure")
        with self.assertRaisesRegex(RuntimeError, "registration failure"):
            register(engine, None, object(), authorize)
        self.assertIs(engine._visual_cache, original)
        self.assertEqual("orb", engine.backend["visual_features"])
        self.assertEqual(0, engine.backend["sift_fallback"]["verified"])
        self.assertNotIn(99, engine._sift_visual_cache)

    def test_rejected_fallback_does_not_cache_pose_authorization(self):
        engine = self.engine()
        calls = []
        def reject(source, rgbd):
            calls.append(engine.backend["visual_features"])
            return None
        for _ in range(2):
            self.assertIsNone(register(engine, None, object(), reject))
        self.assertEqual(["orb", "sift", "orb", "sift"], calls)
        self.assertEqual(0, engine.backend["sift_fallback"]["verified"])
        self.assertIsNone(engine._visual_evidence)

    def test_new_scan_releases_secondary_bank_and_resets_statistics(self):
        with patch.dict(os.environ, KINECT_VISUAL_FEATURES="adaptive", KINECT_CUDA_MATCHING="cpu",
                        KINECT_ADAPTIVE_EXPERIMENTAL="off"), \
                patch.object(ScanEngine, "BLOCK_COUNT", 32):
            engine = ScanEngine(device="cpu", tracking="legacy")
            engine.reset(settings=ScanSettings(voxel_m=.01, final_voxel_m=.005))
            self.assertEqual("orb_then_sift", engine.backend["visual_policy"])
            engine._sift_visual_cache[0] = (object(), object())
            engine.backend["sift_fallback"].update(attempts=8, verified=3)
            engine.reset()
            self.assertEqual({}, engine._sift_visual_cache)
            self.assertEqual({"attempts": 0, "verified": 0}, engine.backend["sift_fallback"])

    def test_unvalidated_resolutions_keep_the_original_orb_policy(self):
        for live, final in ((.005, None), (.005, .005), (.01, None), (.01, .01), (.015, .005)):
            with self.subTest(live=live, final=final):
                policy = select_policy(ScanSettings(voxel_m=live, final_voxel_m=final), "adaptive")
                self.assertEqual("adaptive", policy["visual_policy_requested"])
                self.assertEqual("orb", policy["visual_policy"])
                self.assertEqual("orb", policy["visual_features"])
                self.assertIn("using ORB", policy["visual_policy_reason"])

    def test_experimental_override_is_explicit_and_validated(self):
        settings = ScanSettings()
        policy = select_policy(settings, "adaptive", "on")
        self.assertEqual("orb_then_sift", policy["visual_policy"])
        self.assertEqual("on", policy["adaptive_experimental"])
        self.assertIn("unvalidated", policy["visual_policy_reason"])
        with self.assertRaisesRegex(ValueError, "KINECT_ADAPTIVE_EXPERIMENTAL"):
            select_policy(settings, "adaptive", "true")
        for requested in ("orb", "sift"):
            self.assertEqual(requested, select_policy(settings, requested, "on")["visual_policy"])

    def test_reset_reselects_policy_and_disabled_adaptive_never_calls_fallback(self):
        with patch.dict(os.environ, KINECT_VISUAL_FEATURES="adaptive", KINECT_CUDA_MATCHING="cpu",
                        KINECT_ADAPTIVE_EXPERIMENTAL="off"), \
                patch.object(ScanEngine, "BLOCK_COUNT", 32):
            engine = ScanEngine(device="cpu", tracking="legacy")
            with patch.object(engine, "_visual_register_primary", return_value=sentinel.primary), \
                    patch("scanner_server.adaptive_visual.register", return_value=sentinel.fallback) as fallback:
                self.assertEqual("orb", engine.backend["visual_policy"])
                self.assertIs(sentinel.primary, engine._visual_register(None, None))
                fallback.assert_not_called()
                engine.reset(settings=ScanSettings(voxel_m=.01, final_voxel_m=.005))
                self.assertEqual("orb_then_sift", engine.backend["visual_policy"])
                self.assertIs(sentinel.fallback, engine._visual_register(None, None))
                fallback.assert_called_once()
                engine._sift_visual_cache[0] = (object(), object())
                engine.reset(settings=ScanSettings())
                self.assertEqual("orb", engine.backend["visual_policy"])
                self.assertEqual({}, engine._sift_visual_cache)
                self.assertIs(sentinel.primary, engine._visual_register(None, None))
                fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()

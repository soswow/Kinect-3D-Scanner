"""Measured tag-map priority, optional coverage, and live fallback behavior."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from unittest.mock import patch
import cv2
import numpy as np

from shared.apriltag import TagFrame, without_repeated
from shared.settings import ScanSettings
from scanner_server.appearance import LazyFeatures
from scanner_server.apriltag_tracking import observation
from scanner_server.depth_graph import propose_depth_poses
from scanner_server.engine import ScanEngine
from tests.test_apriltag import tag_scene, FAMILIES


class AprilTagPriorityTests(unittest.TestCase):
    def engine(self, **settings):
        engine = ScanEngine(device="cpu")
        self.addCleanup(engine.shutdown)
        engine.reset(settings=ScanSettings(apriltag_tracking=True,
                                          apriltag_dictionaries=FAMILIES, **settings))
        return engine

    def test_three_tag_views_solve_without_ordinary_features_or_general_search(self):
        engine = self.engine(color_recovery=True)
        for i, shift in enumerate((0, 4, 8)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": i*.1})
        with patch("scanner_server.appearance.extract_features", side_effect=AssertionError("unneeded features")), \
                patch("scanner_server.depth_graph.register_pair", side_effect=AssertionError("unneeded search")):
            poses, report = propose_depth_poses(engine)
        self.assertEqual([0, 1, 2], [i for i, _ in poses], report)
        self.assertTrue(report["apriltag_priority"]["applied"], report)
        self.assertEqual([], report["apriltag_priority"]["recovered_cameras"])
        self.assertEqual(8, report["joint_refinements"][0]["apriltag_landmarks"])
        self.assertAlmostEqual(8*2/525, poses[-1][1][0, 3], delta=.002)

    def test_missing_tags_require_measured_recovery_and_preserve_general_fallback(self):
        engine = self.engine(color_recovery=True)
        for i, shift in enumerate((0, 4, 8, 12)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": i*.1})
        for i in range(3):
            observation(engine, i)
        engine._apriltag_observations[3] = TagFrame()
        with patch("scanner_server.geometry_registration.register_pair", return_value=(None, {"accepted": False})), \
                patch("scanner_server.depth_graph.solve_graph", return_value=([], [], [])):
            poses, report = propose_depth_poses(engine)
        self.assertIsNone(poses)
        self.assertFalse(report["apriltag_priority"]["applied"])
        self.assertEqual([3], report["apriltag_priority"]["remaining_cameras"])
        self.assertTrue(any(p["method"] == "local" for p in report["pairs"]))

    def test_untagged_capture_is_recovered_from_raw_rgb_depth_in_compact_map(self):
        from tests.test_visual_tracking import textured_plane
        engine = self.engine(color_recovery=True)
        for i, shift in enumerate((0, 4, 8, 12)):
            rgb, depth = textured_plane(shift=shift)
            if i != 2:
                tags, _ = tag_scene(shift)
                for x, y in ((110, 100), (430, 280)):
                    rgb[y-10:y+110, x-shift-10:x-shift+110] = 255
                    rgb[y:y+100, x-shift:x-shift+100] = tags[y:y+100, x-shift:x-shift+100]
            engine.store_frame(rgb, depth, {"timestamp_s": i*.1})
        poses, report = propose_depth_poses(engine)
        self.assertEqual([0, 1, 2, 3], [i for i, _ in poses], report)
        self.assertTrue(report["apriltag_priority"]["applied"], report)
        self.assertEqual([2], report["apriltag_priority"]["recovered_cameras"])
        self.assertTrue(any(r["source"] == 2 for r in report["apriltag_priority"]["final_measured_recovery"]))
        np.testing.assert_allclose(poses[2][1][0, 3], 8*2/525, atol=.002)

    def test_duplicate_seen_later_quarantines_earlier_observations_without_mutating_raw_cache(self):
        engine = self.engine()
        engine.store_frame(*tag_scene(), {"timestamp_s": 0})
        engine.store_frame(*tag_scene(families=(FAMILIES[0],)*2), {"timestamp_s": .1})
        first = observation(engine, 0)
        self.assertEqual(2, len(first.tags))
        observation(engine, 1)
        self.assertEqual({(FAMILIES[1], 7)}, set(observation(engine, 0).tags))
        self.assertEqual(2, len(engine._apriltag_observations[0].tags))
        filtered, repeated = without_repeated([first, observation(engine, 1)])
        self.assertEqual({(FAMILIES[0], 7)}, repeated)
        self.assertEqual({(FAMILIES[1], 7)}, set(filtered[0].tags))

    def test_native_rgb_duplicate_is_excluded_even_when_prepared_view_has_one_copy(self):
        engine = self.engine()
        rgb, depth = tag_scene()
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        native, _ = tag_scene(families=(FAMILIES[0],)*2)
        native = cv2.resize(native, (1280, 960), interpolation=cv2.INTER_NEAREST)
        engine.raw_frames[0] = (native, depth)
        frame = observation(engine, 0, rgb, depth)
        self.assertEqual({(FAMILIES[1], 7)}, set(frame.tags))
        self.assertEqual({(FAMILIES[0], 7)}, frame.repeated)

    def test_failed_independent_depth_audit_cannot_authorize_tag_map(self):
        engine = self.engine()
        for i, shift in enumerate((0, 4, 8)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": i*.1})
        rejected = {"accepted": False, "free_space_fraction": .5}
        with patch("scanner_server.depth_graph.scene_visibility", return_value=rejected), \
                patch("scanner_server.depth_graph.solve_graph", return_value=([], [], [])):
            poses, report = propose_depth_poses(engine)
        self.assertIsNone(poses)
        self.assertEqual("final_tag_or_depth_validation_failed", report["apriltag_priority"]["reason"])
        self.assertFalse(report["apriltag_priority"]["applied"])
        self.assertFalse(engine.poses)

    def test_live_tags_skip_normal_estimation_and_missing_tags_reach_color_fallback(self):
        engine = self.engine(color_recovery=True)
        for i, shift in enumerate((0, 4)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": i*.1})
        with patch.object(engine, "_ensure_raw_normals", side_effect=AssertionError("unneeded normals")), \
                patch.object(engine, "_visual_register_primary", side_effect=AssertionError("unneeded color")):
            engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertEqual("apriltag", engine.diagnostics[-1]["visual_evidence"]["kind"])
        # Absence of tags still invokes the existing feature path.
        engine.store_frame(*tag_scene(8), {"timestamp_s": .2})
        engine._apriltag_observations[2] = TagFrame()
        with patch.object(engine, "_visual_register_primary", wraps=engine._visual_register_primary) as fallback:
            engine.process_frames()
        self.assertTrue(fallback.called)

    def test_live_tag_preview_defers_model_levels_until_geometry_fallback(self):
        engine = self.engine(color_recovery=True, confidence_fusion=False)
        def scene(shift):
            rgb, depth = tag_scene(shift)
            # Avoid a surface exactly on a voxel boundary in this synthetic plane.
            depth[:] = 2003
            return rgb, depth
        engine.store_frame(*scene(0), {"timestamp_s": 0})
        engine.process_frames()
        for i in range(1, 4):
            engine.store_frame(*scene(i*4), {"timestamp_s": i*.1})
        with patch.object(engine, "_extract_model_pcd", side_effect=AssertionError("unneeded model levels")):
            engine.process_frames()
        self.assertEqual(4, engine.frame_count)
        self.assertIsNotNone(engine._pending_model_cloud)
        self.assertGreater(len(engine._live_points), 0)
        engine.store_frame(*scene(16), {"timestamp_s": .4})
        engine._apriltag_observations[4] = TagFrame()
        with patch.object(engine, "_visual_register_primary", return_value=None), \
                patch.object(engine, "_extract_model_pcd", wraps=engine._extract_model_pcd) as prepare:
            engine.process_frames()
        prepare.assert_any_call(from_snapshot=True)
        self.assertIsNone(engine._pending_model_cloud)

    def test_lazy_features_cache_failures_and_preserve_sequence_behavior(self):
        calls = []
        features = LazyFeatures(3, lambda i: calls.append(i))
        self.assertTrue(features)
        self.assertEqual(3, len(features))
        self.assertIsNone(features[1])
        self.assertIsNone(features[1])
        self.assertEqual([1], calls)
        self.assertEqual([None, None], features[-2:])
        self.assertEqual([1, 2], calls)
        with self.assertRaises(IndexError):
            features[3]


if __name__ == "__main__":
    unittest.main()

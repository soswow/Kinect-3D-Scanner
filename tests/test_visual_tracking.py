"""Motion accuracy and measured visual authority across sparse fusion updates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scanner_server.engine import ScanEngine
from scanner_server.fragments import (
    Fragment,
    _prepare_fragment,
    _verify_visual_bridge,
    _view,
    propose_fragment_poses,
)
from scanner_server.refinement import motion
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker
from tests.test_quality import scene_frames


def textured_plane(seed=8, shift=0):
    rng = np.random.default_rng(seed)
    rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), (640, 480))
    rgb = cv2.warpAffine(rgb, np.array([[1, 0, -shift], [0, 1, 0]], float), (640, 480))
    return rgb, np.full((480, 640), 2000, np.uint16)


class VisualTrackingTests(unittest.TestCase):
    def test_continuous_motion_has_metric_accuracy_and_seeds_sparse_fusion(self):
        tracker = VisualTracker(ScanSettings(color_recovery=True))
        frames = scene_frames(12)
        reports, errors = [], []
        for i, (rgb, depth, truth) in enumerate(frames):
            report = tracker.update(rgb, depth, {"timestamp_s": i * 0.1, "rgb_depth_delta_ms": 0})
            self.assertTrue(report["valid"], report)
            reports.append(report)
            errors.append(motion(np.linalg.inv(truth) @ np.array(report["camera_to_local"])))
        self.assertLess(np.sqrt(np.mean(np.square(np.array(errors)[:, 0]))), 0.01)
        self.assertLess(errors[-1][0], 0.02)
        self.assertLess(max(angle for _, angle in errors), 2)
        # The server sees only two captures, but receives motion measured on
        # the ten intervening camera frames. No ground-truth pose is supplied.
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True))
        engine.store_frame(*frames[0][:2], {"timestamp_s": 0, "visual_tracking": reports[0]})
        engine.process_frames()
        engine.store_frame(*frames[-1][:2], {"timestamp_s": 1.1, "visual_tracking": reports[-1]})
        guess = engine._tracking_initial_guess()
        self.assertLess(motion(np.linalg.inv(frames[-1][2]) @ guess)[0], 0.02)
        engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertLess(motion(np.linalg.inv(frames[-1][2]) @ engine.cumulative_T)[0], 0.025)

    def test_lost_timing_gap_and_wrong_depth_start_new_segments(self):
        tracker = VisualTracker(ScanSettings(color_recovery=True))
        rgb, depth = textured_plane()
        first = tracker.update(rgb, depth, {"timestamp_s": 0})
        moved = tracker.update(*textured_plane(shift=2), {"timestamp_s": 0.1})
        self.assertTrue(moved["valid"])
        self.assertEqual(first["segment"], moved["segment"])
        gap = tracker.update(rgb, depth, {"timestamp_s": 2})
        self.assertFalse(gap["valid"])
        self.assertNotEqual(moved["segment"], gap["segment"])
        self.assertEqual(0, gap["steps"])
        resumed = tracker.update(rgb, depth, {"timestamp_s": 2.1})
        self.assertTrue(resumed["valid"])
        # Identical color cannot invent motion through inconsistent depth.
        wrong_depth = tracker.update(rgb, depth + 400, {"timestamp_s": 2.2})
        self.assertFalse(wrong_depth["valid"])
        late = tracker.update(rgb, depth, {"timestamp_s": 2.3, "rgb_depth_delta_ms": 21})
        self.assertFalse(late["valid"])
        blank = tracker.update(np.zeros_like(rgb), depth, {"timestamp_s": 2.4})
        self.assertFalse(blank["valid"])

    def test_textured_plane_tracks_but_anonymous_geometry_cannot_override_features(self):
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True))
        engine.store_frame(*textured_plane(), {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(*textured_plane(shift=21), {"timestamp_s": 0.1})
        wrong = np.eye(4)
        wrong[0, 3] = 0.20
        # A plane still fits after sliding along itself. Even an apparently
        # perfect geometric score must not erase the measured feature matches.
        with patch.object(engine, "_icp", return_value=SimpleNamespace(
                transformation=wrong, fitness=1.0, inlier_rmse=0.0)):
            engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        result = engine.diagnostics[-1]
        self.assertEqual("keyframe+visual", result["method"])
        self.assertAlmostEqual(0.08, engine.cumulative_T[0, 3], delta=0.005)
        self.assertGreater(result["visual_evidence"]["feature_support"]["inliers"], 100)
        self.assertEqual(1, len(engine.reconstruction_report()["tracking_edges"]))
        # Blank color gives no planar-motion authority, even with a valid
        # camera-side pose hint. Rejection must leave the volume and pose intact.
        saved = engine.cumulative_T.copy()
        engine.store_frame(np.zeros((480, 640, 3), np.uint8), textured_plane()[1], {
            "timestamp_s": 0.2, "visual_tracking": {
                "valid": True, "segment": "invented", "camera_to_local": np.eye(4).tolist()}})
        engine.process_frames()
        self.assertEqual(2, engine.frame_count)
        np.testing.assert_array_equal(saved, engine.cumulative_T)

    def test_lost_track_can_match_an_earlier_measured_keyframe(self):
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True))
        for i, seed in enumerate((8, 91, 8)):
            engine.store_frame(*textured_plane(seed), {"timestamp_s": i * 0.1})
        last = np.eye(4)
        last[0, 3] = 0.5
        engine.poses = [(0, np.eye(4)), (1, last)]
        engine.cumulative_T = last
        engine._processed_count = 2
        engine._tracking_lost_frames = 1
        rgb, depth = engine.raw_frames[2]
        rgbd = engine._make_rgbd(rgb, depth)
        result, method = engine._register(engine._make_reg_pcd(rgbd), rgbd)
        self.assertEqual("keyframe+visual", method)
        self.assertEqual(0, engine._visual_evidence["target_index"])
        self.assertLess(motion(result.transformation)[0], 0.005)

    def test_pose_hints_require_same_rigid_bounded_segment(self):
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True))
        engine.poses = [(0, np.eye(4))]
        engine._processed_count = 1
        hint = {"valid": True, "segment": "a", "camera_to_local": np.eye(4).tolist()}
        engine.frame_metadata = [{"visual_tracking": hint}, {"visual_tracking": dict(hint)}]
        self.assertIsNotNone(engine._continuous_motion_guess())
        for bad in ({"segment": "b"}, {"valid": False}, {"camera_to_local": [[0]]},
                    {"camera_to_local": np.diag([2, 1, 1, 1]).tolist()}):
            engine.frame_metadata[1]["visual_tracking"] = {**hint, **bad}
            self.assertIsNone(engine._continuous_motion_guess(), bad)
        huge = np.eye(4)
        huge[0, 3] = 2
        engine.frame_metadata[1]["visual_tracking"] = {**hint, "camera_to_local": huge.tolist()}
        self.assertIsNone(engine._continuous_motion_guess())
        engine.frame_metadata[1] = {"visual_tracking": hint, "rgb_depth_delta_ms": 21}
        self.assertIsNone(engine._continuous_motion_guess())

    def test_dense_color_refinement_rejects_large_geometric_slide(self):
        rgb, depth, _ = scene_frames(1)[0]
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True))
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(rgb, depth, {"timestamp_s": 0.1})
        wrong = np.eye(4)
        wrong[0, 3] = 0.20
        with patch("open3d.pipelines.odometry.compute_rgbd_odometry",
                   return_value=(True, np.eye(4), np.eye(6))), patch.object(
                engine, "_icp", return_value=SimpleNamespace(transformation=wrong)), patch.object(
                engine, "_alignment_error", return_value=None):
            self.assertIsNone(engine._color_recovery(engine.model_pcd, engine._last_rgbd))

    def test_final_graph_preserves_measured_texture_and_requires_independent_witnesses(self):
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(color_recovery=True, reconnect_fragments=True))
        for i in range(4):
            engine.store_frame(*textured_plane(shift=i * 10), {"timestamp_s": i * 0.1})
        engine.process_frames()
        self.assertEqual(4, engine.frame_count, engine.diagnostics)
        for index, pose in engine.poses:
            if index >= 2:
                pose[0, 3] += 0.12  # A drifted second segment is only a seed.
        # Force a fragment boundary like the real bounded final pass. Flat
        # depth alone cannot link these maps; two moved, textured cameras can.
        with patch("scanner_server.fragments.MAX_FRAGMENTS", 8), patch(
                "scanner_server.fragments.capture_gap_limit", return_value=0.15):
            engine.frame_metadata[2]["timestamp_s"] = 10
            engine.frame_metadata[3]["timestamp_s"] = 10.1
            poses, report = propose_fragment_poses(engine)
        self.assertIsNotNone(poses, report)
        self.assertEqual(4, len(poses))
        self.assertFalse(report["unconnected_fragments"])
        self.assertTrue(any(e["validation_scope"] == "visual and held-out camera pairs"
                            for e in report["verified_bridges"]))
        for i, pose in poses:
            self.assertAlmostEqual(i * 10 * 2 / 525, pose[0, 3], delta=0.01)
        # Duplicate stationary observations and a lone matching camera do not
        # satisfy independent bridge support, even with hundreds of matches.
        a = Fragment(0, views=[_view(engine, 0), _view(engine, 0)])
        b = Fragment(1, views=[_view(engine, 0), _view(engine, 0)])
        _prepare_fragment(a)
        _prepare_fragment(b)
        self.assertIsNone(_verify_visual_bridge(a, b, np.eye(4), engine.settings.camera))


if __name__ == "__main__":
    unittest.main()

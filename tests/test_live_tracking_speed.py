"""Source-depth pruning must preserve the measured LK correspondence set."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker, measured_rigid_motion, sampled_points


def frames(shift=0, holes=0.18):
    rng = np.random.default_rng(44)
    rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), np.uint8), (640, 480))
    depth = np.full((480, 640), 2000, np.uint16)
    depth[rng.random(depth.shape) < holes] = 0
    # A real discontinuity and holes must never become measured support.
    depth[190:250, 290:350] = 2400
    transform = np.array([[1, 0, -shift], [0, 1, 0]], float)
    return (cv2.warpAffine(rgb, transform, (640, 480)),
            cv2.warpAffine(depth, transform, (640, 480), flags=cv2.INTER_NEAREST))


def legacy_correspondences(gray_a, gray_b, depth_a, depth_b, corners, camera):
    """The unpruned pre-optimization LK measurement path, without a pose solve."""
    new, forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, corners, None, winSize=(21, 21), maxLevel=3)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(
        gray_b, gray_a, new, None, winSize=(21, 21), maxLevel=3)
    a, b = corners.reshape(-1, 2), new.reshape(-1, 2)
    source, source_measured = sampled_points(depth_a, a, camera)
    target, target_measured = sampled_points(depth_b, b, camera)
    good = ((forward.ravel() > 0) & (reverse.ravel() > 0) &
            (np.linalg.norm(a - back.reshape(-1, 2), axis=1) < 0.8) &
            source_measured & target_measured)
    return source[good], target[good], b[good]


class LiveTrackingSpeedTests(unittest.TestCase):
    def test_pruned_flow_preserves_correspondence_identity_and_metric_pose(self):
        settings = ScanSettings(filter_depth=False)
        rgb_a, depth_a = frames()
        tracker = VisualTracker(settings)
        tracker.update(rgb_a, depth_a, {"timestamp_s": 0})
        gray_a = tracker.history[0][0]
        corners = cv2.goodFeaturesToTrack(
            gray_a, 500, 0.015, 9, mask=(depth_a > 0).astype(np.uint8))
        cached = tracker.history[0][2]
        self.assertLess(len(cached), len(corners))
        # Larger shifts also exercise border points and newly occluded depths.
        for shift in (2, 12, 28):
            with self.subTest(shift=shift):
                rgb_b, depth_b = frames(shift)
                gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
                source, target, pixels = legacy_correspondences(
                    gray_a, gray_b, depth_a, depth_b, corners, settings.camera)
                with patch("shared.visual_tracking.cv2.solvePnPRansac",
                           wraps=cv2.solvePnPRansac) as solve, patch(
                               "shared.visual_tracking.measured_rigid_motion",
                               wraps=measured_rigid_motion) as rigid:
                    pose, report = tracker._match_reference(
                        tracker.history[0], gray_b, depth_b, 0.1)
                self.assertIsNotNone(pose, report)
                np.testing.assert_array_equal(source, solve.call_args.args[0])
                np.testing.assert_array_equal(pixels, solve.call_args.args[1])
                np.testing.assert_array_equal(source, rigid.call_args.args[0])
                np.testing.assert_array_equal(target, rigid.call_args.args[1])
                self.assertAlmostEqual(shift * 2 / 525, pose[0, 3], delta=0.002)

    def test_accepted_reference_depth_is_sampled_only_once(self):
        tracker = VisualTracker(ScanSettings(filter_depth=False))
        tracker.update(*frames(), {"timestamp_s": 0})
        old_depth = tracker.history[-1][1]
        with patch("shared.visual_tracking.sampled_points", wraps=sampled_points) as sample:
            report = tracker.update(*frames(2), {"timestamp_s": 0.1})
        self.assertTrue(report["valid"], report)
        self.assertTrue(sample.called)
        self.assertFalse(any(call.args[0] is old_depth for call in sample.call_args_list))
        # The target is sampled at tracked locations and then at the newly
        # detected reference corners; neither step fills unknown depth.
        self.assertEqual(2, sample.call_count)

    def test_too_few_measured_source_features_skip_flow_without_authorizing_motion(self):
        rgb, depth = frames(holes=0)
        checker = np.indices(depth.shape).sum(axis=0) % 2
        depth[checker > 0] += 300
        tracker = VisualTracker(ScanSettings(filter_depth=False))
        first = tracker.update(rgb, depth, {"timestamp_s": 0})
        self.assertTrue(first["valid"])
        self.assertEqual(0, len(tracker.history[-1][2]))
        with patch("shared.visual_tracking.cv2.calcOpticalFlowPyrLK") as flow:
            failed = tracker.update(rgb, depth, {"timestamp_s": 0.1})
        flow.assert_not_called()
        self.assertFalse(failed["valid"])
        self.assertEqual(first["segment"], failed["segment"])
        self.assertEqual(0, failed["steps"])


if __name__ == "__main__":
    unittest.main()

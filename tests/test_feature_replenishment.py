"""Persistent feature lifetimes, spatial top-ups, and failure/recovery authority."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker, sampled_points


def frame():
    rng = np.random.default_rng(8)
    rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), np.uint8), (640, 480))
    return rgb, np.full((480, 640), 2000, np.uint16)


def subset(reference, indices):
    return reference._replace(corners=reference.corners[indices], points=reference.points[indices],
                              ids=reference.ids[indices], born_s=reference.born_s[indices],
                              observations=reference.observations[indices])


class FeatureReplenishmentTests(unittest.TestCase):
    def tracker(self):
        tracker = VisualTracker(ScanSettings(filter_depth=False))
        tracker.debug_enabled = True
        return tracker

    def test_healthy_features_keep_ids_and_age_without_running_corner_detection(self):
        tracker = self.tracker()
        rgb, depth = frame()
        initial = tracker.update(rgb, depth, {"timestamp_s": 0})
        ids = tracker.history[-1].ids.copy()
        with patch("shared.visual_tracking.cv2.goodFeaturesToTrack", wraps=cv2.goodFeaturesToTrack) as detect:
            for step in range(1, 13):
                report = tracker.update(rgb, depth, {"timestamp_s": step * 0.1})
                self.assertTrue(report["valid"], report)
                np.testing.assert_array_equal(ids, tracker.history[-1].ids)
                np.testing.assert_array_equal(step + 1, tracker.history[-1].observations)
                self.assertEqual(0, report["tracks"]["added"])
                self.assertEqual(0, len(tracker.debug_snapshot["corners"]))
        detect.assert_not_called()
        self.assertEqual(initial["segment"], report["segment"])
        self.assertAlmostEqual(1.2, report["tracks"]["median_age_s"])
        self.assertEqual(48, report["tracks"]["occupied_cells"])
        self.assertEqual(5, len(tracker.history))
        # Reports remain JSON serializable; identity arrays are local debug data.
        json.dumps(report, allow_nan=False)

    def test_count_loss_adds_new_ids_and_retains_measured_survivors(self):
        tracker = self.tracker()
        rgb, depth = frame()
        tracker.update(rgb, depth, {"timestamp_s": 0})
        initial = tracker.history[-1]
        depth[:, 420:] = 0
        report = tracker.update(rgb, depth, {"timestamp_s": 0.2})
        self.assertTrue(report["valid"], report)
        self.assertEqual("count", report["tracks"]["replenishment"])
        self.assertGreater(report["tracks"]["retained"], 100)
        self.assertLess(report["tracks"]["retained"], 400)
        self.assertGreater(report["tracks"]["added"], 0)
        current = tracker.history[-1]
        self.assertLessEqual(len(current.ids), 500)
        self.assertEqual(len(current.ids), len(np.unique(current.ids)))
        survivors = np.isin(current.ids, initial.ids)
        np.testing.assert_array_equal(0, current.born_s[survivors])
        np.testing.assert_array_equal(2, current.observations[survivors])
        np.testing.assert_array_equal(0.2, current.born_s[~survivors])
        np.testing.assert_array_equal(1, current.observations[~survivors])
        self.assertTrue(np.all(current.ids[~survivors] > initial.ids.max()))
        points, measured = sampled_points(depth, current.corners, tracker.settings.camera)
        self.assertTrue(measured.all())
        np.testing.assert_array_equal(points, current.points)
        # New corners cannot duplicate an existing subpixel survivor.
        distances = np.linalg.norm(current.corners[survivors, 0, None] - current.corners[~survivors, 0], axis=2)
        self.assertGreaterEqual(float(distances.min()), tracker.MIN_SPACING_PX)

    def test_coverage_loss_replenishes_a_sparse_cell_above_the_count_threshold(self):
        tracker = self.tracker()
        rgb, depth = frame()
        tracker.update(rgb, depth, {"timestamp_s": 0})
        initial = tracker.history[-1]
        cells = tracker._cell_indices(initial.corners)
        survivors = subset(initial, np.flatnonzero(cells != 19))
        self.assertGreater(len(survivors.ids), 400)
        current, added, _, reason = tracker._replenish(initial.gray, depth, survivors, 0.2)
        self.assertEqual("coverage", reason)
        self.assertGreater(len(added), 0)
        np.testing.assert_array_equal(19, tracker._cell_indices(added))
        np.testing.assert_array_equal(survivors.ids, current.ids[:len(survivors.ids)])
        self.assertLessEqual(len(current.ids), 500)

    def test_cooldown_bounds_topups_but_critical_shortage_can_bypass_it(self):
        tracker = self.tracker()
        rgb, depth = frame()
        tracker.update(rgb, depth, {"timestamp_s": 0})
        original = tracker.history[-1]
        partial = depth.copy()
        partial[:, 420:] = 0
        with patch("shared.visual_tracking.cv2.goodFeaturesToTrack", wraps=cv2.goodFeaturesToTrack) as detect:
            report = tracker.update(rgb, partial, {"timestamp_s": 0.1})
            self.assertTrue(report["valid"], report)
            self.assertEqual("cooldown", report["tracks"]["replenishment"])
            detect.assert_not_called()
            report = tracker.update(rgb, partial, {"timestamp_s": 0.2})
            self.assertTrue(report["valid"], report)
            self.assertGreater(report["tracks"]["added"], 0)
            detect.assert_called_once()
        few = subset(original, np.arange(40))
        _, added, _, reason = tracker._replenish(original.gray, depth, few, 0.21)
        self.assertEqual("count", reason)
        self.assertGreater(len(added), 0)

    def test_failed_images_do_not_advance_tracks_or_create_new_ids(self):
        tracker = self.tracker()
        rgb, depth = frame()
        first = tracker.update(rgb, depth, {"timestamp_s": 0})
        reference = tracker.history[-1]
        next_id = tracker._next_feature_id
        with patch("shared.visual_tracking.cv2.goodFeaturesToTrack", wraps=cv2.goodFeaturesToTrack) as detect:
            failed = tracker.update(rgb, depth + 400, {"timestamp_s": 0.1})
            late = tracker.update(rgb, depth, {"timestamp_s": 0.2, "rgb_depth_delta_ms": 21})
            blank = tracker.update(np.zeros_like(rgb), depth, {"timestamp_s": 0.3})
        self.assertFalse(failed["valid"])
        self.assertFalse(late["valid"])
        self.assertFalse(blank["valid"])
        detect.assert_not_called()
        self.assertIs(reference, tracker.history[-1])
        self.assertEqual(next_id, tracker._next_feature_id)
        self.assertEqual(first["segment"], blank["segment"])
        recovered = tracker.update(rgb, depth, {"timestamp_s": 0.4})
        self.assertTrue(recovered["valid"], recovered)
        np.testing.assert_array_equal(reference.ids, tracker.history[-1].ids)
        np.testing.assert_array_equal(2, tracker.history[-1].observations)

    def test_older_reference_recovery_keeps_identity_and_observation_counts(self):
        tracker = self.tracker()
        rgb, depth = frame()
        for step in range(3):
            tracker.update(rgb, depth, {"timestamp_s": step * 0.1})
        latest = tracker.history[-1]
        ids = latest.ids.copy()
        positions = latest.corners.copy()
        match = tracker._match_reference
        def older_only(reference, *args):
            return (None, {}) if reference is latest else match(reference, *args)
        with patch.object(tracker, "_match_reference", side_effect=older_only):
            recovered = tracker.update(rgb, depth, {"timestamp_s": 0.3})
        self.assertTrue(recovered["valid"], recovered)
        self.assertEqual(0.1, recovered["reference_timestamp_s"])
        np.testing.assert_array_equal(ids, tracker.history[-1].ids)
        np.testing.assert_array_equal(4, tracker.history[-1].observations)
        np.testing.assert_array_equal(0, tracker.history[-1].born_s)
        np.testing.assert_array_equal(positions, latest.corners)

    def test_chain_reset_starts_new_lifetimes_without_claiming_verified_motion(self):
        tracker = self.tracker()
        rgb, depth = frame()
        initial = tracker.update(rgb, depth, {"timestamp_s": 0})
        tracker.update(rgb, depth, {"timestamp_s": 0.1})
        reset = tracker.update(rgb, depth, {"timestamp_s": 2})
        self.assertFalse(reset["valid"])
        self.assertNotEqual(initial["segment"], reset["segment"])
        np.testing.assert_array_equal(2, tracker.history[-1].born_s)
        np.testing.assert_array_equal(1, tracker.history[-1].observations)
        self.assertEqual(0, reset["tracks"]["retained"])
        self.assertEqual(0, reset["tracks"]["max_age_s"])

    def test_rgb_corners_without_stable_depth_cannot_seed_a_field(self):
        tracker = self.tracker()
        rgb, depth = frame()
        depth[:, ::2] += 400
        report = tracker.update(rgb, depth, {"timestamp_s": 0})
        self.assertFalse(report["valid"])
        self.assertFalse(tracker.history)
        self.assertEqual(0, report["tracks"]["active"])
        self.assertGreater(tracker.debug_snapshot["detected"], 100)

    def test_less_precise_pose_support_is_retired_from_future_feature_lifetimes(self):
        tracker = self.tracker()
        rgb, depth = frame()
        tracker.update(rgb, depth, {"timestamp_s": 0})
        y, x = np.indices(depth.shape, dtype=np.float32)
        # Most motion is coherent, but some patches drift by up to 2 pixels.
        # This can support a short step without establishing a precise lifetime.
        warped = cv2.remap(rgb, x + 2.0 * np.sin(y / 20), y, cv2.INTER_LINEAR)
        report = tracker.update(warped, depth, {"timestamp_s": 0.2})
        self.assertTrue(report["valid"], report)
        self.assertGreater(report["tracks"]["retired"], 0)
        self.assertEqual(report["inliers"], report["tracks"]["retained"] + report["tracks"]["retired"])
        verified = tracker.debug_snapshot["status"] == VisualTracker.VERIFIED
        self.assertEqual(report["inliers"], int(verified.sum()))

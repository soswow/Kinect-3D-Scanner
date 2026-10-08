"""Diagnostics must reflect measured flow without entering recorded captures."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui import main_window
from kinect_scanner.gui.preferences import ScannerPreferences
from kinect_scanner.gui.tracking_debug import flow_image, flow_summary
from kinect_scanner.worker import KinectWorker
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker
from tests.test_auto_capture import NoCamera, NoTasks


def frame(shift=0):
    rng = np.random.default_rng(8)
    rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), (640, 480))
    rgb = cv2.warpAffine(rgb, np.array([[1, 0, -shift], [0, 1, 0]], float), (640, 480))
    return rgb, np.full((480, 640), 2000, np.uint16)


class TrackingDebugTests(unittest.TestCase):
    def tracker(self):
        tracker = VisualTracker(ScanSettings(color_recovery=True))
        tracker.debug_enabled = True
        return tracker

    def test_snapshot_follows_actual_motion_and_is_separate_from_pose_report(self):
        tracker = self.tracker()
        first = tracker.update(*frame(), {"timestamp_s": 0})
        self.assertIn("no motion measured", tracker.debug_snapshot["reason"])
        self.assertNotIn("status", tracker.debug_snapshot)
        report = tracker.update(*frame(3), {"timestamp_s": 0.1})
        snapshot = tracker.debug_snapshot
        self.assertTrue(report["valid"], report)
        self.assertNotIn("image", report)
        self.assertNotIn("trail_segments", report)
        verified = snapshot["status"] == VisualTracker.VERIFIED
        self.assertEqual(report["inliers"], int(verified.sum()))
        self.assertAlmostEqual(-3, float(np.median((snapshot["target"] - snapshot["source"])[verified, 0])), delta=0.2)
        self.assertEqual(100, snapshot["reference_age_ms"])
        self.assertEqual(first["segment"], report["segment"])
        self.assertIn("Verified camera motion", flow_summary(snapshot))

    def test_trails_keep_twenty_camera_frames_without_extending_pose_history(self):
        tracker = self.tracker()
        oldest = None
        queued = None
        for step in range(26):
            report = tracker.update(*frame(step * 2), {"timestamp_s": step * .1})
            self.assertTrue(report["valid"], report)
            if step == 1:
                queued = tracker.debug_snapshot
                queued_segments = queued["trail_segments"].copy()
            if step == 6:
                reference = tracker.history[-1]
                oldest = dict(zip(reference.ids, reference.corners[:, 0].copy()))
        snapshot = tracker.debug_snapshot
        self.assertEqual(20, snapshot["trail_frame_count"])
        self.assertEqual(20, snapshot["trail_frame_limit"])
        self.assertEqual(5, len(tracker.history))
        self.assertLessEqual(len(snapshot["trail_segments"]), 19 * tracker.MAX_CORNERS)
        ids, counts = np.unique(snapshot["trail_ids"], return_counts=True)
        complete = ids[counts == 19]
        self.assertGreater(len(complete), 100)
        selected = snapshot["trail_ids"] == complete[0]
        path = snapshot["trail_segments"][selected]
        np.testing.assert_array_equal(oldest[complete[0]], path[0, 0])
        np.testing.assert_array_equal(path[:-1, 1], path[1:, 0])
        np.testing.assert_array_equal(np.arange(18, -1, -1), snapshot["trail_age_frames"][selected])
        np.testing.assert_allclose(-2, path[:, 1, 0] - path[:, 0, 0], atol=.15)
        np.testing.assert_array_equal(queued_segments, queued["trail_segments"])
        self.assertIn("20 / 20 camera frames", flow_summary(snapshot))

    def test_replenished_corners_cannot_inherit_a_different_feature_trail(self):
        tracker = self.tracker()
        rgb, depth = frame()
        tracker.update(rgb, depth, {"timestamp_s": 0})
        previous = tracker.history[-1]
        depth[:, 420:] = 0
        report = tracker.update(rgb, depth, {"timestamp_s": .2})
        self.assertTrue(report["valid"], report)
        self.assertGreater(report["tracks"]["added"], 0)
        current = tracker.history[-1]
        ids, before, after = np.intersect1d(previous.ids, current.ids, return_indices=True)
        snapshot = tracker.debug_snapshot
        np.testing.assert_array_equal(ids, snapshot["trail_ids"])
        np.testing.assert_array_equal(previous.corners[before, 0], snapshot["trail_segments"][:, 0])
        np.testing.assert_array_equal(current.corners[after, 0], snapshot["trail_segments"][:, 1])

    def test_failed_observations_break_paths_and_age_out_old_trails(self):
        tracker = self.tracker()
        tracker.update(*frame(), {"timestamp_s": 0})
        tracker.update(*frame(2), {"timestamp_s": .1})
        previous = tracker.debug_snapshot["trail_segments"].copy()
        failed = tracker.update(np.zeros((480, 640, 3), np.uint8), frame()[1], {"timestamp_s": .2})
        self.assertFalse(failed["valid"])
        np.testing.assert_array_equal(previous, tracker.debug_snapshot["trail_segments"])
        recovered = tracker.update(*frame(6), {"timestamp_s": .3})
        self.assertTrue(recovered["valid"], recovered)
        # Recovery still has its current flow arrow, but no fabricated trail
        # through the missing observation at .2 seconds.
        np.testing.assert_array_equal(previous, tracker.debug_snapshot["trail_segments"])
        np.testing.assert_array_equal(2, tracker.debug_snapshot["trail_age_frames"])
        for step in range(20):
            tracker.update(*frame(6), {"timestamp_s": .4 + step * .1, "rgb_depth_delta_ms": 21})
        self.assertIsNone(tracker.debug_snapshot)
        self.assertEqual(20, len(tracker._trail_frames))
        self.assertTrue(all(entry is None for entry in tracker._trail_frames))
        segment = tracker.segment
        tracker.update(*frame(8), {"timestamp_s": 3})
        self.assertNotEqual(segment, tracker.segment)
        self.assertEqual(1, tracker.debug_snapshot["trail_frame_count"])
        self.assertEqual(0, len(tracker.debug_snapshot["trail_segments"]))

    def test_trail_toggle_clears_positions_without_resetting_tracking(self):
        tracker = self.tracker()
        tracker.update(*frame(), {"timestamp_s": 0})
        tracked = tracker.update(*frame(2), {"timestamp_s": .1})
        tracker.debug_enabled = False
        self.assertFalse(tracker._trail_frames)
        self.assertIsNone(tracker.debug_snapshot)
        tracker.update(*frame(4), {"timestamp_s": .2})
        self.assertFalse(tracker._trail_frames)
        tracker.debug_enabled = True
        report = tracker.update(*frame(6), {"timestamp_s": .3})
        self.assertEqual(tracked["segment"], report["segment"])
        self.assertEqual(1, tracker.debug_snapshot["trail_frame_count"])
        self.assertEqual(0, len(tracker.debug_snapshot["trail_segments"]))
        tracker.update(*frame(8), {"timestamp_s": .4})
        self.assertGreater(len(tracker.debug_snapshot["trail_segments"]), 100)

    def test_trail_collection_does_not_change_pose_or_field_acceptance(self):
        debug = self.tracker()
        plain = VisualTracker(debug.settings)
        for step in range(6):
            rgb, depth = frame(step * 2)
            cv2.setRNGSeed(step)
            observed = debug.update(rgb, depth, {"timestamp_s": step * .1})
            cv2.setRNGSeed(step)
            expected = plain.update(rgb, depth, {"timestamp_s": step * .1})
            self.assertEqual(expected["valid"], observed["valid"])
            self.assertEqual(expected["tracks"], observed["tracks"])
            self.assertEqual(expected["steps"], observed["steps"])
            np.testing.assert_array_equal(expected["camera_to_local"], observed["camera_to_local"])
        self.assertFalse(plain._trail_frames)

    def test_toggle_and_rejected_frames_never_reuse_stale_vectors(self):
        tracker = self.tracker()
        tracker.update(*frame(), {"timestamp_s": 0})
        report = tracker.update(*frame(2), {"timestamp_s": 0.1})
        tracker.debug_enabled = False
        without = tracker.update(*frame(3), {"timestamp_s": 0.2})
        self.assertIsNone(tracker.debug_snapshot)
        self.assertEqual(report["segment"], without["segment"])
        tracker.debug_enabled = True
        failed = tracker.update(np.zeros((480, 640, 3), np.uint8), frame()[1], {"timestamp_s": 0.3})
        self.assertFalse(failed["valid"])
        self.assertFalse(np.any(tracker.debug_snapshot["status"] == VisualTracker.VERIFIED))
        self.assertEqual(0, tracker.debug_snapshot["tracks"]["added"])
        self.assertEqual(0, tracker.debug_snapshot["tracks"]["retained"])
        late = tracker.update(*frame(4), {"timestamp_s": 0.4, "rgb_depth_delta_ms": 21})
        self.assertFalse(late["valid"])
        self.assertIsNone(tracker.debug_snapshot)
        recovered = tracker.update(*frame(5), {"timestamp_s": 0.5})
        self.assertTrue(recovered["valid"], recovered)
        self.assertEqual(report["segment"], recovered["segment"])
        self.assertAlmostEqual(300, tracker.debug_snapshot["reference_age_ms"])

    def test_depth_rejection_is_visible_and_cannot_become_verified(self):
        tracker = self.tracker()
        tracker.update(*frame(), {"timestamp_s": 0})
        tracker.update(*frame(1), {"timestamp_s": 0.1})
        # Keep the measured centres but make every 3×3 neighbourhood unstable.
        rgb, depth = frame(2)
        depth[:, ::2] = 1000
        # The latest reference is eligible; the older one has expired. Its
        # skip must not erase the diagnostics from the actual attempt.
        report = tracker.update(rgb, depth, {"timestamp_s": 0.8})
        self.assertFalse(report["valid"])
        status = tracker.debug_snapshot["status"]
        self.assertGreater(int((status == VisualTracker.DEPTH_REJECTED).sum()), 100)
        self.assertFalse(np.any(status == VisualTracker.VERIFIED))
        self.assertAlmostEqual(700, tracker.debug_snapshot["reference_age_ms"])

    def test_debug_uses_native_depth_grid_instead_of_scaling_rgb_coordinates(self):
        settings = ScanSettings(sensor_calibration=load_calibration(), color_recovery=True)
        tracker = VisualTracker(settings)
        tracker.debug_enabled = True
        rgb = cv2.resize(frame()[0], (1280, 1024))
        raw = np.full((480, 640), 750, np.uint16)
        tracker.update(rgb, raw, {"timestamp_s": 0})
        self.assertEqual((480, 640, 3), tracker.debug_snapshot["image"].shape)
        self.assertEqual((1024, 1280, 3), rgb.shape)

    def test_worker_debug_toggle_preserves_tracking_generation(self):
        worker = KinectWorker()
        worker.set_tracking_settings(ScanSettings(color_recovery=True))
        request = worker._tracking_request
        worker.set_tracking_debug(True)
        worker.set_tracking_debug(False)
        self.assertEqual(request, worker._tracking_request)


class TrackingDebugGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = ScannerPreferences(QSettings(os.path.join(self.folder.name, "scanner.ini"), QSettings.Format.IniFormat))
        self.patches = [patch.object(main_window, "KinectWorker", NoCamera),
                        patch.object(main_window, "ServerTaskWorker", NoTasks)]
        for item in self.patches:
            item.start()
        self.window = main_window.MainWindow(preferences=self.store)

    def tearDown(self):
        self.window._close_approved = True
        self.window.close()
        self.window.worker.wait(2500)
        self.window.task_worker.wait(2500)
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def snapshot(self):
        tracker = VisualTracker(ScanSettings(color_recovery=True))
        tracker.debug_enabled = True
        tracker.update(*frame(), {"timestamp_s": 0})
        tracker.update(*frame(3), {"timestamp_s": 0.1})
        return tracker.debug_snapshot

    def test_overlay_toggle_is_available_in_capture_and_persists(self):
        self.window._scanning = True
        self.window._refresh_controls()
        self.assertFalse(self.window.settings_group.isEnabled())
        self.assertTrue(self.window.flow_debug_cb.isEnabled())
        self.window.flow_debug_cb.setChecked(True)
        self.window.flow_windows_cb.setChecked(True)
        self.assertTrue(self.store.read("debug/tracking_flow", False))
        self.assertTrue(self.store.read("debug/tracking_windows", False))
        self.assertTrue(self.window.flow_windows_cb.isEnabled())

    def test_portrait_rotates_diagnostic_presentation_after_native_overlay(self):
        self.window.flow_debug_cb.setChecked(True)
        self.window.orientation_combo.setCurrentIndex(self.window.orientation_combo.findData("portrait_right"))
        debug = self.snapshot()
        with patch.object(self.window, "_set_pixmap") as render:
            self.window._on_frame(*frame(), {"_tracking_debug": debug})
        image = render.call_args.args[0]
        self.assertEqual((image.width(), image.height()), (480, 640))
        self.assertEqual(debug["image"].shape, (480, 640, 3))
        self.assertNotIn("_tracking_debug", self.window._last_frame_metadata)

    def test_debug_pixels_and_arrays_do_not_enter_manual_or_automatic_uploads(self):
        self.window.server_client._connected = True
        self.window._scanning = True
        self.window.flow_debug_cb.setChecked(True)
        snapshot = self.snapshot()
        self.assertGreater(len(snapshot["trail_segments"]), 0)
        # Native RGB shape deliberately differs from the debug tracking image.
        rgb = np.full((1024, 1280, 3), 42, np.uint8)
        depth = np.full((480, 640), 750, np.uint16)
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                metadata = {"_tracking_debug": snapshot, "timestamp_s": time.time()}
                self.window._on_frame(rgb, depth, metadata)
                self.assertEqual(640, self.window.view_label._image.width())
                self.assertNotIn("_tracking_debug", self.window._last_frame_metadata)
                self.assertNotIn("_tracking_debug", self.window._capture_selector.frames[-1][4])
                self.assertIn("_tracking_debug", metadata)
                self.window._capture_frame(select_best=automatic)
                capture = self.window.task_worker.frames[-1]
                self.assertNotIn("_tracking_debug", capture["metadata"])
                np.testing.assert_array_equal(rgb, capture["rgb"])
                np.testing.assert_array_equal(depth, capture["depth"])
        self.window.flow_debug_cb.setChecked(False)
        self.assertEqual(1280, self.window.view_label._image.width())
        self.assertIsNone(self.window._last_tracking_debug)

    def test_rejected_pair_clears_previous_vectors_and_explains_timing_failure(self):
        self.window.flow_debug_cb.setChecked(True)
        rgb, depth = frame()
        self.window._on_frame(rgb, depth, {"_tracking_debug": self.snapshot()})
        self.window._on_frame(rgb, depth, {"visual_tracking": {"reason": "RGB/depth timing exceeds 20 ms"}})
        self.assertIsNone(self.window._last_tracking_debug)
        self.assertIn("timing", self.window.flow_debug_status.text())

    def test_painting_windows_and_invalid_flows_preserves_the_input(self):
        snapshot = self.snapshot()
        before = snapshot["image"].copy()
        snapshot["target"][0] = [np.nan, np.inf]
        with_windows = flow_image(snapshot, show_windows=True)
        without_windows = flow_image(snapshot)
        self.assertFalse(with_windows.isNull())
        self.assertNotEqual(with_windows, without_windows)
        np.testing.assert_array_equal(before, snapshot["image"])
        # A seed-only image has no flows or candidate window indices.
        seed = {**snapshot, "source": np.empty((0, 2)), "target": np.empty((0, 2)), "status": np.empty(0, np.uint8)}
        self.assertFalse(flow_image(seed, show_windows=True).isNull())

    def test_trails_fade_with_age_and_invalid_coordinates_are_skipped(self):
        snapshot = {
            "image": np.zeros((64, 64, 3), np.uint8),
            "trail_segments": np.array([[[10, 10], [50, 10]], [[10, 30], [50, 30]],
                                        [[10, 50], [50, 50]], [[np.nan, 5], [np.inf, 5]]]),
            "trail_age_frames": np.array([18, 9, 0, 0], np.uint8),
            "trail_frame_limit": 20,
        }
        image = flow_image(snapshot)
        greens = [image.pixelColor(30, y).green() for y in (10, 30, 50)]
        self.assertGreater(greens[0], 0)
        self.assertLess(greens[0], greens[1])
        self.assertLess(greens[1], greens[2])
        np.testing.assert_array_equal(0, snapshot["image"])

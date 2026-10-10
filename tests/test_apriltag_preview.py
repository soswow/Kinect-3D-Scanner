"""Tag overlays follow detections, preserve image coordinates, and stay local."""

import json
import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from kinect_scanner.capture_process import DEPTH_SHAPE, RGB_SHAPE
from kinect_scanner.gui import apriltag_preview
from kinect_scanner.gui.widgets import numpy_to_qimage
from kinect_scanner.worker import KinectWorker
from shared.apriltag import AprilTagDetector
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker
from tests.test_apriltag import FAMILIES, tag_scene
from tests.test_capture_worker import wait_for
from tests import test_tracking_debug as tracking_debug_tests


def tag_capture(connection, stop_event, rgb_buffer, depth_buffer):
    rgb = np.frombuffer(rgb_buffer, np.uint8).reshape(RGB_SHAPE)
    depth = np.frombuffer(depth_buffer, np.uint16).reshape(DEPTH_SHAPE)
    rgb[:], depth[:] = tag_scene()
    connection.send(("frame", {"rgb_depth_delta_ms": 0, "timestamp_s": 0}))
    connection.recv()
    rgb[:] = depth[:] = 0
    while not stop_event.wait(.05):
        pass


class AprilTagPreviewWorkerTests(unittest.TestCase):
    def test_worker_delivers_tags_from_copied_pair_without_flow_debug(self):
        frames = []
        worker = KinectWorker(capture_target=tag_capture, rgb_mode="rgb_low_res")
        worker.set_tracking_settings(ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES))
        worker.frame_pair_ready.connect(lambda *args: frames.append(args), Qt.ConnectionType.DirectConnection)
        try:
            worker.start()
            self.assertTrue(wait_for(lambda: bool(frames)))
        finally:
            worker.stop()
            self.assertTrue(worker.wait(2500))
        rgb, depth, metadata = frames[0]
        preview = metadata["_apriltag_preview"]
        self.assertEqual(2, len(preview["detections"]))
        self.assertTrue(all(tag["usable"] for tag in preview["detections"]))
        self.assertNotIn("_tracking_debug", metadata)
        np.testing.assert_array_equal(rgb, preview["image"])
        np.testing.assert_array_equal(rgb, tag_scene()[0])
        np.testing.assert_array_equal(depth, tag_scene()[1])
        json.dumps(metadata["visual_tracking"])
        json.dumps(metadata["motion_history"])


class AprilTagPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tracker(self):
        return VisualTracker(ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES))

    def test_every_detection_has_outline_data_including_unusable_and_ambiguous_tags(self):
        detector = AprilTagDetector(FAMILIES)
        rgb, depth = tag_scene()
        for kwargs, image, raw in (({}, rgb, depth), ({}, rgb, np.zeros_like(depth)),
                                   ({"synchronized": False}, rgb, depth),
                                   ({}, tag_scene(families=(FAMILIES[0],) * 2)[0], depth)):
            with self.subTest(kwargs=kwargs, missing_depth=not raw.any()):
                frame = detector.detect(image, raw, ScanSettings().camera, **kwargs)
                preview = frame.preview()
                self.assertEqual(2, len(preview))
                self.assertTrue(all(tag["id"] == 7 for tag in preview))
                self.assertTrue(all(set(tag) == {"id", "corners", "usable"} for tag in preview))
                self.assertEqual(len(frame.tags), sum(tag["usable"] for tag in preview))
                preview[0]["corners"][:] = 0
                self.assertTrue(frame.preview()[0]["corners"].any())

    def test_preview_is_available_without_flow_debug_and_excluded_from_pose_report(self):
        tracker = self.tracker()
        rgb, depth = tag_scene()
        before = rgb.copy()
        report = tracker.update(rgb, depth, {"timestamp_s": 0})
        self.assertIsNone(tracker.debug_snapshot)
        self.assertEqual(2, len(tracker.apriltag_snapshot["detections"]))
        self.assertNotIn("detections", report["apriltags"])
        json.dumps(report)
        tracker.apriltag_snapshot["image"][:] = 0
        np.testing.assert_array_equal(before, rgb)

    def test_timing_failure_and_expired_chain_keep_only_current_detections(self):
        tracker = self.tracker()
        tracker.update(*tag_scene(), {"timestamp_s": 0})
        tracker.update(*tag_scene(shift=6), {"timestamp_s": .1, "rgb_depth_delta_ms": 21})
        self.assertEqual(2, len(tracker.apriltag_snapshot["detections"]))
        self.assertFalse(any(tag["usable"] for tag in tracker.apriltag_snapshot["detections"]))
        tracker.update(*tag_scene(shift=10), {"timestamp_s": 2})
        self.assertEqual(2, len(tracker.apriltag_snapshot["detections"]))
        tracker.update(np.full((480, 640, 3), 255, np.uint8), tag_scene()[1], {"timestamp_s": 2.1})
        self.assertEqual([], tracker.apriltag_snapshot["detections"])
        tracker.reset()
        self.assertIsNone(tracker.apriltag_snapshot)
        disabled = VisualTracker(ScanSettings())
        disabled.update(*tag_scene(), {"timestamp_s": 0})
        self.assertIsNone(disabled.apriltag_snapshot)

    def test_painter_draws_color_coded_outlines_and_numeric_ids_without_mutating_source(self):
        rgb = np.zeros((240, 480, 3), np.uint8)
        image = numpy_to_qimage(rgb)
        tags = [{"id": 7, "corners": np.array([[80, 100], [180, 100], [180, 200], [80, 200]]), "usable": True},
                {"id": 23, "corners": np.array([[300, 100], [400, 100], [400, 200], [300, 200]]), "usable": False}]
        with patch.object(apriltag_preview.QPainter, "drawText") as text:
            result = apriltag_preview.paint_apriltags(image, tags)
        self.assertEqual(["7", "23"], [call.args[-1] for call in text.call_args_list])
        self.assertGreater(result.pixelColor(130, 100).green(), result.pixelColor(130, 100).red())
        self.assertGreater(result.pixelColor(350, 100).red(), result.pixelColor(350, 100).blue())
        self.assertEqual(0, image.pixelColor(130, 100).green())
        np.testing.assert_array_equal(0, rgb)

    def test_invalid_coordinates_are_skipped_and_edge_labels_stay_inside_image(self):
        image = numpy_to_qimage(np.zeros((80, 80, 3), np.uint8))
        bad = [{"id": 1, "corners": np.full((4, 2), np.nan), "usable": True}]
        self.assertEqual(image, apriltag_preview.paint_apriltags(image, bad))
        edge = [{"id": 123, "corners": np.array([[68, 0], [79, 0], [79, 20], [68, 20]]), "usable": True}]
        with patch.object(apriltag_preview.QPainter, "drawText") as text:
            apriltag_preview.paint_apriltags(image, edge)
        rect = text.call_args.args[0]
        self.assertGreaterEqual(rect.top(), 0)
        self.assertLessEqual(rect.right(), image.width())


class AprilTagPreviewGuiTests(unittest.TestCase):
    setUpClass = classmethod(tracking_debug_tests.TrackingDebugGuiTests.setUpClass.__func__)
    setUp = tracking_debug_tests.TrackingDebugGuiTests.setUp
    tearDown = tracking_debug_tests.TrackingDebugGuiTests.tearDown

    def snapshot(self):
        tracker = VisualTracker(ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES))
        tracker.update(*tag_scene(), {"timestamp_s": 0})
        return tracker.apriltag_snapshot

    def test_overlay_appears_without_flow_debug_and_preview_arrays_never_enter_captures(self):
        self.assertFalse(self.window.flow_debug_cb.isChecked())
        self.window.server_client._connected = True
        self.window._scanning = True
        rgb = np.full((1024, 1280, 3), 42, np.uint8)
        depth = np.full((480, 640), 750, np.uint16)
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                metadata = {"_apriltag_preview": self.snapshot(), "timestamp_s": time.time()}
                self.window._on_frame(rgb, depth, metadata)
                self.assertEqual(640, self.window.view_label._image.width())
                self.assertIn("AprilTags", self.window.camera_title.text())
                self.assertNotIn("_apriltag_preview", self.window._last_frame_metadata)
                self.assertNotIn("_apriltag_preview", self.window._capture_selector.frames[-1][4])
                self.assertIn("_apriltag_preview", metadata)
                self.window._capture_frame(select_best=automatic)
                capture = self.window.task_worker.frames[-1]
                self.assertNotIn("_apriltag_preview", capture["metadata"])
                np.testing.assert_array_equal(rgb, capture["rgb"])
                np.testing.assert_array_equal(depth, capture["depth"])

    def test_portrait_rotation_applies_to_tags_and_image_together(self):
        self.window.orientation_combo.setCurrentIndex(self.window.orientation_combo.findData("portrait_right"))
        snapshot = self.snapshot()
        with patch.object(self.window, "_set_pixmap") as render:
            self.window._on_frame(*tag_scene(), {"_apriltag_preview": snapshot})
        image = render.call_args.args[0]
        self.assertEqual((480, 640), (image.width(), image.height()))

    def test_flow_and_tag_overlays_combine_and_disabling_flow_keeps_tags(self):
        tracker = VisualTracker(ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES))
        tracker.debug_enabled = True
        tracker.update(*tag_scene(), {"timestamp_s": 0})
        self.window.flow_debug_cb.setChecked(True)
        self.window._on_frame(*tag_scene(), {"_apriltag_preview": tracker.apriltag_snapshot,
            "_tracking_debug": tracker.debug_snapshot})
        before = tracker.debug_snapshot["image"].copy()
        self.assertIn("Tracking RGB", self.window.camera_title.text())
        self.window.flow_debug_cb.setChecked(False)
        self.assertIn("AprilTags", self.window.camera_title.text())
        self.assertIsNotNone(self.window._last_apriltag_preview)
        np.testing.assert_array_equal(before, tracker.debug_snapshot["image"])

    def test_new_frame_or_tracking_shutdown_clears_old_tag_overlay(self):
        self.window._on_frame(*tag_scene(), {"_apriltag_preview": self.snapshot()})
        self.window._on_frame(*tag_scene(), {})
        self.assertIsNone(self.window._last_apriltag_preview)
        self.assertIn("Color", self.window.camera_title.text())
        self.window._session_settings = ScanSettings(apriltag_tracking=True).to_dict()
        self.window._scanning = True
        self.window._configure_camera_tracking()
        self.window._on_frame(*tag_scene(), {"_apriltag_preview": self.snapshot()})
        self.window._scanning = False
        self.window._configure_camera_tracking()
        self.assertIsNone(self.window._last_apriltag_preview)


if __name__ == "__main__":
    unittest.main()

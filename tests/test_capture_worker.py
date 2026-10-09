"""Regression tests for driver stalls, retries and shared-buffer ownership."""

import signal
import threading
import time
import unittest
import weakref
from unittest.mock import patch

import cv2
import numpy as np
from PyQt6.QtCore import Qt

from kinect_scanner.capture_process import DEPTH_SHAPE, RGB_SHAPE
from kinect_scanner.worker import KinectWorker
from shared.settings import ScanSettings


def stalled_capture(connection, stop_event, rgb_buffer, depth_buffer):
    # Model a native driver call that ignores both cancellation and SIGTERM.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send(("error", "driver entered uninterruptible call"))
    while True:
        time.sleep(0.1)


def silent_capture(connection, stop_event, rgb_buffer, depth_buffer):
    while True:
        time.sleep(0.1)


def sensor_only_capture(connection, stop_event, rgb_buffer, depth_buffer):
    connection.send(("accelerometer_poll", True))
    while not stop_event.wait(.01):
        connection.send(("sensor_status", {"complete": True}))


def one_frame_capture(connection, stop_event, rgb_buffer, depth_buffer):
    rgb = np.frombuffer(rgb_buffer, np.uint8).reshape(RGB_SHAPE)
    depth = np.frombuffer(depth_buffer, np.uint16).reshape(DEPTH_SHAPE)
    rgb[:] = 42
    depth[:] = 1234
    connection.send(("frame", {"rgb_depth_delta_ms": 12.0}))
    connection.recv()  # Parent must copy before permitting overwrite.
    rgb[:] = 99
    depth[:] = 9999
    while not stop_event.wait(0.05):
        pass


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class CaptureWorkerTests(unittest.TestCase):
    def test_finish_checkpoints_recording_before_stop_without_delivering_more_images(self):
        worker = KinectWorker()
        worker.coalesce_frames = True
        entered, release = threading.Event(), threading.Event()
        def flush(path, stop=False):
            self.assertEqual(path, "/recording")
            self.assertTrue(stop)
            entered.set()
            release.wait(2)
            return {"complete": True, "root": path}
        image = np.zeros((2, 2, 3), np.uint8)
        worker._deliver_frame(image, image, {})
        with patch.object(worker, "flush_sensor_recording", side_effect=flush):
            worker.finish_capture("/recording")
            try:
                self.assertTrue(entered.wait(1))
                self.assertFalse(worker._stop_event.is_set())
                worker._deliver_frame(image, image, {})
                self.assertIsNone(worker.take_latest_frame())
            finally:
                release.set()
                worker._finish_thread.join(2)
        self.assertTrue(worker._stop_event.is_set())

    def test_finish_stops_even_if_sensor_checkpoint_fails(self):
        worker = self.start_worker(silent_capture)
        with patch.object(worker, "flush_sensor_recording", side_effect=OSError("disk full")):
            worker.finish_capture("/recording")
            worker._finish_thread.join(2)
        self.assertTrue(worker.wait(2500))
        self.assertTrue(worker._stop_event.is_set())

    def test_stalled_gui_retains_only_latest_frame_and_one_notification(self):
        worker = KinectWorker()
        worker.coalesce_frames = True
        notifications, legacy, references = [], [], []
        worker.frame_available.connect(lambda: notifications.append(True), Qt.ConnectionType.DirectConnection)
        worker.frame_pair_ready.connect(lambda *args: legacy.append(args), Qt.ConnectionType.DirectConnection)
        for index in range(2000):
            rgb = np.full((10, 10, 3), index % 256, np.uint8)
            depth = np.zeros((10, 10), np.uint16)
            references.append(weakref.ref(rgb))
            worker._deliver_frame(rgb, depth, {"frame_id": index})
        self.assertEqual(len(notifications), 1)
        self.assertEqual(legacy, [])
        self.assertEqual(sum(ref() is not None for ref in references), 1)
        self.assertEqual(worker.preview_frames_replaced, 1999)
        frame = worker.take_latest_frame()
        self.assertEqual(frame[2]["frame_id"], 1999)
        self.assertIsNone(worker.take_latest_frame())
        worker._deliver_frame(rgb, depth, {"frame_id": 2000})
        self.assertEqual(len(notifications), 2)

    def start_worker(self, target, **kwargs):
        worker = KinectWorker(capture_target=target, rgb_mode="rgb_low_res", **kwargs)
        self.addCleanup(self.stop_worker, worker)
        worker.start()
        return worker

    def stop_worker(self, worker):
        worker.stop()
        self.assertTrue(worker.wait(2500), "Camera shutdown must be bounded")

    def test_stop_while_native_capture_is_stalled(self):
        worker = self.start_worker(silent_capture)
        time.sleep(0.5)
        self.stop_worker(worker)

    def test_force_kill_driver_that_ignores_termination(self):
        errors = []
        worker = KinectWorker(capture_target=stalled_capture, retry_delay=10)
        self.addCleanup(self.stop_worker, worker)
        worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(wait_for(lambda: bool(errors)))
        self.stop_worker(worker)

    def test_startup_timeout_reports_failure_and_retries(self):
        errors = []
        worker = KinectWorker(
            capture_target=silent_capture, startup_timeout=0.6, retry_delay=0.1
        )
        self.addCleanup(self.stop_worker, worker)
        worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(wait_for(lambda: len(errors) >= 2))
        self.assertIn("stopped delivering", errors[0])

    def test_sensor_traffic_cannot_hide_image_stall_and_disables_hung_poll(self):
        errors = []
        worker = KinectWorker(capture_target=sensor_only_capture, startup_timeout=.6, retry_delay=10)
        self.addCleanup(self.stop_worker, worker)
        worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(wait_for(lambda: bool(errors) and not worker._accelerometer_enabled))
        self.assertIn("stopped delivering", errors[0])

    def test_copy_frame_before_child_reuses_buffer_and_detect_stream_stall(self):
        frames, errors = [], []
        worker = KinectWorker(
            capture_target=one_frame_capture,
            rgb_mode="rgb_low_res",
            frame_timeout=0.3,
            retry_delay=10,
        )
        self.addCleanup(self.stop_worker, worker)
        worker.frame_pair_ready.connect(
            lambda *args: frames.append(args), Qt.ConnectionType.DirectConnection
        )
        worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(wait_for(lambda: bool(frames) and bool(errors)))
        rgb, depth, metadata = frames[0]
        self.assertTrue(np.all(rgb == 42))
        self.assertTrue(np.all(depth == 1234))
        self.assertEqual(metadata["frame_id"], 1)
        self.assertEqual(metadata["rgb_depth_delta_ms"], 12)

    def test_stop_before_start_never_opens_camera(self):
        worker = KinectWorker(capture_target=stalled_capture)
        worker.stop()
        worker.start()
        self.assertTrue(worker.wait(1000))

    def test_visual_failure_does_not_restart_or_discard_camera_stream(self):
        frames, errors = [], []
        worker = KinectWorker(capture_target=one_frame_capture, rgb_mode="rgb_low_res")
        self.addCleanup(self.stop_worker, worker)
        worker.set_tracking_settings(ScanSettings(color_recovery=True, live_reconstruction=True))
        worker.frame_pair_ready.connect(lambda *args: frames.append(args), Qt.ConnectionType.DirectConnection)
        worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
        with patch("kinect_scanner.worker.VisualTracker") as factory:
            factory.return_value.update.side_effect = cv2.error("No usable optical flow")
            worker.start()
            self.assertTrue(wait_for(lambda: bool(frames)))
            self.stop_worker(worker)
            factory.return_value.reset.assert_called_once()
        self.assertFalse(frames[0][2]["visual_tracking"]["valid"])
        self.assertTrue(np.all(frames[0][0] == 42))
        self.assertFalse(errors)

    def test_debug_snapshot_uses_the_copied_pair_after_driver_buffer_reuse(self):
        frames = []
        worker = KinectWorker(capture_target=one_frame_capture, rgb_mode="rgb_low_res")
        self.addCleanup(self.stop_worker, worker)
        worker.set_tracking_settings(ScanSettings(color_recovery=True, live_reconstruction=True))
        worker.set_tracking_debug(True)
        worker.frame_pair_ready.connect(lambda *args: frames.append(args), Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(wait_for(lambda: bool(frames)))
        self.stop_worker(worker)
        rgb, depth, metadata = frames[0]
        np.testing.assert_array_equal(metadata["_tracking_debug"]["image"], rgb)
        self.assertTrue(np.all(rgb == 42))
        self.assertTrue(np.all(depth == 1234))
        self.assertFalse(metadata["visual_tracking"]["valid"])

"""Exercise the production event loop with a simulated driver and backpressure."""

import json
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kinect_scanner.capture_process import capture_frames
from kinect_scanner.rgb_exposure import ExposureControlUnavailable
from shared.inertial import G
from shared.sensor_recording import journal_snapshot
from shared.settings import ScanSettings
from shared.protocol import pack_frame, unpack_frame_with_metadata
from kinect_scanner.worker import KinectWorker
from PyQt6.QtCore import Qt


class SimulatedDriver:
    RESOLUTION_MEDIUM = 1
    RESOLUTION_HIGH = 2
    DEPTH_11BIT = 1
    VIDEO_IR_10BIT = 2
    VIDEO_RGB = 3

    def __init__(self, stop):
        self.stop = stop
        self.iteration = 0
        self.mode = self.VIDEO_IR_10BIT
        self.maximum_iterations = 55
        self.depth = np.full((480, 640), 500, np.uint16)
        self.rgb = np.zeros((480, 640, 3), np.uint8)

    def init(self): return object()
    def num_devices(self, context): return 1
    def open_device(self, context, index): return object()
    def set_depth_mode(self, *args): return 0
    def set_video_mode(self, device, resolution, mode):
        self.mode = mode
        return 0
    def set_depth_callback(self, device, callback): self.depth_callback = callback
    def set_video_callback(self, device, callback): self.video_callback = callback
    def start_depth(self, *args): return 0
    def stop_depth(self, *args): return 0
    def start_video(self, *args): return 0
    def stop_video(self, *args): return 0
    def close_device(self, *args): return 0
    def shutdown(self, *args): return 0
    def update_tilt_state(self, device): return 10
    def get_tilt_state(self, device):
        return SimpleNamespace(accelerometer_x=0, accelerometer_y=819, accelerometer_z=0)
    def get_mks_accel(self, state): return (0, G, 0)

    def process_events_timeout(self, context):
        self.iteration += 1
        stamp = self.iteration * 2_000_000
        self.depth_callback(None, self.depth, stamp)
        self.rgb[:] = self.iteration % 256
        self.video_callback(None, self.rgb if self.mode == self.VIDEO_RGB else self.depth, stamp)
        if self.iteration >= self.maximum_iterations:
            self.stop.set()
        time.sleep(.01)
        return 0

    def process_events(self, context):
        return self.process_events_timeout(context)


def simulated_child(*args):
    """Spawn-safe fake USB surface; keep the real capture loop and control path."""
    driver = SimulatedDriver(args[1])
    driver.maximum_iterations = 10_000
    with patch.dict(sys.modules, {"freenect": driver}), patch(
        "kinect_scanner.capture_process.RGBExposureControl",
        side_effect=ExposureControlUnavailable("Simulated default auto exposure")):
        capture_frames(*args)


class CaptureLoopTests(unittest.TestCase):
    def test_supervisor_save_barrier_drains_real_child_journal(self):
        self.check_supervisor_save_barrier(full=True)

    def test_default_child_keeps_live_images_and_acceleration_without_recording_camera_arrays(self):
        self.check_supervisor_save_barrier(full=False)

    def check_supervisor_save_barrier(self, full):
        with tempfile.TemporaryDirectory() as folder, patch("kinect_scanner.worker.capture_frames", simulated_child):
            worker = KinectWorker(capture_target=simulated_child, rgb_mode="rgb_low_res", retry_delay=10)
            worker.set_sensor_recording(folder, ScanSettings(record_full_camera_streams=full).to_dict())
            frames, errors = [], []
            worker.frame_pair_ready.connect(lambda *args: frames.append(args), Qt.ConnectionType.DirectConnection)
            worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
            worker.start()
            try:
                deadline = time.monotonic() + 8
                while not frames and not errors and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(frames, errors)
                worker.set_sensor_recording(None)
                snapshot = worker.flush_sensor_recording(folder, stop=True)
                self.assertTrue(snapshot["complete"], snapshot)
                status = snapshot["segments"][0]["status"]
                self.assertTrue(status["closed"])
                if full:
                    self.assertGreater(status["counts"]["rgb"], 0)
                else:
                    self.assertEqual(status["counts"]["rgb"], 0)
                    self.assertEqual(status["counts"]["depth"], 0)
                    self.assertEqual(list(Path(folder).rglob("*.npy")), [])
                self.assertGreater(status["counts"]["accelerometer"], 0)
                time.sleep(.1)
                self.assertEqual(journal_snapshot(folder)["segments"][0]["status"]["index_bytes"], status["index_bytes"])
            finally:
                worker.stop()
                self.assertTrue(worker.wait(2500))
            self.assertFalse(errors)

    def test_full_streams_survive_slow_consumer_and_native_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            stop = threading.Event()
            driver = SimulatedDriver(stop)
            controls = queue.Queue()
            controls.put(("record", {"path": folder, "settings": ScanSettings(record_full_camera_streams=True).to_dict()}))

            class Connection:
                def __init__(self): self.messages = []
                def send(self, message): self.messages.append(message)
                def poll(self): return driver.iteration % 3 == 0
                def recv(self): return "copied"
                def close(self): pass

            connection = Connection()
            rgb_buffer = bytearray(480 * 640 * 3)
            depth_buffer = bytearray(480 * 640 * 2)
            with patch.dict(sys.modules, {"freenect": driver}), patch(
                "kinect_scanner.capture_process.RGBExposureControl",
                side_effect=ExposureControlUnavailable("Simulated default auto exposure")):
                capture_frames(connection, stop, rgb_buffer, depth_buffer, False, controls=controls)
            errors = [value for kind, value in connection.messages if kind == "error"]
            self.assertEqual(errors, [])
            snapshot = journal_snapshot(folder)
            self.assertTrue(snapshot["complete"], snapshot)
            counts = snapshot["segments"][0]["status"]["counts"]
            published = [value for kind, value in connection.messages if kind == "frame"]
            self.assertGreater(counts["rgb"], len(published))
            self.assertGreater(counts["depth"], counts["rgb"])
            self.assertGreater(counts["accelerometer"], 4)
            segment = Path(folder) / snapshot["segments"][0]["generation"]
            rows = [json.loads(row) for row in (segment / "rgb.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["dtype"] == "uint8" for row in rows))
            raw = np.load(segment / rows[0]["image"], allow_pickle=False)
            self.assertTrue(np.all(raw == 31), "Borrowed driver buffer must be copied")
            rgb = np.frombuffer(rgb_buffer, np.uint8).reshape(480, 640, 3)
            depth = np.frombuffer(depth_buffer, np.uint16).reshape(480, 640)
            packet = pack_frame(rgb, depth, published[-1])
            restored_rgb, restored_depth, restored_metadata = unpack_frame_with_metadata(packet)
            np.testing.assert_array_equal(restored_rgb, rgb)
            np.testing.assert_array_equal(restored_depth, depth)
            self.assertEqual(restored_metadata["sensor_recording_segment"], segment.name)
            self.assertIn("accelerometer", restored_metadata)
            self.assertIn("orientation_accelerometer", restored_metadata)
            self.assertEqual(restored_metadata["orientation_host_monotonic_s"], restored_metadata["captured_monotonic_s"])

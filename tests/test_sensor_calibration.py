"""Native calibration, full-resolution transport and measured registration checks."""

import copy
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("KINECT_BLOCK_COUNT", "1000")
import cv2
import numpy as np

from shared.calibration import camera_matrix, prepare_rgbd, raw_depth_to_mm
from shared.protocol import (
    pack_frame,
    pack_frames,
    unpack_frame_with_metadata,
    unpack_frames,
)
from shared.sensor_calibration import (
    DEFAULT_CALIBRATION_PATH,
    SensorCalibration,
    load_calibration,
)
from shared.settings import ScanSettings


class SensorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = load_calibration()
        self.settings = ScanSettings(
            sensor_calibration=self.calibration, filter_depth=False
        )

    def test_publication_preserved_in_portable_settings_and_grid_applied_once(self):
        source = json.loads(DEFAULT_CALIBRATION_PATH.read_text())
        self.assertEqual(source, self.calibration.to_dict())
        restored = ScanSettings.from_dict(
            json.loads(json.dumps(self.settings.to_dict()))
        )
        self.assertEqual(self.settings, restored)
        self.assertAlmostEqual(
            self.settings.camera.cx,
            self.calibration.ir.cx + self.calibration.shift_px[0],
        )
        self.assertEqual(
            (1280, 1024),
            (self.settings.rgb_camera.width, self.settings.rgb_camera.height),
        )
        low = ScanSettings(sensor_calibration=self.calibration, rgb_mode="rgb_low_res")
        self.assertEqual(self.calibration.rgb_low_res, low.rgb_camera)
        self.assertNotEqual(low.rgb_camera.cx * 2, self.settings.rgb_camera.cx)

    def test_guarded_metric_conversion_matches_all_published_lut_entries(self):
        raw = np.arange(2048, dtype=np.uint16)
        metric = raw_depth_to_mm(raw, self.calibration)
        expected = np.fromfile(
            DEFAULT_CALIBRATION_PATH.with_name("raw-to-mm.bin"), dtype="<u2"
        )
        np.testing.assert_array_equal(np.rint(metric).astype(np.uint16), expected)
        self.assertGreater(metric[0], 0)
        self.assertEqual(metric[2047], 0)
        self.assertAlmostEqual(metric[750], 1026.991, delta=0.001)
        self.assertEqual(
            0, raw_depth_to_mm(np.array([65535], np.uint16), self.calibration)[0]
        )

    def test_invalid_geometry_and_encoding_are_rejected(self):
        source = self.calibration.to_dict()
        changes = [
            (lambda x: x.update(schema_version=2)),
            (lambda x: x["ir_to_rgb"].update(R=np.zeros((3, 3)).tolist())),
            (lambda x: x["ir_to_depth"].update(ir_to_depth_shift_px=[0, 0])),
            (lambda x: x["raw_depth_to_mm"]["input"].update(mode="FREENECT_DEPTH_MM")),
            (lambda x: x["intrinsics"]["rgb_high_res"].update(image_size=[640, 480])),
            (lambda x: x["raw_depth_to_mm"].update(scale=float("nan"))),
        ]
        for change in changes:
            with self.subTest(change=change):
                document = copy.deepcopy(source)
                change(document)
                with self.assertRaises(ValueError):
                    SensorCalibration.from_dict(document)

    def test_native_geometry_and_color_follow_measured_pose_and_rgb_lens(self):
        # Encode native high-resolution pixel location in R,G for an independent
        # projection check. This exercises both distortion and the physical pose.
        rgb = np.zeros((1024, 1280, 3), np.uint8)
        rgb[..., 0] = np.arange(1280)[None, :] // 8
        rgb[..., 1] = np.arange(1024)[:, None] // 8
        raw = np.full((480, 640), 750, np.uint16)
        raw[150:180, 150:180] = 2047
        color, depth = prepare_rgbd(rgb, raw, self.settings)
        self.assertEqual((480, 640, 3), color.shape)
        self.assertEqual({0, 1027}, set(np.unique(depth)))
        y, x = 260, 320
        c = self.settings.camera
        z = raw_depth_to_mm(np.array([750], np.uint16), self.calibration)[0]
        point = z * np.array([(x - c.cx) / c.fx, (y - c.cy) / c.fy, 1])
        rvec, _ = cv2.Rodrigues(np.asarray(self.calibration.rotation))
        pixel, _ = cv2.projectPoints(
            point[None, :],
            rvec,
            np.asarray(self.calibration.translation_mm),
            camera_matrix(self.settings.rgb_camera),
            np.asarray(self.settings.rgb_camera.distortion),
        )
        u, v = pixel.ravel()
        np.testing.assert_allclose(color[y, x, :2], [u // 8, v // 8], atol=1)
        self.assertEqual(750, raw[260, 320])
        self.assertEqual(2047, raw[160, 160])

    def test_high_resolution_packets_and_batches_preserve_native_observations(self):
        rng = np.random.default_rng(9)
        rgb = rng.integers(0, 256, (1024, 1280, 3), dtype=np.uint8)
        raw = rng.integers(0, 2048, (480, 640), dtype=np.uint16)
        metadata = {"depth_encoding": "raw_11bit", "frame_id": 3}
        packed = pack_frame(rgb, raw, metadata)
        self.assertEqual(b"RGB3", packed[:4])
        restored = unpack_frame_with_metadata(packed)
        np.testing.assert_array_equal(restored[0], rgb)
        np.testing.assert_array_equal(restored[1], raw)
        self.assertEqual("raw_11bit", restored[2]["depth_encoding"])
        batch = unpack_frames(pack_frames([(rgb, raw, metadata)]), with_metadata=True)
        np.testing.assert_array_equal(batch[0][0], rgb)
        # A V3 declaration cannot expand bounded decompression arbitrarily.
        corrupt = packed.replace(b"1024,1280,3", b"9999,9999,3", 1)
        with self.assertRaises(ValueError):
            unpack_frame_with_metadata(corrupt)

    def test_engine_uses_metric_native_geometry_and_exports_replayable_raw_session(
        self,
    ):
        from scanner_server.engine import ScanEngine
        from scanner_server.session import export_session
        from scripts.replay_scan import load_dataset

        engine = ScanEngine()
        engine.reset(settings=self.settings)
        rgb = np.full((1024, 1280, 3), 120, np.uint8)
        raw = np.full((480, 640), 750, np.uint16)
        metadata = {"frame_id": 1, "timestamp_s": 1, "depth_encoding": "raw_11bit"}
        with self.assertRaises(ValueError):
            engine.store_frame(rgb, raw, {"depth_encoding": "registered_mm"})
        with self.assertRaises(ValueError):
            engine.store_frame(rgb[:480, :640], raw)
        engine.store_frame(rgb, raw, metadata)
        engine.process_frames()
        self.assertEqual(1, engine.frame_count)
        self.assertAlmostEqual(
            np.median(np.asarray(engine.model_pcd.points)[:, 2]), 1.027, delta=0.005
        )
        np.testing.assert_array_equal(engine.raw_frames[0][1], raw)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "session.zip"
            export_session(engine, archive)
            with zipfile.ZipFile(archive) as zf:
                manifest = json.loads(zf.read("manifest.json"))
                self.assertEqual("raw_11bit_disparity", manifest["depth_unit"])
                zf.extractall(Path(tmp) / "replay")
            settings, frames = load_dataset(
                "recording", Path(tmp) / "replay", stride=1, limit=1
            )
            self.assertEqual(self.settings, settings)
            np.testing.assert_array_equal(frames[0][1], raw)
            np.testing.assert_array_equal(frames[0][0], rgb)

    def test_texture_projection_samples_native_high_resolution_pixels(self):
        from scanner_server.texturing import _project

        point = np.array([[0.05, 0.02, 1.2]])
        normals = np.array([[0.0, 0.0, -1.0]])
        rgb = np.zeros((1024, 1280, 3), np.uint8)
        rgb[..., 0] = (np.arange(1280) % 251)[None, :]
        rgb[..., 1] = (np.arange(1024) % 251)[:, None]
        depth = np.full((480, 640), 1.2, np.float32)
        values, weights = _project(
            point,
            normals,
            rgb,
            depth,
            np.eye(4),
            self.settings.camera,
            0.5,
            4,
            self.calibration,
            self.settings.rgb_camera,
        )
        rvec, _ = cv2.Rodrigues(np.asarray(self.calibration.rotation))
        pixel, _ = cv2.projectPoints(
            point * 1000,
            rvec,
            np.asarray(self.calibration.translation_mm),
            camera_matrix(self.settings.rgb_camera),
            np.asarray(self.settings.rgb_camera.distortion),
        )
        u, v = pixel.ravel()
        self.assertGreater(weights[0], 0)
        np.testing.assert_allclose(values[0, :2] * 255, [u % 251, v % 251], atol=0.01)

    def test_gui_requires_complete_calibration_and_switches_rgb_modes(self):
        from dataclasses import asdict

        from PyQt6.QtCore import QThread, pyqtSignal
        from PyQt6.QtWidgets import QApplication

        from kinect_scanner.gui import main_window
        from shared.settings import CameraCalibration

        class NoCamera(QThread):
            frame_ready = pyqtSignal(np.ndarray, np.ndarray)
            error_occurred = pyqtSignal(str)

            def __init__(self, **kwargs):
                super().__init__()
                self.capture_settings = kwargs

            def run(self):
                pass

            def stop(self):
                pass

        app = QApplication.instance() or QApplication([])
        with patch.object(main_window, "KinectWorker", NoCamera):
            window = main_window.MainWindow()
            try:
                self.assertEqual(self.calibration, window._sensor_calibration)
                self.assertEqual(
                    "rgb_high_res", window.worker.capture_settings["rgb_mode"]
                )
                rgb = np.full((1024, 1280, 3), 120, np.uint8)
                raw = np.full((480, 640), 750, np.uint16)
                window._show_scanner(rgb, raw)
                self.assertFalse(window.view_label.pixmap().isNull())
                original = window.worker
                with tempfile.TemporaryDirectory() as tmp:
                    profile = Path(tmp) / "profile.json"
                    # Invalid data leaves both camera and acquisition untouched.
                    profile.write_text('{"intrinsics": {}}')
                    with patch.object(
                        main_window.QFileDialog,
                        "getOpenFileName",
                        return_value=(str(profile), ""),
                    ):
                        window._load_calibration()
                    self.assertIs(original, window.worker)
                    self.assertEqual(self.calibration, window._sensor_calibration)
                    self.assertIn(
                        "Invalid calibration", window.scan_status_label.text()
                    )
                    profile.write_text(json.dumps(asdict(CameraCalibration())))
                    with patch.object(
                        main_window.QFileDialog,
                        "getOpenFileName",
                        return_value=(str(profile), ""),
                    ):
                        window._load_calibration()
                    self.assertIs(original, window.worker)
                    self.assertEqual(self.calibration, window._sensor_calibration)
                    self.assertIn(
                        "Invalid calibration", window.scan_status_label.text()
                    )
                    with self.assertRaises(ValueError):
                        load_calibration(profile)
                    with patch.object(
                        main_window.QFileDialog,
                        "getOpenFileName",
                        return_value=(str(DEFAULT_CALIBRATION_PATH), ""),
                    ):
                        window._load_calibration()
                    self.assertEqual(self.calibration, window._sensor_calibration)
                    self.assertTrue(window.rgb_mode_combo.isEnabled())
                    window.rgb_mode_combo.setCurrentIndex(0)
                    self.assertEqual(
                        "rgb_high_res", window.worker.capture_settings["rgb_mode"]
                    )
                    window.rgb_mode_combo.setCurrentIndex(1)
                    self.assertEqual(
                        "rgb_low_res", window.worker.capture_settings["rgb_mode"]
                    )
                    self.assertIsNone(window._last_rgb)
            finally:
                window.close()
                self.assertTrue(window.worker.wait(2500))
                self.assertTrue(window.task_worker.wait(2500))
                app.processEvents()

    def test_capture_selects_high_rgb_and_raw_depth_after_ir_warmup(self):
        from kinect_scanner.capture_process import capture_frames

        driver = SimpleNamespace(
            RESOLUTION_MEDIUM=1,
            RESOLUTION_HIGH=2,
            DEPTH_11BIT=0,
            VIDEO_RGB=0,
            VIDEO_IR_10BIT=2,
        )
        modes, packets = [], []
        driver.init = lambda: object()
        driver.num_devices = lambda ctx: 1
        driver.open_device = lambda ctx, index: object()
        driver.set_depth_mode = lambda *args: modes.append(("depth", args[1:])) or 0

        def video_mode(dev, resolution, fmt):
            driver.video_mode = fmt
            modes.append(("video", (resolution, fmt)))
            return 0

        driver.set_video_mode = video_mode
        driver.set_depth_callback = lambda dev, callback: setattr(
            driver, "depth_callback", callback
        )
        driver.set_video_callback = lambda dev, callback: setattr(
            driver, "video_callback", callback
        )
        for name in (
            "start_depth",
            "start_video",
            "stop_video",
            "stop_depth",
            "close_device",
            "shutdown",
        ):
            setattr(driver, name, lambda *args: 0)
        ticks = 0

        def events(ctx):
            nonlocal ticks
            ticks += 2_000_000
            driver.depth_callback(None, np.full((480, 640), 750, np.uint16), ticks)
            frame = (
                np.zeros((488, 640), np.uint16)
                if driver.video_mode == driver.VIDEO_IR_10BIT
                else np.full((1024, 1280, 3), 42, np.uint8)
            )
            driver.video_callback(None, frame, ticks)
            return 0

        driver.process_events = driver.process_events_timeout = events
        stop = __import__("threading").Event()

        def send(packet):
            packets.append(packet)
            if packet[0] == "frame":
                stop.set()

        connection = SimpleNamespace(send=send, close=lambda: None)
        rgb = bytearray(1024 * 1280 * 3)
        depth = bytearray(480 * 640 * 2)
        with patch.dict("sys.modules", freenect=driver):
            capture_frames(connection, stop, rgb, depth, high_res=True)
        self.assertIn(("depth", (1, 0)), modes)
        self.assertIn(("video", (2, 0)), modes)
        frames = [payload for kind, payload in packets if kind == "frame"]
        self.assertEqual(1, len(frames), packets)
        self.assertEqual("raw_11bit", frames[0]["depth_encoding"])
        self.assertEqual(10, frames[0]["rgb_fps"])
        self.assertTrue(np.all(np.frombuffer(rgb, np.uint8) == 42))
        self.assertTrue(np.all(np.frombuffer(depth, np.uint16) == 750))


if __name__ == "__main__":
    unittest.main()

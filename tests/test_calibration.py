import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shared.calibration import (
    camera_matrix,
    prepare_metric_depth,
    prepare_rgbd,
    rectification_maps,
)
from shared.device_evidence import board_points, calibrate_corners, plane_evidence
from shared.settings import CameraCalibration, ScanSettings


class CalibrationTests(unittest.TestCase):
    def test_depth_only_path_is_exact_for_native_and_registered_observations(self):
        from dataclasses import replace

        from shared.sensor_calibration import load_calibration

        rng = np.random.default_rng(36)
        native = ScanSettings(sensor_calibration=load_calibration())
        registered = ScanSettings(camera=CameraCalibration(
            distortion=(-0.1, 0.01, 0.001, 0, 0), depth_scale=1.02))
        for settings in (native, registered, ScanSettings()):
            raw = rng.integers(650, 1000, (480, 640), dtype=np.uint16)
            raw[180:230, 280:320] = 2047 if settings.sensor_calibration else 0
            rgb = rng.integers(0, 255, (settings.rgb_camera.height, settings.rgb_camera.width, 3), dtype=np.uint8)
            for filtered in (False, True):
                configured = replace(settings, filter_depth=filtered, roi=(50, 40, 590, 440))
                expected = prepare_rgbd(rgb, raw, configured)[1]
                with patch("shared.calibration.project_rgb", side_effect=AssertionError("unused RGB projection")):
                    actual = prepare_metric_depth(raw, configured)
                np.testing.assert_array_equal(expected, actual)
                self.assertEqual(np.uint16, actual.dtype)
                self.assertTrue(actual.flags.c_contiguous)

    def test_engine_applies_scale_but_keeps_sensor_recording_raw(self):
        from scanner_server.engine import ScanEngine
        engine = ScanEngine()
        engine.reset(settings=ScanSettings(camera=CameraCalibration(depth_scale=1.1)))
        rgb = np.full((480, 640, 3), 120, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(1, engine.frame_count)
        z = np.asarray(engine.model_pcd.points)[:, 2]
        self.assertAlmostEqual(float(np.median(z)), 1.32, delta=.005)
        np.testing.assert_array_equal(engine.raw_frames[0][1], depth)

    def test_calibration_on_independent_views_recovers_known_camera(self):
        camera = CameraCalibration(
            fx=570, fy=575, cx=320, cy=242, distortion=(-0.08, 0.02, 0.001, -0.001, 0)
        )
        objects = board_points(9, 6, 0.025)
        rng = np.random.default_rng(6)
        corners = []
        for i in range(24):
            r = rng.uniform(-0.5, 0.5, 3)
            t = np.array(
                [
                    -0.1 + (i % 4 - 1.5) * 0.09,
                    -0.06 + (i // 4 - 2.5) * 0.035,
                    0.7 + (i % 3) * 0.07,
                ]
            )
            points, _ = cv2.projectPoints(
                objects, r, t, camera_matrix(camera), np.array(camera.distortion)
            )
            corners.append(points + rng.normal(0, 0.03, points.shape))
        measured, report = calibrate_corners(corners, 9, 6, 0.025)
        self.assertTrue(report["accepted"], report)
        self.assertLess(max(report["heldout_rms_px"]), 0.1)
        self.assertAlmostEqual(measured.fx, camera.fx, delta=2)
        self.assertAlmostEqual(measured.fy, camera.fy, delta=2)
        self.assertFalse(report["depth_scale_measured"])

    def test_registered_remap_matches_rgb_and_never_interpolates_depth(self):
        camera = CameraCalibration(distortion=(-0.1, 0, 0, 0, 0), depth_scale=1.02)
        settings = ScanSettings(camera=camera, filter_depth=False)
        depth = np.zeros((480, 640), np.uint16)
        depth[:, 150:450] = 1000
        rgb = np.repeat((depth > 0)[..., None], 3, axis=2).astype(np.uint8) * 255
        color, metric = prepare_rgbd(rgb, depth, settings)
        x, y = rectification_maps(camera)
        expected = cv2.remap(depth, x, y, cv2.INTER_NEAREST)
        np.testing.assert_array_equal(metric, expected * 1.02)
        self.assertEqual({0, 1020}, set(np.unique(metric)))
        self.assertTrue(np.all(color[metric > 0] >= 127))
        self.assertEqual(1000, depth.max())
        self.assertEqual(camera, ScanSettings.from_dict(settings.to_dict()).camera)

    def test_default_calibration_preserves_pixels(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 255, (480, 640, 3), np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        result = prepare_rgbd(rgb, depth, ScanSettings(filter_depth=False))
        np.testing.assert_array_equal(result[0], rgb)
        np.testing.assert_array_equal(result[1], depth)

    def test_plane_precision_and_known_depth_are_distinct(self):
        rng = np.random.default_rng(2)
        depth = np.rint(1020 + rng.normal(0, 2, (480, 640))).astype(np.uint16)
        report = plane_evidence(depth, CameraCalibration(), (100, 100, 540, 380), 1.0)
        self.assertAlmostEqual(report["z_error_m"], 0.02, delta=0.001)
        self.assertLess(report["plane_rmse_m"], 0.003)
        self.assertAlmostEqual(report["suggested_depth_scale"], 1 / 1.02, delta=0.001)
        with self.assertRaises(ValueError):
            plane_evidence(np.zeros_like(depth), CameraCalibration(), (0, 0, 640, 480))

    def test_invalid_lens_fit_and_scale_are_rejected(self):
        with self.assertRaises(ValueError):
            CameraCalibration(distortion=(-2, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            CameraCalibration(depth_scale=float("nan"))
        with self.assertRaises(ValueError):
            calibrate_corners([np.zeros((54, 2))] * 3, 9, 6, 0.025)

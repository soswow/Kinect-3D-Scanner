"""Numerical and compatibility checks for capture performance changes."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import base64
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from kinect_scanner.server_client import ServerClient
from scanner_server.engine import ScanEngine
from scanner_server.weighted_fusion import _integrate_tensor
from shared.calibration import camera_matrix, project_rgb
from shared.config import LIVE_MAX_POINTS
from shared.live import decode_live_geometry, encode_live_message
from shared.protocol import (
    pack_frame,
    pack_frames,
    unpack_frame_with_metadata,
    unpack_frames,
)
from shared.sensor_calibration import load_calibration
from tests.test_quality import scene_frames


class CapturePerformanceTests(unittest.TestCase):
    def test_refresh_time_is_not_double_counted_as_tracking(self):
        engine = ScanEngine(device="cpu")
        with (
            patch(
                "scanner_server.engine.time.monotonic",
                side_effect=[10, 10.1, 10.3, 10.5],
            ),
            engine._stage("tracking"),engine._stage("model_refresh")
        ):
            pass
        self.assertAlmostEqual(200, engine.stage_totals_ms["model_refresh"])
        self.assertAlmostEqual(300, engine.stage_totals_ms["tracking"])
        self.assertAlmostEqual(500, sum(engine.stage_totals_ms.values()))

    def test_cpu_weighted_fusion_matches_tensor_reference(self):
        frames = scene_frames(2)
        results = []
        for reference in (True, False):
            engine = ScanEngine(device="cpu")
            engine.reset(settings=replace(engine.settings, confidence_fusion=True))
            from scanner_server import weighted_fusion

            integration = (
                _integrate_tensor if reference else weighted_fusion._integrate_cpu
            )
            with patch.object(weighted_fusion, "_integrate_cpu", integration):
                for rgb, depth, pose in frames:
                    # Retain depth edges and holes while bounding test memory.
                    depth = depth.copy()
                    depth[:160] = depth[320:] = 0
                    depth[:, :180] = depth[:, 460:] = 0
                    engine._integrate_vbg(rgb, depth, np.linalg.inv(pose))
            hashmap = engine.vbg.hashmap()
            ids = hashmap.active_buf_indices().numpy().astype(int)
            keys = hashmap.key_tensor().numpy()[ids]
            order = np.lexsort(keys.T)
            attributes = {
                name: engine.vbg.attribute(name)
                .numpy()
                .reshape(-1, 4096, 3 if name == "color" else 1)[ids][order]
                .copy()
                for name in ("tsdf", "weight", "color")
            }
            results.append((keys[order].copy(), attributes))
            del engine
        np.testing.assert_array_equal(results[0][0], results[1][0])
        for name in results[0][1]:
            np.testing.assert_allclose(
                results[0][1][name], results[1][1][name], atol=1e-6
            )

    def test_dense_projection_matches_opencv_for_both_native_rgb_modes(self):
        calibration = load_calibration()
        points = np.random.default_rng(9).uniform(
            [-800, -500, 0], [800, 500, 3000], (20, 50, 3)
        )
        points[0] = 0  # Missing depth and the baseline translation remain defined.
        for camera in (calibration.rgb_low_res, calibration.rgb_high_res):
            transformed = (
                points.reshape(-1, 3) @ np.asarray(calibration.rotation).T
                + calibration.translation_mm
            )
            expected, _ = cv2.projectPoints(
                transformed,
                np.zeros(3),
                np.zeros(3),
                camera_matrix(camera),
                np.array(camera.distortion),
            )
            pixels, z = project_rgb(points, calibration, camera)
            np.testing.assert_allclose(
                pixels, expected.reshape(20, 50, 2), rtol=1e-12, atol=1e-8
            )
            np.testing.assert_array_equal(z, transformed[:, 2].reshape(20, 50))

    def test_recovery_reuses_last_accepted_cloud_and_reset_clears_it(self):
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth)
        engine.process_frames()
        anchor = engine._last_reg_pcd
        self.assertIsNotNone(anchor)
        engine._tracking_lost_frames = 1
        # Failed probes must not rebuild or replace the observed reference.
        with (
            patch.object(
                engine, "_make_reg_pcd", side_effect=AssertionError("Rebuilt anchor")
            ),
            patch("scanner_server.refinement._trustworthy", return_value=False),
        ):
            engine._recover_anchor(anchor)
            engine._recover_anchor(anchor)
        self.assertIs(anchor, engine._last_reg_pcd)
        engine.reset()
        self.assertIsNone(engine._last_reg_pcd)

    def test_packed_feedback_roundtrip_and_legacy_client_delivery(self):
        values = np.random.default_rng(7).random((LIVE_MAX_POINTS, 6), dtype=np.float32)
        message = {
            "type": "live",
            "session_id": "test",
            "points": values[:, :3],
            "colors": values[:, 3:],
        }
        packed = encode_live_message(message, packed_geometry=True)
        legacy = encode_live_message(message)
        decoded = decode_live_geometry(json.loads(packed))
        np.testing.assert_array_equal(decoded["points"], values[:, :3])
        np.testing.assert_array_equal(decoded["colors"], values[:, 3:])
        self.assertLess(len(packed), len(legacy) / 3)
        client = ServerClient()
        received = []
        client.live_updated.connect(received.append)
        client._handle_ws_message(json.loads(packed))
        client._handle_ws_message(json.loads(legacy))
        for result in received:
            np.testing.assert_array_equal(result["points"], values[:, :3])
        self.assertEqual(2, len(received))

    def test_packed_feedback_rejects_malformed_and_unbounded_payloads(self):
        valid = json.loads(
            encode_live_message(
                {"type": "live", "points": [[0, 0, 1]], "colors": [[1, 0, 0]]},
                packed_geometry=True,
            )
        )
        for replacement in (
            {"encoding": "unknown"},
            {"count": LIVE_MAX_POINTS + 1},
            {"count": True},
            {"data": "!" * 32},
            {"data": ""},
            {"data": base64.b64encode(np.full(6, np.nan, "<f4").tobytes()).decode()},
        ):
            with self.assertRaises(ValueError):
                decode_live_geometry({"geometry": valid["geometry"] | replacement})

    def test_loopback_packets_are_lossless_in_both_rgb_modes_and_batches(self):
        rng = np.random.default_rng(3)
        depth = rng.integers(0, 2048, (480, 640), np.uint16)
        for shape in ((480, 640, 3), (1024, 1280, 3)):
            rgb = rng.integers(0, 256, shape, np.uint8)
            metadata = {"frame_id": 3, "depth_encoding": "raw_11bit"}
            packet = pack_frame(rgb, depth, metadata, compression_level=0)
            decoded = unpack_frame_with_metadata(packet)
            np.testing.assert_array_equal(rgb, decoded[0])
            np.testing.assert_array_equal(depth, decoded[1])
            self.assertEqual(3, decoded[2]["frame_id"])
            batch = pack_frames([(rgb, depth, metadata)] * 2, compression_level=0)
            for color, observed, info in unpack_frames(batch, with_metadata=True):
                np.testing.assert_array_equal(rgb, color)
                np.testing.assert_array_equal(depth, observed)
                self.assertEqual(3, info["frame_id"])


if __name__ == "__main__":
    unittest.main()

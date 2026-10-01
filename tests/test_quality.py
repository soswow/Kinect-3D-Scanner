"""Regression coverage for actual geometry, protocol and capture ordering."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from shared.capture import timestamp_delta_ms
from shared.depth import prepare_depth
from shared.protocol import (
    pack_frame,
    pack_frames,
    unpack_frame_with_metadata,
    unpack_frames,
)
from shared.recording import RecordingWriter
from shared.settings import CameraCalibration, ScanSettings


def scene_frames(count=8):
    """Raycast a static asymmetric scene from known moving camera poses."""
    scene = o3d.t.geometry.RaycastingScene()
    for size, offset in [
        ((3.0, 2.4, 0.1), (-1.5, -1.2, 2.2)),
        ((0.45, 0.55, 0.25), (-0.4, -0.3, 1.15)),
        ((0.23, 0.4, 0.4), (0.23, -0.05, 1.4)),
        ((0.65, 0.18, 0.32), (-0.1, 0.4, 1.6)),
    ]:
        mesh = o3d.geometry.TriangleMesh.create_box(*size).translate(offset)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    camera = CameraCalibration()
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.0]])
    rng = np.random.default_rng(42)
    frames = []
    for i in range(count):
        pose = np.eye(4)
        pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, i * 0.006, 0))
        pose[:3, 3] = [i * 0.015, i * 0.003, 0]
        rays = scene.create_rays_pinhole(k, np.linalg.inv(pose), 640, 480)
        z = scene.cast_rays(rays)["t_hit"].numpy()
        depth = np.where(np.isfinite(z), z * 1000, 0)
        valid = depth > 0
        depth[valid] += rng.normal(0, 2, valid.sum())
        depth = np.clip(np.rint(depth), 0, 65535).astype(np.uint16)
        depth[rng.random(depth.shape) < 0.03] = 0
        ray_values = rays.numpy()
        points = (
            ray_values[:, :, :3]
            + np.where(np.isfinite(z), z, 0)[:, :, None] * ray_values[:, :, 3:]
        )
        wx, wy, wz = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        # Static world-space texture changes in the image as the camera moves.
        rgb = np.stack(
            [
                127 + 80 * np.sin(wx * 25) + 35 * np.cos(wy * 19),
                127 + 80 * np.sin(wy * 29) + 35 * np.cos(wz * 13),
                127 + 80 * np.sin((wx + wy) * 23) + 35 * np.cos(wz * 17),
            ],
            axis=-1,
        )
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        frames.append((rgb, depth, pose))
    return frames


class QualityTests(unittest.TestCase):
    def test_settings_validate_and_roundtrip(self):
        s = ScanSettings(roi=(100, 80, 540, 400))
        self.assertEqual(s, ScanSettings.from_dict(s.to_dict()))
        for kwargs in (
            {"near_m": 2, "far_m": 1},
            {"voxel_m": 0.0001},
            {"far_m": float("nan")},
            {"roi": (-1, 0, 20, 20)},
        ):
            with self.assertRaises(ValueError):
                ScanSettings(**kwargs)
        with self.assertRaises(ValueError):
            CameraCalibration(image_space="ir")

    def test_clipping_roi_and_isolated_depth(self):
        depth = np.full((480, 640), 1000, np.uint16)
        depth[120, 120] = 1700
        depth[130:135, 130:135] = 200
        depth[140:145, 140:145] = 5000
        prepared = prepare_depth(depth, ScanSettings(roi=(100, 100, 200, 200)))
        self.assertEqual(0, prepared[120, 120])
        self.assertEqual(0, prepared[132, 132])
        self.assertEqual(0, prepared[142, 142])
        self.assertEqual(0, prepared[250, 250])
        self.assertEqual(1000, prepared[160, 160])

    def test_protocol_metadata_legacy_and_malformed(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), 1234, np.uint16)
        for metadata in (None, {"frame_id": 7, "rgb_depth_delta_ms": -12}):
            packed = pack_frame(rgb, depth, metadata)
            color, metric, meta = unpack_frame_with_metadata(packed)
            np.testing.assert_array_equal(rgb, color)
            np.testing.assert_array_equal(depth, metric)
            self.assertEqual(metadata or {}, meta)
            with self.assertRaises(ValueError):
                unpack_frame_with_metadata(packed[:-1])
        batch = pack_frames([(rgb, depth, {"frame_id": 1}), (rgb, depth)])
        self.assertEqual(2, len(unpack_frames(batch, with_metadata=True)))
        with self.assertRaises(ValueError):
            unpack_frames(batch + b"x")
        with self.assertRaises(ValueError):
            pack_frame(rgb, depth.astype(float))
        self.assertAlmostEqual(0.0001, timestamp_delta_ms(3, (1 << 32) - 3))
        self.assertEqual(33.0, timestamp_delta_ms(1_980_000, 0))
        self.assertEqual(-33.0, timestamp_delta_ms(0, 1_980_000))

    def test_duplicate_and_unsynchronised_frames(self):
        e = ScanEngine()
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.ones((480, 640), np.uint16) * 1000
        self.assertTrue(e.store_frame(rgb, depth, {"frame_id": 1})["success"])
        self.assertFalse(e.store_frame(rgb, depth, {"frame_id": 1})["success"])
        self.assertFalse(
            e.store_frame(rgb, depth, {"frame_id": 2, "rgb_depth_delta_ms": 60})[
                "success"
            ]
        )
        self.assertEqual(1, e.stored_count)

    def test_tsdf_uses_configured_truncation(self):
        e = ScanEngine()
        e.reset(settings=ScanSettings(truncation_m=0.02))
        e.vbg = Mock()
        e._integrate_vbg(
            np.zeros((480, 640, 3), np.uint8),
            np.ones((480, 640), np.uint16) * 1000,
            np.eye(4),
        )
        self.assertEqual(
            4,
            e.vbg.compute_unique_block_coordinates.call_args.kwargs[
                "trunc_voxel_multiplier"
            ],
        )
        self.assertEqual(4, e.vbg.integrate.call_args.kwargs["trunc_voxel_multiplier"])

    def test_plane_and_pose_jump_rejected(self):
        e = ScanEngine()
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.ones((480, 640), np.uint16) * 1000
        pcd = e._make_reg_pcd(e._make_rgbd(rgb, depth))
        result = o3d.pipelines.registration.registration_icp(pcd, pcd, 0.05, np.eye(4))
        self.assertIn("flat", e._alignment_error(result, pcd))
        result.transformation = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]
        )
        self.assertIn("Pose jump", e._alignment_error(result, pcd))

    def test_moving_noisy_scene_pose_and_bad_frame(self):
        frames = scene_frames()
        e = ScanEngine()
        for i, (rgb, depth, pose) in enumerate(frames):
            e.store_frame(rgb, depth, {"frame_id": i})
        e.store_frame(
            frames[0][0], np.zeros((480, 640), np.uint16), {"frame_id": "bad"}
        )
        result = e.process_frames()
        self.assertEqual(8, e.frame_count, e.diagnostics)
        self.assertEqual(1, result["errors"])
        self.assertEqual(0, e.unprocessed_count)
        errors = [
            np.linalg.norm(pose[:3, 3] - frames[index][2][:3, 3])
            for index, pose in e.poses
        ]
        self.assertLess(np.sqrt(np.mean(np.square(errors))), 0.012)
        self.assertTrue(e.build_mesh()[0])
        self.assertEqual(0, e.process_frames()["processed"])
        self.assertGreater(len(e.mesh.triangles), 1000)

    def test_color_recovery_seeds_geometry_without_ground_truth(self):
        from dataclasses import replace

        frames = scene_frames(2)
        e = ScanEngine()
        e.reset(settings=replace(ScanSettings(), color_recovery=True))
        e.store_frame(*frames[0][:2], {"timestamp_s": 0.0})
        e.process_frames()
        e.store_frame(*frames[1][:2], {"timestamp_s": 0.1, "rgb_depth_delta_ms": 0})
        e.process_frames()
        self.assertEqual(2, e.frame_count, e.diagnostics)
        self.assertEqual("rgbd+icp", e.diagnostics[-1]["method"])
        self.assertLess(
            np.linalg.norm(e.cumulative_T[:3, 3] - frames[1][2][:3, 3]), 0.01
        )
        e.frame_metadata.append({"rgb_depth_delta_ms": 30})
        self.assertIsNone(e._color_recovery(e.model_pcd, e._last_rgbd))

    def test_motion_recovery_rechecks_geometric_confidence(self):
        frames = scene_frames(3)
        e = ScanEngine()
        for i, frame in enumerate(frames[:2]):
            e.store_frame(*frame[:2], {"timestamp_s": i * 0.1})
            e.process_frames()
        e.store_frame(*frames[2][:2], {"timestamp_s": 0.2})
        original = e._icp
        calls = []

        def weak_initial(*args, **kwargs):
            calls.append(1)
            if len(calls) <= 2:
                return Mock(fitness=0.0, inlier_rmse=1.0, transformation=np.eye(4))
            return original(*args, **kwargs)

        with patch.object(e, "_icp", side_effect=weak_initial):
            e.process_frames()
        self.assertEqual(3, e.frame_count, e.diagnostics)
        self.assertEqual("motion+icp", e.diagnostics[-1]["method"])
        self.assertLess(
            np.linalg.norm(e.cumulative_T[:3, 3] - frames[2][2][:3, 3]), 0.012
        )

    def test_integration_failure_does_not_advance_pose_or_count(self):
        rgb, depth, _ = scene_frames(1)[0]
        e = ScanEngine()
        e.store_frame(rgb, depth)
        with patch.object(
            e, "_integrate_vbg", side_effect=RuntimeError("integration failure")
        ):
            result = e.process_frames()
        self.assertEqual(0, e.frame_count)
        self.assertEqual(1, result["errors"])
        self.assertEqual(0, e.unprocessed_count)
        np.testing.assert_array_equal(np.eye(4), e.cumulative_T)

    def test_global_recovery_bounds_feature_matching_size(self):
        from scanner_server import engine as module

        e = ScanEngine()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.random.default_rng(7).uniform(-2, 2, (20000, 3))
        )
        counts = []

        def feature(pcd, *args):
            counts.append(len(pcd.points))
            return Mock()

        with (
            patch.object(module._REG, "compute_fpfh_feature", side_effect=feature),
            patch.object(
                module._REG,
                "registration_fgr_based_on_feature_matching",
                return_value=Mock(transformation=np.eye(4)),
            ),
            patch.object(e, "_icp", return_value=Mock()),
        ):
            e._fpfh_fallback(cloud, cloud)
        self.assertEqual(2, len(counts))
        self.assertTrue(all(100 < count <= 5000 for count in counts), counts)

    def test_frame_batches_preserve_command_barriers(self):
        from kinect_scanner.server_task_worker import (
            ServerTask,
            ServerTaskWorker,
        )
        from kinect_scanner.server_task_worker import (
            ServerTaskType as T,
        )

        client = Mock()
        client.send_frame.return_value = {"success": True}
        worker = ServerTaskWorker(client)
        frame = lambda i: ServerTask(T.SEND_FRAME, {"rgb": i, "depth": i})
        reset = ServerTask(T.RESET)
        worker.submit(frame(2))
        worker.submit(reset)
        worker.submit(frame(3))
        worker._drain_send_frames(frame(1))
        self.assertEqual(
            [1, 2], [f[0] for f in client.send_frames_batch.call_args.args[0]]
        )
        self.assertIs(reset, worker._pending)
        self.assertEqual(3, worker._queue.get_nowait().kwargs["rgb"])

    def test_recording_is_lossless_and_portable(self):
        import json

        import cv2

        with tempfile.TemporaryDirectory() as folder:
            w = RecordingWriter(Path(folder) / "scan", ScanSettings().to_dict())
            rgb = np.full((480, 640, 3), [12, 34, 56], np.uint8)
            depth = np.full((480, 640), 1234, np.uint16)
            w.append(rgb, depth, {"timestamp_s": 1.25})
            m = json.loads((w.path / "manifest.json").read_text())
            self.assertEqual("millimetres", m["depth_unit"])
            np.testing.assert_array_equal(
                depth,
                cv2.imread(str(w.path / m["frames"][0]["depth"]), cv2.IMREAD_UNCHANGED),
            )
            np.testing.assert_array_equal(
                rgb,
                cv2.cvtColor(
                    cv2.imread(str(w.path / m["frames"][0]["rgb"])), cv2.COLOR_BGR2RGB
                ),
            )


if __name__ == "__main__":
    unittest.main()

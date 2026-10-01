"""Compute backends, portable materials, and validated loop reintegration."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from scanner_server.backend import select_backend
from scanner_server.engine import ScanEngine
from scanner_server.refinement import propose_poses
from scanner_server.session import export_session
from scanner_server.texturing import export_texture, make_textured_mesh
from shared.depth import prepare_depth
from shared.settings import ScanSettings
from tests.test_quality import scene_frames


def texture_scene(depth_mm=1200):
    mesh = o3d.geometry.TriangleMesh.create_box(0.3, 0.3, 0.3).translate(
        (-0.15, -0.15, 1.2)
    )
    mesh.paint_uniform_color([0.8, 0.2, 0.1])
    y, x = np.indices((480, 640))
    rgb = np.stack((x / 640 * 255, y / 480 * 255, np.zeros_like(x)), axis=-1).astype(
        np.uint8
    )
    return SimpleNamespace(
        mesh=mesh,
        poses=[(0, np.eye(4))],
        settings=ScanSettings(),
        raw_frames=[(rgb, np.full((480, 640), depth_mm, np.uint16))],
        frame_metadata=[{}],
    )


class FeatureTests(unittest.TestCase):
    def test_backend_selection_and_unavailable_cuda(self):
        with patch(
            "scanner_server.backend.o3d.core.cuda.is_available", return_value=False
        ):
            device, report = select_backend("auto")
            self.assertEqual("CPU:0", str(device))
            self.assertTrue(report["fallback_reason"])
            with self.assertRaises(RuntimeError):
                select_backend("cuda")
        with self.assertRaises(ValueError):
            select_backend("metal")

    def test_tensor_tracking_motion_and_flat_geometry_gate(self):
        engine = ScanEngine(device="cpu", tracking="tensor")
        frames = scene_frames(4)
        for rgb, depth, _ in frames:
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(4, engine.frame_count, engine.diagnostics)
        self.assertLess(
            np.linalg.norm(engine.cumulative_T[:3, 3] - frames[-1][2][:3, 3]), 0.015
        )
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        engine.reset()
        for _ in range(2):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(1, engine.frame_count)
        self.assertFalse(engine.diagnostics[-1]["success"])
        self.assertIn("tracking", engine.stage_totals_ms)

    @unittest.skipUnless(
        o3d.core.cuda.is_available(), "NVIDIA CUDA hardware/build unavailable"
    )
    def test_cuda_tracking_and_fusion(self):
        engine = ScanEngine(device="cuda")
        for rgb, depth, _ in scene_frames(3):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(3, engine.frame_count, engine.diagnostics)
        self.assertTrue(engine.build_mesh()[0])

    def test_uv_orientation_occlusion_and_material_roundtrip(self):
        engine = texture_scene()
        mesh, report = make_textured_mesh(engine, size=256)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        uv = mesh.visual.uv[mesh.faces].mean(axis=1)
        colors = trimesh.visual.color.uv_to_color(uv, mesh.visual.material.image)
        front = np.isclose(centers[:, 2], 1.2)
        expected = np.stack(
            (
                (centers[front, 0] * 525 / 1.2 + 319.5) / 640 * 255,
                (centers[front, 1] * 525 / 1.2 + 239.5) / 480 * 255,
                np.zeros(front.sum()),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(colors[front, :3], expected, atol=4)
        self.assertGreater(report["projected_fraction"], 0.1)
        _, occluded = make_textured_mesh(texture_scene(900), size=256)
        self.assertEqual(0, occluded["projected_fraction"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            export_texture(engine, path / "scan.glb", size=256)
            scene = trimesh.load(path / "scan.glb", force="scene")
            loaded = next(iter(scene.geometry.values()))
            self.assertIsInstance(loaded.visual, trimesh.visual.texture.TextureVisuals)
            self.assertIsNotNone(loaded.visual.material.baseColorTexture)
            export_texture(engine, path / "scan.zip", "obj.zip", size=256)
            with zipfile.ZipFile(path / "scan.zip") as archive:
                obj = archive.read("scan.obj").decode()
                self.assertIn("mtllib scan.mtl", obj)
                self.assertIn("\nvt ", obj)
                material = archive.read("scan.mtl").decode()
                self.assertIn("map_Kd scan_material.png", material)
                archive.extractall(path / "obj")
            loaded_obj = trimesh.load(path / "obj/scan.obj", force="scene")
            self.assertIsInstance(
                next(iter(loaded_obj.geometry.values())).visual,
                trimesh.visual.texture.TextureVisuals,
            )

    def test_session_lossless_and_estimates_separate_from_reference(self):
        engine = ScanEngine()
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth, {"frame_id": "one", "timestamp_s": 1.0})
        engine.process_frames()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.zip"
            export_session(engine, path)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(Path(temporary) / "session")
            folder = Path(temporary) / "session"
            np.testing.assert_array_equal(
                np.asarray(Image.open(folder / "rgb/000000.png")), rgb
            )
            np.testing.assert_array_equal(
                np.asarray(Image.open(folder / "depth/000000.png")), depth
            )
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(0, manifest["frames"][0]["metadata"]["server_index"])
            self.assertNotIn("reference_pose", manifest["frames"][0])
            report = json.loads((folder / "reconstruction.json").read_text())
            self.assertEqual("camera_to_world", report["pose_convention"])
            self.assertEqual(1, len(report["poses"]))

    def test_synthetic_loop_reduces_injected_drift(self):
        engine = ScanEngine()
        base = scene_frames(10)
        frames = base + base[-2::-1]
        for i, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            drifted = truth.copy()
            drifted[0, 3] += i * 0.004
            engine.poses.append((i, drifted))
        proposed, report = propose_poses(engine)
        self.assertIsNotNone(proposed, report)
        self.assertGreater(report["loops"], 0)
        error = np.sqrt(
            np.mean(
                [
                    np.linalg.norm(p[:3, 3] - f[2][:3, 3]) ** 2
                    for (_, p), f in zip(proposed, frames)
                ]
            )
        )
        self.assertLess(error, 0.008)
        self.assertLess(
            report["validation_after_m2"], report["validation_before_m2"] * 0.98
        )
        np.testing.assert_allclose(proposed[0][1], engine.poses[0][1], atol=1e-8)

    def test_failed_reintegration_preserves_volume_and_poses(self):
        engine = ScanEngine()
        for rgb, depth, _ in scene_frames(3):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        original_volume = engine.vbg
        original_poses = [(i, p.copy()) for i, p in engine.poses]
        with (
            patch(
                "scanner_server.refinement.propose_poses",
                return_value=(original_poses, {}),
            ),
            patch.object(
                engine, "_integrate_vbg", side_effect=RuntimeError("simulated OOM")
            ),
        ):
            engine._refine_volume()
        self.assertIs(original_volume, engine.vbg)
        self.assertFalse(engine.refinement["applied"])
        for (_, a), (_, b) in zip(original_poses, engine.poses):
            np.testing.assert_array_equal(a, b)

    def test_flat_loop_retains_original_trajectory(self):
        engine = ScanEngine()
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        for i in range(10):
            engine.store_frame(rgb, depth)
            pose = np.eye(4)
            pose[0, 3] = 0.03 * (i if i < 5 else 9 - i)
            engine.poses.append((i, pose))
        originals = [(i, p.copy()) for i, p in engine.poses]
        proposed, report = propose_poses(engine)
        self.assertIsNone(proposed, report)
        self.assertEqual(0, report["loops"])
        for (_, a), (_, b) in zip(originals, engine.poses):
            np.testing.assert_array_equal(a, b)

    def test_partial_batch_recording_uses_individual_acknowledgements(self):
        from unittest.mock import Mock
        from shared.recording import RecordingWriter
        from kinect_scanner.server_task_worker import (
            ServerTask,
            ServerTaskType,
            ServerTaskWorker,
        )

        client = Mock()
        client.send_frames_batch.return_value = {
            "success": True,
            "results": [{"success": True, "index": 4}, {"success": False}],
        }
        worker = ServerTaskWorker(client)
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        task = ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth})
        worker.submit(task)
        with tempfile.TemporaryDirectory() as temporary:
            worker._recording = RecordingWriter(
                Path(temporary) / "scan", ScanSettings().to_dict()
            )
            worker._drain_send_frames(task)
            recorded = worker._recording.manifest["frames"]
            self.assertEqual(1, len(recorded))
            self.assertEqual(4, recorded[0]["metadata"]["server_index"])

    def test_refinement_rebuilds_geometry_and_reduces_surface_error(self):
        engine = ScanEngine()
        base = scene_frames(10)
        frames = base + base[-2::-1]
        for i, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            drifted = truth.copy()
            drifted[0, 3] += i * 0.004
            engine.poses.append((i, drifted))
            engine.diagnostics.append(
                {"success": True, "index": i, "pose": drifted.tolist()}
            )
            engine._integrate_vbg(
                rgb, prepare_depth(depth, engine.settings), np.linalg.inv(drifted)
            )
        engine.frame_count = engine._processed_count = len(frames)
        self.assertTrue(engine.build_mesh()[0])
        target = o3d.t.geometry.RaycastingScene()
        for size, offset in [
            ((3.0, 2.4, 0.1), (-1.5, -1.2, 2.2)),
            ((0.45, 0.55, 0.25), (-0.4, -0.3, 1.15)),
            ((0.23, 0.4, 0.4), (0.23, -0.05, 1.4)),
            ((0.65, 0.18, 0.32), (-0.1, 0.4, 1.6)),
        ]:
            target.add_triangles(
                o3d.t.geometry.TriangleMesh.from_legacy(
                    o3d.geometry.TriangleMesh.create_box(*size).translate(offset)
                )
            )

        def surface_error(mesh):
            vertices = np.asarray(mesh.vertices)
            vertices = vertices[vertices[:, 2] < 2.0][
                ::10
            ]  # object detail, excluding the large back wall
            distances = target.compute_distance(
                o3d.core.Tensor(vertices.astype(np.float32))
            ).numpy()
            return float(np.sqrt(np.mean(distances**2)))

        before = surface_error(engine.mesh)
        original_volume = engine.vbg
        engine.settings = replace(engine.settings, refine_poses=True)
        success, result = engine.build_mesh()
        self.assertTrue(success, result)
        self.assertTrue(engine.refinement["applied"], engine.refinement)
        self.assertIsNot(original_volume, engine.vbg)
        self.assertEqual(len(frames), len(engine.original_poses))
        after = surface_error(engine.mesh)
        self.surface_metrics = {
            "before_surface_rmse_m": before,
            "after_surface_rmse_m": after,
            "refinement": engine.refinement,
        }
        self.assertLess(after, before * 0.75, (before, after))
        self.assertLess(after, 0.008)
        committed = engine.vbg
        engine.build_mesh()
        self.assertIs(
            committed, engine.vbg, "Unchanged sessions should not refine repeatedly"
        )


if __name__ == "__main__":
    unittest.main()

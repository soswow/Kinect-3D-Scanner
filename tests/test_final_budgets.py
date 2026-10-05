"""Final fusion budgets, preservation on failure, and bounded live feedback."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from shared.settings import ScanSettings
from tests.test_quality import scene_frames


class FinalBudgetTests(unittest.TestCase):
    def make_engine(self):
        engine = ScanEngine()
        engine.reset(settings=replace(engine.settings, far_m=2.0))
        for rgb, depth, _ in scene_frames(4):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(4, engine.frame_count)
        return engine

    def test_final_reintegration_preserves_live_tracking_and_reuses_cache(self):
        engine = self.make_engine()
        self.assertTrue(engine.build_mesh()[0])
        live, model = engine.vbg, engine.model_pcd
        poses = [(i, p.copy()) for i, p in engine.poses]
        engine.settings = replace(engine.settings, final_voxel_m=0.003)
        ok, result = engine.build_mesh()
        self.assertTrue(ok, result)
        self.assertIs(engine.vbg, live)
        self.assertIs(engine.model_pcd, model)
        self.assertIsNot(engine._final_vbg, live)
        self.assertTrue(result["final_reconstruction"]["applied"])
        self.assertEqual(0.003, result["final_reconstruction"]["voxel_m"])
        self.assertLessEqual(result["final_reconstruction"]["blocks"], 5000)
        for (_, actual), (_, original) in zip(engine.poses, poses):
            np.testing.assert_array_equal(actual, original)
        with patch.object(
            engine,
            "_create_vbg",
            side_effect=AssertionError("Cached volume was rebuilt"),
        ):
            self.assertTrue(engine.build_mesh()[0])
        engine.store_frame(*scene_frames(1)[0][:2])
        self.assertIsNone(engine._final_vbg)

    def test_block_limit_and_native_failure_preserve_previous_result(self):
        engine = self.make_engine()
        self.assertTrue(engine.build_mesh()[0])
        live, mesh, cloud = engine.vbg, engine.mesh, engine.point_cloud
        engine.settings = replace(
            engine.settings, final_voxel_m=0.002, final_block_count=1
        )
        ok, result = engine.build_mesh()
        self.assertFalse(ok)
        self.assertIn("exceeds 1 blocks", result["message"])
        self.assertIs(engine.vbg, live)
        self.assertIs(engine.mesh, mesh)
        self.assertIs(engine.point_cloud, cloud)
        self.assertIsNone(engine._final_vbg)
        with patch.object(
            engine, "_create_vbg", side_effect=RuntimeError("allocation failed")
        ):
            ok, result = engine.build_mesh()
        self.assertFalse(ok)
        self.assertIn("allocation failed", result["message"])
        self.assertIs(engine.mesh, mesh)

    def test_server_queue_age_and_rejection_guidance_are_bounded(self):
        engine = ScanEngine()
        rgb, depth, _ = scene_frames(1)[0]
        for _ in range(6):
            engine.store_frame(rgb, depth)
        snapshot = engine.live_snapshot()
        self.assertIn("Pause", snapshot["guidance"])
        self.assertEqual(6, snapshot["pending_count"])
        self.assertGreaterEqual(snapshot["pending_age_s"], 0)
        engine.process_frames(max_frames=1)
        snapshot = engine.live_snapshot(max_points=200)
        self.assertLessEqual(len(snapshot["points"]), 200)
        engine.store_frame(rgb, np.zeros_like(depth))
        engine.process_frames()
        recovery = engine.live_snapshot()
        self.assertIn("STOP", recovery["guidance"])
        self.assertTrue(recovery["fusion_paused"])
        self.assertEqual(5, recovery["last_tracked_index"])
        self.assertIn("Too few valid depth", recovery["result"]["message"])
        self.assertEqual(1, engine.live_snapshot()["skipped_count"])
        self.assertEqual(0, engine.live_snapshot()["pending_age_s"])

    def test_settings_validate_final_resolution_and_budget(self):
        with self.assertRaises(ValueError):
            ScanSettings(final_voxel_m=0.01)
        with self.assertRaises(ValueError):
            ScanSettings(final_block_count=0)
        s = ScanSettings(final_voxel_m=0.003, final_block_count=1000)
        self.assertEqual(s, ScanSettings.from_dict(s.to_dict()))

    def _weighted_final(self, device):
        engine = ScanEngine(device=device)
        engine.reset(
            settings=replace(
                engine.settings,
                far_m=2,
                confidence_fusion=True,
                final_weight=0.5,
                final_voxel_m=0.004,
            )
        )
        for rgb, depth, _ in scene_frames(4):
            engine.store_frame(rgb, depth)
        live = engine.vbg
        ok, result = engine.build_mesh()
        self.assertTrue(ok, result)
        self.assertEqual(4, engine.frame_count)
        self.assertIs(engine.vbg, live)
        self.assertTrue(result["final_reconstruction"]["applied"])
        self.assertGreater(len(engine.mesh.triangles), 100)

    def test_weighted_final_volume(self):
        self._weighted_final("cpu")

    @unittest.skipUnless(
        o3d.core.cuda.is_available(), "NVIDIA CUDA hardware/build unavailable"
    )
    def test_cuda_weighted_final_volume(self):
        self._weighted_final("cuda")

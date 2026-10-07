"""Finish integration: bounded reintegration, rollback, and reusable results."""

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


class BundleIntegrationTests(unittest.TestCase):
    def engine(self):
        engine = ScanEngine(device="cpu")
        engine.reset(settings=replace(engine.settings, bundle_adjustment=True))
        for rgb, depth, _ in scene_frames(3):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(3, engine.frame_count)
        return engine

    def proposals(self, engine):
        poses = [(i, p.copy()) for i, p in engine.poses]
        for _, pose in poses[1:]:
            pose[0, 3] += 0.001
        return poses

    def test_settings_are_opt_in_and_validate_boolean(self):
        self.assertFalse(ScanSettings().bundle_adjustment)
        settings = ScanSettings(bundle_adjustment=True)
        self.assertEqual(settings, ScanSettings.from_dict(settings.to_dict()))
        with self.assertRaises(ValueError):
            ScanSettings(bundle_adjustment=1)

    def test_finish_rebuilds_then_caches_and_new_capture_invalidates(self):
        engine = self.engine()
        old_volume = engine.vbg
        old_poses = [(i, p.copy()) for i, p in engine.poses]
        with patch("scanner_server.bundle_adjustment.propose_bundle_poses",
                   return_value=(self.proposals(engine), {"applied": False})) as solver:
            ok, report = engine.build_mesh()
            self.assertTrue(ok, report)
            self.assertTrue(report["bundle_adjustment"]["applied"])
            self.assertIsNot(old_volume, engine.vbg)
            self.assertLessEqual(engine.vbg.hashmap().size(), engine.settings.final_block_count)
            committed = engine.vbg
            self.assertTrue(engine.build_mesh()[0])
            self.assertEqual(1, solver.call_count)
            self.assertIs(committed, engine.vbg)
        for (_, before), (_, saved) in zip(old_poses, engine.original_poses):
            np.testing.assert_array_equal(before, saved)
        engine.store_frame(*scene_frames(1)[0][:2])
        self.assertIsNone(engine._bundle_count)
        self.assertEqual("Awaiting final build", engine.bundle_adjustment["reason"])
        self.assertIn("bundle_adjustment", engine.reconstruction_report())

    def test_failed_fusion_preserves_state_and_can_retry(self):
        engine = self.engine()
        old_volume, old_diagnostics = engine.vbg, engine.diagnostics
        old_poses = [(i, p.copy()) for i, p in engine.poses]
        proposed = self.proposals(engine)
        with patch("scanner_server.bundle_adjustment.propose_bundle_poses",
                   return_value=(proposed, {"applied": False})):
            with patch.object(engine, "_integrate_vbg", side_effect=RuntimeError("simulated OOM")):
                self.assertFalse(engine._bundle_volume())
            self.assertIs(old_volume, engine.vbg)
            self.assertIs(old_diagnostics, engine.diagnostics)
            self.assertIsNone(engine._bundle_count)
            self.assertTrue(engine.bundle_adjustment["failed"])
            for (_, before), (_, after) in zip(old_poses, engine.poses):
                np.testing.assert_array_equal(before, after)
            self.assertTrue(engine._bundle_volume(), engine.bundle_adjustment)
            self.assertTrue(engine.bundle_adjustment["applied"])

    def test_bad_anchor_correction_or_budget_never_changes_volume(self):
        for failure in ("anchor", "correction", "budget"):
            with self.subTest(failure=failure):
                engine = self.engine()
                volume = engine.vbg
                poses = self.proposals(engine)
                if failure == "anchor":
                    poses[0][1][0, 3] += 0.01
                elif failure == "correction":
                    poses[-1][1][0, 3] += 0.3
                else:
                    engine.settings = replace(engine.settings, final_block_count=1)
                with patch("scanner_server.bundle_adjustment.propose_bundle_poses",
                           return_value=(poses, {"applied": False})):
                    self.assertFalse(engine._bundle_volume())
                self.assertIs(volume, engine.vbg)
                self.assertIsNone(engine.original_poses)
                self.assertFalse(engine.bundle_adjustment["applied"])

    def test_actual_joint_finish_rebuild_reduces_known_surface_error(self):
        engine = ScanEngine(device="cpu")
        frames = scene_frames(6)
        for index, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            estimate = truth.copy()
            estimate[0, 3] += index * 0.01
            estimate[2, 3] += index * 0.003
            engine.poses.append((index, estimate))
            engine.diagnostics.append({"index": index, "success": True, "pose": estimate.tolist()})
            engine._integrate_vbg(rgb, depth, np.linalg.inv(estimate))
        engine.frame_count = engine._processed_count = len(frames)
        engine.cumulative_T = engine.poses[-1][1].copy()
        self.assertTrue(engine.build_mesh()[0])
        reference = o3d.t.geometry.RaycastingScene()
        for size, offset in [((3.0, 2.4, 0.1), (-1.5, -1.2, 2.2)),
                             ((0.45, 0.55, 0.25), (-0.4, -0.3, 1.15)),
                             ((0.23, 0.4, 0.4), (0.23, -0.05, 1.4)),
                             ((0.65, 0.18, 0.32), (-0.1, 0.4, 1.6))]:
            reference.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
                o3d.geometry.TriangleMesh.create_box(*size).translate(offset)))

        def surface_error():
            points = np.asarray(engine.mesh.vertices)
            points = points[points[:, 2] < 2][::5]
            self.assertGreater(len(points), 100)
            distances = reference.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
            return float(np.sqrt(np.mean(distances**2))), len(points)

        before, before_count = surface_error()
        engine.settings = replace(engine.settings, bundle_adjustment=True)
        ok, report = engine.build_mesh()
        self.assertTrue(ok, report)
        self.assertTrue(engine.bundle_adjustment["applied"], report)
        after, after_count = surface_error()
        self.assertLess(after, before * 0.5, (before, after))
        self.assertGreater(after_count, before_count * 0.8)
        self.assertLess(after, 0.005)
        for index, pose in engine.poses:
            self.assertLess(np.linalg.norm(pose[:3, 3] - frames[index][2][:3, 3]), 0.006)


if __name__ == "__main__":
    unittest.main()

import unittest

import cv2
import numpy as np

from scanner_server.appearance import (
    extract_features,
    propose_transform,
    retrieve_pairs,
)
from shared.settings import CameraCalibration


class AppearanceTests(unittest.TestCase):
    def test_verified_relocalization_recovers_after_lost_pose(self):
        from dataclasses import replace

        from test_quality import scene_frames

        from scanner_server.engine import ScanEngine
        from shared.calibration import prepare_rgbd

        engine = ScanEngine()
        engine.reset(settings=replace(engine.settings, relocalize=True))
        frames = scene_frames(3)
        for rgb, depth, _ in frames:
            engine.store_frame(rgb, depth)
        engine.process_frames()
        rgb, depth, _ = frames[0]
        engine.store_frame(rgb, depth, {"rgb_depth_delta_ms": 0})
        rgb, depth = prepare_rgbd(rgb, depth, engine.settings)
        rgbd = engine._make_rgbd(rgb, depth)
        cloud = engine._make_reg_pcd(rgbd)
        engine.cumulative_T = engine.cumulative_T.copy()
        engine.cumulative_T[0, 3] = 0.6
        engine._tracking_lost_frames = 2
        result = engine._relocalize(cloud, rgbd)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result.transformation[:3, 3]), 0.02)
        engine.frame_metadata[-1]["rgb_depth_delta_ms"] = 30
        self.assertIsNone(engine._relocalize(cloud, rgbd))

    def test_refinement_finds_loop_outside_old_pose_radius(self):
        from test_quality import scene_frames

        from scanner_server.engine import ScanEngine
        from scanner_server.refinement import propose_poses

        engine = ScanEngine()
        base = scene_frames(10)
        frames = base + base[-2::-1]
        for i, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            drifted = truth.copy()
            drifted[0, 3] += i * 0.025
            engine.poses.append((i, drifted))
        self.assertGreater(np.linalg.norm(engine.poses[-1][1][:3, 3]), 0.35)
        poses, report = propose_poses(engine)
        self.assertIsNotNone(poses, report)
        self.assertGreater(report["appearance_loops"], 0)
        self.assertLess(np.linalg.norm(poses[-1][1][:3, 3]), 0.03)

    def fixture(self):
        rng = np.random.default_rng(21)
        rgb = rng.integers(40, 210, (480, 640, 3), np.uint8)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
        depth = np.full((480, 640), 1200, np.uint16)
        return rgb, depth

    def test_retrieval_is_independent_of_drift_and_rejects_wrong_depth(self):
        rgb, depth = self.fixture()
        camera = CameraCalibration()
        a = extract_features(rgb, depth, camera)
        b = extract_features(
            np.clip(rgb.astype(float) * 1.1, 0, 255).astype(np.uint8), depth, camera
        )
        pairs = retrieve_pairs([a] * 4 + [b])
        self.assertEqual((0, 4), pairs[0][1:3])
        pose = propose_transform(a, b, camera)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose, np.eye(4), atol=0.002)
        bad = extract_features(rgb, depth + 300, camera)
        self.assertIsNone(propose_transform(a, bad, camera))

    def test_weak_texture_blur_and_missing_depth_do_not_authorize_pose(self):
        rgb, depth = self.fixture()
        camera = CameraCalibration()
        source = extract_features(rgb, depth, camera)
        for image, metric in [
            (np.full_like(rgb, 100), depth),
            (rgb, np.zeros_like(depth)),
            (cv2.GaussianBlur(rgb, (41, 41), 15), depth),
        ]:
            target = extract_features(image, metric, camera)
            self.assertIsNone(propose_transform(source, target, camera))

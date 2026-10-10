import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scanner_server.appearance import Features
from scanner_server.joint_depth_bundle import refine_motion_graph, landmark_observations, observed_cameras
from shared.settings import CameraCalibration


def fixture():
    camera = CameraCalibration()
    rng = np.random.default_rng(9)
    world = rng.uniform([-.6, -.4, 1.8], [.6, .4, 2.8], (60, 3))
    poses, features = {}, []
    for i in range(3):
        pose = np.eye(4); pose[0, 3] = i*.05
        poses[i] = pose
        points = world-pose[:3, 3]
        pixels = points[:, :2]/points[:, 2, None]*[camera.fx, camera.fy]+[camera.cx, camera.cy]
        features.append(Features(pixels, points, None))
    matches = np.column_stack((np.arange(60), np.arange(60)))
    motion = SimpleNamespace(camera=camera, gravity=[None]*3, visual=False,
                             appearance_pair=lambda a, b: (np.linalg.inv(poses[b]) @ poses[a], matches),
                             features_for_pair=lambda a, b: (features[a], features[b], "test_identities"))
    edges = [{"source": a, "target": b, "transform": np.linalg.inv(poses[b]) @ poses[a]}
             for a, b in ((1, 0), (2, 1), (2, 0))]
    return poses, motion, edges, world


class JointDepthBundleTests(unittest.TestCase):
    def test_cameras_and_landmarks_recover_known_geometry_with_fixed_gauge(self):
        poses, motion, edges, _ = fixture()
        noisy = {i: p.copy() for i, p in poses.items()}
        noisy[1][1, 3] += .015; noisy[2][0, 3] += .025
        result, report = refine_motion_graph(noisy, edges, motion, max_seconds=10)
        self.assertTrue(report["applied"], report)
        self.assertLess(report["final_cost"], report["initial_cost"]*.001)
        np.testing.assert_array_equal(result[0], noisy[0])
        for i in poses:
            np.testing.assert_allclose(result[i], poses[i], atol=1e-4)
        self.assertAlmostEqual(.125, noisy[2][0, 3])

    def test_sparse_derivatives_match_numerical_feature_prior_gravity_and_surface_derivatives(self):
        poses, motion, edges, world = fixture()
        motion.gravity = [{"up": np.array([0., -1., 0.]), "confidence": .9, "verified": False}]*3
        surface = [(2, 1, world[:12]-poses[2][:3, 3], world[:12]-poses[1][:3, 3],
                    np.tile([0., 0., 1.], (12, 1)))]
        planes = [(np.array([0., 0., 2.5]), [(i, world[:15]-poses[i][:3, 3]) for i in range(3)])]

        def verify(residual, initial, jac, **kwargs):
            values = initial.copy(); values[:3] = [.02, -.03, .04]
            rng = np.random.default_rng(1)
            direction = rng.normal(size=len(values))
            epsilon = 1e-7
            numeric = (residual(values+direction*epsilon)-residual(values-direction*epsilon))/(2*epsilon)
            np.testing.assert_allclose(jac(values) @ direction, numeric, rtol=1e-5, atol=1e-5)
            return SimpleNamespace(nfev=1)

        with patch("scanner_server.joint_depth_bundle.depth_constraints", return_value=surface), \
                patch("scanner_server.joint_depth_bundle.plane_observations", return_value=planes), \
                patch("scanner_server.joint_depth_bundle.least_squares", side_effect=verify):
            refine_motion_graph(poses, edges, motion, views=[None]*3)

    def test_ambiguous_identity_cycle_is_discarded_in_full(self):
        poses, motion, edges, _ = fixture()
        original = motion.appearance_pair

        def conflict(a, b):
            pose, matches = original(a, b)
            matches = matches.copy()
            if (a, b) == (2, 0):
                matches[0, 0] = 1
            return pose, matches

        motion.appearance_pair = conflict
        owners, identities, _, _, landmarks, invalid = landmark_observations(poses, edges, motion)
        self.assertGreater(invalid, 0)
        self.assertLess(len(landmarks), 60)
        for identity in np.unique(identities):
            cameras = owners[identities == identity]
            self.assertEqual(len(cameras), len(np.unique(cameras)))

    def test_determination_requires_consistent_distributed_points_and_gauge_connection(self):
        poses, motion, edges, _ = fixture()
        owners, identities, pixels, depths, points, _ = landmark_observations(poses, edges, motion)
        local = np.array([points[identity]-poses[view][:3, 3] for view, identity in zip(owners, identities)])
        self.assertEqual([0, 1, 2], observed_cameras(owners, identities, pixels, depths, local, motion.camera))
        # Reprojection conflicts cannot certify an unconstrained camera.
        local[owners == 2, 0] += .1
        self.assertEqual([0, 1], observed_cameras(owners, identities, pixels, depths, local, motion.camera))
        # A disconnected island does not acquire the gauge camera's authority.
        keep = owners > 0
        self.assertEqual([0], observed_cameras(owners[keep], identities[keep], pixels[keep], depths[keep], local[keep], motion.camera))
        # Many collinear identities cannot determine all camera directions.
        points = np.tile(np.column_stack((np.linspace(-.5, .5, 60), np.zeros(60), np.full(60, 2.))), (3, 1))
        owners = np.repeat(np.arange(3), 60)
        identities = np.tile(np.arange(60), 3)
        pixels = points[:, :2]/points[:, 2, None]*[motion.camera.fx, motion.camera.fy]+[motion.camera.cx, motion.camera.cy]
        self.assertEqual([0], observed_cameras(owners, identities, pixels, points[:, 2], points, motion.camera))


if __name__ == "__main__":
    unittest.main()

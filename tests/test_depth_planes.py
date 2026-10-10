import unittest
from types import SimpleNamespace

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from scanner_server.depth_planes import measured_planes, plane_observations


class DepthPlaneTests(unittest.TestCase):
    def patch(self, distance=2.5):
        normal = np.array([.3, -.2, .93])
        normal /= np.linalg.norm(normal)
        across = np.cross(normal, [0., 1., 0.])
        across /= np.linalg.norm(across)
        along = np.cross(normal, across)
        x, y = np.meshgrid(np.linspace(-.7, .7, 30), np.linspace(-.5, .5, 30))
        return normal, normal*distance+x.ravel()[:, None]*across+y.ravel()[:, None]*along

    def view(self, points):
        return SimpleNamespace(features={}, cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))

    def test_oblique_measured_plane_agrees_across_known_camera_poses(self):
        normal, world = self.patch()
        views, poses = [None]*5, {}
        for node in (0, 2, 4):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('y', node*2, degrees=True).as_matrix()
            pose[:3, 3] = [node*.07, 0., 0.]
            poses[node] = pose
            local = (world-pose[:3, 3]) @ pose[:3, :3]
            views[node] = self.view(local)
            planes = measured_planes(views[node])
            self.assertEqual(1, len(planes))
            self.assertIs(planes, measured_planes(views[node]))
        result = plane_observations(poses, views)
        self.assertEqual(1, len(result))
        plane, observations = result[0]
        np.testing.assert_allclose(plane, normal*2.5, atol=1e-8)
        self.assertEqual({0, 2, 4}, {node for node, _ in observations})
        for node, points in observations:
            measured_world = points @ poses[node][:3, :3].T+poses[node][:3, 3]
            np.testing.assert_allclose(measured_world @ normal, 2.5, atol=1e-8)

    def test_pairwise_similarity_cannot_chain_different_parallel_surfaces(self):
        # Neighboring offsets differ by less than the proposal bound, but the
        # full group disagrees. Complete-link grouping must not invent a plane.
        views, poses = [None]*5, {}
        for node, distance in ((0, 2.5), (2, 2.6), (4, 2.7)):
            _, points = self.patch(distance)
            views[node] = self.view(points)
            poses[node] = np.eye(4)
        self.assertEqual([], plane_observations(poses, views))
        self.assertEqual([], plane_observations({5: np.eye(4)}, views))


if __name__ == '__main__':
    unittest.main()

import unittest
from types import SimpleNamespace

import numpy as np

from scanner_server.depth_surface_constraints import depth_constraints
from shared.settings import CameraCalibration


class DepthSurfaceConstraintTests(unittest.TestCase):
    def plane(self, depth=2.):
        c = CameraCalibration()
        x, y = np.meshgrid(np.arange(30, 610, 8), np.arange(30, 450, 8))
        points = np.column_stack(((x.ravel()-c.cx)*depth/c.fx, (y.ravel()-c.cy)*depth/c.fy,
                                  np.full(x.size, depth)))
        return SimpleNamespace(camera=c, depth=np.full((480, 640), depth), features={},
                               cloud=SimpleNamespace(points=points, normals=np.tile([0., 0., -1.], (len(points), 1))))

    def test_partial_plane_uses_target_measured_depth_and_omits_feature_only_cameras(self):
        views = [self.plane(), self.plane()]
        pose = np.eye(4); pose[2, 3] = .02
        pairs = depth_constraints({0: np.eye(4), 1: pose, 2: pose}, views)
        self.assertEqual(1, len(pairs))
        a, b, source, target, normals = pairs[0]
        self.assertEqual((1, 0), (a, b))
        self.assertGreater(len(source), 100)
        # The target supplies its own raw axial range. Current pose error is
        # a residual, not a manufactured depth correspondence.
        np.testing.assert_allclose(target[:, 2], 2.)
        residual = np.sum((source+pose[:3, 3]-target)*normals, axis=1)
        np.testing.assert_allclose(residual, -.02, atol=1e-12)

    def test_missing_depth_and_occluded_surfaces_do_not_add_correspondences(self):
        source, target = self.plane(), self.plane()
        target.depth[:] = 0
        self.assertEqual([], depth_constraints({0: np.eye(4), 1: np.eye(4)}, [target, source]))
        target.depth[:] = 1.
        self.assertEqual([], depth_constraints({0: np.eye(4), 1: np.eye(4)}, [target, source]))
        self.assertEqual([], depth_constraints({2: np.eye(4)}, [None, None]))


if __name__ == "__main__":
    unittest.main()

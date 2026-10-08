"""Repeated ICP proposals reuse source preparation without changing results."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine


class CountedSource:
    def __init__(self, cloud):
        self.cloud = cloud
        self.downsamples = 0

    def voxel_down_sample(self, voxel):
        self.downsamples += 1
        return self.cloud.voxel_down_sample(voxel)

    def __getattr__(self, name):
        return getattr(self.cloud, name)


def cloud():
    rng = np.random.default_rng(21)
    points = rng.uniform((-0.6, -0.5, 1.1), (0.6, 0.5, 2.2), (10000, 3))
    # Three intersecting surfaces constrain all six pose dimensions.
    points[:4000, 2] = 2.1
    points[4000:7000, 0] = -0.4
    points[7000:, 1] = 0.3
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    return result


class ICPSourceCacheTests(unittest.TestCase):
    def _check_repeated_proposals(self, tracking, device="cpu"):
        engine = ScanEngine(device=device, tracking=tracking)
        target = cloud()
        source = CountedSource(cloud())
        guesses = [np.eye(4) for _ in range(5)]
        for index, guess in enumerate(guesses):
            guess[0, 3] = index * 0.001
        expected = [engine._icp(source, target, guess) for guess in guesses]
        self.assertEqual(15, source.downsamples)
        source.downsamples = 0
        actual = []

        def proposals(current, rgbd):
            actual.extend(engine._icp(current, target, guess) for guess in guesses)
            return actual[-1]

        with patch.object(engine, "_visual_register", side_effect=proposals):
            result, method = engine._register(source)
        self.assertIs(result, actual[-1])
        self.assertEqual("keyframe+visual", method)
        self.assertEqual(3, source.downsamples)
        self.assertFalse(hasattr(engine, "_icp_source_pyramid"))
        tolerance = 1e-7 if tracking == "tensor" else 1e-12
        for before, after in zip(expected, actual):
            # Parallel reductions change last bits even on repeated uncached
            # calls (tensor point data is float32). All measured identities
            # must agree; the tensor pose tolerance is 0.1 micrometres.
            np.testing.assert_allclose(before.transformation, after.transformation,
                                       atol=tolerance, rtol=0)
            a, b = map(np.asarray, (before.correspondence_set, after.correspondence_set))
            np.testing.assert_array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)])
            self.assertEqual(before.fitness, after.fitness)
            self.assertLessEqual(abs(before.inlier_rmse - after.inlier_rmse), tolerance)
        # A later decision rebuilds source levels, including if the caller has
        # reused and changed the original cloud object between captures.
        source.cloud.translate((0.005, 0, 0))
        with patch.object(engine, "_visual_register", side_effect=proposals):
            engine._register(source)
        self.assertEqual(6, source.downsamples)
        self.assertLess(abs(actual[-1].transformation[0, 3] + 0.005), 1e-6)

    def test_legacy_proposals_preserve_transform_correspondences_and_support(self):
        self._check_repeated_proposals("legacy")

    def test_tensor_cpu_proposals_preserve_transform_correspondences_and_support(self):
        self._check_repeated_proposals("tensor")

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_tensor_cuda_proposals_preserve_transform_correspondences_and_support(self):
        self._check_repeated_proposals("tensor", "cuda")

    def test_failed_decision_discards_cache(self):
        engine = ScanEngine(device="cpu", tracking="legacy")
        source = CountedSource(cloud())
        target = cloud()

        def fail(current, rgbd):
            engine._icp(current, target, np.eye(4))
            raise RuntimeError("Injected proposal failure")

        with patch.object(engine, "_visual_register", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "Injected proposal failure"):
                engine._register(source)
        self.assertEqual(3, source.downsamples)
        self.assertFalse(hasattr(engine, "_icp_source_pyramid"))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.benchmark_confidence_fusion import (
    camera_for,
    depth_average_report,
    fuse_surface,
    legacy_confidence,
    observations,
    plane_depth,
)
from shared.confidence import depth_confidence


class DepthConfidenceTests(unittest.TestCase):
    def test_perspective_plane_incidence_matches_known_normal(self):
        camera = camera_for()
        y, x = np.indices((camera.height, camera.width))
        rx, ry = (x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy
        for angle in (0, 35, 55):
            with self.subTest(angle=angle):
                truth = plane_depth(camera, angle_degrees=angle)
                slope = np.tan(np.deg2rad(angle))
                cosine = (1 + slope * rx) / np.sqrt((1 + slope**2) * (1 + rx**2 + ry**2))
                interior = np.s_[8:-8, 8:-8]
                for quantized in (False, True):
                    measured = np.rint(truth * 1000).astype(np.uint16) if quantized else truth * 1000
                    actual = depth_confidence(measured, camera)
                    sigma = 0.001 + 0.002 * (measured / 1000)**2
                    expected = np.clip((0.003 / sigma)**2, 0.05, 1) * cosine**2
                    tolerance = 0.015 if quantized else 0.0001
                    self.assertLess(np.mean(np.abs(actual[interior] - expected[interior])), tolerance)

    def test_noisy_plane_retains_confidence_without_changing_measurements(self):
        camera = camera_for()
        depth = next(observations(plane_depth(camera), 1))
        unchanged = depth.copy()
        weight = depth_confidence(depth, camera)
        np.testing.assert_array_equal(depth, unchanged)
        self.assertGreater(float(weight[8:-8, 8:-8].mean()), 0.9)
        self.assertEqual(0, np.count_nonzero(weight[8:-8, 8:-8] == 0))
        self.assertEqual(np.float32, weight.dtype)
        self.assertTrue(weight.flags.c_contiguous)

    def test_holes_steps_outliers_and_small_surfaces_are_preserved(self):
        camera = camera_for()
        depth = np.full((camera.height, camera.width), 1000, np.uint16)
        depth[40:60, 30:50] = 0
        depth[:, 90:] = 1200
        depth[20, 20] = 1300
        weight = depth_confidence(depth, camera)
        self.assertTrue(np.all(weight[40:60, 30:50] == 0))
        self.assertTrue(np.all(weight[:, 89:91] == 0))
        self.assertEqual(0, weight[20, 20])
        # A fit must never borrow support from across the nearby hole/step.
        # Unknown orientation contributes conservatively; it is not erased.
        self.assertAlmostEqual(0.25, weight[50, 29], places=6)
        self.assertAlmostEqual(0.25, weight[60, 88], places=6)
        ribbon = np.zeros_like(depth)
        ribbon[20:100, 76:81] = 1000
        ribbon_weight = depth_confidence(ribbon, camera)
        self.assertGreater(ribbon_weight[60, 78], 0.99)
        self.assertGreater(ribbon_weight[60, 76], 0)
        self.assertTrue(np.all(ribbon_weight[ribbon == 0] == 0))

    def test_independent_repeated_measurements_reduce_error_and_add_coverage(self):
        camera = camera_for()
        truth = plane_depth(camera)
        four = depth_average_report(truth, camera, 4, depth_confidence)
        many = depth_average_report(truth, camera, 32, depth_confidence)
        legacy = depth_average_report(truth, camera, 4, legacy_confidence)
        self.assertLess(many["rmse_m"], four["rmse_m"] * 0.45)
        self.assertGreater(four["completeness_at_weight_2"], 0.95)
        self.assertEqual(1, many["completeness_at_weight_2"])
        self.assertLess(four["missing_aware_capped_rmse_m"], legacy["missing_aware_capped_rmse_m"] * 0.4)

    def test_oblique_repeated_measurements_improve_with_adequate_evidence(self):
        camera = camera_for()
        truth = plane_depth(camera, angle_degrees=55)
        actual = depth_average_report(truth, camera, 32, depth_confidence)
        legacy = depth_average_report(truth, camera, 32, legacy_confidence)
        self.assertGreater(actual["completeness_at_weight_2"], 0.99)
        self.assertLess(actual["rmse_m"], legacy["rmse_m"] * 0.9)

    def test_nonfinite_or_missing_depth_never_acquires_confidence(self):
        camera = camera_for(5, 4)
        depth = np.array(
            [[0, np.nan, np.inf, -10, 1000], [1000] * 5, [1000] * 5, [1000] * 5],
            dtype=np.float32,
        )
        weight = depth_confidence(depth, camera)
        self.assertTrue(np.all(np.isfinite(weight)))
        self.assertTrue(np.all(weight[0, :4] == 0))
        self.assertGreater(weight[0, 4], 0)

    def test_actual_tsdf_repeated_fusion_improves_surface_error_and_preserves_holes(self):
        camera = camera_for(120, 90)
        truth = plane_depth(camera)
        truth[30:50, 25:45] = 0
        truth[:, 75:] = 1.06
        actual = fuse_surface(truth, camera, 12, depth_confidence)
        legacy = fuse_surface(truth, camera, 12, legacy_confidence)
        self.assertGreater(actual["completeness_within_0_01m"], 0.99)
        self.assertEqual(0, actual["points_projecting_into_holes"])
        self.assertLess(actual["axial_rmse_m"], legacy["axial_rmse_m"] * 0.95)
        self.assertLessEqual(actual["missing_aware_capped_rmse_m"], legacy["missing_aware_capped_rmse_m"])

    def test_actual_tsdf_retains_fine_measured_surface_amplitude(self):
        camera = camera_for(120, 90)
        _, x = np.indices((camera.height, camera.width))
        truth = 1 + 0.01 * np.sin(2 * np.pi * x / 24)
        actual = fuse_surface(truth, camera, 16, depth_confidence, detail_period_pixels=24)
        self.assertGreater(actual["completeness_within_0_01m"], 0.99)
        self.assertGreater(actual["detail_amplitude_m"], 0.009)
        self.assertLess(actual["detail_amplitude_m"], 0.011)
        self.assertLess(actual["axial_rmse_m"], 0.001)

    def test_numpy_native_and_tensor_fusion_use_the_same_confidence(self):
        import open3d as o3d

        from scanner_server.weighted_fusion import _integrate_cpu, _integrate_tensor
        from shared.native import kernels

        core = o3d.core
        camera = SimpleNamespace(width=16, height=12, fx=8.3, fy=7.6, cx=7.5, cy=5.5)
        engine = SimpleNamespace(
            settings=SimpleNamespace(camera=camera), device=core.Device("CPU:0"),
            max_depth_m=1.5, sdf_trunc=0.08,
        )
        coordinates = np.stack(np.meshgrid(np.arange(-2, 2), np.arange(-3, 3), np.arange(3, 9), indexing="ij"), axis=-1)
        blocks = core.Tensor(coordinates.reshape(-1, 3).astype(np.int32))
        depth = np.full((12, 16), 1000, np.uint16)
        depth[4:6, 7:9] = 0
        depth[2, 3] = 1400
        rgb = np.full((12, 16, 3), 130, np.uint8)
        confidence = depth_confidence(depth, camera)
        modes = ["numpy", "tensor"]
        if kernels() is not None:
            modes.append("native")
        results = []
        for mode in modes:
            volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"), attr_dtypes=(core.float32,) * 3,
                attr_channels=((1,), (1,), (3,)), voxel_size=0.05,
                block_resolution=4, block_count=256, device=engine.device,
            )
            with patch.dict("os.environ", {"KINECT_NATIVE": "on" if mode == "native" else "off"}):
                integrate = _integrate_tensor if mode == "tensor" else _integrate_cpu
                for _ in range(2):
                    integrate(engine, volume, blocks, rgb, depth, np.eye(4), confidence)
            hashmap = volume.hashmap()
            ids = hashmap.active_buf_indices().numpy().astype(np.int64)
            keys = hashmap.key_tensor().numpy()[ids]
            order = np.lexsort(keys.T)
            results.append({
                name: volume.attribute(name).numpy().reshape(-1, 64, 3 if name == "color" else 1)[ids][order].copy()
                for name in ("tsdf", "weight", "color")
            })
        for result in results[1:]:
            for name in result:
                np.testing.assert_allclose(results[0][name], result[name], atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()

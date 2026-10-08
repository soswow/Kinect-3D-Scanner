"""Real-device parity, gating and shared-storage tests for fused CUDA TSDF."""

import os
import sys
from contextlib import nullcontext
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import open3d as o3d

from scanner_server.cuda_fusion import FusionUpdateError, selection, integrate, _kernel
from scanner_server.weighted_fusion import _integrate_tensor, integrate_weighted


class SelectionTests(unittest.TestCase):
    def test_optional_compiler_errors_are_reported_before_updates(self):
        class CompileFailure(Exception):
            pass

        compiled = Mock()
        compiled.compile.side_effect = CompileFailure("compiler unavailable")
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()),
                             RawKernel=Mock(return_value=compiled))
        with patch.dict(sys.modules, cupy=cp):
            module, kernel, reason = _kernel.__wrapped__(0)
        self.assertIsNone(module)
        self.assertIsNone(kernel)
        self.assertEqual("compiler unavailable", reason)

    def test_tensor_override_never_initializes_optional_cuda_compiler(self):
        with patch.dict(os.environ, KINECT_CUDA_FUSION="tensor"), patch(
                "scanner_server.cuda_fusion._kernel", side_effect=AssertionError("unexpected compilation")):
            self.assertEqual("tensor", selection(o3d.core.Device("CUDA:0"))[2]["implementation"])

    def test_failed_gpu_update_is_never_retried_with_tensor_updates(self):
        engine = SimpleNamespace(device=o3d.core.Device("CUDA:0"), backend={},
                                 settings=SimpleNamespace(camera=None))
        depth = np.full((2, 2), 1000, np.uint16)
        with patch("scanner_server.weighted_fusion.depth_confidence", return_value=np.ones((2, 2), np.float32)), patch(
                "scanner_server.cuda_fusion.selection", return_value=(object(), object(), {})), patch(
                "scanner_server.cuda_fusion.integrate", side_effect=RuntimeError("partial update failure")), patch(
                "scanner_server.weighted_fusion._integrate_tensor") as fallback:
            with self.assertRaisesRegex(RuntimeError, "partial update failure"):
                integrate_weighted(engine, None, None, np.zeros((2, 2, 3), np.uint8), depth, np.eye(4))
            fallback.assert_not_called()

    def test_explicit_request_never_silently_falls_back(self):
        device = o3d.core.Device("CUDA:0")
        with patch("scanner_server.cuda_fusion._kernel", return_value=(None, None, "missing")):
            with patch.dict(os.environ, KINECT_CUDA_FUSION="auto"):
                self.assertEqual("missing", selection(device)[2]["fallback_reason"])
            with patch.dict(os.environ, KINECT_CUDA_FUSION="fused"):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    selection(device)
        with patch.dict(os.environ, KINECT_CUDA_FUSION="invalid"):
            with self.assertRaises(ValueError):
                selection(device)


@unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
class CUDAFusionTests(unittest.TestCase):
    def test_failure_after_a_completed_chunk_is_reported_as_a_partial_update(self):
        device = o3d.core.Device("CUDA:0")
        with patch.dict(os.environ, KINECT_CUDA_FUSION="auto"):
            cp, kernel, status = selection(device)
        if cp is None:
            self.skipTest(status["fallback_reason"])
        core = o3d.core
        camera = SimpleNamespace(width=8, height=4, fx=1, fy=1, cx=3.5, cy=1.5)
        engine = SimpleNamespace(settings=SimpleNamespace(camera=camera), device=device,
                                 max_depth_m=1.5, sdf_trunc=.1)
        volume = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32,)*3, attr_channels=((1,), (1,), (3,)),
            voxel_size=.05, block_resolution=2, block_count=2048, device=device)
        coordinates = np.array([(x, y, 10) for x in range(-16, 17) for y in range(-16, 17)], np.int32)
        blocks = core.Tensor(coordinates, device=device)
        calls = []
        def fail_second_chunk(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("Injected second chunk failure")
            kernel(*args)
        with self.assertRaisesRegex(FusionUpdateError, "second chunk failure"):
            integrate(engine, volume, blocks, np.full((4, 8, 3), 130, np.uint8),
                np.full((4, 8), 1000, np.uint16), np.eye(4), np.ones((4, 8), np.float32), cp, fail_second_chunk)
        self.assertEqual(2, len(calls))
        self.assertGreater(float(volume.attribute("weight").cpu().numpy().sum()), 0)

    def test_half_pixel_rounding_and_out_of_image_voxels(self):
        device = o3d.core.Device("CUDA:0")
        with patch.dict(os.environ, KINECT_CUDA_FUSION="auto"):
            cp, kernel, status = selection(device)
        if cp is None:
            self.skipTest(status["fallback_reason"])
        core = o3d.core
        camera = SimpleNamespace(width=4, height=4, fx=1, fy=1, cx=0, cy=0)
        engine = SimpleNamespace(settings=SimpleNamespace(camera=camera),
            device=device, max_depth_m=1.5, sdf_trunc=0.1)
        volume = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32,)*3, attr_channels=((1,), (1,), (3,)),
            voxel_size=0.5, block_resolution=4, block_count=8, device=device)
        blocks = core.Tensor([[0, 0, 0], [-1, 0, 0]], dtype=core.int32, device=device)
        rgb = np.full((4, 4, 3), 130, np.uint8)
        depth = np.full((4, 4), 1000, np.uint16)
        confidence = np.zeros((4, 4), np.float32)
        confidence[0, 1] = 0.75
        integrate(engine, volume, blocks, rgb, depth, np.eye(4), confidence, cp, kernel)
        points, flat = volume.voxel_coordinates_and_flattened_indices()
        xyz, ids = points.cpu().numpy(), flat.cpu().numpy().reshape(-1)
        weight = volume.attribute("weight").cpu().numpy().reshape(-1)
        for x, expected in ((0.5, 0.75), (-0.5, 0)):
            selected = np.all(xyz == [x, 0, 1], axis=1)
            self.assertEqual(1, np.count_nonzero(selected))
            self.assertEqual(expected, weight[ids[selected][0]])

    def test_repeated_updates_match_tensor_with_holes_and_nonidentity_pose(self):
        with patch.dict(os.environ, KINECT_CUDA_FUSION="auto"):
            cp, kernel, status = selection(o3d.core.Device("CUDA:0"))
        if cp is None:
            self.skipTest(status["fallback_reason"])
        core = o3d.core
        camera = SimpleNamespace(width=16, height=12, fx=8.3, fy=7.6, cx=7.5, cy=5.5)
        engine = SimpleNamespace(settings=SimpleNamespace(camera=camera),
            device=core.Device("CUDA:0"), max_depth_m=1.5, sdf_trunc=0.08)
        coordinates = np.stack(np.meshgrid(np.arange(-2, 2), np.arange(-3, 3),
            np.arange(3, 9), indexing="ij"), axis=-1)
        blocks = core.Tensor(coordinates.reshape(-1, 3).astype(np.int32), device=engine.device)
        depth = np.full((12, 16), 1000, np.uint16)
        depth[4:6, 7:9] = 0
        depth[2, 3] = 1600  # beyond the depth gate
        rng = np.random.default_rng(31)
        rgb = rng.integers(0, 256, (12, 16, 3), np.uint8)
        confidence = rng.uniform(0, 1, (12, 16)).astype(np.float32)
        confidence[0, :] = 0
        confidence[1, 0] = np.nan
        for angle in (0, 0.013, -0.273):
            transform = np.eye(4)
            c, s = np.cos(angle), np.sin(angle)
            transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            transform[:3, 3] = [0.043, -0.012, 0.021]
            results = []
            for fused in (False, True):
                volume = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                    attr_dtypes=(core.float32,)*3, attr_channels=((1,), (1,), (3,)),
                    voxel_size=0.05, block_resolution=4, block_count=256, device=engine.device)
                for _ in range(3):
                    if fused:
                        integrate(engine, volume, blocks, rgb, depth, transform, confidence, cp, kernel)
                    else:
                        _integrate_tensor(engine, volume, blocks, rgb, depth, transform, confidence)
                h = volume.hashmap()
                ids = h.active_buf_indices().cpu().numpy().astype(np.int64)
                keys = h.key_tensor().cpu().numpy()[ids]
                order = np.lexsort(keys.T)
                results.append({name: volume.attribute(name).cpu().numpy().reshape(
                    -1, 64, 3 if name == "color" else 1)[ids][order].copy()
                    for name in ("tsdf", "weight", "color")})
            for name in results[0]:
                np.testing.assert_allclose(results[0][name], results[1][name], atol=2e-6, rtol=2e-6)
            self.assertTrue(np.any(results[1]["weight"] > 0))


if __name__ == "__main__":
    unittest.main()

"""Allocation follows extent and actual available memory, on CPU and CUDA."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scanner_server.fusion_memory import BYTES_PER_BLOCK, MIB, available_gpu_bytes, plan_fusion


class FusionMemoryTests(unittest.TestCase):
    def test_allocation_scales_with_extent_and_includes_workspace(self):
        with patch("scanner_server.fusion_memory.available_memory", return_value={"RAM": 8 * 1024 * MIB}):
            small = plan_fusion("CPU:0", 100, .005)
            large = plan_fusion("CPU:0", 13302, .005)
        self.assertEqual("automatic", large["allocation"])
        self.assertGreater(large["allocated_blocks"], 13302)
        self.assertLess(large["allocated_blocks"], 14000)
        self.assertAlmostEqual(large["attribute_budget_mib"], large["allocated_blocks"] * BYTES_PER_BLOCK / MIB)
        self.assertGreater(large["attribute_budget_mib"], small["attribute_budget_mib"])
        self.assertGreaterEqual(large["estimated_workspace_mib"], 256)

    def test_shortage_names_the_limiting_pool_and_retains_requested_resolution(self):
        for device, pools, name in (("CPU:0", {"RAM": 300 * MIB}, "RAM"),
                                    ("CUDA:0", {"RAM": 8 * 1024 * MIB, "GPU memory": 300 * MIB}, "GPU memory"),
                                    ("CUDA:0", {"RAM": 200 * MIB, "GPU memory": 8 * 1024 * MIB}, "RAM")):
            with self.subTest(device=device, name=name), \
                    patch("scanner_server.fusion_memory.available_memory", return_value=pools):
                with self.assertRaisesRegex(ValueError, f"Not enough {name} to reconstruct at 5 mm"):
                    plan_fusion(device, 13302, .005)

    def test_cuda_uses_context_memory_and_releases_only_unused_cupy_blocks(self):
        context = Mock()
        context.__enter__ = Mock()
        context.__exit__ = Mock(return_value=False)
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Mock(return_value=context),
                                                  runtime=SimpleNamespace(memGetInfo=Mock(return_value=(123456, 999999)))),
                             get_default_memory_pool=Mock(return_value=Mock()))
        with patch.dict(sys.modules, {"cupy": cp}):
            self.assertEqual(123456, available_gpu_bytes(SimpleNamespace(get_id=lambda: 2)))
        cp.cuda.Device.assert_called_once_with(2)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_called_once()

    def test_cuda_cli_fallback_maps_visible_device_and_has_a_timeout(self):
        with patch.dict(sys.modules, {"cupy": None}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}), \
                patch("scanner_server.fusion_memory.subprocess.run", return_value=SimpleNamespace(stdout="4096\n")) as run:
            self.assertEqual(4096 * MIB, available_gpu_bytes(SimpleNamespace(get_id=lambda: 1)))
        self.assertIn("GPU-second", run.call_args.args[0])
        self.assertEqual(3, run.call_args.kwargs["timeout"])

    def test_cuda_probe_failure_does_not_guess_an_arbitrary_block_budget(self):
        with patch.dict(sys.modules, {"cupy": None}), \
                patch("scanner_server.fusion_memory.subprocess.run", side_effect=OSError("no driver")):
            with self.assertRaisesRegex(RuntimeError, "Cannot read available GPU memory"):
                available_gpu_bytes(SimpleNamespace(get_id=lambda: 0))


if __name__ == "__main__":
    unittest.main()

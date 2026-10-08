"""CUDA input capability/fallback tests and optional real-device exact parity."""

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from scanner_server.cuda_input import CudaInputError, InputPreparation, preparation_signature, selection
from shared.calibration import prepare_rgbd
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings


def cpu_identity(rgb, raw, settings):
    return np.array(rgb,copy=True),np.array(raw,copy=True)


class FakeGpu:
    def __init__(self, settings):
        self.settings = settings
        self.host_calls = 0
        self.resident_calls = 0

    def prepare_host(self, rgb, raw):
        self.host_calls += 1
        return cpu_identity(rgb,raw,self.settings)

    def prepare(self, rgb, raw):
        self.resident_calls += 1
        return {"color":np.array(rgb,copy=True),"depth":np.array(raw,copy=True)}

    def synchronize(self):
        pass

    def metadata(self):
        return {"mocked":True}


class InputSelectionTests(unittest.TestCase):
    def test_default_is_off(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(selection("CUDA:0")["requested"],"off")

    def test_invalid_mode_and_explicit_cpu_backend_reject(self):
        with patch.dict(os.environ,KINECT_CUDA_INPUT="invalid"):
            with self.assertRaisesRegex(ValueError,"off, auto, or on"):
                selection("CUDA:0")
        with patch.dict(os.environ,KINECT_CUDA_INPUT="on"):
            with self.assertRaisesRegex(CudaInputError,"backend uses CPU"):
                selection("CPU:0")

    def test_auto_cpu_has_clear_reason(self):
        with patch.dict(os.environ,KINECT_CUDA_INPUT="auto"):
            self.assertIn("not selected",selection("CPU:0")["fallback_reason"])


class InputPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = ScanSettings(sensor_calibration=load_calibration(),rgb_mode="rgb_low_res")
        cls.rgb = np.full((480,640,3),83,np.uint8)
        cls.raw = np.full((480,640),800,np.uint16)

    def adapter(self, mode="on", factory=FakeGpu, device="CUDA:0"):
        with patch.dict(os.environ,KINECT_CUDA_INPUT=mode):
            return InputPreparation(device,{},factory=factory)

    def test_off_preserves_cpu_callable_and_does_not_construct_gpu(self):
        factory = Mock(side_effect=AssertionError("CuPy must not initialize"))
        adapter = self.adapter("off",factory)
        cpu = Mock(side_effect=cpu_identity)
        actual = adapter.prepare(self.rgb,self.raw,self.settings,cpu)
        cpu.assert_called_once_with(self.rgb,self.raw,self.settings)
        factory.assert_not_called()
        np.testing.assert_array_equal(actual[1],self.raw)
        self.assertEqual(adapter.status["cpu_batches"],1)
        self.assertEqual(adapter.status["confidence_device"],"CPU:0")

    def test_auto_registered_input_falls_back_once_per_configuration(self):
        adapter = self.adapter("auto",Mock(side_effect=AssertionError("No native calibration")))
        settings = ScanSettings()
        cpu = Mock(side_effect=cpu_identity)
        for _ in range(2):
            adapter.prepare(self.rgb,self.raw,settings,cpu)
        self.assertEqual(cpu.call_count,2)
        self.assertEqual(adapter.status["fallback_batches"],2)
        self.assertIn("calibrated native",adapter.status["fallback_reason"])

    def test_explicit_registered_input_fails_clearly(self):
        adapter = self.adapter()
        with self.assertRaisesRegex(CudaInputError,"calibrated native"):
            adapter.prepare(self.rgb,self.raw,ScanSettings(),cpu_identity)
        self.assertEqual(adapter.status["gpu_batches"],0)

    def test_optional_import_failure_is_recorded_without_retrying_each_frame(self):
        factory = Mock(side_effect=ImportError("CuPy unavailable"))
        adapter = self.adapter("auto",factory)
        for _ in range(2):
            adapter.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        factory.assert_called_once()
        self.assertIn("CuPy unavailable",adapter.status["fallback_reason"])

    def test_probe_failure_blocks_explicit_cuda_and_auto_retains_raw_input(self):
        class BadGpu(FakeGpu):
            def prepare_host(self,rgb,raw):
                colors,depth = super().prepare_host(rgb,raw)
                depth[0,0] += 1
                return colors,depth
        before = self.raw.copy()
        adapter = self.adapter("on",BadGpu)
        with self.assertRaisesRegex(CudaInputError,"probe changed"):
            adapter.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        auto = self.adapter("auto",BadGpu)
        actual = auto.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        np.testing.assert_array_equal(actual[1],self.raw)
        np.testing.assert_array_equal(self.raw,before)
        self.assertEqual(auto.status["implementation"],"cpu")
        self.assertEqual(auto.status["gpu_batches"],0)

    def test_success_checks_three_full_size_images_then_uses_gpu_without_cpu(self):
        adapter = self.adapter()
        cpu = Mock(side_effect=cpu_identity)
        adapter.prepare(self.rgb,self.raw,self.settings,cpu)
        self.assertEqual(adapter.status["probe_images"],3)
        self.assertEqual(cpu.call_count,3)
        cpu.reset_mock()
        adapter.prepare(self.rgb,self.raw,self.settings,cpu)
        cpu.assert_not_called()
        self.assertEqual(adapter.status["gpu_batches"],2)
        self.assertEqual(adapter.status["implementation"],"cuda")

    def test_resident_reuses_validation_without_downloads(self):
        adapter = self.adapter()
        adapter.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        cpu = Mock(side_effect=AssertionError("Resident output must not download or re-probe"))
        with patch.object(adapter._gpu,"prepare_host",side_effect=AssertionError("Unexpected host copy")):
            actual = adapter.prepare_resident(self.rgb,self.raw,self.settings,cpu)
        self.assertEqual(set(actual),{"color","depth"})
        self.assertEqual(adapter.status["resident_batches"],1)
        self.assertEqual(adapter._gpu.resident_calls,1)
        cpu.assert_not_called()

    def test_signature_invalidates_camera_roi_filter_range_and_mode_but_not_voxels(self):
        before = preparation_signature(self.settings)
        for change in ({"near_m":.7},{"far_m":3.5},{"filter_depth":False},
                       {"roi":(50,40,600,440)},{"rgb_mode":"rgb_high_res"}):
            with self.subTest(change=change):
                self.assertNotEqual(before,preparation_signature(replace(self.settings,**change)))
        changed_calibration = replace(self.settings.sensor_calibration,
            translation_mm=(20.,*self.settings.sensor_calibration.translation_mm[1:]))
        self.assertNotEqual(before,preparation_signature(replace(self.settings,sensor_calibration=changed_calibration)))
        self.assertEqual(before,preparation_signature(replace(self.settings,voxel_m=.01,final_voxel_m=.005)))

    def test_config_change_reprobes_and_new_session_clears_statistics(self):
        factory = Mock(side_effect=FakeGpu)
        adapter = self.adapter(factory=factory)
        adapter.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        first = factory.call_count
        adapter.prepare(self.rgb,self.raw,replace(self.settings,far_m=3.5),cpu_identity)
        self.assertEqual(factory.call_count,first+2)
        self.assertEqual(adapter.status["probe_images"],6)
        fresh = self.adapter(factory=factory)
        self.assertEqual(fresh.status["gpu_batches"],0)
        self.assertEqual(fresh.status["probe_images"],0)

    def test_invalid_signed_dtype_and_shape_fail_before_gpu_kernel(self):
        factory = Mock(side_effect=FakeGpu)
        for raw in (self.raw.astype(np.int16),self.raw[:-1]):
            adapter = self.adapter(factory=factory)
            with self.assertRaisesRegex(CudaInputError,"raw uint16 depth"):
                adapter.prepare(self.rgb,raw,self.settings,cpu_identity)
        factory.assert_not_called()

    def test_sync_failure_is_sticky_and_never_retried_on_cpu(self):
        adapter = self.adapter("auto")
        adapter.prepare(self.rgb,self.raw,self.settings,cpu_identity)
        cpu = Mock(side_effect=AssertionError("Unsafe CPU retry"))
        with patch.object(adapter._gpu,"prepare_host",side_effect=RuntimeError("kernel fault")), \
             patch.object(adapter._gpu,"synchronize",side_effect=RuntimeError("sticky fault")):
            with self.assertRaisesRegex(CudaInputError,"synchronization failed"):
                adapter.prepare(self.rgb,self.raw,self.settings,cpu)
        with self.assertRaisesRegex(CudaInputError,"synchronization failed"):
            adapter.prepare(self.rgb,self.raw,replace(self.settings,far_m=3.5),cpu)
        cpu.assert_not_called()


class RealCudaInputTests(unittest.TestCase):
    def setUp(self):
        import open3d as o3d
        if not o3d.core.cuda.is_available():
            self.skipTest("CUDA-enabled Open3D is unavailable")
        try:
            import cupy as cp
        except ImportError:
            self.skipTest("Optional CuPy is unavailable")
        self.cp = cp
        self.settings = ScanSettings(sensor_calibration=load_calibration(),rgb_mode="rgb_low_res")
        y,x = np.indices((480,640),dtype=np.int32)
        self.raw = (650+(x*3+y)%350).astype(np.uint16)
        self.raw[31:47,55:71] = 2047
        self.raw[200:209,300:309] = 65535
        self.rgb = np.random.default_rng(56).integers(0,256,(480,640,3),dtype=np.uint8)

    def test_real_host_parity_and_input_immutability(self):
        before = self.raw.copy()
        with patch.dict(os.environ,KINECT_CUDA_INPUT="on"):
            adapter = InputPreparation("CUDA:0",{})
        actual = adapter.prepare(self.rgb,self.raw,self.settings,prepare_rgbd)
        expected = prepare_rgbd(self.rgb,self.raw,self.settings)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(self.raw,before)
        self.assertTrue(adapter.status["probe_passed"])
        self.assertEqual(adapter.status["probe_images"],3)
        self.assertEqual(adapter.status["compatibility"]["cuda_device"],0)
        self.assertEqual(adapter.status["confidence_device"],"CPU:0")

    def test_real_resident_on_explicit_null_stream_avoids_host_copies(self):
        with patch.dict(os.environ,KINECT_CUDA_INPUT="on"):
            adapter = InputPreparation("CUDA:0",{})
        stream = self.cp.cuda.Stream(non_blocking=True)
        with stream:
            adapter.prepare(self.rgb,self.raw,self.settings,prepare_rgbd)
            with patch.object(adapter._gpu,"prepare_host",side_effect=AssertionError("Unexpected download")):
                actual = adapter.prepare_resident(self.rgb,self.raw,self.settings,prepare_rgbd)
        expected = prepare_rgbd(self.rgb,self.raw,self.settings)
        self.assertEqual(actual["color"].device.id,0)
        self.assertEqual(actual["depth"].device.id,0)
        np.testing.assert_array_equal(self.cp.asnumpy(actual["color"]),expected[0])
        np.testing.assert_array_equal(self.cp.asnumpy(actual["depth"]),expected[1])


if __name__=="__main__":
    unittest.main()

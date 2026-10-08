"""Exact confidence selection, probe/fallback lifecycle and optional CUDA parity."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from scanner_server.cuda_confidence import (
    ConfidencePreparation,CudaConfidenceUnsafeError,NativeConfidence,confidence_signature,selection)
from scanner_server.cuda_input import CudaInputError,InputPreparation
from shared.confidence import depth_confidence


def camera(**changes):
    values=dict(width=96,height=48,fx=59.1723,fy=60.6218,cx=47.3721,cy=23.6804)
    return SimpleNamespace(**{**values,**changes})


def cpu_identity(depth,camera):
    return np.ascontiguousarray(depth,dtype=np.float32)/np.float32(1000)


class FakeGpu:
    def __init__(self,camera):self.camera=camera;self.host_calls=0
    def prepare_host(self,depth):
        self.host_calls+=1
        return cpu_identity(depth,self.camera)
    def synchronize(self):pass
    def metadata(self):return {'mocked':True}


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.camera=camera()
        self.depth=np.full((48,96),900,np.uint16)
    def adapter(self,mode='on',factory=FakeGpu,device='CUDA:0',backend=None):
        with patch.dict(os.environ,KINECT_CUDA_CONFIDENCE=mode):
            return ConfidencePreparation(device,{} if backend is None else backend,factory=factory)
    def test_default_off_and_invalid_mode(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual('off',selection('CUDA:0')['requested'])
        with patch.dict(os.environ,KINECT_CUDA_CONFIDENCE='invalid'):
            with self.assertRaisesRegex(ValueError,'off, auto, or on'):selection('CUDA:0')
    def test_explicit_cpu_rejects_and_auto_cpu_records_reason(self):
        with patch.dict(os.environ,KINECT_CUDA_CONFIDENCE='on'):
            with self.assertRaisesRegex(CudaInputError,'backend uses CPU'):selection('CPU:0')
        adapter=self.adapter('auto',device='CPU:0')
        actual=adapter.prepare(self.depth,self.camera,cpu_identity)
        np.testing.assert_array_equal(actual,cpu_identity(self.depth,self.camera))
        self.assertIn('not selected',adapter.status['reason'])
        self.assertEqual(1,adapter.status['fallback_calls'])
    def test_off_preserves_original_callable_without_gpu_or_config_probe(self):
        factory=Mock(side_effect=AssertionError('No CUDA setup'))
        adapter=self.adapter('off',factory=factory)
        cpu=Mock(side_effect=cpu_identity)
        with patch('scanner_server.cuda_confidence.runtime_configuration',side_effect=AssertionError('No probe')):
            adapter.prepare(self.depth,self.camera,cpu)
        cpu.assert_called_once_with(self.depth,self.camera)
        factory.assert_not_called()
        self.assertEqual(1,adapter.status['cpu_calls'])
    def test_import_failure_falls_back_once_per_configuration(self):
        factory=Mock(side_effect=ImportError('CuPy unavailable'))
        adapter=self.adapter('auto',factory=factory)
        for _ in range(2):adapter.prepare(self.depth,self.camera,cpu_identity)
        factory.assert_called_once()
        self.assertEqual(2,adapter.status['fallback_calls'])
        self.assertIn('CuPy unavailable',adapter.status['reason'])
    def test_explicit_retry_preserves_original_failure_and_does_not_retry_factory(self):
        factory=Mock(side_effect=ImportError('CuPy unavailable'))
        adapter=self.adapter('on',factory=factory)
        reasons=[]
        for _ in range(3):
            with self.assertRaises(CudaInputError) as failure:
                adapter.prepare(self.depth,self.camera,cpu_identity)
            reasons.append(str(failure.exception))
        self.assertEqual([reasons[0]]*3,reasons)
        self.assertEqual(reasons[0],adapter.status['reason'])
        factory.assert_called_once()
    def test_strict_bit_probe_rejects_even_tiny_difference(self):
        class BadGpu(FakeGpu):
            def prepare_host(self,depth):
                result=super().prepare_host(depth);result.view(np.uint32)[0,1]^=1
                return result
        adapter=self.adapter(factory=BadGpu)
        with self.assertRaisesRegex(CudaInputError,'float32 bits'):adapter.prepare(self.depth,self.camera,cpu_identity)
        auto=self.adapter('auto',factory=BadGpu)
        actual=auto.prepare(self.depth,self.camera,cpu_identity)
        np.testing.assert_array_equal(actual,cpu_identity(self.depth,self.camera))
        self.assertEqual(0,auto.status['gpu_calls'])
    def test_nonfinite_probe_rejected_before_fusion(self):
        class BadGpu(FakeGpu):
            def prepare_host(self,depth):
                result=super().prepare_host(depth);result[0,0]=np.nan;return result
        with self.assertRaisesRegex(CudaInputError,'nonfinite'):
            self.adapter(factory=BadGpu).prepare(self.depth,self.camera,cpu_identity)
    def test_success_runs_three_probes_then_gpu_without_cpu(self):
        adapter=self.adapter();cpu=Mock(side_effect=cpu_identity)
        adapter.prepare(self.depth,self.camera,cpu)
        self.assertEqual(3,cpu.call_count)
        self.assertEqual(3,adapter.status['compatibility_probes'])
        cpu.reset_mock();adapter.prepare(self.depth,self.camera,cpu)
        cpu.assert_not_called();self.assertEqual(2,adapter.status['gpu_calls'])
        self.assertEqual('CUDA:0',adapter.status['device'])
    def test_camera_and_installed_runtime_changes_reprobe(self):
        factory=Mock(side_effect=FakeGpu);adapter=self.adapter(factory=factory)
        adapter.prepare(self.depth,self.camera,cpu_identity)
        changed=camera(fx=58.31)
        adapter.prepare(self.depth,changed,cpu_identity)
        self.assertEqual(2,factory.call_count)
        with patch('scanner_server.cuda_confidence.runtime_configuration',return_value={'different':'runtime'}):
            adapter.prepare(self.depth,changed,cpu_identity)
        self.assertEqual(3,factory.call_count)
        self.assertEqual(9,adapter.status['compatibility_probes'])
    def test_signature_uses_only_confidence_pinhole_fields(self):
        original=confidence_signature(self.camera)
        self.camera.distortion=(.1,.2,0,0,0)
        self.assertEqual(original,confidence_signature(self.camera))
        self.assertNotEqual(original,confidence_signature(camera(cx=49)))
    def test_new_session_clears_stats_and_device_metadata(self):
        backend={'cuda_input':{'confidence_device':'CPU:0'}}
        adapter=self.adapter(backend=backend);adapter.prepare(self.depth,self.camera,cpu_identity)
        self.assertEqual('CUDA:0',backend['cuda_input']['confidence_device'])
        self.assertEqual('CUDA:0',backend['stage_devices']['depth_confidence'])
        fresh=self.adapter(backend=backend)
        self.assertEqual(0,fresh.status['gpu_calls'])
        self.assertEqual('CPU:0',backend['cuda_input']['confidence_device'])
    def test_input_metadata_preserves_independent_confidence_device(self):
        backend={'depth_confidence':{'device':'CUDA:0'}}
        with patch.dict(os.environ,KINECT_CUDA_INPUT='off'):
            adapter=InputPreparation('CUDA:0',backend)
        self.assertEqual('CUDA:0',adapter.status['confidence_device'])
        self.assertEqual('CUDA:0',backend['stage_devices']['depth_confidence'])
    def test_invalid_dtype_dimensions_and_camera_reject_before_factory(self):
        factory=Mock(side_effect=FakeGpu)
        for depth,c in ((self.depth.astype(np.int16),self.camera),(self.depth[:-1],self.camera),
                        (self.depth,camera(fx=0)),(self.depth,camera(fx=1e-80)),
                        (np.zeros((2,3),np.uint16),camera(width=3,height=2))):
            with self.subTest(camera=c):
                with self.assertRaises(CudaInputError):self.adapter(factory=factory).prepare(depth,c,cpu_identity)
        factory.assert_not_called()
    def test_sticky_sync_failure_never_retries_on_cpu_or_new_camera(self):
        adapter=self.adapter('auto');adapter.prepare(self.depth,self.camera,cpu_identity)
        cpu=Mock(side_effect=AssertionError('No CPU retry'))
        with patch.object(adapter._gpu,'prepare_host',side_effect=RuntimeError('Kernel fault')), \
             patch.object(adapter._gpu,'synchronize',side_effect=RuntimeError('Sticky failure')):
            with self.assertRaisesRegex(CudaInputError,'synchronization failed'):adapter.prepare(self.depth,self.camera,cpu)
        with self.assertRaisesRegex(CudaInputError,'synchronization failed'):adapter.prepare(self.depth,camera(fx=52),cpu)
        cpu.assert_not_called()
    def test_constructor_sticky_failure_is_hard_even_in_auto(self):
        factory=Mock(side_effect=CudaConfidenceUnsafeError('Initialization synchronization failed'))
        adapter=self.adapter('auto',factory=factory);cpu=Mock()
        for c in (self.camera,camera(fx=52)):
            with self.assertRaisesRegex(CudaInputError,'Initialization'):adapter.prepare(self.depth,c,cpu)
        factory.assert_called_once();cpu.assert_not_called()
    def test_weighted_fusion_off_keeps_patchable_cpu_callable(self):
        from scanner_server.weighted_fusion import integrate_weighted
        adapter=self.adapter('off',device='CPU:0')
        engine=SimpleNamespace(device='CPU:0',settings=SimpleNamespace(camera=self.camera),_confidence_preparation=adapter)
        expected=np.ones(self.depth.shape,np.float32)
        with patch('scanner_server.weighted_fusion.depth_confidence',return_value=expected) as cpu, \
             patch('scanner_server.weighted_fusion._integrate_cpu') as integrate:
            integrate_weighted(engine,object(),object(),np.zeros((*self.depth.shape,3),np.uint8),self.depth,np.eye(4))
        cpu.assert_called_once_with(self.depth,self.camera)
        self.assertIs(expected,integrate.call_args.args[-1])
    def test_weighted_fusion_hard_failure_occurs_before_volume_write(self):
        from scanner_server.weighted_fusion import integrate_weighted
        adapter=SimpleNamespace(prepare=Mock(side_effect=CudaInputError('Explicit confidence failure')))
        engine=SimpleNamespace(device='CPU:0',settings=SimpleNamespace(camera=self.camera),_confidence_preparation=adapter)
        with patch('scanner_server.weighted_fusion._integrate_cpu') as integrate:
            with self.assertRaises(CudaInputError):integrate_weighted(engine,object(),object(),None,self.depth,np.eye(4))
        integrate.assert_not_called()


class RealCudaConfidenceTests(unittest.TestCase):
    def setUp(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1:self.skipTest('CUDA unavailable')
        except (ImportError,RuntimeError) as exc:self.skipTest(str(exc))
        self.cp=cp
        from shared.settings import ScanSettings
        self.camera=ScanSettings().camera
        y,x=np.indices((self.camera.height,self.camera.width))
        self.depth=(850+x//8+y//12).astype(np.uint16)
        self.depth[21:24,31:34]=0
    def test_real_exact_host_probes_and_immutable_depth(self):
        before=self.depth.copy()
        with patch.dict(os.environ,KINECT_CUDA_CONFIDENCE='on'):
            adapter=ConfidencePreparation('CUDA:0',{})
        actual=adapter.prepare(self.depth,self.camera,depth_confidence)
        expected=depth_confidence(self.depth,self.camera)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_array_equal(self.depth,before)
        self.assertEqual(3,adapter.status['compatibility_probes'])
        self.assertEqual(0,adapter.status['compatibility']['cuda_device'])
    def test_real_explicit_device_and_null_stream(self):
        stream=self.cp.cuda.Stream(non_blocking=True)
        with stream:
            gpu=NativeConfidence(self.camera,0)
            actual=gpu.prepare_host(self.depth[::-1])
            gpu.synchronize()
        expected=depth_confidence(self.depth[::-1],self.camera)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


if __name__=='__main__':unittest.main()

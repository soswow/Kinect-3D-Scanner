"""Opt-in exact CUDA engineering confidence, with bounded compatibility checks.

The equations and conservative support/zero gates are shared.confidence's CPU
reference. Row/column FMA order follows OpenCV 5.0.0 filter.simd.hpp; integer
squares use rounded multiplication because CuPy 13.6.0 power calls powf.
Installed OpenCV dispatch is checked empirically before any fusion update.

Official source provenance:
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.simd.hpp
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.dispatch.cpp
https://github.com/cupy/cupy/blob/v13.6.0/cupy/_core/_routines_math.pyx

All 290 supplied archive views and boundary probes matched every confidence
float32 bit during development. Bounded probes cannot prove all future inputs
or all library installations. Default off preserves the original CPU callable;
auto records a safe fallback; on raises when compatibility cannot be shown.
"""
import hashlib
import json
import os
import time

import numpy as np

from .cuda_input import CudaInputError

class CudaConfidenceUnsafeError(CudaInputError):
    """CUDA cleanup failed while allocating or computing temporary weights."""

FILTER_SOURCE = r'''
__device__ int mirror(int p, int n) {
    if (p<0) return -p;
    if (p>=n) return 2*n-2-p;
    return p;
}
extern "C" __global__ void row_filter(
    const float* input, float* output, const float* weights,
    int width, int height, int size, int derivative) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=width*height) return;
    int x=i%width,y=i/width,r=size/2;
    if (size==3) {
        float left=input[y*width+mirror(x-1,width)];
        float center=input[i],right=input[y*width+mirror(x+1,width)];
        output[i]=derivative ? (right-left)*weights[2]
            : fmaf(center,weights[1],(left+right)*weights[2]);
    } else {
        float sum=0.0f;
        for (int k=0;k<size;++k)
            sum=fmaf(input[y*width+mirror(x+k-r,width)],weights[k],sum);
        output[i]=sum;
    }
}
extern "C" __global__ void column_filter(
    const float* input, float* output, const float* weights,
    int width, int height, int size, int derivative) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=width*height) return;
    int x=i%width,y=i/width,r=size/2;
    float sum=derivative ? 0.0f : input[i]*weights[r];
    for (int k=1;k<=r;++k) {
        float top=input[mirror(y-k,height)*width+x];
        float bottom=input[mirror(y+k,height)*width+x];
        sum=fmaf(derivative ? bottom-top : bottom+top,weights[r+k],sum);
    }
    output[i]=sum;
}
extern "C" __global__ void moment_rows(
    const float* input, float* average_row, float* derivative_row,
    const float* average, const float* derivative,
    int width, int height, int size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=width*height) return;
    int x=i%width,y=i/width,r=size/2;
    if (size==3) {
        float left=input[y*width+mirror(x-1,width)];
        float center=input[i],right=input[y*width+mirror(x+1,width)];
        average_row[i]=fmaf(center,average[1],(left+right)*average[2]);
        derivative_row[i]=(right-left)*derivative[2];
    } else {
        float mean=0.0f,slope=0.0f;
        for (int k=0;k<size;++k) {
            float sample=input[y*width+mirror(x+k-r,width)];
            mean=fmaf(sample,average[k],mean);
            slope=fmaf(sample,derivative[k],slope);
        }
        average_row[i]=mean;
        derivative_row[i]=slope;
    }
}
extern "C" __global__ void moment_columns(
    const float* average_row, const float* derivative_row,
    float* mean, float* slope_x, float* slope_y,
    const float* average, const float* derivative,
    int width, int height, int size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=width*height) return;
    int x=i%width,y=i/width,r=size/2;
    float m=average_row[i]*average[r];
    float a=derivative_row[i]*average[r];
    float b=0.0f;
    for (int k=1;k<=r;++k) {
        int top=mirror(y-k,height)*width+x,bottom=mirror(y+k,height)*width+x;
        float mt=average_row[top],mb=average_row[bottom];
        m=fmaf(mt+mb,average[r+k],m);
        a=fmaf(derivative_row[top]+derivative_row[bottom],average[r+k],a);
        b=fmaf(mb-mt,derivative[r+k],b);
    }
    mean[i]=m;slope_x[i]=a;slope_y[i]=b;
}
'''

class NativeConfidence:
    """Temporary images on the selected CUDA device and the explicit null stream."""
    def __init__(self,camera,device=0):
        import cupy as cp
        self.cp,self.camera=cp,camera
        self.cuda_device=cp.cuda.Device(device)
        self.shape=(camera.height,camera.width)
        self.count=camera.width*camera.height
        self.launch=((self.count+255)//256,),(256,)
        try:
            with self.cuda_device,cp.cuda.Stream.null:
                module=cp.RawModule(code=FILTER_SOURCE,options=('--std=c++11','--fmad=false'))
                self.moment_rows=module.get_function('moment_rows')
                self.moment_columns=module.get_function('moment_columns')
                self.coefficients={}
                for size in (7,3):
                    offsets=np.arange(size,dtype=np.float32)-size//2
                    derivative=offsets/np.dot(offsets,offsets)
                    average=np.full(size,1/size,np.float32)
                    self.coefficients[size]=(cp.asarray(average),cp.asarray(derivative))
        except Exception as exc:
            try:
                self.synchronize()
            except Exception as cleanup:
                raise CudaConfidenceUnsafeError(f'CUDA confidence initialization synchronization failed: {cleanup}') from exc
            raise

    def moments(self,image,size):
        cp=self.cp
        average,derivative=self.coefficients[size]
        row,dx=cp.empty_like(image),cp.empty_like(image)
        mean,a,b=(cp.empty_like(image) for _ in range(3))
        h,w=image.shape
        dimensions=(np.int32(w),np.int32(h),np.int32(size))
        self.moment_rows(*self.launch,(image,row,dx,average,derivative,*dimensions))
        self.moment_columns(*self.launch,(row,dx,mean,a,b,average,derivative,*dimensions))
        return mean,a,b

    def prepare_host(self,depth):
        with self.cuda_device,self.cp.cuda.Stream.null:
            device_depth=self.cp.ascontiguousarray(self.cp.asarray(depth))
            return self.cp.asnumpy(self._calculate(device_depth))

    def synchronize(self):
        with self.cuda_device,self.cp.cuda.Stream.null:
            self.cp.cuda.Stream.null.synchronize()

    def metadata(self):
        return {**runtime_configuration(),'cupy':self.cp.__version__,
            'cuda_device':self.cuda_device.id,'reduction_order':'opencv-5.0.0-symmetric-fma',
            'squares':'float32-multiply','filters':'fused-three-moments'}

    def _calculate(self, depth):
        from cupyx.scipy import ndimage
        cp, c = (self.cp, self.camera)
        z = depth.astype(cp.float32) / cp.float32(1000)
        valid = cp.isfinite(z) & (z > 0)
        z = cp.where(valid, z, cp.float32(0))
        padded = cp.pad(z, 1)
        edge = cp.zeros(self.shape, cp.bool_)
        tolerance = cp.maximum(cp.float32(0.02), cp.float32(0.02) * z)
        for neighbour in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
            edge |= (neighbour > 0) & (cp.abs(neighbour - z) > tolerance)
        inverse = cp.where(valid, cp.float32(1) / cp.where(valid, z, cp.float32(1)), cp.float32(0))
        y, x = cp.indices(self.shape, dtype=cp.float32)
        rx, ry = ((x - cp.float32(c.cx)) / cp.float32(c.fx), (y - cp.float32(c.cy)) / cp.float32(c.fy))
        ray_length = cp.sqrt(cp.float32(1) + cp.square(rx) + cp.square(ry))
        cosine, fitted = (cp.full(self.shape, 0.5, cp.float32), cp.zeros(self.shape, cp.bool_))
        for size in (7, 3):
            complete = ndimage.minimum_filter(valid, size=size, mode='constant', cval=0)
            crossing = ndimage.maximum_filter(edge, size=size, mode='constant', cval=0)
            use = complete & ~crossing & ~fitted
            mean, a, b = self.moments(inverse, size)
            a = a * cp.float32(c.fx)
            b = b * cp.float32(c.fy)
            intercept = mean - a * rx - b * ry
            length = cp.sqrt(a * a + b * b + intercept * intercept)
            estimate = mean / cp.maximum(length * ray_length, cp.float32(1e-10))
            cosine = cp.where(use, cp.clip(estimate, 0, 1), cosine)
            fitted |= use
        sigma = cp.float32(0.001) + cp.float32(0.002) * cp.square(z)
        weight = cp.clip(cp.square(cp.float32(0.003) / sigma), 0.05, 1) * cp.square(cosine)
        weight = cp.where(~valid | edge | fitted & (cosine < cp.float32(0.25)), cp.float32(0), weight)
        return cp.ascontiguousarray(weight, dtype=cp.float32)


def selection(device):
    requested=os.environ.get('KINECT_CUDA_CONFIDENCE','off').lower()
    if requested not in ('off','auto','on'):
        raise ValueError('KINECT_CUDA_CONFIDENCE must be off, auto, or on')
    cuda=str(device).startswith('CUDA:')
    if requested=='on' and not cuda:
        raise CudaInputError('CUDA confidence requested but the scanning backend uses CPU')
    return {'requested':requested,'implementation':'cpu','device':'CPU:0',
        'gpu_calls':0,'cpu_calls':0,'fallback_calls':0,'compatibility_probes':0,
        'setup_ms':0.0,'configuration_sha256':None,'probe_passed':None,
        'reason':'CUDA scanning backend not selected' if requested=='auto' and not cuda else None}


def runtime_configuration():
    import cv2
    configuration=getattr(np.__config__,'CONFIG',{})
    numpy_build=hashlib.sha256(json.dumps(configuration,sort_keys=True,default=str).encode()).hexdigest()
    return {'numpy':np.__version__,'numpy_build_sha256':numpy_build,'opencv':cv2.__version__,
        'opencv_build_sha256':hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        'opencv_optimized':cv2.useOptimized(),'opencv_threads':cv2.getNumThreads(),
        'opencv_ipp_enabled':cv2.ipp.useIPP(),'opencv_ipp_version':cv2.ipp.getIppVersion(),
        'opencv_ipp_not_exact':cv2.ipp.useIPP_NotExact() if hasattr(cv2.ipp,'useIPP_NotExact') else None}


def confidence_signature(camera):
    camera_fields={name:getattr(camera,name) for name in ('width','height','fx','fy','cx','cy')}
    value={'camera':camera_fields,'runtime':runtime_configuration()}
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def supported_input(depth,camera):
    if not isinstance(depth,np.ndarray) or depth.dtype!=np.uint16 or depth.shape!=(camera.height,camera.width):
        raise ValueError('CUDA confidence requires prepared uint16 metric depth with camera dimensions')
    if min(camera.width,camera.height)<4 or camera.width*camera.height>4_194_304:
        raise ValueError('CUDA confidence supports image dimensions >= 4 and at most 4194304 pixels')
    values=np.asarray([camera.fx,camera.fy,camera.cx,camera.cy],np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values)>np.finfo(np.float32).max):
        raise ValueError('CUDA confidence requires finite float32 pinhole parameters and positive focal lengths')
    if min(values.astype(np.float32)[:2])<=0:
        raise ValueError('CUDA confidence requires finite float32 pinhole parameters and positive focal lengths')


class ConfidencePreparation:
    """Session-owned checked weights; setup/fallback completes before mutation."""
    def __init__(self,device,backend,*,factory=None):
        self.device,self.backend=str(device),backend
        self.status=selection(device)
        self.backend['depth_confidence']=self.status
        self._factory=factory or (lambda camera:NativeConfidence(camera,int(self.device.split(':')[1])))
        self._gpu=None
        self._signature=None
        self._failed_reason=None
        self._unsafe_failure=False
        self._stage_device('CPU:0')

    def _stage_device(self,value):
        self.backend.setdefault('stage_devices',{})['depth_confidence']=value
        self.backend.get('cuda_input',{})['confidence_device']=value
        self.status.update(device=value,implementation='cuda' if value.startswith('CUDA:') else 'cpu')

    def _cpu(self,depth,camera,cpu_prepare,*,fallback=False):
        self.status['cpu_calls']+=1
        self.status['fallback_calls']+=int(fallback)
        return cpu_prepare(depth,camera)

    def _compare(self,depth,camera,cpu_prepare,label):
        expected=cpu_prepare(depth,camera)
        actual=self._gpu.prepare_host(depth)
        self.status['compatibility_probes']+=1
        if expected.dtype!=np.float32 or actual.dtype!=np.float32 or expected.shape!=actual.shape:
            raise CudaInputError(f'CUDA confidence {label} probe changed weight shape or dtype')
        if not np.all(np.isfinite(actual)):
            raise CudaInputError(f'CUDA confidence {label} probe produced nonfinite weights')
        changed=int(np.count_nonzero(expected.view(np.uint32)!=actual.view(np.uint32)))
        if changed:
            masks=int(np.count_nonzero((expected==0)!=(actual==0)))
            raise CudaInputError(f'CUDA confidence {label} probe changed {changed} float32 bits and {masks} zero masks')
        return actual

    def _numeric_probes(self,camera,cpu_prepare):
        y,x=np.indices((camera.height,camera.width),dtype=np.float32)
        rx=(x-np.float32(camera.cx))/np.float32(camera.fx)
        inverse=np.maximum(np.float32(.1),np.float32(1)+np.float32(np.sqrt(15))*rx)
        plane=np.clip(np.rint(np.float32(1000)/inverse),1,65535).astype(np.uint16)
        plane[:2]=0
        plane[10:12,10:12]=0
        plane[-12:-4,-12:-4]=65535
        self._compare(plane,camera,cpu_prepare,'incidence threshold/support')
        depth=(850+(x.astype(np.int32)+y.astype(np.int32)//2)%100).astype(np.uint16)
        depth[::31,::29]=0
        depth[-4:]=0
        depth[30:34,30:34]=1
        depth[50:55,50:55]=65535
        self._compare(depth,camera,cpu_prepare,'numeric/border/hole/discontinuity')

    def _failure(self,exc):
        if isinstance(exc,CudaConfidenceUnsafeError):
            self._unsafe_failure=True
        if self._gpu is not None:
            try:self._gpu.synchronize()
            except Exception as sync_exc:
                self._unsafe_failure=True
                exc=CudaConfidenceUnsafeError(f'CUDA confidence synchronization failed: {sync_exc}')
        self._failed_reason=f'CUDA confidence unavailable: {exc}'
        self.status.update(reason=self._failed_reason,probe_passed=False)
        self._gpu=None
        self._stage_device('CPU:0')
        if self._unsafe_failure or self.status['requested']=='on':
            raise CudaInputError(self._failed_reason) from exc

    def prepare(self,depth,camera,cpu_prepare):
        mode=self.status['requested']
        if mode=='off':return self._cpu(depth,camera,cpu_prepare)
        if not self.device.startswith('CUDA:'):
            return self._cpu(depth,camera,cpu_prepare,fallback=True)
        if self._unsafe_failure:raise CudaInputError(self._failed_reason)
        try:
            signature=confidence_signature(camera)
        except Exception as exc:
            self._failure(exc)
            return self._cpu(depth,camera,cpu_prepare,fallback=True)
        if signature!=self._signature:
            self._gpu=None
            self._signature,self._failed_reason=signature,None
            self.status.update(configuration_sha256=signature,reason=None,probe_passed=None)
            self._stage_device('CPU:0')
        if self._failed_reason is not None:
            if mode=='on':raise CudaInputError(self._failed_reason)
            return self._cpu(depth,camera,cpu_prepare,fallback=True)
        started=time.perf_counter()
        setup=False
        try:
            supported_input(depth,camera)
            setup=self._gpu is None
            if setup:
                self._gpu=self._factory(camera)
                self._numeric_probes(camera,cpu_prepare)
                actual=self._compare(depth,camera,cpu_prepare,'first observation')
                self.status.update(compatibility=self._gpu.metadata(),probe_passed=True)
                self._stage_device(self.device)
            else:
                actual=self._gpu.prepare_host(depth)
            self.status['gpu_calls']+=1
            return actual
        except Exception as exc:
            self._failure(exc)
            return self._cpu(depth,camera,cpu_prepare,fallback=True)
        finally:
            if setup:self.status['setup_ms']+=(time.perf_counter()-started)*1000

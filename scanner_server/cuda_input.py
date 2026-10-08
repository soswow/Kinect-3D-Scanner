"""Opt-in native RGB/depth CUDA preparation before any volume mutation.

Only measured disparity conversion, image rectification, conservative depth
rejection, lens projection/occlusion and color sampling run on CUDA. Original
CPU depth_confidence and all geometric authorization gates remain unchanged.
Compatibility probes compare exact CPU outputs for native dimensions/current
calibration and a known numeric boundary before accepting a GPU result.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np

from shared.calibration import pinhole_rays, raw_depth_lut, rectification_maps

CUDA_SOURCE = r'''
extern "C" __global__ void rectify_clip(
    const unsigned short* raw, const int* source, const double* lut,
    double* metric, unsigned short* clipped, int count, int width,
    double near_mm, double far_mm, int x0, int y0, int x1, int y1) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int p = source[i];
    unsigned short code = p < 0 ? 2047 : raw[p];
    double mm = code <= 2047 ? lut[code] : 0.0;
    metric[i] = mm;
    int rounded = __double2int_rn(mm);
    int x = i % width, y = i / width;
    clipped[i] = rounded > 0 && rounded >= near_mm && rounded <= far_mm &&
        x >= x0 && x < x1 && y >= y0 && y < y1 ? rounded : 0;
}
extern "C" __global__ void reject_isolated(
    const unsigned short* clipped, unsigned short* depth,
    int count, int width, int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int value = clipped[i];
    if (!value) { depth[i] = 0; return; }
    int x = i % width, y = i / width, agree = 0;
    double tolerance = fmax(15.0, value * 0.01);
    int q;
    if (y > 0) { q = clipped[i-width]; agree += q > 0 && abs(q-value) <= tolerance; }
    if (y+1 < height) { q = clipped[i+width]; agree += q > 0 && abs(q-value) <= tolerance; }
    if (x > 0) { q = clipped[i-1]; agree += q > 0 && abs(q-value) <= tolerance; }
    if (x+1 < width) { q = clipped[i+1]; agree += q > 0 && abs(q-value) <= tolerance; }
    depth[i] = agree >= 2 ? value : 0;
}
extern "C" __global__ void project_occlusion(
    const double* metric, const unsigned short* depth, const double* rays,
    const double* transform, const double* camera, float* map_x,
    float* map_y, double* rgb_z, int* rgb_index, unsigned long long* nearest,
    int count, int rgb_width, int rgb_height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    double mm = metric[i];
    double px = rays[i*3] * mm, py = rays[i*3+1] * mm, pz = mm;
    // Match the current NumPy/BLAS three-term dot's accumulation at the tested
    // half-bin regression. Projection below retains the native unfused math.
    double point_x = fma(pz,transform[2],fma(py,transform[1],px*transform[0])) + transform[9];
    double point_y = fma(pz,transform[5],fma(py,transform[4],px*transform[3])) + transform[10];
    double point_z = fma(pz,transform[8],fma(py,transform[7],px*transform[6])) + transform[11];
    double inverse_z = point_z != 0.0 ? 1.0 / point_z : 1.0;
    double x = point_x * inverse_z, y = point_y * inverse_z;
    double xx = x*x, yy = y*y, xy = x*y, r2 = xx+yy;
    double radial = 1.0 + r2 * (camera[4] + r2 * (camera[5] + r2*camera[8]));
    double u = (x*radial + 2.0*camera[6]*xy + camera[7]*(r2+2.0*xx))*camera[0] + camera[2];
    double v = (y*radial + camera[6]*(r2+2.0*yy) + 2.0*camera[7]*xy)*camera[1] + camera[3];
    map_x[i] = (float)u; map_y[i] = (float)v; rgb_z[i] = point_z;
    bool candidate = depth[i] > 0 && point_z > 0.0 &&
        u >= 0.0 && u < rgb_width-1 && v >= 0.0 && v < rgb_height-1;
    int pixel = candidate ? __double2int_rn(v)*rgb_width + __double2int_rn(u) : -1;
    rgb_index[i] = pixel;
    if (candidate) atomicMin(nearest+pixel, __double_as_longlong(point_z));
}
extern "C" __global__ void sample_color(
    const unsigned char* rgb, const float* map_x, const float* map_y,
    const double* rgb_z, const int* rgb_index, const double* nearest,
    unsigned char* colors, unsigned char* visible,
    int count, int rgb_width, int rgb_height, int interpolation) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int pixel = rgb_index[i]; double z = rgb_z[i];
    bool use = pixel >= 0 && z <= nearest[pixel] + fmax(15.0, z*0.01);
    visible[i] = use;
    if (!use) { colors[i*3] = 0; colors[i*3+1] = 0; colors[i*3+2] = 0; return; }
    // The installed IPP backend can use continuous float32 coefficients;
    // OpenCV's portable sampler uses nearest-even five-bit fixed-point maps.
    int uq = __float2int_rn(map_x[i]*32.0f), vq = __float2int_rn(map_y[i]*32.0f);
    int x = interpolation ? (int)floorf(map_x[i]) : uq >> 5;
    int y = interpolation ? (int)floorf(map_y[i]) : vq >> 5;
    int ax = uq & 31, ay = vq & 31;
    float fx = map_x[i]-x, fy = map_y[i]-y;
    int weights[4] = {(32-ax)*(32-ay), ax*(32-ay), (32-ax)*ay, ax*ay};
    int sx[4] = {x,x+1,x,x+1}, sy[4] = {y,y,y+1,y+1};
    for (int c = 0; c < 3; ++c) {
        if (interpolation) {
            float values[4];
            for (int k=0;k<4;++k) values[k] = sx[k]>=0 && sx[k]<rgb_width && sy[k]>=0 && sy[k]<rgb_height
                ? rgb[(sy[k]*rgb_width+sx[k])*3+c] : 0;
            float top = fmaf(values[1]-values[0],fx,values[0]);
            float bottom = fmaf(values[3]-values[2],fx,values[2]);
            int rounded = __float2int_rn(fmaf(bottom-top,fy,top));
            colors[i*3+c] = rounded;
            continue;
        }
        int sum = 0;
        for (int k = 0; k < 4; ++k) {
            if (sx[k] >= 0 && sx[k] < rgb_width && sy[k] >= 0 && sy[k] < rgb_height)
                sum += weights[k] * rgb[(sy[k]*rgb_width+sx[k])*3+c];
        }
        colors[i*3+c] = (sum+512) >> 10;
    }
}
'''


class NativeRgbd:
    """Calibrated native RGB/depth only; no confidence or volume updates."""

    def __init__(self, settings, device=0):
        import cv2
        import cupy as cp

        if settings.sensor_calibration is None:
            raise ValueError("CUDA input preparation requires calibrated native disparity input")
        self.cp, self.settings = cp, settings
        self.cuda_device = cp.cuda.Device(device)
        with self.cuda_device, cp.cuda.Stream.null:
            self._initialize()

    def _initialize(self):
        import cv2
        cp, settings = self.cp, self.settings
        self.interpolation = self.detect_interpolation()
        c = settings.camera
        self.shape, self.count = (c.height, c.width), c.height*c.width
        mx, my = rectification_maps(c)
        source = np.arange(self.count, dtype=np.float32).reshape(self.shape)
        source = cv2.remap(source, mx, my, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=-1).astype(np.int32)
        self.source = cp.asarray(source)
        self.lut = cp.asarray(raw_depth_lut(settings.sensor_calibration))
        self.rays = cp.asarray(pinhole_rays(c))
        calibration = settings.sensor_calibration
        self.transform = cp.asarray(np.r_[np.asarray(calibration.rotation).ravel(), calibration.translation_mm])
        rc = settings.rgb_camera
        self.camera = cp.asarray([rc.fx, rc.fy, rc.cx, rc.cy, *rc.distortion], dtype=cp.float64)
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11", "--fmad=false"))
        self.rectify = module.get_function("rectify_clip")
        self.reject = module.get_function("reject_isolated")
        self.project = module.get_function("project_occlusion")
        self.sample = module.get_function("sample_color")
        self.launch = ((self.count+255)//256,), (256,)

    @staticmethod
    def detect_interpolation():
        """Probe behavior rather than assume an OpenCV/IPP version implies it."""
        import cv2
        rng = np.random.default_rng(31)
        rgb = rng.integers(0,256,(64,64,3),dtype=np.uint8)
        mx,my = [rng.uniform(1,61,(64,64)).astype(np.float32) for _ in range(2)]
        expected = cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        ix,iy = np.floor(mx).astype(int),np.floor(my).astype(int)
        ax,ay = (mx-ix.astype(np.float32))[...,None],(my-iy.astype(np.float32))[...,None]
        a,b,c,d = [v.astype(np.float32) for v in (rgb[iy,ix],rgb[iy,ix+1],rgb[iy+1,ix],rgb[iy+1,ix+1])]
        # Emulate the installed IPP difference-form float32 FMA sequence in
        # float64, rounding to float32 after each fused step.
        top = (a.astype(np.float64)+(b-a).astype(np.float64)*ax.astype(np.float64)).astype(np.float32)
        bottom = (c.astype(np.float64)+(d-c).astype(np.float64)*ax.astype(np.float64)).astype(np.float32)
        continuous = np.rint((top.astype(np.float64)+(bottom-top).astype(np.float64)*ay.astype(np.float64)).astype(np.float32)).astype(np.uint8)
        if np.array_equal(continuous,expected):
            return 1
        uq,vq = np.rint(mx*32).astype(int),np.rint(my*32).astype(int)
        ix,iy,ax,ay = uq>>5,vq>>5,uq&31,vq&31
        a,b,c,d = [v.astype(np.int32) for v in (rgb[iy,ix],rgb[iy,ix+1],rgb[iy+1,ix],rgb[iy+1,ix+1])]
        ax,ay = ax[...,None],ay[...,None]
        quantized = ((a*(32-ax)*(32-ay)+b*ax*(32-ay)+c*(32-ax)*ay+d*ax*ay)+512)>>10
        if np.array_equal(quantized.astype(np.uint8),expected):
            return 0
        raise RuntimeError("Installed OpenCV color interpolation is unsupported by CUDA input preparation")

    def prepare(self, rgb, raw):
        with self.cuda_device, self.cp.cuda.Stream.null:
            return self._prepare(rgb, raw)

    def _prepare(self, rgb, raw):
        cp, settings = self.cp, self.settings
        rgb, raw = cp.asarray(rgb), cp.asarray(raw)
        if raw.shape != self.shape or raw.dtype != cp.uint16:
            raise ValueError("Require native uint16 depth with calibrated dimensions")
        rc = settings.rgb_camera
        if rgb.shape != (rc.height, rc.width, 3) or rgb.dtype != cp.uint8:
            raise ValueError("Require native uint8 RGB with calibrated dimensions")
        # Raw kernels index flat storage; noncontiguous views still describe
        # valid observations and must be copied before pointer-based reads.
        rgb, raw = cp.ascontiguousarray(rgb), cp.ascontiguousarray(raw)
        metric = cp.empty(self.shape, cp.float64)
        clipped = cp.empty(self.shape, cp.uint16)
        h, w = self.shape
        x0, y0, x1, y1 = settings.roi or (0,0,w,h)
        self.rectify(*self.launch, (raw, self.source, self.lut, metric, clipped,
            np.int32(self.count), np.int32(w), np.float64(settings.near_m*1000),
            np.float64(settings.far_m*1000), *map(np.int32,(x0,y0,x1,y1))))
        depth = cp.empty_like(clipped) if settings.filter_depth else clipped
        if settings.filter_depth:
            self.reject(*self.launch, (clipped, depth, np.int32(self.count), np.int32(w), np.int32(h)))
        map_x, map_y = cp.empty(self.shape, cp.float32), cp.empty(self.shape, cp.float32)
        rgb_z, rgb_index = cp.empty(self.shape, cp.float64), cp.empty(self.shape, cp.int32)
        nearest = cp.full(rgb.shape[:2], cp.inf, cp.float64)
        self.project(*self.launch, (metric, depth, self.rays, self.transform, self.camera,
            map_x, map_y, rgb_z, rgb_index, nearest,
            np.int32(self.count), np.int32(rc.width), np.int32(rc.height)))
        colors, visible = cp.empty((*self.shape,3), cp.uint8), cp.empty(self.shape, cp.uint8)
        self.sample(*self.launch, (rgb, map_x, map_y, rgb_z, rgb_index, nearest,
            colors, visible, np.int32(self.count), np.int32(rc.width), np.int32(rc.height),np.int32(self.interpolation)))
        return {"color":colors, "depth":depth}


    def prepare_host(self, rgb, raw):
        with self.cuda_device, self.cp.cuda.Stream.null:
            result = self._prepare(rgb, raw)
            return self.cp.asnumpy(result["color"]), self.cp.asnumpy(result["depth"])

    def synchronize(self):
        with self.cuda_device, self.cp.cuda.Stream.null:
            self.cp.cuda.Stream.null.synchronize()

    def metadata(self):
        import cv2
        return {"cupy": self.cp.__version__, "numpy": np.__version__,
                "cuda_device": self.cuda_device.id,
                "opencv": cv2.__version__, "opencv_ipp_enabled": cv2.ipp.useIPP(),
                "opencv_ipp_version": cv2.ipp.getIppVersion(),
                "color_sampler": "continuous_float32_fma" if self.interpolation else "opencv_fixed_5bit"}


class CudaInputError(RuntimeError):
    """Explicit CUDA input request or compatibility validation failed."""


def selection(device):
    requested = os.environ.get("KINECT_CUDA_INPUT", "off").lower()
    if requested not in ("off", "auto", "on"):
        raise ValueError("KINECT_CUDA_INPUT must be off, auto, or on")
    cuda = str(device).startswith("CUDA:")
    if requested == "on" and not cuda:
        raise CudaInputError("CUDA input preparation requested but the scanning backend uses CPU")
    return {"requested":requested,"implementation":"cpu","device":"CPU:0",
        "confidence_device":"CPU:0","gpu_batches":0,"cpu_batches":0,
        "fallback_batches":0,"resident_batches":0,"probe_images":0,"setup_ms":0.0,
        "fallback_reason":"CUDA scanning backend not selected" if requested=="auto" and not cuda else None}


def preparation_signature(settings):
    """Only fields that change calibrated RGB/depth output or array dimensions."""
    calibration = settings.sensor_calibration
    value = {"camera":asdict(settings.camera),"rgb_camera":asdict(settings.rgb_camera),
        "rgb_mode":settings.rgb_mode,"near_m":settings.near_m,"far_m":settings.far_m,
        "filter_depth":settings.filter_depth,"roi":settings.roi,
        "calibration":None if calibration is None else {
            "document":calibration.document_json,"rotation":calibration.rotation,
            "translation_mm":calibration.translation_mm,"a":calibration.a_per_code,
            "b":calibration.b,"scale":calibration.scale}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def input_shapes(rgb, raw, settings):
    c, rc = settings.camera, settings.rgb_camera
    if not isinstance(raw,np.ndarray) or raw.dtype!=np.uint16 or raw.shape!=(c.height,c.width):
        raise ValueError("CUDA input requires raw uint16 depth with the calibrated native dimensions")
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(rc.height,rc.width,3):
        raise ValueError("CUDA input requires uint8 RGB with the calibrated native dimensions")


class InputPreparation:
    """Session-owned preparation with lazy CUDA setup and explicit fallback.

    Off never imports CuPy or runs compatibility probes. Auto disables itself
    for a failed calibration/configuration after a safe, recorded CPU fallback.
    On raises clearly. Preparation owns only temporary images, and completes
    before the caller can integrate either live or candidate volumes.
    """

    def __init__(self, device, backend, *, factory=None):
        self.device, self.backend = str(device), backend
        self.status = selection(device)
        self.backend["cuda_input"] = self.status
        self._factory = factory or (lambda settings: NativeRgbd(settings,
            device=int(self.device.split(":")[1])))
        self._gpu = None
        self._signature = None
        self._failed_reason = None
        self._unsafe_failure = False
        self._stage_device("CPU:0")

    def _stage_device(self, value):
        stages = self.backend.setdefault("stage_devices",{})
        stages["native_rgbd_preparation"] = value
        stages["depth_filter"] = value
        confidence_device = self.backend.get("depth_confidence", {}).get("device", "CPU:0")
        stages["depth_confidence"] = confidence_device
        self.status["confidence_device"] = confidence_device
        self.status.update(device=value,implementation="cuda" if value.startswith("CUDA:") else "cpu")

    def _cpu(self, rgb, raw, settings, cpu_prepare, *, fallback=False):
        self.status["cpu_batches"] += 1
        self.status["fallback_batches"] += int(fallback)
        return cpu_prepare(rgb,raw,settings)

    def _compare(self, rgb, raw, settings, gpu, cpu_prepare, label):
        expected = cpu_prepare(rgb,raw,settings)
        actual = gpu.prepare_host(rgb,raw)
        self.status["probe_images"] += 1
        for name,before,after in zip(("color","depth"),expected,actual):
            if before.shape!=after.shape or before.dtype!=after.dtype:
                raise CudaInputError(f"CUDA input {label} probe changed {name} shape or dtype")
            changed = int(np.count_nonzero(before!=after))
            if changed:
                raise CudaInputError(f"CUDA input {label} probe changed {changed} {name} values")
        return actual

    def _calibration_probe(self, settings, cpu_prepare):
        c,rc = settings.camera, settings.rgb_camera
        y,x = np.indices((c.height,c.width),dtype=np.int32)
        raw = (650+(x*3+y)%350).astype(np.uint16)
        raw[45:61,65:81] = 2047
        raw[155:161,235:241] = 65535
        rgb = np.random.default_rng(31).integers(0,256,(rc.height,rc.width,3),dtype=np.uint8)
        self._compare(rgb,raw,settings,self._gpu,cpu_prepare,"native calibration")

    def _boundary_probe(self, cpu_prepare):
        # This full-size low-resolution sample distinguishes scalar and BLAS
        # transforms at a color rounding boundary on the recorded calibration.
        from shared.sensor_calibration import load_calibration
        from shared.settings import ScanSettings
        calibration = load_calibration()
        calibration = replace(calibration,rgb_low_res=replace(calibration.rgb_low_res,cx=306.32208255633054))
        settings = ScanSettings(sensor_calibration=calibration,rgb_mode="rgb_low_res",filter_depth=False)
        raw = np.random.default_rng(718).integers(650,851,(480,640),dtype=np.uint16)
        _,x = np.indices((480,640))
        rgb = np.repeat(((x%2)*32)[...,None],3,axis=2).astype(np.uint8)
        self._compare(rgb,raw,settings,self._factory(settings),cpu_prepare,"BLAS/color boundary")

    def _failure(self, exc):
        # A failed image-stage operation has not touched a reconstruction, but
        # a sticky CUDA error must still be surfaced instead of hiding it by
        # retrying preparation on CPU and proceeding to GPU fusion.
        if self._gpu is not None:
            try:
                self._gpu.synchronize()
            except Exception as sync_exc:
                self.status["fallback_reason"] = f"CUDA input synchronization failed: {sync_exc}"
                self._failed_reason = self.status["fallback_reason"]
                self._unsafe_failure = True
                self._gpu = None
                self._stage_device("CPU:0")
                raise CudaInputError(self._failed_reason) from exc
        self._failed_reason = f"CUDA input preparation unavailable: {exc}"
        self.status["fallback_reason"] = self._failed_reason
        self._gpu = None
        self._stage_device("CPU:0")
        if self.status["requested"] == "on":
            raise CudaInputError(self._failed_reason) from exc

    def _call(self, rgb, raw, settings, cpu_prepare, *, resident):
        mode = self.status["requested"]
        if mode=="off":
            if resident:
                raise CudaInputError("Resident CUDA input requires KINECT_CUDA_INPUT=auto or on")
            return self._cpu(rgb,raw,settings,cpu_prepare)
        if not self.device.startswith("CUDA:"):
            if resident:
                raise CudaInputError(self.status["fallback_reason"])
            return self._cpu(rgb,raw,settings,cpu_prepare,fallback=True)
        if self._unsafe_failure:
            raise CudaInputError(self._failed_reason)
        signature = preparation_signature(settings)
        if signature!=self._signature:
            self._gpu = None
            self._signature, self._failed_reason = signature, None
            self.status.update(calibration_signature=signature,fallback_reason=None)
            self._stage_device("CPU:0")
        if self._failed_reason is not None:
            if mode=="on" or resident:
                raise CudaInputError(self._failed_reason)
            return self._cpu(rgb,raw,settings,cpu_prepare,fallback=True)
        started = time.perf_counter()
        setup = self._gpu is None
        try:
            if settings.sensor_calibration is None:
                raise ValueError("Only calibrated native disparity streams support CUDA input preparation")
            input_shapes(rgb,raw,settings)
            if setup:
                self._gpu = self._factory(settings)
                self._calibration_probe(settings,cpu_prepare)
                self._boundary_probe(cpu_prepare)
                actual = self._compare(rgb,raw,settings,self._gpu,cpu_prepare,"first observation")
                metadata = self._gpu.metadata()
                self.status.update(compatibility=metadata,probe_passed=True)
                self._stage_device(self.device)
            elif not resident:
                actual = self._gpu.prepare_host(rgb,raw)
            if resident:
                actual = self._gpu.prepare(rgb,raw)
                self._gpu.synchronize()
                self.status["resident_batches"] += 1
            self.status["gpu_batches"] += 1
            return actual
        except Exception as exc:
            self.status["probe_passed"] = False
            self._failure(exc)
            if resident:
                raise CudaInputError(self._failed_reason) from exc
            return self._cpu(rgb,raw,settings,cpu_prepare,fallback=True)
        finally:
            if setup:
                self.status["setup_ms"] += (time.perf_counter()-started)*1000

    def prepare(self, rgb, raw, settings, cpu_prepare):
        return self._call(rgb,raw,settings,cpu_prepare,resident=False)

    def prepare_resident(self, rgb, raw, settings, cpu_prepare):
        """Checked device arrays for future preview consumers; never confidence.

        The first call completes the same host compatibility checks. This
        method is not used by the current CPU tracking/final authorizers.
        """
        return self._call(rgb,raw,settings,cpu_prepare,resident=True)

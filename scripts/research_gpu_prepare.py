"""Standalone native Kinect CUDA preparation experiment; never used by server.

Keep calibrated disparity LUT, nearest lens rectification, depth rejection,
RGB occlusion, and the installed OpenCV color interpolation. Inputs and outputs
remain device arrays until the caller explicitly requests a host copy. The
confidence implementation is separately checked against the CPU reference;
separable floating-point filters need a tolerance rather than bit identity.
"""

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


class NativeGpuPrepare:
    """Research image preprocessor with calibration cached on the device."""

    def __init__(self, settings):
        import cv2
        import cupy as cp

        if settings.sensor_calibration is None:
            raise ValueError("This experiment requires calibrated native disparity input")
        self.cp, self.settings = cp, settings
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
        raise RuntimeError("Installed OpenCV color interpolation is unsupported by this research prototype")

    def prepare(self, rgb, raw, *, confidence=False):
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
        return {"color":colors, "depth":depth, "confidence":self.confidence(depth) if confidence else None,
                "map_x":map_x, "map_y":map_y, "visible":visible, "metric":metric}

    def confidence(self, depth):
        """Match CPU engineering weights with the original conservative gates."""
        from cupyx.scipy import ndimage
        cp, c = self.cp, self.settings.camera
        depth = cp.asarray(depth)
        if depth.shape != self.shape:
            raise ValueError("Confidence requires the calibrated depth dimensions")
        z = depth.astype(cp.float32)/cp.float32(1000)
        valid = cp.isfinite(z) & (z > 0)
        z = cp.where(valid,z,cp.float32(0))
        padded = cp.pad(z,1)
        edge = cp.zeros(self.shape,cp.bool_)
        tolerance = cp.maximum(cp.float32(.02),cp.float32(.02)*z)
        for neighbour in (padded[:-2,1:-1],padded[2:,1:-1],padded[1:-1,:-2],padded[1:-1,2:]):
            edge |= (neighbour > 0) & (cp.abs(neighbour-z)>tolerance)
        inverse = cp.where(valid,cp.float32(1)/cp.where(valid,z,cp.float32(1)),cp.float32(0))
        y,x = cp.indices(self.shape,dtype=cp.float32)
        rx,ry = (x-cp.float32(c.cx))/cp.float32(c.fx),(y-cp.float32(c.cy))/cp.float32(c.fy)
        ray_length = cp.sqrt(cp.float32(1)+rx**2+ry**2)
        cosine, fitted = cp.full(self.shape,.5,cp.float32), cp.zeros(self.shape,cp.bool_)
        for size in (7,3):
            complete = ndimage.minimum_filter(valid,size=size,mode="constant",cval=0)
            crossing = ndimage.maximum_filter(edge,size=size,mode="constant",cval=0)
            use = complete & ~crossing & ~fitted
            offsets = cp.arange(size,dtype=cp.float32)-size//2
            derivative = offsets/cp.sum(offsets*offsets)
            average = cp.full(size,1/size,cp.float32)
            def separable(kx,ky):
                return ndimage.correlate1d(ndimage.correlate1d(inverse,kx,axis=1,mode="mirror"),ky,axis=0,mode="mirror")
            mean = separable(average,average)
            a,b = separable(derivative,average)*cp.float32(c.fx), separable(average,derivative)*cp.float32(c.fy)
            intercept = mean-a*rx-b*ry
            length = cp.sqrt(a*a+b*b+intercept*intercept)
            estimate = mean/cp.maximum(length*ray_length,cp.float32(1e-10))
            cosine = cp.where(use,cp.clip(estimate,0,1),cosine)
            fitted |= use
        sigma = cp.float32(.001)+cp.float32(.002)*z**2
        weight = cp.clip((cp.float32(.003)/sigma)**2,.05,1)*cosine**2
        weight = cp.where(~valid|edge|(fitted&(cosine<cp.float32(.25))),cp.float32(0),weight)
        return cp.ascontiguousarray(weight,dtype=cp.float32)

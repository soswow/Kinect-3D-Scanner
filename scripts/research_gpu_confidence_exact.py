"""Standalone attempt to reproduce OpenCV confidence reductions on CUDA.

OpenCV 5.0.0 filter.simd.hpp uses a general FMA row reduction for size7,
symmetry-paired size3 rows, and paired column reductions. The earlier cupyx
generic-correlate experiment changed four confidence zero-mask decisions and
remains rejected. This variant never alters the production CPU confidence.

Official implementation examined (release tag, not a moving branch):
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.simd.hpp
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.dispatch.cpp

IPP dispatch is installation dependent. The CPU-only probe records whether
IPP on/off changes these six filters; a source-derived order is not evidence
that an installed binary selected that implementation. GPU acceptance requires
actual bit identity of confidence, including zero masks, on the tested inputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('KINECT_NATIVE','on')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np

from scripts.research_gpu_prepare import NativeGpuPrepare

REFERENCES = [
    "https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.simd.hpp",
    "https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/filter.dispatch.cpp",
    "https://github.com/cupy/cupy/blob/v13.6.0/cupy/_core/_routines_math.pyx",
]

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


def coefficients(size):
    offsets = np.arange(size,dtype=np.float32)-size//2
    return np.full(size,1/size,np.float32),offsets/np.dot(offsets,offsets)


def fma_numpy(a,b,c):
    """Probe emulation: product/add in float64, one float32 output rounding."""
    return (np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64)+
            np.asarray(c,dtype=np.float64)).astype(np.float32)


def cpu_order_filter(image,size,dx,dy):
    average,derivative = coefficients(size)
    h,w=image.shape;r=size//2
    padded=np.pad(image,((0,0),(r,r)),mode="reflect")
    if size==3:
        left,center,right=padded[:,:w],padded[:,1:w+1],padded[:,2:w+2]
        row=(right-left)*derivative[2] if dx else fma_numpy(center,average[1],(left+right)*average[2])
    else:
        weights=derivative if dx else average
        row=np.zeros_like(image)
        for k in range(size):
            row=fma_numpy(padded[:,k:k+w],weights[k],row)
    padded=np.pad(row,((r,r),(0,0)),mode="reflect")
    weights=derivative if dy else average
    out=np.zeros_like(image) if dy else row*weights[r]
    for k in range(1,r+1):
        top,bottom=padded[r-k:r-k+h],padded[r+k:r+k+h]
        out=fma_numpy(bottom-top if dy else bottom+top,weights[r+k],out)
    return out


def cpu_probe():
    initial_ipp=cv2.ipp.useIPP()
    image=np.random.default_rng(59).uniform(.2,2,(48,96)).astype(np.float32)
    variants={}
    try:
        for ipp in (False,True):
            cv2.ipp.setUseIPP(ipp)
            rows=[]
            for size in (7,3):
                average,derivative=coefficients(size)
                for dx,dy in ((0,0),(1,0),(0,1)):
                    cpu=cv2.sepFilter2D(image,-1,derivative if dx else average,
                        derivative if dy else average)
                    proposed=cpu_order_filter(image,size,dx,dy)
                    rows.append({"size":size,"dx":dx,"dy":dy,
                        "changed_values":int(np.count_nonzero(cpu!=proposed)),
                        "max_absolute_difference":float(np.max(np.abs(cpu-proposed))),
                        "cpu":cpu})
            variants[str(ipp)]=rows
        differences=[]
        for off,on in zip(variants['False'],variants['True']):
            differences.append({"size":off['size'],"dx":off['dx'],"dy":off['dy'],
                "ipp_on_off_changed_values":int(np.count_nonzero(off['cpu']!=on['cpu']))})
        for rows in variants.values():
            for row in rows:row.pop('cpu')
        return {"opencv":cv2.__version__,"opencv_threads":cv2.getNumThreads(),"opencv_optimized":cv2.useOptimized(),
            "ipp_version":cv2.ipp.getIppVersion(),"initial_ipp":initial_ipp,
            "source_version":"5.0.0","official_references":REFERENCES,
            "cpu_emulation":variants,"ipp_comparison":differences,
            "scope":"Small CPU filter probe only; no CUDA performance or scanner claims. Float64-emulated FMA is separately checked by actual GPU filters."}
    finally:
        cv2.ipp.setUseIPP(initial_ipp)


class ExactConfidencePrepare(NativeGpuPrepare):
    def __init__(self,settings):
        if min(settings.camera.width,settings.camera.height)<4:
            raise ValueError('Confidence reduction prototype requires both image dimensions >= 4')
        super().__init__(settings)
        cp=self.cp
        module=cp.RawModule(code=FILTER_SOURCE,options=("--std=c++11","--fmad=false"))
        self.row_filter=module.get_function("row_filter")
        self.column_filter=module.get_function("column_filter")
        self.coefficients={size:tuple(cp.asarray(v) for v in coefficients(size)) for size in (7,3)}

    def sep_filter(self,image,size,dx,dy):
        cp=self.cp
        average,derivative=self.coefficients[size]
        row=cp.empty_like(image);out=cp.empty_like(image)
        h,w=image.shape
        self.row_filter(*self.launch,(image,row,derivative if dx else average,
            np.int32(w),np.int32(h),np.int32(size),np.int32(dx)))
        self.column_filter(*self.launch,(row,out,derivative if dy else average,
            np.int32(w),np.int32(h),np.int32(size),np.int32(dy)))
        return out

    def moments(self,image,size):
        return tuple(self.sep_filter(image,size,dx,dy) for dx,dy in ((0,0),(1,0),(0,1)))

    def confidence(self,depth,trace=None):
        from cupyx.scipy import ndimage
        cp,c=self.cp,self.settings.camera
        z=depth.astype(cp.float32)/cp.float32(1000)
        valid=cp.isfinite(z)&(z>0)
        z=cp.where(valid,z,cp.float32(0))
        padded=cp.pad(z,1)
        edge=cp.zeros(self.shape,cp.bool_)
        tolerance=cp.maximum(cp.float32(.02),cp.float32(.02)*z)
        for neighbour in (padded[:-2,1:-1],padded[2:,1:-1],padded[1:-1,:-2],padded[1:-1,2:]):
            edge|=(neighbour>0)&(cp.abs(neighbour-z)>tolerance)
        inverse=cp.where(valid,cp.float32(1)/cp.where(valid,z,cp.float32(1)),cp.float32(0))
        y,x=cp.indices(self.shape,dtype=cp.float32)
        rx,ry=(x-cp.float32(c.cx))/cp.float32(c.fx),(y-cp.float32(c.cy))/cp.float32(c.fy)
        # CuPy power(float32, 2) calls powf; use one rounded multiplication,
        # matching NumPy's square fast path rather than approximate power.
        ray_length=cp.sqrt(cp.float32(1)+cp.square(rx)+cp.square(ry))
        if trace is not None:
            trace.update(z=z,valid=valid,edge=edge,inverse=inverse,rx=rx,ry=ry,ray_length=ray_length)
        cosine,fitted=cp.full(self.shape,.5,cp.float32),cp.zeros(self.shape,cp.bool_)
        for size in (7,3):
            complete=ndimage.minimum_filter(valid,size=size,mode="constant",cval=0)
            crossing=ndimage.maximum_filter(edge,size=size,mode="constant",cval=0)
            use=complete&~crossing&~fitted
            mean,a,b=self.moments(inverse,size)
            a=a*cp.float32(c.fx)
            b=b*cp.float32(c.fy)
            intercept=mean-a*rx-b*ry
            length=cp.sqrt(a*a+b*b+intercept*intercept)
            estimate=mean/cp.maximum(length*ray_length,cp.float32(1e-10))
            if trace is not None:
                trace.update({f'{size}.{name}':value for name,value in (
                    ('complete',complete),('crossing',crossing),('use',use),('mean',mean),
                    ('a',a),('b',b),('intercept',intercept),('length',length),('estimate',estimate))})
            cosine=cp.where(use,cp.clip(estimate,0,1),cosine)
            fitted|=use
        sigma=cp.float32(.001)+cp.float32(.002)*cp.square(z)
        weight=cp.clip(cp.square(cp.float32(.003)/sigma),.05,1)*cp.square(cosine)
        weight=cp.where(~valid|edge|(fitted&(cosine<cp.float32(.25))),cp.float32(0),weight)
        if trace is not None:
            trace.update(cosine=cosine,fitted=fitted,sigma=sigma,weight=weight)
        return cp.ascontiguousarray(weight,dtype=cp.float32)


class FusedConfidencePrepare(ExactConfidencePrepare):
    """Share samples across the three filters, preserving each reduction order."""
    def __init__(self,settings):
        super().__init__(settings)
        module=self.cp.RawModule(code=FILTER_SOURCE,options=("--std=c++11","--fmad=false"))
        self.moment_rows=module.get_function('moment_rows')
        self.moment_columns=module.get_function('moment_columns')

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


def cpu_confidence_trace(depth,camera):
    """Diagnostic replay of the existing CPU stages, checked against its result."""
    from shared.confidence import depth_confidence
    z=depth.astype(np.float32)/1000
    valid=np.isfinite(z)&(z>0)
    z[~valid]=0
    padded=np.pad(z,1)
    neighbours=np.stack((padded[:-2,1:-1],padded[2:,1:-1],padded[1:-1,:-2],padded[1:-1,2:]))
    edge=np.any((neighbours>0)&(np.abs(neighbours-z)>np.maximum(.02,.02*z)),axis=0)
    inverse=np.zeros_like(z)
    np.divide(1,z,out=inverse,where=valid)
    y,x=np.indices(z.shape,dtype=np.float32)
    rx,ry=(x-camera.cx)/camera.fx,(y-camera.cy)/camera.fy
    ray_length=np.sqrt(1+rx**2+ry**2)
    trace=dict(z=z,valid=valid,edge=edge,inverse=inverse,rx=rx,ry=ry,ray_length=ray_length)
    cosine=np.full(z.shape,.5,np.float32)
    fitted=np.zeros(z.shape,bool)
    for size in (7,3):
        support=np.ones((size,size),np.uint8)
        complete=cv2.erode(valid.astype(np.uint8),support,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        crossing=cv2.dilate(edge.astype(np.uint8),support,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        use=complete&~crossing&~fitted
        average,derivative=coefficients(size)
        mean=cv2.sepFilter2D(inverse,-1,average,average)
        a=cv2.sepFilter2D(inverse,-1,derivative,average)*camera.fx
        b=cv2.sepFilter2D(inverse,-1,average,derivative)*camera.fy
        intercept=mean-a*rx-b*ry
        length=np.sqrt(a**2+b**2+intercept**2)
        estimate=mean/np.maximum(length*ray_length,1e-10)
        trace.update({f'{size}.{name}':value for name,value in (
            ('complete',complete),('crossing',crossing),('use',use),('mean',mean),
            ('a',a),('b',b),('intercept',intercept),('length',length),('estimate',estimate))})
        cosine[use]=np.clip(estimate[use],0,1)
        fitted|=use
    sigma=.001+.002*z**2
    weight=np.clip((.003/sigma)**2,.05,1)*cosine**2
    weight[~valid|edge|(fitted&(cosine<.25))]=0
    trace.update(cosine=cosine,fitted=fitted,sigma=sigma,weight=weight)
    if not np.array_equal(weight,depth_confidence(depth,camera)):
        raise AssertionError('Diagnostic CPU replay did not match production confidence')
    return trace


def diagnose(args):
    """Separate diagnostic mode; transfers/stage capture are excluded from timing."""
    import zipfile
    from scripts.benchmark_native_kernels import read_frame
    from scripts.profile_session import file_hash,source_hash
    from shared.calibration import prepare_rgbd
    from shared.settings import ScanSettings
    with zipfile.ZipFile(args.archive) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        settings=ScanSettings.from_dict(manifest['settings'])
        rgb,raw=read_frame(archive,manifest['frames'][args.frame])
    _,depth=prepare_rgbd(rgb,raw,settings)
    cpu=cpu_confidence_trace(depth,settings.camera)
    variant=FusedConfidencePrepare if args.fused_filters else ExactConfidencePrepare
    prepare=variant(settings)
    gpu={}
    prepare.confidence(prepare.cp.asarray(depth),trace=gpu)
    rows=[]
    for name,reference in cpu.items():
        actual=prepare.cp.asnumpy(gpu[name])
        changed=reference!=actual
        indices=np.argwhere(changed)[:8]
        delta=np.abs(reference.astype(np.float64)-actual.astype(np.float64))
        rows.append({'stage':name,'changed_values':int(np.count_nonzero(changed)),
            'max_difference':float(delta.max()),
            'examples':[{'index':index.tolist(),'cpu':reference[tuple(index)].item(),
                'gpu':actual[tuple(index)].item()} for index in indices]})
    report={'kind':'experimental-confidence-stage-diagnostic','archive':str(args.archive),
        'input_sha256':file_hash(args.archive),'source_sha256':source_hash(),
        'script_sha256':file_hash(Path(__file__)),'frame':args.frame,'fused_filters':args.fused_filters,
        'official_references':REFERENCES,'stages':rows,
        'scope':'One-frame diagnostic only; no performance claim. CPU trace checked against production result.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0


def main():
    if '--opencv-threads' in sys.argv:
        position=sys.argv.index('--opencv-threads')
        threads=int(sys.argv[position+1])
        if threads<1:raise ValueError('Require positive OpenCV thread count')
        cv2.setNumThreads(threads)
        del sys.argv[position:position+2]
    if '--diagnose' in sys.argv:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--diagnose',action='store_true')
        parser.add_argument('archive',type=Path)
        parser.add_argument('--frame',type=int,default=0)
        parser.add_argument('--fused-filters',action='store_true')
        parser.add_argument('--output',type=Path,required=True)
        return diagnose(parser.parse_args())
    if "--cpu-probe" in sys.argv:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--cpu-probe",action="store_true")
        parser.add_argument("--output",type=Path,required=True)
        args=parser.parse_args()
        report=cpu_probe()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
        return 0
    import scripts.benchmark_gpu_prepare as benchmark
    from scripts.profile_session import file_hash
    original_difference=benchmark.difference
    def strict_difference(reference,actual,tolerance):
        result=original_difference(reference,actual,tolerance)
        if reference['confidence'] is not None:
            result['confidence_changed_values']=int(np.count_nonzero(
                reference['confidence'].view(np.uint32)!=actual['confidence'].view(np.uint32)))
            result['passed']=result['passed'] and not result['confidence_changed_values']
        return result
    fused='--fused-filters' in sys.argv
    if fused:sys.argv.remove('--fused-filters')
    benchmark.NativeGpuPrepare=FusedConfidencePrepare if fused else ExactConfidencePrepare
    benchmark.difference=strict_difference
    result=benchmark.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    report=json.loads(output.read_text(encoding='utf-8'))
    report.update(kind='experimental-opencv-reduction-order-confidence',
        confidence_variant_sha256=file_hash(Path(__file__)),official_references=REFERENCES,
        source_version='5.0.0',strict_confidence_bit_identity_required=True,
        fused_filters=fused,
        opencv_threads=cv2.getNumThreads(),omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
        cpu_filter_probe=cpu_probe(),
        limitations='Research only. Source-derived SIMD reduction order does not prove installed IPP dispatch. Strict actual-image confidence bit identity is required; production CPU confidence is unchanged.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    sys.exit(main())

"""Batched ORB/SIFT retrieval on CUDA with mutual ratio filtering.

OpenCV SIFT stores quantized integer descriptors as float32. For values in
[0, 255] and 128 channels, dot products and squared distances fit the exact
integer range of float32. Other float descriptors retain the CPU matcher.
Only the selected two distances need square roots. A rounded-distance tie
cannot pass the strict ratio test, so squared-distance selection preserves
every accepted nearest-neighbour identity.
"""

import os
from functools import lru_cache

import numpy as np

MAX_ORB_FEATURES = 1200
MAX_SIFT_FEATURES = 1280  # SIFT retains equal-response ties beyond nfeatures.
MAX_BANK_VIEWS = 40

CODE = r'''
extern "C" __global__ void distances(const unsigned int* a, const unsigned int* b,
                                     unsigned short* d, int n, int m) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  if (k >= n*m) return;
  int i = k/m, j = k%m, v = 0;
  #pragma unroll
  for (int w=0; w<8; ++w) v += __popc(a[i*8+w] ^ b[j*8+w]);
  d[k] = v;
}
extern "C" __global__ void forward(const unsigned short* d, const int* offsets,
                                   int* out, int n, int m, int banks) {
  int lane=threadIdx.x%32;
  int row=(blockDim.x*blockIdx.x+threadIdx.x)/32;
  if (row >= n*banks) return;
  int bank=row/n, query=row%n, start=offsets[bank], end=offsets[bank+1];
  unsigned int first=0xffffffff, second=0xffffffff;
  for (int j=start+lane; j<end; j+=32) {
    unsigned int key=((unsigned int)d[query*m+j]<<16) | (j-start);
    if(key<first) {second=first; first=key;} else if(key<second) second=key;
  }
  for(int delta=16;delta;delta/=2) {
    unsigned int a=__shfl_down_sync(0xffffffff,first,delta);
    unsigned int b=__shfl_down_sync(0xffffffff,second,delta);
    if(lane+delta<32) {
      if(a<first) {second=first; first=a;} else if(a<second) second=a;
      if(b<second) second=b;
    }
  }
  if(lane==0) out[row]=(second!=0xffffffff && 4*(first>>16)<3*(second>>16))
    ? (int)(first&65535) : -1;
}
extern "C" __global__ void backward(const unsigned short* d, int* out, int n, int m) {
  int j=blockDim.x*blockIdx.x+threadIdx.x;
  if(j>=m) return;
  unsigned int first=0xffffffff, second=0xffffffff;
  for(int i=0;i<n;++i) {
    unsigned int key=((unsigned int)d[i*m+j]<<16) | i;
    if(key<first) {second=first; first=key;} else if(key<second) second=key;
  }
  out[j]=(second!=0xffffffff && 4*(first>>16)<3*(second>>16)) ? (int)(first&65535) : -1;
}
extern "C" __global__ void mutual(int* out, const int* back, const int* offsets,
                                  int n, int banks) {
  int k=blockDim.x*blockIdx.x+threadIdx.x;
  if(k>=n*banks || out[k]<0) return;
  if(back[offsets[k/n]+out[k]] != k%n) out[k]=-1;
}
extern "C" __global__ void forward_l2(const float* d, const int* offsets,
                                     int* out, int n, int m, int banks) {
  int lane=threadIdx.x%32;
  int row=(blockDim.x*blockIdx.x+threadIdx.x)/32;
  if(row>=n*banks) return;
  int bank=row/n, query=row%n, start=offsets[bank], end=offsets[bank+1];
  unsigned long long first=~0ull, second=~0ull;
  for(int j=start+lane;j<end;j+=32) {
    unsigned long long key=((unsigned long long)__float_as_uint(d[query*m+j])<<32)|(j-start);
    if(key<first) {second=first;first=key;} else if(key<second) second=key;
  }
  for(int delta=16;delta;delta/=2) {
    unsigned long long a=__shfl_down_sync(0xffffffff,first,delta);
    unsigned long long b=__shfl_down_sync(0xffffffff,second,delta);
    if(lane+delta<32) {
      if(a<first) {second=first;first=a;} else if(a<second) second=a;
      if(b<second) second=b;
    }
  }
  if(lane==0) out[row]=(second!=~0ull && (double)sqrtf(__uint_as_float(first>>32))<0.75*(double)sqrtf(__uint_as_float(second>>32)))
    ? (int)(first&0xffffffff) : -1;
}
extern "C" __global__ void backward_l2(const float* d, int* out, int n, int m) {
  int j=blockDim.x*blockIdx.x+threadIdx.x;
  if(j>=m) return;
  unsigned long long first=~0ull, second=~0ull;
  for(int i=0;i<n;++i) {
    unsigned long long key=((unsigned long long)__float_as_uint(d[i*m+j])<<32)|i;
    if(key<first) {second=first;first=key;} else if(key<second) second=key;
  }
  out[j]=(second!=~0ull && (double)sqrtf(__uint_as_float(first>>32))<0.75*(double)sqrtf(__uint_as_float(second>>32)))
    ? (int)(first&0xffffffff) : -1;
}
'''

@lru_cache(maxsize=8)
def _kernels(device_id):
    import cupy as cp

    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=CODE)
        return tuple(module.get_function(n) for n in ('distances', 'forward', 'backward', 'mutual', 'forward_l2', 'backward_l2'))


def selection(device):
    mode = os.environ.get("KINECT_CUDA_MATCHING", "cpu").lower()
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError("KINECT_CUDA_MATCHING must be auto, cpu, or cuda")
    status = {"requested": mode, "implementation": "cpu", "fallback_reason": None}
    if mode == "cpu":
        return status
    try:
        if not str(device).startswith("CUDA"):
            if mode == "auto":
                return status
            raise RuntimeError("CUDA matching requested for a CPU engine")
        _kernels(int(str(device).split(":")[1]))
    except Exception as exc:
        if mode == "cuda":
            raise RuntimeError(f"CUDA descriptor matching unavailable: {exc}") from exc
        status["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return status
    status["implementation"] = "cuda"
    return status


class Matcher:
    """A bounded bank of immutable Features owned by the engine's keyframes."""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.kernels = _kernels(device_id)
        self.cache = {}

    def match(self, source, targets):
        import cupy as cp

        empty = lambda: np.empty((0, 2), dtype=int)
        results = [empty() for _ in targets]
        a = source.descriptors
        if a is None or a.ndim != 2 or len(a) < 40:
            return None
        orb = a.dtype == np.uint8 and a.shape[1] == 32
        sift = a.dtype == np.float32 and a.shape[1] == 128
        if not orb and not sift:
            return None
        valid = [i for i, f in enumerate(targets) if f.descriptors is not None
                 and f.descriptors.dtype == a.dtype and f.descriptors.ndim == 2
                 and f.descriptors.shape[1] == a.shape[1] and len(f.descriptors) >= 40]
        if not valid:
            return results
        selected = [targets[i] for i in valid]
        if sift and any(not np.isfinite(d).all() or not np.equal(d, np.rint(d)).all()
                        or np.any(d < 0) or np.any(d > 255)
                        for d in [a] + [f.descriptors for f in selected]):
            return None
        max_features = MAX_ORB_FEATURES if orb else MAX_SIFT_FEATURES
        if (len(a) > max_features or any(len(f.descriptors) > max_features for f in selected)
                or len(selected) > MAX_BANK_VIEWS):
            return None  # Respect the live bank's allocation bound.
        keep = {id(f) for f in selected}
        self.cache = {key: value for key, value in self.cache.items() if key in keep}
        offsets = np.cumsum([0] + [len(f.descriptors) for f in selected], dtype=np.int32)
        n,m=len(a),int(offsets[-1])
        banks = len(selected)
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            for feature in selected:
                key = id(feature)
                if key not in self.cache or self.cache[key][0] is not feature:
                    self.cache[key] = (feature, cp.asarray(np.ascontiguousarray(feature.descriptors)))
            a = cp.asarray(np.ascontiguousarray(a))
            b = cp.concatenate([self.cache[id(feature)][1] for feature in selected])
            off=cp.asarray(offsets)
            out=cp.empty((banks,n),cp.int32); back=cp.empty(m,cp.int32)
            args=lambda *v: tuple(np.int32(x) for x in v)
            if orb:
                d = cp.empty((n,m),cp.uint16)
                self.kernels[0](((n*m+255)//256,),(256,),(a,b,d,*args(n,m)))
                forward, backward = self.kernels[1:3]
            else:
                # Quantized SIFT values make this float32 algebra exact.
                d = cp.maximum((a*a).sum(axis=1)[:,None] + (b*b).sum(axis=1)[None,:] - 2*(a@b.T), 0)
                forward, backward = self.kernels[4:6]
            forward(((n*banks*32+255)//256,),(256,),(d,off,out,*args(n,m,banks)))
            backward(((m+255)//256,),(256,),(d,back,*args(n,m)))
            self.kernels[3](((n*banks+255)//256,),(256,),(out,back,off,*args(n,banks)))
            indices=cp.asnumpy(out)
        for index, row in zip(valid, indices):
            results[index] = np.column_stack((np.flatnonzero(row >= 0), row[row >= 0]))
        return results

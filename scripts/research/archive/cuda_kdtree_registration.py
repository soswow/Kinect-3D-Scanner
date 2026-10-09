"""Isolated CUDA KD-tree research; never selected by the scanner backend.

SciPy builds a balanced tree on the original double coordinates. CUDA searches
that tree without changing coordinates; ambiguous ties use Open3D's CPU tree.
The original legacy Huber estimator and stopping rules live in NearestICP.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import hashlib
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from scanner_server.cuda_nn_registration import NearestICP


CODE = r'''
extern "C" __global__ void nearest(
  const double* points, const int* order, const int* nodes,
  const double* boxes, const double* queries, int* output,
  double* distances, int n, double radius2) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=n) return;
  const double* q=queries+3*row;
  double guard=1e-12*fmax(1.,q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
  double best=radius2+guard, second=1./0.;
  int selected=-1, stack[64], top=1;stack[0]=0;
  while(top) {
    int node=stack[--top];
    const double* box=boxes+6*node;
    double bound=0.;
    #pragma unroll
    for(int d=0;d<3;++d) {
      double delta=fmax(fmax(box[d]-q[d],q[d]-box[3+d]),0.);
      bound+=delta*delta;
    }
    if(bound>best+guard) continue;
    const int* info=nodes+4*node;
    if(info[0]<0) {
      for(int j=info[2];j<info[3];++j) {
        int id=order[j];const double* p=points+3*id;
        double x=q[0]-p[0],y=q[1]-p[1],z=q[2]-p[2];
        double distance=(x*x+y*y)+z*z;
        if(distance<best) {second=best;best=distance;selected=id;}
        else if(distance<second) second=distance;
      }
    } else {
      int a=info[0],b=info[1];
      const double* ab=boxes+6*a;const double* bb=boxes+6*b;
      double da=0.,db=0.;
      #pragma unroll
      for(int d=0;d<3;++d) {
        double x=fmax(fmax(ab[d]-q[d],q[d]-ab[3+d]),0.);
        double y=fmax(fmax(bb[d]-q[d],q[d]-bb[3+d]),0.);
        da+=x*x;db+=y*y;
      }
      if(top>61) {selected=-2;break;}
      if(da<db) {stack[top++]=b;stack[top++]=a;}
      else {stack[top++]=a;stack[top++]=b;}
    }
  }
  // A double-distance tie can have a different legacy traversal choice.
  if(selected>=0 && second-best<=guard) selected=-2;
  output[row]=selected;distances[row]=best;
}
'''


@lru_cache(maxsize=8)
def kernel(device_id):
    import cupy as cp
    with cp.cuda.Device(device_id):
        return cp.RawKernel(CODE, "nearest", options=("--fmad=false",))


class KDTreeICP(NearestICP):
    def __init__(self, device, max_clouds=64):
        self.device = o3d.core.Device(str(device))
        if not str(self.device).startswith("CUDA"):
            raise ValueError("Research CUDA KD-tree requires CUDA")
        self.device_id = int(str(self.device).split(":")[1])
        self.max_clouds = max_clouds
        self.cache = OrderedDict()
        self.statistics = {"cloud_uploads": 0, "index_builds": 0, "cache_hits": 0,
                           "searches": 0, "pose_iterations": 0, "exact_cpu_queries": 0}
        self.kernel = kernel(self.device_id)

    def _dataset(self, target, radius):
        import cupy as cp
        points = np.asarray(target.points)
        digest = hashlib.blake2b(memoryview(points).cast("B"), digest_size=16).digest()
        cached = self.cache.pop(id(target), None)
        if cached is None or cached[0] is not target or cached[1] != digest:
            tree = cKDTree(points, leafsize=16, copy_data=True)
            rows, boxes = [], []
            def visit(node):
                index = len(rows)
                rows.append(None)
                selected = points[tree.indices[node.start_idx:node.end_idx]]
                boxes.append(np.concatenate((selected.min(axis=0), selected.max(axis=0))))
                if node.split_dim < 0:
                    rows[index] = (-1, -1, node.start_idx, node.end_idx)
                else:
                    a, b = visit(node.lesser), visit(node.greater)
                    rows[index] = (a, b, 0, 0)
                return index
            visit(tree.tree)
            with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
                data = tuple(cp.asarray(array) for array in (
                    points, tree.indices.astype(np.int32), np.asarray(rows, np.int32), np.asarray(boxes)))
            cached = target, digest, data
            self.statistics["cloud_uploads"] += 1
            self.statistics["index_builds"] += 1
        else:
            self.statistics["cache_hits"] += 1
        self.cache[id(target)] = cached
        while len(self.cache) > self.max_clouds:
            self.cache.popitem(last=False)
        return cached[2]

    def _correspondences(self, source, target, search, radius):
        import cupy as cp
        points = np.asarray(source.points)
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            queries = cp.asarray(points)
            output = cp.empty(len(points), cp.int32)
            distances = cp.empty(len(points), cp.float64)
            self.kernel(((len(points)+127)//128,), (128,),
                        (*search, queries, output, distances, np.int32(len(points)), np.float64(radius*radius)))
            nearest = cp.asnumpy(output)
        uncertain = nearest == -2
        if np.any(uncertain):
            tree = o3d.geometry.KDTreeFlann(target)
            for row in np.flatnonzero(uncertain):
                count, ids, _ = tree.search_hybrid_vector_3d(points[row], radius, 1)
                nearest[row] = ids[0] if count else -1
            self.statistics["exact_cpu_queries"] += int(np.count_nonzero(uncertain))
        selected = np.flatnonzero(nearest >= 0)
        difference = points[selected]-np.asarray(target.points)[nearest[selected]]
        squared = np.sum(difference*difference, axis=1)
        inside = squared < radius*radius
        selected, squared = selected[inside], squared[inside]
        pairs = np.column_stack((selected, nearest[selected])).astype(np.int32)
        self.statistics["searches"] += 1
        rmse = float(np.sqrt(np.sum(squared)/len(selected))) if len(selected) else 0.
        return pairs, len(selected)/len(points), rmse

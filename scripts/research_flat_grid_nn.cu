// Standalone full-radius flattened-range candidate iterator ablation.
// Relative to pure parallel lookup, only candidate lane assignment changes.
// Original doubles, distinct-two minima and five-column output stay intact.
// NVRTC supplies device math builtins without host-system headers.

__device__ inline unsigned long long packed(int x,int y,int z) {
    return ((unsigned long long)(x+1048576)<<42) |
           ((unsigned long long)(y+1048576)<<21) |
           (unsigned long long)(z+1048576);
}
__device__ inline bool before(double d,int id,double old,int oldid) {
    return id>=0 && (oldid<0 || d<old || (d==old && id<oldid));
}
__device__ inline void add(double d,int id,double &a,int &ai,double &b,int &bi) {
    if(id<0 || id==ai || id==bi) return;
    if(before(d,id,a,ai)) { b=a;bi=ai;a=d;ai=id; }
    else if(before(d,id,b,bi)) {b=d;bi=id;}
}

extern "C" __global__ void flat_grid_nearest_two(
    const double *points,const int *ids,const unsigned long long *keys,
    const unsigned int *offsets,unsigned int cells,
    const double *queries,unsigned int count,int shift,double *output) {
    unsigned int row=blockIdx.x;
    if(row>=count) return;
    unsigned int tid=threadIdx.x;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    // Launch contract: exactly128 threads, original (N,3) contiguous doubles.
    __shared__ unsigned int starts[27],ends[27];
    __shared__ unsigned int prefix[28];
    __shared__ double minima[128],seconds[128];
    __shared__ int minimum_ids[128],second_ids[128];
    __shared__ unsigned int visits[128];
    __shared__ bool supported;
    __shared__ int query_cell[3];
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(tid==0) {
        double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
        supported=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&
            isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
            scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
        double fx=floor(sx),fy=floor(sy),fz=floor(sz);
        supported=supported&&fx>=-1048576&&fx<=1048575&&
            fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
        if(supported) {
            query_cell[0]=(int)fx;query_cell[1]=(int)fy;query_cell[2]=(int)fz;
        }
    }
    // Every lane observes the same validated query cell and support decision.
    __syncthreads();
    if(!supported) {
        if(tid==0) {output[5*row]=-3;output[5*row+1]=-1;output[5*row+2]=output[5*row+3]=infinity;output[5*row+4]=0;}
        return;
    }
    if(tid<27) {
        // Same cell order as the original nested dx/dy/dz loops.
        int xx=query_cell[0]+(int)(tid/9)-1;
        int yy=query_cell[1]+(int)((tid/3)%3)-1;
        int zz=query_cell[2]+(int)(tid%3)-1;
        starts[tid]=ends[tid]=0;
        // Check before packing: never wrap a signed neighbour cell.
        if(xx>=-1048576&&xx<=1048575&&yy>=-1048576&&yy<=1048575&&zz>=-1048576&&zz<=1048575) {
            unsigned long long key=packed(xx,yy,zz);
            unsigned int low=0,high=cells;
            while(low<high) {unsigned int mid=low+(high-low)/2;if(keys[mid]<key)low=mid+1;else high=mid;}
            if(low<cells&&keys[low]==key) {starts[tid]=offsets[low];ends[tid]=offsets[low+1];}
        }
    }
    // All 27 ranges are published before any of the 128 lanes scans points.
    __syncthreads();
    if(tid==0) {
        prefix[0]=0;
        for(unsigned int cell=0;cell<27;cell++) prefix[cell+1]=prefix[cell]+ends[cell]-starts[cell];
    }
    // Disjoint cell ranges contain at most the one-million-point target.
    __syncthreads();
    double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int seen=0;
    for(unsigned int ordinal=tid;ordinal<prefix[27];ordinal+=128) {
        // First nonempty range whose exclusive cumulative end exceeds ordinal.
        // Repeated prefix values for empty cells are skipped by upper_bound.
        unsigned int low=0,high=27;
        while(low<high) {
            unsigned int middle=low+(high-low)/2;
            if(prefix[middle+1]<=ordinal) low=middle+1;
            else high=middle;
        }
        unsigned int index=starts[low]+ordinal-prefix[low];
        double dx=__dsub_rn(qx,points[3*index]);
        double dy=__dsub_rn(qy,points[3*index+1]);
        double dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);seen++;
    }
    minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;visits[tid]=seen;
    __syncthreads();
    for(unsigned int stride=64;stride;stride>>=1) {
        if(tid<stride) {
            a=minima[tid];b=seconds[tid];ai=minimum_ids[tid];bi=second_ids[tid];
            add(minima[tid+stride],minimum_ids[tid+stride],a,ai,b,bi);
            add(seconds[tid+stride],second_ids[tid+stride],a,ai,b,bi);
            minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;
            visits[tid]+=visits[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0) {
        // Bounded IDs/visit counts<=one million are exact float64 integers.
        // One contiguous return buffer needs a single device-to-host transfer.
        output[5*row]=minimum_ids[0];output[5*row+1]=second_ids[0];
        output[5*row+2]=minima[0];output[5*row+3]=seconds[0];output[5*row+4]=visits[0];
    }
}

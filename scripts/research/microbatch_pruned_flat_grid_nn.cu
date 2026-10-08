// Research-only centre-first exact two-neighbour grid traversal.
// Same dyadic 27-cell domain/output as the exhaustive flat shader. Distances
// retain explicit binary64 RN subtraction/product/left-associated sum.
// First evaluate the containing cell. Exclude another entire cell only when
// its downward-rounded AABB distance exceeds an outward-expanded second-best
// distance. The 64epsilon expansion covers the RN metric's five operations.
// No new target index, coordinate quantization, or approximate search.

__device__ inline unsigned long long packed(int x, int y, int z) {
    return ((unsigned long long)(x+1048576)<<42) |
           ((unsigned long long)(y+1048576)<<21) |
           (unsigned long long)(z+1048576);
}
__device__ inline bool before(double d,int id,double old,int oldid) {
    return id>=0 && (oldid<0 || d<old || (d==old && id<oldid));
}
__device__ inline void add(double d,int id,double &a,int &ai,double &b,int &bi) {
    if(id<0 || id==ai || id==bi) return;
    if(before(d,id,a,ai)) {b=a;bi=ai;a=d;ai=id;}
    else if(before(d,id,b,bi)) {b=d;bi=id;}
}
__device__ inline double gap_lower(double q,int cell,int shift) {
    double lo=scalbn((double)cell,-shift), hi=scalbn((double)(cell+1),-shift);
    return q<lo ? fmax(0.0,__dsub_rd(lo,q)) :
           (q>hi ? fmax(0.0,__dsub_rd(q,hi)) : 0.0);
}
extern "C" __global__ void flat_grid_nearest_two(
    const double *points,const int *ids,const unsigned long long *keys,
    const unsigned int *offsets,unsigned int cells,
    const double *queries,unsigned int count,int shift,double *output) {
    unsigned int row=blockIdx.x, tid=threadIdx.x;
    if(row>=count) return;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    const double epsilon=__longlong_as_double(0x3cb0000000000000LL);
    __shared__ unsigned int starts[27],ends[27],prefix[28],visits[128];
    __shared__ double minima[128],seconds[128],threshold;
    __shared__ int minimum_ids[128],second_ids[128],query_cell[3];
    __shared__ bool supported;
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(tid==0) {
        double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
        supported=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&
            isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
            scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
        double fx=floor(sx),fy=floor(sy),fz=floor(sz);
        supported=supported&&fx>=-1048576&&fx<=1048575&&
            fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
        if(supported) {query_cell[0]=(int)fx;query_cell[1]=(int)fy;query_cell[2]=(int)fz;}
    }
    __syncthreads();
    if(!supported) {
        if(tid==0) {output[5*row]=-3;output[5*row+1]=-1;
            output[5*row+2]=output[5*row+3]=infinity;output[5*row+4]=0;}
        return;
    }
    if(tid<27) {
        int xx=query_cell[0]+(int)(tid/9)-1;
        int yy=query_cell[1]+(int)((tid/3)%3)-1;
        int zz=query_cell[2]+(int)(tid%3)-1;
        starts[tid]=ends[tid]=0;
        if(xx>=-1048576&&xx<=1048575&&yy>=-1048576&&yy<=1048575&&zz>=-1048576&&zz<=1048575) {
            unsigned long long key=packed(xx,yy,zz);
            unsigned int low=0,high=cells;
            while(low<high) {unsigned int mid=low+(high-low)/2;if(keys[mid]<key)low=mid+1;else high=mid;}
            if(low<cells&&keys[low]==key) {starts[tid]=offsets[low];ends[tid]=offsets[low+1];}
        }
    }
    __syncthreads();
    double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int seen=0;
    for(unsigned int index=starts[13]+tid;index<ends[13];index+=128) {
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
    if(tid==0) threshold=__dmul_ru(seconds[0],1.0+64.0*epsilon);
    __syncthreads();
    // Every cell's closed dyadic box encloses its original sorted points.
    if(tid<27 && tid!=13 && starts[tid]<ends[tid]) {
        int xx=query_cell[0]+(int)(tid/9)-1;
        int yy=query_cell[1]+(int)((tid/3)%3)-1;
        int zz=query_cell[2]+(int)(tid%3)-1;
        double gx=gap_lower(qx,xx,shift),gy=gap_lower(qy,yy,shift),gz=gap_lower(qz,zz,shift);
        double lower=__dadd_rd(__dadd_rd(__dmul_rd(gx,gx),__dmul_rd(gy,gy)),__dmul_rd(gz,gz));
        if(lower>threshold) starts[tid]=ends[tid];
    }
    __syncthreads();
    if(tid==0) {
        starts[13]=ends[13];prefix[0]=0;
        for(unsigned int cell=0;cell<27;cell++) prefix[cell+1]=prefix[cell]+ends[cell]-starts[cell];
    }
    __syncthreads();
    // Only lane zero carries the already reduced centre result into pass two.
    if(tid==0) {a=minima[0];b=seconds[0];ai=minimum_ids[0];bi=second_ids[0];seen=visits[0];}
    else {a=b=infinity;ai=bi=-1;seen=0;}
    for(unsigned int ordinal=tid;ordinal<prefix[27];ordinal+=128) {
        unsigned int low=0,high=27;
        while(low<high) {unsigned int mid=low+(high-low)/2;if(prefix[mid+1]<=ordinal)low=mid+1;else high=mid;}
        unsigned int index=starts[low]+ordinal-prefix[low];
        double dx=__dsub_rn(qx,points[3*index]);
        double dy=__dsub_rn(qy,points[3*index+1]);
        double dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);seen++;
    }
    // Lane zero must finish reading the centre totals before other lanes can
    // overwrite shared slots in this second reduction (a real inter-warp race).
    __syncthreads();
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
    if(tid==0) {output[5*row]=minimum_ids[0];output[5*row+1]=second_ids[0];
        output[5*row+2]=minima[0];output[5*row+3]=seconds[0];output[5*row+4]=visits[0];}
}

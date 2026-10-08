// Research-only warp-per-query centre-pruned 27-cell exact two-neighbour search.
// Four independent queries per128-thread block; no cross-warp communication.
// Shared target layout and FP64 RN metrics match the frozen flat reference.
// Launch ceil(query_count/4) blocks of128. No approximate retrieval.
__device__ inline unsigned long long packed(int x,int y,int z) {
    return ((unsigned long long)(x+1048576)<<42) |
           ((unsigned long long)(y+1048576)<<21) | (unsigned long long)(z+1048576);
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
    double lo=scalbn((double)cell,-shift),hi=scalbn((double)(cell+1),-shift);
    return q<lo ? fmax(0.0,__dsub_rd(lo,q)) : (q>hi ? fmax(0.0,__dsub_rd(q,hi)) : 0.0);
}
extern "C" __global__ void flat_grid_nearest_two(
    const double *points,const int *ids,const unsigned long long *keys,
    const unsigned int *offsets,unsigned int cells,
    const double *queries,unsigned int count,int shift,double *output) {
    unsigned int warp=threadIdx.x/32,lane=threadIdx.x%32,row=4*blockIdx.x+warp;
    if(row>=count) return;
    const unsigned int mask=0xffffffffu;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    __shared__ unsigned int starts[4][27],ends[4][27],prefix[4][28];
    __shared__ int query_cell[4][3];
    __shared__ bool supported[4];
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(lane==0) {
        double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
        bool ok=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&
            isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
            scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
        double fx=floor(sx),fy=floor(sy),fz=floor(sz);
        ok=ok&&fx>=-1048576&&fx<=1048575&&fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
        supported[warp]=ok;
        if(ok) {query_cell[warp][0]=(int)fx;query_cell[warp][1]=(int)fy;query_cell[warp][2]=(int)fz;}
    }
    __syncwarp(mask);
    if(!supported[warp]) {
        if(lane==0) {output[5*row]=-3;output[5*row+1]=-1;
            output[5*row+2]=output[5*row+3]=infinity;output[5*row+4]=0;}
        return;
    }
    if(lane<27) {
        int xx=query_cell[warp][0]+(int)(lane/9)-1;
        int yy=query_cell[warp][1]+(int)((lane/3)%3)-1;
        int zz=query_cell[warp][2]+(int)(lane%3)-1;
        starts[warp][lane]=ends[warp][lane]=0;
        if(xx>=-1048576&&xx<=1048575&&yy>=-1048576&&yy<=1048575&&zz>=-1048576&&zz<=1048575) {
            unsigned long long key=packed(xx,yy,zz);unsigned int low=0,high=cells;
            while(low<high) {unsigned int mid=low+(high-low)/2;if(keys[mid]<key)low=mid+1;else high=mid;}
            if(low<cells&&keys[low]==key) {starts[warp][lane]=offsets[low];ends[warp][lane]=offsets[low+1];}
        }
    }
    __syncwarp(mask);
    double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int visits=0;
    for(unsigned int index=starts[warp][13]+lane;index<ends[warp][13];index+=32) {
        double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);visits++;
    }
    for(unsigned int stride=16;stride;stride>>=1) {
        double other_a=__shfl_down_sync(mask,a,stride),other_b=__shfl_down_sync(mask,b,stride);
        int other_ai=__shfl_down_sync(mask,ai,stride),other_bi=__shfl_down_sync(mask,bi,stride);
        unsigned int other_visits=__shfl_down_sync(mask,visits,stride);
        if(lane<stride) {add(other_a,other_ai,a,ai,b,bi);add(other_b,other_bi,a,ai,b,bi);visits+=other_visits;}
    }
    const double epsilon=__longlong_as_double(0x3cb0000000000000LL);
    double threshold=__dmul_ru(__shfl_sync(mask,b,0),1.0+64.0*epsilon);
    if(lane<27 && lane!=13 && starts[warp][lane]<ends[warp][lane]) {
        int xx=query_cell[warp][0]+(int)(lane/9)-1;
        int yy=query_cell[warp][1]+(int)((lane/3)%3)-1;
        int zz=query_cell[warp][2]+(int)(lane%3)-1;
        double gx=gap_lower(qx,xx,shift),gy=gap_lower(qy,yy,shift),gz=gap_lower(qz,zz,shift);
        double lower=__dadd_rd(__dadd_rd(__dmul_rd(gx,gx),__dmul_rd(gy,gy)),__dmul_rd(gz,gz));
        if(lower>threshold) starts[warp][lane]=ends[warp][lane];
    }
    if(lane==13) starts[warp][13]=ends[warp][13];
    if(lane!=0) {a=b=infinity;ai=bi=-1;visits=0;}
    __syncwarp(mask);
    if(lane==0) {
        prefix[warp][0]=0;
        for(unsigned int cell=0;cell<27;cell++) prefix[warp][cell+1]=prefix[warp][cell]+ends[warp][cell]-starts[warp][cell];
    }
    __syncwarp(mask);
    for(unsigned int ordinal=lane;ordinal<prefix[warp][27];ordinal+=32) {
        unsigned int low=0,high=27;
        while(low<high) {unsigned int mid=low+(high-low)/2;if(prefix[warp][mid+1]<=ordinal)low=mid+1;else high=mid;}
        unsigned int index=starts[warp][low]+ordinal-prefix[warp][low];
        double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);visits++;
    }
    for(unsigned int stride=16;stride;stride>>=1) {
        double other_a=__shfl_down_sync(mask,a,stride),other_b=__shfl_down_sync(mask,b,stride);
        int other_ai=__shfl_down_sync(mask,ai,stride),other_bi=__shfl_down_sync(mask,bi,stride);
        unsigned int other_visits=__shfl_down_sync(mask,visits,stride);
        if(lane<stride) {add(other_a,other_ai,a,ai,b,bi);add(other_b,other_bi,a,ai,b,bi);visits+=other_visits;}
    }
    if(lane==0) {output[5*row]=ai;output[5*row+1]=bi;output[5*row+2]=a;output[5*row+3]=b;output[5*row+4]=visits;}
}

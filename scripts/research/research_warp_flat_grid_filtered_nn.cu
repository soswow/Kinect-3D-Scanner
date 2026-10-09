// NEW standalone warp-per-query host-certified float screen. Original XYZ/IDs
// and every RN64 distance remain authoritative; all old shaders unchanged.
// Four32-lane queries/block; no cross-warp barriers; original5 output columns.
// Caller computes E using bound helper with immutable targetM and querycap.
// Research-only warp-per-query exhaustive 27-cell exact two-neighbour search.
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
__device__ inline bool before32(float d,int id,float old,int oldid) {
    return id>=0&&(oldid<0||d<old||(d==old&&id<oldid));
}
__device__ inline void add32(float d,int id,float &a,int &ai,float &b,int &bi) {
    if(id<0||id==ai||id==bi)return;
    if(before32(d,id,a,ai)){b=a;bi=ai;a=d;ai=id;}
    else if(before32(d,id,b,bi)){b=d;bi=id;}
}
__device__ inline float approximate(float qx,float qy,float qz,const float *shadow,unsigned int index) {
    float dx=__fsub_rn(qx,shadow[3*index]);
    float dy=__fsub_rn(qy,shadow[3*index+1]);
    float dz=__fsub_rn(qz,shadow[3*index+2]);
    return __fadd_rn(__fadd_rn(__fmul_rn(dx,dx),__fmul_rn(dy,dy)),__fmul_rn(dz,dz));
}
extern "C" __global__ void make_float_shadow(const double *points,unsigned int scalar_count,float *shadow) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<scalar_count)shadow[i]=__double2float_rn(points[i]);
}
extern "C" __global__ void warp_filtered_grid_nearest_two(
    const double *points,const float *shadow,const int *ids,const unsigned long long *keys,
    const unsigned int *offsets,unsigned int cells,
    const double *queries,unsigned int count,int shift,double query_cap,double certified_error,double *output,unsigned int *diagnostic) {
    unsigned int warp=threadIdx.x/32,lane=threadIdx.x%32,row=4*blockIdx.x+warp;
    if(row>=count) return;
    const unsigned int mask=0xffffffffu;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    __shared__ unsigned int starts[4][27],ends[4][27],prefix[4][28];
    __shared__ int query_cell[4][3];
    __shared__ bool supported[4],screenable[4];
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(lane==0) {
        double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
        bool ok=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&
            isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
            scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
        double fx=floor(sx),fy=floor(sy),fz=floor(sz);
        ok=ok&&fx>=-1048576&&fx<=1048575&&fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
        supported[warp]=ok;
        double q=fmax(fabs(qx),fmax(fabs(qy),fabs(qz)));
        screenable[warp]=ok&&shift>=0&&shift<=20&&isfinite(query_cap)&&query_cap>=0&&query_cap<=1048576.&&q<=query_cap&&isfinite(certified_error)&&certified_error>=0;
        if(ok) {query_cell[warp][0]=(int)fx;query_cell[warp][1]=(int)fy;query_cell[warp][2]=(int)fz;}
    }
    __syncwarp(mask);
    if(!supported[warp]) {
        if(lane==0) {output[5*row]=-3;output[5*row+1]=-1;
            output[5*row+2]=output[5*row+3]=infinity;output[5*row+4]=0;diagnostic[3*row]=diagnostic[3*row+1]=diagnostic[3*row+2]=0;}
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
    if(lane==0) {
        prefix[warp][0]=0;
        for(unsigned int cell=0;cell<27;cell++) prefix[warp][cell+1]=prefix[warp][cell]+ends[warp][cell]-starts[warp][cell];
    }
    __syncwarp(mask);
    // Host computed certified_error for all supported original candidates
    // with maxabs(query)<=query_cap. Outside that cap keeps the full exact pass.
    bool screen=screenable[warp],attempted=screen;
    double limit=infinity;
    float fq_x=__double2float_rn(qx),fq_y=__double2float_rn(qy),fq_z=__double2float_rn(qz);
    if(screen) {
        float a=__int_as_float(0x7f800000),b=a;int ai=-1,bi=-1;unsigned int invalid=0;
        for(unsigned int ordinal=lane;ordinal<prefix[warp][27];ordinal+=32) {
            unsigned int low=0,high=27;
            while(low<high) {unsigned int mid=low+(high-low)/2;if(prefix[warp][mid+1]<=ordinal)low=mid+1;else high=mid;}
            unsigned int index=starts[warp][low]+ordinal-prefix[warp][low];
            float d=approximate(fq_x,fq_y,fq_z,shadow,index);
            if(!isfinite(d)||d<0)invalid=1;else add32(d,ids[index],a,ai,b,bi);
        }
        bool bad=__any_sync(mask,invalid!=0);
        for(unsigned int stride=16;stride;stride>>=1) {
            float other_a=__shfl_down_sync(mask,a,stride),other_b=__shfl_down_sync(mask,b,stride);
            int other_ai=__shfl_down_sync(mask,ai,stride),other_bi=__shfl_down_sync(mask,bi,stride);
            if(lane<stride) {add32(other_a,other_ai,a,ai,b,bi);add32(other_b,other_bi,a,ai,b,bi);}
        }
        float second=__shfl_sync(mask,b,0);int second_id=__shfl_sync(mask,bi,0);
        screen=!bad&&second_id>=0&&isfinite(second);
        if(lane==0&&screen)limit=__dadd_ru((double)second,__dmul_ru(2.,certified_error));
        limit=__shfl_sync(mask,limit,0);screen=screen&&isfinite(limit);
    }
    double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int visits=0,rechecks=0;
    for(unsigned int ordinal=lane;ordinal<prefix[warp][27];ordinal+=32) {
        unsigned int low=0,high=27;
        while(low<high) {unsigned int mid=low+(high-low)/2;if(prefix[warp][mid+1]<=ordinal)low=mid+1;else high=mid;}
        unsigned int index=starts[warp][low]+ordinal-prefix[warp][low];
        visits++;
        if(screen&&(double)approximate(fq_x,fq_y,fq_z,shadow,index)>limit)continue;
        double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);rechecks++;
    }
    for(unsigned int stride=16;stride;stride>>=1) {
        double other_a=__shfl_down_sync(mask,a,stride),other_b=__shfl_down_sync(mask,b,stride);
        int other_ai=__shfl_down_sync(mask,ai,stride),other_bi=__shfl_down_sync(mask,bi,stride);
        unsigned int other_visits=__shfl_down_sync(mask,visits,stride);
        unsigned int other_rechecks=__shfl_down_sync(mask,rechecks,stride);
        if(lane<stride) {add(other_a,other_ai,a,ai,b,bi);add(other_b,other_bi,a,ai,b,bi);visits+=other_visits;rechecks+=other_rechecks;}
    }
    if(lane==0) {output[5*row]=ai;output[5*row+1]=bi;output[5*row+2]=a;output[5*row+3]=b;output[5*row+4]=visits;
        diagnostic[3*row]=rechecks;diagnostic[3*row+1]=((attempted?1:0)+(screen?1:0))*visits;diagnostic[3*row+2]=screen?1:0;}
}

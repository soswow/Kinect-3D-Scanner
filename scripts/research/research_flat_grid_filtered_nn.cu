// Separate two-pass screening proof. Original grid/ranges and RN64 metric stay
// authoritative. Launch exactly128 threads; output keeps original5 columns.
// Shadow preparation and screening must use the same fmad=false/FTZ settings.
__device__ inline unsigned long long packed(int x,int y,int z) {
    return ((unsigned long long)(x+1048576)<<42)|((unsigned long long)(y+1048576)<<21)|(unsigned long long)(z+1048576);
}
__device__ inline bool before(double d,int id,double old,int oldid) {
    return id>=0&&(oldid<0||d<old||(d==old&&id<oldid));
}
__device__ inline void add(double d,int id,double &a,int &ai,double &b,int &bi) {
    if(id<0||id==ai||id==bi)return;
    if(before(d,id,a,ai)){b=a;bi=ai;a=d;ai=id;}
    else if(before(d,id,b,bi)){b=d;bi=id;}
}
__device__ inline bool before32(float d,int id,float old,int oldid) {
    return id>=0&&(oldid<0||d<old||(d==old&&id<oldid));
}
__device__ inline void add32(float d,int id,float &a,int &ai,float &b,int &bi) {
    if(id<0||id==ai||id==bi)return;
    if(before32(d,id,a,ai)){b=a;bi=ai;a=d;ai=id;}
    else if(before32(d,id,b,bi)){b=d;bi=id;}
}
__device__ inline double upadd(double a,double b){return __dadd_ru(a,b);}
__device__ inline double upmul(double a,double b){return __dmul_ru(a,b);}
__device__ inline double metric_error(double d,double delta,double u,double mu) {
    double h=upadd(d,delta),h2=upmul(h,h);
    double term=upadd(upmul(u,h2),upmul(4.,mu));
    double first=upadd(upmul(upmul(2.,u),upadd(h2,term)),upmul(4.,mu));
    double second=upadd(upmul(u,upadd(upmul(3.,upadd(h2,term)),first)),upmul(4.,mu));
    double cast=upmul(3.,upadd(upmul(upmul(2.,d),delta),upmul(delta,delta)));
    return upadd(upadd(cast,upmul(3.,term)),upadd(first,second));
}
__device__ inline double screening_error(double m,double q,int shift) {
    // Original neighbour-cell candidates have |p_axis-q_axis|<2w; 3w
    // is a conservative enclosure. M is exact cached target maxabs.
    double d=3.*scalbn(1.,-shift),u=scalbn(1.,-24),mu=scalbn(1.,-126);
    double pbound=fmin(m,upadd(q,d));
    double c=upadd(upmul(u,upadd(pbound,q)),upmul(2.,mu));
    double delta=upadd(upadd(c,upmul(u,upadd(d,c))),upmul(4.,mu));
    double u64=scalbn(1.,-53),mu64=scalbn(1.,-1022);
    double delta64=upadd(upmul(u64,d),mu64);
    return upadd(metric_error(d,delta,u,mu),metric_error(d,delta64,u64,mu64));
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
extern "C" __global__ void filtered_flat_grid_nearest_two(
    const double *points,const float *shadow,const int *ids,const unsigned long long *keys,
    const unsigned int *offsets,unsigned int cells,const double *queries,unsigned int count,
    int shift,double target_max,double *output,unsigned int *diagnostic) {
    unsigned int row=blockIdx.x;if(row>=count)return;
    unsigned int tid=threadIdx.x;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    const float infinity32=__int_as_float(0x7f800000);
    __shared__ unsigned int starts[27],ends[27],prefix[28],visits[128],rechecks[128],bad,attempted_screen;
    __shared__ double minima[128],seconds[128],error,limit;
    __shared__ float approximate_minima[128],approximate_seconds[128];
    __shared__ int minimum_ids[128],second_ids[128],query_cell[3];
    __shared__ bool supported,screen;
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(tid==0){
        double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
        supported=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
            scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
        double fx=floor(sx),fy=floor(sy),fz=floor(sz);
        supported=supported&&fx>=-1048576&&fx<=1048575&&fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
        if(supported){query_cell[0]=(int)fx;query_cell[1]=(int)fy;query_cell[2]=(int)fz;}
        double q=fmax(fabs(qx),fmax(fabs(qy),fabs(qz)));
        screen=supported&&shift>=0&&shift<=20&&isfinite(target_max)&&target_max>=0&&target_max<=1048576.&&q<=1048576.;
        error=screen?screening_error(target_max,q,shift):infinity;
        screen=screen&&isfinite(error)&&error>=0;attempted_screen=screen?1:0;bad=0;limit=infinity;
    }
    __syncthreads();
    if(!supported){
        if(tid==0){output[5*row]=-3;output[5*row+1]=-1;output[5*row+2]=output[5*row+3]=infinity;output[5*row+4]=0;
            diagnostic[3*row]=diagnostic[3*row+1]=diagnostic[3*row+2]=0;}
        return;
    }
    if(tid<27){
        int xx=query_cell[0]+(int)(tid/9)-1,yy=query_cell[1]+(int)((tid/3)%3)-1,zz=query_cell[2]+(int)(tid%3)-1;
        starts[tid]=ends[tid]=0;
        if(xx>=-1048576&&xx<=1048575&&yy>=-1048576&&yy<=1048575&&zz>=-1048576&&zz<=1048575){
            unsigned long long key=packed(xx,yy,zz);unsigned int low=0,high=cells;
            while(low<high){unsigned int mid=low+(high-low)/2;if(keys[mid]<key)low=mid+1;else high=mid;}
            if(low<cells&&keys[low]==key){starts[tid]=offsets[low];ends[tid]=offsets[low+1];}
        }
    }
    __syncthreads();
    if(tid==0){prefix[0]=0;for(unsigned int cell=0;cell<27;cell++)prefix[cell+1]=prefix[cell]+ends[cell]-starts[cell];}
    __syncthreads();
    float fq_x=__double2float_rn(qx),fq_y=__double2float_rn(qy),fq_z=__double2float_rn(qz);
    if(screen){
        float a=infinity32,b=infinity32;int ai=-1,bi=-1;
        for(unsigned int ordinal=tid;ordinal<prefix[27];ordinal+=128){
            unsigned int low=0,high=27;
            while(low<high){unsigned int middle=low+(high-low)/2;if(prefix[middle+1]<=ordinal)low=middle+1;else high=middle;}
            unsigned int index=starts[low]+ordinal-prefix[low];
            float d=approximate(fq_x,fq_y,fq_z,shadow,index);
            if(!isfinite(d)||d<0)atomicOr(&bad,1u);
            else add32(d,ids[index],a,ai,b,bi);
        }
        approximate_minima[tid]=a;approximate_seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;
        __syncthreads();
        for(unsigned int stride=64;stride;stride>>=1){
            if(tid<stride){
                a=approximate_minima[tid];b=approximate_seconds[tid];ai=minimum_ids[tid];bi=second_ids[tid];
                add32(approximate_minima[tid+stride],minimum_ids[tid+stride],a,ai,b,bi);
                add32(approximate_seconds[tid+stride],second_ids[tid+stride],a,ai,b,bi);
                approximate_minima[tid]=a;approximate_seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;
            }
            __syncthreads();
        }
        if(tid==0){
            screen=bad==0&&second_ids[0]>=0&&isfinite(approximate_seconds[0]);
            if(screen){limit=upadd((double)approximate_seconds[0],upmul(2.,error));screen=isfinite(limit);}
        }
        __syncthreads();
    }
    double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int seen=0,exact=0;
    for(unsigned int ordinal=tid;ordinal<prefix[27];ordinal+=128){
        unsigned int low=0,high=27;
        while(low<high){unsigned int middle=low+(high-low)/2;if(prefix[middle+1]<=ordinal)low=middle+1;else high=middle;}
        unsigned int index=starts[low]+ordinal-prefix[low];seen++;
        if(screen&&(double)approximate(fq_x,fq_y,fq_z,shadow,index)>limit)continue;
        // Exactly the original RN64 distance expression; no FMA/no approximation.
        double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);
        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
        add(d,ids[index],a,ai,b,bi);exact++;
    }
    minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;visits[tid]=seen;rechecks[tid]=exact;
    __syncthreads();
    for(unsigned int stride=64;stride;stride>>=1){
        if(tid<stride){
            a=minima[tid];b=seconds[tid];ai=minimum_ids[tid];bi=second_ids[tid];
            add(minima[tid+stride],minimum_ids[tid+stride],a,ai,b,bi);add(seconds[tid+stride],second_ids[tid+stride],a,ai,b,bi);
            minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;visits[tid]+=visits[tid+stride];rechecks[tid]+=rechecks[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        output[5*row]=minimum_ids[0];output[5*row+1]=second_ids[0];output[5*row+2]=minima[0];output[5*row+3]=seconds[0];output[5*row+4]=visits[0];
        diagnostic[3*row]=rechecks[0];diagnostic[3*row+1]=(attempted_screen+(screen?1:0))*visits[0];diagnostic[3*row+2]=screen?1:0;
    }
}

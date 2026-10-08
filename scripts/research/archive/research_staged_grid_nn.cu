// Separate staged dyadic-grid experiment: original geometry stays binary64.
// Device math builtins supplied by NVRTC; no host-system headers.
__device__ inline unsigned long long packed(int x,int y,int z) {
    return ((unsigned long long)(x+1048576)<<42)|((unsigned long long)(y+1048576)<<21)|(unsigned long long)(z+1048576);
}
__device__ inline bool before(double d,int id,double old,int oldid) {
    return id>=0&&(oldid<0||d<old||(d==old&&id<oldid));
}
__device__ inline void add(double d,int id,double &a,int &ai,double &b,int &bi) {
    if(id<0||id==ai||id==bi)return;
    if(before(d,id,a,ai)){b=a;bi=ai;a=d;ai=id;}else if(before(d,id,b,bi)){b=d;bi=id;}
}
__device__ inline bool interval_coordinate(double x) {
    // FTZ may affect float input operands: endpoints must be zero or normal.
    return x==0.0||fabs(x)>=(double)__int_as_float(0x00800000);
}
__device__ inline float gap_lower(float qlo,float qhi,float plo,float phi) {
    return fmaxf(0.0f,fmaxf(__fsub_rd(qlo,phi),__fsub_rd(plo,qhi)));
}

extern "C" __global__ void staged_grid_nearest_two(
    // Three rows: original doubles, IDs, keys, offsets, intervals, valid flags,
    // cell count, dyadic shift; zero point pointer skips unsupported stage.
    const unsigned long long *tables,const double *radii,
    const double *queries,unsigned int count,double original_radius,double *output) {
    unsigned int row=blockIdx.x,tid=threadIdx.x;
    if(row>=count)return;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    // No incomplete inner-only scan may be consumed as a complete miss.
    if(tables[16]==0){
        if(tid==0){output[7*row]=-3;output[7*row+1]=-1;output[7*row+2]=output[7*row+3]=infinity;output[7*row+4]=output[7*row+5]=output[7*row+6]=0.0;}
        return;
    }
    const double epsilon64=__longlong_as_double(0x3cb0000000000000LL);
    const double full_tau=__dmul_rn(original_radius,original_radius);
    __shared__ unsigned int starts[27],ends[27];
    __shared__ double minima[128],seconds[128];
    __shared__ int minimum_ids[128],second_ids[128];
    __shared__ unsigned int visits[128],pruned[128];
    __shared__ bool supported,done,query_interval_valid;
    __shared__ float query_intervals[6],threshold;
    double qx=queries[3*row],qy=queries[3*row+1],qz=queries[3*row+2];
    if(tid==0){
        query_interval_valid=interval_coordinate(qx)&&interval_coordinate(qy)&&interval_coordinate(qz);
        query_intervals[0]=__double2float_rd(qx);query_intervals[1]=__double2float_ru(qx);
        query_intervals[2]=__double2float_rd(qy);query_intervals[3]=__double2float_ru(qy);
        query_intervals[4]=__double2float_rd(qz);query_intervals[5]=__double2float_ru(qz);
    }
    __syncthreads();
    unsigned long long visit_bits=0,prune_bits=0;unsigned int executed=0,total=0;
    double final_a=infinity,final_b=infinity;int final_ai=-3,final_bi=-1;
    for(int stage=0;stage<3;stage++){
        const unsigned long long *meta=tables+8*stage;
        if(meta[0]==0)continue;
        const double *points=(const double*)meta[0];const int *ids=(const int*)meta[1];
        const unsigned long long *keys=(const unsigned long long*)meta[2];
        const unsigned int *offsets=(const unsigned int*)meta[3];
        const float *intervals=(const float*)meta[4];const unsigned char *valid=(const unsigned char*)meta[5];
        unsigned int cells=(unsigned int)meta[6];int shift=(int)meta[7];
        const double tau=__dmul_rn(radii[stage],radii[stage]);
        if(tid==0){
            double sx=scalbn(qx,shift),sy=scalbn(qy,shift),sz=scalbn(qz,shift);
            supported=isfinite(qx)&&isfinite(qy)&&isfinite(qz)&&isfinite(sx)&&isfinite(sy)&&isfinite(sz)&&
                scalbn(sx,-shift)==qx&&scalbn(sy,-shift)==qy&&scalbn(sz,-shift)==qz;
            double fx=floor(sx),fy=floor(sy),fz=floor(sz);
            supported=supported&&fx>=-1048576&&fx<=1048575&&fy>=-1048576&&fy<=1048575&&fz>=-1048576&&fz<=1048575;
            // The outward float ceiling includes every original CPU strict hit:
            // dCPU >= real squared distance*(1-u64)^5; 64eps covers this bound.
            threshold=__double2float_ru(__dmul_rn(tau,1.0+64.0*epsilon64));
            done=false;
            if(supported){
                int x=(int)fx,y=(int)fy,z=(int)fz,n=0;
                for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++,n++){
                    int xx=x+dx,yy=y+dy,zz=z+dz;starts[n]=ends[n]=0;
                    if(xx<-1048576||xx>1048575||yy<-1048576||yy>1048575||zz<-1048576||zz>1048575)continue;
                    unsigned long long key=packed(xx,yy,zz);unsigned int low=0,high=cells;
                    while(low<high){unsigned int mid=low+(high-low)/2;if(keys[mid]<key)low=mid+1;else high=mid;}
                    if(low<cells&&keys[low]==key){starts[n]=offsets[low];ends[n]=offsets[low+1];}
                }
            }
        }
        __syncthreads();
        if(!supported){
            if(tid==0&&stage==2){final_ai=-3;final_bi=-1;final_a=final_b=infinity;}
            // All warps must consume this stage's shared flag before warp0
            // can overwrite supported/ranges while preparing the next stage.
            __syncthreads();
            continue;
        }
        double a=infinity,b=infinity;int ai=-1,bi=-1;unsigned int seen=0,skipped=0;
        for(int cell=0;cell<27;cell++)for(unsigned int index=starts[cell]+tid;index<ends[cell];index+=128){
            seen++;
            if(query_interval_valid&&valid[index]){
                float gx=gap_lower(query_intervals[0],query_intervals[1],intervals[6*index],intervals[6*index+1]);
                float gy=gap_lower(query_intervals[2],query_intervals[3],intervals[6*index+2],intervals[6*index+3]);
                float gz=gap_lower(query_intervals[4],query_intervals[5],intervals[6*index+4],intervals[6*index+5]);
                float lower=__fadd_rd(__fadd_rd(__fmul_rd(gx,gx),__fmul_rd(gy,gy)),__fmul_rd(gz,gz));
                if(lower>threshold){skipped++;continue;}
            }
            double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);
            double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
            add(d,ids[index],a,ai,b,bi);
        }
        minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;visits[tid]=seen;pruned[tid]=skipped;
        __syncthreads();
        for(unsigned int stride=64;stride;stride>>=1){
            if(tid<stride){
                a=minima[tid];b=seconds[tid];ai=minimum_ids[tid];bi=second_ids[tid];
                add(minima[tid+stride],minimum_ids[tid+stride],a,ai,b,bi);
                add(seconds[tid+stride],second_ids[tid+stride],a,ai,b,bi);
                minima[tid]=a;seconds[tid]=b;minimum_ids[tid]=ai;second_ids[tid]=bi;
                visits[tid]+=visits[tid+stride];pruned[tid]+=pruned[tid+stride];
            }
            __syncthreads();
        }
        if(tid==0){
            total+=visits[0];executed|=(1u<<stage);
            visit_bits|=((unsigned long long)visits[0]<<(20*stage));prune_bits|=((unsigned long long)pruned[0]<<(20*stage));
            final_a=minima[0];final_b=seconds[0];final_ai=minimum_ids[0];final_bi=second_ids[0];
            // FULL-radius margin ensures an outside-stage competitor cannot
            // be a possible tie to an early winner under the original policy.
            double margin=__dmul_rn(64.0*epsilon64,fmax(full_tau,fabs(final_a)));
            bool interior=isfinite(final_a)&&final_ai>=0&&final_a<__dsub_rn(tau,margin);
            double tie_margin=__dmul_rn(64.0*epsilon64,fmax(full_tau,fmax(fabs(final_a),fabs(final_b))));
            bool separate=!isfinite(final_b)||__dsub_rn(final_b,final_a)>tie_margin;
            done=stage==2||(interior&&separate);
        }
        __syncthreads();
        if(done)break;
    }
    if(tid==0){
        output[7*row]=final_ai;output[7*row+1]=final_bi;output[7*row+2]=final_a;output[7*row+3]=final_b;output[7*row+4]=total;
        // Each stage visits at most one million unique IDs:20bits*3+3 mask
        // bits fits63bits. Payload is never used as float arithmetic/JSON.
        output[7*row+5]=__longlong_as_double((long long)(visit_bits|((unsigned long long)executed<<60)));
        output[7*row+6]=__longlong_as_double((long long)prune_bits);
    }
}

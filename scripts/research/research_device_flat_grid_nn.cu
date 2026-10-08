// Separate device result classification/exception transport for exact flat NN.
// Frozen flat shader produces original-double first/second minima unchanged.
// Fixed128 lanes. No device execution failure is converted to a CPU recovery.
extern "C" __global__ void classify_flat_results(
    const double *raw,unsigned int count,unsigned int target_count,double r2,
    int direct_miss,int audit_hits,int audit_misses,int *nearest,double *squared,
    unsigned int *reasons,unsigned int *flagged,unsigned long long *totals) {
    unsigned int row=blockIdx.x*128+threadIdx.x,tid=threadIdx.x;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    const double tiny=__longlong_as_double(0x0010000000000000LL);
    const double epsilon=__longlong_as_double(0x3cb0000000000000LL);
    // totals: flagged, direct hits, declared misses, CPUfallback, unsupported,
    // uncertain, visited, malformed, audited hits, audited misses.
    __shared__ unsigned long long sums[10][128];
    __shared__ unsigned int warp_bases[4],global_base;
    unsigned long long local[10]={0,0,0,0,0,0,0,0,0,0};
    unsigned int reason=0;
    if(row<count){
        double id=raw[5*row],second=raw[5*row+1],a=raw[5*row+2],b=raw[5*row+3],visits=raw[5*row+4];
        bool valid=isfinite(id)&&floor(id)==id&&(id==-3.0||id==-1.0||(id>=0&&id<target_count))&&
            isfinite(second)&&floor(second)==second&&(second==-1.0||(second>=0&&second<target_count))&&
            (second<0||second!=id)&&((isfinite(a)&&a>=0)||a==infinity)&&
            ((isfinite(b)&&b>=0)||b==infinity)&&a<=b&&
            ((id>=0&&isfinite(a))||(id<0&&a==infinity))&&
            ((second>=0&&isfinite(b))||(second<0&&b==infinity))&&
            isfinite(visits)&&visits>=0&&visits<=target_count&&floor(visits)==visits;
        nearest[row]=-1;squared[row]=infinity;
        if(!valid){local[7]=1;}
        else {
            bool finite=isfinite(a),unsupported=id==-3.0;
            double margin=__dmul_rn(64.0*epsilon,fmax(fmax(fabs(a),fabs(r2)),tiny));
            bool boundary=finite&&fabs(__dsub_rn(a,r2))<=margin;
            double tie_margin=__dmul_rn(64.0*epsilon,fmax(fmax(fabs(a),fabs(b)),fabs(r2)));
            bool tie=finite&&isfinite(b)&&__dsub_rn(b,a)<=tie_margin;
            bool uncertain=boundary||tie;
            bool missing=!unsupported&&(!finite||a>=r2);
            bool fallback=unsupported||uncertain||(!direct_miss&&missing);
            int found=missing?-1:(int)id;
            nearest[row]=found;squared[row]=found>=0?a:infinity;
            if(unsupported)reason|=1;
            if(uncertain)reason|=2;
            if(!direct_miss&&missing)reason|=4;
            bool hit=!fallback&&found>=0;
            bool miss=direct_miss&&missing&&!uncertain;
            if(hit&&audit_hits)reason|=8;
            if(miss&&audit_misses)reason|=16;
            local[1]=hit;local[2]=miss;local[3]=fallback;local[4]=unsupported;local[5]=uncertain;
            local[6]=(unsigned long long)visits;local[8]=hit&&audit_hits;local[9]=miss&&audit_misses;
        }
        reasons[row]=reason;
    }
    // Compact CPU-only/audit rows once: one atomic per block, not per query.
    unsigned int lane=tid&31,warp=tid>>5;
    unsigned int mask=__ballot_sync(0xffffffffu,reason!=0);
    if(lane==0)warp_bases[warp]=__popc(mask);
    __syncthreads();
    if(tid==0){
        unsigned int total=0;
        for(int w=0;w<4;w++){unsigned int n=warp_bases[w];warp_bases[w]=total;total+=n;}
        global_base=total?(unsigned int)atomicAdd(totals,(unsigned long long)total):0;
    }
    __syncthreads();
    if(reason){
        unsigned int prior=__popc(mask&((1u<<lane)-1u));
        flagged[global_base+warp_bases[warp]+prior]=row;
    }
    for(int k=1;k<10;k++)sums[k][tid]=local[k];
    __syncthreads();
    for(unsigned int stride=64;stride;stride>>=1){
        if(tid<stride)for(int k=1;k<10;k++)sums[k][tid]+=sums[k][tid+stride];
        __syncthreads();
    }
    if(tid==0)for(int k=1;k<10;k++)if(sums[k][0])atomicAdd(totals+k,sums[k][0]);
}

extern "C" __global__ void pack_flat_cpu_rows(
    const double *queries,const double *raw,const int *nearest,const unsigned int *reasons,
    const unsigned int *flagged,unsigned int count,double *packet) {
    unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    unsigned int row=flagged[index];
    packet[8*index]=(double)row;
    packet[8*index+1]=queries[3*row];packet[8*index+2]=queries[3*row+1];packet[8*index+3]=queries[3*row+2];
    packet[8*index+4]=(double)nearest[row];packet[8*index+5]=(double)reasons[row];
    packet[8*index+6]=raw[5*row+2];packet[8*index+7]=raw[5*row+3];
}

extern "C" __global__ void scatter_flat_cpu_results(
    const unsigned int *rows,const int *corrected,unsigned int count,const double *queries,
    const double *original_target,int *nearest,double *squared) {
    unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    unsigned int row=rows[index];int id=corrected[index];nearest[row]=id;
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    if(id<0){squared[row]=infinity;return;}
    double dx=__dsub_rn(queries[3*row],original_target[3*id]);
    double dy=__dsub_rn(queries[3*row+1],original_target[3*id+1]);
    double dz=__dsub_rn(queries[3*row+2],original_target[3*id+2]);
    squared[row]=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
}

extern "C" __global__ void scatter_flat_provided_metrics(
    const unsigned int *rows,const int *corrected,const double *distances,
    unsigned int count,int *nearest,double *squared) {
    unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    unsigned int row=rows[index];nearest[row]=corrected[index];squared[row]=distances[index];
}

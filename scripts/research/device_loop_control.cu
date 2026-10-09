// Research-only control for a bounded, explicit-stream FP64 ICP lane.
// Mathematical kernels are guarded copies of separately pinned source files.
// This control/composition evaluator is a NEW method; CPU trajectory proof is
// mandatory. No scanner runtime imports or historical proof authority.
enum LoopPhase { PREPARE=0, SOLVE=1, QUERY=2, EQUATIONS=3,
                 NN_BLOCK=4, SOLVE_BLOCK=5, DONE=6, FAULT=7 };
__device__ __forceinline__ double loop_radius_squared(int stage) {
    return stage==0 ? .12*.12 : stage==1 ? .06*.06 : .03*.03;
}
__device__ __forceinline__ int loop_limit(int stage) {
    return stage==0 ? 40 : stage==1 ? 30 : 20;
}

extern "C" __global__ void loop_reset_counters(int* control,
        unsigned long long* counters) {
    if(threadIdx.x || blockIdx.x || (control[0]!=PREPARE && control[0]!=QUERY)) return;
    for(int i=0;i<10;i++) counters[i]=0;
}

extern "C" __global__ void loop_build_system(const double* totals,
        double* matrix,double* gradient,int* status,double* update,
        double* composed,const double* pose,const int* control) {
    if(threadIdx.x || blockIdx.x || control[0]!=SOLVE) return;
    int term=0;
    for(int a=0;a<6;a++) for(int b=a;b<6;b++) {
        matrix[6*a+b]=matrix[6*b+a]=totals[term++];
    }
    for(int a=0;a<6;a++) gradient[a]=totals[21+a];
    // Original zero-correspondence iteration uses an identity update, then
    // evaluates metrics once and can converge. Do not factor a zero system.
    if(totals[28]==0.) {
        status[0]=0;
        for(int i=0;i<16;i++) {
            update[i]=(i%5==0)?1.:0.; composed[i]=pose[i];
        }
    }
}

extern "C" __global__ void loop_commit_update(int* control,const int* status,
        const double* update,const double* composed,double* pose,double* metrics) {
    if(threadIdx.x || blockIdx.x || control[0]!=SOLVE) return;
    if(status[0]) { control[0]=SOLVE_BLOCK; control[4]=status[0]; return; }
    for(int i=0;i<16;i++) if(!isfinite(update[i]) || !isfinite(composed[i])) {
        control[0]=FAULT; control[4]=6; return;
    }
    metrics[2]=metrics[0]; metrics[3]=metrics[1];
    for(int i=0;i<16;i++) pose[i]=composed[i];
    control[2]++; control[3]++; control[0]=QUERY;
}

extern "C" __global__ void loop_classified(int* control,
        const unsigned long long* c,unsigned long long* cumulative,unsigned int count,unsigned int target_count,
        int audit_hits,int audit_misses) {
    if(threadIdx.x || blockIdx.x || (control[0]!=PREPARE && control[0]!=QUERY)) return;
    // No provisional nearest ID can enter equations until these guards and
    // any CPU exception/audit transport have completed.
    bool bad=c[7] || c[0]>count || c[0]!=c[3]+c[8]+c[9] ||
        c[1]+c[2]+c[3]!=count || c[4]>c[3] || c[5]>c[3] ||
        c[6]>(unsigned long long)count*target_count ||
        c[8]!=(audit_hits?c[1]:0) || c[9]!=(audit_misses?c[2]:0);
    if(bad) { control[0]=FAULT; control[4]=1; return; }
    for(int i=0;i<10;i++) cumulative[i]+=c[i];
    control[6]=control[0]; // PREPARE/QUERY distinguishes initial vs update eval.
    control[5]++; control[0]=c[0]?NN_BLOCK:EQUATIONS;
}

extern "C" __global__ void loop_resume_nn(int* control) {
    if(!threadIdx.x && !blockIdx.x && control[0]==NN_BLOCK) control[0]=EQUATIONS;
}

extern "C" __global__ void loop_metrics(int* control,const double* totals,
        unsigned int count,double* metrics) {
    if(threadIdx.x || blockIdx.x || control[0]!=EQUATIONS) return;
    bool bad=totals[29]!=0. || !isfinite(totals[28]) || totals[28]<0. ||
        totals[28]>count || floor(totals[28])!=totals[28] ||
        !isfinite(totals[27]) || totals[27]<0.;
    for(int i=0;i<27;i++) bad=bad || !isfinite(totals[i]);
    if(bad) { control[0]=FAULT; control[4]=2; return; }
    double fitness=totals[28]/count;
    double rmse=totals[28]>0. ? sqrt(totals[27]/totals[28]) : 0.;
    if(!isfinite(fitness)||!isfinite(rmse)) {control[0]=FAULT;control[4]=3;return;}
    metrics[0]=fitness; metrics[1]=rmse;
    bool finished=control[6]==QUERY &&
        ((fabs(metrics[2]-fitness)<1e-6 && fabs(metrics[3]-rmse)<1e-6) ||
          control[2]>=loop_limit(control[1]));
    if(finished) {
        control[1]++; control[2]=0;
        control[0]=control[1]==3 ? DONE : PREPARE;
    } else control[0]=SOLVE;
}

extern "C" __global__ void loop_small_packet(const int* control,
        const unsigned long long* counters,const unsigned long long* cumulative,const double* metrics,
        unsigned long long* packet) {
    if(threadIdx.x || blockIdx.x) return;
    for(int i=0;i<8;i++) packet[i]=(unsigned long long)control[i];
    for(int i=0;i<10;i++) packet[8+i]=counters[i];
    for(int i=0;i<10;i++) packet[18+i]=cumulative[i];
    for(int i=0;i<4;i++) packet[28+i]=__double_as_longlong(metrics[i]);
}

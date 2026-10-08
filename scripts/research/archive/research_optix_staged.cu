// Unmeasured research: radius staging entirely inside one raygen launch.
// Eight payload registers replace the old volatile local-state pointer.
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <math.h>

struct Params {
    OptixTraversableHandle scenes[4];
    const double* points;
    const double* queries;
    int* indices;
    double* distances;
    unsigned long long* visits;
    unsigned int* traces;
    double radius2[4];
    unsigned int count, bins;
};
extern "C" { __constant__ Params params; }
static_assert(sizeof(Params)==120,"Staged launch parameter ABI differs");

static __forceinline__ __device__ double unpack(unsigned int low, unsigned int high) {
    return __longlong_as_double((static_cast<unsigned long long>(high)<<32) | low);
}
static __forceinline__ __device__ void best_payload(double best, int id) {
    unsigned long long bits=__double_as_longlong(best);
    optixSetPayload_0(static_cast<unsigned int>(bits));
    optixSetPayload_1(static_cast<unsigned int>(bits>>32));
    optixSetPayload_4(static_cast<unsigned int>(id));
}
static __forceinline__ __device__ void second_payload(double second, int id) {
    unsigned long long bits=__double_as_longlong(second);
    optixSetPayload_2(static_cast<unsigned int>(bits));
    optixSetPayload_3(static_cast<unsigned int>(bits>>32));
    optixSetPayload_5(static_cast<unsigned int>(id));
}

extern "C" __global__ void __raygen__nearest() {
    unsigned int row=optixGetLaunchIndex().x;
    if(row>=params.count) return;
    const double* q=params.queries+3*row;
    const float3 origin=make_float3(float(q[0]),float(q[1]),float(q[2]));
    const double infinity=__longlong_as_double(0x7ff0000000000000LL);
    const double guard=1.e-12*fmax(1.,(q[0]*q[0]+q[1]*q[1])+q[2]*q[2]);
    unsigned long long total_visits=0;
    unsigned int trace_count=0;
    int result=-1;
    double squared=infinity;
    for(unsigned int bin=0;bin<params.bins;++bin) {
        unsigned int best_low=0,best_high=0x7ff00000u;
        unsigned int second_low=0,second_high=0x7ff00000u;
        unsigned int best_id=0xffffffffu,second_id=0xffffffffu,visits_low=0,visits_high=0;
        optixTrace(params.scenes[bin],origin,make_float3(1.f,0.f,0.f),
            0.f,1.e-20f,0.f,255,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0,1,0,best_low,best_high,second_low,second_high,best_id,second_id,visits_low,visits_high);
        ++trace_count;
        total_visits+=(static_cast<unsigned long long>(visits_high)<<32) | visits_low;
        double best=unpack(best_low,best_high), second=unpack(second_low,second_high);
        int id=static_cast<int>(best_id);
        // Identical uncertainty policy to each original separate bin launch.
        if(id>=0 && (second-best<=guard || fabs(best-params.radius2[bin])<=guard)) {
            result=-2;
            break;
        }
        if(id>=0 && best<params.radius2[bin]) {
            result=id;
            squared=best;
            break;
        }
        // Empty narrow-radius searches expand; only the last empty scene can
        // certify a complete-radius miss in the host's experimental policy.
    }
    params.indices[row]=result;
    params.distances[row]=squared;
    params.visits[row]=total_visits;
    params.traces[row]=trace_count;
}

extern "C" __global__ void __intersection__nearest() {
    int id=static_cast<int>(optixGetPrimitiveIndex());
    unsigned int visits=optixGetPayload_6()+1;
    optixSetPayload_6(visits);
    if(visits==0) optixSetPayload_7(optixGetPayload_7()+1);
    int old_id=static_cast<int>(optixGetPayload_4());
    int old_second_id=static_cast<int>(optixGetPayload_5());
    if(id==old_id || id==old_second_id) return;
    const double* q=params.queries+3*optixGetLaunchIndex().x;
    const double* p=params.points+3*id;
    double x=q[0]-p[0],y=q[1]-p[1],z=q[2]-p[2];
    double squared=(x*x+y*y)+z*z; // Compile with --fmad=false.
    double best=unpack(optixGetPayload_0(),optixGetPayload_1());
    double second=unpack(optixGetPayload_2(),optixGetPayload_3());
    if(squared<best) {
        second_payload(best,old_id);
        best_payload(squared,id);
    } else if(squared<second) {
        second_payload(squared,id);
    }
    // No reported intersection: RTX must enumerate every conservative AABB.
}

extern "C" __global__ void __miss__nearest() {}

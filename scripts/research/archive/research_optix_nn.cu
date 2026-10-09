// Standalone research: RTX AABB retrieval, original double-coordinate distances.
// No hit is reported: traversal enumerates the entire conservative radius cube.
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <math.h>

struct Params {
    OptixTraversableHandle scene;
    const double* points;
    const double* queries;
    int* indices;
    double* distances;
    unsigned int* visits;
    double radius2;
    unsigned int count;
    unsigned int padding;
};
extern "C" { __constant__ Params params; }
struct QueryState { double best, second; int id, second_id; unsigned int visits; };

extern "C" __global__ void __raygen__nearest() {
    unsigned int row = optixGetLaunchIndex().x;
    if (row >= params.count) return;
    const double* q = params.queries + 3 * row;
    double infinity = __longlong_as_double(0x7ff0000000000000LL);
    volatile QueryState state = {infinity, infinity, -1, -1, 0};
    unsigned long long address = reinterpret_cast<unsigned long long>(&state);
    unsigned int low = static_cast<unsigned int>(address);
    unsigned int high = static_cast<unsigned int>(address >> 32);
    // A positive axis ray starts strictly inside every relevant expanded AABB.
    // Positive tiny tmax avoids an invalid zero-length trace interval.
    optixTrace(params.scene, make_float3(float(q[0]), float(q[1]), float(q[2])),
               make_float3(1.f, 0.f, 0.f), 0.f, 1.e-20f, 0.f, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 1, 0, low, high);
    double guard = 1.e-12 * fmax(1., (q[0]*q[0] + q[1]*q[1]) + q[2]*q[2]);
    int id = state.id;
    // Preserve original CPU traversal's tie and strict-radius choices.
    // A missing candidate also falls back to the original tree in the host.
    if (id >= 0 && (state.second - state.best <= guard ||
                   fabs(state.best - params.radius2) <= guard)) id = -2;
    if (state.best >= params.radius2 && id != -2) id = -1;
    params.indices[row] = id;
    params.distances[row] = state.best;
    params.visits[row] = state.visits;
}

extern "C" __global__ void __intersection__nearest() {
    unsigned long long address = static_cast<unsigned long long>(optixGetPayload_0()) |
                                 (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    volatile QueryState* state = reinterpret_cast<volatile QueryState*>(address);
    int id = static_cast<int>(optixGetPrimitiveIndex());
    state->visits++;
    // Spatial splitting may revisit a primitive; second-best must be a distinct ID.
    if (id == state->id || id == state->second_id) return;
    unsigned int row = optixGetLaunchIndex().x;
    const double* q = params.queries + 3 * row;
    const double* p = params.points + 3 * id;
    double x = q[0] - p[0], y = q[1] - p[1], z = q[2] - p[2];
    double distance = (x*x + y*y) + z*z;  // compiled with --fmad=false
    if (distance < state->best) {
        state->second = state->best; state->second_id = state->id;
        state->best = distance; state->id = id;
    } else if (distance < state->second) {
        state->second = distance; state->second_id = id;
    }
}

extern "C" __global__ void __miss__nearest() {}

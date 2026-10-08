// No fast math or fused multiply-add: preserve the float32 reference equations.
extern "C" __global__ void integrate_weighted(
    const float* points, const long long* indices, long long count,
    const float* depth, const float* confidence, const unsigned char* rgb,
    const float* transform, float* tsdf, float* weight, float* color,
    int width, int height, float fx, float fy, float cx, float cy,
    float max_depth, float truncation) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float px = points[3*i], py = points[3*i+1], pz = points[3*i+2];
    float x = ((px*transform[0] + py*transform[1]) + pz*transform[2]) + transform[3];
    float y = ((px*transform[4] + py*transform[5]) + pz*transform[6]) + transform[7];
    float z = ((px*transform[8] + py*transform[9]) + pz*transform[10]) + transform[11];
    if (!(z > 0)) return;
    float safe = fmaxf(z, 1e-6f);
    float u_float = roundf(x*fx/safe + cx);
    float v_float = roundf(y*fy/safe + cy);
    // Check in floating point before conversion, including nonfinite projection.
    if (!(u_float >= 0 && u_float < width && v_float >= 0 && v_float < height)) return;
    int pixel = (int)v_float * width + (int)u_float;
    float observed = depth[pixel], incoming = confidence[pixel];
    float sdf = observed - z;
    if (!(observed > 0 && observed <= max_depth && incoming > 0 && sdf >= -truncation)) return;
    long long id = indices[i];
    float old = weight[id], total = old + incoming;
    float distance = fminf(sdf/truncation, 1.0f);
    tsdf[id] = (tsdf[id]*old + distance*incoming)/total;
    for (int channel = 0; channel < 3; ++channel)
        color[3*id+channel] = (color[3*id+channel]*old + rgb[3*pixel+channel]*incoming)/total;
    weight[id] = total;
}

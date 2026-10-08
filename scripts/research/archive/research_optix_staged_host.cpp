// Standalone single-launch staged traversal. Proven original DLL stays separate.
#define NOMINMAX
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>

static thread_local std::string error;
#define CUDA_CHECK(expr) do { CUresult result=(expr); if(result!=CUDA_SUCCESS) \
    throw std::runtime_error(std::string(#expr)+" CUDA error "+std::to_string(int(result))); } while(0)
#define OPTIX_CHECK(expr) do { OptixResult result=(expr); if(result!=OPTIX_SUCCESS) \
    throw std::runtime_error(std::string(#expr)+" OptiX error "+std::to_string(int(result))); } while(0)
#define EXPORT extern "C" __declspec(dllexport)
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct Context {
    OptixDeviceContext context{}; OptixModule module{}; OptixPipeline pipeline{};
    OptixProgramGroup groups[3]{}; CUdeviceptr sbt_memory{}, params{};
    OptixShaderBindingTable sbt{};
    ~Context() {
        if(params) cuMemFree(params);
        if(sbt_memory) cuMemFree(sbt_memory);
        if(pipeline) optixPipelineDestroy(pipeline);
        for(auto group:groups) if(group) optixProgramGroupDestroy(group);
        if(module) optixModuleDestroy(module);
        if(context) optixDeviceContextDestroy(context);
    }
};
struct Scene {
    Context* owner{}; OptixTraversableHandle handle{}; CUdeviceptr output{}; size_t bytes{};
    ~Scene() { if(output) cuMemFree(output); }
};
struct Params {
    OptixTraversableHandle scenes[4];
    CUdeviceptr points,queries,indices,distances,visits,traces;
    double radius2[4]; unsigned int count,bins;
};
static_assert(sizeof(Params)==120,"Staged launch parameter ABI differs");
EXPORT const char* nn_error() { return error.c_str(); }

EXPORT void* nn_create(const char* ptx, size_t length) {
    Context* owner = new Context();
    try {
        OPTIX_CHECK(optixInit()); CUcontext current{}; CUDA_CHECK(cuCtxGetCurrent(&current));
        if(!current) throw std::runtime_error("CuPy must create the current CUDA context first");
        OptixDeviceContextOptions options{};
        OPTIX_CHECK(optixDeviceContextCreate(current,&options,&owner->context));
        OptixModuleCompileOptions module_options{};
        module_options.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_options.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        OptixPipelineCompileOptions pipeline_options{};
        pipeline_options.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_options.numPayloadValues=8; pipeline_options.numAttributeValues=2;
        pipeline_options.pipelineLaunchParamsVariableName="params";
        pipeline_options.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        char log[8192]{}; size_t log_length=sizeof(log);
        OptixResult status=optixModuleCreate(owner->context,&module_options,&pipeline_options,
            ptx,length,log,&log_length,&owner->module);
        if(status!=OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX module: ")+log);
        OptixProgramGroupDesc description[3]{};
        description[0].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        description[0].raygen.module=owner->module;
        description[0].raygen.entryFunctionName="__raygen__nearest";
        description[1].kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
        description[1].miss.module=owner->module;
        description[1].miss.entryFunctionName="__miss__nearest";
        description[2].kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        description[2].hitgroup.moduleIS=owner->module;
        description[2].hitgroup.entryFunctionNameIS="__intersection__nearest";
        OptixProgramGroupOptions group_options{}; log_length=sizeof(log);
        status=optixProgramGroupCreate(owner->context,description,3,&group_options,
                                      log,&log_length,owner->groups);
        if(status!=OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX groups: ")+log);
        OptixPipelineLinkOptions link{}; link.maxTraceDepth=1; log_length=sizeof(log);
        status=optixPipelineCreate(owner->context,&pipeline_options,&link,owner->groups,3,
                                  log,&log_length,&owner->pipeline);
        if(status!=OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX pipeline: ")+log);
        OptixStackSizes stack{};
        for(auto group:owner->groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(group,&stack,owner->pipeline));
        unsigned int from_traversal{},from_state{},continuation{};
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack,1,0,0,&from_traversal,&from_state,&continuation));
        OPTIX_CHECK(optixPipelineSetStackSize(owner->pipeline,from_traversal,from_state,continuation,1));
        Record records[3]{};
        for(int i=0;i<3;++i) OPTIX_CHECK(optixSbtRecordPackHeader(owner->groups[i],&records[i]));
        CUDA_CHECK(cuMemAlloc(&owner->sbt_memory,sizeof(records)));
        CUDA_CHECK(cuMemcpyHtoD(owner->sbt_memory,records,sizeof(records)));
        owner->sbt.raygenRecord=owner->sbt_memory;
        owner->sbt.missRecordBase=owner->sbt_memory+sizeof(Record);
        owner->sbt.missRecordStrideInBytes=sizeof(Record); owner->sbt.missRecordCount=1;
        owner->sbt.hitgroupRecordBase=owner->sbt_memory+2*sizeof(Record);
        owner->sbt.hitgroupRecordStrideInBytes=sizeof(Record); owner->sbt.hitgroupRecordCount=1;
        CUDA_CHECK(cuMemAlloc(&owner->params,sizeof(Params)));
        return owner;
    } catch(const std::exception& failure) { error=failure.what(); delete owner; return nullptr; }
}
EXPORT void nn_destroy(void* owner) { delete static_cast<Context*>(owner); }
EXPORT void nn_scene_destroy(void* scene) { delete static_cast<Scene*>(scene); }
EXPORT size_t nn_scene_bytes(void* scene) { return static_cast<Scene*>(scene)->bytes; }
EXPORT void* nn_build(void* context, CUdeviceptr aabbs, unsigned int count) {
    Scene* scene=new Scene(); CUdeviceptr temporary{};
    try {
        auto owner=static_cast<Context*>(context);
        scene->owner=owner;
        unsigned int flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput input{}; input.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers=&aabbs;
        input.customPrimitiveArray.numPrimitives=count;
        input.customPrimitiveArray.flags=&flags; input.customPrimitiveArray.numSbtRecords=1;
        OptixAccelBuildOptions options{}; options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation=OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(owner->context,&options,&input,1,&sizes));
        CUDA_CHECK(cuMemAlloc(&temporary,sizes.tempSizeInBytes));
        CUDA_CHECK(cuMemAlloc(&scene->output,sizes.outputSizeInBytes));
        scene->bytes=sizes.outputSizeInBytes;
        OPTIX_CHECK(optixAccelBuild(owner->context,0,&options,&input,1,temporary,sizes.tempSizeInBytes,
            scene->output,sizes.outputSizeInBytes,&scene->handle,nullptr,0));
        CUDA_CHECK(cuCtxSynchronize()); cuMemFree(temporary); return scene;
    } catch(const std::exception& failure) {
        error=failure.what(); if(temporary) cuMemFree(temporary); delete scene; return nullptr;
    }
}
EXPORT int nn_launch_staged(void* context, void* const* scenes, const double* radius2,
    unsigned int bins, CUdeviceptr points, CUdeviceptr queries,
    CUdeviceptr indices, CUdeviceptr distances, CUdeviceptr visits, CUdeviceptr traces,
    unsigned int count) {
    try {
        auto owner=static_cast<Context*>(context);
        if(!owner || !scenes || !radius2 || !count || !bins || bins>4)
            throw std::runtime_error("Invalid staged query dimensions or ownership");
        Params params{};
        params.points=points; params.queries=queries; params.indices=indices;
        params.distances=distances; params.visits=visits; params.traces=traces;
        params.count=count; params.bins=bins;
        for(unsigned int i=0;i<bins;++i) {
            if(!scenes[i] || static_cast<Scene*>(scenes[i])->owner!=owner ||
               !std::isfinite(radius2[i]) || radius2[i]<=0 ||
               (i && radius2[i]<=radius2[i-1]))
                throw std::runtime_error("Staged scenes need positive increasing squared radii");
            params.scenes[i]=static_cast<Scene*>(scenes[i])->handle;
            params.radius2[i]=radius2[i];
        }
        CUDA_CHECK(cuMemcpyHtoD(owner->params,&params,sizeof(params)));
        OPTIX_CHECK(optixLaunch(owner->pipeline,0,owner->params,sizeof(params),&owner->sbt,count,1,1));
        CUDA_CHECK(cuCtxSynchronize()); return 0;
    } catch(const std::exception& failure) { error=failure.what(); return -1; }
}

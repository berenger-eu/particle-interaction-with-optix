//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//
// Copyright (c) Inria 2024
//

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixSphere.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include "SpTimer.hpp"


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


typedef SbtRecord<RayGenDataLJ>   RayGenSbtRecordLJ;
typedef SbtRecord<MissDataLJ>     MissSbtRecordLJ;
typedef SbtRecord<HitGroupDataLJ> HitGroupSbtRecordLJ;


void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 3.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 60.0f );
    cam.setAspectRatio( (float)width / (float)height );
}



static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}


std::pair<double,double> core(const int nbSpheres, const float sphereRadius, const std::string outfile,
          const int width, const int height, const bool checkResult, const bool gensurface){
    double timebuild = 0;
    double timecompute = 0;

    try
    {
        std::vector<float3> sphereVertices;
        std::vector<float> sphereEnergy;

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixInit() );
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // sphere build input
            std::vector<float> sphereRadii;
            if(gensurface){
                for(int idxSphere = 0; idxSphere < nbSpheres; idxSphere++)
                {
                    double theta = 2.0 * M_PI * drand48(); // Random angle between 0 and 2π
                    double phi = acos(1.0 - 2.0 * drand48()); // Random angle between 0 and π

                    // Convert spherical coordinates to Cartesian coordinates
                    double x = sin(phi) * cos(theta) + 0.5;
                    double y = sin(phi) * sin(theta) + 0.5;
                    double z = cos(phi) + 0.5;

                    const float3 sphereVertex = make_float3( x, y, z);
                    sphereVertices.push_back( sphereVertex );
                }                    
            }
            else{
                for(int idxSphere = 0; idxSphere < nbSpheres; idxSphere++)
                {
                    const float3 sphereVertex = make_float3( 1.0f * (drand48()), 
                                                    1.0f * (drand48()), 
                                                    1.0f * (drand48()) );
                    sphereVertices.push_back( sphereVertex );
                }
            }
            // We add espilon to radius to avoid numerical issues
            sphereRadii.push_back( sphereRadius*0.81649658092f + 1e-5f ); // sqrt(2/3)

            CUdeviceptr d_vertex_buffer;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertex_buffer ), sphereVertices.size() * sizeof( float3 ) ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_vertex_buffer ), sphereVertices.data(),
                                    sphereVertices.size() * sizeof( float3 ), cudaMemcpyHostToDevice ) );

            CUdeviceptr d_radius_buffer;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_radius_buffer ), sphereRadii.size() * sizeof( float ) ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_radius_buffer ), sphereRadii.data(), sphereRadii.size() * sizeof( float ),
                                    cudaMemcpyHostToDevice ) );

            OptixBuildInput sphere_input = {};

            sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
            sphere_input.sphereArray.numVertices   = nbSpheres;
            sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;
            sphere_input.sphereArray.singleRadius = 1;

            uint32_t sphere_input_flags[1]         = {OPTIX_GEOMETRY_FLAG_NONE};
            sphere_input.sphereArray.flags         = sphere_input_flags;
            sphere_input.sphereArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &sphere_input, 1, &gas_buffer_sizes ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

            // non-compacted output
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                                    compactedSizeOffset + 8 ) );

            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

            SpTimer timer;

            OPTIX_CHECK( optixAccelBuild( context,
                                          0,  // CUDA stream
                                          &accel_options, &sphere_input,
                                          1,  // num build inputs
                                          d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                          d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                          &emitProperty,  // emitted property list
                                          1               // num emitted properties
                                          ) );

            timer.stop();
            std::cout << "[TIME] Time to build gas: " << timer.getElapsed() << " s" << std::endl;
            timebuild = timer.getElapsed();

            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

            CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
            CUDA_CHECK( cudaFree( (void*)d_vertex_buffer ) );
            CUDA_CHECK( cudaFree( (void*)d_radius_buffer ) );

            size_t compacted_gas_size;
            CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

            if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
            {
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

                // use handle as input and output
                OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

                CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
            }
            else
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        // Ray tracing to generate a view of the scene
        {
            //
            // Create module
            //
            OptixModule module = nullptr;
            OptixModule sphere_module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

                pipeline_compile_options.usesMotionBlur                   = false;
                pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipeline_compile_options.numPayloadValues                 = 3;
                pipeline_compile_options.numAttributeValues               = 1;
                pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

                size_t      inputSize = 0;
                const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixSphere.cu", inputSize );

                OPTIX_CHECK_LOG( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options, input,
                                                    inputSize, LOG, &LOG_SIZE, &module ) );

                OptixBuiltinISOptions builtin_is_options = {};

                builtin_is_options.usesMotionBlur      = false;
                builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
                OPTIX_CHECK_LOG( optixBuiltinISModuleGet( context, &module_compile_options, &pipeline_compile_options,
                                                        &builtin_is_options, &sphere_module ) );
            }

            //
            // Create program groups
            //
            OptixProgramGroup raygen_prog_group   = nullptr;
            OptixProgramGroup miss_prog_group     = nullptr;
            OptixProgramGroup hitgroup_prog_group = nullptr;
            {
                OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

                OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
                raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module            = module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &raygen_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &raygen_prog_group
                            ) );

                OptixProgramGroupDesc miss_prog_group_desc  = {};
                miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module            = module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &miss_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &miss_prog_group
                            ) );

                OptixProgramGroupDesc hitgroup_prog_group_desc = {};
                hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
                hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
                hitgroup_prog_group_desc.hitgroup.moduleIS            = sphere_module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &hitgroup_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &hitgroup_prog_group
                            ) );
            }

            //
            // Link pipeline
            //
            OptixPipeline pipeline = nullptr;
            {
                const uint32_t    max_trace_depth  = 1;
                OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

                OptixPipelineLinkOptions pipeline_link_options = {};
                pipeline_link_options.maxTraceDepth            = max_trace_depth;
                OPTIX_CHECK_LOG( optixPipelineCreate(
                            context,
                            &pipeline_compile_options,
                            &pipeline_link_options,
                            program_groups,
                            sizeof( program_groups ) / sizeof( program_groups[0] ),
                            LOG, &LOG_SIZE,
                            &pipeline
                            ) );

                OptixStackSizes stack_sizes = {};
                for( auto& prog_group : program_groups )
                {
                    OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
                }

                uint32_t direct_callable_stack_size_from_traversal;
                uint32_t direct_callable_stack_size_from_state;
                uint32_t continuation_stack_size;
                OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                        0,  // maxCCDepth
                                                        0,  // maxDCDEpth
                                                        &direct_callable_stack_size_from_traversal,
                                                        &direct_callable_stack_size_from_state, &continuation_stack_size ) );
                OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                        direct_callable_stack_size_from_state, continuation_stack_size,
                                                        1  // maxTraversableDepth
                                                        ) );
            }

            //
            // Set up shader binding table
            //
            OptixShaderBindingTable sbt = {};
            {
                CUdeviceptr  raygen_record;
                const size_t raygen_record_size = sizeof( RayGenSbtRecord );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
                sutil::Camera cam;
                configureCamera( cam, width, height );
                RayGenSbtRecord rg_sbt;
                rg_sbt.data ={};
                rg_sbt.data.cam_eye = cam.eye();
                cam.UVWFrame( rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w );
                OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( raygen_record ),
                            &rg_sbt,
                            raygen_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                CUdeviceptr miss_record;
                size_t      miss_record_size = sizeof( MissSbtRecord );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
                MissSbtRecord ms_sbt;
                ms_sbt.data = { 0.3f, 0.1f, 0.2f };
                OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( miss_record ),
                            &ms_sbt,
                            miss_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                CUdeviceptr hitgroup_record;
                size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
                HitGroupSbtRecord hg_sbt;
                OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( hitgroup_record ),
                            &hg_sbt,
                            hitgroup_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                sbt.raygenRecord                = raygen_record;
                sbt.missRecordBase              = miss_record;
                sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
                sbt.missRecordCount             = 1;
                sbt.hitgroupRecordBase          = hitgroup_record;
                sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
                sbt.hitgroupRecordCount         = 1;
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

            //
            // launch
            //
            {
                CUstream stream;
                CUDA_CHECK( cudaStreamCreate( &stream ) );

                Params params;
                params.image        = output_buffer.map();
                params.image_width  = width;
                params.image_height = height;
                params.origin_x     = width / 2;
                params.origin_y     = height / 2;
                params.handle       = gas_handle;

                CUdeviceptr d_param;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_param ),
                            &params, sizeof( params ),
                            cudaMemcpyHostToDevice
                            ) );

                OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
                CUDA_SYNC_CHECK();

                output_buffer.unmap();
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
            }

            //
            // Display results
            //
            {
                sutil::ImageBuffer buffer;
                buffer.data         = output_buffer.getHostPointer();
                buffer.width        = width;
                buffer.height       = height;
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
                if( outfile.empty() )
                    sutil::displayBufferWindow( "Multi sphere", buffer );
                else
                    sutil::saveImage( outfile.c_str(), buffer, false );
            }

            //
            // Cleanup
            //
            {
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
                // Will be reused CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

                OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
                OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
                OPTIX_CHECK( optixModuleDestroy( module ) );
                OPTIX_CHECK( optixModuleDestroy( sphere_module ) );

                // Will be reused OPTIX_CHECK( optixDeviceContextDestroy( context ) );
            }
        }

        // Ray to compute LJ
        {
            //
            // Create module
            //
            OptixModule module = nullptr;
            OptixModule sphere_module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

                pipeline_compile_options.usesMotionBlur                   = false;
                pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipeline_compile_options.numPayloadValues                 = 5;
                pipeline_compile_options.numAttributeValues               = 1;
                pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

                size_t      inputSize = 0;
                const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixLJ.cu", inputSize );

                OPTIX_CHECK_LOG( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options, input,
                                                    inputSize, LOG, &LOG_SIZE, &module ) );

                OptixBuiltinISOptions builtin_is_options = {};

                builtin_is_options.usesMotionBlur      = false;
                builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
                OPTIX_CHECK_LOG( optixBuiltinISModuleGet( context, &module_compile_options, &pipeline_compile_options,
                                                        &builtin_is_options, &sphere_module ) );
            }

            //
            // Create program groups
            //
            OptixProgramGroup raygen_prog_group   = nullptr;
            OptixProgramGroup miss_prog_group     = nullptr;
            OptixProgramGroup hitgroup_prog_group = nullptr;
            {
                OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

                OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
                raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module            = module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &raygen_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &raygen_prog_group
                            ) );

                OptixProgramGroupDesc miss_prog_group_desc  = {};
                miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module            = module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &miss_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &miss_prog_group
                            ) );

                OptixProgramGroupDesc hitgroup_prog_group_desc = {};
                hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ch";
                hitgroup_prog_group_desc.hitgroup.moduleCH            = nullptr;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
                hitgroup_prog_group_desc.hitgroup.moduleIS            = sphere_module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                            context,
                            &hitgroup_prog_group_desc,
                            1,   // num program groups
                            &program_group_options,
                            LOG, &LOG_SIZE,
                            &hitgroup_prog_group
                            ) );
            }

            //
            // Link pipeline
            //
            OptixPipeline pipeline = nullptr;
            {
                const uint32_t    max_trace_depth  = 1;
                OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

                OptixPipelineLinkOptions pipeline_link_options = {};
                pipeline_link_options.maxTraceDepth            = max_trace_depth;
                OPTIX_CHECK_LOG( optixPipelineCreate(
                            context,
                            &pipeline_compile_options,
                            &pipeline_link_options,
                            program_groups,
                            sizeof( program_groups ) / sizeof( program_groups[0] ),
                            LOG, &LOG_SIZE,
                            &pipeline
                            ) );

                OptixStackSizes stack_sizes = {};
                for( auto& prog_group : program_groups )
                {
                    OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
                }

                uint32_t direct_callable_stack_size_from_traversal;
                uint32_t direct_callable_stack_size_from_state;
                uint32_t continuation_stack_size;
                OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                        0,  // maxCCDepth
                                                        0,  // maxDCDEpth
                                                        &direct_callable_stack_size_from_traversal,
                                                        &direct_callable_stack_size_from_state, &continuation_stack_size ) );
                OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                        direct_callable_stack_size_from_state, continuation_stack_size,
                                                        1  // maxTraversableDepth
                                                        ) );
            }

            //
            // Set up shader binding table
            //
            OptixShaderBindingTable sbt = {};
            {
                CUdeviceptr  raygen_record;
                const size_t raygen_record_size = sizeof( RayGenSbtRecordLJ );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
                RayGenSbtRecordLJ rg_sbt;
                OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( raygen_record ),
                            &rg_sbt,
                            raygen_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                CUdeviceptr miss_record;
                size_t      miss_record_size = sizeof( MissSbtRecordLJ );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
                MissSbtRecordLJ ms_sbt;
                OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( miss_record ),
                            &ms_sbt,
                            miss_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                CUdeviceptr hitgroup_record;
                size_t      hitgroup_record_size = sizeof( HitGroupSbtRecordLJ );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
                HitGroupSbtRecordLJ hg_sbt;
                OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( hitgroup_record ),
                            &hg_sbt,
                            hitgroup_record_size,
                            cudaMemcpyHostToDevice
                            ) );

                sbt.raygenRecord                = raygen_record;
                sbt.missRecordBase              = miss_record;
                sbt.missRecordStrideInBytes     = sizeof( MissSbtRecordLJ );
                sbt.missRecordCount             = 1;
                sbt.hitgroupRecordBase          = hitgroup_record;
                sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecordLJ );
                sbt.hitgroupRecordCount         = 1;
            }

            CUdeviceptr output_buffer;
            {
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &output_buffer ), nbSpheres * sizeof( float ) ) );
                CUDA_CHECK( cudaMemset( reinterpret_cast<void*>( output_buffer ), 0, nbSpheres * sizeof( float ) ));
            }

            //
            // launch
            //
            {
                const size_t sphereVerticesCopyLeadingDim = ((nbSpheres+31)/32)*32;
                std::vector<float> sphereVerticesCopy(sphereVerticesCopyLeadingDim*3);
                for(size_t idxPos = 0 ; idxPos < sphereVertices.size() ; ++idxPos)
                {
                    const auto pos = sphereVertices[idxPos];
                    sphereVerticesCopy[idxPos] = pos.x;
                    sphereVerticesCopy[idxPos + sphereVerticesCopyLeadingDim] = pos.y;
                    sphereVerticesCopy[idxPos + sphereVerticesCopyLeadingDim*2] = pos.z;
                }
                CUdeviceptr d_sphereVertices;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sphereVertices ), 
                                        sphereVerticesCopyLeadingDim*3 * sizeof( float ) ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_sphereVertices ),
                            sphereVerticesCopy.data(), sphereVerticesCopyLeadingDim*3 * sizeof( float ),
                            cudaMemcpyHostToDevice
                            ) );
            

                CUstream stream;
                CUDA_CHECK( cudaStreamCreate( &stream ) );

                ParamsLJ params;
                params.num_points = nbSpheres;
                params.leading_dim = sphereVerticesCopyLeadingDim;
                params.points     = reinterpret_cast<float*>(d_sphereVertices);
                params.c          = sphereRadius;
                params.energy     = reinterpret_cast<float*>(output_buffer);
                params.handle     = gas_handle;

                CUdeviceptr d_param;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( ParamsLJ ) ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_param ),
                            &params, sizeof( params ),
                            cudaMemcpyHostToDevice
                            ) );

                SpTimer timer;

                const int nbRays = 3;
                OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( ParamsLJ ), 
                                          &sbt, nbSpheres, nbRays, /*depth=*/1 ) );
                CUDA_SYNC_CHECK();

                timer.stop();
                std::cout << "[TIME] Time to compute LJ: " << timer.getElapsed() << " s" << std::endl;
                timecompute = timer.getElapsed();

                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
            }

            //
            // Cleanup
            //
            {
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

                OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
                OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
                OPTIX_CHECK( optixModuleDestroy( module ) );
                OPTIX_CHECK( optixModuleDestroy( sphere_module ) );

                OPTIX_CHECK( optixDeviceContextDestroy( context ) );

                // Added
                sphereEnergy.resize(nbSpheres, 0);
                CUDA_CHECK( cudaMemcpy( sphereEnergy.data(), reinterpret_cast<void*>( output_buffer ), 
                                       nbSpheres * sizeof( float ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( output_buffer ) ) );
            }
        }

        if(checkResult){
            // We will compute the LJ interaction between the particles on the cpu
            // to compare the results
            // We will compute only for the first 100 particles
            for(size_t idxTarget = 0 ; idxTarget < std::min(100UL, sphereVertices.size()) ; ++idxTarget){
                float energy = 0.0f;

                for(size_t idxSource = 0 ; idxSource < sphereVertices.size() ; ++idxSource){
                    if(idxSource != idxTarget){
                        const auto posSource = sphereVertices[idxSource];
                        const auto posTarget = sphereVertices[idxTarget];
                        const auto diff = posSource - posTarget;
                        const float dist = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                        if(dist < sphereRadius){
                            std::cout << " - SRC pos = " << sphereVertices[idxSource].x << " " << sphereVertices[idxSource].y << " " << sphereVertices[idxSource].z << std::endl;
                            energy += 4.0f * (pow(1.0f/dist, 12) - pow(1.0f/dist, 6));
                        }
                    }
                }
                std::cout << "Energy for particle " << idxTarget << " is " << energy 
                    << " it has been computed as " << (sphereEnergy[idxTarget]) << std::endl;
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }


    return std::make_pair(timebuild, timecompute);
}


#include <chrono>
#include <ctime>

std::string getFilename(const bool gensurface){
    // Get the current date and time
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Format the date and time
    std::tm* ptm = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", ptm);

    // Create the filename with the date and time
    std::string filename = "results-sphere"
                             + std::string(gensurface ? "-surface" : "")
                             + "-" + std::string(buffer) + ".csv";
    return filename;
}

struct ResultFrame{
    struct AResult{
        double timeInit;
        double timeCompute;
        double timeTotal;
    };

    int nbParticles;
    long int nbInteractions;
    int nbLoops;
    int boxDiv;
    std::vector<AResult> results;
};


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --nbspheres | -ns           Specify the number of particles/spheres\n";
    std::cerr << "         --boxdivisor | -bd          Specify the divisor to compute the cutoff (cutoff = 1/x) \n";
    std::cerr << "         --runbench | -rb            Ask to run the benchmark (erase all the other arguments) \n";
    std::cerr << "         --gensurface | -gs          Ask to generate the surface of the sphere (default is random) \n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}

int main( int argc, char* argv[] )
{
    std::string outfile;
    int         nbSpheres = 10;
    int         boxDivisor = 2;
    int         width  = 1024;
    int         height =  768;
    bool        runBench = false;
    bool        gensurface = false;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "--nbspheres" || arg == "-ns" )
        {
            if( i < argc - 1 )
            {
                nbSpheres = atoi(argv[++i]);
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "--boxdivisor" || arg == "-bd" )
        {
            if( i < argc - 1 )
            {
                boxDivisor = atoi(argv[++i]);
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "--runbench" || arg == "-rb" )
        {
            runBench = true;
        }
        else if( arg == "--gensurface" || arg == "-gs" )
        {
            gensurface = true;
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if(runBench){
        if(gensurface){
            {// Fake first run to warm up the GPU
                const float  sphereRadius = 0.5;
                core(10, sphereRadius, outfile, width, height, false, gensurface);
            }
            std::vector<ResultFrame> results;

            const int NbLoops = 5;// put 200
            const int MaxParticlesPerCell = 32;
            const int MaxBoxDiv = 32;// put 32
            for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
                const int nbBoxes = boxDiv*boxDiv*boxDiv;
                for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 2){
                    const double particlePerCell = double(nbParticles)/double(nbBoxes);
                    const double expectedNbNeighbors = 9*particlePerCell;
                    const double sphereRadius = acos(1. - ((2*expectedNbNeighbors)/nbParticles));
                    const int nbSpheres = nbParticles;

                    std::cout << "NbParticles: " << nbParticles << std::endl;
                    std::cout << "BoxDiv: " << boxDiv << std::endl;
                    std::cout << "CellWidth: " << sphereRadius << std::endl;
                    std::cout << "NbLoops: " << NbLoops << std::endl;
                    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                        std::pair<double,double> timeInitCompute = core(nbSpheres, sphereRadius, outfile, width, height, false, gensurface);

                        ResultFrame frame;
                        frame.nbParticles = nbParticles;
                        frame.nbInteractions = nbParticles*(nbParticles/nbBoxes)*27;
                        frame.nbLoops = NbLoops;
                        frame.boxDiv = boxDiv;
                        frame.results.push_back({timeInitCompute.first, timeInitCompute.second, timeInitCompute.first+timeInitCompute.second});
                        results.push_back(frame);
                    }
                }
            }

            std::ofstream file(getFilename(gensurface));
            {
                file << "NbParticles,NbInteractions,NbLoops,boxDiv,nbCells,partspercell,timeinit,timecompute,timetotal";
                file << std::endl;
            }
            for(const ResultFrame& frame : results){
                for(const ResultFrame::AResult& res : frame.results){
                    file << frame.nbParticles << "," << frame.nbInteractions << "," << frame.nbLoops << "," << frame.boxDiv << "," << frame.boxDiv*frame.boxDiv*frame.boxDiv;
                    file << "," << double(frame.nbParticles)/(frame.boxDiv*frame.boxDiv*frame.boxDiv);
                    file << "," << res.timeInit  << "," << res.timeCompute << "," << res.timeTotal;
                    file << std::endl;
                }
            }
        }
        else{
            {// Fake first run to warm up the GPU
                const float  sphereRadius = 0.5;
                core(10, sphereRadius, outfile, width, height, false, gensurface);
            }
            std::vector<ResultFrame> results;

            const float BoxWidth = 1.0;
            const int NbLoops = 5;// put 200
            const int MaxParticlesPerCell = 32;
            const int MaxBoxDiv = 32;// put 32
            for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
                const float cellWidth = BoxWidth/boxDiv;
                const int nbBoxes = boxDiv*boxDiv*boxDiv;
                for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 2){
                    const float  sphereRadius = cellWidth;
                    const int nbSpheres = nbParticles;

                    std::cout << "NbParticles: " << nbParticles << std::endl;
                    std::cout << "BoxDiv: " << boxDiv << std::endl;
                    std::cout << "CellWidth: " << cellWidth << std::endl;
                    std::cout << "NbLoops: " << NbLoops << std::endl;
                    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                        std::pair<double,double> timeInitCompute = core(nbSpheres, sphereRadius, outfile, width, height, false, gensurface);

                        ResultFrame frame;
                        frame.nbParticles = nbParticles;
                        frame.nbInteractions = nbParticles*(nbParticles/nbBoxes)*27;
                        frame.nbLoops = NbLoops;
                        frame.boxDiv = boxDiv;
                        frame.results.push_back({timeInitCompute.first, timeInitCompute.second, timeInitCompute.first+timeInitCompute.second});
                        results.push_back(frame);
                    }
                }
            }

            std::ofstream file(getFilename(gensurface));
            {
                file << "NbParticles,NbInteractions,NbLoops,boxDiv,nbCells,partspercell,timeinit,timecompute,timetotal";
                file << std::endl;
            }
            for(const ResultFrame& frame : results){
                for(const ResultFrame::AResult& res : frame.results){
                    file << frame.nbParticles << "," << frame.nbInteractions << "," << frame.nbLoops << "," << frame.boxDiv << "," << frame.boxDiv*frame.boxDiv*frame.boxDiv;
                    file << "," << double(frame.nbParticles)/(frame.boxDiv*frame.boxDiv*frame.boxDiv);
                    file << "," << res.timeInit  << "," << res.timeCompute << "," << res.timeTotal;
                    file << std::endl;
                }
            }
        }
    }
    else{
        const float  sphereRadius = 1/double(boxDivisor);

        std::cout << "[LOG] nb spheres = " << nbSpheres << std::endl;
        std::cout << "[LOG] sphere radius = " << sphereRadius << std::endl;
        std::cout << "[LOG] box divisor = " << boxDivisor << std::endl;
        std::cout << "[LOG] outfile = " << outfile << std::endl;
        std::cout << "[LOG] width = " << width << std::endl;
        std::cout << "[LOG] height = " << height << std::endl;

        core(nbSpheres, sphereRadius, outfile, width, height, true, gensurface);
    }

    return 0;
}

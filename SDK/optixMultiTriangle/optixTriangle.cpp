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

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixTriangle.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
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

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

typedef SbtRecord<RayGenDataLJ>     RayGenSbtRecordLJ;
typedef SbtRecord<MissDataLJ>       MissSbtRecordLJ;
typedef SbtRecord<HitGroupDataLJ>   HitGroupSbtRecordLJ;

// void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
// {
//     cam.setEye( {0.0f, 0.0f, 2.0f} );
//     cam.setLookat( {0.0f, 0.0f, 0.0f} );
//     cam.setUp( {0.0f, 1.0f, 3.0f} );
//     cam.setFovY( 45.0f );
//     cam.setAspectRatio( (float)width / (float)height );
// }

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    // Set the eye position to be at a diagonal position
    cam.setEye( {1.0f, 1.0f, 2.0f} );

    // Set the lookat position to be at the center
    cam.setLookat( {0.0f, 0.0f, 0.0f} );

    // Set the up vector to keep the camera oriented correctly
    cam.setUp( {0.0f, 1.0f, 0.0f} );

    // Set the field of view
    cam.setFovY( 45.0f );

    // Set the aspect ratio based on the provided width and height
    cam.setAspectRatio( (float)width / (float)height );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


std::pair<double,double> core(const int nbPoints, const float cutoffRadius, const std::string outfile,
          const int width, const int height, const bool checkResult){
    double timebuild = 0;
    double timecompute = 0;
    try
    {
        std::vector<float3> points;
        // Create random points positions between O and 1
        // TODO
        // for (int i = 0; i < nbPoints; i++)
        // {
        //     float3 point = make_float3( 1.0f * (drand48()), 
        //                                             1.0f * (drand48()), 
        //                                             1.0f * (drand48()) );
        //     points.push_back(point);
        // }
        points.push_back(make_float3(0, 0.5, 0));
        points.push_back(make_float3(0, 0.6, 0));

        std::vector<float> pointsEnergy;
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
#ifdef DEBUG
            // This may incur significant performance cost and should only be done during development.
            options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }


        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // OPTIX_BUILD_FLAG_NONE;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // Triangle build input: simple list of three vertices
            std::vector<float3> vertices;
            // For each point we create four triangles
            // that create a panel in front and behind.
            // So each panel is composed of two triangles.
            for(int i = 0; i < nbPoints; i++)
            {
                const float epsilon = 1.001f;
                const float3 point = points[i];
                std::array<float3, 8> corners;
                for(int idxCorner = 0 ; idxCorner < 8 ; ++idxCorner){
                    corners[idxCorner].z = point.z + (idxCorner&1 ? cutoffRadius/2 : -cutoffRadius/2 ) * epsilon;
                    corners[idxCorner].y = point.y + (idxCorner&2 ? cutoffRadius/2 : -cutoffRadius/2 ) * epsilon;
                    corners[idxCorner].x = point.x + (idxCorner&4 ? cutoffRadius/2 : -cutoffRadius/2 ) * epsilon;
                    // TODO
                    std::cout << " - Corner " << idxCorner << " = " << corners[idxCorner].x << " " << corners[idxCorner].y << " " << corners[idxCorner].z << std::endl;
                }
                vertices.push_back(corners[0]);
                vertices.push_back(corners[1]);
                vertices.push_back(corners[3]);

                vertices.push_back(corners[0]);
                vertices.push_back(corners[2]);
                vertices.push_back(corners[3]);

                vertices.push_back(corners[4]);
                vertices.push_back(corners[5]);
                vertices.push_back(corners[7]);

                vertices.push_back(corners[4]);
                vertices.push_back(corners[6]);
                vertices.push_back(corners[7]);
            }

            const size_t vertices_size = sizeof( float3 )*vertices.size();
            CUdeviceptr d_vertices=0;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_vertices ),
                        vertices.data(),
                        vertices_size,
                        cudaMemcpyHostToDevice
                        ) );

            // Our build input is a simple list of non-indexed triangle vertices
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
            triangle_input.triangleArray.vertexBuffers = &d_vertices;
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        &triangle_input,
                        1, // Number of build inputs
                        &gas_buffer_sizes
                        ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_temp_buffer_gas ),
                        gas_buffer_sizes.tempSizeInBytes
                        ) );
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_gas_output_buffer ),
                        gas_buffer_sizes.outputSizeInBytes
                        ) );

            SpTimer timer;

            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,                  // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                        ) );

            timer.stop();
            std::cout << "[TIME] Time to build gas: " << timer.getElapsed() << " s" << std::endl;
            timebuild = timer.getElapsed();

            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
        }

        {
            //
            // Create module
            //
            OptixModule module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                OptixModuleCompileOptions module_compile_options = {};
    #if !defined( NDEBUG )
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif

                pipeline_compile_options.usesMotionBlur        = false;
                pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipeline_compile_options.numPayloadValues      = 3;
                pipeline_compile_options.numAttributeValues    = 3;
                pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

                size_t      inputSize  = 0;
                const char* input      = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixTriangle.cu", inputSize );

                OPTIX_CHECK_LOG( optixModuleCreate(
                            context,
                            &module_compile_options,
                            &pipeline_compile_options,
                            input,
                            inputSize,
                            LOG, &LOG_SIZE,
                            &module
                            ) );
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
                RayGenSbtRecord rg_sbt;
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

                sutil::Camera cam;
                configureCamera( cam, width, height );

                Params params;
                params.image        = output_buffer.map();
                params.image_width  = width;
                params.image_height = height;
                params.handle       = gas_handle;
                params.cam_eye      = cam.eye();
                cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

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
                    sutil::displayBufferWindow( "Multi triangles", buffer );
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
                // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

                OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
                OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
                OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
                OPTIX_CHECK( optixModuleDestroy( module ) );

                // OPTIX_CHECK( optixDeviceContextDestroy( context ) );
            }
        }


        {
            //
            // Create module
            //
            OptixModule module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                OptixModuleCompileOptions module_compile_options = {};
    #if !defined( NDEBUG )
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif

                pipeline_compile_options.usesMotionBlur        = false;
                pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipeline_compile_options.numPayloadValues      = 7;
                pipeline_compile_options.numAttributeValues    = 3;
                pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

                size_t      inputSize  = 0;
                const char* input      = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixLJ.cu", inputSize );

                OPTIX_CHECK_LOG( optixModuleCreate(
                            context,
                            &module_compile_options,
                            &pipeline_compile_options,
                            input,
                            inputSize,
                            LOG, &LOG_SIZE,
                            &module
                            ) );
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
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &output_buffer ), nbPoints * sizeof( float ) ) );
                CUDA_CHECK( cudaMemset( reinterpret_cast<void*>( output_buffer ), 0, nbPoints * sizeof( float ) ));
            }

            //
            // launch
            //
            {
                const size_t pointsCopyLeadingDim = ((nbPoints+31)/32)*32;
                std::vector<float> pointsCopy(pointsCopyLeadingDim*3);
                for(size_t idxPos = 0 ; idxPos < points.size() ; ++idxPos)
                {
                    const auto pos = points[idxPos];
                    pointsCopy[idxPos] = pos.x;
                    pointsCopy[idxPos + pointsCopyLeadingDim] = pos.y;
                    pointsCopy[idxPos + pointsCopyLeadingDim*2] = pos.z;
                }
                CUdeviceptr d_points;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_points ), 
                                        pointsCopyLeadingDim*3 * sizeof( float ) ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_points ),
                            pointsCopy.data(), pointsCopyLeadingDim*3 * sizeof( float ),
                            cudaMemcpyHostToDevice
                            ) );
            

                CUstream stream;
                CUDA_CHECK( cudaStreamCreate( &stream ) );

                ParamsLJ params;
                params.num_points = nbPoints;
                params.leading_dim = pointsCopyLeadingDim;
                params.points     = reinterpret_cast<float*>(d_points);
                params.c          = cutoffRadius;
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

                const int nbRays = 4;
                OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( ParamsLJ ), &sbt, nbPoints, nbRays, /*depth=*/1 ) );
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

                OPTIX_CHECK( optixDeviceContextDestroy( context ) );

                // Added
                pointsEnergy.resize(nbPoints, 0);
                CUDA_CHECK( cudaMemcpy( pointsEnergy.data(), reinterpret_cast<void*>( output_buffer ), 
                                       nbPoints * sizeof( float ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( output_buffer ) ) );
            }
        }

        if(checkResult){
            // We will compute the LJ interaction between the particles on the cpu
            // to compare the results
            // We will compute only for the first 100 particles
            for(size_t idxTarget = 0 ; idxTarget < std::min(100UL, points.size()) ; ++idxTarget){
                float energy = 0.0f;

                for(size_t idxSource = 0 ; idxSource < points.size() ; ++idxSource){
                    if(idxSource != idxTarget){
                        const auto posSource = points[idxSource];
                        const auto posTarget = points[idxTarget];
                        const auto diff = posSource - posTarget;
                        const float dist = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                        if(dist < cutoffRadius){
                            std::cout << " - SRC pos = " << points[idxSource].x << " " << points[idxSource].y << " " << points[idxSource].z << std::endl;
                            energy += 4.0f * (pow(1.0f/dist, 12) - pow(1.0f/dist, 6));
                        }
                    }
                }
                std::cout << "Energy for particle " << idxTarget << " is " << energy 
                    << " it has been computed as " << (pointsEnergy[idxTarget]) << std::endl;
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

std::string getFilename(){
    // Get the current date and time
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Format the date and time
    std::tm* ptm = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", ptm);

    // Create the filename with the date and time
    std::string filename = "results-" + std::string(buffer) + ".csv";
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
    std::cerr << "         --nbpoints | -np           Specify the number of particles\n";
    std::cerr << "         --boxdivisor | -bd          Specify the divisor to compute the cutoff (cutoff = 1/x) \n";
    std::cerr << "         --runbench | -rb            Ask to run the benchmark (erase all the other arguments) \n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}

int main( int argc, char* argv[] )
{
    std::string outfile;
    int         nbPoints = 10;
    int         boxDivisor = 2;
    int         width  = 1024;
    int         height =  768;
    bool        runBench = false;

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
        else if( arg == "--nbpoints" || arg == "-np" )
        {
            if( i < argc - 1 )
            {
                nbPoints = atoi(argv[++i]);
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
        std::vector<ResultFrame> results;

        const float BoxWidth = 1.0;
        const int NbLoops = 5;// put 200
        const int MaxParticlesPerCell = 32;
        const int MaxBoxDiv = 32;// put 32
        for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
            const float cellWidth = BoxWidth/boxDiv;
            const int nbBoxes = boxDiv*boxDiv*boxDiv;
            for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 2){
                const float  cutoffRadius = cellWidth;
                const int nbPoints = nbParticles;

                std::cout << "NbParticles: " << nbParticles << std::endl;
                std::cout << "BoxDiv: " << boxDiv << std::endl;
                std::cout << "CellWidth: " << cellWidth << std::endl;
                std::cout << "NbLoops: " << NbLoops << std::endl;
                std::cout << "NbLoops: " << NbLoops << std::endl;

                std::pair<double,double> timeInitCompute = core(nbPoints, cutoffRadius, outfile, width, height, false);

                ResultFrame frame;
                frame.nbParticles = nbParticles;
                frame.nbInteractions = nbParticles*(nbParticles/nbBoxes)*27;
                frame.nbLoops = NbLoops;
                frame.boxDiv = boxDiv;
                frame.results.push_back({timeInitCompute.first, timeInitCompute.second, timeInitCompute.first+timeInitCompute.second});
                results.push_back(frame);
            }
        }

        std::ofstream file(getFilename());
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
        const float  cutoffRadius = 1/double(boxDivisor);

        std::cout << "[LOG] nb points = " << nbPoints << std::endl;
        std::cout << "[LOG] cutoff cutoffRadius = " << cutoffRadius << std::endl;
        std::cout << "[LOG] box divisor = " << boxDivisor << std::endl;
        std::cout << "[LOG] outfile = " << outfile << std::endl;
        std::cout << "[LOG] width = " << width << std::endl;
        std::cout << "[LOG] height = " << height << std::endl;

        core(nbPoints, cutoffRadius, outfile, width, height, true);
    }

    return 0;
}


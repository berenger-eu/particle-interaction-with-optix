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

#include "optixSphere.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ ParamsLJ params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float*                 energy
        )
{
    unsigned int p0;
    p0 = __float_as_uint( *energy );

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0);

    (*energy) = __uint_as_float( p0 );
}

static __forceinline__ __device__ void setPayloadEnergy( float p )
{
    optixSetPayload_0( __float_as_uint( p ) );
}


static __forceinline__ __device__ float getPayloadEnergy()
{
    return __uint_as_float( optixGetPayload_0() );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    const int point_index = idx.x;
    const int ray_index = idx.y;

    const RayGenDataLJ* rtData = (RayGenDataLJ*)optixGetSbtDataPointer();
    const Point point = params.points[point_index];
    const float c = params.c;

    const float3 ray_origins[3] = {
        make_float3(point.position.x - c, point.position.y, point.position.z),
        make_float3(point.position.x, point.position.y - c, point.position.z),
        make_float3(point.position.x, point.position.y, point.position.z - c)
    };

    const float3 ray_directions[3] = {
        make_float3(2 * c, 0, 0),
        make_float3(0, 2 * c, 0),
        make_float3(0, 0, 2 * c)
    };

    const float3 origin = ray_origins[ray_index];
    const float3 direction = normalize(ray_directions[ray_index]);

    float payload_energy = 0;
    trace( params.handle,
            origin,
            direction,
            0.00f,  // tmin
            2 * c,  // tmax
            &payload_energy );

    params.energy[point_index] += payload_energy;
}


extern "C" __global__ void __miss__ms()
{
    // setPayloadEnergy(0.0f);
}


static __forceinline__ __device__  float distance(const float3 p1,
                                                  const float3 p2) {
    // Particles are distributed randomly in the box, so we need a softening factor
    const float softening = 1e-5;
    return sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) + (p2.z - p1.z)*(p2.z - p1.z) + softening);
}

// Function to calculate the Lennard-Jones potential between two particles
static __forceinline__ __device__  float  lennardJonesPotential(const float3 p1, 
                                                                const float3 p2, 
                                                                const float epsilon, 
                                                                const float sigma) {
    const float r = distance(p1, p2);
    const float sigma_d_r = sigma / r;
    const float r6 = (sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r);
    const float r12 = r6 * r6;
    const float result = float(4) * epsilon * (r12 - r6);
    return result;
}


extern "C" __global__ void __closesthit__ch()
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );

    // float  t_hit = optixGetRayTmax();
    // Backface hit not used.
    // float  t_hit2 = __uint_as_float( optixGetAttribute_0() ); 
    // float3 world_raypos = ray_orig + t_hit * ray_dir;
    // float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    // float3 obj_normal   = ( obj_raypos - make_float3( q ) ) / q.w;
    // float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );

    const float epsilon = 1.0f;
    const float sigma = 1.0f;
    const float energy = lennardJonesPotential(ray_orig, make_float3(q.x, q.y, q.z), 
                                               epsilon, sigma);

    setPayloadEnergy( energy );
}

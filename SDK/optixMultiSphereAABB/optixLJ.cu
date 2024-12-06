//
// Copyright (c) Inria 2024
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
        float                  c,
        float*                 energy
        )
{
    unsigned int p0, p1;
    p0 = __float_as_uint( c );
    p1 = __float_as_uint( *energy );

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1);
    
    (*energy) += __uint_as_float( p1 );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    const int point_index = idx.x;

    float3 point;
    point.x = params.points[point_index];
    point.y = params.points[point_index + params.leading_dim];
    point.z = params.points[point_index + params.leading_dim*2];
    const float c = params.c;
    // const float half_ray = params.c*0.81649658092f; // sqrt(2/3)

    // const int ray_index = idx.y;
    // const float3 ray_origins[3] = {
    //     make_float3(point.x - half_ray, point.y, point.z),
    //     make_float3(point.x, point.y - half_ray, point.z),
    //     make_float3(point.x, point.y, point.z - half_ray)
    // };
    // const float3 ray_directions[3] = {
    //     make_float3(2 * half_ray, 0, 0),
    //     make_float3(0, 2 * half_ray, 0),
    //     make_float3(0, 0, 2 * half_ray)
    // };
    // const float3 origin = ray_origins[ray_index];
    // const float3 direction = normalize(ray_directions[ray_index]);

    const float3 direction = make_float3(0, 0, 1);

    float payload_energy = 0;
    trace( params.handle,
            point,
            direction,
            0.00f,  // tmin
            1E-16f,  // tmax
            c,
            &payload_energy );

    atomicAdd(&params.energy[point_index], payload_energy);
}


static __forceinline__ __device__ float getPayloadC()
{
    return __uint_as_float( optixGetPayload_0() );
}

static __forceinline__ __device__ void setPayloadEnergy( float p )
{
    optixSetPayload_1( __float_as_uint( p ) );
}

static __forceinline__ __device__ float getPayloadEnergy()
{
    return __uint_as_float( optixGetPayload_1() );
}


extern "C" __global__ void __miss__ms()
{
}


// Function to calculate the Lennard-Jones potential between two particles
static __forceinline__ __device__  float  lennardJonesPotential(const float3 p1, 
                                                                const float3 p2, 
                                                                const float dist_squared,
                                                                const float epsilon, 
                                                                const float sigma) {
    const float r = sqrt(dist_squared);//distance(p1, p2);
    const float sigma_d_r = sigma / r;
    const float r6 = (sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r);
    const float r12 = r6 * r6;
    const float result = float(4) * epsilon * (r12 - r6);
    return result;
}


extern "C" __global__ void __anyhit__ch()
{
    optixIgnoreIntersection();
}



extern "C" __global__ void __intersection__sphere()
{
    // Retrieve sphere center and radius
    const int point_index = optixGetPrimitiveIndex();
    float3 sphere_center;
    sphere_center.x = params.points[point_index];
    sphere_center.y = params.points[point_index + params.leading_dim];
    sphere_center.z = params.points[point_index + params.leading_dim*2];
    const float radius = getPayloadC();

    // Retrieve ray origin
    const float3 ray_orig = optixGetWorldRayOrigin();

    // Compute squared distance from ray origin to sphere center
    const float3 offset = ray_orig - sphere_center;
    const float distance_squared = dot(offset, offset);
    const float radius_squared = radius * radius;

    // // // TODO remove
    // {
    //     const int primitive_index = optixGetPrimitiveIndex();
    //     const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
    //     printf("primitive_index %d sbtGASIndex %d] ray_orig: %f %f %f  -- sphere_center: %f %f %f \n",
    //             primitive_index, sbtGASIndex, 
    //             ray_orig.x, ray_orig.y, ray_orig.z, sphere_center.x, sphere_center.y, sphere_center.z);
    // }

    // Check if the ray origin is inside the sphere
    if (distance_squared <= radius_squared && distance_squared >= 1E-16f)
    {
        const float epsilon = 1.0f;
        const float sigma = 1.0f;
        const float energy = lennardJonesPotential(sphere_center, ray_orig, distance_squared,
                                                    epsilon, sigma);

        setPayloadEnergy( getPayloadEnergy() + energy );

        // Report intersection at the origin
        optixReportIntersection(0.0f,// t = 0.0f
                                0 // kind
                                ); 
    }
}

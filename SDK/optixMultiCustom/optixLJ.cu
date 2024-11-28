//
// Copyright (c) Inria 2024
//

#include <optix.h>

#include "optixMultiCustom.h"
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
        float3                 partPos,
        float                  c,
        float*                 energy
        )
{
    unsigned int p0, p1, p2, p3, p4;
    p0 = __float_as_uint( partPos.x );
    p1 = __float_as_uint( partPos.y );
    p2 = __float_as_uint( partPos.z );
    p3 = __float_as_uint( *energy );
    p4 = __float_as_uint( c );

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT ,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2, p3, p4);
    
    (*energy) += __uint_as_float( p3 );
}

static __forceinline__ __device__ float getPayloadC()
{
    return __uint_as_float( optixGetPayload_4() );
}

static __forceinline__ __device__ void setPayloadEnergy( float p )
{
    optixSetPayload_3( __float_as_uint( p ) );
}

static __forceinline__ __device__ float getPayloadEnergy()
{
    return __uint_as_float( optixGetPayload_3() );
}

static __forceinline__ __device__ float3 getPayloadPartPos()
{
    float3 point;
    point.x = __uint_as_float( optixGetPayload_0() );
    point.y = __uint_as_float( optixGetPayload_1() );
    point.z = __uint_as_float( optixGetPayload_2() );
    return point;
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    const int point_index = idx.x;

    const RayGenDataLJ* rtData = (RayGenDataLJ*)optixGetSbtDataPointer();
    float3 point;
    point.x = params.points[point_index];
    point.y = params.points[point_index + params.leading_dim];
    point.z = params.points[point_index + params.leading_dim*2];
    const float c = params.c;

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
            point,
            c,
            &payload_energy );

    atomicAdd(&params.energy[point_index], payload_energy);
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
    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );

    const float3 point = getPayloadPartPos();
    const float3 diff_pos{fabsf(point.x - q.x), fabsf(point.y - q.y), fabsf(point.z - q.z)};
    const float3 diff_pos_squared{diff_pos.x * diff_pos.x, diff_pos.y * diff_pos.y, diff_pos.z * diff_pos.z};
    const float dist_squared = diff_pos_squared.x + diff_pos_squared.y + diff_pos_squared.z;
    const float c = getPayloadC();

    if(dist_squared < c*c){
        const float epsilon = 1.0f;
        const float sigma = 1.0f;
        const float energy = lennardJonesPotential(point, make_float3(q.x, q.y, q.z), dist_squared,
                                                    epsilon, sigma);

        setPayloadEnergy( getPayloadEnergy() + energy );
    }

    // Backface hit not used.
    // float  t_hit2 = __uint_as_float( optixGetAttribute_0() ); 
    // float3 world_raypos = ray_orig + t_hit * ray_dir;
    // float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    // float3 obj_normal   = ( obj_raypos - make_float3( q ) ) / q.w;
    // float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );
    // optixGetRayTmax();
    // optixTerminateRay();
    optixIgnoreIntersection();
}
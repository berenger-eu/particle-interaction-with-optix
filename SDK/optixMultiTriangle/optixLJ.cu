//
// Copyright (c) Inria 2024
//

#include <optix.h>

#include "optixTriangle.h"
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
        float*                 energy,
        unsigned int           ray_idx
        )
{
    unsigned int p0, p1, p2, p3, p4, p5;
    p0 = __float_as_uint( partPos.x );
    p1 = __float_as_uint( partPos.y );
    p2 = __float_as_uint( partPos.z );
    p3 = __float_as_uint( *energy );
    p4 = __float_as_uint( c );
    p5 = ray_idx;

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
            p0, p1, p2, p3, p4, p5);
    
    (*energy) += __uint_as_float( p3 );
}
static __forceinline__ __device__ unsigned int getPayloadRayidx()
{
    return optixGetPayload_5();
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
    const float half_c = params.c/2;
    const float half_ray = (params.c/2);

    // Coordinates are:
    //  1 --- 5
    //  |\   |\
    //  | \  | \
    //  0 -3- 4  7
    //   \ |  \ |   
    //    \|   \|
    //     2 --- 6
    // First triangle is 0,1,3
    // Second triangle is 0,2,3
    // Third triangle is 4,5,7
    // Fourth triangle is 4,6,7

    float3 origin;
    float3 direction;
    {
        float ycoef = (idx.y & 1) ? -1.0f : 1.0f;
        float zcoef = (idx.y & 2) ? -1.0f : 1.0f;

        origin = make_float3(point.x - half_ray,
                             point.y + half_c * ycoef,
                             point.z + half_c * zcoef);
        direction = make_float3(1, 0, 0);
    }

    float payload_energy = 0;
    const float feps = 1.19209290e-07F;
    const float tmin = feps;
    const float tmax = (2 * half_ray) + (2 * half_ray) * 0.00001;
    trace( params.handle,
            origin,
            direction,
            tmin,
            tmax,
            point,
            c,
            &payload_energy ,
            idx.y);
    
    atomicAdd(&params.energy[point_index], payload_energy);
}


extern "C" __global__ void __miss__ms()
{
}


static __forceinline__ __device__  float  distance(const float3& p1, const float3& p2) {
    return sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) + (p2.z - p1.z)*(p2.z - p1.z));
}

// Function to calculate the Lennard-Jones potential between two particles
static __forceinline__ __device__  float  lennardJonesPotential(const float3 p1, 
                                                                const float3 p2, 
                                                                const float dist_p1_p2,
                                                                const float epsilon, 
                                                                const float sigma) {
    const float r = dist_p1_p2;//distance(p1, p2);
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

    float3 vertices[3];
    optixGetTriangleVertexData( gas, prim_idx, sbtGASIndex, 0.f, vertices );

    float3 q;
    q.y = (max(vertices[0].y,max(vertices[1].y, vertices[2].y)) + min(vertices[0].y,min(vertices[1].y, vertices[2].y)))/2;
    q.z = (max(vertices[0].z,max(vertices[1].z, vertices[2].z)) + min(vertices[0].z,min(vertices[1].z, vertices[2].z)))/2;

    const float c = getPayloadC();
    if((prim_idx % 4) < 2){
        q.x = vertices[0].x + c/2;
    }
    else{
        q.x = vertices[0].x - c/2;
    }

    const float3 point = getPayloadPartPos();

    const float dist_p1_p2 = distance(point, q);
    if(dist_p1_p2 < c && dist_p1_p2 > 0.0001){
        // const float3 ray_orig = optixGetWorldRayOrigin();
        // const float3 ray_dir  = optixGetWorldRayDirection();

        const unsigned int ray_idx = getPayloadRayidx();
        const bool is_ray_for_compute = (point.y != q.y && point.z != q.z) ||
                                        ((point.z < q.z && ray_idx == 0) || (point.z > q.z && ray_idx == 2)) ||
                                        ((point.y < q.y && ray_idx == 0) || (point.y > q.y && ray_idx == 1)) ||
                                        ray_idx == 0;// y and z are same

        if(is_ray_for_compute){
            // const float3 hit_position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
            const float epsilon = 1.0f;
            const float sigma = 1.0f;
            const float energy = lennardJonesPotential(point, q, dist_p1_p2,
                                                       epsilon, sigma);

            setPayloadEnergy( getPayloadEnergy() + energy );
        }
    }

    // Backface hit not used.
    // float  t_hit2 = __uint_as_float( optixGetAttribute_0() ); 
    // float3 world_raypos = ray_orig + t_hit * ray_dir;
    // float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    // float3 obj_normal   = ( obj_raypos - make_float3( q ) ) / q.w;
    // float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );
    // optixTerminateRay();
    optixIgnoreIntersection();
}

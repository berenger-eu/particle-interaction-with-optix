
This repository contains the source code of OptixLJ, a project where we compute particle interactions with a cutoff method on GPU, managing the interaction lists using OptiX and ray tracing.

You can look at the related publication https://inria.hal.science/hal-04677813 .

# Organization

- The `Cuda-version` directory contains the pure CUDA implementation, where a grid of cells is used.
- The `SDK/optixMultiSphere` and `SDK/optixMultiTriangle` directories contain the OptiX-based implementations that use built-in primitives.
- The `SDK/optixMultiSphereAABB` and `SDK/optixMultiCustom` directories contain the AABB based implementations.

# Compiling

The CUDA version should be compiled with CMake, but it is required to specify the CUDA SM version. For example, you can compile with:

```bash
cmake .. -DCUDA_SM=75 -DCMAKE_BUILD_TYPE=Release
```

The OptiX implementations have been put directly into the SDK, which should also be compiled with CMake. Note that it is advised to build in a directory outside of the SDK directory.

You can also look at the batch files used to compile and execute the bench `a100.batch` and `rtx8000.batch`.

# License

Apart from specific files located in `SDK/optixMultiSphere`, `SDK/optixMultiTriangle`, `optixMultiSphereAABB` and `optixMultiCustom`, which are under the MIT License, the rest of the SDK directory is under the NVIDIA proprietary license.

`Cuda-version` is also under the MIT License.

# How the code is build

In each of the four directories the same organization is used. We put here some snippets, but the latest version will be in the code.
For example, in the `optixMultiTriangle` directory, there are:
- `optixTriangle.cpp` There is the code to generate the particles:
```cpp
if(gensurface){
            for(int idxSphere = 0; idxSphere < nbPoints; idxSphere++)
            {
                double theta = 2.0 * M_PI * drand48(); // Random angle between 0 and 2π
                double phi = acos(1.0 - 2.0 * drand48()); // Random angle between 0 and π

                // Convert spherical coordinates to Cartesian coordinates
                double x = sin(phi) * cos(theta) + 0.5;
                double y = sin(phi) * sin(theta) + 0.5;
                double z = cos(phi) + 0.5;

                const float3 sphereVertex = make_float3( x, y, z);
                points.push_back( sphereVertex );
            }                    
        }
        else{
            for (int i = 0; i < nbPoints; i++)
            {
                float3 point = make_float3( 1.0f * (drand48()), 
                                                        1.0f * (drand48()), 
                                                        1.0f * (drand48()) );
                points.push_back(point);
            }
        }
```

We convert and copy the geometric data:
```cpp
            for(int i = 0; i < nbPoints; i++)
            {
                const float3 point = points[i];
                std::array<float3, 8> corners;
                for(int idxCorner = 0 ; idxCorner < 8 ; ++idxCorner){
                    corners[idxCorner].z = point.z + (idxCorner&1 ? add_smallest_increment(cutoffRadius/2) : subtract_smallest_increment(-cutoffRadius/2) );
                    corners[idxCorner].y = point.y + (idxCorner&2 ? add_smallest_increment(cutoffRadius/2) : subtract_smallest_increment(-cutoffRadius/2) );
                    corners[idxCorner].x = point.x + (idxCorner&4 ? cutoffRadius/2 : -cutoffRadius/2 );
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
```

We launch the rays:
```cpp
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
```
- In the `.cu` file (like `optixLJ.cu`) there is the function that generates a ray:
```cpp
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
```
The interaction kernel:
```cpp
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
```
And the hit callback:
```cpp
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
```
- In `optixTriangle.cpp` again, there is the code that generates the different test cases:
```cpp
    const int NbLoops = 5;// put 200
    const int MaxParticlesPerCell = 32;
    const int MaxBoxDiv = 256;// put 32
    for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
        const int nbBoxes = boxDiv*boxDiv*boxDiv;
        for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 2){
            const double particlePerCell = double(nbParticles)/double(nbBoxes);
            const double expectedNbNeighbors = 9*particlePerCell;
            const double coef = 1. - ((2*expectedNbNeighbors)/nbParticles);
            const double validCoef = std::min(1.0, std::max(-1.0, coef));
            const double sphereRadius = acos(validCoef);

            std::cout << "NbParticles: " << nbParticles << std::endl;
            std::cout << "BoxDiv: " << boxDiv << std::endl;
            std::cout << "CellWidth: " << sphereRadius << std::endl;
            std::cout << "NbLoops: " << NbLoops << std::endl;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::pair<double,double> timeInitCompute = core(nbPoints, sphereRadius, outfile, width, height, false, gensurface);

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
```

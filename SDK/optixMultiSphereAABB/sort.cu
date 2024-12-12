#include <cuda_runtime.h>
#include "sort.cuh"

#include "prefixsum.hpp"

__global__ void count_per_cell(const int N, const float3* pointsInput,
                     int* counter, float radius, const float width){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        const float3 point = pointsInput[id];
        const int cell_x = (int) (point.x / radius);
        const int cell_y = (int) (point.y / radius);
        const int cell_z = (int) (point.z / radius);
        const int cell_id = cell_x + cell_y * width + cell_z * width * width;
        assert(cell_id < width * width * width);
        atomicAdd(&counter[cell_id], 1);
    }
}


__global__ void move_particles(const int N, float3* pointsOutput, const float3* pointsInput,
                               int* counter, float radius, const float width){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        const float3 point = pointsInput[id];
        const int cell_x = (int) (point.x / radius);
        const int cell_y = (int) (point.y / radius);
        const int cell_z = (int) (point.z / radius);
        const int cell_id = cell_x + cell_y * width + cell_z * width * width;
        assert(cell_id < width * width * width);
        const int insertPos = atomicAdd(&counter[cell_id], 1);
        pointsOutput[insertPos] = point;
    }
}


__host__ void reorder(const int N, float3* pointsOutput, const float3* pointsInput,
                     int* particlesPerCell, int* prefixParCell, float radius, cudaStream_t stream){
    const int width = 1.0f / radius;
    const int nbCells = width * width * width;
    
    cudaMemsetAsync(particlesPerCell, nbCells * sizeof(int), 0, stream);

    count_per_cell<<<(N + 255) / 256, 256, 0, stream>>>(N, pointsInput, particlesPerCell, radius, width);

    cudaMemset(prefixParCell, 0, (nbCells+1)*sizeof(int));

    PrefixFullV2(particlesPerCell, prefixParCell+1, nbCells, stream);

    move_particles<<<(N + 255) / 256, 256, 0, stream>>>(N, pointsOutput, pointsInput, 
                                                        prefixParCell, radius, width);
}

#include <cuda_runtime.h>
#include "sort.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

__global__ void count_per_cell(const int N, float4* points,
                    float radius, const float width){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        const float4 point = points[id];
        const int cell_x = (int) (point.x / radius);
        const int cell_y = (int) (point.y / radius);
        const int cell_z = (int) (point.z / radius);
        const int cell_id = cell_x + cell_y * width + cell_z * width * width;
        points[id].w = cell_id;
    }
}

struct float4_compare {
    __host__ __device__
    bool operator()(const float4 &a, const float4 &b) const {
        return a.w < b.w;
    }
};


__host__ void reorder(const int N, float4* points, float radius){
    const int width = 1.0f / radius;
    
    count_per_cell<<<(N + 255) / 256, 256>>>(N, points, radius, width);

    // Copy data to device
    thrust::device_vector<float4> d_array(points, points + N);

    // Sort on the device using the custom comparator
    thrust::sort(d_array.begin(), d_array.end(), float4_compare());

    cudaMemcpy(points, thrust::raw_pointer_cast(d_array.data()), 
               N * sizeof(float4), cudaMemcpyDeviceToDevice);
}

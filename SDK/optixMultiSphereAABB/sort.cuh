#ifndef SORT_CUH
#define SORT_CUH

__host__ void reorder(const int N, float3* pointsOutput, const float3* pointsInput,
                     int* particlesPerCell, int* prefixParCell, float radius, cudaStream_t stream);


#endif
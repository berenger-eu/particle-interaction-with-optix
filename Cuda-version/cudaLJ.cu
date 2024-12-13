#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <list>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpTimer.hpp"
#include "point3d.hpp"
#include "prefixsum.hpp"

#ifndef CUDA_ASSERT
#define CUDA_ASSERT(X)\
{\
    cudaError_t ___resCuda = (X);\
    if ( cudaSuccess != ___resCuda ){\
    printf("Error: fails, %s (%s line %d)\nbCols", cudaGetErrorString(___resCuda), __FILE__, __LINE__ );\
    exit(1);\
    }\
    }
#endif

template <class NumType>
__device__ __host__ NumType distance(const Point3D<NumType>& p1, const Point3D<NumType>& p2) {
    // Particles are distributed randomly in the box, so we need a softening factor
    const NumType softening = 1e-5;
    return sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) + (p2.z - p1.z)*(p2.z - p1.z) + softening);
}

template <class NumType>
// Function to calculate the Lennard-Jones potential between two particles
__device__ __host__ NumType  lennardJonesPotential(const Point3D<NumType>& p1, const Point3D<NumType>& p2, 
                                         const NumType epsilon, const NumType sigma) {
    const NumType r = distance(p1, p2);
    assert(r != 0);
    assert(!isnan(r));
    assert(!isinf(r));

    const NumType sigma_d_r = sigma / r;
    assert(!isnan(sigma_d_r) && !isinf(sigma_d_r));

    const NumType r6 = (sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r)*(sigma_d_r*sigma_d_r);
    assert(!isnan(r6) && !isinf(r6));

    const NumType r12 = r6 * r6;
    assert(!isnan(r12) && !isinf(r12));

    const NumType result = NumType(4) * epsilon * (r12 - r6);
    assert(!isnan(result) && !isinf(result));
    return result;
}

// ParticlesContainer is an array of pointers
// for the particles' data.
template <typename FloatType>
struct ParticlesContainer{
    FloatType* x;
    FloatType* y;
    FloatType* z;
    FloatType* v;
    int* index;
    int nbParticles;

    void swap(ParticlesContainer& inOther){
        std::swap(this->x, inOther.x);
        std::swap(this->y, inOther.y);
        std::swap(this->z, inOther.z);
        std::swap(this->v, inOther.v);
        std::swap(this->index, inOther.index);
        std::swap(this->nbParticles, inOther.nbParticles);
    }

    void copy(const ParticlesContainer& inOther){
        assert(this->nbParticles == inOther.nbParticles);
        CUDA_ASSERT( cudaMemcpy(this->x, inOther.x, inOther.nbParticles*sizeof(FloatType), cudaMemcpyDeviceToDevice) );
        CUDA_ASSERT( cudaMemcpy(this->y, inOther.y, inOther.nbParticles*sizeof(FloatType), cudaMemcpyDeviceToDevice) );
        CUDA_ASSERT( cudaMemcpy(this->z, inOther.z, inOther.nbParticles*sizeof(FloatType), cudaMemcpyDeviceToDevice) );
        CUDA_ASSERT( cudaMemcpy(this->v, inOther.v, inOther.nbParticles*sizeof(FloatType), cudaMemcpyDeviceToDevice) );
        CUDA_ASSERT( cudaMemcpy(this->index, inOther.index, inOther.nbParticles*sizeof(int), cudaMemcpyDeviceToDevice) );
    }
};

// A cell is simply an interval telling
// where the particle of a cell are located
// in an ParticlesContainer
struct CellDescriptor{
    int nbParticleInCell;
    int offsetInContainer;
};

// Init the table of random value, only useful because we move
// particles randmly
__global__ void InitRandStates(curandState_t* particleCurandStates, const int inNbElements){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int idxPart = uniqueIdx; idxPart < inNbElements ; idxPart += nbThreads){
        curand_init(1234, uniqueIdx, 0, &particleCurandStates[idxPart]);
    }
}


// This function will convert a position into a 3D coord
// in the grid
template <typename NumType>
__device__ Point3D<int> PosToCellCoord(const Point3D<NumType>& inCellWidth,
                                                   const NumType inPosX, const NumType inPosY,
                                                   const NumType inPosZ){
    return Point3D<int>{
        static_cast<int>(inPosX / inCellWidth.x),
        static_cast<int>(inPosY / inCellWidth.y),
        static_cast<int>(inPosZ / inCellWidth.z)
    };
}

template <typename NumType>
__device__ Point3D<int> PosToCellCoord(const Point3D<NumType>& inCellWidth,
                                                   const Point3D<NumType>& inPos){
    return PosToCellCoord(inCellWidth, inPos.x, inPos.y, inPos.z);
}

// Convert a coord in the grid into a linear index
__device__ int CellCoordToIndex(const Point3D<int>& inGridDim,
                                              const Point3D<int>& inCellCoord){
    assert(0 <= inCellCoord.x && inCellCoord.x < inGridDim.x);
    assert(0 <= inCellCoord.y && inCellCoord.y < inGridDim.y);
    assert(0 <= inCellCoord.z && inCellCoord.z < inGridDim.z);
    return ((inCellCoord.z * inGridDim.y) + inCellCoord.y ) * inGridDim.x + inCellCoord.x;
}

__device__ Point3D<int> IndexToCoord(const Point3D<int>& inGridDim,
                                          int inIndex){
    Point3D<int> coord;
    coord.z = inIndex/(inGridDim.y*inGridDim.x);
    coord.y = (inIndex - coord.z*inGridDim.y*inGridDim.x)/inGridDim.x;
    coord.x = inIndex%inGridDim.x;
    assert(0 <= coord.x && coord.x < inGridDim.x);
    assert(0 <= coord.y && coord.y < inGridDim.y);
    assert(0 <= coord.z && coord.z < inGridDim.z);
    return coord;
}

// Periodic boundary condition
template <typename NumType>
__device__ NumType PBC(const NumType inPos, const NumType inLimit){
    if(inPos < 0) return inPos + inLimit;
    else if(inPos >= inLimit) return inPos-inLimit;
    return inPos;
}

// Min
template <typename NumType>
__device__ NumType M_Min(const NumType inVal1, const NumType inVal2){
    return inVal1 < inVal2 ? inVal1 : inVal2;
}

// Max
template <typename NumType>
__device__ NumType M_Max(const NumType inVal1, const NumType inVal2){
    return inVal1 > inVal2 ? inVal1 : inVal2;
}

// Abs
template <typename NumType>
__device__ NumType M_Abs(const NumType inVal){
    return 0 < inVal ? inVal : -inVal;
}

template <typename NumType>
__global__ void InitParticles(const Point3D<NumType> inBoxWidth,
                              const Point3D<NumType> inCellWidth,
                              ParticlesContainer<NumType> inOutParticles,
                              const NumType espilon,
                              curandState_t* particleCurandStates){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t curandState = particleCurandStates[uniqueIdx];

    for(int idxPart = uniqueIdx; idxPart < inOutParticles.nbParticles ; idxPart += nbThreads){
        inOutParticles.x[idxPart] = max(NumType(0), (curand_uniform(&curandState)-espilon)* inBoxWidth.x);
        assert(0 <= inOutParticles.x[idxPart] && inOutParticles.x[idxPart] < inBoxWidth.x);
        inOutParticles.y[idxPart] = max(NumType(0), (curand_uniform(&curandState)-espilon)* inBoxWidth.y);
        assert(0 <= inOutParticles.y[idxPart] && inOutParticles.y[idxPart] < inBoxWidth.y);
        inOutParticles.z[idxPart] = max(NumType(0), (curand_uniform(&curandState)-espilon)* inBoxWidth.z);
        assert(0 <= inOutParticles.z[idxPart] && inOutParticles.z[idxPart] < inBoxWidth.z);
        inOutParticles.index[idxPart] = idxPart;
        inOutParticles.v[idxPart] = -1;
    }
    particleCurandStates[uniqueIdx] = curandState;
}

template <typename NumType>
__global__ void InitParticlesSphere(
                              ParticlesContainer<NumType> inOutParticles,
                              const NumType espilon,
                              curandState_t* particleCurandStates){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t curandState = particleCurandStates[uniqueIdx];

    for(int idxPart = uniqueIdx; idxPart < inOutParticles.nbParticles ; idxPart += nbThreads){
        NumType theta = 2.0 * M_PI * curand_uniform(&curandState); // Random angle between 0 and 2π
        NumType phi = acos(1.0 - 2.0 * curand_uniform(&curandState)); // Random angle between 0 and π

        inOutParticles.x[idxPart] = min(NumType(2 - 1e-6f ), (sin(phi) * cos(theta) + 1));
        assert(0 <= inOutParticles.x[idxPart] && inOutParticles.x[idxPart] < 2);
        inOutParticles.y[idxPart] = min(NumType(2 - 1e-6f ), (sin(phi) * sin(theta) + 1));
        assert(0 <= inOutParticles.y[idxPart] && inOutParticles.y[idxPart] < 2);
        inOutParticles.z[idxPart] = min(NumType(2 - 1e-6f ), (cos(phi) + 1));
        assert(0 <= inOutParticles.z[idxPart] && inOutParticles.z[idxPart] < 2);
        inOutParticles.index[idxPart] = idxPart;
        inOutParticles.v[idxPart] = -1;
    }
    particleCurandStates[uniqueIdx] = curandState;
}


template <typename NumType>
__global__ void PrintParticles(const ParticlesContainer<NumType> inParticles){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int idxPart = uniqueIdx; idxPart < inParticles.nbParticles ; idxPart += nbThreads){
        printf("Particle %d: %e %e %e\n", idxPart, inParticles.x[idxPart], inParticles.y[idxPart], inParticles.z[idxPart]);
    }
}

template <typename NumType>
__global__ void CheckEqual(const ParticlesContainer<NumType> inParticles1,
                           const ParticlesContainer<NumType> inParticles2){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(inParticles1.nbParticles == inParticles2.nbParticles);

    for(int idxPart = uniqueIdx; idxPart < inParticles1.nbParticles ; idxPart += nbThreads){
        assert(inParticles1.x[idxPart] == inParticles2.x[idxPart]);
        assert(inParticles1.y[idxPart] == inParticles2.y[idxPart]);
        assert(inParticles1.z[idxPart] == inParticles2.z[idxPart]);
        assert(inParticles1.index[idxPart] == inParticles2.index[idxPart]);
        const NumType diff = (inParticles1.v[idxPart] == 0 ? fabs(inParticles2.v[idxPart]) : fabs((inParticles1.v[idxPart]-inParticles2.v[idxPart])/inParticles1.v[idxPart]));
        if(!(diff <= 1e-5)){
            printf("Error: %d %e %e => %e\n", idxPart, inParticles1.v[idxPart], inParticles2.v[idxPart], diff);
        }
        assert(diff <= 1e-5);
    }
}


template <typename NumType>
__global__ void ComputeNbInteractions(const Point3D<NumType> inCellWidth,
                                           const Point3D<int> inGridDim,
                                           ParticlesContainer<NumType> inOutParticles,
                                           const CellDescriptor* inCells,
                                           unsigned long long int* inOutNbInteractions){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long int nbInteractions = 0;

    for(int idxPart = uniqueIdx ; idxPart < inOutParticles.nbParticles ; idxPart += nbThreads){
        NumType potential = 0;
        Point3D<NumType> partPos;
        partPos.x = inOutParticles.x[idxPart];
        partPos.y = inOutParticles.y[idxPart];
        partPos.z = inOutParticles.z[idxPart];
        // Compute new cell coord
        const Point3D<int> coord = PosToCellCoord(inCellWidth, partPos);

        assert(coord.x < inGridDim.x && coord.y < inGridDim.y && coord.z < inGridDim.z);

        const int cellIdx = CellCoordToIndex(inGridDim, coord);
        {// Current cell
            const CellDescriptor currentCell = inCells[cellIdx];
            nbInteractions += currentCell.nbParticleInCell-1;
        }

        for(int idxCellX = M_Max(0, coord.x - 1) ; idxCellX <= M_Min(inGridDim.x-1, coord.x + 1) ; ++idxCellX){
            for(int idxCellY = M_Max(0, coord.y - 1) ; idxCellY <= M_Min(inGridDim.y-1, coord.y + 1) ; ++idxCellY){
                for(int idxCellZ = M_Max(0, coord.z - 1) ; idxCellZ <= M_Min(inGridDim.z-1, coord.z + 1) ; ++idxCellZ){
                    if(idxCellX != coord.x || idxCellY != coord.y || idxCellZ != coord.z){
                        const Point3D<int> otherCellCoord{idxCellX, idxCellY, idxCellZ};
                        const int otherCellIdx = CellCoordToIndex(inGridDim, otherCellCoord);
                        const CellDescriptor otherCell = inCells[otherCellIdx];
                        nbInteractions += otherCell.nbParticleInCell;
                    }
                }
            }
        }
    }
    atomicAdd(inOutNbInteractions, nbInteractions);
}

template <typename NumType>
__global__ void ComputeParticleInterationsParPartNoLoop(const Point3D<NumType> inCellWidth,
                                           const Point3D<int> inGridDim,
                                           ParticlesContainer<NumType> inOutParticles,
                                           const CellDescriptor* inCells){
    const int nbThreads = blockDim.x * gridDim.x;
    assert(inOutParticles.nbParticles <= nbThreads);
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(uniqueIdx < inOutParticles.nbParticles){
        NumType potential = 0;
        Point3D<NumType> partPos;
        partPos.x = inOutParticles.x[uniqueIdx];
        partPos.y = inOutParticles.y[uniqueIdx];
        partPos.z = inOutParticles.z[uniqueIdx];
        // Compute new cell coord
        Point3D<int> coord = PosToCellCoord(inCellWidth, partPos);

        assert(coord.x < inGridDim.x && coord.y < inGridDim.y && coord.z < inGridDim.z);

        for(int idxZ = M_Max(0, coord.z-1) ; idxZ <= M_Min(inGridDim.z-1, coord.z+1) ; ++idxZ){
            for(int idxY = M_Max(0, coord.y-1) ; idxY <= M_Min(inGridDim.y-1, coord.y+1) ; ++idxY){
                for(int idxX = M_Max(0, coord.x-1) ; idxX <= M_Min(inGridDim.x-1, coord.x+1) ; ++idxX){
                    const int cellIdx = CellCoordToIndex(inGridDim, Point3D<int>{idxX, idxY, idxZ});
                    int startIdx = inCells[cellIdx].offsetInContainer;
                    int endIdx = inCells[cellIdx].offsetInContainer + inCells[cellIdx].nbParticleInCell;
                    for(int idxOtherPart = startIdx ; idxOtherPart < endIdx ; ++idxOtherPart){
                        if(idxOtherPart != uniqueIdx){
                            Point3D<NumType> otherPos;
                            otherPos.x = inOutParticles.x[idxOtherPart];
                            otherPos.y = inOutParticles.y[idxOtherPart];
                            otherPos.z = inOutParticles.z[idxOtherPart];
                            // Compute with other
                            potential += lennardJonesPotential<NumType>(partPos, otherPos, 1.0, 1.0);
                        }
                    }
                }
            }
        }

        inOutParticles.v[uniqueIdx] = potential;
    }    
}



template <typename NumType>
__global__ void ComputeNbParticlePerCells(const Point3D<NumType> inCellWidth,
                                        const Point3D<int> inGridDim,
                                        const ParticlesContainer<NumType> inParticles,
                                        int* inOutParticlePerCells){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int currentCellId = -1;
    int currentCellAddition = 0;

    for(int idxPart = uniqueIdx ; idxPart < inParticles.nbParticles ; idxPart += nbThreads){
        // Compute new cell coord
        const Point3D<int> coord = PosToCellCoord(inCellWidth, inParticles.x[idxPart],
                                                       inParticles.y[idxPart], inParticles.z[idxPart]);

        assert(coord.x < inGridDim.x && coord.y < inGridDim.y && coord.z < inGridDim.z);

        // Current cell unique index
        const int cellIdx = CellCoordToIndex(inGridDim, coord);
        assert(cellIdx < inGridDim.x*inGridDim.y*inGridDim.z);

        // If the current particle is not in the same cell, update the previous cell counter
        if(cellIdx != currentCellId){
            if(currentCellId != -1){
                const int old = atomicAdd(&inOutParticlePerCells[currentCellId], currentCellAddition) + currentCellAddition;
            }
            // Reset the counters
            currentCellId = cellIdx;
            currentCellAddition = 0;
        }
        currentCellAddition += 1;
    }

    // Save the remaining
    if(currentCellId != -1){
        const int old = atomicAdd(&inOutParticlePerCells[currentCellId], currentCellAddition) + currentCellAddition;
    }
}

template <typename NumType>
void CheckEqualCpu(const ParticlesContainer<NumType>& inParticles1,
                   const ParticlesContainer<NumType>& inParticles2){
    assert(inParticles1.nbParticles == inParticles2.nbParticles);

    // Move from gpu to cpu
    std::vector<NumType> inParticles1_x(inParticles1.nbParticles);
    std::vector<NumType> inParticles1_y(inParticles1.nbParticles);
    std::vector<NumType> inParticles1_z(inParticles1.nbParticles);
    std::vector<int> inParticles1_index(inParticles1.nbParticles);
    std::vector<NumType> inParticles1_v(inParticles1.nbParticles);
    CUDA_ASSERT( cudaMemcpy(inParticles1_x.data(), inParticles1.x, inParticles1.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles1_y.data(), inParticles1.y, inParticles1.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles1_z.data(), inParticles1.z, inParticles1.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles1_index.data(), inParticles1.index, inParticles1.nbParticles*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles1_v.data(), inParticles1.v, inParticles1.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );

    std::vector<NumType> inParticles2_x(inParticles2.nbParticles);
    std::vector<NumType> inParticles2_y(inParticles2.nbParticles);
    std::vector<NumType> inParticles2_z(inParticles2.nbParticles);
    std::vector<int> inParticles2_index(inParticles2.nbParticles);
    std::vector<NumType> inParticles2_v(inParticles2.nbParticles);
    CUDA_ASSERT( cudaMemcpy(inParticles2_x.data(), inParticles2.x, inParticles2.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles2_y.data(), inParticles2.y, inParticles2.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles2_z.data(), inParticles2.z, inParticles2.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles2_index.data(), inParticles2.index, inParticles2.nbParticles*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(inParticles2_v.data(), inParticles2.v, inParticles2.nbParticles*sizeof(NumType), cudaMemcpyDeviceToHost) );


    for(int idxPart = 0 ; idxPart < inParticles1.nbParticles ; ++idxPart){
        assert(inParticles1_x[idxPart] == inParticles2_x[idxPart]);
        assert(inParticles1_y[idxPart] == inParticles2_y[idxPart]);
        assert(inParticles1_z[idxPart] == inParticles2_z[idxPart]);
        assert(inParticles1_index[idxPart] == inParticles2_index[idxPart]);
        const NumType diff = (inParticles1_v[idxPart] == 0 ? fabs(inParticles2_v[idxPart]) : fabs((inParticles1_v[idxPart]-inParticles2_v[idxPart])/inParticles1_v[idxPart]));
        if(!(diff <= 1e-5)){
            printf("Error: %d %e %e => %e\n", idxPart, inParticles1_v[idxPart], inParticles2_v[idxPart], diff);
        }
        assert(diff <= 1e-5);
    }
}


template <typename NumType>
__global__ void InitNewCells(const int* inPartialPrefixParCell,
                             CellDescriptor* inOutCells,
                             const int inNbCells){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int idxCell = uniqueIdx; idxCell < inNbCells ; idxCell += nbThreads){
        inOutCells[idxCell].nbParticleInCell = inPartialPrefixParCell[idxCell+1]-inPartialPrefixParCell[idxCell];
        inOutCells[idxCell].offsetInContainer = inPartialPrefixParCell[idxCell];
    }
}

template <typename NumType>
__global__ void MoveParticlesToNewCells(const Point3D<NumType> inCellWidth,
                                    const Point3D<int> inGridDim,
                                    const ParticlesContainer<NumType> inParticles,
                                     ParticlesContainer<NumType> outParticles,
                                     int* inTicketForCells,
                                     const CellDescriptor* inCells){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int idxPart = uniqueIdx ; idxPart < inParticles.nbParticles ; idxPart += nbThreads){
        // Compute new cell coord
        const Point3D<int> coord = PosToCellCoord(inCellWidth, inParticles.x[idxPart],
                                                       inParticles.y[idxPart], inParticles.z[idxPart]);

        assert(coord.x < inGridDim.x && coord.y < inGridDim.y && coord.z < inGridDim.z);

        // Compute new cell index
        const int cellIdx = CellCoordToIndex(inGridDim, coord);
        assert(cellIdx < inGridDim.x*inGridDim.y*inGridDim.z);
        // Get the position to store the current particle
        const int ticket = atomicAdd(&inTicketForCells[cellIdx], 1);
        assert(ticket < inCells[cellIdx].nbParticleInCell);
        const int storeIdx = ticket + inCells[cellIdx].offsetInContainer;
        // Store the particle
        outParticles.x[storeIdx] = inParticles.x[idxPart];
        outParticles.y[storeIdx] = inParticles.y[idxPart];
        outParticles.z[storeIdx] = inParticles.z[idxPart];
        outParticles.index[storeIdx] = inParticles.index[idxPart];
    }
}


template <typename NumType>
auto AllocateParticles(const int inNbParticles, std::list<void*>& cuPtrToDelete){
    ParticlesContainer<NumType> particles;
    CUDA_ASSERT( cudaMalloc(&particles.x, inNbParticles*sizeof(NumType)) );
    cuPtrToDelete.push_back(particles.x);
    CUDA_ASSERT( cudaMalloc(&particles.y, inNbParticles*sizeof(NumType)) );
    cuPtrToDelete.push_back(particles.y);
    CUDA_ASSERT( cudaMalloc(&particles.z, inNbParticles*sizeof(NumType)) );
    cuPtrToDelete.push_back(particles.z);
    CUDA_ASSERT( cudaMalloc(&particles.index, inNbParticles*sizeof(int)) );
    cuPtrToDelete.push_back(particles.index);
    CUDA_ASSERT( cudaMalloc(&particles.v, inNbParticles*sizeof(NumType)) );
    cuPtrToDelete.push_back(particles.v);
    particles.nbParticles = inNbParticles;
    return particles;
}

void deletePtrs(std::list<void*>& inPtrToDelete){
    for(void* ptr : inPtrToDelete){
        CUDA_ASSERT( cudaFree(ptr) );
    }
    // For the print
    CUDA_ASSERT( cudaDeviceSynchronize());
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


template <typename NumType = float>
auto executeSimulation(const int inNbParticles, const int inNbLoops, 
                        const NumType boxWidth1Dim, const NumType inCutoff, 
                        const bool inCheckResult, const bool gensurface){
    const int DefaultNbThreads = 128;
    const int DefaultNbBlocks  = std::max(int((inNbParticles+DefaultNbThreads-1)/DefaultNbThreads), 1);

    // Keep ptr to delete
    std::list<void*> cuPtrToDelete;
    // Configuration
    const Point3D<NumType> origin{0, 0, 0};
    const Point3D<NumType> BoxWidth{boxWidth1Dim, boxWidth1Dim, boxWidth1Dim};
    const Point3D<NumType> CellWidth{inCutoff, inCutoff, inCutoff};
    const Point3D<int> GridDim{static_cast<int>(BoxWidth.x/inCutoff),
                                    static_cast<int>(BoxWidth.y/inCutoff),
                                    static_cast<int>(BoxWidth.z/inCutoff)};
    const int NbCells = GridDim.x * GridDim.y * GridDim.z;

    std::cout << "Start Execution" << std::endl;
    std::cout << " - gensurface: " << gensurface << std::endl;
    std::cout << " - NbParticles: " << inNbParticles << std::endl;
    std::cout << " - NbLoops: " << inNbLoops << std::endl;
    std::cout << " - Cutoff: " << inCutoff << std::endl;
    std::cout << " - Space box dimension: " << BoxWidth.x << " / " << BoxWidth.y << " / " << BoxWidth.z << std::endl;
    std::cout << " - Space cell dimension: " << CellWidth.x << " / " << CellWidth.y << " / " << CellWidth.z << std::endl;
    std::cout << " - Space grid dimension: " << GridDim.x << " / " << GridDim.y << " / " << GridDim.z << std::endl;
    std::cout << " - Number of cells: " << NbCells << std::endl;
    std::cout << " - Avergage particles per cell: " << double(inNbParticles)/NbCells << std::endl;
    std::cout << " - Number of CUDA blocks: " << DefaultNbBlocks << std::endl;
    std::cout << " - Number of CUDA threads: " << DefaultNbThreads << std::endl;
    const double expectedAveragePartPerCell = inNbParticles/double(GridDim.x*GridDim.y*GridDim.z);
    const long int expectedNbInteractions = expectedAveragePartPerCell*27*inNbParticles*inNbLoops;
    std::cout << " - Expected number of interactions: " << expectedNbInteractions << std::endl;

    // Number of flops per interaction
    const int NbFlopsPerInteraction = 18;
    std::cout << " - Number of flops per interaction: " << NbFlopsPerInteraction << std::endl;

    // Allocate the particles
    ParticlesContainer<NumType> particles     = AllocateParticles<NumType>(inNbParticles, cuPtrToDelete);
    ParticlesContainer<NumType> particlesSwap = AllocateParticles<NumType>(inNbParticles, cuPtrToDelete);

    // Random states
    curandState_t* particleCurandStates;
    {
        CUDA_ASSERT( cudaMalloc(&particleCurandStates, DefaultNbBlocks*DefaultNbThreads*sizeof(curandState_t)));
        cuPtrToDelete.push_back(particleCurandStates);

        InitRandStates<<<DefaultNbBlocks, DefaultNbThreads>>>(particleCurandStates, DefaultNbBlocks*DefaultNbThreads);
        CUDA_ASSERT( cudaDeviceSynchronize());
    }

    // The number of particles for each cell
    int* particlePerCells;
    {
        CUDA_ASSERT( cudaMalloc(&particlePerCells, NbCells*sizeof(int)));
        cuPtrToDelete.push_back(particlePerCells);
    }
    // Prefix sum for the particles per cells
    int* prefixParCell;
    {
        CUDA_ASSERT( cudaMalloc(&prefixParCell, (NbCells+1)*sizeof(int)));
        cuPtrToDelete.push_back(prefixParCell);
    }
    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream));
    // Build the cells (nb particles and offset in the container)
    CellDescriptor* cells;
    {
        CUDA_ASSERT( cudaMalloc(&cells, NbCells*sizeof(CellDescriptor)) );
        cuPtrToDelete.push_back(cells);
    }
    // Compute the number of interactions
    unsigned long long int NbInteractions = 0;
    unsigned long long int* cuNbInteractions;
    {
        CUDA_ASSERT( cudaMalloc(&cuNbInteractions, sizeof(unsigned long long int)) );
        cuPtrToDelete.push_back(cuNbInteractions);
    }

    std::vector<ResultFrame::AResult> results;

    SpTimer initTimer;
    SpTimer computeTimer;

    for(int idxLoop = 0 ; idxLoop < inNbLoops ; ++idxLoop){
        // Init particles and set random positions
        if(gensurface==true){
            InitParticlesSphere<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(particles, 
                                                                      std::numeric_limits<NumType>::epsilon(),
                                                                      particleCurandStates);
        }else{
            InitParticles<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(BoxWidth, CellWidth, particles, 
                                                                      std::numeric_limits<NumType>::epsilon(),
                                                                      particleCurandStates);
        }
        CUDA_ASSERT( cudaDeviceSynchronize());

        initTimer.start();
        CUDA_ASSERT( cudaMemset(particlePerCells, 0, NbCells*sizeof(int)) );

        // Find out the number of particles per cells
        ComputeNbParticlePerCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, GridDim, particles, particlePerCells);
        CUDA_ASSERT( cudaDeviceSynchronize());

        // Compute the prefix
        CUDA_ASSERT( cudaMemset(prefixParCell, 0, (NbCells+1)*sizeof(int)) );
        PrefixFullV2(particlePerCells, prefixParCell+1, NbCells, stream);
        CUDA_ASSERT(cudaStreamSynchronize(stream));

        // Init the cells
        CUDA_ASSERT( cudaMemset(cells, 0, NbCells*sizeof(CellDescriptor)) );
        InitNewCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(prefixParCell, cells, NbCells);
        CUDA_ASSERT( cudaDeviceSynchronize());
        
        // Reset the array to reuse it as ticket
        CUDA_ASSERT( cudaMemset(particlePerCells, 0, NbCells*sizeof(int)) );
        MoveParticlesToNewCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, GridDim, particles,
                                                                            particlesSwap, particlePerCells, cells);
        particlesSwap.swap(particles);
        initTimer.stop();

        CUDA_ASSERT( cudaMemset(cuNbInteractions, 0, sizeof(unsigned long long int)) );
        ComputeNbInteractions<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, 
                                                                        GridDim, 
                                                                        particles, 
                                                                        cells,
                                                                        cuNbInteractions);
        CUDA_ASSERT( cudaDeviceSynchronize());
        unsigned long long int newNbInteractions = 0;
        CUDA_ASSERT( cudaMemcpy(&newNbInteractions, cuNbInteractions, sizeof(unsigned long long int), cudaMemcpyDeviceToHost) );
        NbInteractions += newNbInteractions;

        computeTimer.start();
        ComputeParticleInterationsParPartNoLoop<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth,
                                                                GridDim,
                                                                particles,
                                                                cells);
        CUDA_ASSERT( cudaDeviceSynchronize());
        computeTimer.stop();

        results.push_back(ResultFrame::AResult{
            initTimer.getElapsed(),
            computeTimer.getElapsed(),
            (initTimer.getElapsed() + computeTimer.getElapsed())});
    }

    CUDA_ASSERT(cudaStreamDestroy(stream));

    std::cout << " - Nb interactions: " << NbInteractions << std::endl;
    std::cout << " - Time init: " << initTimer.getCumulated() << "s" << std::endl;
    std::cout << " - Time compute: " << computeTimer.getCumulated() << "s" << std::endl;
    std::cout << " - Total: " << (initTimer.getCumulated() + computeTimer.getCumulated()) << "s" << std::endl;

    std::cout << " - Per interactions: " << (initTimer.getCumulated() + computeTimer.getCumulated())/NbInteractions << std::endl;
    std::cout << " - GFlops/s: " << (NbInteractions*NbFlopsPerInteraction)/(initTimer.getCumulated() + computeTimer.getCumulated())/1e9 << std::endl;
    std::cout << "     - Compute only GFlops/s: " << (NbInteractions*NbFlopsPerInteraction)/computeTimer.getCumulated()/1e9 << std::endl;
    std::cout << " - Interactions/s: " << NbInteractions/(initTimer.getCumulated() + computeTimer.getCumulated()) << std::endl;

    deletePtrs(cuPtrToDelete);

    return std::make_tuple(NbInteractions, std::move(results));
}


#include <chrono>
#include <ctime>

auto getFilename(const bool gensurface){
    // Get the current date and time
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Format the date and time
    std::tm* ptm = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", ptm);

    // Create the filename with the date and time
    std::string filename = "results"
                             + std::string(gensurface ? "-surface" : "")
                             + "-" + std::string(buffer) + ".csv";
    return filename;
}

#include <fstream>

int main(int argc, char** argv){
    // nvcc -gencode arch=compute_75,code=sm_75 cutrix/testParticlesPaper.cu -o test.exe --std=c++17 -O3 --ptxas-options=-v
    // nvcc -gencode arch=compute_75,code=sm_75 cutrix/testParticlesPaper.cu -o test.exe --std=c++17 -O0 -g --ptxas-options=-v -lgomp -lineinfo --generate-line-info
    //executeSimulation<double>(8, 1, 1./2., true);
    //return 0;

    bool gensurface = false;
    if( argc == 2 &&  (std::string(argv[1]) == "--gensurface" || std::string(argv[1]) == "-gs" ))
    {
        gensurface = true;
    }

    std::vector<ResultFrame> allResults;

    using NumType = float;
    if(gensurface){
        const int NbLoops = 5;// put 200
        const int MaxParticlesPerCell = 64;
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

                const float boxWidth = std::ceil(2.0 / sphereRadius) * sphereRadius;
                const int gridDim = boxWidth/sphereRadius;
                const float cellWidth = boxWidth/gridDim;

                auto [nbInteractions, results] = executeSimulation<NumType>(nbParticles, NbLoops, 
                                            boxWidth, cellWidth, false, gensurface);

                ResultFrame frame{nbParticles, nbInteractions, NbLoops, boxDiv, std::move(results)};

                allResults.emplace_back(std::move(frame));
            }
        }
    }
    else{
        const NumType BoxWidth = 1.0;
        const int NbLoops = 5;// put 200
        const int MaxParticlesPerCell = 32;
        const int MaxBoxDiv = 32;// put 32
        for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
            const NumType cellWidth = BoxWidth/boxDiv;
            const int nbBoxes = boxDiv*boxDiv*boxDiv;
            for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 2){
                std::cout << "NbParticles: " << nbParticles << std::endl;
                std::cout << "BoxDiv: " << boxDiv << std::endl;
                std::cout << "CellWidth: " << cellWidth << std::endl;
                std::cout << "NbLoops: " << NbLoops << std::endl;

                auto [nbInteractions, results] = executeSimulation<NumType>(nbParticles, NbLoops, 
                                                    1, cellWidth, false, gensurface);

                ResultFrame frame{nbParticles, nbInteractions, NbLoops, boxDiv, std::move(results)};

                allResults.emplace_back(std::move(frame));
            }
        }
    }

    // We will print the results in a csv file
    // the first columns will contains nbParticles, nbInteractions, NbLoops, boxDiv
    // followed by the number of cells (boxDiv^3)
    // then we put the results for each method using the vecMethodName-[time, gflops, interactionsPerSecond]

    // Open the file
    std::ofstream file(getFilename(gensurface));
    {
        file << "NbParticles,NbInteractions,NbLoops,boxDiv,nbCells,partspercell,interactionsperparticle,timeinit,timecompute,timetotal";
        file << std::endl;
    }
    for(const ResultFrame& frame : allResults){
        for(const ResultFrame::AResult& res : frame.results){
            file << frame.nbParticles << "," << frame.nbInteractions << "," << frame.nbLoops << "," << frame.boxDiv << "," << frame.boxDiv*frame.boxDiv*frame.boxDiv;
            file << "," << double(frame.nbParticles)/(frame.boxDiv*frame.boxDiv*frame.boxDiv) << "," << double(frame.nbInteractions)/frame.nbParticles;
            file << "," << res.timeInit  << "," << res.timeCompute << "," << res.timeTotal;
            file << std::endl;
        }
    }

	return 0;
}


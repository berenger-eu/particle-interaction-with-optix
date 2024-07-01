#include <iostream>
#include <list>
#include <vector>
#include <iostream>


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
                              curandState_t* particleCurandStates){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t curandState = particleCurandStates[uniqueIdx];

    for(int idxPart = uniqueIdx; idxPart < inOutParticles.nbParticles ; idxPart += nbThreads){
        inOutParticles.x[idxPart] = curand_uniform(&curandState)* inBoxWidth.x;
        assert(0 <= inOutParticles.x[idxPart] && inOutParticles.x[idxPart] < inBoxWidth.x);
        inOutParticles.y[idxPart] = curand_uniform(&curandState)* inBoxWidth.y;
        assert(0 <= inOutParticles.y[idxPart] && inOutParticles.y[idxPart] < inBoxWidth.y);
        inOutParticles.z[idxPart] = curand_uniform(&curandState)* inBoxWidth.z;
        assert(0 <= inOutParticles.z[idxPart] && inOutParticles.z[idxPart] < inBoxWidth.z);
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
                                        int* inOutParticlePerCells,
                                        int* inOutMaxNbPartPerCell){
    const int nbThreads = blockDim.x * gridDim.x;
    const int uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int currentCellId = -1;
    int currentCellAddition = 0;
    int maxNbPartPerCell = 0;

    __shared__ int sharedMaxNbPartPerCell;
    if(threadIdx.x == 0){
        sharedMaxNbPartPerCell = 0;
    }
    __syncthreads();

    for(int idxPart = uniqueIdx ; idxPart < inParticles.nbParticles ; idxPart += nbThreads){
        // Compute new cell coord
        const Point3D<int> coord = PosToCellCoord(inCellWidth, inParticles.x[idxPart],
                                                       inParticles.y[idxPart], inParticles.z[idxPart]);
        // Current cell unique index
        const int cellIdx = CellCoordToIndex(inGridDim, coord);
        assert(cellIdx < inGridDim.x*inGridDim.y*inGridDim.z);

        // If the current particle is not in the same cell, update the previous cell counter
        if(cellIdx != currentCellId){
            if(currentCellId != -1){
                const int old = atomicAdd(&inOutParticlePerCells[currentCellId], currentCellAddition) + currentCellAddition;
                maxNbPartPerCell = M_Max(maxNbPartPerCell, old);
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
        maxNbPartPerCell = M_Max(maxNbPartPerCell, old);
    }

    // Save the max
    atomicMax(&sharedMaxNbPartPerCell, maxNbPartPerCell);
    __syncthreads();
    if(threadIdx.x == 0){
        atomicMax(inOutMaxNbPartPerCell, sharedMaxNbPartPerCell);
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
        double time;
        double gflops;
        double interactionsPerSecond;
    };
    struct V2Stats{
        int nbBlocks;
        int nbThreads;
        int smSize;
    };

    int nbParticles;
    long int nbInteractions;
    int nbLoops;
    int boxDiv;
    std::vector<AResult> results;
};


template <typename NumType = float>
auto executeSimulation(const int inNbParticles, const int inNbLoops, const NumType inCutoff, const bool inCheckResult){
    const int DefaultNbThreads = 128;
    const int DefaultNbBlocks  = std::max(int((inNbParticles+DefaultNbThreads-1)/DefaultNbThreads), 1);

    // Keep ptr to delete
    std::list<void*> cuPtrToDelete;
    // Configuration
    const Point3D<NumType> origin{0, 0, 0};
    const Point3D<NumType> BoxWidth{1, 1, 1};
    const Point3D<NumType> CellWidth{inCutoff, inCutoff, inCutoff};
    const Point3D<int> GridDim{static_cast<int>(BoxWidth.x/inCutoff),
                                    static_cast<int>(BoxWidth.y/inCutoff),
                                    static_cast<int>(BoxWidth.z/inCutoff)};
    const int NbCells = GridDim.x * GridDim.y * GridDim.z;

    std::cout << "Start Execution" << std::endl;
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

    // Allocate the particles
    ParticlesContainer<NumType> particles     = AllocateParticles<NumType>(inNbParticles, cuPtrToDelete);

    // Random states
    {
        curandState_t* particleCurandStates;
        CUDA_ASSERT( cudaMalloc(&particleCurandStates, DefaultNbBlocks*DefaultNbThreads*sizeof(curandState_t)));
        InitRandStates<<<DefaultNbBlocks, DefaultNbThreads>>>(particleCurandStates, DefaultNbBlocks*DefaultNbThreads);
        CUDA_ASSERT( cudaDeviceSynchronize());

        // Init particles and set random positions
        InitParticles<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(BoxWidth, CellWidth, particles, particleCurandStates);
        CUDA_ASSERT( cudaDeviceSynchronize());
        CUDA_ASSERT( cudaFree(particleCurandStates) );
    }

    // The number of particles for each cell
    int* particlePerCells;
    {
        CUDA_ASSERT( cudaMalloc(&particlePerCells, NbCells*sizeof(int)));
        cuPtrToDelete.push_back(particlePerCells);
        CUDA_ASSERT( cudaMemset(particlePerCells, 0, NbCells*sizeof(int)) );
    }

    // Find out the number of particles per cells
    int maxNbPartPerCell = -1;
    {
        int* cuMaxNbPartPerCell;
        CUDA_ASSERT( cudaMalloc(&cuMaxNbPartPerCell, sizeof(int)) );
        CUDA_ASSERT( cudaMemset(cuMaxNbPartPerCell, 0, sizeof(int)) );
        ComputeNbParticlePerCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, GridDim, particles, particlePerCells, cuMaxNbPartPerCell);
        CUDA_ASSERT( cudaDeviceSynchronize());

        CUDA_ASSERT( cudaMemcpy(&maxNbPartPerCell, cuMaxNbPartPerCell, sizeof(int), cudaMemcpyDeviceToHost) );

        std::cout << " - Max particles per cell: " << maxNbPartPerCell << std::endl;
    }

    // Prefix sum for the particles per cells
    int* prefixParCell;
    {
        CUDA_ASSERT( cudaMalloc(&prefixParCell, (NbCells+1)*sizeof(int)));
        CUDA_ASSERT( cudaMemset(prefixParCell, 0, (NbCells+1)*sizeof(int)) );
        cuPtrToDelete.push_back(prefixParCell);

        // Compute the prefix
        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        PrefixFullV2(particlePerCells, prefixParCell+1, NbCells, stream);
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        CUDA_ASSERT(cudaStreamDestroy(stream));
    }

    // Build the cells (nb particles and offset in the container)
    CellDescriptor* cells;
    {
        CUDA_ASSERT( cudaMalloc(&cells, NbCells*sizeof(CellDescriptor)) );
        CUDA_ASSERT( cudaMemset(cells, 0, NbCells*sizeof(CellDescriptor)) );
        cuPtrToDelete.push_back(cells);

        // Init the cells
        InitNewCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(prefixParCell, cells, NbCells);
        CUDA_ASSERT( cudaDeviceSynchronize());
    }
    // Move particles to new cells (reorder the particles)
    {
        // Reset the array to reuse it as ticket
        CUDA_ASSERT( cudaMemset(particlePerCells, 0, NbCells*sizeof(int)) );

        std::list<void*> cuPtrToDeleteSwap;
        ParticlesContainer<NumType> particlesSwap = AllocateParticles<NumType>(inNbParticles, cuPtrToDeleteSwap);
        MoveParticlesToNewCells<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, GridDim, particles,
                                                                            particlesSwap, particlePerCells, cells);
        CUDA_ASSERT( cudaDeviceSynchronize());
        particles.copy(particlesSwap);
        for(void* ptrToFree : cuPtrToDeleteSwap){
            CUDA_ASSERT( cudaFree(ptrToFree) );
        }
    }

    // Number of flops per interaction
    const int NbFlopsPerInteraction = 18;
    std::cout << " - NbFlopsPerInteraction: " << NbFlopsPerInteraction << std::endl;

    // Compute the number of interactions
    unsigned long long int NbInteractions = 0;
    {
        unsigned long long int* cuNbInteractions;
        CUDA_ASSERT( cudaMalloc(&cuNbInteractions, sizeof(unsigned long long int)) );
        CUDA_ASSERT( cudaMemset(cuNbInteractions, 0, sizeof(unsigned long long int)) );
        ComputeNbInteractions<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth, 
                                                                        GridDim, 
                                                                        particles, 
                                                                        cells,
                                                                        cuNbInteractions);
        CUDA_ASSERT( cudaDeviceSynchronize());

        CUDA_ASSERT( cudaMemcpy(&NbInteractions, cuNbInteractions, sizeof(unsigned long long int), cudaMemcpyDeviceToHost) );

        std::cout << " - NbInteractions: " << NbInteractions << std::endl;
    }


    std::vector<ResultFrame::AResult> results;

    /////////////////////////////////////////////////////////////////////////////
    {
        std::cout << "# Par-Part-NoLoop:" << std::endl;
        
        std::list<void*> localCuPtrToDelete;
        ParticlesContainer<NumType> particles_original = AllocateParticles<NumType>(inNbParticles, localCuPtrToDelete);
        particles_original.copy(particles);
        CUDA_ASSERT( cudaDeviceSynchronize());
        SpTimer timer;
        for(int idxLoop = 0 ; idxLoop < inNbLoops ; ++idxLoop){
            ComputeParticleInterationsParPartNoLoop<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(CellWidth,
                                                                            GridDim,
                                                                            particles_original,
                                                                            cells);
        }
        CUDA_ASSERT( cudaDeviceSynchronize());
        timer.stop();
        std::cout << " - Time: " << timer.getElapsed() << "s" << std::endl;
        std::cout << " - Per iterations: " << timer.getElapsed()/inNbLoops << "s" << std::endl;
        std::cout << " - Per interations: " << timer.getElapsed()/(inNbLoops*NbInteractions) << "s" << std::endl;
        std::cout << " - GFlops/s: " << (inNbLoops*NbInteractions*NbFlopsPerInteraction)/timer.getElapsed()/1e9 << std::endl;

        if(inCheckResult){
            std::cout << "Check particles_original" << std::endl;
            CheckEqualCpu<NumType>(particles, particles_original);
            CheckEqual<NumType><<<DefaultNbBlocks, DefaultNbThreads>>>(particles, particles_original);
            CUDA_ASSERT( cudaDeviceSynchronize());
        }
        deletePtrs(localCuPtrToDelete);

        results.push_back(ResultFrame::AResult{timer.getElapsed()/inNbLoops, 
                (inNbLoops*NbInteractions*NbFlopsPerInteraction)/timer.getElapsed()/1e9, 
                NbInteractions/timer.getElapsed()});
    }
    /////////////////////////////////////////////////////////////////////////////

    deletePtrs(cuPtrToDelete);

    return std::make_tuple(NbInteractions, std::move(results));
}


#include <chrono>
#include <ctime>

auto getFilename(){
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

#include <fstream>

int main(){
    // nvcc -gencode arch=compute_75,code=sm_75 cutrix/testParticlesPaper.cu -o test.exe --std=c++17 -O3 --ptxas-options=-v
    // nvcc -gencode arch=compute_75,code=sm_75 cutrix/testParticlesPaper.cu -o test.exe --std=c++17 -O0 -g --ptxas-options=-v -lgomp -lineinfo --generate-line-info
    //executeSimulation<double>(8, 1, 1./2., true);
    //return 0;

    std::vector<ResultFrame> allResults;

    using NumType = float;
    const NumType BoxWidth = 1.0;
    const int NbLoops = 200;// put 200
    const int MaxParticlesPerCell = 128;
    const int MaxBoxDiv = 32;// put 32
    for(int boxDiv = 2 ; boxDiv <= MaxBoxDiv ; boxDiv *= 2){
        const NumType cellWidth = BoxWidth/boxDiv;
        const int nbBoxes = boxDiv*boxDiv*boxDiv;
        for(int nbParticles = nbBoxes ; nbParticles <= nbBoxes*MaxParticlesPerCell ; nbParticles *= 10){
            std::cout << "NbParticles: " << nbParticles << std::endl;
            std::cout << "BoxDiv: " << boxDiv << std::endl;
            std::cout << "CellWidth: " << cellWidth << std::endl;
            std::cout << "NbLoops: " << NbLoops << std::endl;

            auto [nbInteractions, results] = executeSimulation<NumType>(nbParticles, NbLoops, cellWidth, false);

            ResultFrame frame{nbParticles, nbInteractions, NbLoops, boxDiv, std::move(results)};

            allResults.emplace_back(std::move(frame));
        }
    }


    // We will print the results in a csv file
    // the first columns will contains nbParticles, nbInteractions, NbLoops, boxDiv
    // followed by the number of cells (boxDiv^3)
    // then we put the results for each method using the vecMethodName-[time, gflops, interactionsPerSecond]

    // Open the file
    std::ofstream file(getFilename());
    {
        file << "NbParticles,NbInteractions,NbLoops,boxDiv,nbCells,partspercell,interactionsperparticle,Par-Part-NoLoop-time,Par-Part-NoLoop-gflops,Par-Part-NoLoop-interactionsPerSecond";
        file << std::endl;
    }
    for(const ResultFrame& frame : allResults){
        file << frame.nbParticles << "," << frame.nbInteractions << "," << frame.nbLoops << "," << frame.boxDiv << "," << frame.boxDiv*frame.boxDiv*frame.boxDiv;
        file << "," << double(frame.nbParticles)/(frame.boxDiv*frame.boxDiv*frame.boxDiv) << "," << double(frame.nbInteractions)/frame.nbParticles;
        for(const ResultFrame::AResult& res : frame.results){
            if(res.time != 0){
                file << "," << res.time << "," << res.gflops << "," << res.interactionsPerSecond;
            }
            else{
                file << "," << "" << "," << "" << "," << "";   
            }
        }
        file << std::endl;
    }

	return 0;
}


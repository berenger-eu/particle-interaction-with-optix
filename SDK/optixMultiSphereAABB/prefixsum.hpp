#ifndef PREFIXSUM_HPP
#define PREFIXSUM_HPP

#include <cstring>
#include <cmath>
#include <cassert>
#include <numeric>
#include <cstdio>

#include <cuda_runtime.h>

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

template <typename IndexType>
__device__ __host__ /*constexpr*/ int Log2OfPowerOfTwo(const IndexType inPowerOfTwo){
#ifdef __CUDA_ARCH__
    return __ffs(static_cast<size_t>(inPowerOfTwo)) - 1;
#else
    return ffs(static_cast<size_t>(inPowerOfTwo)) - 1;
#endif
}

template <typename IndexType>
__device__ __host__ constexpr int MyCuMin(const IndexType& inVal1, const IndexType& inVal2){
    return (inVal1 < inVal2 ? inVal1 : inVal2);
}

template <typename IndexType>
__device__ __host__ constexpr int MyCuMax(const IndexType& inVal1, const IndexType& inVal2){
    return (inVal1 > inVal2 ? inVal1 : inVal2);
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template <typename NumType, typename IndexType, const int SharedMemSize, const bool isFixedSize>
__global__ void BuildPartialPrefix_AllSize( const NumType* inValues,
                                           NumType* inOutPartialReduction,
                                           const IndexType inLimite,
                                           const IndexType inStepSize){
    static_assert((SharedMemSize != 0) && ((SharedMemSize & (SharedMemSize - 1)) == 0), "must be power of 2");

    const IndexType NbValues = (inLimite+inStepSize-1)/inStepSize;

    if constexpr (isFixedSize){
        assert(SharedMemSize <= NbValues);
        assert(SharedMemSize == blockDim.x*2);
    }
    else{
        assert(NbValues < SharedMemSize);
        assert(0 < NbValues);
    }

    __shared__ NumType prefixTree[SharedMemSize];

    [[maybe_unused]] const IndexType idxWindow = blockIdx.x*SharedMemSize;

    // We start by copying SharedMemSize values
    // if isFixedSize, we know there are perfectly SharedMemSize values to copy
    // and SharedMemSize/2 threads
    // else there are NbValues+1/2 threads and we copy only NbValues values

    if constexpr (isFixedSize){
        prefixTree[threadIdx.x]   = inValues[MyCuMin(inLimite-1u, (idxWindow+threadIdx.x+1)*inStepSize-1)];
        prefixTree[threadIdx.x+blockDim.x] = inValues[MyCuMin(inLimite-1u, (idxWindow+threadIdx.x+blockDim.x+1)*inStepSize-1)];
        __syncthreads();
    }
    else{
        for (int idxVal = threadIdx.x+NbValues ; idxVal < SharedMemSize; idxVal += blockDim.x){
            prefixTree[idxVal] = 0;
        }

        prefixTree[threadIdx.x]   = inValues[MyCuMin(inLimite-1u, (threadIdx.x+1)*inStepSize-1)];
        if(threadIdx.x+blockDim.x < NbValues){
            prefixTree[threadIdx.x+blockDim.x] = inValues[MyCuMin(inLimite-1u, (threadIdx.x+blockDim.x+1)*inStepSize-1)];
        }
        __syncthreads();
    }

    // In the upward pass, we consider the values are the leaves
    // and we build the upper levels directly on the nodes

    {
        // Upward pass
        const IndexType TreeHeight = Log2OfPowerOfTwo(SharedMemSize);
        IndexType jumpSize = 2;
        IndexType limite = SharedMemSize;

        for(IndexType idxLevel = TreeHeight-1 ; idxLevel >= 0 ; --idxLevel){
            const IndexType jumpSizeDiv2 = jumpSize/2;
            const IndexType idxNode = threadIdx.x*jumpSize+jumpSize-1;
            if(idxNode < limite){// TODO remove ?
                prefixTree[idxNode] += prefixTree[idxNode-jumpSizeDiv2];
            }
            jumpSize *= 2;
            //limite /= 2;
            __syncthreads();
        }
    }

    {
        // Downward pass
        const IndexType TreeHeight = Log2OfPowerOfTwo(SharedMemSize);
        IndexType jumpSize = (SharedMemSize >> (2-1));
        IndexType limite = 2;

        for(IndexType idxLevel = 2 ; idxLevel <= TreeHeight ; ++idxLevel){
            const IndexType jumpSizeDiv2 = jumpSize/2;
            if(threadIdx.x < limite-1){
                IndexType idxNode = threadIdx.x*jumpSize+jumpSize+jumpSizeDiv2-1;
                prefixTree[idxNode] += prefixTree[idxNode-jumpSizeDiv2];
            }
            jumpSize /= 2;
            limite *= 2;
            __syncthreads();
        }
    }

    // Copy back the same way as in the first step

    if constexpr (isFixedSize){
        // Copy result, we know there are SharedMemSize values for SharedMemSize/2 threads
        inOutPartialReduction[idxWindow+threadIdx.x]   = prefixTree[threadIdx.x];
        inOutPartialReduction[idxWindow+threadIdx.x+blockDim.x] = prefixTree[threadIdx.x+blockDim.x];
        //__syncthreads();
    }
    else {
        // Copy result
        inOutPartialReduction[threadIdx.x]   = prefixTree[threadIdx.x];
        if(threadIdx.x+blockDim.x < NbValues){
            inOutPartialReduction[threadIdx.x+blockDim.x] = prefixTree[threadIdx.x+blockDim.x];
        }
        //__syncthreads();
    }
}

template <typename NumType, typename IndexType>
__global__ void ApplyPartialPrefixV3( const NumType* inPartialPrefixSum,
                                   NumType* inValues,
                                   const IndexType inNbValues,
                                   const IndexType inWindowSize){

    const IndexType idxVal = threadIdx.x+blockDim.x*blockIdx.x;

    if(idxVal < inNbValues){
        inValues[idxVal] += inPartialPrefixSum[idxVal/inWindowSize];
    }
}

template <typename ValueType, typename IndexType, int ShareMemSize = 512*2>
void PrefixFullV2(const ValueType* inValues,
                ValueType* inOutPrefixSum,
                const IndexType inLimite,
                cudaStream_t& inStream,
                  const IndexType inStepSize = 1){
    static_assert(std::is_signed_v<IndexType>, "IndexType must be signed");

    const IndexType NbValues = (inLimite+inStepSize-1)/inStepSize;

    const IndexType NbCompleteBlocks = (NbValues/ShareMemSize);
    if(NbCompleteBlocks){
        const int MaxThreads = ShareMemSize/2;
        const IndexType LimiteCompleteBlocks = std::min(inLimite, NbCompleteBlocks*ShareMemSize*inStepSize);

        BuildPartialPrefix_AllSize<ValueType, IndexType, ShareMemSize, true><<< NbCompleteBlocks, MaxThreads, 0, inStream>>>(inValues,
                                                                                             inOutPrefixSum,
                                                                                             LimiteCompleteBlocks,
                                                                                             inStepSize);
        //CUDA_ASSERT(cudaStreamSynchronize(inStream));
    }

    const bool HasIncompleteBlock = (NbValues%ShareMemSize);
    if(HasIncompleteBlock){
        const IndexType startingValueIdx = NbCompleteBlocks*ShareMemSize;
        const IndexType remainingNbValues = (NbValues-startingValueIdx);
        const IndexType remainingLimite = inLimite-NbCompleteBlocks*ShareMemSize*inStepSize;
        const int MaxThreads = (remainingNbValues+1)/2;
        BuildPartialPrefix_AllSize<ValueType, IndexType, ShareMemSize, false><<< 1, MaxThreads, 0, inStream>>>(inValues + startingValueIdx*inStepSize,
                                                                                                                      inOutPrefixSum + startingValueIdx,
                                                                                                                      remainingLimite,
                                                                                                                      inStepSize);
    }

    if(NbCompleteBlocks > 1 || (NbCompleteBlocks%ShareMemSize != 0 && HasIncompleteBlock)){
        const IndexType NbPrefixValuesToReduce = (NbCompleteBlocks + (HasIncompleteBlock ? 1 : 0));
        ValueType* incompletePrefix;
        CUDA_ASSERT(cudaMallocAsync(&incompletePrefix, sizeof(ValueType)*NbPrefixValuesToReduce, inStream));

        const IndexType LimiteToProceed = NbValues;
        PrefixFullV2<ValueType, IndexType, ShareMemSize>(inOutPrefixSum, incompletePrefix, LimiteToProceed, inStream, ShareMemSize);

        const int MaxThreads = ShareMemSize/2;
        const IndexType NbPrefixValuesToProcess = NbValues-ShareMemSize;
        const int NbThreadBlocks = (NbPrefixValuesToProcess+MaxThreads-1)/MaxThreads;
        ApplyPartialPrefixV3<ValueType, IndexType><<< NbThreadBlocks, MaxThreads, 0, inStream>>>(incompletePrefix,
                                                                                           inOutPrefixSum+ShareMemSize,
                                                                                           NbPrefixValuesToProcess,
                                                                                           ShareMemSize);


        CUDA_ASSERT(cudaFreeAsync(incompletePrefix, inStream));
    }

    CUDA_ASSERT(cudaStreamSynchronize(inStream));
}


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template <class ElementType, typename IndexType>
__device__ void BuildPrefix_SM_device(ElementType* prefix, IndexType N) {
    // Upward pass
    IndexType js = 2;
    while (js <= N) {
        IndexType jsd2 = js / 2;
        for (IndexType idN = threadIdx.x*js + js - 1; idN < N; idN += blockDim.x*js) {
            prefix[idN] += prefix[idN - jsd2];
        }
        js *= 2;
        __syncthreads();
    }

    // Downward pass
    js = max(4, js/2);
    while (js > 1) {
        IndexType jsd2 = js / 2;
        for (IndexType idN = threadIdx.x*js + js + jsd2 - 1; idN < N; idN += blockDim.x*js) {
            prefix[idN] += prefix[idN - jsd2];
        }
        js = jsd2;
        __syncthreads();
    }
}


template <typename NumType, typename IndexType>
__global__ void BuildPrefix_SM( NumType* prefixTree, const IndexType inSize){
    extern __shared__ IndexType buffer[];

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        buffer[idx] = prefixTree[idx];
    }

    __syncthreads();

    BuildPrefix_SM_device(buffer, inSize);

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        prefixTree[idx] = buffer[idx];
    }
}


#endif

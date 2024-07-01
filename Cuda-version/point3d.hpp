#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief      Template version of float3, int3, ...
 */
template <class NumType> struct Point3D;

template <> struct Point3D<float> : public float3
{
    using float3::float3;

    __forceinline__ __device__ __host__ Point3D() : float3(){};
    __forceinline__ __device__ __host__ Point3D(const float3 &inOther)
        : float3(inOther){};

    __forceinline__ __device__ __host__ Point3D(float inx, float iny, float inz)
        : float3(float3{inx, iny, inz}){};

    __forceinline__ __device__ __host__ Point3D &operator=(
        const float3 &inOther)
    {
        float3::operator=(inOther);
        return *this;
    }
};

static_assert(sizeof(Point3D<float>) == sizeof(float3) && sizeof(float3) == 12,
              "Point3D<float> is not the same size as float3");

template <> struct Point3D<double> : public double3
{
    using double3::double3;

    __forceinline__ __device__ __host__ Point3D() : double3(){};
    __forceinline__ __device__ __host__ Point3D(const double3 &inOther)
        : double3(inOther){};

    __forceinline__ __device__ __host__ Point3D(double inx, double iny,
                                                double inz)
        : double3(double3{inx, iny, inz}){};

    __forceinline__ __device__ __host__ Point3D &operator=(
        const double3 &inOther)
    {
        double3::operator=(inOther);
        return *this;
    }
};
static_assert(sizeof(Point3D<double>) == sizeof(double3) && sizeof(double3) == 24,
              "Point3D<float> is not the same size as float3");

template <> struct Point3D<int> : public int3
{
    using int3::int3;

    __forceinline__ __device__ __host__ Point3D() : int3(){};
    __forceinline__ __device__ __host__ Point3D(const int3 &inOther)
        : int3(inOther){};

    __forceinline__ __device__ __host__ Point3D(int inx, int iny, int inz)
        : int3(int3{inx, iny, inz}){};

    __forceinline__ __device__ __host__ Point3D &operator=(const int3 &inOther)
    {
        int3::operator=(inOther);
        return *this;
    }
};

template <> struct Point3D<uint32_t> : public uint3
{
    using uint3::uint3;

    __forceinline__ __device__ __host__ Point3D() : uint3(){};
    __forceinline__ __device__ __host__ Point3D(const uint3 &inOther)
        : uint3(inOther){};

    __forceinline__ __device__ __host__ Point3D(uint32_t inx, uint32_t iny,
                                                uint32_t inz)
        : uint3(uint3{inx, iny, inz}){};

    __forceinline__ __device__ __host__ Point3D &operator=(const uint3 &inOther)
    {
        uint3::operator=(inOther);
        return *this;
    }
};


inline __device__ __host__ dim3 int3ToDim3(const int3 &inPoint)
{
    assert(inPoint.x >= 0 && inPoint.y >= 0 && inPoint.z >= 0);
    return dim3{static_cast<unsigned int>(inPoint.x),
                static_cast<unsigned int>(inPoint.y),
                static_cast<unsigned int>(inPoint.z)};
}

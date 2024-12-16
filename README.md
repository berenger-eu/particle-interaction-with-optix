
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

# License

Apart from specific files located in `SDK/optixMultiSphere`, `SDK/optixMultiTriangle`, `optixMultiSphereAABB` and `optixMultiCustom`, which are under the MIT License, the rest of the SDK directory is under the NVIDIA proprietary license.

`Cuda-version` is also under the MIT License.

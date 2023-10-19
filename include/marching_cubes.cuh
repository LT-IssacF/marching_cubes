#ifndef MARCHING_CUBES_CUH
#define MARCHING_CUBES_CUH

#include "types.cuh"

class MarchingCubes {
public:
    __device__ static void getCubeIndex(Voxel &cube, float isovalue);

    __device__ static _Point linearInterpolation(const Point &p1, const Point &p2, float isovalue);

    __device__ static void getIntersections(Voxel &cube, float isovalue);

    __device__ static void getCubesTriangle(Voxel &cube);
};

__global__ void kernel_getCubeIndex(Voxel *cube, float isovalue, int cnt);

__global__ void kernel_getIntersections(Voxel *cubes, float isovalue, int cnt);

__global__ void kernel_getCubesTriangle(Voxel *cubes, int cnt);

#endif
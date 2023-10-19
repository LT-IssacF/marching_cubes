#include <marching_cubes.cuh>
#include <extra_arrays.cuh>

// ----- kernel 并行算法核心 -----
// 分配的总线程数大于任务数（最简单的一种）：只在 offset 小于任务数时进入 device
// 分配的总线程数小于任务数（需要重复多次使用同一线程）：使用 stride = gridDim * blockDim

__global__ void kernel_getCubeIndex(Voxel *cubes, float isovalue, int cnt) {
    int stride = gridDim.x * blockDim.x;
    int iteration_cnt = cnt / stride, iteration_remainder = cnt % stride;
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    for (int i = 0; i < iteration_cnt; i++) {
        MarchingCubes::getCubeIndex(cubes[stride * i + offset], isovalue);
    } if (offset < iteration_remainder) {
        MarchingCubes::getCubeIndex(cubes[stride * iteration_cnt + offset], isovalue);
    }
}

// ### blockDim must be above 12
__global__ void kernel_getIntersections(Voxel *cubes, float isovalue, int cnt) {
    int realBlockDim = (blockDim.x / 12) * 12; // 以12个线程为单位，每个 block 个内多余的线程将舍弃
    int stride = gridDim.x * realBlockDim;
    int iteration_cnt = cnt * 12 / stride, iteration_remainder = cnt * 12 % stride; // 循环次数 = 任务数 * 每个任务需要的线程数 / 总线程数

    if (threadIdx.x < realBlockDim) { // 舍弃每块多余的线程
        int offset = realBlockDim * blockIdx.x + threadIdx.x;
        for (int i = 0; i < iteration_cnt; i++) {
            MarchingCubes::getIntersections(cubes[(stride / 12) * i + (offset / 12)], isovalue); // stride / 12 = 全部线程一次迭代能够处理的任务数，offset / 12 = 本板块线程应处理的任务序号
        } if (offset < iteration_remainder) {
            MarchingCubes::getIntersections(cubes[(stride / 12) * iteration_cnt + (offset / 12)], isovalue);
        }
    }
}

// ### blockDim must be above 15
__global__ void kernel_getCubesTriangle(Voxel *cubes, int cnt) {
    int realBlockDim = (blockDim.x / 15) * 15; // 以15个线程为单位，每个线程去 triTable 中检查自己的 edgeNum 是否不为-1，然后在 intersections 中取出 point
    int stride = gridDim.x * realBlockDim;
    int iteration_cnt = cnt * 15 / stride, iteration_remainder = cnt * 15 % stride;

    if (threadIdx.x < realBlockDim) {
        int offset = realBlockDim * blockIdx.x + threadIdx.x;
        for (int i = 0; i < iteration_cnt; i++) {
            MarchingCubes::getCubesTriangle(cubes[(stride / 15) * i + (offset / 15)]);
        } if (offset < iteration_remainder) {
            MarchingCubes::getCubesTriangle(cubes[(stride / 15) * iteration_cnt + (offset / 15)]);
        }
    }
}

__device__ void MarchingCubes::getCubeIndex(Voxel &cube, float isovalue) {
    int index = 0; // 每个线程算一个 index
    for (int i = 0; i < 8; i++) {
        if (cube.vertices[i].value < isovalue) {
            index |= 1 << i;
        }
    }
    cube.index = index;
}

__device__ _Point MarchingCubes::linearInterpolation(const Point &p1, const Point &p2, float isovalue) {
    return p1 + (p2 - p1) * ((isovalue - p1.value) / (p2.value - p1.value));
}

__device__ void MarchingCubes::getIntersections(Voxel &cube, float isovalue) {
    int realThreadIdx = threadIdx.x % 12; // 12个线程处理一个 cube，每个线程处理一条 edge 上的 intersection
    if ((edgeTable[cube.index] >> realThreadIdx) & 1) {
        cube.intersections[realThreadIdx] = linearInterpolation(cube.vertices[edgeToVertices[realThreadIdx][0]], cube.vertices[edgeToVertices[realThreadIdx][1]], isovalue);
    }
}

__device__ void MarchingCubes::getCubesTriangle(Voxel &cube) {
    int realThreadIdx = threadIdx.x % 15;
    int edgeNum = triTable[cube.index][realThreadIdx];
    if (edgeNum >= 0) {
        cube.triangle[realThreadIdx] = cube.intersections[edgeNum];
    }
}
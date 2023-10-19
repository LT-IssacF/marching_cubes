#include <marching_cubes.cuh>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CUBES 2

void module_getCubeIndex(Voxel *host_cubes, Voxel *device_cubes, float isovalue, int cnt);
void module_getIntersections(Voxel *host_cubes, Voxel *device_cubes, float isovalue, int cnt);
void module_getCubesTriangle(Voxel *host_cubes, Voxel *device_cubes, int cnt);

int main() {
    float isovalue = 0.25f;
    Voxel host_cubes[CUBES], *device_cubes;
    host_cubes[0] = {
        {
            {0.f, 0.f, 0.f, 0.f},
            {1.f, 0.f, 0.f, 0.5f},
            {1.f, 0.f, 1.f, 0.5f},
            {0.f, 0.f, 1.f, 0.f},
            {0.f, 1.f, 0.f, 0.5f},
            {1.f, 1.f, 0.f, 0.5f},
            {1.f, 1.f, 1.f, 0.5f},
            {0.f, 1.f, 1.f, 0.5f}
        }
    };
    host_cubes[1] = {
        {
            {0.f, 0.f, 0.f, 0.5f},
            {1.f, 0.f, 0.f, 0.5f},
            {1.f, 0.f, 1.f, 0.5f},
            {0.f, 0.f, 1.f, 0.f},
            {0.f, 1.f, 0.f, 0.5f},
            {1.f, 1.f, 0.f, 0.5f},
            {1.f, 1.f, 1.f, 0.5f},
            {0.f, 1.f, 1.f, 0.5f}
        }
    };

    // cudaMalloc 在 device 上分配的地址更像是分配了一个临时地址，故显存数据只能在本函数内和本函数调用的函数内使用，无法在调用它的函数中使用
    // 所以如果想利用之前分配的显存，那么分配过程最好不要包装，统一写在最顶层的调用函数如 main 里最好
    checkCudaErrors(cudaMalloc((void**)&device_cubes, CUBES * sizeof(Voxel)));

    module_getCubeIndex(host_cubes, device_cubes, isovalue, CUBES);

    module_getIntersections(host_cubes, device_cubes, isovalue, CUBES);

    module_getCubesTriangle(host_cubes, device_cubes, CUBES);

    return 0;
}

// ----- module -----

void module_getCubeIndex(Voxel *host_cubes, Voxel *device_cubes, float isovalue, int cnt) {
    checkCudaErrors(cudaMemcpy(device_cubes, host_cubes, CUBES * sizeof(Voxel), cudaMemcpyHostToDevice)); // 但是传送数据并不受影响

    kernel_getCubeIndex<<<64, 64>>>(device_cubes, isovalue, cnt);

    checkCudaErrors(cudaMemcpy(host_cubes, device_cubes, cnt * sizeof(Voxel), cudaMemcpyDeviceToHost));

    for (int i = 0; i < CUBES; i++) {
        std::cout << "cubeIndex[" << i << "]: " << host_cubes[i].index << std::endl;
    }
}

void module_getIntersections(Voxel *host_cubes, Voxel *device_cubes, float isovalue, int cnt) {
    kernel_getIntersections<<<64, 64>>>(device_cubes, isovalue, cnt);

    checkCudaErrors(cudaMemcpy(host_cubes, device_cubes, cnt * sizeof(Voxel), cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; i++) {
        for (int j = 0; j < 12; j++) {
            if (host_cubes[i].intersections[j].x != 1.17549e-38f) {
                std::cout << std::right << "intersection[" << i << "][" << std::setw(2) << j << "]: " << 
                std::setw(11) << host_cubes[i].intersections[j].x << " " << std::setw(11) << host_cubes[i].intersections[j].y << " "  << std::setw(11) << host_cubes[i].intersections[j].z << std::endl;
            }
        }
    }
}

void module_getCubesTriangle(Voxel *host_cubes, Voxel *device_cubes, int cnt) {
    kernel_getCubesTriangle<<<64, 64>>>(device_cubes, cnt);

    checkCudaErrors(cudaMemcpy(host_cubes, device_cubes, cnt * sizeof(Voxel), cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; i++) {
        for (int j = 0; j < 15; j++) {
            if (host_cubes[i].triangle[j].x != 1.17549e-38f) {
                std::cout << std::right << "triangle[" << i << "][" << j << "]: " <<
                std:: setw(11) << host_cubes[i].triangle[j].x << std:: setw(11) << host_cubes[i].triangle[j].y << std:: setw(11) << host_cubes[i].triangle[j].z << std::endl;
            }
        }
    }
}
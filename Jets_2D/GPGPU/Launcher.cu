// Standard Library
#include <stdio.h>

// CUDA
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

// File Header
#include "Jets_2D/GPGPU/Launcher.hpp"
#include "Jets_2D/GPGPU/State.hpp"

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"


__host__ __device__ int Div_Up(int a, int b)
{
    return (a + b - 1) / b;
}

// __global__ void Coop_Test_Kernel(int in)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;

//     if (idx >= 16)
//     {
//         return;
//     }

//     printf("%d. Thread: %d\n", in, idx);

//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();
//     grid.sync();

//     printf("%d. Thread: %d\n", in, idx * 100);

//     grid.sync();
// }

// void Launch_Coop_Test_Kernel(int in1, int in2)
// {
//     // void* arr[] = {&in1, &in2};

//     // cudaLaunchCooperativeKernel((void*)Coop_Test_Kernel, 4, 4, (void**)arr);
//     // cudaCheck(cudaDeviceSynchronize());

//     Cooperative_Kernel_Launch(Coop_Test_Kernel, 16, in1, in2);

//     int dev = 0;
//     int supportsCoopLaunch = 0;
//     cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

//     printf("Is coop supported? %d\n", supportsCoopLaunch);

//     printf("Launch_Coop_Test_Kernel\n");
// }

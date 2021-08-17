
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include "Tests/GPGPU/Coop_Test.hpp"

__global__ void Coop_Test_Kernel(int in)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    printf("%d. Thread: %d0\n", in, idx);
}

void Launch_Coop_Test_Kernel(int in)
{
    int* p_in = &in;

    int* arr = {p_in};

    //cuLaunchCooperativeKernel((CUfunction)Coop_Test_Kernel, 1, 1, 1, 32, 1, 1, 0, 0, (void**)arr);

    printf("Launch_Coop_Test_Kernel\n");
}

#include <stdio.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include "Tests/GPGPU/Coop_Test.hpp"

#include "Jets_2D/GPGPU/GPErrorCheck.hpp"

template <typename F, typename... Args>
void for_each_argument_address(F f, Args&&... args)
{
    [](...){}((f( (void*) &std::forward<Args>(args) ), 0)...);
}

template<typename KernelFunction, typename... KernelParameters>
inline void cooperative_launch(const KernelFunction& kernel_function, KernelParameters... parameters)
{
    void* arguments_ptrs[sizeof...(KernelParameters)];
    int arg_index = 0;
    for_each_argument_address([&](void * x) {arguments_ptrs[arg_index++] = x;}, parameters...);
    cudaLaunchCooperativeKernel<KernelFunction>(&kernel_function, 4, 32, arguments_ptrs, 0, 0);
}

__global__ void Coop_Test_Kernel(int in)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    printf("%d. Thread: %d\n", in, idx);

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    printf("%d. Thread: %d\n", in, idx * 100);

    grid.sync();
}

void Launch_Coop_Test_Kernel(int in1, int in2)
{
    void* arr[] = {&in1, &in2};

    cudaLaunchCooperativeKernel((void*)Coop_Test_Kernel, 4, 4, (void**)arr);
    // cudaCheck(cudaDeviceSynchronize());

    // cooperative_launch(Coop_Test_Kernel, in1, in2);

    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

    printf("Is coop supported? %d\n", supportsCoopLaunch);

    printf("Launch_Coop_Test_Kernel\n");
}
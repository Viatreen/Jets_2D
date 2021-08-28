#pragma once

// Standard Library
#include <utility>

// CUDA
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

__host__ __device__ int Div_Up(int a, int b);

template <typename Function, typename... Args>
void For_Each_Argument(Function Lambda_Function, Args&&... args)
{
    [](...){}((Lambda_Function( (void*)&std::forward<Args>(args) ), 0)...);
}

template<typename KernelFunction, typename... KernelParameters>
void CUDA_Kernel_Launch(const KernelFunction& Kernel_Function, int TotalThreads, KernelParameters... Parameters)
{
    void* ArgumentPtrs[sizeof...(KernelParameters)];
    int ArgumentIndex = 0;
    For_Each_Argument([&](void * x){ ArgumentPtrs[ArgumentIndex++] = x; }, Parameters...);

    cudaCheck(cudaLaunchKernel((void*)&Kernel_Function, Div_Up(TotalThreads, BLOCK_SIZE), BLOCK_SIZE, ArgumentPtrs, 0, 0));
}

template<typename KernelFunction, typename... KernelParameters>
void Cooperative_Kernel_Launch(const KernelFunction& Kernel_Function, int TotalThreads, KernelParameters... Parameters)
{
    void* ArgumentPtrs[sizeof...(KernelParameters)];
    int ArgumentIndex = 0;
    For_Each_Argument([&](void * x){ ArgumentPtrs[ArgumentIndex++] = x; }, Parameters...);

    cudaCheck(cudaLaunchCooperativeKernel((void*)&Kernel_Function, Div_Up(TotalThreads, BLOCK_SIZE), BLOCK_SIZE, ArgumentPtrs, 0, 0));
}

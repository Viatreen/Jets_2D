#include "Helper.h"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__host__ __device__ int Div_Up(int a, int b)
{
    return (a + b - 1) / b;
}
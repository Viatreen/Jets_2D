#pragma once

#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

__global__ void Coop_Test_Kernel(int in);
void Launch_Coop_Test_Kernel(int in);

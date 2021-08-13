#pragma once

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Project Headers
#include "Jets_2D/GPGPU/State.h"
#include "Jets_2D/Config.h"

__global__ void RunEpoch(MatchState* Match, CraftState* C, GraphicsObjectPointer* Buffer, config* Config, int Opponent_ID_Weights);
__global__ void ScoreCumulativeCalc(CraftState* C);

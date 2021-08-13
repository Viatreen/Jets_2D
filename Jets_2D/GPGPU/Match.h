#pragma once

// Standard Library
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Project Headers
#include "Jets_2D/GPGPU/State.h"

__device__ void WeightsMutateAndTransfer(CraftState* C, config* Config, int SourceIndex, int TargetIndex);
__global__ void RoundAssignPlace(CraftState* Crafts);
__global__ void RoundTieFix(CraftState* Crafts);
__global__ void RoundPrintFirstPlace(CraftState* C, int RoundNumber);
__global__ void IDAssign(CraftState* C, config* Config);
__global__ void WeightsAndIDTempSave(CraftState* C, temp* Temp);
__global__ void WeightsAndIDTransfer(CraftState* C, temp* Temp); // TODO: Combine transfer and mutate kernels
__global__ void WeightsMutate(CraftState* C, config* Config);
__global__ void ScoreTempSave(CraftState* C);
__global__ void ScoreTransfer(CraftState* C);
__global__ void ResetScoreCumulative(CraftState* Crafts);
__global__ void Init(CraftState* C);
__device__ void Reset(CraftState* Crafts, int idx, GraphicsObjectPointer Buffer, float PositionX, float PositionY, float AngleStart);
__global__ void ResetMatch(MatchState* Match, CraftState* Crafts, GraphicsObjectPointer* Buffer, int PositionNumber, float AngleStart);

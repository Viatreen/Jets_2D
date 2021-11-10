#pragma once

#include "Jets_2D/GPGPU/NeuralNet_Eval.hpp"

#include "Jets_2D/GPGPU/State.hpp"

__host__ void Test_Neural_Net_Eval(CraftState* C);
__global__ void Print_Layer_Eval(CraftState* C, unsigned int Layer);
__global__ void Print_Layer_Eval_Delta_Neurons(CraftState* C, unsigned int Layer);
__global__ void Print_Layer_Eval_Delta_Weights(CraftState* C, unsigned int Layer);

#pragma once

// CUDA
#include <cuda_runtime.h>

// Project Headers
#include "Jets_2D/GPGPU/State.hpp"

__device__ void RELU_Activate(float& Neuron);
__device__ void State_Processing(CraftState* C, GraphicsObjectPointer* Buffer, int ID_Opponent, int ID_Craft, int ID_Weight, int TimeStep);
__device__ void Run_Neural_Net(CraftState* C, bool Do_Activation, int ID_Neurons, int ID_Weights, int TimeStep);
__device__ void Environment_To_Input_Neurons(CraftState* C, int ID_Neurons, int ID_Weights, int TimeStep);
__device__ void Output_Neurons_To_Action(CraftState* C, int ID_Craft, GraphicsObjectPointer* Buffer, int TimStep);
__device__ void BackPropagate(CraftState* C, int Craft_ID);

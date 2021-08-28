#pragma once
// CUDA
#include <cuda_runtime.h>

// Project Headers
#include "Jets_2D/GPGPU/State.hpp"

struct neuron_Indices
{
    unsigned int Origin_Neuron_Index;
    unsigned int Target_Neuron_Index;
};

__global__ void  Zeroize_Non_Input_Layers(CraftState* C);
__global__ void  Reset_Neural_Net_Eval_Delta_Neuron(CraftState* C);
__global__ void  RELU_Activate_Layer_Eval(CraftState *C, unsigned int Layer);
__host__   float Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(CraftState *C);
__global__ void  Initialize_Neural_Net_Eval(CraftState *C);
__host__   void  BackPropagate_Eval_Host(CraftState *C, float Target_Result);
__global__ void  BackPropagate_Eval_Compute_Deltas_To_Neurons(CraftState *C); //, float Target_Result);
__global__ void  BackPropagate_Eval_Account_For_Activation_Slope(CraftState* C, int Layer);
__global__ void  BackPropagate_Eval_Compute_Deltas(CraftState* C, int Layer);
__global__ void  Run_Neural_Net_Eval(CraftState *C);
__device__ neuron_Indices Get_Neuron_Indices(CraftState *C, const unsigned int &Weight_Index, unsigned int Layer);
__device__ __host__ int Get_Weight_Amount_In_Layer(int Layer);
__device__ __host__ int Get_Weight_Begin_Index(int Layer);
__device__ __host__ int Get_Neuron_Amount_In_Layer(int Layer);
__device__ __host__ int Get_Neuron_Begin_Index(int Layer);

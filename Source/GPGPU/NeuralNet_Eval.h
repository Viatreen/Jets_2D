#pragma once
// CUDA
#include <cuda_runtime.h>

// Project Headers
#include "GPGPU/State.h"

struct Weight_Characteristic
{
    unsigned int Layer;
    unsigned int Origin_Neuron_Index;
    unsigned int Target_Neuron_Index;
    unsigned int Neuron_Index_Layer_Begin;
    unsigned int Origin_Neuron_Index_Within_Layer;
    unsigned int Target_Neuron_Index_Within_Layer;
    unsigned int Weight_Index;
    unsigned int Weight_Index_Layer_Begin;
    unsigned int Weight_Index_Within_Layer;
    unsigned int Next_Layer_Size;
};
struct neuron_Indices
{
    unsigned int Origin_Neuron_Index;
    unsigned int Target_Neuron_Index;
};

__global__ void  Reset_Neural_Net_Eval(CraftState* C);
__global__ void  Reset_Neural_Net_Eval_Delta_Neuron(CraftState* C);
__global__ void  RELU_Activate_Layer_Eval(CraftState *C, unsigned int Layer);
__host__   float Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(CraftState *C);
__host__   void  Test_Neural_Net_Eval(CraftState *C);
__global__ void  Initialize_Neural_Net_Eval(CraftState *C);
__host__   void  BackPropagate_Eval_Host(CraftState *C, float Target_Result);
__global__ void  BackPropagate_Eval_Compute_Deltas_To_Neurons(CraftState *C, int Layer, float Target_Result);
__global__ void  BackPropagate_Eval_Account_For_Activation_Slope(CraftState* C, int Layer);
__global__ void  BackPropagate_Eval_Compute_Deltas(CraftState* C, int Layer);
__global__ void  Print_Layer_Eval(CraftState *C, unsigned int Layer);
__global__ void Print_Layer_Eval_Delta_Neurons(CraftState* C, unsigned int Layer);
__global__ void Print_Layer_Eval_Delta_Weights(CraftState* C, unsigned int Layer);
__global__ void  Run_Neural_Net_Eval(CraftState *C, unsigned int Layer);
__device__ void  Run_Neural_Net_Layer_Eval(CraftState *C, const unsigned int &Weight_Index, const bool &Do_Activation, unsigned int Layer, neuron_Indices Neuron_Value);
__device__ neuron_Indices Get_Neuron_Indices(CraftState *C, const unsigned int &Weight_Index, unsigned int Layer);
__device__ __host__ int   Get_Weight_Amount_In_Layer(int Layer);
__device__ int   Get_Weight_Begin_Index(int Layer);
__device__ __host__ int   Get_Neuron_Amount_In_Layer(int Layer);
__device__ __host__ int Get_Neuron_Begin_Index(int Layer);
__device__ void  BackPropagate_Eval_Old(CraftState *C);
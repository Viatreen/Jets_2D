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

struct Weight_Characteristic_Global
{
	unsigned int Layer[WEIGHT_AMOUNT_EVAL];
	unsigned int Origin_Neuron_Index[WEIGHT_AMOUNT_EVAL];
	unsigned int Target_Neuron_Index[WEIGHT_AMOUNT_EVAL];
	unsigned int Neuron_Index_Layer_Begin[WEIGHT_AMOUNT_EVAL];
	unsigned int Origin_Neuron_Index_Within_Layer[WEIGHT_AMOUNT_EVAL];
	unsigned int Target_Neuron_Index_Within_Layer[WEIGHT_AMOUNT_EVAL];
	unsigned int Weight_Index[WEIGHT_AMOUNT_EVAL];
	unsigned int Weight_Index_Layer_Begin[WEIGHT_AMOUNT_EVAL];
	unsigned int Weight_Index_Within_Layer[WEIGHT_AMOUNT_EVAL];
	unsigned int Next_Layer_Size[WEIGHT_AMOUNT_EVAL];
};

struct neuron_Indices
{
	unsigned int Origin_Neuron_Index;
	unsigned int Target_Neuron_Index;
};

__device__ void BackPropagate_Populate_Neurons(CraftState* C, const unsigned int &Weight_Index);
__device__ void BackPropagate_Eval(CraftState* C);
__global__ void RELU_Activate_Layer(CraftState* C, unsigned int Layer);
__host__ void Test_Neural_Net_Eval(CraftState* C);
__global__ void Create_Neural_Net_Eval(CraftState* C);
__global__ void Print_Layer_Eval(CraftState* C, unsigned int Layer);
__global__ void Run_Neural_Net_Eval(CraftState* C, unsigned int Layer);
__device__ void Populate_Weight_Data(CraftState* C, Weight_Characteristic_Global* WG, const unsigned int &Weight_Index);
__device__ void Copy_Weight_Characteristics_From_Global(Weight_Characteristic_Global* WG, Weight_Characteristic& W);
__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, const unsigned int &Weight_Index, const bool &Do_Activation, unsigned int Layer);
__device__ void Activate_Layer_Eval(CraftState* C, const unsigned int &Neuron_Index);
__device__ neuron_Indices Get_Neuron_Indices(CraftState *C, const unsigned int &Weight_Index, unsigned int Layer);

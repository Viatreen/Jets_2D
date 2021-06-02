#pragma once

__device__ void BackPropagate_Eval(CraftState* C, unsigned int Weight_Index);
__host__ void Run_Neural_Net_Eval(Craftstate* C, bool Do_Activation);
__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, unsigned int Weight_Index, unsigned int Layer_Index);
__device__ void Activate_Layer_Eval(CraftState* C, unsigned int Layer_Index, unsigned int Neuron_Index_Within_Layer);

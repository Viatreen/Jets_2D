// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

// Standard Library
#include <stdio.h>
#include <iostream>
#include <iomanip>

// File Header
#include "Jets_2D/GPGPU/NeuralNet_Eval.hpp"

// Project Headers
#include "Jets_2D/GPGPU/NeuralNet.hpp"
#include "Jets_2D/GPGPU/Helper.hpp"

__global__ void Initialize_Neural_Net_Eval(CraftState* C)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Grid_Size = blockDim.x * gridDim.x;

    if (idx < NEURON_AMOUNT_INPUT_EVAL)
    {
        C->Eval_Network.Neuron[idx] = 1.f;
    }

    unsigned int Batch_Amount = (WEIGHT_AMOUNT_EVAL + Grid_Size - 1) / Grid_Size;

    for (unsigned int i = 0; i < Batch_Amount; i++)
    {
        unsigned int Weight_Index = Grid_Size * i + idx;
        if (Weight_Index < WEIGHT_AMOUNT_EVAL)
        {
            C->Eval_Network.Weight[Weight_Index] = 0.09f;
            // if (Weight_Index == 500)
            // {
            //     C->Eval_Network.Weight[Weight_Index] += 0.31;
            //     printf("Weight: %f\n", C->Eval_Network.Weight[Weight_Index]);
            // }
        }
    }
}

__host__ void BackPropagate_Eval_Host(CraftState* C, float Target_Result)
{
    float Result = Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(C);

    Reset_Neural_Net_Eval_Delta_Neuron<<<Div_Up(NEURON_AMOUNT_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C);
    cudaDeviceSynchronize();

    float Delta_Error = Target_Result - Result;

    for (int Layer = LAYER_AMOUNT_EVAL - 2; Layer >= 0; Layer--)
    {
        BackPropagate_Eval_Compute_Deltas_To_Neurons<<<Div_Up(Get_Weight_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer, Delta_Error);
        cudaDeviceSynchronize();
        // int Neuron_Amount = Get_Neuron_Amount_In_Layer(Layer);
        //BackPropagate_Eval_Account_For_Activation_Slope<<<Div_Up(Neuron_Amount, BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        //cudaDeviceSynchronize();
        BackPropagate_Eval_Compute_Deltas<<<Div_Up(Get_Weight_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaDeviceSynchronize();
    }
}

__global__ void BackPropagate_Eval_Compute_Deltas_To_Neurons(CraftState* C, int Layer, float Delta_Error)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= Get_Weight_Amount_In_Layer(Layer))
    {
        return;
    }

    int Weight_Index = Get_Weight_Begin_Index(Layer) + idx;
    neuron_Indices Neuron_Value = Get_Neuron_Indices(C, Weight_Index, Layer);

    if (Layer == LAYER_AMOUNT_EVAL - 2) // Last Layer
    {
        // Calculate delta neurons for the first hidden layer based on input layer
        C->Eval_Network.Delta_Result_Over_Delta_Weight[Weight_Index] = C->Eval_Network.Neuron[Neuron_Value.Origin_Neuron_Index] * Delta_Error;
        atomicAdd(&C->Eval_Network.Delta_Neuron[Neuron_Value.Origin_Neuron_Index], C->Eval_Network.Neuron[Neuron_Value.Origin_Neuron_Index]);

        // if (idx < Get_Neuron_Amount_In_Layer(Layer))
        // {
        //     C->Eval_Network.Delta_Neuron[Neuron_Value.Target_Neuron_Index] = 1.f;
        // }

        // printf("idx: %d, WI: %d, ON: %d, TN: %d, TotalN: %d\n", idx, Weight_Index, Neuron_Value.Origin_Neuron_Index, Neuron_Value.Target_Neuron_Index, NEURON_AMOUNT_EVAL);

        // TODO: Implement after cooperative groups is working
        // grid.sync();
    }
    else
    {
        float BackPropagate_Signal = C->Eval_Network.Weight[Weight_Index] * C->Eval_Network.Delta_Neuron[Neuron_Value.Target_Neuron_Index];

        // printf("Thread: %d, BackPropagate_Signal: %f\n", idx, BackPropagate_Signal);
        // printf("Thread: %d, Target_Neuron_Index: %d\n", idx, Neuron_Value.Target_Neuron_Index);

        atomicAdd(&C->Eval_Network.Delta_Neuron[Neuron_Value.Origin_Neuron_Index], BackPropagate_Signal);

        // TODO: Implement after cooperative groups is working
        // grid.sync();
    }

    // TODO: Implement after cooperative groups is working
    // grid.sync();
}

__global__ void BackPropagate_Eval_Account_For_Activation_Slope(CraftState* C, int Layer)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx >= NEURON_AMOUNT_HIDDEN_EVAL)
    {
        return;
    }

    int Neuron_Index = NEURON_AMOUNT_INPUT_EVAL + (Layer - 1) * NEURON_AMOUNT_HIDDEN_EVAL + idx;

    if (C->Eval_Network.Neuron[Neuron_Index] > 1.f || C->Eval_Network.Neuron[Neuron_Index] < -1.f)
    {
        C->Eval_Network.Delta_Neuron[Neuron_Index] *= NETWORK_ACTIVATION_SLOPE;
    }

    // TODO: Implement after cooperative groups is working
    // grid.sync();
}

__global__ void BackPropagate_Eval_Compute_Deltas(CraftState* C, int Layer)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx >= Get_Weight_Amount_In_Layer(Layer))
    {
        return;
    }

    int Weight_Index = Get_Weight_Begin_Index(Layer) + idx;

    neuron_Indices Neuron_Value = Get_Neuron_Indices(C, Weight_Index, Layer);

    C->Eval_Network.Delta_Result_Over_Delta_Weight[Weight_Index] = C->Eval_Network.Neuron[Neuron_Value.Origin_Neuron_Index] * C->Eval_Network.Delta_Neuron[Neuron_Value.Target_Neuron_Index];

    // TODO: Implement after cooperative groups is working
    // grid.sync();
}

__global__ void Reset_Neural_Net_Eval(CraftState* C)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NEURON_AMOUNT_EVAL - NEURON_AMOUNT_INPUT_EVAL)
    {
        C->Eval_Network.Neuron[NEURON_AMOUNT_INPUT_EVAL + idx] = 0.f;
    }
}

__global__ void Reset_Neural_Net_Eval_Delta_Neuron(CraftState* C)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NEURON_AMOUNT_EVAL)
    {
        C->Eval_Network.Delta_Neuron[idx] = 0.f;
    }
}

__global__ void RELU_Activate_Layer_Eval(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int Layer_Size = Get_Neuron_Amount_In_Layer(Layer);
    unsigned int Layer_Begin_Index = Get_Neuron_Begin_Index(Layer);

    if (idx < Layer_Size && Layer != LAYER_AMOUNT_EVAL - 1)
    {
        RELU_Activate(C->Eval_Network.Neuron[Layer_Begin_Index + idx]);
    }
}

__host__ float Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(CraftState* C)
{
    Reset_Neural_Net_Eval<<<Div_Up(NEURON_AMOUNT_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C);
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < LAYER_AMOUNT_EVAL; i++)
    {
        Run_Neural_Net_Eval<<<Div_Up(WEIGHT_AMOUNT_MAX_LAYER_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C, i);
        cudaDeviceSynchronize();
        // RELU_Activate_Layer_Eval<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(C, i);
        // cudaDeviceSynchronize();
    }

    float Result;
    cudaMemcpy(&Result, &C->Eval_Network.Neuron[OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL], sizeof(float) , cudaMemcpyDeviceToHost);

    // std::cout << "Result 1: " << std::setprecision(6) << Result << std::endl;

    return Result;
}

// TODO: Move to test file
__host__ void Test_Neural_Net_Eval(CraftState* C)
{
    Initialize_Neural_Net_Eval<<<Div_Up(WEIGHT_AMOUNT_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C);
    cudaDeviceSynchronize();

    // std::cout << "Run 1" << std::endl;
    float Result = Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(C);

    for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        Print_Layer_Eval<<<Div_Up(NEURON_AMOUNT_MAX_LAYER_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaDeviceSynchronize();
        std::cout << std::endl;
    }

    // std::cout << "Run 2" << std::endl;
    // Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(C);
    // for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++)
    // {
    //     std::cout << "Layer " << Layer << ": ";
    //     Print_Layer_Eval<<<Div_Up(NEURON_AMOUNT_MAX_LAYER_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
    //     cudaDeviceSynchronize();
    //     std::cout << std::endl;
    // }

    // std::cout << "Run 1" << std::endl;
    float Target_Result = 1.f;
    BackPropagate_Eval_Host(C, Target_Result);

    std::cout << "BackPropagation:" << std::endl;
    std::cout << "Target Result: " << Target_Result << ", Actual Result: " << Result << std::endl;

    std::cout << "Neurons" << std::endl;
    for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        // Print_Layer_Eval_Delta_Neurons<<<NEURON_AMOUNT_INPUT_EVAL, BLOCK_SIZE>>>(C, Layer);
        Print_Layer_Eval_Delta_Neurons<<<Div_Up(Get_Neuron_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaDeviceSynchronize();
        std::cout << std::endl;
    }

    std::cout << "Weights" << std::endl;
    for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL - 1; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        // Print_Layer_Eval_Delta_Neurons<<<NEURON_AMOUNT_INPUT_EVAL, BLOCK_SIZE>>>(C, Layer);
        Print_Layer_Eval_Delta_Weights<<<Div_Up(Get_Weight_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaDeviceSynchronize();
        std::cout << std::endl;
    }

    // std::cout << "Run 2" << std::endl;
    // BackPropagate_Eval_Host(C, Target_Result);
    // std::cout << "BackPropagation:" << std::endl;
    // std::cout << "Target Result: " << Target_Result << ", Actual Result: " << Result << std::endl;
    // for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL - 1; Layer++)
    // {
    //     std::cout << "Layer " << Layer << ": ";
    //     // Print_Layer_Eval_Delta_Neurons<<<NEURON_AMOUNT_INPUT_EVAL, BLOCK_SIZE>>>(C, Layer);
    //     Print_Layer_Eval_Delta_Weights<<<Div_Up(Get_Weight_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
    //     cudaDeviceSynchronize();
    //     std::cout << std::endl;
    // }
}

__global__ void Print_Layer_Eval(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int Layer_Size = Get_Neuron_Amount_In_Layer(Layer);
    int Neuron_Begin_Index = Get_Neuron_Begin_Index(Layer);

    if (idx < Layer_Size)
    {
        printf("%9.6f, ", C->Eval_Network.Neuron[Neuron_Begin_Index + idx]);
    }
}

__global__ void Print_Layer_Eval_Delta_Neurons(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= Get_Neuron_Amount_In_Layer(Layer))
    {
        return;
    }

    int Neuron_Index = Get_Neuron_Begin_Index(Layer) + idx;

    // printf("Idx: %d, Neuron_Index: %d\n", idx, Neuron_Index);

    printf("%7.2f, ", C->Eval_Network.Delta_Neuron[Neuron_Index]);
}

__global__ void Print_Layer_Eval_Delta_Weights(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 20 || idx >= Get_Weight_Amount_In_Layer(Layer))
    // if (idx >= Get_Weight_Amount_In_Layer(Layer))
    {
        return;
    }

    int Weight_Index = Get_Weight_Begin_Index(Layer) + idx;

    // printf("Idx: %d, Weight_Index: %d\n", idx, Weight_Index);

    printf("%7.2f, ", C->Eval_Network.Delta_Result_Over_Delta_Weight[Weight_Index]);
}

__global__ void Run_Neural_Net_Eval(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO Grid sync placeholder
    // cooperative_groups::grid_group grid = cooperative_groups::this_grid()

    // for (unsigned int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++) // TODO: Add back after grid sync is established
    {
        int Weight_Index = Get_Weight_Begin_Index(Layer) + idx;

        if (idx < Get_Weight_Amount_In_Layer(Layer))
        {
            int Weight_Index = Get_Weight_Begin_Index(Layer) + idx;
            neuron_Indices Neuron_Value = Get_Neuron_Indices(C, Weight_Index, Layer);
            Run_Neural_Net_Layer_Eval(C, Weight_Index, true, Layer, Neuron_Value);
        }

        // grid.sync();
        // TODO: Activate Layers
        // grid.sync();
    }
}

__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, const unsigned int &Weight_Index, const bool &Do_Activation, unsigned int Layer, neuron_Indices Neuron_Value)
{
    float Weight = C->Eval_Network.Weight[Weight_Index];
    float Neuron = C->Eval_Network.Neuron[Neuron_Value.Origin_Neuron_Index];

    float Signal = Neuron * Weight;

    atomicAdd(&C->Eval_Network.Neuron[Neuron_Value.Target_Neuron_Index], Signal);

    // TODO: synchronize grid
    // grid.sync();
}

// Get origin neuron index and target neuron index of each weight. Make sure the Weight_Index value is bounds checked before passed in
__device__ neuron_Indices Get_Neuron_Indices(CraftState *C, const unsigned int &Weight_Index, unsigned int Layer)
{
    neuron_Indices Result;

    // First Layer
    if (Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL)
    {
        Result.Origin_Neuron_Index = Weight_Index / NEURON_AMOUNT_HIDDEN_EVAL;
        Result.Target_Neuron_Index = NEURON_AMOUNT_INPUT_EVAL + Weight_Index % NEURON_AMOUNT_HIDDEN_EVAL;
    }
    // Hidden Layers
    else if (Weight_Index < OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL)
    {
        unsigned int Weight_Index_Within_Layer = (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) % (WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL);
        unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / NEURON_AMOUNT_HIDDEN_EVAL;
        unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % NEURON_AMOUNT_HIDDEN_EVAL;

        unsigned int Layer_Index = 1 + (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) / WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
        Result.Origin_Neuron_Index = NEURON_AMOUNT_INPUT_EVAL + (Layer_Index - 1) * NEURON_AMOUNT_HIDDEN_EVAL + Origin_Neuron_Index_Within_Layer;
        Result.Target_Neuron_Index = NEURON_AMOUNT_INPUT_EVAL + Layer_Index * NEURON_AMOUNT_HIDDEN_EVAL + Target_Neuron_Index_Within_Layer;
    }
    // Last Layer
    else
    {
        unsigned int Weight_Index_Within_Layer = Weight_Index - OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL;

        // unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / NEURON_AMOUNT_OUTPUT_EVAL;
        // unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % NEURON_AMOUNT_OUTPUT_EVAL;
        unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / NEURON_AMOUNT_OUTPUT_EVAL;
        unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % NEURON_AMOUNT_OUTPUT_EVAL;

        Result.Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - NEURON_AMOUNT_HIDDEN_EVAL + Origin_Neuron_Index_Within_Layer;
        Result.Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + Target_Neuron_Index_Within_Layer;
    }

    return Result;
}

__device__ __host__ int Get_Weight_Amount_In_Layer(int Layer)
{
    if (Layer == 0)
    {
        return WEIGHT_AMOUNT_INPUT_LAYER_EVAL;
    }
    else if (Layer < LAYER_AMOUNT_EVAL - 2)
    {
        return WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
    }
    else
    {
        return WEIGHT_AMOUNT_OUTPUT_LAYER_EVAL;
    }
}

__device__ int Get_Weight_Begin_Index(int Layer)
{
    if (Layer == 0)
    {
        return 0;
    }
    else
    {
        return WEIGHT_AMOUNT_INPUT_LAYER_EVAL + (Layer - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
    }
}

__device__ __host__ int Get_Neuron_Amount_In_Layer(int Layer)
{
    if (Layer == 0)
    {
        return NEURON_AMOUNT_INPUT_EVAL;
    }
    else if (Layer < LAYER_AMOUNT_EVAL - 1)
    {
        return NEURON_AMOUNT_HIDDEN_EVAL;
    }
    else
    {
        return NEURON_AMOUNT_OUTPUT_EVAL;
    }
}

__device__ __host__ int Get_Neuron_Begin_Index(int Layer)
{
    if (Layer == 0)
    {
         return 0;
    }
    // Hidden Layers
    else if (Layer < LAYER_AMOUNT_EVAL - 1)
    {
        return NEURON_AMOUNT_INPUT_EVAL + NEURON_AMOUNT_HIDDEN_EVAL * (Layer - 1);
    }
    // Last Layer (Has 1 output neuron)
    else
    {
        return OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
    }
}

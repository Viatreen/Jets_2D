// Standard Library
#include <iostream>

// CUDA library
#include <cuda_runtime.h>

// Project Headers
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"
#include "Jets_2D/GPGPU/Launcher.hpp"
#include "Jets_2D/GPGPU/State.hpp"

// File Header
#include "Tests/GPGPU/NeuralNet_Eval.test.hpp"

__host__ void Test_Neural_Net_Eval(CraftState* C)
{
    Initialize_Neural_Net_Eval<<<Div_Up(WEIGHT_AMOUNT_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C);
    cudaCheck(cudaDeviceSynchronize());

    float Result = Run_Neural_Net_Eval_This_Is_The_Function_Until_Sync_Is_Figured_Out(C);

    for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        Print_Layer_Eval<<<Div_Up(NEURON_AMOUNT_MAX_LAYER_EVAL, BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaCheck(cudaDeviceSynchronize());
        std::cout << std::endl;
    }

    float Target_Result = 1.f;
    BackPropagate_Eval_Host(C, Target_Result);
    cudaCheck(cudaDeviceSynchronize());

    std::cout << "BackPropagation:" << std::endl;
    std::cout << "Target Result: " << Target_Result << ", Actual Result: " << Result << std::endl;

    std::cout << "Delta Neurons" << std::endl;
    for (int Layer = 0; Layer < LAYER_AMOUNT_HIDDEN_EVAL; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        Print_Layer_Eval_Delta_Neurons<<<Div_Up(Get_Neuron_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaCheck(cudaDeviceSynchronize());
        std::cout << std::endl;
    }

    std::cout << "Delta Weights" << std::endl;
    for (int Layer = 0; Layer < LAYER_AMOUNT_EVAL - 1; Layer++)
    {
        std::cout << "Layer " << Layer << ": ";
        Print_Layer_Eval_Delta_Weights<<<Div_Up(Get_Weight_Amount_In_Layer(Layer), BLOCK_SIZE), BLOCK_SIZE>>>(C, Layer);
        cudaCheck(cudaDeviceSynchronize());
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
        printf("%8.6f, ", C->Eval_Network.Neuron[Neuron_Begin_Index + idx]);
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

    printf("%6.2f, ", C->Eval_Network.Delta_Neuron[Neuron_Index]);
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

    printf("%6.2f,", C->Eval_Network.Delta_Result_Over_Delta_Weight[Weight_Index]);
}

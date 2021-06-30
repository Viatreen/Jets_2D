// CUDA
#include <curand_kernel.h>

// Standard Library
#include <stdio.h>
#include <iostream>

// File Header
#include "NeuralNet_Eval.h"
#include "NeuralNet.h"
#include <cooperative_groups.h>

// Project Headers
#include "NeuralNet.h"

__device__ void BackPropagate_Populate_Neurons(CraftState* C, const unsigned int &Weight_Index)
{
    //C->Eval_Network.Delta_Neuron[];
}

__device__ void BackPropagate_Eval(CraftState* C, Weight_Characteristic *W)
{
    // Make sure Weight_Index is not greater than or equal to weight amount

    float Origin_Neuron = C->Eval_Network.Neuron[W->Origin_Neuron_Index];
    float Weight = C->Eval_Network.Weight[W->Weight_Index];

    float Delta_First_Neuron = Origin_Neuron * Weight;

    if (W->Layer == LAYER_AMOUNT_EVAL - 1)
    {
        C->Eval_Network.Delta_Weight[W->Weight_Index] = Delta_First_Neuron;
        return;
    }

    float Target_Neuron = C->Eval_Network.Neuron[W->Target_Neuron_Index];

    if (Target_Neuron > 1.f || Target_Neuron < -1.f)
    {
        Delta_First_Neuron *= NETWORK_ACTIVATION_SLOPE;
    }

    if (W->Layer == LAYER_AMOUNT_EVAL - 2)
    {
        unsigned int Final_Weight_Index
            = WEIGHT_AMOUNT_INPUT_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL + W->Target_Neuron_Index;
        float Final_Weight = C->Eval_Network.Weight[Final_Weight_Index];

        C->Eval_Network.Delta_Weight[W->Weight_Index] = Delta_First_Neuron * Final_Weight;
        return;
    }

    float Delta_Neuron_Previous_Layer[LAYER_SIZE_HIDDEN_EVAL];
    float Delta_Neuron_Next_Layer[LAYER_SIZE_HIDDEN_EVAL];

    // First broadcast from the target neuron
    for (unsigned int i = 0; i < LAYER_SIZE_HIDDEN_EVAL; i++)
    {
        unsigned int First_Broadcast_Neuron_Weight_Index = W->Weight_Index + WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL + (W->Target_Neuron_Index - W->Origin_Neuron_Index) * LAYER_SIZE_HIDDEN_EVAL - W->Target_Neuron_Index + i;

        float First_Broadcast_Neuron_Weight = C->Eval_Network.Weight[First_Broadcast_Neuron_Weight_Index];
        Delta_Neuron_Previous_Layer[i] = Delta_First_Neuron * First_Broadcast_Neuron_Weight;
    }

    for (unsigned int Layer_Index = W->Layer + 2; Layer_Index < LAYER_AMOUNT_EVAL - 2; Layer_Index++)
    {
        unsigned int Broadcast_Neuron_Index_Begin = LAYER_SIZE_INPUT_EVAL + LAYER_SIZE_HIDDEN_EVAL * (Layer_Index - 1);
        unsigned int Broadcast_Weight_Index_Begin = WEIGHT_AMOUNT_INPUT_LAYER_EVAL + WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL * (Layer_Index - 1);

        for (int Origin_Delta_Neuron_Index = 0; Origin_Delta_Neuron_Index < LAYER_SIZE_HIDDEN_EVAL; Origin_Delta_Neuron_Index++)
        {
            int Broadcast_Delta_Neuron_Index = Broadcast_Neuron_Index_Begin + Origin_Delta_Neuron_Index;
            float Broadcast_Delta_Neuron = Delta_Neuron_Previous_Layer[Origin_Delta_Neuron_Index];

            if (C->Eval_Network.Neuron[Broadcast_Delta_Neuron_Index] > 1.f || C->Eval_Network.Neuron[Broadcast_Delta_Neuron_Index] < -1.f)
            {
                Broadcast_Delta_Neuron *= NETWORK_ACTIVATION_SLOPE;
            }

            for (int Target_Delta_Neuron_Index = 0; Target_Delta_Neuron_Index < LAYER_SIZE_HIDDEN_EVAL; Target_Delta_Neuron_Index++)
            {
                int Broadcast_Weight_Index = Broadcast_Weight_Index_Begin + Origin_Delta_Neuron_Index * LAYER_SIZE_HIDDEN_EVAL + Target_Delta_Neuron_Index;
                float Broadcast_Weight = C->Weight[Broadcast_Weight_Index];

                Delta_Neuron_Next_Layer[Target_Delta_Neuron_Index] += Broadcast_Weight * Broadcast_Delta_Neuron;
            }
        }
    }

    float Delta_Output_Neuron = 0.f;

    // Calculate final delta output neuron value
    for (int i = 0; i < LAYER_SIZE_HIDDEN_EVAL; i++)
    {
        unsigned int Weight_Begin_Index = WEIGHT_AMOUNT_INPUT_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
        unsigned int Last_Bottle_Neuron_Weight_Index = Weight_Begin_Index + i;
        float Last_Bottle_Neuron_Weight = C->Weight[Last_Bottle_Neuron_Weight_Index];

        unsigned int Last_Bottle_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + i;
        float Neuron = C->Eval_Network.Neuron[Last_Bottle_Neuron_Index];

        if (Neuron > 1.f || Neuron < -1.f)
        {
            Last_Bottle_Neuron_Weight *= NETWORK_ACTIVATION_SLOPE;
        }

        float Bottle_Delta_Neuron = Delta_Neuron_Previous_Layer[i] * Last_Bottle_Neuron_Weight;

        Delta_Output_Neuron += Bottle_Delta_Neuron;
    }

    C->Eval_Network.Delta_Weight[W->Weight_Index] = Delta_Output_Neuron;
}

__global__ void Create_Neural_Net_Eval(CraftState* C)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Grid_Size = blockDim.x * gridDim.x;

    if (idx < LAYER_SIZE_INPUT_EVAL)
    {
        C->Eval_Network.Neuron[idx] = 1.f;
    }

    unsigned int Batch_Amount = (WEIGHT_AMOUNT_EVAL + Grid_Size - 1) / Grid_Size;
    
    for (unsigned int i = 0; i < Batch_Amount; i++)
    {
        unsigned int Weight_Index = Grid_Size * i + idx;
        if (Weight_Index < WEIGHT_AMOUNT_EVAL)
        {
            C->Eval_Network.Weight[Weight_Index] = 1.f;
        }
    }

}

__global__ void RELU_Activate_Layer(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int Layer_Size;
    unsigned int Layer_Begin_Index;

    if (Layer == 0)
    {
        Layer_Size = LAYER_SIZE_INPUT_EVAL;
        Layer_Begin_Index = 0;
    }
    else if (Layer == LAYER_AMOUNT_EVAL - 1)
    {
        Layer_Size = LAYER_SIZE_OUTPUT_EVAL;
        Layer_Begin_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
    }
    else
    {
        Layer_Size = LAYER_SIZE_HIDDEN_EVAL;
        Layer_Begin_Index = LAYER_SIZE_INPUT_EVAL + (Layer - 1) * LAYER_SIZE_HIDDEN_EVAL;
    }

    if (idx < Layer_Size)
    {
        RELU_Activate(C->Eval_Network.Neuron[Layer_Begin_Index + idx]);
    }
}

__host__ void Test_Neural_Net_Eval(CraftState* C)
{
    Create_Neural_Net_Eval<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(C);
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < LAYER_AMOUNT_EVAL; i++)
    {
        Run_Neural_Net_Eval<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(C, i);
        cudaDeviceSynchronize();
        RELU_Activate_Layer<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(C, i);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < LAYER_AMOUNT_EVAL; i++)
    {
        unsigned int Layer = i;
        std::cout << "Layer " << Layer << ": ";
        Print_Layer_Eval<<<1000, BLOCK_SIZE>>>(C, Layer);
        cudaDeviceSynchronize();
        std::cout << std::endl;
    }
}

__global__ void Print_Layer_Eval(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (Layer == 0)
    {
         if (idx < LAYER_SIZE_INPUT_EVAL)
         {
             printf("%9.2f, ", C->Eval_Network.Neuron[idx]);
         }
    }
    // Last Layer (Has 1 output neuron)
    else if (Layer == LAYER_AMOUNT_EVAL - 1)
    {
        if (idx < LAYER_SIZE_OUTPUT_EVAL)
        {
            unsigned int Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + idx;
            printf("%9.2f, ", C->Eval_Network.Neuron[Neuron_Index]);
        }
    }
    // Hidden Layers
    else
    {
        if (idx < LAYER_SIZE_HIDDEN_EVAL)
        {
            unsigned int Neuron_Index = LAYER_SIZE_INPUT_EVAL + LAYER_SIZE_HIDDEN_EVAL * (Layer - 1) + idx;
            printf("%9.2f, ", C->Eval_Network.Neuron[Neuron_Index]);
        }
    }
}

__global__ void Run_Neural_Net_Eval(CraftState* C, unsigned int Layer)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Grid_Size = blockDim.x * gridDim.x;

    // TODO Grid sync placeholder
    // cooperative_groups::grid_group grid = cooperative_groups::this_grid()

    // for (unsigned int Layer = 0; Layer < LAYER_AMOUNT_EVAL; Layer++)
    {
        if (Layer == 0)
        {
            unsigned int Iterations = (WEIGHT_AMOUNT_INPUT_LAYER_EVAL + Grid_Size - 1) / Grid_Size;

            for (unsigned int i = 0; i < Iterations; i++)
            {
                unsigned int Weight_Index = Grid_Size * i + idx;

                if (Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL)
                {
                    Run_Neural_Net_Layer_Eval(C, Weight_Index, true, Layer);
                }
            }
        }
        // Last Layer (Has 1 output neuron)
        else if (Layer == LAYER_AMOUNT_EVAL - 1)
        {
            unsigned int Iterations = (WEIGHT_AMOUNT_OUTPUT_LAYER_EVAL + Grid_Size - 1) / Grid_Size;

            for (unsigned int i = 0; i < Iterations; i++)
            {
                unsigned int Weight_Index = OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL + Grid_Size * i + idx;

                if (Weight_Index < WEIGHT_AMOUNT_EVAL)
                {
                    Run_Neural_Net_Layer_Eval(C, Weight_Index, true, Layer);

                    printf("Weight Index: %d\n", Weight_Index - OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL);
                    // if (idx == 0)
                    //     printf("Weight Amount Eval: %d\n", WEIGHT_AMOUNT_EVAL);
                }
            }
        }
        // Hidden Layers
        else
        {
            unsigned int Iterations = (WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL + Grid_Size - 1) / Grid_Size;

            for (unsigned int i = 0; i < Iterations; i++)
            {
                unsigned int Weight_Index = WEIGHT_AMOUNT_INPUT_LAYER_EVAL + WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL * (Layer - 1) + Grid_Size * i + idx;

                if (Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL + WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL * Layer && Weight_Index < OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL)
                {
                    Run_Neural_Net_Layer_Eval(C, Weight_Index, true, Layer);
                }
            }
        }

        // TODO: Actually implement when cooperative launch is setup
        // grid.sync();
        // TODO: Activate Layers
        // grid.sync();
    }
}

__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, const unsigned int &Weight_Index, const bool &Do_Activation, unsigned int Layer)
{
    neuron_Indices Neuron_Value = Get_Neuron_Indices(C, Weight_Index, Layer);

    float Weight = C->Eval_Network.Weight[Weight_Index];
    float Neuron = C->Eval_Network.Neuron[Neuron_Value.Origin_Neuron_Index];

    // if (Weight_Index ==  1206)
    // {
    //     printf("Input Size: %d, Idx: %d, Origin %d, Target %d\n", LAYER_SIZE_INPUT_EVAL, Weight_Index, Neuron_Value.Origin_Neuron_Index, Neuron_Value.Target_Neuron_Index);
    // }
    // if (Weight_Index ==  1207)
    // {
    //     printf("Input Size: %d, Idx: %d, Origin %d, Target %d\n", LAYER_SIZE_INPUT_EVAL, Weight_Index, Neuron_Value.Origin_Neuron_Index, Neuron_Value.Target_Neuron_Index);
    // }
    
    float Signal = Neuron * Weight;

    if (Weight_Index < 100)
    {
        printf("Signal: %f, TN: %d\n", Signal, Neuron_Value.Target_Neuron_Index);
    }

    atomicAdd(&C->Eval_Network.Neuron[Neuron_Value.Target_Neuron_Index], Signal);

    // TODO: Finish this
    // synchronize grid
}

// Must make sure Neuron_Index does not exceed neuron amount and is the proper subset
__device__ void Activate_Layer_Eval(CraftState* C, const unsigned int &Neuron_Index)
{
    RELU_Activate(C->Eval_Network.Neuron[Neuron_Index]);
}

// Get origin neuron index and target neuron index of each weight. Make sure the Weight_Index value is bounds checked before passed in 
__device__ neuron_Indices Get_Neuron_Indices(CraftState *C, const unsigned int &Weight_Index, unsigned int Layer)
{
    neuron_Indices Result;

    // First Layer
    if (Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL)
    {
        Result.Origin_Neuron_Index = Weight_Index / LAYER_SIZE_HIDDEN_EVAL;
        Result.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Weight_Index % LAYER_SIZE_HIDDEN_EVAL;
    }
    // Hidden Layers
    else if (Weight_Index < OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL)
    {
        unsigned int Weight_Index_Within_Layer = (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) % (WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL);
        unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / LAYER_SIZE_HIDDEN_EVAL;
        unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % LAYER_SIZE_HIDDEN_EVAL;

        unsigned int Layer_Index = 1 + (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) / WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
        Result.Origin_Neuron_Index = LAYER_SIZE_INPUT_EVAL + (Layer_Index - 1) * LAYER_SIZE_HIDDEN_EVAL + Origin_Neuron_Index_Within_Layer;
        Result.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Layer_Index * LAYER_SIZE_HIDDEN_EVAL + Target_Neuron_Index_Within_Layer;

        if (Layer == 1)
        {
            printf("#: %d, #N: %d, WI: %d, WL: %d, ON: %d, TN: %d, W: %8.3f\n", WEIGHT_AMOUNT_EVAL, NEURON_AMOUNT_EVAL, Weight_Index, Weight_Index_Within_Layer, Result.Origin_Neuron_Index, Result.Target_Neuron_Index, C->Eval_Network.Weight[Weight_Index]);
        }
    }
    // Last Layer
    else
    {
        unsigned int Weight_Index_Within_Layer = Weight_Index - OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL;

        unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / LAYER_SIZE_OUTPUT_EVAL;
        unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % LAYER_SIZE_OUTPUT_EVAL;

        Result.Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - LAYER_SIZE_HIDDEN_EVAL + Origin_Neuron_Index_Within_Layer;
        Result.Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + Target_Neuron_Index_Within_Layer;

        // printf("#: %d, #N: %d, WI: %d, WL: %d, ON: %d, TN: %d\n", WEIGHT_AMOUNT_EVAL, NEURON_AMOUNT_EVAL, Weight_Index, Weight_Index_Within_Layer, Result.Origin_Neuron_Index, Result.Target_Neuron_Index);
    }

    return Result;
}

/* TODO: Test if it's faster to have a pre-allocated array of the characteristic
of each weight and store it on global memory, or if it's faster to calculate each
weight characteristic each time the network is forward or back propagated.
The reasoning is that it takes 270 cycles to retrieve the data from global memory
versus perhaps a few dozen cycles to populate on stack and only a few cycles to retrieve it

Storing in global memory would be cleaner

Will Weight_Characteristic struct even fit on the registers of each thread
*/

__device__ void Populate_Weight_Data(CraftState* C, Weight_Characteristic_Global* WG, const unsigned int &Weight_Index)
{
    // Weight_Characteristic W;

    // W.Weight_Index = Weight_Index;

    // if (W.Weight_Index >= WEIGHT_AMOUNT_EVAL)
    //     return;

    // if (W.Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL)
    // {
    //     W.Layer = 0;
    //     W.Weight_Index_Within_Layer = Weight_Index;

    //     W.Origin_Neuron_Index_Within_Layer = Weight_Index / LAYER_SIZE_HIDDEN_EVAL;
    //     W.Target_Neuron_Index_Within_Layer = Weight_Index % LAYER_SIZE_HIDDEN_EVAL;

    //     W.Origin_Neuron_Index = W.Origin_Neuron_Index_Within_Layer;
    //     W.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + W.Target_Neuron_Index_Within_Layer;
    // }
    // else if (W.Weight_Index < WEIGHT_AMOUNT_INPUT_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL)
    // {
    //     W.Layer = 1 + (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) / (WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL);
    //     W.Weight_Index_Within_Layer = (Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL) % (WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL);

    //     W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / LAYER_SIZE_HIDDEN_EVAL;
    //     W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % LAYER_SIZE_HIDDEN_EVAL;

    //     W.Origin_Neuron_Index = LAYER_SIZE_INPUT_EVAL + (W.Layer - 1) * LAYER_SIZE_HIDDEN_EVAL + W.Origin_Neuron_Index_Within_Layer;
    //     W.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + W.Layer * LAYER_SIZE_HIDDEN_EVAL + W.Target_Neuron_Index_Within_Layer;
    // }
    // else
    // {
    //     W.Layer = LAYER_AMOUNT_EVAL - 1;
    //     W.Weight_Index_Within_Layer = W.Weight_Index - WEIGHT_AMOUNT_INPUT_LAYER_EVAL - (LAYER_AMOUNT_HIDDEN_EVAL - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;

    //     W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / LAYER_SIZE_OUTPUT_EVAL;
    //     W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % LAYER_SIZE_OUTPUT_EVAL;

    //     W.Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - LAYER_SIZE_HIDDEN_EVAL + W.Origin_Neuron_Index_Within_Layer;
    //     W.Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + W.Target_Neuron_Index_Within_Layer;
    // }

    // if (W.Layer != LAYER_AMOUNT_EVAL - 1)
    // {
    //     if (W.Layer != 0)    // Hidden Layer
    //     {
    //         if (W.Weight_Index_Within_Layer >= WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL)
    //         {
    //             return;
    //         }

    //         W.Neuron_Index_Layer_Begin = LAYER_SIZE_INPUT_EVAL + (W.Layer - 1) * LAYER_SIZE_HIDDEN_EVAL;
    //         W.Weight_Index_Layer_Begin = WEIGHT_AMOUNT_INPUT_LAYER_EVAL + (W.Layer - 1) * WEIGHT_AMOUNT_HIDDEN_LAYER_EVAL;
    //     }
    //     else    // Input Layer
    //     {
    //         if (W.Weight_Index_Within_Layer >= WEIGHT_AMOUNT_INPUT_LAYER_EVAL)
    //         {
    //             return;
    //         }

    //         W.Neuron_Index_Layer_Begin = 0;
    //         W.Weight_Index_Layer_Begin = 0;
    //     }

    //     W.Next_Layer_Size = LAYER_SIZE_HIDDEN_EVAL;
    // }
    // else        // Output Layer
    // {
    //     if (W.Weight_Index_Within_Layer >= LAYER_SIZE_OUTPUT_EVAL * LAYER_SIZE_HIDDEN_EVAL)
    //     {
    //         return;
    //     }

    //     W.Neuron_Index_Layer_Begin = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
    //     W.Weight_Index_Layer_Begin = OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL;
    //     W.Next_Layer_Size = LAYER_SIZE_OUTPUT_EVAL;
    // }

    // W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / W.Next_Layer_Size;
    // W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % W.Next_Layer_Size;

    // WG->Layer[W.Weight_Index]                            = W.Layer;
    // WG->Origin_Neuron_Index[W.Weight_Index]              = W.Origin_Neuron_Index;
    // WG->Target_Neuron_Index[W.Weight_Index]              = W.Target_Neuron_Index;
    // WG->Neuron_Index_Layer_Begin[W.Weight_Index]         = W.Neuron_Index_Layer_Begin;
    // WG->Origin_Neuron_Index_Within_Layer[W.Weight_Index] = W.Origin_Neuron_Index_Within_Layer;
    // WG->Target_Neuron_Index_Within_Layer[W.Weight_Index] = W.Target_Neuron_Index_Within_Layer;
    // WG->Weight_Index[W.Weight_Index]                     = W.Weight_Index;
    // WG->Weight_Index_Layer_Begin[W.Weight_Index]         = W.Weight_Index_Layer_Begin;
    // WG->Weight_Index_Within_Layer[W.Weight_Index]        = W.Weight_Index_Within_Layer;
    // WG->Next_Layer_Size[W.Weight_Index]                  = W.Next_Layer_Size;
}

__device__ void Populate_Delta_Neurons(CraftState* C, const unsigned int &Weight_Index)
{
    // neuron_Indices Neuron_Values = Get_Neuron_Indices(Weight_Index);


}

__device__ void Copy_Weight_Characteristics_From_Global(Weight_Characteristic_Global* WG, Weight_Characteristic *W)
{
    W->Layer                            = WG->Layer[W->Weight_Index];
    W->Origin_Neuron_Index              = WG->Origin_Neuron_Index[W->Weight_Index];
    W->Target_Neuron_Index              = WG->Target_Neuron_Index[W->Weight_Index];
    W->Neuron_Index_Layer_Begin         = WG->Neuron_Index_Layer_Begin[W->Weight_Index];
    W->Origin_Neuron_Index_Within_Layer = WG->Origin_Neuron_Index_Within_Layer[W->Weight_Index];
    W->Target_Neuron_Index_Within_Layer = WG->Target_Neuron_Index_Within_Layer[W->Weight_Index];
    W->Weight_Index                     = WG->Weight_Index[W->Weight_Index];
    W->Weight_Index_Layer_Begin         = WG->Weight_Index_Layer_Begin[W->Weight_Index];
    W->Weight_Index_Within_Layer        = WG->Weight_Index_Within_Layer[W->Weight_Index];
    W->Next_Layer_Size                  = WG->Next_Layer_Size[W->Weight_Index];
}
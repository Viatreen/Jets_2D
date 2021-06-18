


// File Header
#include "NeuralNet_Eval.h"

// Project Headers
#include "NeuralNet.h"

__device__ void BackPropagate_Eval(CraftState* C, Weight_Characteristic *W)
{
    // Make sure W->W->Weight_Index is not greater than or equal to weight amount

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
            = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + W->Target_Neuron_Index;
        float Final_Weight = C->Eval_Network.Weight[Final_Weight_Index];

        C->Eval_Network.Delta_Weight[W->Weight_Index] = Delta_First_Neuron * Final_Weight;
        return;
    }

    float Delta_Neuron_Previous_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];
    float Delta_Neuron_Next_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];

    // First broadcast from the target neuron
    for (unsigned int i = 0; i < NEURONS_PER_HIDDEN_LAYER_EVAL; i++)
    {
        unsigned int First_Broadcast_Neuron_Weight_Index = W->Weight_Index + NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (W->Target_Neuron_Index - W->Origin_Neuron_Index) * NEURONS_PER_HIDDEN_LAYER_EVAL - W->Target_Neuron_Index + i;

        float First_Broadcast_Neuron_Weight = C->Eval_Network.Weight[First_Broadcast_Neuron_Weight_Index];
        Delta_Neuron_Previous_Layer[i] = Delta_First_Neuron * First_Broadcast_Neuron_Weight;
    }

    for (unsigned int Layer_Index = W->Layer + 2; Layer_Index < LAYER_AMOUNT_EVAL - 2; Layer_Index++)
    {
        unsigned int Broadcast_Neuron_Index_Begin = LAYER_SIZE_INPUT_EVAL + NEURONS_PER_HIDDEN_LAYER_EVAL * (Layer_Index - 1);
        unsigned int Broadcast_Weight_Index_Begin = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL * (Layer_Index - 1);

        for (int Origin_Delta_Neuron_Index = 0; Origin_Delta_Neuron_Index < NEURONS_PER_HIDDEN_LAYER_EVAL; Origin_Delta_Neuron_Index++)
        {
            int Broadcast_Delta_Neuron_Index = Broadcast_Neuron_Index_Begin + Origin_Delta_Neuron_Index;
            float Broadcast_Delta_Neuron = Delta_Neuron_Previous_Layer[Origin_Delta_Neuron_Index];

            if (C->Eval_Network.Neuron[Broadcast_Delta_Neuron_Index] > 1.f || C->Eval_Network.Neuron[Broadcast_Delta_Neuron_Index] < -1.f)
            {
                Broadcast_Delta_Neuron *= NETWORK_ACTIVATION_SLOPE;
            }

            for (int Target_Delta_Neuron_Index = 0; Target_Delta_Neuron_Index < NEURONS_PER_HIDDEN_LAYER_EVAL; Target_Delta_Neuron_Index++)
            {
                int Broadcast_Weight_Index = Broadcast_Weight_Index_Begin + Origin_Delta_Neuron_Index * NEURONS_PER_HIDDEN_LAYER_EVAL + Target_Delta_Neuron_Index;
                float Broadcast_Weight = C->Weight[Broadcast_Weight_Index];

                Delta_Neuron_Next_Layer[Target_Delta_Neuron_Index] += Broadcast_Weight * Broadcast_Delta_Neuron;
            }
        }
    }

    float Delta_Output_Neuron = 0.f;

    // Calculate final delta output neuron value
    for (int i = 0; i < NEURONS_PER_HIDDEN_LAYER_EVAL; i++)
    {
        unsigned int Weight_Begin_Index = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;
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

__device__ void Run_Neural_Net_Eval()
{

}

/* TODO: Test if it's faster to have a pre-allocated array of the characteristic
of each weight and store it on global memory, or if it's faster to calculate each
weight characteristic each time the network is forward or back propagated.
The reasoning is that it takes 270 cycles to retrieve the data from global memory
versus perhaps a few dozen cycles to populate on stack and only a few cycles to retrieve it

Storing in global memory would be cleaner

Will Weight_Characteristic struct even fit on the registers of each thread
*/

__device__ void Populate_Weight_Data(CraftState* C, Weight_Characteristic_Global* WG, unsigned int Weight_Index)
{
    Weight_Characteristic W;

    W.Weight_Index = Weight_Index;

    if (W.Weight_Index >= WEIGHT_COUNT_EVAL)
        return;

    if (W.Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
    {
        W.Layer = 0;
        W.Weight_Index_Within_Layer = Weight_Index;

        W.Origin_Neuron_Index_Within_Layer = Weight_Index / NEURONS_PER_HIDDEN_LAYER_EVAL;
        W.Target_Neuron_Index_Within_Layer = Weight_Index % NEURONS_PER_HIDDEN_LAYER_EVAL;

        W.Origin_Neuron_Index = W.Origin_Neuron_Index_Within_Layer;
        W.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + W.Target_Neuron_Index_Within_Layer;
    }
    else if (W.Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
    {
        W.Layer = 1 + (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) / (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);
        W.Weight_Index_Within_Layer = (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) % (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);

        W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / NEURONS_PER_HIDDEN_LAYER_EVAL;
        W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % NEURONS_PER_HIDDEN_LAYER_EVAL;

        W.Origin_Neuron_Index = LAYER_SIZE_INPUT_EVAL + (W.Layer - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL + W.Origin_Neuron_Index_Within_Layer;
        W.Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + W.Layer * NEURONS_PER_HIDDEN_LAYER_EVAL + W.Target_Neuron_Index_Within_Layer;
    }
    else
    {
        W.Layer = LAYER_AMOUNT_EVAL - 1;
        W.Weight_Index_Within_Layer = W.Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL - (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;

        W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / LAYER_SIZE_OUTPUT_EVAL;
        W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % LAYER_SIZE_OUTPUT_EVAL;

        W.Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - NEURONS_PER_HIDDEN_LAYER_EVAL + W.Origin_Neuron_Index_Within_Layer;
        W.Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + W.Target_Neuron_Index_Within_Layer;
    }

    if (W.Layer != LAYER_AMOUNT_EVAL - 1)
    {
        if (W.Layer != 0)    // Hidden Layer
        {
            if (W.Weight_Index_Within_Layer >= NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
            {
                return;
            }

            W.Neuron_Index_Layer_Begin = LAYER_SIZE_INPUT_EVAL + (W.Layer - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL;
            W.Weight_Index_Layer_Begin = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (W.Layer - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;
        }
        else    // Input Layer
        {
            if (W.Weight_Index_Within_Layer >= LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
            {
                return;
            }

            W.Neuron_Index_Layer_Begin = 0;
            W.Weight_Index_Layer_Begin = 0;
        }

        W.Next_Layer_Size = NEURONS_PER_HIDDEN_LAYER_EVAL;
    }
    else        // Output Layer
    {
        if (W.Weight_Index_Within_Layer >= LAYER_SIZE_OUTPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
        {
            return;
        }

        W.Neuron_Index_Layer_Begin = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
        W.Weight_Index_Layer_Begin = OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL;
        W.Next_Layer_Size = LAYER_SIZE_OUTPUT_EVAL;
    }

    W.Origin_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer / W.Next_Layer_Size;
    W.Target_Neuron_Index_Within_Layer = W.Weight_Index_Within_Layer % W.Next_Layer_Size;

    WG->Layer[W.Weight_Index]                            = W.Layer;
    WG->Origin_Neuron_Index[W.Weight_Index]              = W.Origin_Neuron_Index;
    WG->Target_Neuron_Index[W.Weight_Index]              = W.Target_Neuron_Index;
    WG->Neuron_Index_Layer_Begin[W.Weight_Index]         = W.Neuron_Index_Layer_Begin;
    WG->Origin_Neuron_Index_Within_Layer[W.Weight_Index] = W.Origin_Neuron_Index_Within_Layer;
    WG->Target_Neuron_Index_Within_Layer[W.Weight_Index] = W.Target_Neuron_Index_Within_Layer;
    WG->Weight_Index[W.Weight_Index]                     = W.Weight_Index;
    WG->Weight_Index_Layer_Begin[W.Weight_Index]         = W.Weight_Index_Layer_Begin;
    WG->Weight_Index_Within_Layer[W.Weight_Index]        = W.Weight_Index_Within_Layer;
    WG->Next_Layer_Size[W.Weight_Index]                  = W.Next_Layer_Size;
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

__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, Weight_Characteristic *W, bool Do_Activation)
{
    float Neuron = C->Eval_Network.Neuron[W->Origin_Neuron_Index];
    float Weight = C->Eval_Network.Weight[W->Weight_Index];

    float Signal = Neuron * Weight;

    // TODO: Finish this
    // atomicAdd(&C->Eval_Network.Neuron[])
}

// Must make sure Neuron_Index does not exceed neuron amount and is the proper subset
__device__ void Activate_Layer_Eval(CraftState* C, unsigned int Neuron_Index)
{
    RELU_Activate(C->Eval_Network.Neuron[Neuron_Index]);
}
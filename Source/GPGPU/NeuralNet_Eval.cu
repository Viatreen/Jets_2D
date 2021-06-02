


// File Header
#include "NeuralNet_Eval.h"

// Project Headers
#include "NeuralNet.h"

__device__ void BackPropagate_Eval(CraftState* C, unsigned int Weight_Index)
{
	if (Weight_Index >= WEIGHT_COUNT_EVAL)
		return;

	// Neuron index definition of Weight Index
	// There is only one output neuron for the evaluation network
	unsigned int Layer;
	unsigned int Origin_Neuron_Index;
	unsigned int Target_Neuron_Index;
	{
		unsigned int Origin_Neuron_Index_Within_Layer;
		unsigned int Target_Neuron_Index_Within_Layer;
		unsigned int Weight_Index_Within_Layer;

		if (Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
		{
			Layer = 0;
			Weight_Index_Within_Layer = Weight_Index;

			Origin_Neuron_Index_Within_Layer = Weight_Index / NEURONS_PER_HIDDEN_LAYER_EVAL;
			Target_Neuron_Index_Within_Layer = Weight_Index % NEURONS_PER_HIDDEN_LAYER_EVAL;

			Origin_Neuron_Index = Origin_Neuron_Index_Within_Layer;
			Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Target_Neuron_Index_Within_Layer;
		}
		else if (Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
		{
			Layer = 1 + (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) / (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);
			Weight_Index_Within_Layer = (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) % (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);

			Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / NEURONS_PER_HIDDEN_LAYER_EVAL;
			Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % NEURONS_PER_HIDDEN_LAYER_EVAL;

			Origin_Neuron_Index = LAYER_SIZE_INPUT_EVAL + (Layer - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL + Origin_Neuron_Index_Within_Layer;
			Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Layer * NEURONS_PER_HIDDEN_LAYER_EVAL + Target_Neuron_Index_Within_Layer;
		}
		else
		{
			Layer = LAYER_AMOUNT_EVAL - 1;
			Weight_Index_Within_Layer = Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL - (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;

			Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / LAYER_SIZE_OUTPUT_EVAL;
			Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % LAYER_SIZE_OUTPUT_EVAL;

			Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - NEURONS_PER_HIDDEN_LAYER_EVAL + Origin_Neuron_Index_Within_Layer;
			Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + Target_Neuron_Index_Within_Layer;
		}
	}

	float Origin_Neuron = C->Eval_Network.Neuron[Origin_Neuron_Index];
	float Weight = C->Eval_Network.Weight[Weight_Index];

	float Delta_First_Neuron = Origin_Neuron * Weight;

	if (Layer == LAYER_AMOUNT_EVAL - 1)
	{
		C->Eval_Network.Delta_Weight[Weight_Index] = Delta_First_Neuron;
		return;
	}

	float Target_Neuron = C->Eval_Network.Neuron[Target_Neuron_Index];

	if (Target_Neuron > 1.f || Target_Neuron < -1.f)
	{
		Delta_First_Neuron *= NETWORK_ACTIVATION_SLOPE;
	}

	if (Layer == LAYER_AMOUNT_EVAL - 2)
	{
		unsigned int Final_Weight_Index
			= LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + Target_Neuron_Index;
		float Final_Weight = C->Eval_Network.Weight[Final_Weight_Index];

		C->Eval_Network.Delta_Weight[Weight_Index] = Delta_First_Neuron * Final_Weight;
		return;
	}

	float Delta_Neuron_Previous_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];
	float Delta_Neuron_Next_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];

	// First broadcast from the target neuron
	for (unsigned int i = 0; i < NEURONS_PER_HIDDEN_LAYER_EVAL; i++)
	{
		unsigned int First_Broadcast_Neuron_Weight_Index = Weight_Index + NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (Target_Neuron_Index - Origin_Neuron_Index) * NEURONS_PER_HIDDEN_LAYER_EVAL - Target_Neuron_Index + i;

		float First_Broadcast_Neuron_Weight = C->Eval_Network.Weight[First_Broadcast_Neuron_Weight_Index];
		Delta_Neuron_Previous_Layer[i] = Delta_First_Neuron * First_Broadcast_Neuron_Weight;
	}

	for (unsigned int Layer_Index = Layer + 2; Layer_Index < LAYER_AMOUNT_EVAL - 2; Layer_Index++)
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

	C->Eval_Network.Delta_Weight[Weight_Index] = Delta_Output_Neuron;
}

__host__ void Run_Neural_Net_Eval(Craftstate* C, bool Do_Activation)
{
	
}

__device__ void Run_Neural_Net_Layer_Eval(CraftState* C, bool Do_Activation, unsigned int Weight_Index_Within_Layer, unsigned int Layer_Index)
{
	unsigned int Next_Layer_Size;
	unsigned int Weight_Index_Layer_Begin;
	unsigned int Neuron_Index_Layer_Begin;

	if (Layer_Index != LAYER_AMOUNT_EVAL - 1)
	{
		if (Layer_Index != 0)	// Hidden Layer
		{
			if (Weight_Index_Within_Layer >= NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
			{
				return;
			}

			Neuron_Index_Layer_Begin = LAYER_SIZE_INPUT_EVAL + (Layer_Index - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL;
			Weight_Index_Layer_Begin = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (Layer_Index - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;
		}
		else	// Input Layer
		{
			if (Weight_Index_Within_Layer >= LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
			{
				return;
			}

			Neuron_Index_Layer_Begin = 0;
			Weight_Index_Layer_Begin = 0;
		}

		Next_Layer_Size = NEURONS_PER_HIDDEN_LAYER_EVAL;
	}
	else		// Output Layer
	{
		if (Weight_Index_Within_Layer >= LAYER_SIZE_OUTPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
		{
			return;
		}

		Neuron_Index_Layer_Begin = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
		Weight_Index_Layer_Begin = OUTPUT_LAYER_WEIGHT_BEGIN_IDX_EVAL;
		Next_Layer_Size = LAYER_SIZE_OUTPUT_EVAL;
	}

	unsigned int Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / Next_Layer_Size;
	unsigned int Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % Next_Layer_Size;

	float Neuron = C->Eval_Network.Neuron[Neuron_Index_Layer_Begin + Origin_Neuron_Index_Within_Layer];
	float Weight = C->Eval_Network.Weight[Weight_Index_Layer_Begin + Weight_Index_Within_Layer];

	float Signal = Neuron * Weight;

	unsigned int Target_Neuron_Index = Neuron_Index_Layer_Begin + Next_Layer_Size
	atomicAdd(&C->Eval_Network.Neuron[])
}

__device__ void Activate_Layer_Eval(CraftState* C, unsigned int Layer_Index, unsigned int Neuron_Index_Within_Layer)
{
	unsigned int Neuron_Layer_Begin_Index;

	if (Layer_Index == 0)
	{
		if (Neuron_Index >= INPUT_LAYER_SIZE_EVAL)
		{
			return;
		}

		Neuron_Layer_Begin_Index = 0;
	}
	else if (Layer_Index < LAYER_AMOUNT_EVAL - 1)
	{
		if (Neuron_Index >= NEURONS_PER_HIDDEN_LAYER_EVAL)
		{
			return;
		}

		Neuron_Layer_Begin_Index = LAYER_SIZE_INPUT_EVAL + NEURONS_PER_HIDDEN_LAYER_EVAL * (Layer_Index - 1);
	}
	else  // Output layer
	{
		if (Neuron_Index >= LAYER_SIZE_OUTPUT_EVAL)
		{
			return;
		}

		Neuron_Layer_Begin_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL;
	}

	RELU_Activate(C->Eval_Network.Neuron[Neuron_Layer_Begin_Index + Neuron_Index]);

}
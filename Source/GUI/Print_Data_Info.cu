// Standard Library
#include <iostream>

// File Header
#include "GUI/Print_Data_Info.h"

// Project Headers
#include "Config.h"
#include "GPGPU/Match.h"
#include "GPGPU/State.h"

void Print_Data_Info()
{
	std::cout << "Number of crafts: " << CRAFT_COUNT << std::endl;
	// Output estimated memory usage for non-neural network data
	size_t StateSize = (sizeof(CraftState) - sizeof(float) * WEIGHT_COUNT - 2 * ((NEURON_COUNT + 1 + 1) * sizeof(float)) - sizeof(eval_Network)) * 2 / 1024 / 1024;
	std::cout << "Size of state: " << StateSize << " MB" << std::endl;

	//unsigned long long StateHistorySize = StateSize * FRAMERATE_PHYSICS * int(TIME_MATCH);
	std::cout << "GPU memory required: " << (sizeof(CraftState) + sizeof(temp) + sizeof(MatchState) + sizeof(eval_Network)) / 1024 / 1024 << " MB" << std::endl;

	std::cout << std::endl << "Policy Network:" << std::endl;
	std::cout << "Number of layers: " << LAYER_AMOUNT << std::endl;
	std::cout << "Number of input neurons: " << LAYER_SIZE_INPUT << std::endl;
	std::cout << "Number of neurons per hidden layer: " << NEURONS_PER_HIDDEN_LAYER << std::endl;
	std::cout << "Number of output neurons: " << LAYER_SIZE_OUTPUT << std::endl;
	std::cout << "Number of neurons: " << NEURON_COUNT << std::endl;
	std::cout << "Number of weights: " << WEIGHT_COUNT << std::endl << std::endl;

	std::cout << "Evaluation Network:" << std::endl;
	std::cout << "Number of layers: " << LAYER_AMOUNT_EVAL << std::endl;
	std::cout << "Number of input neurons: " << LAYER_SIZE_INPUT_EVAL << std::endl;
	std::cout << "Number of neurons per hidden layer: " << NEURONS_PER_HIDDEN_LAYER_EVAL << std::endl;
	std::cout << "Number of output neurons: " << LAYER_SIZE_OUTPUT_EVAL << std::endl;
	std::cout << "Number of neurons: " << NEURON_COUNT_EVAL << std::endl;
	std::cout << "Number of weights: " << WEIGHT_COUNT_EVAL << std::endl << std::endl;

}
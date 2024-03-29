// Standard Library
#include <iostream>

// File Header
#include "Jets_2D/GUI/Print_Data_Info.hpp"

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/GPGPU/Match.hpp"
#include "Jets_2D/GPGPU/State.hpp"

void Print_Data_Info()
{
	std::cout << "Number of crafts: " << CRAFT_COUNT << std::endl;
	// Output estimated memory usage for non-neural network data
	size_t StateSize = (sizeof(CraftState) - sizeof(float) * WEIGHT_AMOUNT - 2 * ((NEURON_AMOUNT + 1 + 1) * sizeof(float)) - sizeof(eval_Network)) * 2 / 1024 / 1024;
	std::cout << "Size of state: " << StateSize << " MB" << std::endl;

	//unsigned long long StateHistorySize = StateSize * FRAMERATE_PHYSICS * int(TIME_MATCH);
	std::cout << "GPU memory required: " << (sizeof(CraftState) + sizeof(temp) + sizeof(MatchState) + sizeof(eval_Network)) / 1024 / 1024 << " MB" << std::endl;

	std::cout << std::endl << "Policy Network:" << std::endl;
	std::cout << "Number of layers: " << LAYER_AMOUNT << std::endl;
	std::cout << "Number of input neurons: " << LAYER_SIZE_INPUT << std::endl;
	std::cout << "Number of neurons per hidden layer: " << LAYER_SIZE_HIDDEN << std::endl;
	std::cout << "Number of output neurons: " << LAYER_SIZE_OUTPUT << std::endl;
	std::cout << "Number of neurons: " << NEURON_AMOUNT << std::endl;
	std::cout << "Number of weights: " << WEIGHT_AMOUNT << std::endl << std::endl;

	std::cout << "Evaluation Network:" << std::endl;
	std::cout << "Number of layers: " << LAYER_AMOUNT_EVAL << std::endl;
	std::cout << "Number of input neurons: " << NEURON_AMOUNT_INPUT_EVAL << std::endl;
	std::cout << "Number of neurons per hidden layer: " << NEURON_AMOUNT_HIDDEN_EVAL << std::endl;
	std::cout << "Number of output neurons: " << NEURON_AMOUNT_OUTPUT_EVAL << std::endl;
	std::cout << "Number of neurons: " << NEURON_AMOUNT_EVAL << std::endl;
	std::cout << "Number of weights: " << WEIGHT_AMOUNT_EVAL << std::endl << std::endl;

}
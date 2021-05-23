// Standard Library
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdint.h>
#include <stdlib.h>
#include <istream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdlib.h>

//#pragma message("hello this is a message")
//
//#ifdef _WIN32
//#include <minwindef.h>	// Removes glad APIENTRY redefine warning
//#pragma message("WIN32 defined")
//#else
//#pragma message("WIN32 not defined")
//#endif

#define GLFW_INCLUDE_NONE

// OpenGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
 
// ImGUI
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// Project Headers
#include "ErrorCheck.h"
#include "GPGPU/GPU_Error_Check.h"
#include "GL/GLSetup.h"
#include "GPGPU/Round.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/MapVertexBuffer.h"
#include "GPGPU/Match.h"
#include "GPGPU/State.h"
#include "GUI/GUI.h"
#include "Graphics/GrSetup.h"

float PrintWeights[CRAFT_COUNT * WEIGHT_COUNT];

int main()
{
	std::cout << "Begin" << std::endl;
	// Setup
	Timer = std::chrono::steady_clock::now();
	GL::Setup();
	Mem::Setup();
	Setup();
	Graphics::Setup();
	Init<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
	cudaCheck(cudaDeviceSynchronize());

	std::cout << "Number of crafts: " << CRAFT_COUNT << std::endl;
	std::cout << "Number of weights: " << WEIGHT_COUNT << std::endl;
	// Output estimated memory usage for backpropagating advantage function
	size_t StateSize = (sizeof(CraftState) - sizeof(float) * WEIGHT_COUNT - 2 * ((NEURON_COUNT + 1 + 1) * sizeof(float) - sizeof(curandState))) *  2 / 1024 / 1024;
	std::cout << "Size of state: " << StateSize << " MB" << std::endl;

	//unsigned long long StateHistorySize = StateSize * FRAMERATE_PHYSICS * int(TIME_MATCH);
	std::cout << "GPU memory required: " << ( sizeof(CraftState) + sizeof(temp) + sizeof(MatchState) ) / 1024 / 1024 << " MB" << std::endl;

	std::cout << "Number of Layers: " << LAYER_AMOUNT << std::endl;
	std::cout << "Number of Neurons: " << NEURON_COUNT << std::endl;
	std::cout << "Input neuron amount: " << LAYER_SIZE_INPUT << std::endl;
	std::cout << "Neurons per hidden layer: " << NEURONS_PER_HIDDEN_LAYER << std::endl;
	std::cout << "Output neuron amount: " << LAYER_SIZE_OUTPUT << std::endl;

	TimerStartup = float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count()) / 1000.f;

	// Game Loop
	while (!glfwWindowShouldClose(window))
	{
		int h_TournamentEpochNumber = 0;
		cudaCheck(cudaMemcpy(&Match->TournamentEpochNumber, &h_TournamentEpochNumber, sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaDeviceSynchronize());

		// Original Side of the Circle
		Round();

		std::cout << "Round " << RoundNumber;
		RoundAssignPlace<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		RoundPrintFirstPlace<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());
		
		RoundEnd();

		h_Config->RoundNumber = RoundNumber;
		SyncConfigArray();
		
		WeightsAndIDTempSave<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
		cudaCheck(cudaDeviceSynchronize());

		WeightsAndIDTransfer<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
		cudaCheck(cudaDeviceSynchronize());

		WeightsMutate<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
		cudaCheck(cudaDeviceSynchronize());

		IDAssign<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
		cudaCheck(cudaDeviceSynchronize());

		ResetScoreCumulative<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		RoundEnd2();
	}

	// Cleanup
	Mem::Shutdown();
	Shutdown();
	Graphics::Shutdown();

	std::cout << "End" << std::endl;

	return 0;
}
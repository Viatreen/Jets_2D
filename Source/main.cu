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

#ifdef _WIN32
#include <Windows.h>		// Removes glad APIENTRY redefine warning
#endif

// OpenGL
#define GLFW_INCLUDE_NONE
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
#include "GPGPU/GPErrorCheck.h"
#include "GL/GLSetup.h"
#include "GPGPU/Round.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/MapVertexBuffer.h"
#include "GPGPU/Match.h"
#include "GPGPU/NeuralNet_Eval.h"
#include "GPGPU/State.h"
#include "GUI/GUI.h"
#include "GUI/Print_Data_Info.h"
#include "Graphics/GrSetup.h"

int main()
{
	std::cout << "Begin" << std::endl;
	// Setup
	Timer = std::chrono::steady_clock::now();
	GL::Setup();
	Mem::Setup();
	Setup();
	Graphics::Setup();
	// TODO: Fix this function
	// Init<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
	cudaCheck(cudaDeviceSynchronize());

	// Print_Data_Info();
	Test_Neural_Net_Eval(Crafts);
	return 0;

	TimerStartup = float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count()) / 1000.f;

	// Game Loop
	while (!glfwWindowShouldClose(window))
	{
		int h_TournamentEpochNumber = 0;
		cudaCheck(cudaMemcpy(&Match->TournamentEpochNumber, &h_TournamentEpochNumber, sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaDeviceSynchronize());

		// Original Side of the Circle
		Round();

		//std::cout << "Round " << RoundNumber;
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

		return 0;
	}

	// Cleanup
	Mem::Shutdown();
	Shutdown();
	Graphics::Shutdown();

	std::cout << "End" << std::endl;

	return 0;
}
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
#include "GL/GLSetup.h"
#include "GPGPU/Round.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/MapVertexBuffer.h"
#include "GPGPU/Match.h"
#include "GPGPU/State.h"
#include "GUI/GUI.h"

int main()
{
	// Startup
	Timer = std::chrono::steady_clock::now();
	GL::Setup();
	Mem::Setup();
	Setup();
	Graphics::Setup();
	Init<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
	cudaCheck(cudaDeviceSynchronize());

	// Output estimated memory usage for backpropagating advantage function
	unsigned long long StateSize = (sizeof(CraftState) - sizeof(float) * WEIGHT_COUNT - 2 * ((NEURON_COUNT + 1 + 1) * sizeof(float) - sizeof(curandState))) *  2 / 1024 / 1024;
	std::cout << "Size of state: " << StateSize << " MB" << std::endl;

	unsigned long long AdvantageSize = StateSize * FRAMERATE_PHYSICS * int(TIME_MATCH);
	std::cout << "GPU memory required: " << AdvantageSize << " MB" << std::endl;

	std::cout << "Number of Layers: " << LAYER_AMOUNT << std::endl;
	std::cout << "Number of Neurons: " << NEURON_COUNT << std::endl;

	TimerStartup = float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count()) / 1000.f;

	// Game Loop
	while (!glfwWindowShouldClose(window))
	{
		int h_TournamentEpochNumber = 0;
		cudaCheck(cudaMemcpy(&Match->TournamentEpochNumber, &h_TournamentEpochNumber, sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaDeviceSynchronize());

		// Original Side of the Circle
		for (int i = 0; i < OPPONENT_COUNT && !glfwWindowShouldClose(window); i++)
			Round();

		RoundAssignPlace<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		// TODO: Build 1 kernel from these
		for (int i = 0; i < 10; i++)
		{
			RoundTieFix<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
			cudaCheck(cudaDeviceSynchronize());
		}
		
		RoundEnd();

		h_Config->RoundNumber = RoundNumber;
		SyncConfigArray();

		IDAssign<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
		cudaCheck(cudaDeviceSynchronize());

		IDTempSave<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		IDTransfer<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		int ScoreCumulative[CRAFT_COUNT];
		cudaCheck(cudaMemcpy(ScoreCumulative, Crafts->ScoreCumulative, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

		int ID[CRAFT_COUNT];
		cudaCheck(cudaMemcpy(ID, Crafts->ID, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

		int Place[CRAFT_COUNT];
		cudaCheck(cudaMemcpy(Place, Crafts->Place, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

		float FirstWeight1[CRAFT_COUNT];
		for (int i = 0; i < CRAFT_COUNT; i++)
			cudaCheck(cudaMemcpy(&FirstWeight1[i], &Crafts->Weights[i], sizeof(float), cudaMemcpyDeviceToHost));

		float ThirtiethWeight1[CRAFT_COUNT];
		for (int i = 0; i < CRAFT_COUNT; i++)
			cudaCheck(cudaMemcpy(&ThirtiethWeight1[i], &Crafts->Weights[29 * WEIGHT_COUNT + i], sizeof(float), cudaMemcpyDeviceToHost));
		
		WeightsTempSave<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
		cudaCheck(cudaDeviceSynchronize());

		WeightsTransfer<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
		cudaCheck(cudaDeviceSynchronize());

		WeightsMutate<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
		cudaCheck(cudaDeviceSynchronize());

		ScoreTempSave<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		ScoreTransfer<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		int ScoreCumulative2[CRAFT_COUNT];
		cudaCheck(cudaMemcpy(ScoreCumulative2, Crafts->ScoreCumulative, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

		float FirstWeight2[CRAFT_COUNT];
		for (int i = 0; i < CRAFT_COUNT; i++)
			cudaCheck(cudaMemcpy(&FirstWeight2[i], &Crafts->Weights[i], sizeof(float), cudaMemcpyDeviceToHost));

		float ThirtiethWeight2[CRAFT_COUNT];
		for (int i = 0; i < CRAFT_COUNT; i++)
			cudaCheck(cudaMemcpy(&ThirtiethWeight2[i], &Crafts->Weights[29 * WEIGHT_COUNT + i], sizeof(float), cudaMemcpyDeviceToHost));

		std::cout << "Results" << std::endl;
		for (int i = 0; i < CRAFT_COUNT; i++)
			std::cout << std::setw(3) << i << " ID: " << std::setw(5) << ID[i] 
			<< " Score: " << std::setw(4) << ScoreCumulative[i] 
			<< " Place: " << std::setw(3) << Place[i] 
			<< " 1st Weight: " << std::setw(12) << FirstWeight1[i]
			<< " 30th Weight: " << std::setw(12) << ThirtiethWeight1[i]
			<< " Sorted:: Score: " << std::setw(5) << ScoreCumulative2[i]
			<< " 1st Weight: " << std::setw(12) << FirstWeight2[i]
			<< " 30th Weight: " << std::setw(12) << ThirtiethWeight2[i] << std::endl;

		/*std::cout << "Score" << std::endl;
		for (int i = 0; i < CRAFT_COUNT; i++)
			std::cout << std::setw(3) << i << " Score: " << std::setw(5) << ScoreCumulative2[i] << std::endl;*/

		ResetScoreCumulative<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
		cudaCheck(cudaDeviceSynchronize());

		RoundEnd2();
	}

	// Cleanup
	Mem::Shutdown();
	Shutdown();
	Graphics::Shutdown();

	return 0;
}
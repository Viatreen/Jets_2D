// Standard Library
#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <Windows.h>        // Removes glad APIENTRY redefine warning
#endif

// OpenGL
#define GLFW_INCLUDE_NONE
#include "glad/glad.h"
#include "GLFW/glfw3.h"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
// Project Headers
#include "Jets_2D/ErrorCheck.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"
#include "Jets_2D/GPGPU/Round.hpp"
#include "Jets_2D/GPGPU/GPSetup.hpp"
#include "Jets_2D/GPGPU/Launcher.hpp"
#include "Jets_2D/GPGPU/MapVertexBuffer.hpp"
#include "Jets_2D/GPGPU/Match.hpp"
#include "Jets_2D/GPGPU/NeuralNet_Eval.hpp"
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GL/GLSetup.hpp"
#include "Jets_2D/Graphics/GrSetup.hpp"
#include "Jets_2D/GUI/GUI.hpp"
#include "Jets_2D/GUI/Print_Data_Info.hpp"

// Tests
#include "Tests/GPGPU/NeuralNet_Eval.test.hpp"

int main()
{
    std::cout << "Begin" << std::endl;

    Print_Data_Info();

    // Setup
    Timer = std::chrono::steady_clock::now();
    GL::Setup();
    Mem::Setup();
    GUI::Setup();
    Graphics::Setup();
    Init<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
    cudaCheck(cudaDeviceSynchronize());

    Test_Neural_Net_Eval(Crafts);

    GUI::TimerStartup = float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count()) / 1000.f;

    bool Do_Mutations = true;

    glfwSetWindowShouldClose(window, 1);

    // Game Loop
    while (!glfwWindowShouldClose(window))
    {
        int h_TournamentEpochNumber = 0;
        cudaCheck(cudaMemcpy(&Match->TournamentEpochNumber, &h_TournamentEpochNumber, sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());

        // Original Side of the Circle
        Round();

        RoundAssignPlace<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
        cudaCheck(cudaDeviceSynchronize());

        RoundPrintFirstPlace<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, GUI::RoundNumber);
        cudaCheck(cudaDeviceSynchronize());

        GUI::RoundEnd();

        h_Config->RoundNumber = GUI::RoundNumber;
        SyncConfigArray();
        
        if (Do_Mutations)
        {
            WeightsAndIDTempSave<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
            cudaCheck(cudaDeviceSynchronize());

            WeightsAndIDTransfer<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, Temp);
            cudaCheck(cudaDeviceSynchronize());

            WeightsMutate<<<FIT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
            cudaCheck(cudaDeviceSynchronize());

            IDAssign<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts, d_Config);
            cudaCheck(cudaDeviceSynchronize());
        }

        ResetScoreCumulative<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
        cudaCheck(cudaDeviceSynchronize());

        GUI::RoundEnd2();
    }

    // Cleanup
    Mem::Shutdown();
    // GUI::Shutdown();
    Graphics::Shutdown();

    std::cout << "End" << std::endl;

    return 0;
}

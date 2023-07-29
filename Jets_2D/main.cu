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
#include "Jets_2D/GL/GLSetup.hpp"
#include "Jets_2D/GPGPU/Round.hpp"
#include "Jets_2D/GPGPU/GPSetup.hpp"
#include "Jets_2D/GPGPU/MapVertexBuffer.hpp"
#include "Jets_2D/GPGPU/Match.hpp"
#include "Jets_2D/GPGPU/NeuralNet_Eval.hpp"
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GUI/GUI.hpp"
#include "Jets_2D/GUI/Print_Data_Info.hpp"
#include "Jets_2D/Graphics/GrSetup.hpp"

extern bool exit_round;

int main()
{
    std::cout << "Begin" << std::endl;
    Print_Data_Info();

    // Setup
    Timer = std::chrono::steady_clock::now();
    std::cout << "Setting up OpenGL" << std::endl;
    GL::Setup();
    std::cout << "Allocating memory" << std::endl;
    Mem::Setup();
    std::cout << "Setting up graphical user interface" << std::endl;
    GUI::Setup();
    std::cout << "Setting up graphics objects" << std::endl;
    Graphics::Setup();
    std::cout << "Initializing craft state" << std::endl;
    Init<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
    cudaCheck(cudaDeviceSynchronize());
    std::cout << "Initialization done" << std::endl;


    // Test_Neural_Net_Eval(Crafts);

    GUI::TimerStartup = float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count()) / 1000.f;

    bool Do_Mutations = true;

    // Game Loop
    while (!glfwWindowShouldClose(window))
    {
        Round();

        if (!exit_round) {
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


            GUI::RoundEnd2();
        }
        else {
            exit_round = false;
        }

        ResetScoreCumulative<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);
        cudaCheck(cudaDeviceSynchronize());
    }

    // Cleanup
    Mem::Shutdown();
    GUI::Shutdown();
    Graphics::Shutdown();

    std::cout << "End" << std::endl;

    return 0;
}
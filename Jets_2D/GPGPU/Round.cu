// File Header
#include "Jets_2D/GPGPU/Round.hpp"

// Standard Library
#include <chrono>

// CUDA
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#ifdef _WIN32
#include <Windows.h>        // Removes glad APIENTRY redefine warning
#endif

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/GPGPU/Match.hpp"
#include "Jets_2D/GPGPU/Epoch.hpp"
#include "Jets_2D/GPGPU/MapVertexBuffer.hpp"
#include "Jets_2D/Graphics/Draw.hpp"
#include "Jets_2D/Graphics/GrSetup.hpp"
#include "Jets_2D/Graphics/Component.hpp"
#include "Jets_2D/GUI/GUI.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"
#include "Jets_2D/GPGPU/Cooperative_Call.hpp"

std::chrono::steady_clock::time_point Timer;

void Round()
{
    int Opponent_ID_Weights = rand() % GUI::OpponentRankRange;
    float AngleStart;

    for (int Angle = 0; Angle < 2; Angle++) // TODO: Change back to 2 ater mutation test
    {
        if (Angle == 0)
            AngleStart = (float(rand()) / RAND_MAX * 2.f - 1.f) * 5.f + 30.f;
        else if (Angle == 1)
            AngleStart = -(float(rand()) / RAND_MAX * 2.f - 1.f) * 5.f - 30.f;

        // One once when opponent is on left side and once when opponent is on right side
        for (int PositionNumber = 0; PositionNumber < 2; PositionNumber++)
        {
            GPGPU::CUDA_Map();
            ResetMatch<<<MATCH_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Match, Crafts, d_Buffer, PositionNumber, AngleStart);
            cudaCheck(cudaDeviceSynchronize());
            GPGPU::CUDA_Unmap();

            while (!h_AllDone && !glfwWindowShouldClose(window))
            {
                if ((float)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - Timer).count() > 1000.f / FRAMES_PER_SECOND)
                {
                    Timer = std::chrono::steady_clock::now();

                    glfwPollEvents();

                    if (!GUI::Pause)
                    {
                        cudaCheck(cudaDeviceSynchronize());

                        GLCheck(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
                        GLCheck(glClearColor(GUI::clear_color.x, GUI::clear_color.y, GUI::clear_color.z, GUI::clear_color.w));
                        glfwGetFramebufferSize(window, &GL::ScreenWidth, &GL::ScreenHeight);
                        Inputs::DoMovement();

                        h_AllDone = true;
                        cudaCheck(cudaMemcpy(&Match->AllDone, &h_AllDone, sizeof(bool), cudaMemcpyHostToDevice));
                        cudaCheck(cudaDeviceSynchronize());

                        GPGPU::CUDA_Map();
                        cudaCheck(cudaDeviceSynchronize());

                        // std::cout << "Match: " << Match << std::endl;
                        // std::cout << "Crafts: " << Crafts << std::endl;
                        // std::cout << "d_Buffer: " << d_Buffer << std::endl;
                        // std::cout << "d_Config: " << d_Config << std::endl;
                        // std::cout << "Opponent_ID_Weights: " << Opponent_ID_Weights << std::endl;

                        // cudaDeviceProp DeviceProp;
                        // cudaGetDeviceProperties(&DeviceProp, 0);

                        //void *Args[] { &Match, &Crafts, &d_Buffer, &d_Config, &Opponent_ID_Weights };
                        //cudaLaunchCooperativeKernel((void*)RunEpoch, 50, 50, Args);
                        
                        //Cooperative_Launch(RunEpoch, DeviceProp.multiProcessorCount, CRAFT_COUNT, Match, Crafts, d_Buffer, d_Config, Opponent_ID_Weights);
                        RunEpoch<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Match, Crafts, d_Buffer, d_Config, Opponent_ID_Weights);
                        cudaCheck(cudaDeviceSynchronize());
                        GPGPU::CUDA_Unmap();

                        cudaCheck(cudaMemcpy(&h_AllDone, &Match->AllDone, sizeof(bool), cudaMemcpyDeviceToHost));
                        cudaCheck(cudaDeviceSynchronize());

                        Graphics::Draw();
                    }

                    GUI::Run(Opponent_ID_Weights, PositionNumber, AngleStart);
                    glfwSwapBuffers(window);
                }
            }

            ScoreCumulativeCalc<<<CRAFT_COUNT / BLOCK_SIZE, BLOCK_SIZE>>>(Crafts);

            GUI::MatchEnd();
            h_AllDone = false;
        }
    }
}

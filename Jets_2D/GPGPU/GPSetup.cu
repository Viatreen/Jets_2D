// File Headers
#include "Jets_2D/GPGPU/GPSetup.hpp"

// Standard Library
#include <iostream>

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Project Headers
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"

CraftState* Crafts;

MatchState* Match;
temp* Temp;
config* d_Config;
GraphicsObjectPointer   Buffer;     // Filled by CUDA_Map and copied to global memory
GraphicsObjectPointer* d_Buffer;    // Global memory version

config* h_Config;   // Host side variable. Requirement, whenever this is changed, it must be uploaded to GPU.

bool h_AllDone = false;  // Breaks up epoch iterations so as to not trip Windows GPU watchdog timer and also to allow real-time rendering

namespace Mem
{
    void Setup()
    {
        cudaCheck(cudaMalloc(&Match, sizeof(MatchState)));

        cudaCheck(cudaMalloc(&Crafts, sizeof(CraftState)));
        cudaCheck(cudaDeviceSynchronize());

        // std::cout << "Neuron Address: " << &Crafts->Neuron << "-0x" << std::hex << (unsigned long)(&Crafts->Neuron) + sizeof(float) * 2 * CRAFT_COUNT * NEURON_AMOUNT << ", Weight Address: "  << &Crafts->Weight << "-0x" << std::hex << (unsigned long)(&Crafts->Weight) + sizeof(float) * CRAFT_COUNT * WEIGHT_AMOUNT << std::endl;

        cudaCheck(cudaMalloc(&Temp, sizeof(temp)));
        cudaCheck(cudaDeviceSynchronize());

        h_Config = new config();

        cudaCheck(cudaMalloc(&d_Config, sizeof(config)));
        cudaCheck(cudaMemcpy(d_Config, h_Config, sizeof(config), cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMalloc(&d_Buffer, sizeof(GraphicsObjectPointer)));
        cudaCheck(cudaDeviceSynchronize());
    }
    void Shutdown()
    {
        cudaCheck(cudaFree(Match));
        cudaCheck(cudaFree(Crafts));
        cudaCheck(cudaFree(Temp));

        delete h_Config;
        cudaCheck(cudaFree(d_Config));
    }
}

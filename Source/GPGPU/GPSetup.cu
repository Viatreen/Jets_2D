// File Headers
#include "GPSetup.h"

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Headers
#include "GPGPU/State.h"
#include "GPGPU/GPErrorCheck.h"

CraftState* Crafts;

MatchState* Match;
temp* Temp;
config* d_Config;
GraphicsObjectPointer	Buffer;		// Filled by CUDA_Map and copied to global memory
GraphicsObjectPointer* d_Buffer;	// Global memory version

config* h_Config;	// Host side variable. Requirement, whenever this is changed, it must be uploaded to GPU.

bool h_AllDone = false;	 // Breaks up epoch iterations so as to not trip Windows GPU watchdog timer and also to allow real-time rendering

namespace Mem
{
	void Setup()
	{
		cudaCheck(cudaMalloc(&Match, sizeof(MatchState)));

		cudaCheck(cudaMalloc(&Crafts, sizeof(CraftState)));
		cudaCheck(cudaDeviceSynchronize());

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

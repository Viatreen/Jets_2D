#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Headers
#include "GPGPU/State.h"
#include "ErrorCheck.h"

//namespace GPGPU
//{
MatchState				*Match;
CraftPtrArr				*Crafts;
TempPtrArr				*Temp;
config					*d_Config;
GraphicsObjectPointer	Buffer;		// Filled by CUDA_Map and copied to global memory
GraphicsObjectPointer	*d_Buffer;	// Global memory version

config					*h_Config;	// Host side variable. Requirement, whenever this is changed, it must be uploaded to GPU.

bool h_AllDone = false;	 // Breaks up epoch iterations so as to not trip Windows GPU watchdog timer and also to allow real-time rendering

CraftPtrArr CraftsDevicePointers;
TempPtrArr  WeightsTempDevicePointers;

namespace Mem
{
	void Setup()
	{
		cudaCheck(cudaMalloc(&Match, sizeof(MatchState)));

		cudaCheck(cudaMalloc(&Crafts, sizeof(CraftPtrArr)));
		cudaCheck(cudaDeviceSynchronize());
		
		for (int i = 0; i < WARP_COUNT; i++)
		{
			cudaCheck(cudaMalloc(&CraftsDevicePointers.Warp[i], sizeof(CraftState)));
			cudaCheck(cudaMemcpy(&Crafts->Warp[i], &CraftsDevicePointers.Warp[i], sizeof(CraftState*), cudaMemcpyHostToDevice));
			cudaCheck(cudaDeviceSynchronize());
		}

		cudaCheck(cudaMalloc(&Temp, sizeof(TempPtrArr)));
		cudaCheck(cudaDeviceSynchronize());
		
		for (int i = 0; i < 1; i++)	// TODO: Fix for loop high index
		{
			cudaCheck(cudaMalloc(&WeightsTempDevicePointers.Warp[i], sizeof(temp)));
			cudaCheck(cudaMemcpy(&Temp->Warp[i], &WeightsTempDevicePointers.Warp[i], sizeof(CraftState*), cudaMemcpyHostToDevice));	// TODO: Free this
			cudaCheck(cudaDeviceSynchronize());
		}
		
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
		for (int i = 0; i < WARP_COUNT; i++)
			cudaCheck(cudaFree(CraftsDevicePointers.Warp[i]));

		cudaCheck(cudaFree(Temp));		
		for (int i = 0; i < 1; i++)
			cudaCheck(cudaFree(WeightsTempDevicePointers.Warp[i]));

		delete h_Config;
		cudaCheck(cudaFree(d_Config));
	}
}
//}
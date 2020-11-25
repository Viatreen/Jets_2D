#pragma once

// Project Headers
#include "Graphics/Component.h"
#include "Graphics/Circle.h"
#include "Graphics/Thrust.h"
#include "GPGPU/State.h"

namespace GPGPU
{
	void CUDA_Map()
	{
		for (int i = 0; i < WARP_COUNT; i++)
		{
			Craft::Fuselage[i]->CUDA_Map(Buffer.Fuselage[i]);
			Craft::Wing[i]->CUDA_Map(Buffer.Wing[i]);
			Craft::Cannon[i]->CUDA_Map(Buffer.Cannon[i]);
		
			for (int j = 0; j < 4; j++)
			{
				Craft::Engine[i][j]->CUDA_Map(Buffer.Engine[i][j]);
				Craft::ThrustLong[i][j]->CUDA_Map(Buffer.ThrustLong[i][j]);
				Craft::ThrustShort[i][j]->CUDA_Map(Buffer.ThrustShort[i][j]);
			}
		
			for (int j = 0; j < BULLET_COUNT_MAX; j++)
				Craft::Bullet[i][j]->CUDA_Map(Buffer.Bullet[i][j]);
		}
		
		// TODO: Use constant memory for buffer
		cudaCheck(cudaMemcpy(d_Buffer, &Buffer, sizeof(GraphicsObjectPointer), cudaMemcpyHostToDevice));	// Copy buffer pointers to global memory
		
		cudaCheck(cudaDeviceSynchronize());
	}

	void CUDA_Unmap()
	{
		for (int i = 0; i < WARP_COUNT; i++)
		{
			Craft::Fuselage[i]->CUDA_Unmap();
			Craft::Wing[i]->CUDA_Unmap();
			Craft::Cannon[i]->CUDA_Unmap();
		
			for (int j = 0; j < 4; j++)
			{
				Craft::Engine[i][j]->CUDA_Unmap();
				Craft::ThrustLong[i][j]->CUDA_Unmap();
				Craft::ThrustShort[i][j]->CUDA_Unmap();
			}
		
			for (int j = 0; j < BULLET_COUNT_MAX; j++)
				Craft::Bullet[i][j]->CUDA_Unmap();
		}
		
		cudaCheck(cudaDeviceSynchronize());
	}
}
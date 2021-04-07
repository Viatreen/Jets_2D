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
		Craft::Fuselage->CUDA_Map(Buffer.Fuselage);
		Craft::Wing->CUDA_Map(Buffer.Wing);
		Craft::Cannon->CUDA_Map(Buffer.Cannon);
		
		for (int j = 0; j < 4; j++)
		{
			Craft::Engine[j]->CUDA_Map(Buffer.Engine[j]);
			Craft::ThrustLong[j]->CUDA_Map(Buffer.ThrustLong[j]);
			Craft::ThrustShort[j]->CUDA_Map(Buffer.ThrustShort[j]);
		}
		
		for (int j = 0; j < BULLET_COUNT_MAX; j++)
			Craft::Bullet[j]->CUDA_Map(Buffer.Bullet[j]);
		
		// TODO: Use constant memory for buffer
		cudaCheck(cudaMemcpy(d_Buffer, &Buffer, sizeof(GraphicsObjectPointer), cudaMemcpyHostToDevice));	// Copy buffer pointers to global memory
		
		cudaCheck(cudaDeviceSynchronize());
	}

	void CUDA_Unmap()
	{
		Craft::Fuselage->CUDA_Unmap();
		Craft::Wing->CUDA_Unmap();
		Craft::Cannon->CUDA_Unmap();

		for (int j = 0; j < 4; j++)
		{
			Craft::Engine[j]->CUDA_Unmap();
			Craft::ThrustLong[j]->CUDA_Unmap();
			Craft::ThrustShort[j]->CUDA_Unmap();
		}

		for (int j = 0; j < BULLET_COUNT_MAX; j++)
			Craft::Bullet[j]->CUDA_Unmap();

		cudaCheck(cudaDeviceSynchronize());
	}
}
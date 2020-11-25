#pragma once

// OpenGL
#include <glad/glad.h>

// CUDA
#include <cuda_gl_interop.h>

// Project Headers
#include "Config.h"
#include "ErrorCheck.h"

namespace Craft
{
	struct Thrust
	{
		// OpenGL
		unsigned int VA, VB, EB;

		// CUDA
		cudaGraphicsResource_t VertexBufferCuResource;

		Thrust()
		{
			// Retrieve GL indices and load to GPU memory
			GLCheck(glGenVertexArrays(1, &VA));
			GLCheck(glGenBuffers(1, &VB));
			GLCheck(glGenBuffers(1, &EB));
			GLCheck(glBindVertexArray(VA));
			
			GLCheck(glBindBuffer(GL_ARRAY_BUFFER, VB));
			GLCheck(glBufferData(GL_ARRAY_BUFFER, 15 * WARP_SIZE * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW));

			// Vertex Positions
			GLCheck(glEnableVertexAttribArray(0));
			GLCheck(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0));

			GLCheck(glEnableVertexAttribArray(1));
			GLCheck(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(1 * 3 * WARP_SIZE * 2 * sizeof(float))));

			// Vertex Colors
			GLCheck(glEnableVertexAttribArray(2));
			GLCheck(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * 3 * WARP_SIZE * 2 * sizeof(float))));

			GLCheck(glEnableVertexAttribArray(3));
			GLCheck(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(3 * 3 * WARP_SIZE * 2 * sizeof(float))));

			GLCheck(glEnableVertexAttribArray(4));
			GLCheck(glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(4 * 3 * WARP_SIZE * 2 * sizeof(float))));

			unsigned int* ThrustIndexOrder = new unsigned int[3 * WARP_SIZE * 2];

			for (int i = 0; i < WARP_SIZE * 2; i++)
				for (int j = 0; j < 3; j++)
					ThrustIndexOrder[3 * i + j] = WARP_SIZE * 2 * j + i;

			GLCheck(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));
			GLCheck(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * WARP_SIZE * 2 * sizeof(unsigned int), ThrustIndexOrder, GL_DYNAMIC_DRAW));
			delete[] ThrustIndexOrder;
			
			//std::cout << "Register Thrust" << std::endl;

			cudaCheck(cudaGraphicsGLRegisterBuffer(&VertexBufferCuResource, VB, 0));		// TODO: Find out which flags to set (Final function parameter)
		}
		void CUDA_Map(float*& d_VertexBuffer)
		{
			// Create CUDA graphics object pointer
			cudaCheck(cudaGraphicsMapResources(1, &VertexBufferCuResource));

			// Direct pointer to OpenGL VBO
			size_t num_bytes;
			cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&d_VertexBuffer, &num_bytes, VertexBufferCuResource));
		}

		void CUDA_Unmap()
		{
			cudaCheck(cudaGraphicsUnmapResources(1, &VertexBufferCuResource));
		}

		~Thrust()
		{
			GLCheck(glDeleteVertexArrays(1, &VA));
			GLCheck(glDeleteBuffers(1, &VB));
		}

		void Draw()
		{
			GLCheck(glBindVertexArray(VA));
			GLCheck(glDrawElements(GL_TRIANGLES, 3 * WARP_SIZE * 2, GL_UNSIGNED_INT, (void*)0));
			GLCheck(glBindVertexArray(0));
		}
	};

	Thrust* ThrustLong[WARP_COUNT][4];
	Thrust* ThrustShort[WARP_COUNT][4];
}

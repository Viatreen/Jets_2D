// File Header
#include "Component.h"

// Standard Library
#include <iostream>

// OpenGL
#include "glad/glad.h"

// CUDA
#include <cuda_gl_interop.h>

// Project Headers
#include "Config.h"
#include "ErrorCheck.h"
#include "GPGPU/GPErrorCheck.h"

namespace Craft
{
	Component* Wing;
	Component* Cannon;
	Component* Engine[4];	// Engine 0 is left most

	Component::Component()
	{
		// This wont be needed anymore with show and conceal vertices functions
		// Create interleaved arrays of vertices in order to cause CUDA manipulation of vertex buffer data to be coalesced

		// Retrieve GL indices and load to GPU memory
		GLCheck(glGenVertexArrays(1, &VA));
		GLCheck(glGenBuffers(1, &VB));
		GLCheck(glGenBuffers(1, &EB));
		GLCheck(glBindVertexArray(VA));
		GLCheck(glBindBuffer(GL_ARRAY_BUFFER, VB));
		GLCheck(glBufferData(GL_ARRAY_BUFFER, 4 * 5 * CRAFT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW));

		unsigned int* IndexOrder = new unsigned int[CRAFT_COUNT * 2 * 5];

		for (int i = 0; i < CRAFT_COUNT * 2; i++)
		{
			for (int j = 0; j < 4; j++)
				IndexOrder[5 * i + j] = j * CRAFT_COUNT * 2 + i;

			IndexOrder[5 * i + 4] = 0xFFFFFFFF;	// Draw restart primitive
		}

		GLCheck(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));
		GLCheck(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 5 * CRAFT_COUNT * 2 * sizeof(unsigned int), IndexOrder, GL_DYNAMIC_DRAW));
		delete[] IndexOrder;

		// TODO: What would the stride be

		// Vertex Positions
		GLCheck(glEnableVertexAttribArray(0));
		GLCheck(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0));

		GLCheck(glEnableVertexAttribArray(1));
		GLCheck(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(1 * 4 * CRAFT_COUNT * 2 * sizeof(float))));

		// Vertex Colors												   
		GLCheck(glEnableVertexAttribArray(2));
		GLCheck(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * 4 * CRAFT_COUNT * 2 * sizeof(float))));

		GLCheck(glEnableVertexAttribArray(3));
		GLCheck(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(3 * 4 * CRAFT_COUNT * 2 * sizeof(float))));

		GLCheck(glEnableVertexAttribArray(4));
		GLCheck(glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(4 * 4 * CRAFT_COUNT * 2 * sizeof(float))));

		// CUDA/GL Interoperation
		//cudaCheck(cudaGraphicsResourceSetMapFlags(d_VertexBuffer, 0));
		cudaCheck(cudaGraphicsGLRegisterBuffer(&VertexBufferCuResource, VB, 0)); // cudaGraphicsRegisterFlagsWriteDiscard));		// TODO: Find out which flags to set (Final function parameter)
	}
	void Component::CUDA_Map(float*& d_VertexBuffer)
	{
		// Create CUDA graphics object pointer
		cudaCheck(cudaGraphicsMapResources(1, &VertexBufferCuResource));

		// Direct pointer to OpenGL VBO
		size_t num_bytes;
		cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&d_VertexBuffer, &num_bytes, VertexBufferCuResource));
		//std::cout << "Size: " << num_bytes << std::endl;
	}

	void Component::CUDA_Unmap()
	{
		cudaCheck(cudaGraphicsUnmapResources(1, &VertexBufferCuResource));
	}

	Component::~Component()
	{
		GLCheck(glDeleteVertexArrays(1, &VA));
		GLCheck(glDeleteBuffers(1, &VB));
		GLCheck(glDeleteBuffers(1, &EB));
	}

	void Component::Draw()
	{
		GLCheck(glBindVertexArray(VA));
		GLCheck(glDrawElements(GL_TRIANGLE_FAN, 5 * CRAFT_COUNT * 2, GL_UNSIGNED_INT, (void*)0));
		GLCheck(glBindVertexArray(0));
	}
};	// End namespace Craft
// File Header
#include "Jets_2D/Graphics/Circle.hpp"

// Standard Library
#include <iostream>

// OpenGL
#include <glad/glad.h>

// CUDA
#include <cuda_gl_interop.h>

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/ErrorCheck.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"


namespace Craft
{
    Circle* Fuselage;
    Circle* Bullet[BULLET_COUNT_MAX];
    
    // Create interleaved arrays of vertices in order to cause CUDA manipulation of vertex buffer data to be coalesced
    // Size needs to include number of circle edge vertices plus center vertex
    Circle::Circle(int VertexCount)
    {
        // Retrieve GL indices and load to GPU memory
        GLCheck(glGenVertexArrays(1, &VA));
        GLCheck(glGenBuffers(1, &VB));
        GLCheck(glGenBuffers(1, &EB));
        GLCheck(glBindVertexArray(VA));
        GLCheck(glBindBuffer(GL_ARRAY_BUFFER, VB));
        GLCheck(glBufferData(GL_ARRAY_BUFFER, 5 * (VertexCount + 1) * CRAFT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW));

        unsigned int* IndexOrder = new unsigned int[CRAFT_COUNT * 2 * (VertexCount + 3)];   // Circle elements. Center vertex + edge vertices + first edge vertex + primitive restart

        // TODO: Connect first and last circle edge vertices
        for (int i = 0; i < CRAFT_COUNT * 2; i++)
        {
            for (int j = 0; j < VertexCount + 1; j++)
                IndexOrder[(VertexCount + 3) * i + j] = j * CRAFT_COUNT * 2 + i;

            IndexOrder[(VertexCount + 3) * i + VertexCount + 1] = CRAFT_COUNT * 2 + i;          // Tie circle back to first edge vertex
            IndexOrder[(VertexCount + 3) * i + VertexCount + 2] = 0xFFFFFFFF;   // Primitive restart
        }

        //for (int i = 0; i < CRAFT_COUNT * (FUSELAGE_VERT_COUNT + 3); i++)
        //  std::cout << "Fuselage Index[" << i << "]: " << IndexOrder[i] << std::endl;

        GLCheck(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));
        GLCheck(glBufferData(GL_ELEMENT_ARRAY_BUFFER, (VertexCount + 3) * CRAFT_COUNT * 2 * sizeof(unsigned int), IndexOrder, GL_DYNAMIC_DRAW));
        delete[] IndexOrder;

        // Vertex Positions
        GLCheck(glEnableVertexAttribArray(0));
        GLCheck(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0));

        GLCheck(glEnableVertexAttribArray(1));
        GLCheck(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(1 * (VertexCount + 1) * CRAFT_COUNT * 2 * sizeof(float))));

        // Vertex Colors                                                   
        GLCheck(glEnableVertexAttribArray(2));
        GLCheck(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * (VertexCount + 1) * CRAFT_COUNT * 2 * sizeof(float))));

        GLCheck(glEnableVertexAttribArray(3));
        GLCheck(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(3 * (VertexCount + 1) * CRAFT_COUNT * 2 * sizeof(float))));

        GLCheck(glEnableVertexAttribArray(4));
        GLCheck(glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(4 * (VertexCount + 1) * CRAFT_COUNT * 2 * sizeof(float))));

        // CUDA/GL Interoperation
        //cudaCheck(cudaGraphicsResourceSetMapFlags(d_VertexBuffer, 0));
        cudaCheck(cudaGraphicsGLRegisterBuffer(&VertexBufferCuResource, VB, 0)); // cudaGraphicsRegisterFlagsWriteDiscard));        // TODO: Find out which flags to set (Final function parameter)
    }
    void Circle::CUDA_Map(float*& d_VertexBuffer)
    {
        // Create CUDA graphics object pointer
        cudaCheck(cudaGraphicsMapResources(1, &VertexBufferCuResource));

        // Direct pointer to OpenGL VBO
        size_t num_bytes;
        cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&d_VertexBuffer, &num_bytes, VertexBufferCuResource));
        //std::cout << "Size: " << num_bytes << std::endl;
    }

    void Circle::CUDA_Unmap()
    {
        cudaCheck(cudaGraphicsUnmapResources(1, &VertexBufferCuResource));
    }

    Circle::~Circle()
    {
        GLCheck(glDeleteVertexArrays(1, &VA));
        GLCheck(glDeleteBuffers(1, &VB));
        GLCheck(glDeleteBuffers(1, &EB));
    }

    void Circle::Draw(int VertexCount)
    {
        GLCheck(glBindVertexArray(VA));
        GLCheck(glDrawElements(GL_TRIANGLE_FAN, (VertexCount + 3) * CRAFT_COUNT * 2, GL_UNSIGNED_INT, (void*)0));
        GLCheck(glBindVertexArray(0));
    }
}   // End Craft namespace
#pragma once

// Standard Library
#include <iostream>

// OpenGL
#include <glad/glad.h>

// CUDA
#include <cuda_gl_interop.h>

// Project Headers
#include "Jets_2D/Config.h"
#include "Jets_2D/ErrorCheck.h"
#include "Jets_2D/GPGPU/GPErrorCheck.h"


namespace Craft
{
    struct Circle
    {
        // OpenGL
        unsigned int VA, VB, EB;

        // CUDA
        cudaGraphicsResource_t VertexBufferCuResource;

        // Create interleaved arrays of vertices in order to cause CUDA manipulation of vertex buffer data to be coalesced
        // Size needs to include number of circle edge vertices plus center vertex
        Circle(int VertexCount);
        void CUDA_Map(float*& d_VertexBuffer);
        void CUDA_Unmap();
        ~Circle();
        void Draw(int VertexCount);
    };  // End Component struct

    extern Circle *Fuselage;
    extern Circle *Bullet[BULLET_COUNT_MAX];
}   // End Craft namespace
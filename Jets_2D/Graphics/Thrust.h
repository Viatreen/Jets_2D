#pragma once

// OpenGL
#include "glad/glad.h"

// CUDA
#include <cuda_gl_interop.h>

namespace Craft
{
    struct Thrust
    {
        // OpenGL
        unsigned int VA, VB, EB;

        // CUDA
        cudaGraphicsResource_t VertexBufferCuResource;

        Thrust();
        void CUDA_Map(float*& d_VertexBuffer);
        void CUDA_Unmap();
        ~Thrust();
        void Draw();
    };

    extern Thrust* ThrustLong[4];
    extern Thrust* ThrustShort[4];
}

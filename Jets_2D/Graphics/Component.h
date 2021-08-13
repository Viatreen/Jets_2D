#pragma once

#include "glad/glad.h"

// CUDA
#include <cuda_gl_interop.h>

namespace Craft
{
    struct Component
    {
        // OpenGL
        unsigned int VA, VB, EB;

        // CUDA
        cudaGraphicsResource_t VertexBufferCuResource;

        Component();
        void CUDA_Map(float*& d_VertexBuffer);
        void CUDA_Unmap();
        ~Component();
        void Draw();
    };  // End Component struct

    extern Component *Wing;
    extern Component *Cannon;
    extern Component *Engine[4];    // Engine 0 is left most
}   // End Craft namespace

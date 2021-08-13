#pragma once

// Project Headers
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/ErrorCheck.hpp"

extern CraftState* Crafts;

extern MatchState               *Match;
extern temp                     *Temp;
extern config                   *d_Config;
extern GraphicsObjectPointer    Buffer;     // Filled by CUDA_Map and copied to global memory
extern GraphicsObjectPointer    *d_Buffer;  // Global memory version

extern config                   *h_Config;  // Host side variable. Requirement, whenever this is changed, it must be uploaded to GPU.

extern bool h_AllDone;   // Breaks up epoch iterations so as to not trip Windows GPU watchdog timer and also to allow real-time rendering

namespace Mem
{
    void Setup();
    void Shutdown();
}

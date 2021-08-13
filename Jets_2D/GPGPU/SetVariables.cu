// File Headers
#include "Jets_2D/GPGPU/SetVariables.h"

// Standard Library
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Project Headers
#include "Jets_2D/Config.h"
#include "Jets_2D/ErrorCheck.h"
#include "Jets_2D/GPGPU/GPSetup.h"
#include "Jets_2D/GPGPU/GPErrorCheck.h"


// Turns on rendering of all matches
void RenderAllMatches()
{
    bool* RenderOnArray = new bool[CRAFT_COUNT];
    for (int i = 0; i < CRAFT_COUNT; i++)
        RenderOnArray[i] = true;
    cudaCheck(cudaMemcpy(Match->RenderOn, RenderOnArray, CRAFT_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnArray;

    bool* RenderOnFirstFrameArray = new bool[CRAFT_COUNT];
    for (int i = 0; i < CRAFT_COUNT; i++)
        RenderOnFirstFrameArray[i] = true;
    cudaCheck(cudaMemcpy(Match->RenderOnFirstFrame, RenderOnFirstFrameArray, CRAFT_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnFirstFrameArray;
}

// Turns on rendering of fit matches
void RenderFitMatches()
{
    bool* RenderOnArray = new bool[MATCH_COUNT];
    for (int i = 0; i < MATCH_COUNT; i++)
        if (i < FIT_COUNT)
            RenderOnArray[i] = true;
        else
            RenderOnArray[i] = false;

    cudaCheck(cudaMemcpy(Match->RenderOn, RenderOnArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnArray;

    // TODO: Check if I need to use RenderOnFirstFrame
    bool* RenderOnFirstFrameArray = new bool[MATCH_COUNT];
    bool* RenderOffFirstFrameArray = new bool[MATCH_COUNT];
    for (int i = 0; i < MATCH_COUNT; i++)
        if (i < FIT_COUNT)
        {
            RenderOnFirstFrameArray[i] = true;
            RenderOffFirstFrameArray[i] = false;
        }
        else
        {
            RenderOnFirstFrameArray[i] = false;
            RenderOffFirstFrameArray[i] = true;
        }

    cudaCheck(cudaMemcpy(Match->RenderOnFirstFrame, RenderOnFirstFrameArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(Match->RenderOffFirstFrame, RenderOffFirstFrameArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnFirstFrameArray;
    delete[] RenderOffFirstFrameArray;
}

// Turns on rendering of match of best craft
void RenderBestMatch()
{
    bool* RenderOnArray = new bool[MATCH_COUNT];
    for (int i = 1; i < MATCH_COUNT; i++)
        RenderOnArray[i] = false;
    RenderOnArray[0] = true;
    cudaCheck(cudaMemcpy(Match->RenderOn, RenderOnArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnArray;

    bool* RenderOffFirstFrameArray = new bool[MATCH_COUNT];
    bool* RenderOnFirstFrameArray = new bool[MATCH_COUNT];

    for (int i = 0; i < MATCH_COUNT; i++)
    {
        RenderOffFirstFrameArray[i] = true;
        RenderOnFirstFrameArray[i] = false;
    }

    RenderOffFirstFrameArray[0] = false;
    RenderOnFirstFrameArray[0] = true;

    cudaCheck(cudaMemcpy(Match->RenderOffFirstFrame, RenderOffFirstFrameArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());

    delete[] RenderOffFirstFrameArray;
    delete[] RenderOnFirstFrameArray;
}

// Turns off all match rendering
void RenderNoMatches()
{
    bool* RenderOnArray = new bool[MATCH_COUNT];
    for (int i = 0; i < MATCH_COUNT; i++)
        RenderOnArray[i] = false;
    cudaCheck(cudaMemcpy(Match->RenderOn, RenderOnArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOnArray;

    bool* RenderOffFirstFrameArray = new bool[MATCH_COUNT];
    for (int i = 0; i < MATCH_COUNT; i++)
        RenderOffFirstFrameArray[i] = true;
    cudaCheck(cudaMemcpy(Match->RenderOffFirstFrame, RenderOffFirstFrameArray, MATCH_COUNT * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    delete[] RenderOffFirstFrameArray;
}

// Call everytime h_Config is modifed
void SyncConfigArray()
{
    h_Config->IterationsPerCall = round(h_Config->TimeSpeed / TIME_STEP / FRAMES_PER_SECOND);

    cudaCheck(cudaMemcpy(d_Config, h_Config, sizeof(config), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
}

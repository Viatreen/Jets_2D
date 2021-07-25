// File Header
#include "Epoch.h"

// CUDA
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

// Project Headers
#include "Config.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/Match.h"
#include "GPGPU/NeuralNet.h"
#include "GPGPU/Vertices.h"
#include "GPGPU/Physic.h"
#include "GPGPU/State.h"
#include "GPGPU/GPErrorCheck.h"

__global__ void RunEpoch(MatchState *Match, CraftState *C, GraphicsObjectPointer *Buffer, config *Config, int Opponent_ID_Weights)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    // if (idx == 0)
    // {
    //  printf("Match pointer: %p, Craft: %p, Buffer: %p, Config: %p, Opp_ID: %d\n", Match, C, Buffer, Config, Opponent_ID_Weights);
    // }

    // Process a few physics time steps
    for (int TimeStepIteration = 0; TimeStepIteration < Config->IterationsPerCall && !Match->Done[idx]; TimeStepIteration++)
    {
        // cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        // grid.sync();

        if (Match->ElapsedTicks[idx] % (FRAMERATE_NN_PHYSICS) == 0)
        {
            // Trainee neural network
            if (C->Active[idx])
                State_Processing(C, Buffer, idx + CRAFT_COUNT, idx, idx);
            // Opponent neural network
            if (C->Active[idx + CRAFT_COUNT])
                // Each opponent has their own set of neurons and state but uses the same weight vector as all other oppenents
                State_Processing(C, Buffer, idx, idx + CRAFT_COUNT, Opponent_ID_Weights);
        }

        if (C->Active[idx])
            Physic(Match, C, Config, false);
        if (C->Active[idx + CRAFT_COUNT])
            Physic(Match, C, Config, true);

        if (C->Active[idx] )
            CollisionDetect(C, idx, idx + CRAFT_COUNT);

        // TODO: Convert to single function
        BulletMechanics(Buffer, C, idx, idx + CRAFT_COUNT, Config);
        BulletMechanics(Buffer, C, idx + CRAFT_COUNT, idx, Config);

        C->Score[idx]               = C->ScoreTime[idx]               + C->ScoreFuelEfficiency[idx]               + C->ScoreDistance[idx]  / 1000.f              + C->ScoreBullet[idx]               - C->ScoreBullet[idx + CRAFT_COUNT] / 10.f;    // TODO: Consider floating point precision for accumuulating score

        if (idx == 0 && Match->ElapsedTicks[idx] % 10 == 0)
            printf("Jet 0 ScoreTime: %2.6f, \tScoreFuelEfficiency: %2.6f, \tPosition.x: %2.6f, \tPosition.y: %2.6f\n", C->ScoreTime[idx], C->ScoreFuelEfficiency[idx], C->Position.X[idx], C->Position.Y[idx]);

        if (!C->Active[idx] || Match->ElapsedTicks[idx] == (Config->TimeStepLimit - 1))
        {
            Match->Done[idx] = true;

            ConcealVertices(Buffer, idx, idx + CRAFT_COUNT);
            C->Score[idx]               = C->ScoreTime[idx]               + C->ScoreFuelEfficiency[idx]               + C->ScoreDistance[idx]  / 1000.f              + C->ScoreBullet[idx]               - C->ScoreBullet[idx + CRAFT_COUNT] / 10.f;    // TODO: Consider floating point precision for accumuulating score
            if (idx == 0)
                printf("Ticks: %d, Jet 0 Score: %f\n", Match->ElapsedTicks[idx], C->Score[idx]);
            C->Score[idx + CRAFT_COUNT] = C->ScoreTime[idx + CRAFT_COUNT] + C->ScoreFuelEfficiency[idx + CRAFT_COUNT] + C->ScoreDistance[idx + CRAFT_COUNT] / 1000.f + C->ScoreBullet[idx + CRAFT_COUNT] - C->ScoreBullet[idx]  / 10.f;              // Score of opponent does not matter except
        }

        Match->ElapsedTicks[idx]++;
    } // End main step iteration for loop 

    // If craft is going from rendering to hidden, conceal its vertices
    if (Match->RenderOffFirstFrame[idx] == true)
    {
        ConcealVertices(Buffer, idx, idx + CRAFT_COUNT);
        Match->RenderOffFirstFrame[idx] = false;
    }

    // If craft is going from hidden to rendering, create its vertices
    if (Match->RenderOnFirstFrame[idx] == true)
    {
        ShowVertices(C, Buffer, idx, idx + CRAFT_COUNT);
        Match->RenderOnFirstFrame[idx] = false;
    }

    // If match has render turned on, process its vertices
    if (Match->RenderOn[idx] && !Match->Done[idx])
    {
        GraphicsProcessing(C, Buffer, idx, idx + CRAFT_COUNT);
    }

    if (!Match->Done[idx])
    {
        Match->AllDone = false;
    }
}   // End RunEpoch function

__global__ void ScoreCumulativeCalc(CraftState *C)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    C->ScoreCumulative[idx]               += C->Score[idx];
    C->ScoreCumulative[idx + CRAFT_COUNT] += C->Score[idx + CRAFT_COUNT];
}

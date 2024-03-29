// File Header
#include "Jets_2D/GPGPU/Match.hpp"

// Standard Library
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Project Headers
#include "Jets_2D/GPGPU/Vertices.hpp"
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GPGPU/NeuralNet.hpp"

__device__ void WeightsMutateAndTransfer(CraftState* C, config* Config, int SourceIndex, int TargetIndex)
{
    for (int i = 0; i < WEIGHT_AMOUNT; i++)
    {
        float Chance = curand_uniform(&C->RandState[SourceIndex]);

        if (Chance < Config->MutationFlipChance)
            C->Weight[CRAFT_COUNT * i + TargetIndex] = -C->Weight[CRAFT_COUNT * i + SourceIndex];
        else if (Chance < Config->MutationFlipChance + Config->MutationScaleChance)
            C->Weight[CRAFT_COUNT * i + TargetIndex]
            = (1.f - Config->MutationScale + 2.f * Config->MutationScale * curand_uniform(&C->RandState[TargetIndex])) * C->Weight[CRAFT_COUNT * i + SourceIndex];
        else if (Chance < Config->MutationFlipChance + Config->MutationScaleChance + Config->MutationSlideChance)
            C->Weight[CRAFT_COUNT * i + TargetIndex] = C->Weight[CRAFT_COUNT * i + SourceIndex] + Config->MutationSigma * curand_normal(&C->RandState[SourceIndex]);
        else
            C->Weight[CRAFT_COUNT * i + TargetIndex] = C->Weight[CRAFT_COUNT * i + SourceIndex];

        if (C->Weight[CRAFT_COUNT * i + TargetIndex] > Config->WeightMax)
            C->Weight[CRAFT_COUNT * i + TargetIndex] = Config->WeightMax;
        if (C->Weight[CRAFT_COUNT * i + TargetIndex] < -Config->WeightMax)
            C->Weight[CRAFT_COUNT * i + TargetIndex] = -Config->WeightMax;
    }
}

__global__ void RoundAssignPlace(CraftState* Crafts)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    Crafts->Place[idx] = 0;

    // Assign place to each craft
    for (int i = 0; i < CRAFT_COUNT; i++)
        if (i != idx)
            if (Crafts->ScoreCumulative[idx] < Crafts->ScoreCumulative[i])
                Crafts->Place[idx]++;
}

__global__ void RoundTieFix(CraftState* Crafts)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    // Deal with score ties
    // This function isn't completely thread-safe, but errors are unlikely and will not cause program to fail and will hardly interfere with learning
    for (int i = 0; i < CRAFT_COUNT; i++)   // TODO: Lookup: Will parallelism be affected if I use "int i = idx"?
    {
        if (i > idx && i != idx && Crafts->Place[idx] == Crafts->Place[i] )
        {
            for (int j = 0; j < CRAFT_COUNT; j++)
            {
                if (j != idx && j != i && Crafts->Place[j] > Crafts->Place[idx])
                {
                    atomicAdd(&Crafts->Place[j], 1);    // Only compatible with compute >=6.0
                }
            }
            atomicAdd(&Crafts->Place[idx], 1);          // Only compatible with compute >=6.0
        }
    }
}

__global__ void RoundPrintFirstPlace(CraftState* C, int RoundNumber)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (C->Place[idx] == 0) {
        // printf("Round: %4d, First place ID: %6d, Score: %7.2f\n", RoundNumber, idx, C->ScoreCumulative[idx] / MATCHES_PER_ROUND);
    }
}

__global__ void IDAssign(CraftState* C, config* Config)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    /*if (idx == 0)
        printf("Round Number: %d\n", Config->RoundNumber);*/

    if (idx < CRAFT_COUNT - FIT_COUNT)
    {
        for (int i = 1; i < CRAFT_COUNT / FIT_COUNT; i++)
            // Round number starts at 1 and is initialized. First round encountered by this function is round 2
            C->ID[FIT_COUNT + idx] = CRAFT_COUNT + (Config->RoundNumber - 2) * (CRAFT_COUNT - FIT_COUNT) + idx;
    }
}

__global__ void WeightsAndIDTempSave(CraftState* C, temp* Temp) // TODO: Fix this. Higher place is better
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (C->Place[idx] < FIT_COUNT)
    {
        int PlaceID = C->Place[idx];

        for (int i = 0; i < WEIGHT_AMOUNT; i++)
            Temp->Weight[FIT_COUNT * i + PlaceID] = C->Weight[CRAFT_COUNT * i + idx];

        C->TempID[PlaceID] = C->ID[idx];
    }
}

// TODO: Combine transfer and mutate kernels
__global__ void WeightsAndIDTransfer(CraftState* C, temp* Temp)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    for (int i = 0; i < WEIGHT_AMOUNT; i++)
        C->Weight[CRAFT_COUNT * i + idx] = Temp->Weight[FIT_COUNT * i + idx];

    C->ID[idx] = C->TempID[idx];
}

__global__ void WeightsMutate(CraftState* C, config* Config)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

#pragma unroll
    for (int i = 1; i < CRAFT_COUNT / FIT_COUNT; i++)
        WeightsMutateAndTransfer(C, Config, idx, FIT_COUNT * i + idx);
}

__global__ void ScoreTempSave(CraftState* C)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    int PlaceID = C->Place[idx] % CRAFT_COUNT;

    C->ScoreTemp[PlaceID] = C->ScoreCumulative[idx];
}

__global__ void ScoreTransfer(CraftState* C)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    C->ScoreCumulative[idx] = C->ScoreTemp[idx];
}

__global__ void ResetScoreCumulative(CraftState* Crafts)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    // Reset Score Cumulative
    Crafts->ScoreCumulative[idx] = 0.f;
    Crafts->ScoreCumulative[idx + CRAFT_COUNT] = 0.f;
}

__device__ void print_Weights(CraftState* C, int ID_Neurons, int ID_Weights)
{
    // Calculate output neurons
    // for (unsigned int Input = 0; Input < LAYER_SIZE_HIDDEN; Input++)
    // {
    //  for (unsigned int Output = 0; Output < LAYER_SIZE_OUTPUT; Output++)
    //  {
    //      unsigned int Output_Index = LAYER_SIZE_INPUT + LAYER_AMOUNT_HIDDEN * LAYER_SIZE_HIDDEN + Output;
    //      unsigned int Input_Index  = LAYER_SIZE_INPUT + (LAYER_AMOUNT_HIDDEN - 1) * LAYER_SIZE_HIDDEN + Input;

    //      unsigned int Weight_Index
    //          = LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN
    //          + WEIGHT_AMOUNT_HIDDEN_LAYER * (LAYER_AMOUNT_HIDDEN - 1)
    //          + Input * LAYER_SIZE_OUTPUT
    //          + Output;

    //      // if (Weight_Index > WEIGHT_AMOUNT)
    //      {
    //          printf("Number of Weights: %d, Index: %d, Input: %d, Output: %d, Index into Array: %d, Total Number of Weights: %d, Begin Index: %d\n", WEIGHT_AMOUNT,
    //              Weight_Index, Input, Output, CRAFT_COUNT * Weight_Index + ID_Weights, CRAFT_COUNT * WEIGHT_AMOUNT, 
    //              WEIGHT_AMOUNT_INPUT_LAYER + WEIGHT_AMOUNT_HIDDEN_LAYER * (LAYER_AMOUNT_HIDDEN - 1));
    //      }

    //      C->Neuron[2 * CRAFT_COUNT * Output_Index + ID_Neurons] += C->Neuron[2 * CRAFT_COUNT * Input_Index + ID_Neurons] * C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
    //      C->Neuron[2 * CRAFT_COUNT * Output_Index + ID_Neurons] += C->Neuron[2 * CRAFT_COUNT * Input_Index + ID_Neurons];
    //      C->Neuron[2 * CRAFT_COUNT * Output_Index + ID_Neurons] += C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
    //      C->Neuron[0] += C->Neuron[2 * CRAFT_COUNT * Input_Index + ID_Neurons] * C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
    //  }
    // }
}

__device__ void Shrink_Weights(CraftState* C)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    for (int i = 0; i < LAYER_SIZE_INPUT; i++)
    {
        C->Neuron[2 * CRAFT_COUNT * i + idx] = 0.5f;
    }

    int Shrink_Count = 0;

    // if (idx < 32)
        // print_Weights(C, idx, idx);

    bool Too_Large = true;

    while (Too_Large)
    {
        Run_Neural_Net(C, false, idx, idx, Shrink_Count);

        float Sum = 0.f;
        for (int i = 0; i < LAYER_SIZE_OUTPUT; i++)
        {
            Sum += abs(C->Neuron[2 * CRAFT_COUNT * ( OUTPUT_LAYER_NEURON_BEGIN_INDEX + i ) + idx]);
        }
        if (Sum > 0.5f * float(LAYER_SIZE_OUTPUT))
        {
            if (idx == 0)
            {
                Shrink_Count++;
            }
            for (int i = 0; i < WEIGHT_AMOUNT; i++)
            {
                C->Weight[CRAFT_COUNT * i + idx] *= SHRINK_COEFFICIENT_WEIGHTS;
            }
        }
        else
        {
            Too_Large = false;
            /*if (idx == 0)
            {
                printf("Number of shrink cycles for idx(0): %d\n", Shrink_Count);

                for (int i = 0; i < LAYER_SIZE_INPUT; i++)
                {
                    float Value = C->Neuron[2 * CRAFT_COUNT * i + idx];
                    printf("%46.6f ", Value);
                    if (i < LAYER_SIZE_HIDDEN)
                    {
                        for (int j = 0; j < LAYER_AMOUNT_HIDDEN; j++)
                        {
                            float Value = C->Neuron[2 * CRAFT_COUNT * ( j * LAYER_SIZE_HIDDEN + LAYER_SIZE_INPUT + i ) + idx];
                            printf("%46.6f ", Value);
                        }
                    }
                    if (i < LAYER_SIZE_OUTPUT)
                    {
                        float Value = C->Neuron[2 * CRAFT_COUNT * ( OUTPUT_LAYER_NEURON_BEGIN_INDEX + i ) + idx];
                        if (i >= LAYER_SIZE_HIDDEN)
                            for (int k = 0; k < 47 * LAYER_AMOUNT_HIDDEN; k++)
                                printf(" ");
                        printf("%46.6f", Value);
                    }
                    printf("\n");
                }

                printf("With activation:\n");
                Run_Neural_Net(C, true, idx, idx);
                for (int i = 0; i < LAYER_SIZE_INPUT; i++)
                {
                    float Value = C->Neuron[2 * CRAFT_COUNT * i + idx];
                    printf("%46.6f ", Value);
                    if (i < LAYER_SIZE_HIDDEN)
                    {
                        for (int j = 0; j < LAYER_AMOUNT_HIDDEN; j++)
                        {
                            float Value = C->Neuron[2 * CRAFT_COUNT * (j * LAYER_SIZE_HIDDEN + LAYER_SIZE_INPUT + i) + idx];
                            printf("%46.6f ", Value);
                        }
                    }
                    if (i < LAYER_SIZE_OUTPUT)
                    {
                        float Value = C->Neuron[2 * CRAFT_COUNT * (OUTPUT_LAYER_NEURON_BEGIN_INDEX + i) + idx];
                        if (i >= LAYER_SIZE_HIDDEN)
                            for (int k = 0; k < 47 * LAYER_AMOUNT_HIDDEN; k++)
                                printf(" ");
                        printf("%46.6f", Value);
                    }
                    printf("\n");
                }

                printf("\n");
                printf("First 25 weights:\n");
                for (int i = 0; i < 25; i++)
                    printf("Weight(%d): %10.6f\n", i, C->Weight[2 * CRAFT_COUNT * i + idx]);
            }*/
        }
    }
}

__global__ void Init(CraftState* C)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    curand_init(124, idx, 0, &(C->RandState[idx]));

    for (int i = 0; i < WEIGHT_AMOUNT; i++)
        C->Weight[CRAFT_COUNT * i + idx] = (curand_uniform(&C->RandState[idx]) - 0.5f) * 2.f * WEIGHTS_MULTIPLIER;

    // Shrink_Weights(C);

    //if (idx < CRAFT_COUNT)
    //{
    //  // Engine 1 Angle = -Engine 0 Angle
    //  // Engine 2 Angle =  Engine 0 Angle
    //  // Engine 3 Angle = -Engine 0 Angle
    //  for (int i = 0; i < 3 * LAYER_SIZE_HIDDEN; i++)
    //  {
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = -C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = -C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //  }
    //  // Brake output neuron stays the same
    //  for (int i = 3 * LAYER_SIZE_INPUT; i < 4 * LAYER_SIZE_INPUT; i++)
    //  {
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //  }
    //  // Thrust Neurons neuron stays the same
    //  for (int i = 0; i < LAYER_SIZE_INPUT; i++)
    //  {
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 1 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 2 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //      C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 3 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
    //  }
    //}

    C->ID[idx] = idx;

// #pragma unroll
    // Set Bias Neurons
    for (int i = 0; i < SENSORS_BIAS_NEURON_AMOUNT; i++)
    {
        C->Neuron[(LAYER_SIZE_INPUT - SENSORS_BIAS_NEURON_AMOUNT) * 2 * CRAFT_COUNT + 2 * CRAFT_COUNT * i + idx] = 1.f;                 // Trainee
        C->Neuron[(LAYER_SIZE_INPUT - SENSORS_BIAS_NEURON_AMOUNT) * 2 * CRAFT_COUNT + 2 * CRAFT_COUNT * i + idx + CRAFT_COUNT] = 1.f;   // Opponent
    }
} // End init function

__device__ void Reset(CraftState* Crafts, int idx, GraphicsObjectPointer* Buffer, float PositionX, float PositionY, float AngleStart)
{
    Crafts->Position.X[idx] = PositionX;
    Crafts->Position.Y[idx] = PositionY;
    Crafts->Velocity.X[idx] = 0.f;
    Crafts->Velocity.Y[idx] = 0.f;
    Crafts->Acceleration.X[idx] = 0.f;
    Crafts->Acceleration.Y[idx] = 0.f;
    Crafts->Angle[idx] = AngleStart * PI / 180.f;
    Crafts->AngularVelocity[idx] = 0.f;
    Crafts->AngularAcceleration[idx] = 0.f;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        Crafts->Engine[i].Angle[idx] = 0.f;
        Crafts->Engine[i].AngularVelocity[idx] = 0.f;
        Crafts->Engine[i].AngularAcceleration[idx] = 0.f;
        Crafts->Engine[i].ThrustNormalized[idx] = THRUST_NORMALIZED_INITIAL;
        Crafts->Engine[i].ThrustNormalizedTemp[idx] = THRUST_NORMALIZED_INITIAL;
    }

    Crafts->Cannon.Angle[idx] = 0.f;
    Crafts->Cannon.AngularVelocity[idx] = 0.f;
    Crafts->Cannon.AngularAcceleration[idx] = 0.f;
    Crafts->BulletCounter[idx] = 0;
    Crafts->BulletTimer[idx] = 0;

#pragma unroll
    for (int i = 0; i < BULLET_COUNT_MAX; i++)
    {
        Crafts->Bullet[i].Active[idx] = false;
        //ConcealBullet(Buffer, idx, i);
    }

    Crafts->Score[idx] = 0.f;
    Crafts->ScoreTime[idx] = 0.f;
    Crafts->ScoreFuelEfficiency[idx] = 0.f;
    Crafts->ScoreBullet[idx] = 0.f;
    Crafts->ScoreDistance[idx] = 0.f;
    Crafts->Active[idx] = true;

    for (int i = 0; i < SENSORS_MEMORY_COUNT; i++)
        Crafts->Neuron[(SENSORS_MEMORY_START + i) * CRAFT_COUNT * 2 + idx] = 0.f;
}   // End Reset function

__global__ void ResetMatch(MatchState* Match, CraftState* Crafts, GraphicsObjectPointer* Buffer, int PositionNumber, float AngleStart)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    Match->Done[idx] = false;
    Match->ElapsedTicks[idx] = 0;

    if (PositionNumber == 0)
    {
        Reset(Crafts, idx, Buffer, -LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
        Reset(Crafts, idx + CRAFT_COUNT, Buffer, LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
    }
    else if (PositionNumber == 1)
    {
        Reset(Crafts, idx, Buffer, LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
        Reset(Crafts, idx + CRAFT_COUNT, Buffer, -LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
    }

    /*if (idx == 0)
    {
        printf("Trainee : Position X: %f, Position Y: %f\n", Crafts->Position.X[idx], Crafts->Position.Y[idx]);
        printf("Opponent: Position X: %f, Position Y: %f\n", Crafts->Position.X[idx + CRAFT_COUNT], Crafts->Position.Y[idx + CRAFT_COUNT]);
    }*/

    // TODO: Optimize this
    ConcealVertices(Buffer, idx, idx + CRAFT_COUNT);

    ShowVertices(Crafts, Buffer, idx, idx + CRAFT_COUNT);
}   // End reset function

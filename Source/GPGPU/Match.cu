// File Header
#include "Match.h"

// Standard Library
#include <cmath>

// CUDA
#include "cuda_runtime.h"
#include "curand_kernel.h"

// Project Headers
#include "GPGPU/Vertices.h"
#include "GPGPU/State.h"

__device__ void WeightsMutateAndTransfer(CraftState* C, config* Config, int SourceIndex, int TargetIndex)
{
	for (int i = 0; i < WEIGHT_COUNT; i++)
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
	for (int i = 0; i < CRAFT_COUNT; i++)
		if (i != idx)
			if (Crafts->Place[idx] == Crafts->Place[i])
				if (i > idx)
					atomicAdd(&(Crafts->Place[idx]), 1);					// Only compatible with compute >=6.0
					// TODO: Add one place to all crafts below it
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

__global__ void WeightsAndIDTempSave(CraftState* C, temp* Temp)	// TODO: Fix this. Higher place is better
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	if (C->Place[idx] < FIT_COUNT)
	{
		int PlaceID = C->Place[idx];

		for (int i = 0; i < WEIGHT_COUNT; i++)
			Temp->Weight[FIT_COUNT * i + PlaceID] = C->Weight[CRAFT_COUNT * i + idx];

		C->TempID[PlaceID] = C->ID[idx];
	}
}

// TODO: Combine transfer and mutate kernels
__global__ void WeightsAndIDTransfer(CraftState* C, temp* Temp)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	for (int i = 0; i < WEIGHT_COUNT; i++)
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
	Crafts->ScoreCumulative[idx] = 0;
	Crafts->ScoreCumulative[idx + CRAFT_COUNT] = 0;
}

__global__ void Init(CraftState* C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	curand_init(124, idx, 0, &(C->RandState[idx]));

	for (int i = 0; i < WEIGHT_COUNT; i++)
		C->Weight[CRAFT_COUNT * i + idx] = (curand_uniform(&C->RandState[idx]) - 0.5f) * 2.f * WEIGHTS_MULTIPLIER;

	//if (idx < CRAFT_COUNT)
	//{
	//	// Engine 1 Angle = -Engine 0 Angle
	//	// Engine 2 Angle =  Engine 0 Angle
	//	// Engine 3 Angle = -Engine 0 Angle
	//	for (int i = 0; i < 3 * LAYER_SIZE_HIDDEN; i++)
	//	{
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = -C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = -C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//	}
	//	// Brake output neuron stays the same
	//	for (int i = 3 * LAYER_SIZE_INPUT; i < 4 * LAYER_SIZE_INPUT; i++)
	//	{
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] =  C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//	}
	//	// Thrust Neurons neuron stays the same
	//	for (int i = 0; i < LAYER_SIZE_INPUT; i++)
	//	{
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 1 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 2 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//		C[WarpID]->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 3 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx] = C->Weight[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * CRAFT_COUNT + idx];
	//	}
	//}

	C->ID[idx] = idx;

#pragma unroll
	for (int i = 0; i < SENSORS_BIAS_NEURON_COUNT; i++)
		C->Neuron[(LAYER_SIZE_INPUT - SENSORS_BIAS_NEURON_COUNT) * 2 * CRAFT_COUNT + 2 * CRAFT_COUNT * i + idx] = 1.f;	// Bias Neuron
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

	Crafts->Score[idx] = 0;
	Crafts->ScoreTime[idx] = 0;
	Crafts->ScoreBullet[idx] = 0;
	Crafts->ScoreDistance[idx] = 0;
	Crafts->Active[idx] = true;

	for (int i = 0; i < SENSORS_MEMORY_COUNT; i++)
		Crafts->Neuron[(SENSORS_MEMORY_START + i) * CRAFT_COUNT * 2 + idx] = 0.f;
}	// End Reset function

__global__ void ResetMatch(MatchState* Match, CraftState* Crafts, GraphicsObjectPointer* Buffer, int PositionNumber, float AngleStart)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	if (idx == 0)
		Match->TournamentEpochNumber++;

	Match->Done[idx] = false;
	Match->ElapsedSteps[idx] = 0;

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

	// TODO: Optimize this
	ConcealVertices(Buffer, idx, idx + CRAFT_COUNT);

	ShowVertices(Crafts, Buffer, idx, idx + CRAFT_COUNT);
}	// End reset function
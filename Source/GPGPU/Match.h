#pragma once

// Standard Library
#include <cmath>

// CUDA
#include "cuda_runtime.h"
#include "curand_kernel.h"

// Project Headers
#include "GPGPU/Vertices.h"

__device__ void WeightsMutateAndTransfer(CraftPtrArr *C, config *Config, int SourceIndex, int TargetIndex)
{
	int SourceWarpID = SourceIndex / WARP_SIZE;
	int SourceID = SourceIndex % WARP_SIZE;

	int TargetWarpID = TargetIndex / WARP_SIZE;
	int TargetID = TargetIndex % WARP_SIZE;

	for (int i = 0; i < WEIGHT_COUNT; i++)
	{
		float Chance = curand_uniform(&C->Warp[SourceWarpID]->RandState[SourceID]);

		if (Chance < Config->MutationFlipChance)
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] = -C->Warp[SourceWarpID]->Weights[WARP_SIZE * i + SourceID];
		else if (Chance < Config->MutationFlipChance + Config->MutationScaleChance)
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID]
			= (1.f - Config->MutationScale + 2.f * Config->MutationScale * curand_uniform(&C->Warp[TargetWarpID]->RandState[TargetID])) * C->Warp[SourceWarpID]->Weights[WARP_SIZE * i + SourceID];
		else if (Chance < Config->MutationFlipChance + Config->MutationScaleChance + Config->MutationSlideChance)
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] = C->Warp[SourceWarpID]->Weights[WARP_SIZE * i + SourceID] + Config->MutationSigma * curand_normal(&C->Warp[SourceWarpID]->RandState[SourceID]);
		else
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] = C->Warp[SourceWarpID]->Weights[WARP_SIZE * i + SourceID];

		if (C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] > Config->WeightMax)
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] = Config->WeightMax;
		if (C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] < -Config->WeightMax)
			C->Warp[TargetWarpID]->Weights[WARP_SIZE * i + TargetID] = -Config->WeightMax;
	}
}

__global__ void RoundAssignPlace(CraftPtrArr *Crafts)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	Crafts->Warp[WarpID]->Place[ID] = 0;

	// Assign place to each craft
	for (int i = 0; i < CRAFT_COUNT; i++)
		if (i != idx)
			if (Crafts->Warp[WarpID]->ScoreCumulative[ID] < Crafts->Warp[i / WARP_SIZE]->ScoreCumulative[i % WARP_SIZE])
				Crafts->Warp[WarpID]->Place[ID]++;
}

__global__ void RoundTieFix(CraftPtrArr *Crafts)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	// Deal with score ties
	for (int i = 0; i < CRAFT_COUNT; i++)
		if (i != idx)
			if (Crafts->Warp[WarpID]->Place[ID] == Crafts->Warp[i / WARP_SIZE]->Place[i % WARP_SIZE])
				if (i > idx)
					atomicAdd(&(Crafts->Warp[WarpID]->Place[ID]), 1);					// Only compatible with compute >=6.0
}

__global__ void IDAssign(CraftPtrArr *C, config *Config)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = (idx + FIT_COUNT) / WARP_SIZE;
	int ID = (idx + FIT_COUNT) % WARP_SIZE;

	/*if (idx == 0)
		printf("Round Number: %d\n", Config->RoundNumber);*/

	C->Warp[WarpID]->ID[ID] = (Config->RoundNumber + 1) * FIT_COUNT + idx;
}

__global__ void IDTempSave(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	if (C->Warp[WarpID]->Place[ID] < FIT_COUNT)
	{
		int PlaceWarpID = C->Warp[WarpID]->Place[ID] / WARP_SIZE;
		int PlaceID = C->Warp[WarpID]->Place[ID] % WARP_SIZE;

		C->Warp[PlaceWarpID]->TempID[PlaceID] = C->Warp[WarpID]->ID[ID];
	}
}

__global__ void IDTransfer(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	C->Warp[WarpID]->ID[ID] = C->Warp[WarpID]->TempID[ID];
}

__global__ void WeightsTempSave(CraftPtrArr *C, TempPtrArr *Temp)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	if (C->Warp[WarpID]->Place[ID] < FIT_COUNT)
	{
		int PlaceWarpID = C->Warp[WarpID]->Place[ID] / WARP_SIZE;
		int PlaceID = C->Warp[WarpID]->Place[ID] % WARP_SIZE;

		for (int i = 0; i < WEIGHT_COUNT; i++)
			Temp->Warp[PlaceWarpID]->Weights[WARP_SIZE * i + PlaceID] = C->Warp[WarpID]->Weights[WARP_SIZE * i + ID];
	}
}

// TODO: Combine transfer and mutate kernels
__global__ void WeightsTransfer(CraftPtrArr *C, TempPtrArr *Temp)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;
	
	for (int i = 0; i < WEIGHT_COUNT; i++)
		C->Warp[WarpID]->Weights[WARP_SIZE * i + ID] = Temp->Warp[WarpID]->Weights[WARP_SIZE * i + ID];
}

__global__ void WeightsMutate(CraftPtrArr *C, config *Config)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

#pragma unroll
	for (int i = 1; i < CRAFT_COUNT / FIT_COUNT; i++)
		WeightsMutateAndTransfer(C, Config, idx, FIT_COUNT * i + idx);
}

__global__ void ScoreTempSave(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	int PlaceWarpID = C->Warp[WarpID]->Place[ID] / WARP_SIZE;
	int PlaceID = C->Warp[WarpID]->Place[ID] % WARP_SIZE;

	C->Warp[PlaceWarpID]->ScoreTemp[PlaceID] = C->Warp[WarpID]->ScoreCumulative[ID];
}

__global__ void ScoreTransfer(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	C->Warp[WarpID]->ScoreCumulative[ID] = C->Warp[WarpID]->ScoreTemp[ID];
}

__global__ void ResetScoreCumulative(CraftPtrArr *Crafts)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	// Reset Score Cumulative
	Crafts->Warp[WarpID]->ScoreCumulative[ID] = 0;
	Crafts->Warp[WarpID]->ScoreCumulative[ID + WARP_SIZE] = 0;
}

__global__ void Init(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	curand_init(124, idx, 0, &(C->Warp[WarpID]->RandState[ID]));

	for (int i = 0; i < WEIGHT_COUNT; i++)
		C->Warp[WarpID]->Weights[WARP_SIZE * i + ID] = (curand_uniform(&C->Warp[WarpID]->RandState[ID]) - 0.5f) * 2.f * WEIGHTS_MULTIPLIER;

	//if (idx < CRAFT_COUNT)
	//{
	//	// Engine 1 Angle = -Engine 0 Angle
	//	// Engine 2 Angle =  Engine 0 Angle
	//	// Engine 3 Angle = -Engine 0 Angle
	//	for (int i = 0; i < 3 * LAYER_SIZE_HIDDEN; i++)
	//	{
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] = -C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] =  C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] = -C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//	}
	//	// Brake output neuron stays the same
	//	for (int i = 3 * LAYER_SIZE_INPUT; i < 4 * LAYER_SIZE_INPUT; i++)
	//	{
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  4 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] =  C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN +  8 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] =  C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 12 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] =  C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//	}
	//	// Thrust Neurons neuron stays the same
	//	for (int i = 0; i < LAYER_SIZE_INPUT; i++)
	//	{
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 1 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] = C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 2 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] = C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//		C[WarpID]->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + 3 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx] = C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + 21 * LAYER_SIZE_HIDDEN + i) * WARP_SIZE + idx];
	//	}
	//}

	C->Warp[WarpID]->ID[ID] = idx;

#pragma unroll
	for (int i = 0; i < 2; i++)
		C->Warp[WarpID]->Neuron[(LAYER_SIZE_INPUT - 1) * WARP_SIZE * 2 + ID + WARP_SIZE * i] = 1.f;	// Bias Neuron
} // End init function

__device__ void Reset(CraftState *Crafts, int WarpID, int idx, GraphicsObjectPointer Buffer, float PositionX, float PositionY, float AngleStart)
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
		//ConcealBullet(Buffer, WarpID, idx, i);
	}

	Crafts->Score[idx] = 0;
	Crafts->ScoreTime[idx] = 0;
	Crafts->ScoreBullet[idx] = 0;
	Crafts->ScoreDistance[idx] = 0;
	Crafts->Active[idx] = true;

	for (int i = 0; i < SENSORS_MEMORY_COUNT; i++)
		Crafts->Neuron[(SENSORS_MEMORY_START + i) * WARP_SIZE * 2 + idx] = 0.f;
}	// End Reset function

__global__ void ResetMatch(MatchState *Match, CraftPtrArr *Crafts, GraphicsObjectPointer *Buffer, int PositionNumber, float AngleStart)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int ID = idx % WARP_SIZE;

	if (idx == 0)
		Match->TournamentEpochNumber++;

	Match->Done[idx] = false;
	Match->ElapsedSteps[idx] = 0;

	if (PositionNumber == 0)
	{
		Reset(Crafts->Warp[WarpID], WarpID, ID, *Buffer, -LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
		Reset(Crafts->Warp[WarpID], WarpID, ID + WARP_SIZE, *Buffer, LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
	}
	else if (PositionNumber == 1)
	{
		Reset(Crafts->Warp[WarpID], WarpID, ID, *Buffer, LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
		Reset(Crafts->Warp[WarpID], WarpID, ID + WARP_SIZE, *Buffer, -LIFE_RADIUS * 2.f / 3.f, 0.f, AngleStart);
	}

	// TODO: Optimize this
	ConcealVertices(*Buffer, WarpID, ID, ID + WARP_SIZE);

	ShowVertices(Crafts->Warp[WarpID], *Buffer, WarpID, ID, ID + WARP_SIZE);
}	// End reset function

#pragma once

// CUDA
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Headers
#include "Config.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/Match.h"
#include "GPGPU/NeuralNet.h"
#include "GPGPU/Vertices.h"
#include "GPGPU/Physic.h"
#include "GPGPU/State.h"

//namespace GPGPU
//{
__global__ void RunEpoch(MatchState *Match, CraftPtrArr *C, GraphicsObjectPointer *Buffer, config *Config, int OpponentID)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int id = idx % WARP_SIZE;

	// Process a few physics time steps
	for (int TimeStepIteration = 0; TimeStepIteration < Config->IterationsPerCall && !Match->Done[idx]; TimeStepIteration++)
	{
		if (Match->ElapsedSteps[idx] % (FRAMERATE_NN_PHYSICS) == 0)
		{
			if (C->Warp[WarpID]->Active[id])
				NeuralNet(C->Warp[WarpID], *Buffer, WarpID, id,				false,	0,						C->Warp[OpponentID / WARP_SIZE]);	// TODO: Just make weights to read from a float*
			if (C->Warp[WarpID]->Active[id + WARP_SIZE])
				NeuralNet(C->Warp[WarpID], *Buffer, WarpID, id + WARP_SIZE,	true,	OpponentID % WARP_SIZE, C->Warp[OpponentID / WARP_SIZE]);
		}

		if (C->Warp[WarpID]->Active[id])
			Physic(Match, C->Warp[WarpID], id, Config);
		if (C->Warp[WarpID]->Active[id + WARP_SIZE])
			Physic(Match, C->Warp[WarpID], id + WARP_SIZE, Config);

		if (C->Warp[WarpID]->Active[id])
			CollisionDetect(C->Warp[WarpID], id, id + WARP_SIZE);

		// TODO: Convert to single function
		BulletMechanics(*Buffer, C->Warp[WarpID], WarpID, id, id + WARP_SIZE, Config);
		BulletMechanics(*Buffer, C->Warp[WarpID], WarpID, id + WARP_SIZE, id, Config);

		Match->ElapsedSteps[idx]++;

		if (Match->ElapsedSteps[idx] > Config->TimeStepLimit)
			Match->Done[idx] = true;
		else  if (!C->Warp[WarpID]->Active[id])
			Match->Done[idx] = true;

		if (Match->Done[idx])
		{
			ConcealVertices(*Buffer, WarpID, id, id + WARP_SIZE);
			C->Warp[WarpID]->Score[id]				= C->Warp[WarpID]->ScoreTime[id]			 + C->Warp[WarpID]->ScoreDistance[id] / 1000				+ C->Warp[WarpID]->ScoreBullet[id]				- C->Warp[WarpID]->ScoreBullet[id + WARP_SIZE] / 4;
			C->Warp[WarpID]->Score[id + WARP_SIZE]	= C->Warp[WarpID]->ScoreTime[id + WARP_SIZE] + C->Warp[WarpID]->ScoreDistance[id + WARP_SIZE] / 1000	+ C->Warp[WarpID]->ScoreBullet[id + WARP_SIZE]	- C->Warp[WarpID]->ScoreBullet[id] / 4;	// Score of opponent does not matter
		}
	} // End main step iteration for loop 

	// If craft is going from rendering to hidden, conceal its vertices
	if (Match->RenderOffFirstFrame[idx] == true)
	{
		ConcealVertices(*Buffer, WarpID, id, id + WARP_SIZE);
		Match->RenderOffFirstFrame[idx] = false;
	}

	// If craft is going from hidden to rendering, create its vertices
	if (Match->RenderOnFirstFrame[idx] == true)
	{
		ShowVertices(C->Warp[WarpID], *Buffer, WarpID, id, id + WARP_SIZE);
		Match->RenderOnFirstFrame[idx] = false;
	}

	// If match has render turned on, process its vertices
	if (Match->RenderOn[idx] && !Match->Done[idx])
		GraphicsProcessing(C->Warp[WarpID], *Buffer, WarpID, id, id + WARP_SIZE);

	if (!Match->Done[idx])
		Match->AllDone = false;
}	// End RunEpoch function	

__global__ void ScoreCumulativeCalc(CraftPtrArr *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int WarpID = idx / WARP_SIZE;
	int id = idx % WARP_SIZE;

	C->Warp[WarpID]->ScoreCumulative[id] += C->Warp[WarpID]->Score[id];
	C->Warp[WarpID]->ScoreCumulative[id + WARP_SIZE] += C->Warp[WarpID]->Score[id + WARP_SIZE];
}
//} // End namespace GPGPU

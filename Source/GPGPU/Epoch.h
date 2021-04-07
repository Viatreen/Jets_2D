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
__global__ void RunEpoch(MatchState *Match, CraftState *C, GraphicsObjectPointer *Buffer, config *Config, int OpponentID)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	// Process a few physics time steps
	for (int TimeStepIteration = 0; TimeStepIteration < Config->IterationsPerCall && !Match->Done[idx]; TimeStepIteration++)
	{
		if (Match->ElapsedSteps[idx] % (FRAMERATE_NN_PHYSICS) == 0)
		{
			if (C->Active[idx])
				NeuralNet(C, *Buffer, idx,				 false,	0,          &C[0]);	// TODO: Just make weights to read from a float*
			if (C->Active[idx + CRAFT_COUNT])
				NeuralNet(C, *Buffer, idx + CRAFT_COUNT, true,	OpponentID, &C[0]);
		}

		if (C->Active[idx] )
			Physic(Match, C, idx, Config);
		if (C->Active[idx + CRAFT_COUNT])
			Physic(Match, C, idx + CRAFT_COUNT, Config);

		if (C->Active[idx] )
			CollisionDetect(C, idx, idx + CRAFT_COUNT);

		// TODO: Convert to single function
		BulletMechanics(*Buffer, C, idx, idx + CRAFT_COUNT, Config);
		BulletMechanics(*Buffer, C, idx + CRAFT_COUNT, idx, Config);

		Match->ElapsedSteps[idx]++;

		if (Match->ElapsedSteps[idx] > Config->TimeStepLimit)
			Match->Done[idx] = true;
		else  if (!C->Active[idx] )
			Match->Done[idx] = true;

		if (Match->Done[idx])
		{
			ConcealVertices(*Buffer, idx, idx + CRAFT_COUNT);
			C->Score[idx] 				= C->ScoreTime[idx] 			  + C->ScoreDistance[idx]  / 1000				+ C->ScoreBullet[idx] 				- C->ScoreBullet[idx + CRAFT_COUNT] / 4;
			C->Score[idx + CRAFT_COUNT]	= C->ScoreTime[idx + CRAFT_COUNT] + C->ScoreDistance[idx + CRAFT_COUNT] / 1000	+ C->ScoreBullet[idx + CRAFT_COUNT]	- C->ScoreBullet[idx]  / 4;	// Score of opponent does not matter
		}
	} // End main step iteration for loop 

	// If craft is going from rendering to hidden, conceal its vertices
	if (Match->RenderOffFirstFrame[idx] == true)
	{
		ConcealVertices(*Buffer, idx, idx + CRAFT_COUNT);
		Match->RenderOffFirstFrame[idx] = false;
	}

	// If craft is going from hidden to rendering, create its vertices
	if (Match->RenderOnFirstFrame[idx] == true)
	{
		ShowVertices(C, *Buffer, idx, idx + CRAFT_COUNT);
		Match->RenderOnFirstFrame[idx] = false;
	}

	// If match has render turned on, process its vertices
	if (Match->RenderOn[idx] && !Match->Done[idx])
		GraphicsProcessing(C, *Buffer, idx, idx + CRAFT_COUNT);

	if (!Match->Done[idx])
		Match->AllDone = false;
}	// End RunEpoch function	

__global__ void ScoreCumulativeCalc(CraftState *C)
{
	int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

	C->ScoreCumulative[idx]				  += C->Score[idx];
	C->ScoreCumulative[idx + CRAFT_COUNT] += C->Score[idx + CRAFT_COUNT];
}
//} // End namespace GPGPU

#pragma once

#ifndef _DEBUG
#define _DEBUG
#endif

// CUDA
#include "cuda_runtime.h"

// Project Headers
#include "GPGPU/State.h"
#include "GPGPU/Vertices.h"

__device__ void Physic(MatchState *Match, CraftState *CS, int IdxCraft, config *Config)
{
#ifdef _DEBUG
	if (CS->Position.X[IdxCraft] != CS->Position.X[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Position X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Position.X[IdxCraft]);
	if (CS->Position.Y[IdxCraft] != CS->Position.Y[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Position Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Position.Y[IdxCraft]);
	if (CS->Velocity.X[IdxCraft] != CS->Velocity.X[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Velocity X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Velocity.X[IdxCraft]);
	if (CS->Velocity.Y[IdxCraft] != CS->Velocity.Y[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Velocity Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Velocity.Y[IdxCraft]);
	if (CS->Acceleration.X[IdxCraft] != CS->Acceleration.X[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Acceleration X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Acceleration.X[IdxCraft]);
	if (CS->Acceleration.Y[IdxCraft] != CS->Acceleration.Y[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Acceleration Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Acceleration.Y[IdxCraft]);
	if (CS->Angle[IdxCraft] != CS->Angle[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Angle NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Angle[IdxCraft]);
	if (CS->AngularVelocity[IdxCraft] != CS->AngularVelocity[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Angular Velocity NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->AngularVelocity[IdxCraft]);
	if (CS->AngularAcceleration[IdxCraft] != CS->AngularAcceleration[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), AngularAcceleration NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->AngularAcceleration[IdxCraft]);
	if (CS->Cannon.Angle[IdxCraft] != CS->Cannon.Angle[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Cannon Angle NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.Angle[IdxCraft]);
	if (CS->Cannon.AngularVelocity[IdxCraft] != CS->Cannon.AngularVelocity[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Cannon Angle Vel NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.AngularVelocity[IdxCraft]);
	if (CS->Cannon.AngularAcceleration[IdxCraft] != CS->Cannon.AngularAcceleration[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Cannon Angle Acc NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.AngularAcceleration[IdxCraft]);
	if (CS->Score[IdxCraft] != CS->Score[IdxCraft])
		printf("Physic - Before- Craft(%d), ES(%d), Score NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Score[IdxCraft]);

	for (int i = 0; i < 4; i++)
	{
		if (CS->Engine[i].Angle[IdxCraft]	!= CS->Engine[i].Angle[IdxCraft])
			printf("Physic - Before- Craft(%d), Eng(%d), ES(%d), Angle NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].Angle[IdxCraft]);
		if (CS->Engine[i].AngularVelocity[IdxCraft] != CS->Engine[i].AngularVelocity[IdxCraft])
			printf("Physic - Before- Craft(%d), Eng(%d), ES(%d), Angle Vel NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].AngularVelocity[IdxCraft]);
		if (CS->Engine[i].AngularAcceleration[IdxCraft] != CS->Engine[i].AngularAcceleration[IdxCraft])
			printf("Physic - Before- Craft(%d), Eng(%d), ES(%d), Angle Acc NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].AngularAcceleration[IdxCraft]);
		if (CS->Engine[i].Thrust[IdxCraft] != CS->Engine[i].Thrust[IdxCraft])
			printf("Physic - Before- Craft(%d), Eng(%d), ES(%d), Thrust NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].Thrust[IdxCraft]);
		if (CS->Engine[i].ThrustNormalized[IdxCraft] != CS->Engine[i].ThrustNormalized[IdxCraft])
			printf("Physic - Before- Craft(%d), Eng(%d), ES(%d), Thrust Norm NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].ThrustNormalized[IdxCraft]);
	}
#endif

#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		////////////////////////////////////////////////////////////////////////////////////
		//// Process new thrust
		// Check for limits
		if (CS->Engine[i].ThrustNormalized[IdxCraft] < THRUST_MIN)
			CS->Engine[i].ThrustNormalized[IdxCraft] = THRUST_MIN;
		else if (CS->Engine[i].ThrustNormalized[IdxCraft] > 1.f)
			CS->Engine[i].ThrustNormalized[IdxCraft] = 1.f;

		// Ensure accurate throttle response
		if (CS->Engine[i].ThrustNormalized[IdxCraft] - CS->Engine[i].ThrustNormalizedTemp[IdxCraft] > THRUST_RAMP_MAX)
			CS->Engine[i].ThrustNormalized[IdxCraft] = CS->Engine[i].ThrustNormalizedTemp[IdxCraft] + THRUST_RAMP_MAX;
		else if (CS->Engine[i].ThrustNormalized[IdxCraft] - CS->Engine[i].ThrustNormalizedTemp[IdxCraft] < -THRUST_RAMP_MAX)
			CS->Engine[i].ThrustNormalized[IdxCraft] = CS->Engine[i].ThrustNormalizedTemp[IdxCraft] - THRUST_RAMP_MAX;

		CS->Engine[i].Thrust[IdxCraft] = CS->Engine[i].ThrustNormalized[IdxCraft] * THRUST_MAX;

		CS->Engine[i].ThrustNormalizedTemp[IdxCraft] = CS->Engine[i].ThrustNormalized[IdxCraft];

		////////////////////////////////////////////////////////////////////////////////////
		// Process engine rotation
		// Check for limits

	}

	// Craft change in acceleration, velocity, and position
	CS->Acceleration.X[IdxCraft] = 0.f;
	CS->Acceleration.Y[IdxCraft] = 0.f;

	// TODO: Make this more efficient by removing loop
#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		CS->Acceleration.X[IdxCraft] += CS->Engine[i].Thrust[IdxCraft] * cos(CS->Angle[IdxCraft] + CS->Engine[i].Angle[IdxCraft] + PI / 2.f);
		CS->Acceleration.Y[IdxCraft] += CS->Engine[i].Thrust[IdxCraft] * sin(CS->Angle[IdxCraft] + CS->Engine[i].Angle[IdxCraft] + PI / 2.f);
	}
	CS->Acceleration.X[IdxCraft] *= CRAFT_MASS_INVERSE;
	CS->Acceleration.Y[IdxCraft] = CS->Acceleration.Y[IdxCraft] * CRAFT_MASS_INVERSE - GRAVITY;

	CS->Velocity.X[IdxCraft] += CS->Acceleration.X[IdxCraft] * TIME_STEP;
	CS->Velocity.Y[IdxCraft] += CS->Acceleration.Y[IdxCraft] * TIME_STEP;

	CS->Position.X[IdxCraft] += CS->Velocity.X[IdxCraft] * TIME_STEP;
	CS->Position.Y[IdxCraft] += CS->Velocity.Y[IdxCraft] * TIME_STEP;

	// Craft distance from path
	float Distance = __fsqrt_ru(CS->Position.X[IdxCraft] * CS->Position.X[IdxCraft] + CS->Position.Y[IdxCraft] * CS->Position.Y[IdxCraft]);

	if (Distance > (LIFE_RADIUS - FUSELAGE_RADIUS) || Distance != Distance)
	{
		CS->Active[IdxCraft] = false;
	}
	else
	{
		CS->ScoreTime[IdxCraft]++;
		//CS->ScoreDistance[IdxCraft] += int(1000.f * pow((LIFE_RADIUS - Distance) / LIFE_RADIUS, 2.f));

#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			CS->Engine[i].AngularVelocity[IdxCraft]	+= CS->Engine[i].AngularAcceleration[IdxCraft] * TIME_STEP;			
			CS->Engine[i].Angle[IdxCraft]			+= CS->Engine[i].AngularVelocity[IdxCraft] * TIME_STEP;

			while (CS->Engine[i].Angle[IdxCraft] > PI)
				CS->Engine[i].Angle[IdxCraft] -= 2 * PI;
			while (CS->Engine[i].Angle[IdxCraft] < -PI)
				CS->Engine[i].Angle[IdxCraft] += 2 * PI;

			if (i == 0 || i == 2)
			{
				if (CS->Engine[i].Angle[IdxCraft] > ENGINE_ANGLE_MAX_IN)
				{
					CS->Engine[i].Angle[IdxCraft] = ENGINE_ANGLE_MAX_IN;
					CS->Engine[i].AngularVelocity[IdxCraft] = 0;
				}
				else if (CS->Engine[i].Angle[IdxCraft] < -ENGINE_INBOARD_ANGLE_MAX_OUT)
				{
					CS->Engine[i].Angle[IdxCraft] = -ENGINE_INBOARD_ANGLE_MAX_OUT;
					CS->Engine[i].AngularVelocity[IdxCraft] = 0;
				}
			}
			else
			{
				if (CS->Engine[i].Angle[IdxCraft] > ENGINE_INBOARD_ANGLE_MAX_OUT)
				{
					CS->Engine[i].Angle[IdxCraft] = ENGINE_INBOARD_ANGLE_MAX_OUT;
					CS->Engine[i].AngularVelocity[IdxCraft] = 0;
				}
				else if (CS->Engine[i].Angle[IdxCraft] < -ENGINE_ANGLE_MAX_IN)
				{
					CS->Engine[i].Angle[IdxCraft] = -ENGINE_ANGLE_MAX_IN;
					CS->Engine[i].AngularVelocity[IdxCraft] = 0;
				}
			}
		}

		// TODO: Start using instant center of rotation instead of just CG
		// TODO: Optimize function by not using "+ PI/2" in the trig functions. Use trig equivalents

		// Whole craft angle
		CS->AngularAcceleration[IdxCraft] =
			(CS->Engine[0].Thrust[IdxCraft] * __sinf(CS->Engine[0].Angle[IdxCraft] + PI / 2.f) * ENGINE_0_DISTANCE
				+ CS->Engine[1].Thrust[IdxCraft] * __sinf(CS->Engine[1].Angle[IdxCraft] + PI / 2.f) * ENGINE_1_DISTANCE
				+ CS->Engine[2].Thrust[IdxCraft] * __sinf(CS->Engine[2].Angle[IdxCraft] + PI / 2.f) * ENGINE_2_DISTANCE
				+ CS->Engine[3].Thrust[IdxCraft] * __sinf(CS->Engine[3].Angle[IdxCraft] + PI / 2.f) * ENGINE_3_DISTANCE
				+ (CS->Engine[0].Thrust[IdxCraft] * __cosf(CS->Engine[0].Angle[IdxCraft] + PI / 2.f)
					+  CS->Engine[1].Thrust[IdxCraft] * __cosf(CS->Engine[1].Angle[IdxCraft] + PI / 2.f)
					+  CS->Engine[2].Thrust[IdxCraft] * __cosf(CS->Engine[2].Angle[IdxCraft] + PI / 2.f)
					+  CS->Engine[3].Thrust[IdxCraft] * __cosf(CS->Engine[3].Angle[IdxCraft] + PI / 2.f)) * CG_OFFSET_Y)
			* CRAFT_ROTATIONAL_INERTIA_INVERSE;

		CS->AngularVelocity[IdxCraft]	+= CS->AngularAcceleration[IdxCraft] * TIME_STEP;
		CS->Angle[IdxCraft]				+= CS->AngularVelocity[IdxCraft] * TIME_STEP;
		
		while (CS->Angle[IdxCraft] > PI)
		{
			CS->Angle[IdxCraft] -= 2 * PI;
			//CS->ScoreTime[IdxCraft] -= 200;
		}
		while (CS->Angle[IdxCraft] < -PI)
		{
			CS->Angle[IdxCraft] += 2 * PI;
			//CS->ScoreTime[IdxCraft] -= 200;
		}

		// Canon
		CS->Cannon.AngularVelocity[IdxCraft] += CS->Cannon.AngularAcceleration[IdxCraft] * TIME_STEP;

		if (CS->Cannon.AngularVelocity[IdxCraft] >  CANNON_VELOCITY_MAX)
			CS->Cannon.AngularVelocity[IdxCraft] =  CANNON_VELOCITY_MAX;
		if (CS->Cannon.AngularVelocity[IdxCraft] < -CANNON_VELOCITY_MAX)
			CS->Cannon.AngularVelocity[IdxCraft] = -CANNON_VELOCITY_MAX;

		CS->Cannon.Angle[IdxCraft] += CS->Cannon.AngularVelocity[IdxCraft] * TIME_STEP;

		while (CS->Cannon.Angle[IdxCraft] > PI)
			CS->Cannon.Angle[IdxCraft] -= 2 * PI;
		while (CS->Cannon.Angle[IdxCraft] < -PI)
			CS->Cannon.Angle[IdxCraft] += 2 * PI;
	}

#ifdef _DEBUG
	if (CS->Position.X[IdxCraft] != CS->Position.X[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Position X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Position.X[IdxCraft]);
	if (CS->Position.Y[IdxCraft] != CS->Position.Y[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Position Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Position.Y[IdxCraft]);
	if (CS->Velocity.X[IdxCraft] != CS->Velocity.X[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Velocity X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Velocity.X[IdxCraft]);
	if (CS->Velocity.Y[IdxCraft] != CS->Velocity.Y[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Velocity Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Velocity.Y[IdxCraft]);
	if (CS->Acceleration.X[IdxCraft] != CS->Acceleration.X[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Acceleration X NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Acceleration.X[IdxCraft]);
	if (CS->Acceleration.Y[IdxCraft] != CS->Acceleration.Y[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Acceleration Y NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Acceleration.Y[IdxCraft]);
	if (CS->Angle[IdxCraft] != CS->Angle[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Angle NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Angle[IdxCraft]);
	if (CS->AngularVelocity[IdxCraft] != CS->AngularVelocity[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Angular Velocity NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->AngularVelocity[IdxCraft]);
	if (CS->AngularAcceleration[IdxCraft] != CS->AngularAcceleration[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), AngularAcceleration NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->AngularAcceleration[IdxCraft]);
	if (CS->Cannon.Angle[IdxCraft] != CS->Cannon.Angle[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Cannon Angle NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.Angle[IdxCraft]);
	if (CS->Cannon.AngularVelocity[IdxCraft] != CS->Cannon.AngularVelocity[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Cannon Angle Vel NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.AngularVelocity[IdxCraft]);
	if (CS->Cannon.AngularAcceleration[IdxCraft] != CS->Cannon.AngularAcceleration[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Cannon Angle Acc NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Cannon.AngularAcceleration[IdxCraft]);
	if (CS->Score[IdxCraft] != CS->Score[IdxCraft])
		printf("Physic - After- Craft(%d), ES(%d), Score NaN, %f\n", IdxCraft, Match->ElapsedSteps[IdxCraft], CS->Score[IdxCraft]);

	for (int i = 0; i < 4; i++)
	{
		if (CS->Engine[i].Angle[IdxCraft]	!= CS->Engine[i].Angle[IdxCraft])
			printf("Physic - After- Craft(%d), Eng(%d), ES(%d), Angle NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].Angle[IdxCraft]);
		if (CS->Engine[i].AngularVelocity[IdxCraft] != CS->Engine[i].AngularVelocity[IdxCraft])
			printf("Physic - After- Craft(%d), Eng(%d), ES(%d), Angle Vel NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].AngularVelocity[IdxCraft]);
		if (CS->Engine[i].AngularAcceleration[IdxCraft] != CS->Engine[i].AngularAcceleration[IdxCraft])
			printf("Physic - After- Craft(%d), Eng(%d), ES(%d), Angle Acc NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].AngularAcceleration[IdxCraft]);
		if (CS->Engine[i].Thrust[IdxCraft] != CS->Engine[i].Thrust[IdxCraft])
			printf("Physic - After- Craft(%d), Eng(%d), ES(%d), Thrust NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].Thrust[IdxCraft]);
		if (CS->Engine[i].ThrustNormalized[IdxCraft] != CS->Engine[i].ThrustNormalized[IdxCraft])
			printf("Physic - After- Craft(%d), Eng(%d), ES(%d), Thrust Norm NaN, %f\n", IdxCraft, i, Match->ElapsedSteps[IdxCraft], CS->Engine[i].ThrustNormalized[IdxCraft]);
	}
#endif
	} // End physic function

	// TODO: Fix this
	  ////////////////////////////////////////////////////
	  //// Craft on craft collision detection and response
__device__ void CollisionDetect(CraftState *C, int idx1, int idx2)
{
	// Check if fuselages are close to each other with simple math 
	if (fabs(C->Position.X[idx1] - C->Position.X[idx2]) < 2 * FUSELAGE_RADIUS && fabs(C->Position.Y[idx1] - C->Position.Y[idx2]) < 2 * FUSELAGE_RADIUS)
	{
		// Check for collision
		//__fsqrt_ru
		float DistanceSquared = pow(C->Position.X[idx1] - C->Position.X[idx2], 2.f) + pow(C->Position.Y[idx1] - C->Position.Y[idx2], 2.f);
		if (DistanceSquared < 2 * FUSELAGE_RADIUS * 2 * FUSELAGE_RADIUS)
		{
			float Distance = __fsqrt_ru(DistanceSquared);

			if (Distance < 0.01f)
			{
				Distance = 0.01f;
				C->Position.X[idx1] -= 0.01f;
			}

			//Velocity-Position Dot Product
			float DotProductVelPos = (C->Velocity.X[idx1] - C->Velocity.X[idx2]) * (C->Position.X[idx1] - C->Position.X[idx2]) + (C->Velocity.Y[idx1] - C->Velocity.Y[idx2]) * (C->Position.Y[idx1] - C->Position.Y[idx2]);

			float DistanceHorizontal	= C->Position.X[idx1] - C->Position.X[idx2];
			float DistanceVertical		= C->Position.Y[idx1] - C->Position.Y[idx2];

			C->Velocity.X[idx1] = C->Velocity.X[idx1] - DotProductVelPos / DistanceSquared * DistanceHorizontal;
			C->Velocity.Y[idx1] = C->Velocity.Y[idx1] - DotProductVelPos / DistanceSquared * DistanceVertical;

			C->Velocity.X[idx2] = C->Velocity.X[idx2] + DotProductVelPos / DistanceSquared * DistanceHorizontal;
			C->Velocity.Y[idx2] = C->Velocity.Y[idx2] + DotProductVelPos / DistanceSquared * DistanceVertical;

			// Find vector between each fuselage and normalize it
			float NormalVectorX = (C->Position.X[idx1] - C->Position.X[idx2]) / Distance;
			float NormalVectorY = (C->Position.Y[idx1] - C->Position.Y[idx2]) / Distance;

			// Enforce no overlap. Could be more accurate by splitting previous timestep into pre and post collision, but this saves on compute
			// TODO: Check if += and -= are in the correct place
			float Overlap = (2.f * FUSELAGE_RADIUS - Distance) / 2.f + 0.1f;	// Amount to move each ball so they are no longer overlapping

			C->Position.X[idx1] += Overlap * NormalVectorX;
			C->Position.Y[idx1] += Overlap * NormalVectorY;

			C->Position.X[idx2] -= Overlap * NormalVectorX;
			C->Position.Y[idx2] -= Overlap * NormalVectorY;

			// For formula, see Wiki - Elastic Collision
		}
	}	// End check for collision detection
}	// End CollisionDetect function

__device__ void BulletMechanics(GraphicsObjectPointer Buffer, CraftState *CS, int WarpID, int ID1, int ID2, config *Config)
{
	for (int i = 0; i < BULLET_COUNT_MAX; i++)
	{
		if (CS->Bullet[i].Active[ID1])
		{
			CS->Bullet[i].Position.X[ID1] += CS->Bullet[i].Velocity.X[ID1] * TIME_STEP;
			CS->Bullet[i].Velocity.Y[ID1] -= GRAVITY * TIME_STEP;
			CS->Bullet[i].Position.Y[ID1] += CS->Bullet[i].Velocity.Y[ID1] * TIME_STEP;

			// TODO: Turn on friendly fire
			float DistanceSquared = pow(CS->Bullet[i].Position.X[ID1], 2.f) + pow(CS->Bullet[i].Position.Y[ID1], 2.f);
			if (DistanceSquared > pow(LIFE_RADIUS - BULLET_RADIUS, 2.f))
			{
				CS->Bullet[i].Active[ID1] = false;
				CS->BulletCounter[ID1]--;
				ConcealBullet(Buffer, WarpID, ID1, i);
			}
			else if (pow(CS->Bullet[i].Position.X[ID1] - CS->Position.X[ID2], 2.f)
				+ pow(CS->Bullet[i].Position.Y[ID1] - CS->Position.Y[ID2], 2.f)
				< pow(FUSELAGE_RADIUS + BULLET_RADIUS, 2.f))
			{
				CS->Bullet[i].Active[ID1] = false;
				CS->BulletCounter[ID1]--;
				ConcealBullet(Buffer, WarpID, ID1, i);
				CS->ScoreBullet[ID1] += Config->BulletDamage;		// Score increases more than total possible boundary score
			}
		}
	}
}

__device__ void ShootBullet(CraftState *CS, int WarpID, int ID, GraphicsObjectPointer Buffer)
{
	// Check which bullets are avialable to be launched
	int i = 0;
	while (CS->Bullet[i].Active[ID]) { i++; }
	CS->Bullet[i].Active[ID] = true;

	float ComponentX = __cosf(CS->Angle[ID] + CS->Cannon.Angle[ID] + PI/2);
	float ComponentY = __sinf(CS->Angle[ID] + CS->Cannon.Angle[ID] + PI/2);
	CS->Bullet[i].Position.X[ID] = (FUSELAGE_RADIUS + 3 * BULLET_RADIUS) * ComponentX + CS->Position.X[ID];
	CS->Bullet[i].Position.Y[ID] = (FUSELAGE_RADIUS + 3 * BULLET_RADIUS) * ComponentY + CS->Position.Y[ID];

	CS->Bullet[i].Velocity.X[ID] = BULLET_VELOCITY_INITIAL * ComponentX;
	CS->Bullet[i].Velocity.Y[ID] = BULLET_VELOCITY_INITIAL * ComponentY;

	ShowBullet(Buffer, WarpID, ID, i);

	CS->BulletTimer[ID]			 = 0;
	CS->BulletCounter[ID]++;
}
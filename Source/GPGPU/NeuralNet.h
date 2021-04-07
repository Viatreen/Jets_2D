#pragma once

// Standard Library
#include "math.h"

// CUDA
#include "cuda_runtime.h"
#include "crt/math_functions.h" 

// Project Headers
#include "Config.h"
#include "GPGPU/State.h"
#include "GPGPU/Physic.h"


__device__ void NeuralNet(CraftState *C, GraphicsObjectPointer Buffer, int ID, bool IsOpponent, int OpponentID, CraftState *C_Opponent)
{
	int IdxOpponent;
	if (IsOpponent)
		IdxOpponent = ID - CRAFT_COUNT;
	else
		IdxOpponent = ID + CRAFT_COUNT;

#ifdef _DEBUGs
	if (C->Position.X[ID] != C->Position.X[ID])
		printf("NN Before- Craft(%d), Position X NaN, %f\n", ID, C->Position.X[ID]);
	if (C->Position.Y[ID] != C->Position.Y[ID])
		printf("NN Before- Craft(%d), Position Y NaN, %f\n", ID, C->Position.Y[ID]);
	if (C->Velocity.X[ID] != C->Velocity.X[ID])
		printf("NN Before- Craft(%d), Velocity X NaN, %f\n", ID, C->Velocity.X[ID]);
	if (C->Velocity.Y[ID] != C->Velocity.Y[ID])
		printf("NN Before- Craft(%d), Velocity Y NaN, %f\n", ID, C->Velocity.Y[ID]);
	if (C->Acceleration.X[ID] != C->Acceleration.X[ID])
		printf("NN Before- Craft(%d), Acceleration X NaN, %f\n", ID, C->Acceleration.X[ID]);
	if (C->Acceleration.Y[ID] != C->Acceleration.Y[ID])
		printf("NN Before- Craft(%d), Acceleration Y NaN, %f\n", ID, C->Acceleration.Y[ID]);
	if (C->Angle[ID] != C->Angle[ID])
		printf("NN Before- Craft(%d), Angle NaN, %f\n", ID, C->Angle[ID]);
	if (C->AngularVelocity[ID] != C->AngularVelocity[ID])
		printf("NN Before- Craft(%d), Angular Velocity NaN, %f\n", ID, C->AngularVelocity[ID]);
	if (C->AngularAcceleration[ID] != C->AngularAcceleration[ID])
		printf("NN Before- Craft(%d), AngularAcceleration NaN, %f\n", ID, C->AngularAcceleration[ID]);
	if (C->Cannon.Angle[ID] != C->Cannon.Angle[ID])
		printf("NN Before- Craft(%d), Cannon Angle NaN, %f\n", ID, C->Cannon.Angle[ID]);
	if (C->Cannon.AngularVelocity[ID] != C->Cannon.AngularVelocity[ID])
		printf("NN Before- Craft(%d), Cannon Angle Vel NaN, %f\n", ID, C->Cannon.AngularVelocity[ID]);
	if (C->Cannon.AngularAcceleration[ID] != C->Cannon.AngularAcceleration[ID])
		printf("NN Before- Craft(%d), Cannon Angle Acc NaN, %f\n", ID, C->Cannon.AngularAcceleration[ID]);
	if (C->Score[ID] != C->Score[ID])
		printf("NN Before- Craft(%d), Score NaN, %f\n", ID, C->Score[ID]);

	for (int i = 0; i < 4; i++)
	{
		if (C->Engine[i].Angle[ID] != C->Engine[i].Angle[ID])
			printf("NN Before- Craft(%d), Eng(%d), Angle NaN, %f\n", ID, i, C->Engine[i].Angle[ID]);
		if (C->Engine[i].AngularVelocity[ID] != C->Engine[i].AngularVelocity[ID])
			printf("NN Before- Craft(%d), Eng(%d), Angle Vel NaN, %f\n", ID, i, C->Engine[i].AngularVelocity[ID]);
		if (C->Engine[i].AngularAcceleration[ID] != C->Engine[i].AngularAcceleration[ID])
			printf("NN Before- Craft(%d), Eng(%d), Angle Acc NaN, %f\n", ID, i, C->Engine[i].AngularAcceleration[ID]);
		if (C->Engine[i].Thrust[ID] != C->Engine[i].Thrust[ID])
			printf("NN Before- Craft(%d), Eng(%d), Thrust NaN, %f\n", ID, i, C->Engine[i].Thrust[ID]);
		if (C->Engine[i].ThrustNormalized[ID] != C->Engine[i].ThrustNormalized[ID])
			printf("NN Before- Craft(%d), Eng(%d), Thrust Norm NaN, %f\n", ID, i, C->Engine[i].ThrustNormalized[ID]);
	}

	for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID] != C->Neuron[CRAFT_COUNT * 2 * i + ID])
		{
			printf("1 NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID, i, C->Neuron[CRAFT_COUNT * 2 * i + ID]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID] = 0.f;
		}
#endif

	///////////////////////////////////////////////////////////////////////
	//// Input

	// TODO: Add neuron for angular velocity and neuron 1 - abs(ang. vel.)

	// Distance to circle of life in 8 directions. Starting at angle 0. Scaled so that diameter is 1
#pragma unroll
	for (int i = 0; i < SENSORS_EDGE_DISTANCE_COUNT / 2; i++)
	{
		// TODO: Optimize this
		float Angle = C->Angle[ID] + i * (2.f * PI) / SENSORS_EDGE_DISTANCE_COUNT;

		/*float dx = cos(Angle);
		float dy = sin(Angle);*/

		float dx, dy;
		__sincosf(Angle, &dy, &dx);

		//float dr = 1.f;

		float x1 = C->Position.X[ID];
		float x2 = C->Position.X[ID] + dx;
		float y1 = C->Position.Y[ID];
		float y2 = C->Position.Y[ID] + dy;
		float D = x1 * y2 - x2 * y1;

		float sign;
		if (dy < 0.f)
			sign = -1.f;
		else
			sign = 1.f;

		float x1i = D * dy + sign * dx * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);
		float x2i = D * dy - sign * dx * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);

		float y1i = -D * dx + fabs(dy) *  __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);
		float y2i = -D * dx - fabs(dy) *  __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);

		float Distance = __fsqrt_ru((x1i - x2i) * (x1i - x2i) + (y1i - y2i) * (y1i - y2i));

		float PartDistance = __fsqrt_ru((x1i - C->Position.X[ID]) * (x1i - C->Position.X[ID]) + (y1i - C->Position.Y[ID]) * (y1i - C->Position.Y[ID]));
		float DistanceRight, DistanceLeft;

		if (x1i > x2i)
		{
			DistanceRight = PartDistance;
			DistanceLeft = Distance - PartDistance;
		}
		else
		{
			DistanceRight = Distance - PartDistance;
			DistanceLeft = PartDistance;
		}

		if (dx > 0.f)
		{
			C->Neuron[32 * 2 * i + ID] = DistanceRight / (2.f * LIFE_RADIUS);
			C->Neuron[32 * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID] = DistanceLeft / (2.f * LIFE_RADIUS);
		}
		else
		{
			C->Neuron[32 * 2 * i + ID] = DistanceLeft / (2.f * LIFE_RADIUS);
			C->Neuron[32 * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID] = DistanceRight / (2.f * LIFE_RADIUS);
		}

		C->Neuron[32 * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT) + ID] = 1.f - C->Neuron[32 * 2 * i + ID];	// Versed distance
		C->Neuron[32 * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID] = 1.f - C->Neuron[32 * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID];
	}

	// Velocity Input
	// TODO: Get more creative with this
	{
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = C->Velocity.X[ID] * SENSORS_VELOCITY_SCALE;
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = C->Velocity.Y[ID] * SENSORS_VELOCITY_SCALE;

		// Versed Velocity
		float VelocityXSign;
		if (C->Velocity.X[ID] < 0.f)
			VelocityXSign = -1.f;
		else
			VelocityXSign = 1.f;

		float VelocityYSign;
		if (C->Velocity.Y[ID] < 0.f)
			VelocityYSign = -1.f;
		else
			VelocityYSign = 1.f;

		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 2 + ID] = VelocityXSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID]));
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 3 + ID] = VelocityYSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID]));
	}

	// Angular Velocity Input
	// TODO: Get more creative with this
	{
		C->Neuron[SENSORS_ANG_VEL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = C->AngularVelocity[ID] * SENSORS_ANG_VEL_SCALE;

		// Versed Angular Velocity
		float AngVelSign;
		if (C->AngularVelocity[ID] < 0.f)
			AngVelSign = -1.f;
		else
			AngVelSign = 1.f;

		C->Neuron[SENSORS_ANG_VEL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = AngVelSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID]));
	}

	// TODO: Add inertial force of acceleration to this
#pragma unroll
	for (int i = 0; i < SENSORS_EXTERNAL_FORCE_COUNT; i++)
	{
		float AngleMomentumVsGravity = -atan2(C->Acceleration.X[ID], GRAVITY + C->Acceleration.Y[ID]);
		float Magnitude = sqrt(pow(GRAVITY + C->Acceleration.Y[ID], 2.f) + pow(C->Acceleration.X[ID], 2.f)) / GRAVITY;
		float Angle = C->Angle[ID] + AngleMomentumVsGravity + i * (2.f * PI) / SENSORS_EXTERNAL_FORCE_COUNT + PI / 2.f;

		while (Angle > PI)
			Angle -= 2.f * PI;
		while (Angle < -PI)
			Angle += 2.f * PI;

		if (Angle > PI / 2.f || Angle < -PI / 2.f)
		{
			C->Neuron[SENSORS_EXTERNAL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 0.f;
			C->Neuron[(SENSORS_EXTERNAL_START + SENSORS_EXTERNAL_FORCE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = Magnitude;
		}
		else
		{
			C->Neuron[SENSORS_EXTERNAL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = Magnitude * __cosf(Angle);
			C->Neuron[(SENSORS_EXTERNAL_START + SENSORS_EXTERNAL_FORCE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 1 - Magnitude * __cosf(Angle);
		}
	}

	// Engine Angle Sensing
#pragma unroll
	for (int i = 0; i < 4; i++)	// Engine Count
	{
		for (int j = 0; j < SENSORS_ENGINE_ANGLE_COUNT; j++)
		{
			float SensorAngle = C->Angle[ID] + j * (2.f * PI) / SENSORS_ENGINE_ANGLE_COUNT + PI / 2.f;

			while (SensorAngle > PI)
				SensorAngle -= 2.f * PI;
			while (SensorAngle < -PI)
				SensorAngle += 2.f * PI;

			if (SensorAngle > PI / 2.f || SensorAngle < -PI / 2.f)
			{
				C->Neuron[(6 * i + SENSORS_ENGINE_ANGLE_START) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * j + ID] = 0.f;
				C->Neuron[(6 * i + SENSORS_ENGINE_ANGLE_START + SENSORS_ENGINE_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * j + ID] = 1.f;
			}
			else
			{
				C->Neuron[(6 * i + SENSORS_ENGINE_ANGLE_START) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * j + ID] = __cosf(SensorAngle);
				C->Neuron[(6 * i + SENSORS_ENGINE_ANGLE_START + SENSORS_ENGINE_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * j + ID] = 1 - __cosf(SensorAngle);
			}
		}
	}

	// Opponent Angle
#pragma unroll
	for (int i = 0; i < SENSORS_OPPONENT_ANGLE_COUNT; i++)
	{
		float OpponentAngleAbsolute = atan2(C->Position.Y[IdxOpponent] - C->Position.Y[ID], C->Position.X[IdxOpponent] - C->Position.X[ID]);

		float Angle = C->Angle[ID] - OpponentAngleAbsolute + i * (2.f * PI) / SENSORS_OPPONENT_ANGLE_COUNT + PI / 2.f;

		while (Angle > PI)
			Angle -= 2.f * PI;
		while (Angle < -PI)
			Angle += 2.f * PI;

		if (Angle > PI / 2.f || Angle < -PI / 2.f)
		{
			C->Neuron[SENSORS_OPPONENT_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 0.f;
			C->Neuron[(SENSORS_OPPONENT_ANGLE_START + SENSORS_OPPONENT_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 1.f;
		}
		else
		{
			C->Neuron[SENSORS_OPPONENT_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = __cosf(Angle);
			C->Neuron[(SENSORS_OPPONENT_ANGLE_START + SENSORS_OPPONENT_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 1 - __cosf(Angle);
		}
	}

	// Opponent Distance
	{
		float Distance = sqrt(pow(C->Position.Y[IdxOpponent] - C->Position.Y[ID], 2.f) + pow(C->Position.X[IdxOpponent] - C->Position.X[ID], 2.f));

		C->Neuron[SENSORS_OPPONENT_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = Distance * (1.f / (2.f * LIFE_RADIUS));
		C->Neuron[SENSORS_OPPONENT_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = 1.f - Distance * (1.f / (2.f * LIFE_RADIUS));
	}

	// Bullet Angle
	// TODO: Figure out something for non-active bullet
#pragma unroll
	for (int i = 0; i < SENSORS_BULLET_ANGLE_COUNT; i++)
	{
		float BulletAngleAbsolute = atan2(C->Bullet->Position.Y[IdxOpponent] - C->Position.Y[ID], C->Bullet->Position.X[IdxOpponent] - C->Position.X[ID]);

		float Angle = C->Angle[ID] - BulletAngleAbsolute + i * (2.f * PI) / SENSORS_BULLET_ANGLE_COUNT + PI / 2.f;

		while (Angle > PI)
			Angle -= 2.f * PI;
		while (Angle < -PI)
			Angle += 2.f * PI;

		if (Angle > PI / 2.f || Angle < -PI / 2.f)
		{
			C->Neuron[SENSORS_BULLET_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 0.f;
			C->Neuron[(SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 1.f;
		}
		else
		{
			C->Neuron[SENSORS_BULLET_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = __cosf(Angle);
			C->Neuron[(SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID] = 1 - __cosf(Angle);
		}
	}

	// Bullet Distance
	{
		if (C->Bullet->Active[IdxOpponent])
		{
			float Distance = sqrt(pow(C->Bullet->Position.Y[IdxOpponent] - C->Position.Y[ID], 2.f) + pow(C->Bullet->Position.X[IdxOpponent] - C->Position.X[ID], 2.f));

			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = Distance * (1.f / (2.f * LIFE_RADIUS));
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = 1.f - Distance * (1.f / (2.f * LIFE_RADIUS));
		}
		else
		{
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = 1.f;
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = 0.f;
		}
	}

	// Angle
	// TODO: Move to right above angular velocity
	float sign;
	if (C->Angle[ID] < 0.f)
		sign = -1.f;
	else
		sign = 1.f;

	C->Neuron[SENSORS_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID] = C->Angle[ID] / PI;
	C->Neuron[SENSORS_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID] = sign * (1.f - fabs(C->Angle[ID] / PI));

	// Memory from Feedback
	//C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + 0) + ID] = C->Neuron[CRAFT_COUNT * 2 * (LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + 25 + 0) + ID];

#pragma unroll
	//for (int i = 1; i < SENSORS_MEMORY_COUNT; i++)
	//{
	//	C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + i) + ID] *= float((1 << i) - 1) / float(1 << i);
	//	C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + i) + ID] += C->Neuron[CRAFT_COUNT * 2 * (LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + 25 + i) + ID] / float(1 << i);
	//}

#ifdef _DEBUGs
	for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID] != C->Neuron[CRAFT_COUNT * 2 * i + ID])
		{
			printf("2 NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID, i, C->Neuron[CRAFT_COUNT * 2 * i + ID]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID] = 0.f;
		}
#endif

	// TODO: Add opponent and bullet detection

	///////////////////////////////////////////////////////////////////////////
	//// Hidden Layer and Output

	int CraftID;

	if (IsOpponent)
		CraftID = OpponentID;
	else
		CraftID = ID;

	// Clear hidden and output neurons for loop addition
//#pragma unroll
	for (int i = LAYER_SIZE_INPUT; i < NEURON_COUNT; i++)
		C->Neuron[CRAFT_COUNT * 2 * i + ID] = 0.f;

	int LayerCountArray[]	= LAYER_ARRAY;
	int LayerBeginIndex[]	= LAYER_BEGIN_INDEX;
	int WeightBegin[]		= WEIGHT_BEGIN_INDEX_ARRAY;

	for (int LayerNumber = 0; LayerNumber < LAYER_AMOUNT - 1; LayerNumber++)
	{
		for (int TargetNeuron = 0; TargetNeuron < LayerCountArray[LayerNumber + 1]; TargetNeuron++)
		{
			// Loop through target Neurons first because these are to be multiplied and summed
			for (short int OriginatingNeuron = 0; OriginatingNeuron < LayerCountArray[LayerNumber]; OriginatingNeuron++)		// Loop through originating gm_Neurons for each target neuron
			{
				if (IsOpponent)
					C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID]
					+= C_Opponent->Weights[(WeightBegin[LayerNumber] + TargetNeuron * LayerCountArray[LayerNumber] + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[LayerBeginIndex[LayerNumber] + OriginatingNeuron * CRAFT_COUNT * 2 + ID];
				else
					C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID]
					+= C->Weights[(WeightBegin[LayerNumber] + TargetNeuron * LayerCountArray[LayerNumber] + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[LayerBeginIndex[LayerNumber] + OriginatingNeuron * CRAFT_COUNT * 2 + ID];
			}

			// Rectify
			if (C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID] > 1.f)
				C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID] = 1.f;
			else if (C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID] < -1.f)
				C->Neuron[(LayerBeginIndex[LayerNumber + 1] + TargetNeuron) * CRAFT_COUNT * 2 + ID] = -1.f;
		}
	}

//	// Input to hidden
////#pragma unroll
//	for (short int TargetNeuron = 0; TargetNeuron < LAYER_SIZE_HIDDEN; TargetNeuron++)
//	{
//		// Loop through target Neurons first because these are to be multiplied and summed
//		for (short int OriginatingNeuron = 0; OriginatingNeuron < LAYER_SIZE_INPUT; OriginatingNeuron++)		// Loop through originating gm_Neurons for each target neuron
//		{
//			if (IsOpponent)
//				C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID]
//				+= C_Opponent->Weights[(TargetNeuron * LAYER_SIZE_INPUT + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[OriginatingNeuron * CRAFT_COUNT * 2 + ID];
//			else
//				C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID]
//				+= C->Weights[(TargetNeuron * LAYER_SIZE_INPUT + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[OriginatingNeuron * CRAFT_COUNT * 2 + ID];
//		}
//
//		// Rectify
//		if (C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID] > 1.f)
//			C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID] = 1.f;
//		else if (C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID] < -1.f)
//			C->Neuron[(LAYER_SIZE_INPUT + TargetNeuron) * CRAFT_COUNT * 2 + ID] = -1.f;
//	}
//
//	// Hidden to output
////#pragma unroll
//	for (short int TargetNeuron = 0; TargetNeuron < LAYER_SIZE_OUTPUT; TargetNeuron++)
//	{
//		for (short int OriginatingNeuron = 0; OriginatingNeuron < LAYER_SIZE_HIDDEN; OriginatingNeuron++)
//		{
//			if (IsOpponent)
//				C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID]
//				+= C_Opponent->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + TargetNeuron * LAYER_SIZE_HIDDEN + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[(LAYER_SIZE_INPUT + OriginatingNeuron) * CRAFT_COUNT * 2 + ID];
//			else
//			C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID]
//				+= C->Weights[(LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + TargetNeuron * LAYER_SIZE_HIDDEN + OriginatingNeuron) * CRAFT_COUNT + CraftID] * C->Neuron[(LAYER_SIZE_INPUT + OriginatingNeuron) * CRAFT_COUNT * 2 + ID];
//		}
//
//		// Clamp output neurons [0.0,1.0]
//		//if (TargetNeuron < LAYER_SIZE_OUTPUT - SENSORS_MEMORY_COUNT)
//			__saturatef(C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID]);
//		// Memory neurons can be negative
//		//else
//		//	if (C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID] > 1.f)
//		//		C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID] = 1.f;
//		//	else if (C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID] < -1.f)
//		//		C->Neuron[(LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + TargetNeuron) * CRAFT_COUNT * 2 + ID] = -1.f;
//	}

	////////////////////////////////////////////////////////////////////////////
	//// Output Conversion

	// Engine Angle Command
#pragma unroll
	for (int i = 0; i < 4; i++)	// 4 Engines
	{
		float P0	= C->Neuron[(0 + 4 * i + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float P1	= C->Neuron[(1 + 4 * i + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float P2	= C->Neuron[(2 + 4 * i + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float Brake = C->Neuron[(3 + 4 * i + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];

		if (P0 + P1 + P2 == 0.f)
			P0 += 0.01f;

		// TODO: This looks like it could be infinity. Double check
		//float dx = 0.5f * sqrt(3.f) * (P2 - P1) / (P0 + P1 + P2);
		//float dy = (P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);

		float dx = -(P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);
		float dy = 0.5f * __fsqrt_ru(3.f) * (P2 - P1) / (P0 + P1 + P2);

		// Find target angle
		float CommandAngle = atan2(dy, dx);

		//C->Engine[i].Angle[ID] = CommandAngle;

		float AngleDifference = CommandAngle - C->Engine[i].Angle[ID];

		// TODO: Think about this
		if (AngleDifference > PI)
			AngleDifference -= 2.f * PI;
		else if (AngleDifference < -PI)
			AngleDifference += -2.f * PI;

		if (abs(AngleDifference) < 2.f * PI / 180.f)
		{
			C->Engine[i].AngularVelocity[ID] = 0.f;
			C->Engine[i].AngularAcceleration[ID] = 0.f;
		}
		else
		{
			// Bang Bang control law
			// Find target angle direction
			float AngleSign;
			if (AngleDifference < 0.f)
				AngleSign = -1.f;
			else
				AngleSign = 1.f;

			float AngleVel = C->Engine[i].AngularVelocity[ID];

			// Apply strength and brake
			float Strength = (1.f - Brake);

			if ((2.f / ENGINE_ANGULAR_ACCEL) * AngleSign * AngleVel > sqrt(pow(AngleVel, 2.f) + fabs(2.f * ENGINE_ANGULAR_ACCEL * AngleDifference)))
				C->Engine[i].AngularAcceleration[ID] = -AngleSign * ENGINE_ANGULAR_ACCEL * Strength;
			else
				C->Engine[i].AngularAcceleration[ID] = AngleSign * ENGINE_ANGULAR_ACCEL * Strength;
		}
	}

	// Cannon Angle Command
	{
		float P0	= C->Neuron[(16 + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float P1	= C->Neuron[(17 + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float P2	= C->Neuron[(18 + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];
		float Brake = C->Neuron[(19 + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];

		if (P0 + P1 + P2 == 0.f)
			P0 += 0.01f;

		//float dx = 0.5f * sqrt(3.f) * (P2 - P1) / (P0 + P1 + P2);
		//float dy = (P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);

		float dx = (P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);
		float dy = 0.5f * __fsqrt_ru(3.f) * (P1 - P2) / (P0 + P1 + P2);

		// Find target angle
		float CommandAngle = atan2(dy, dx);
		//C->Cannon.Angle[ID] = CommandAngle;
		C->CannonCommandAngle[ID] = CommandAngle;

		float AngleDifference = CommandAngle - C->Cannon.Angle[ID];

		// TODO: Think about this
		if (AngleDifference > PI)
			AngleDifference -= 2.f * PI;
		else if (AngleDifference < -PI)
			AngleDifference += -2.f * PI;

		if (abs(AngleDifference) < 2.f * PI / 180.f)
		{
			C->Cannon.AngularVelocity[ID] = 0.f;
			C->Cannon.AngularAcceleration[ID] = 0.f;
		}
		else
		{
			// Bang Bang control law
			float AngleSign;
			if (AngleDifference < 0.f)
				AngleSign = -1.f;
			else
				AngleSign = 1.f;

			float AngleVel = C->Cannon.AngularVelocity[ID];

			float Strength = (1.f - Brake);
			C->CannonStrength[ID] = Strength;

			if ((2.f / CANNON_ANGULAR_ACCEL) * AngleSign * AngleVel > sqrt(pow(AngleVel, 2.f) + fabs(2.f * CANNON_ANGULAR_ACCEL * AngleDifference)))
				C->Cannon.AngularAcceleration[ID] = -AngleSign * CANNON_ANGULAR_ACCEL * Strength;
			else
				C->Cannon.AngularAcceleration[ID] = AngleSign * CANNON_ANGULAR_ACCEL * Strength;
		}

		// TODO: Evaluate
	}

	// Cannon fire decision
	if (C->Neuron[(20 + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID] > 0.5f)
		if (C->BulletTimer[ID] > (int)(BULLET_INTERVAL_MIN / TIME_STEP))
			if (C->BulletCounter[ID] < BULLET_COUNT_MAX)
				ShootBullet(C, ID, Buffer);
	C->BulletTimer[ID]++;

	// Thrust
#pragma unroll
	for (int i = 0; i < 4; i++)
		C->Engine[i].ThrustNormalized[ID] = C->Neuron[((21 + i) + LayerBeginIndex[LAYER_AMOUNT - 1]) * CRAFT_COUNT * 2 + ID];

#ifdef _DEBUGs
	for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID] != C->Neuron[CRAFT_COUNT * 2 * i + ID])
		{
			printf("NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID, i, C->Neuron[CRAFT_COUNT * 2 * i + ID]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID] = 0.f;
		}

	if (C->Position.X[ID] != C->Position.X[ID])
		printf("NN After- Craft(%d), Position X NaN, %f\n", ID, C->Position.X[ID]);
	if (C->Position.Y[ID] != C->Position.Y[ID])
		printf("NN After- Craft(%d), Position Y NaN, %f\n", ID, C->Position.Y[ID]);
	if (C->Velocity.X[ID] != C->Velocity.X[ID])
		printf("NN After- Craft(%d), Velocity X NaN, %f\n", ID, C->Velocity.X[ID]);
	if (C->Velocity.Y[ID] != C->Velocity.Y[ID])
		printf("NN After- Craft(%d), Velocity Y NaN, %f\n", ID, C->Velocity.Y[ID]);
	if (C->Acceleration.X[ID] != C->Acceleration.X[ID])
		printf("NN After- Craft(%d), Acceleration X NaN, %f\n", ID, C->Acceleration.X[ID]);
	if (C->Acceleration.Y[ID] != C->Acceleration.Y[ID])
		printf("NN After- Craft(%d), Acceleration Y NaN, %f\n", ID, C->Acceleration.Y[ID]);
	if (C->Angle[ID] != C->Angle[ID])
		printf("NN After- Craft(%d), Angle NaN, %f\n", ID, C->Angle[ID]);
	if (C->AngularVelocity[ID] != C->AngularVelocity[ID])
		printf("NN After- Craft(%d), Angular Velocity NaN, %f\n", ID, C->AngularVelocity[ID]);
	if (C->AngularAcceleration[ID] != C->AngularAcceleration[ID])
		printf("NN After- Craft(%d), AngularAcceleration NaN, %f\n", ID, C->AngularAcceleration[ID]);
	if (C->Cannon.Angle[ID] != C->Cannon.Angle[ID])
		printf("NN After- Craft(%d), Cannon Angle NaN, %f\n", ID, C->Cannon.Angle[ID]);
	if (C->Cannon.AngularVelocity[ID] != C->Cannon.AngularVelocity[ID])
		printf("NN After- Craft(%d), Cannon Angle Vel NaN, %f\n", ID, C->Cannon.AngularVelocity[ID]);
	if (C->Cannon.AngularAcceleration[ID] != C->Cannon.AngularAcceleration[ID])
		printf("NN After- Craft(%d), Cannon Angle Acc NaN, %f\n", ID, C->Cannon.AngularAcceleration[ID]);
	if (C->Score[ID] != C->Score[ID])
		printf("NN After- Craft(%d), Score NaN, %f\n", ID, C->Score[ID]);

	for (int i = 0; i < 4; i++)
	{
		if (C->Engine[i].Angle[ID] != C->Engine[i].Angle[ID])
			printf("NN After- Craft(%d), Eng(%d), Angle NaN, %f\n", ID, i, C->Engine[i].Angle[ID]);
		if (C->Engine[i].AngularVelocity[ID] != C->Engine[i].AngularVelocity[ID])
			printf("NN After- Craft(%d), Eng(%d), Angle Vel NaN, %f\n", ID, i, C->Engine[i].AngularVelocity[ID]);
		if (C->Engine[i].AngularAcceleration[ID] != C->Engine[i].AngularAcceleration[ID])
			printf("NN After- Craft(%d), Eng(%d), Angle Acc NaN, %f\n", ID, i, C->Engine[i].AngularAcceleration[ID]);
		if (C->Engine[i].Thrust[ID] != C->Engine[i].Thrust[ID])
			printf("NN After- Craft(%d), Eng(%d), Thrust NaN, %f\n", ID, i, C->Engine[i].Thrust[ID]);
		if (C->Engine[i].ThrustNormalized[ID] != C->Engine[i].ThrustNormalized[ID])
			printf("NN After- Craft(%d), Eng(%d), Thrust Norm NaN, %f\n", ID, i, C->Engine[i].ThrustNormalized[ID]);
}
#endif
}

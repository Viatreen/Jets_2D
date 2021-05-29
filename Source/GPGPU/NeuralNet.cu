// File Header
#include "NeuralNet.h"

// Standard Library
#include "math.h"
#include "stdio.h"

// CUDA
#include "cuda_runtime.h"

// Project Headers
#include "Config.h"
#include "GPGPU/State.h"
#include "GPGPU/Physic.h"



__forceinline__ __device__ void RELU_Activate(float& Neuron)
{
	if (Neuron > 1.f)
		Neuron = NETWORK_ACTIVATION_SLOPE * Neuron + 1.f - NETWORK_ACTIVATION_SLOPE;	// y = mx + b
	else if (Neuron < -1.f)
		Neuron = NETWORK_ACTIVATION_SLOPE * Neuron - 1.f + NETWORK_ACTIVATION_SLOPE;
}

__device__ void State_Processing(CraftState* C, GraphicsObjectPointer* Buffer, int ID_Opponent, int ID_Craft, int ID_Weight)
{
	///////////////////////////////////////////////////////////////////////////
	//// Environment Input to Input Neuron Conversion

	Environment_To_Input_Neurons(C, ID_Opponent, ID_Craft);

	///////////////////////////////////////////////////////////////////////////
	//// Neural Net Processing

	// TODO: Change activation back to true
	Run_Neural_Net(C, true, ID_Craft, ID_Weight);

	////////////////////////////////////////////////////////////////////////////
	//// Output Conversion

	Output_Neurons_To_Action(C, ID_Craft, Buffer);
}

__device__ void Environment_To_Input_Neurons(CraftState* C, int ID_Opponent, int ID_Craft)
{
#ifdef _DEBUGs
	if (C->Position.X[ID_Craft] != C->Position.X[ID_Craft])
		printf("NN Before- Craft(%d), Position X NaN, %f\n", ID_Craft, C->Position.X[ID_Craft]);
	if (C->Position.Y[ID_Craft] != C->Position.Y[ID_Craft])
		printf("NN Before- Craft(%d), Position Y NaN, %f\n", ID_Craft, C->Position.Y[ID_Craft]);
	if (C->Velocity.X[ID_Craft] != C->Velocity.X[ID_Craft])
		printf("NN Before- Craft(%d), Velocity X NaN, %f\n", ID_Craft, C->Velocity.X[ID_Craft]);
	if (C->Velocity.Y[ID_Craft] != C->Velocity.Y[ID_Craft])
		printf("NN Before- Craft(%d), Velocity Y NaN, %f\n", ID_Craft, C->Velocity.Y[ID_Craft]);
	if (C->Acceleration.X[ID_Craft] != C->Acceleration.X[ID_Craft])
		printf("NN Before- Craft(%d), Acceleration X NaN, %f\n", ID_Craft, C->Acceleration.X[ID_Craft]);
	if (C->Acceleration.Y[ID_Craft] != C->Acceleration.Y[ID_Craft])
		printf("NN Before- Craft(%d), Acceleration Y NaN, %f\n", ID_Craft, C->Acceleration.Y[ID_Craft]);
	if (C->Angle[ID_Craft] != C->Angle[ID_Craft])
		printf("NN Before- Craft(%d), Angle NaN, %f\n", ID_Craft, C->Angle[ID_Craft]);
	if (C->AngularVelocity[ID_Craft] != C->AngularVelocity[ID_Craft])
		printf("NN Before- Craft(%d), Angular Velocity NaN, %f\n", ID_Craft, C->AngularVelocity[ID_Craft]);
	if (C->AngularAcceleration[ID_Craft] != C->AngularAcceleration[ID_Craft])
		printf("NN Before- Craft(%d), AngularAcceleration NaN, %f\n", ID_Craft, C->AngularAcceleration[ID_Craft]);
	if (C->Cannon.Angle[ID_Craft] != C->Cannon.Angle[ID_Craft])
		printf("NN Before- Craft(%d), Cannon Angle NaN, %f\n", ID_Craft, C->Cannon.Angle[ID_Craft]);
	if (C->Cannon.AngularVelocity[ID_Craft] != C->Cannon.AngularVelocity[ID_Craft])
		printf("NN Before- Craft(%d), Cannon Angle Vel NaN, %f\n", ID_Craft, C->Cannon.AngularVelocity[ID_Craft]);
	if (C->Cannon.AngularAcceleration[ID_Craft] != C->Cannon.AngularAcceleration[ID_Craft])
		printf("NN Before- Craft(%d), Cannon Angle Acc NaN, %f\n", ID_Craft, C->Cannon.AngularAcceleration[ID_Craft]);
	if (C->Score[ID_Craft] != C->Score[ID_Craft])
		printf("NN Before- Craft(%d), Score NaN, %f\n", ID_Craft, C->Score[ID_Craft]);

	for (int i = 0; i < 4; i++)
	{
		if (C->Engine[i].Angle[ID_Craft] != C->Engine[i].Angle[ID_Craft])
			printf("NN Before- Craft(%d), Eng(%d), Angle NaN, %f\n", ID_Craft, i, C->Engine[i].Angle[ID_Craft]);
		if (C->Engine[i].AngularVelocity[ID_Craft] != C->Engine[i].AngularVelocity[ID_Craft])
			printf("NN Before- Craft(%d), Eng(%d), Angle Vel NaN, %f\n", ID_Craft, i, C->Engine[i].AngularVelocity[ID_Craft]);
		if (C->Engine[i].AngularAcceleration[ID_Craft] != C->Engine[i].AngularAcceleration[ID_Craft])
			printf("NN Before- Craft(%d), Eng(%d), Angle Acc NaN, %f\n", ID_Craft, i, C->Engine[i].AngularAcceleration[ID_Craft]);
		if (C->Engine[i].Thrust[ID_Craft] != C->Engine[i].Thrust[ID_Craft])
			printf("NN Before- Craft(%d), Eng(%d), Thrust NaN, %f\n", ID_Craft, i, C->Engine[i].Thrust[ID_Craft]);
		if (C->Engine[i].ThrustNormalized[ID_Craft] != C->Engine[i].ThrustNormalized[ID_Craft])
			printf("NN Before- Craft(%d), Eng(%d), Thrust Norm NaN, %f\n", ID_Craft, i, C->Engine[i].ThrustNormalized[ID_Craft]);
	}

	for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] != C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft])
		{
			printf("1 NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID_Craft, i, C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
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
		float Angle = C->Angle[ID_Craft] + i * (2.f * PI) / SENSORS_EDGE_DISTANCE_COUNT;

		/*float dx = cos(Angle);
		float dy = sin(Angle);*/

		float dx, dy;
		__sincosf(Angle, &dy, &dx);

		//float dr = 1.f;

		float x1 = C->Position.X[ID_Craft];
		float x2 = C->Position.X[ID_Craft] + dx;
		float y1 = C->Position.Y[ID_Craft];
		float y2 = C->Position.Y[ID_Craft] + dy;
		float D = x1 * y2 - x2 * y1;

		float sign;
		if (dy < 0.f)
			sign = -1.f;
		else
			sign = 1.f;

		float x1i = D * dy + sign * dx * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);
		float x2i = D * dy - sign * dx * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);

		float y1i = -D * dx + fabs(dy) * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);
		float y2i = -D * dx - fabs(dy) * __fsqrt_ru(LIFE_RADIUS * LIFE_RADIUS - D * D);

		float Distance = __fsqrt_ru((x1i - x2i) * (x1i - x2i) + (y1i - y2i) * (y1i - y2i));

		float PartDistance = __fsqrt_ru((x1i - C->Position.X[ID_Craft]) * (x1i - C->Position.X[ID_Craft]) + (y1i - C->Position.Y[ID_Craft]) * (y1i - C->Position.Y[ID_Craft]));
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
			C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] = DistanceRight / (2.f * LIFE_RADIUS);
			C->Neuron[CRAFT_COUNT * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID_Craft] = DistanceLeft / (2.f * LIFE_RADIUS);
		}
		else
		{
			C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] = DistanceLeft / (2.f * LIFE_RADIUS);
			C->Neuron[CRAFT_COUNT * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID_Craft] = DistanceRight / (2.f * LIFE_RADIUS);
		}

		C->Neuron[CRAFT_COUNT * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT) + ID_Craft] = 1.f - C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft];	// Versed distance
		C->Neuron[CRAFT_COUNT * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID_Craft] = 1.f - C->Neuron[CRAFT_COUNT * 2 * (i + SENSORS_EDGE_DISTANCE_COUNT / 2) + ID_Craft];
	}

	// Velocity Input
	// TODO: Get more creative with this
	{
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = C->Velocity.X[ID_Craft] * SENSORS_VELOCITY_SCALE;
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = C->Velocity.Y[ID_Craft] * SENSORS_VELOCITY_SCALE;

		// Versed Velocity
		float VelocityXSign;
		if (C->Velocity.X[ID_Craft] < 0.f)
			VelocityXSign = -1.f;
		else
			VelocityXSign = 1.f;

		float VelocityYSign;
		if (C->Velocity.Y[ID_Craft] < 0.f)
			VelocityYSign = -1.f;
		else
			VelocityYSign = 1.f;

		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 2 + ID_Craft] = VelocityXSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft]));
		C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 3 + ID_Craft] = VelocityYSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft]));
	}

	// Angular Velocity Input
	// TODO: Get more creative with this
	{
		C->Neuron[SENSORS_ANG_VEL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = C->AngularVelocity[ID_Craft] * SENSORS_ANG_VEL_SCALE;

		// Versed Angular Velocity
		float AngVelSign;
		if (C->AngularVelocity[ID_Craft] < 0.f)
			AngVelSign = -1.f;
		else
			AngVelSign = 1.f;

		C->Neuron[SENSORS_ANG_VEL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = AngVelSign * (1.f - fabs(C->Neuron[SENSORS_VELOCITY_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft]));
	}

	// TODO: Add inertial force of acceleration to this
#pragma unroll
	for (int i = 0; i < SENSORS_EXTERNAL_FORCE_COUNT; i++)
	{
		float AngleMomentumVsGravity = -atan2(C->Acceleration.X[ID_Craft], GRAVITY + C->Acceleration.Y[ID_Craft]);
		float Magnitude = sqrt(pow(GRAVITY + C->Acceleration.Y[ID_Craft], 2.f) + pow(C->Acceleration.X[ID_Craft], 2.f)) / GRAVITY;
		float Angle = C->Angle[ID_Craft] + AngleMomentumVsGravity + i * (2.f * PI) / SENSORS_EXTERNAL_FORCE_COUNT + PI / 2.f;

		while (Angle > PI)
			Angle -= 2.f * PI;
		while (Angle < -PI)
			Angle += 2.f * PI;

		if (Angle > PI / 2.f || Angle < -PI / 2.f)
		{
			C->Neuron[SENSORS_EXTERNAL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
			C->Neuron[(SENSORS_EXTERNAL_START + SENSORS_EXTERNAL_FORCE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = Magnitude;
		}
		else
		{
			C->Neuron[SENSORS_EXTERNAL_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = Magnitude * __cosf(Angle);
			C->Neuron[(SENSORS_EXTERNAL_START + SENSORS_EXTERNAL_FORCE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1 - Magnitude * __cosf(Angle);
		}
	}

	// Engine Angle Sensing
#pragma unroll
	for (int i = 0; i < 4; i++)	// Engine Count
	{
		for (int j = 0; j < SENSORS_ENGINE_ANGLE_COUNT; j++)
		{
			float SensorAngle = C->Angle[ID_Craft] + j * (2.f * PI) / SENSORS_ENGINE_ANGLE_COUNT + PI / 2.f;

			while (SensorAngle > PI)
				SensorAngle -= 2.f * PI;
			while (SensorAngle < -PI)
				SensorAngle += 2.f * PI;

			if (SensorAngle > PI / 2.f || SensorAngle < -PI / 2.f)
			{
				C->Neuron[2 * CRAFT_COUNT * (SENSORS_ENGINE_ANGLE_START + 2 * SENSORS_ENGINE_ANGLE_COUNT * i + j) + ID_Craft] = 0.f;
				C->Neuron[2 * CRAFT_COUNT * (SENSORS_ENGINE_ANGLE_START + 2 * SENSORS_ENGINE_ANGLE_COUNT * i + j + SENSORS_ENGINE_ANGLE_COUNT) + ID_Craft] = 1.f;
			}
			else
			{
				C->Neuron[2 * CRAFT_COUNT * (SENSORS_ENGINE_ANGLE_START + 2 * SENSORS_ENGINE_ANGLE_COUNT * i + j) + ID_Craft] = __cosf(SensorAngle);
				C->Neuron[2 * CRAFT_COUNT * (SENSORS_ENGINE_ANGLE_START + 2 * SENSORS_ENGINE_ANGLE_COUNT * i + j + SENSORS_ENGINE_ANGLE_COUNT) + ID_Craft] = 1 - __cosf(SensorAngle);
			}
		}
	}

	// Opponent Angle
#pragma unroll
	for (int i = 0; i < SENSORS_OPPONENT_ANGLE_COUNT; i++)
	{
		float OpponentAngleAbsolute = atan2(C->Position.Y[ID_Opponent] - C->Position.Y[ID_Craft], C->Position.X[ID_Opponent] - C->Position.X[ID_Craft]);

		float Angle = C->Angle[ID_Craft] - OpponentAngleAbsolute + i * (2.f * PI) / SENSORS_OPPONENT_ANGLE_COUNT + PI / 2.f;

		while (Angle > PI)
			Angle -= 2.f * PI;
		while (Angle < -PI)
			Angle += 2.f * PI;

		if (Angle > PI / 2.f || Angle < -PI / 2.f)
		{
			C->Neuron[SENSORS_OPPONENT_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
			C->Neuron[(SENSORS_OPPONENT_ANGLE_START + SENSORS_OPPONENT_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1.f;
		}
		else
		{
			C->Neuron[SENSORS_OPPONENT_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = __cosf(Angle);
			C->Neuron[(SENSORS_OPPONENT_ANGLE_START + SENSORS_OPPONENT_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1 - __cosf(Angle);
		}
	}

	// Opponent Distance
	{
		float Distance = sqrt(pow(C->Position.Y[ID_Opponent] - C->Position.Y[ID_Craft], 2.f) + pow(C->Position.X[ID_Opponent] - C->Position.X[ID_Craft], 2.f));

		C->Neuron[SENSORS_OPPONENT_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = Distance * (1.f / (2.f * LIFE_RADIUS));
		C->Neuron[SENSORS_OPPONENT_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = 1.f - Distance * (1.f / (2.f * LIFE_RADIUS));
	}

	// Bullet Angle
	// TODO: Figure out something for bullets that aren't in index 0
	float BulletAngleAbsolute = atan2(C->Bullet[0].Position.Y[ID_Opponent] - C->Position.Y[ID_Craft], C->Bullet[0].Position.X[ID_Opponent] - C->Position.X[ID_Craft]);
	
#pragma unroll
	for (int i = 0; i < SENSORS_BULLET_ANGLE_COUNT; i++)
	{
		if (C->BulletCounter[ID_Opponent] > 0)
		{
			float Angle = C->Angle[ID_Craft] - BulletAngleAbsolute + i * (2.f * PI) / SENSORS_BULLET_ANGLE_COUNT + PI / 2.f;

			while (Angle > PI)
				Angle -= 2.f * PI;
			while (Angle < -PI)
				Angle += 2.f * PI;

			if (Angle > PI / 2.f || Angle < -PI / 2.f)
			{
				C->Neuron[SENSORS_BULLET_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
				C->Neuron[(SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1.f;
			}
			else
			{
				C->Neuron[SENSORS_BULLET_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = __cosf(Angle);
				C->Neuron[(SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1 - __cosf(Angle);
			}
		}
		else
		{
			C->Neuron[SENSORS_BULLET_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
			C->Neuron[(SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT) * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * i + ID_Craft] = 1.f;
		}
	}

	// Bullet Distance
	{
		if (C->Bullet[0].Active[ID_Opponent])
		{
			float Distance = sqrt(pow(C->Bullet[0].Position.Y[ID_Opponent] - C->Position.Y[ID_Craft], 2.f) + pow(C->Bullet[0].Position.X[ID_Opponent] - C->Position.X[ID_Craft], 2.f));

			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = Distance * (1.f / (2.f * LIFE_RADIUS));
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = 1.f - Distance * (1.f / (2.f * LIFE_RADIUS));
		}
		else
		{
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = 1.f;
			C->Neuron[SENSORS_BULLET_DISTANCE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = 0.f;
		}
	}


	// Angle
	// TODO: Move to right above angular velocity
	float sign;
	if (C->Angle[ID_Craft] < 0.f)
		sign = -1.f;
	else
		sign = 1.f;

	C->Neuron[SENSORS_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 0 + ID_Craft] = C->Angle[ID_Craft] / PI;
	C->Neuron[SENSORS_ANGLE_START * CRAFT_COUNT * 2 + CRAFT_COUNT * 2 * 1 + ID_Craft] = sign * (1.f - fabs(C->Angle[ID_Craft] / PI));

	// Memory from Feedback
	//C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + 0) + ID_Craft] = C->Neuron[CRAFT_COUNT * 2 * (LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + 25 + 0) + ID_Craft];

	//#pragma unroll
	//for (int i = 1; i < SENSORS_MEMORY_COUNT; i++)
	//{
	//	C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + i) + ID_Craft] *= float((1 << i) - 1) / float(1 << i);
	//	C->Neuron[CRAFT_COUNT * 2 * (SENSORS_MEMORY_START + i) + ID_Craft] += C->Neuron[CRAFT_COUNT * 2 * (LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN + 25 + i) + ID_Craft] / float(1 << i);
	//}

	/*for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] != C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft])
		{
			printf("NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID_Craft, i, C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
		}*/
}

__device__ void Run_Neural_Net(CraftState* C, bool Do_Activation, int ID_Neurons, int ID_Weights)
{
	// Init network to zero (Except for input neurons)
	for (unsigned int i = LAYER_SIZE_INPUT; i < NEURON_COUNT; i++)
		C->Neuron[2 * CRAFT_COUNT * i + ID_Neurons] = 0.f;

	// Calculate values of first hidden layer
	for (unsigned int Input = 0; Input < LAYER_SIZE_INPUT; Input++)
	{
		for (unsigned int Output = LAYER_SIZE_INPUT; Output < LAYER_SIZE_INPUT + NEURONS_PER_HIDDEN_LAYER; Output++)
		{
			unsigned int Weight_Index = Input * NEURONS_PER_HIDDEN_LAYER + Output; // TODO: Investigate. This is never 0

			C->Neuron[2 * CRAFT_COUNT * Output + ID_Neurons] += C->Neuron[2 * CRAFT_COUNT * Input + ID_Neurons] * C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
		}
	}

	// Activate first hidden layer
	for (unsigned int i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++)
	{
		unsigned int Index = i + LAYER_SIZE_INPUT;

		if (Do_Activation)
			RELU_Activate(C->Neuron[2 * CRAFT_COUNT * Index + ID_Neurons]);
	}

	// Calculate values for neurons of hidden layers
	for (unsigned int Layer = 1; Layer < LAYER_AMOUNT_HIDDEN; Layer++)
	{
		for (unsigned int Input = 0; Input < NEURONS_PER_HIDDEN_LAYER; Input++)
		{
			for (unsigned int Output = 0; Output < NEURONS_PER_HIDDEN_LAYER; Output++)
			{
				unsigned int Output_Index = LAYER_SIZE_INPUT + Layer * NEURONS_PER_HIDDEN_LAYER + Output;
				unsigned int Input_Index  = LAYER_SIZE_INPUT + (Layer - 1) * NEURONS_PER_HIDDEN_LAYER + Input;

				unsigned int Weight_Index
					= LAYER_SIZE_INPUT * NEURONS_PER_HIDDEN_LAYER
					+ NEURONS_PER_HIDDEN_LAYER * NEURONS_PER_HIDDEN_LAYER * (Layer - 1)
					+ Input * NEURONS_PER_HIDDEN_LAYER
					+ Output;

				C->Neuron[2 * CRAFT_COUNT * Output_Index + ID_Neurons] += C->Neuron[2 * CRAFT_COUNT * Input_Index + ID_Neurons] * C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
			}
		}

		for (unsigned int Output = 0; Output < NEURONS_PER_HIDDEN_LAYER; Output++)
		{
			unsigned int Index = LAYER_SIZE_INPUT + Layer * NEURONS_PER_HIDDEN_LAYER + Output;

			if (Do_Activation)
				for (unsigned int i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++)
					RELU_Activate(C->Neuron[2 * CRAFT_COUNT * Index + ID_Neurons]);
		}
	}

	// Calculate output neurons
	for (unsigned int Input = 0; Input < NEURONS_PER_HIDDEN_LAYER; Input++)
	{
		for (unsigned int Output = 0; Output < LAYER_SIZE_OUTPUT; Output++)
		{
			unsigned int Output_Index = LAYER_SIZE_INPUT + LAYER_AMOUNT_HIDDEN * NEURONS_PER_HIDDEN_LAYER + Output;
			unsigned int Input_Index  = LAYER_SIZE_INPUT + (LAYER_AMOUNT_HIDDEN - 1) * NEURONS_PER_HIDDEN_LAYER + Input;

			unsigned int Weight_Index
				= LAYER_SIZE_INPUT * NEURONS_PER_HIDDEN_LAYER
				+ NEURONS_PER_HIDDEN_LAYER * NEURONS_PER_HIDDEN_LAYER * (LAYER_AMOUNT_HIDDEN - 1)
				+ Input * LAYER_SIZE_OUTPUT
				+ Output;

			C->Neuron[2 * CRAFT_COUNT * Output_Index + ID_Neurons] += C->Neuron[2 * CRAFT_COUNT * Input_Index + ID_Neurons] * C->Weight[CRAFT_COUNT * Weight_Index + ID_Weights];
		}
	}
}

__device__ void Output_Neurons_To_Action(CraftState *C, int ID_Craft, GraphicsObjectPointer* Buffer)
{
	// Engine Angle Command
#pragma unroll
	for (int i = 0; i < 4; i++)	// 4 Engines
	{
		float P0	= C->Neuron[(0 + 4 * i + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float P1	= C->Neuron[(1 + 4 * i + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float P2	= C->Neuron[(2 + 4 * i + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float Brake = C->Neuron[(3 + 4 * i + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];

		if (P0 + P1 + P2 == 0.f)
			P0 += 0.01f;

		// TODO: This looks like it could be infinity. Double check
		//float dx = 0.5f * sqrt(3.f) * (P2 - P1) / (P0 + P1 + P2);
		//float dy = (P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);

		float dx = -(P0 - 0.5f * (P1 + P2)) / (P0 + P1 + P2);
		float dy = 0.5f * __fsqrt_ru(3.f) * (P2 - P1) / (P0 + P1 + P2);

		// Find target angle
		float CommandAngle = atan2(dy, dx);

		//C->Engine[i].Angle[ID_Craft] = CommandAngle;

		float AngleDifference = CommandAngle - C->Engine[i].Angle[ID_Craft];

		// TODO: Think about this
		if (AngleDifference > PI)
			AngleDifference -= 2.f * PI;
		else if (AngleDifference < -PI)
			AngleDifference += -2.f * PI;

		if (abs(AngleDifference) < 2.f * PI / 180.f)
		{
			C->Engine[i].AngularVelocity[ID_Craft] = 0.f;
			C->Engine[i].AngularAcceleration[ID_Craft] = 0.f;
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

			float AngleVel = C->Engine[i].AngularVelocity[ID_Craft];

			// Apply strength and brake
			float Strength = (1.f - Brake);

			if ((2.f / ENGINE_ANGULAR_ACCEL) * AngleSign * AngleVel > sqrt(pow(AngleVel, 2.f) + fabs(2.f * ENGINE_ANGULAR_ACCEL * AngleDifference)))
				C->Engine[i].AngularAcceleration[ID_Craft] = -AngleSign * ENGINE_ANGULAR_ACCEL * Strength;
			else
				C->Engine[i].AngularAcceleration[ID_Craft] = AngleSign * ENGINE_ANGULAR_ACCEL * Strength;

			if (C->Engine[i].AngularAcceleration[ID_Craft] > ENGINE_ANGULAR_ACCEL)
				C->Engine[i].AngularAcceleration[ID_Craft] = ENGINE_ANGULAR_ACCEL;
			else if (C->Engine[i].AngularAcceleration[ID_Craft] < -ENGINE_ANGULAR_ACCEL)
				C->Engine[i].AngularAcceleration[ID_Craft] = -ENGINE_ANGULAR_ACCEL;
		}
	}

	// Cannon Angle Command
	{
		float P0	= C->Neuron[(16 + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float P1	= C->Neuron[(17 + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float P2	= C->Neuron[(18 + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];
		float Brake = C->Neuron[(19 + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];

		if (P0 + P1 + P2 == 0.f)
			P0 += 0.01f;

		//float dx = 0.5f * sqrt(3.f) * (P2 - P1) / (P0 + P1 + P2);
		//float dy = (P0 - 0.5f*(P1 + P2)) / (P0 + P1 + P2);

		float dx = (P0 - 0.5f * (P1 + P2)) / (P0 + P1 + P2);
		float dy = 0.5f * __fsqrt_ru(3.f) * (P1 - P2) / (P0 + P1 + P2);

		// Find target angle
		float CommandAngle = atan2(dy, dx);
		//C->Cannon.Angle[ID_Craft] = CommandAngle;
		C->CannonCommandAngle[ID_Craft] = CommandAngle;

		float AngleDifference = CommandAngle - C->Cannon.Angle[ID_Craft];

		// TODO: Think about this
		if (AngleDifference > PI)
			AngleDifference -= 2.f * PI;
		else if (AngleDifference < -PI)
			AngleDifference += -2.f * PI;

		if (abs(AngleDifference) < 2.f * PI / 180.f)
		{
			C->Cannon.AngularVelocity[ID_Craft] = 0.f;
			C->Cannon.AngularAcceleration[ID_Craft] = 0.f;
		}
		else
		{
			// Bang Bang control law
			float AngleSign;
			if (AngleDifference < 0.f)
				AngleSign = -1.f;
			else
				AngleSign = 1.f;

			float AngleVel = C->Cannon.AngularVelocity[ID_Craft];

			float Strength = (1.f - Brake);
			C->CannonStrength[ID_Craft] = Strength;

			if ((2.f / CANNON_ANGULAR_ACCEL) * AngleSign * AngleVel > sqrt(pow(AngleVel, 2.f) + fabs(2.f * CANNON_ANGULAR_ACCEL * AngleDifference)))
				C->Cannon.AngularAcceleration[ID_Craft] = -AngleSign * CANNON_ANGULAR_ACCEL * Strength;
			else
				C->Cannon.AngularAcceleration[ID_Craft] = AngleSign * CANNON_ANGULAR_ACCEL * Strength;

			if (C->Cannon.AngularAcceleration[ID_Craft] > CANNON_MAX_ANGULAR_ACCEL)
				C->Cannon.AngularAcceleration[ID_Craft] = CANNON_MAX_ANGULAR_ACCEL;
			else if (C->Cannon.AngularAcceleration[ID_Craft] < -CANNON_MAX_ANGULAR_ACCEL)
				C->Cannon.AngularAcceleration[ID_Craft] = -CANNON_MAX_ANGULAR_ACCEL;
		}

		// TODO: Evaluate
	}

	// Cannon fire decision
	if (C->Neuron[(20 + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft] > 0.5f)
		if (C->BulletTimer[ID_Craft] > (int)(BULLET_INTERVAL_MIN / TIME_STEP))
			if (C->BulletCounter[ID_Craft] < BULLET_COUNT_MAX)
				ShootBullet(C, ID_Craft, Buffer);
	C->BulletTimer[ID_Craft]++;

	// Thrust
#pragma unroll
	for (int i = 0; i < 4; i++)
		C->Engine[i].ThrustNormalized[ID_Craft] = C->Neuron[((21 + i) + OUTPUT_LAYER_NEURON_BEGIN_INDEX) * CRAFT_COUNT * 2 + ID_Craft];

#ifdef _DEBUGs
	for (int i = 0; i < NEURON_COUNT; i++)
		if (C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] != C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft])
		{
			printf("NaN Neuron, Thread(%d) Neuron(%d): %f\n", ID_Craft, i, C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft]);
			C->Neuron[CRAFT_COUNT * 2 * i + ID_Craft] = 0.f;
		}

	if (C->Position.X[ID_Craft] != C->Position.X[ID_Craft])
		printf("NN After- Craft(%d), Position X NaN, %f\n", ID_Craft, C->Position.X[ID_Craft]);
	if (C->Position.Y[ID_Craft] != C->Position.Y[ID_Craft])
		printf("NN After- Craft(%d), Position Y NaN, %f\n", ID_Craft, C->Position.Y[ID_Craft]);
	if (C->Velocity.X[ID_Craft] != C->Velocity.X[ID_Craft])
		printf("NN After- Craft(%d), Velocity X NaN, %f\n", ID_Craft, C->Velocity.X[ID_Craft]);
	if (C->Velocity.Y[ID_Craft] != C->Velocity.Y[ID_Craft])
		printf("NN After- Craft(%d), Velocity Y NaN, %f\n", ID_Craft, C->Velocity.Y[ID_Craft]);
	if (C->Acceleration.X[ID_Craft] != C->Acceleration.X[ID_Craft])
		printf("NN After- Craft(%d), Acceleration X NaN, %f\n", ID_Craft, C->Acceleration.X[ID_Craft]);
	if (C->Acceleration.Y[ID_Craft] != C->Acceleration.Y[ID_Craft])
		printf("NN After- Craft(%d), Acceleration Y NaN, %f\n", ID_Craft, C->Acceleration.Y[ID_Craft]);
	if (C->Angle[ID_Craft] != C->Angle[ID_Craft])
		printf("NN After- Craft(%d), Angle NaN, %f\n", ID_Craft, C->Angle[ID_Craft]);
	if (C->AngularVelocity[ID_Craft] != C->AngularVelocity[ID_Craft])
		printf("NN After- Craft(%d), Angular Velocity NaN, %f\n", ID_Craft, C->AngularVelocity[ID_Craft]);
	if (C->AngularAcceleration[ID_Craft] != C->AngularAcceleration[ID_Craft])
		printf("NN After- Craft(%d), AngularAcceleration NaN, %f\n", ID_Craft, C->AngularAcceleration[ID_Craft]);
	if (C->Cannon.Angle[ID_Craft] != C->Cannon.Angle[ID_Craft])
		printf("NN After- Craft(%d), Cannon Angle NaN, %f\n", ID_Craft, C->Cannon.Angle[ID_Craft]);
	if (C->Cannon.AngularVelocity[ID_Craft] != C->Cannon.AngularVelocity[ID_Craft])
		printf("NN After- Craft(%d), Cannon Angle Vel NaN, %f\n", ID_Craft, C->Cannon.AngularVelocity[ID_Craft]);
	if (C->Cannon.AngularAcceleration[ID_Craft] != C->Cannon.AngularAcceleration[ID_Craft])
		printf("NN After- Craft(%d), Cannon Angle Acc NaN, %f\n", ID_Craft, C->Cannon.AngularAcceleration[ID_Craft]);
	if (C->Score[ID_Craft] != C->Score[ID_Craft])
		printf("NN After- Craft(%d), Score NaN, %f\n", ID_Craft, C->Score[ID_Craft]);

	for (int i = 0; i < 4; i++)
	{
		if (C->Engine[i].Angle[ID_Craft] != C->Engine[i].Angle[ID_Craft])
			printf("NN After- Craft(%d), Eng(%d), Angle NaN, %f\n", ID_Craft, i, C->Engine[i].Angle[ID_Craft]);
		if (C->Engine[i].AngularVelocity[ID_Craft] != C->Engine[i].AngularVelocity[ID_Craft])
			printf("NN After- Craft(%d), Eng(%d), Angle Vel NaN, %f\n", ID_Craft, i, C->Engine[i].AngularVelocity[ID_Craft]);
		if (C->Engine[i].AngularAcceleration[ID_Craft] != C->Engine[i].AngularAcceleration[ID_Craft])
			printf("NN After- Craft(%d), Eng(%d), Angle Acc NaN, %f\n", ID_Craft, i, C->Engine[i].AngularAcceleration[ID_Craft]);
		if (C->Engine[i].Thrust[ID_Craft] != C->Engine[i].Thrust[ID_Craft])
			printf("NN After- Craft(%d), Eng(%d), Thrust NaN, %f\n", ID_Craft, i, C->Engine[i].Thrust[ID_Craft]);
		if (C->Engine[i].ThrustNormalized[ID_Craft] != C->Engine[i].ThrustNormalized[ID_Craft])
			printf("NN After- Craft(%d), Eng(%d), Thrust Norm NaN, %f\n", ID_Craft, i, C->Engine[i].ThrustNormalized[ID_Craft]);
	}
#endif
}

//*
__device__ void BackPropagate(CraftState* C, int Craft_ID)
{
	// Assumes all hidden layers are the same size

	// Result to solve
	float Delta_Output_Neuron = 0.f;
	float Delta_Weight = 1.f;	// TODO: Figure out the notation


	// TODO: Assign this based on Thread ID
	int Weight_Neuron_Origin = 3;
	int Weight_Neuron_Destination = 2;
	int Layer = 1;
	int Target_Output_Neuron = 3;

	int Weight_Index = 0;
	int Origin_Neuron_Index;
	int Target_Neuron_Index;
	int Previous_Layer_Size;

	if (Layer == 0)
	{
		Origin_Neuron_Index = Weight_Neuron_Origin;
		Target_Neuron_Index = LAYER_SIZE_INPUT + Weight_Neuron_Destination;

		Previous_Layer_Size = LAYER_SIZE_INPUT;
	}
	else
	{
		Weight_Index += LAYER_SIZE_INPUT * NEURONS_PER_HIDDEN_LAYER;
		Weight_Index += NEURONS_PER_HIDDEN_LAYER * NEURONS_PER_HIDDEN_LAYER * (Layer - 1);

		Origin_Neuron_Index = LAYER_SIZE_INPUT + NEURONS_PER_HIDDEN_LAYER * (Layer - 1) + Weight_Neuron_Origin;
		Target_Neuron_Index = LAYER_SIZE_INPUT + NEURONS_PER_HIDDEN_LAYER * Layer	    + Weight_Neuron_Destination;

		Previous_Layer_Size = NEURONS_PER_HIDDEN_LAYER;
	}

	Weight_Index += Weight_Neuron_Origin * Previous_Layer_Size + Weight_Neuron_Destination;
	float Weight = C->Weight[CRAFT_COUNT * Weight_Index + Craft_ID];

	float Origin_Neuron = C->Neuron[2 * CRAFT_COUNT * Origin_Neuron_Index + Craft_ID];
	float Target_Neuron = C->Neuron[2 * CRAFT_COUNT * Target_Neuron_Index + Craft_ID];

	if (Layer == LAYER_AMOUNT_HIDDEN)
	{
		if (Weight_Neuron_Destination != Target_Neuron_Index)
		{
			return;
		}
		else
		{
			Delta_Output_Neuron = Origin_Neuron * Weight;
			return;
		}
	}
	
	float Delta_First_Neuron = Origin_Neuron * Weight;
	if (Target_Neuron > 1.f || Target_Neuron < -1.f)
		Delta_First_Neuron *= NETWORK_ACTIVATION_SLOPE;

	float Delta_Neuron_Previous_Layer[NEURONS_PER_HIDDEN_LAYER];
	float Delta_Neuron_Next_Layer[NEURONS_PER_HIDDEN_LAYER];

	// Populate Delta_Neuron_Previous_Layer
	// First loop is just the delta of one neuron broadcasting to the next layer
	// TODO: Combine these 2 loops
	for (int i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++)
	{
		// TODO: Fix this indexing
		int First_Broadcast_Neuron_Weight_Index = Weight_Index + NEURONS_PER_HIDDEN_LAYER * Weight_Neuron_Origin + i;
		float First_Broadcast_Neuron_Weight = C->Weight[CRAFT_COUNT * First_Broadcast_Neuron_Weight_Index + Craft_ID];
		Delta_Neuron_Previous_Layer[i] = First_Broadcast_Neuron_Weight * Delta_First_Neuron;
	}

	for (int i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++)
	{
		int Target_Delta_Neuron_Index = LAYER_SIZE_INPUT + NEURONS_PER_HIDDEN_LAYER * (Layer - 1) + i;
		float Target_Delta_Neuron = C->Neuron[2 * CRAFT_COUNT * Target_Delta_Neuron_Index + Craft_ID];
		if (Target_Delta_Neuron > 1.f || Target_Delta_Neuron < -1.f)
		{
			Delta_Neuron_Previous_Layer[i] *= NETWORK_ACTIVATION_SLOPE;
		}
	}

	for (int Layer_Index = Layer + 2; Layer_Index < LAYER_AMOUNT - 2; Layer_Index++)
	{
		int Broadcast_Neuron_Index_Begin = LAYER_SIZE_INPUT + NEURONS_PER_HIDDEN_LAYER * (Layer_Index - 1);
		int Broadcast_Weight_Index_Begin = LAYER_SIZE_INPUT * NEURONS_PER_HIDDEN_LAYER + NEURONS_PER_HIDDEN_LAYER * NEURONS_PER_HIDDEN_LAYER * (Layer_Index - 1);

		// TODO: Combine this loop and the next loop
		for (int Origin_Delta_Neuron_Index = 0; Origin_Delta_Neuron_Index < NEURONS_PER_HIDDEN_LAYER; Origin_Delta_Neuron_Index++)
		{
			int Broadcast_Delta_Neuron_Index = Broadcast_Neuron_Index_Begin + Origin_Delta_Neuron_Index;
			float Broadcast_Delta_Neuron = Delta_Neuron_Previous_Layer[Origin_Delta_Neuron_Index];

			for (int Target_Delta_Neuron_Index = 0; Target_Delta_Neuron_Index < NEURONS_PER_HIDDEN_LAYER; Target_Delta_Neuron_Index++)
			{
				int Broadcast_Weight_Index = Broadcast_Weight_Index_Begin + Origin_Delta_Neuron_Index * NEURONS_PER_HIDDEN_LAYER + Target_Delta_Neuron_Index;
				float Broadcast_Weight = C->Weight[CRAFT_COUNT * Broadcast_Weight_Index + Craft_ID];

				Delta_Neuron_Next_Layer[Target_Delta_Neuron_Index] += Broadcast_Weight * Broadcast_Delta_Neuron;
			}
		}

		for (int j = 0; j < NEURONS_PER_HIDDEN_LAYER; j++)
		{
			float Target_Neuron_For_Delta = C->Neuron[2 * CRAFT_COUNT * (Broadcast_Neuron_Index_Begin + j) + Craft_ID];
			if (Target_Neuron_For_Delta > 1.f || Target_Neuron_For_Delta < -1.f)
			{
				Delta_Neuron_Previous_Layer[j] = NETWORK_ACTIVATION_SLOPE * Delta_Neuron_Next_Layer[j];
			}
			else
			{
				Delta_Neuron_Previous_Layer[j] = Delta_Neuron_Next_Layer[j];
			}

			Delta_Neuron_Next_Layer[j] = 0.f;
		}
	}

	for (int i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++)
	{
		int Weight_Begin_Index = LAYER_SIZE_INPUT * NEURONS_PER_HIDDEN_LAYER + (LAYER_AMOUNT_HIDDEN - 1) * NEURONS_PER_HIDDEN_LAYER * NEURONS_PER_HIDDEN_LAYER;
		// TODO: Fix this indexing
		int Last_Bottle_Neuron_Weight_Index = Weight_Begin_Index + Target_Output_Neuron + (LAYER_SIZE_OUTPUT - 1) * i;
		float Last_Bottle_Neuron_Weight = C->Weight[CRAFT_COUNT * Last_Bottle_Neuron_Weight_Index + Craft_ID];

		float Bottle_Delta_Neuron = Delta_Neuron_Previous_Layer[i] * Last_Bottle_Neuron_Weight;

		Delta_Output_Neuron += Bottle_Delta_Neuron;
	}

	return;

	// There is not activation on the output layer
	
}
/**/

__device__ void BackPropagate_Eval(CraftState* C, unsigned int Weight_Index)
{
	if (Weight_Index >= WEIGHT_COUNT_EVAL)
		return;

	// Neuron index definition of Weight Index
	// There is only one output neuron for the evaluation network
	unsigned int Layer;
	unsigned int Origin_Neuron_Index;
	unsigned int Target_Neuron_Index;

	unsigned int Origin_Neuron_Index_Within_Layer;
	unsigned int Target_Neuron_Index_Within_Layer;
	unsigned int Weight_Index_Within_Layer;

	if (Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
	{
		Layer = 0;
		Weight_Index_Within_Layer = Weight_Index;

		Origin_Neuron_Index_Within_Layer = Weight_Index / NEURONS_PER_HIDDEN_LAYER_EVAL;
		Target_Neuron_Index_Within_Layer = Weight_Index % NEURONS_PER_HIDDEN_LAYER_EVAL;

		Origin_Neuron_Index = Origin_Neuron_Index_Within_Layer;
		Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Target_Neuron_Index_Within_Layer;
	}
	else if (Weight_Index < LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL)
	{
		Layer = 1 + (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) / (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);
		Weight_Index_Within_Layer = (Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL) % (NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL);

		Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / NEURONS_PER_HIDDEN_LAYER_EVAL;
		Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % NEURONS_PER_HIDDEN_LAYER_EVAL;

		Origin_Neuron_Index = LAYER_SIZE_INPUT_EVAL + (Layer - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL + Origin_Neuron_Index_Within_Layer;
		Target_Neuron_Index = LAYER_SIZE_INPUT_EVAL + Layer * NEURONS_PER_HIDDEN_LAYER_EVAL + Target_Neuron_Index_Within_Layer;
	}
	else
	{
		Layer = LAYER_AMOUNT_EVAL - 1;
		Weight_Index_Within_Layer = Weight_Index - LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL - (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL;

		Origin_Neuron_Index_Within_Layer = Weight_Index_Within_Layer / LAYER_SIZE_OUTPUT_EVAL;
		Target_Neuron_Index_Within_Layer = Weight_Index_Within_Layer % LAYER_SIZE_OUTPUT_EVAL;

		Origin_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL - NEURONS_PER_HIDDEN_LAYER_EVAL + Origin_Neuron_Index_Within_Layer;
		Target_Neuron_Index = OUTPUT_LAYER_NEURON_BEGIN_IDX_EVAL + Target_Neuron_Index_Within_Layer;
	}

	float Origin_Neuron = C->Eval_Network.Neuron[Origin_Neuron_Index];
	float Weight = C->Eval_Network.Weight[Weight_Index];
	
	float Delta_First_Neuron = Origin_Neuron * Weight;
	
	if (Layer == LAYER_AMOUNT_EVAL - 1)
	{
		C->Eval_Network.Delta_Weight[Weight_Index] = Delta_First_Neuron;
		return;
	}

	float Target_Neuron = C->Eval_Network.Neuron[Target_Neuron_Index];

	if (Target_Neuron > 1.f || Target_Neuron < -1.f)
	{
		Delta_First_Neuron *= NETWORK_ACTIVATION_SLOPE;
	}

	if (Layer == LAYER_AMOUNT_EVAL - 2)
	{
		unsigned int Final_Weight_Index
			= LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (LAYER_AMOUNT_HIDDEN_EVAL - 1) * NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + Target_Neuron_Index;
		float Final_Weight = C->Eval_Network.Weight[Final_Weight_Index];
		
		C->Eval_Network.Delta_Weight[Weight_Index] = Delta_First_Neuron * Final_Weight;
		return;
	}

	float Delta_Neuron_Previous_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];
	float Delta_Neuron_Next_Layer[NEURONS_PER_HIDDEN_LAYER_EVAL];

	// First broadcast from the target neuron
	for (unsigned int i = 0; i < NEURONS_PER_HIDDEN_LAYER_EVAL; i++)
	{
		unsigned int First_Broadcast_Neuron_Weight_Index
			= Weight_Index + NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + (Target_Neuron_Index - Origin_Neuron_Index) * NEURONS_PER_HIDDEN_LAYER_EVAL - Target_Neuron_Index;
		float First_Broadcast_Neuron_Weight = C->Eval_Network.Weight[First_Broadcast_Neuron_Weight_Index];
		Delta_Neuron_Previous_Layer[i] = Delta_First_Neuron * First_Broadcast_Neuron_Weight;
	}

	for (unsigned int Layer_Index = Layer + 2; Layer_Index < LAYER_AMOUNT_EVAL - 2; Layer_Index++)
	{
		unsigned int Origin_Neuron_Index_Begin = LAYER_SIZE_INPUT_EVAL + NEURONS_PER_HIDDEN_LAYER_EVAL * (Layer_Index - 1);
		unsigned int Broadcast_Neuron_Index_Begin = LAYER_SIZE_INPUT_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL + NEURONS_PER_HIDDEN_LAYER_EVAL * NEURONS_PER_HIDDEN_LAYER_EVAL * (Layer_Index - 1);
	}



	C->Eval_Network.Delta_Weight[Weight_Index] = 0.f;
}
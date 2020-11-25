#pragma once

// CUDA
#include "curand_kernel.h"

// Project Headers
#include "Config.h"

//namespace GPGPU
//{
struct Vec2
{
	float X[2 * WARP_SIZE];
	float Y[2 * WARP_SIZE];
};

struct engine
{
	float Angle[2 * WARP_SIZE];
	float AngularVelocity[2 * WARP_SIZE];
	float AngularAcceleration[2 * WARP_SIZE];
	float Thrust[2 * WARP_SIZE];
	float ThrustNormalized[2 * WARP_SIZE];
	float ThrustNormalizedTemp[2 * WARP_SIZE];
};

struct cannon
{
	float Angle[2 * WARP_SIZE];
	float AngularVelocity[2 * WARP_SIZE];
	float AngularAcceleration[2 * WARP_SIZE];
};

struct bullet
{
	Vec2 Position;
	Vec2 Velocity;
	bool Active[2 * WARP_SIZE];
};

struct temp
{
	float Weights[WARP_SIZE * WEIGHT_COUNT];
};

struct TempPtrArr
{
	temp *Warp[WARP_COUNT];	// This only needs to be fit count
};

struct CraftState
{
	float Weights[WARP_SIZE * WEIGHT_COUNT];

	float Neuron[2 * WARP_SIZE * NEURON_COUNT];
	Vec2 Position;
	Vec2 Velocity;
	Vec2 Acceleration;
	float Angle[2 * WARP_SIZE];	// Up is 0.f
	float AngularVelocity[2 * WARP_SIZE];
	float AngularAcceleration[2 * WARP_SIZE];
	engine Engine[4];
	
	cannon Cannon;							// Up is 0.f
	bullet Bullet[10];
	short BulletCounter[2 * WARP_SIZE];		// Number of active bullets
	short BulletTimer[2 * WARP_SIZE];		// Craft can only shoot a bullet every so often

	curandState RandState[2 * WARP_SIZE];
	int Score[2 * WARP_SIZE];
	int ScoreTime[2 * WARP_SIZE];
	int ScoreBullet[2 * WARP_SIZE];
	int ScoreDistance[2 * WARP_SIZE];
	
	// Tournament Variables
	int ScoreCumulative[2 * WARP_SIZE];
	int ScoreTemp[2 * WARP_SIZE];
	int Place[WARP_SIZE];
	int ID[WARP_SIZE];
	int TempID[WARP_SIZE];
	
	bool Active[2 * WARP_SIZE];	// Craft becomes inactive when colliding with wall. Other craft still active and able to shoot.

	// For output only
	float CannonCommandAngle[2 * WARP_SIZE];
	float CannonStrength[2 * WARP_SIZE];

	// TODO: More points for earlier bullet hits
	// TODO: Make score bars fixed on the window for each side
	// TODO: Move MatchState parameters to CraftState
};

struct CraftPtrArr
{
	CraftState *Warp[WARP_COUNT];
};

struct MatchState
{
	bool Done[MATCH_COUNT];
	unsigned int ElapsedSteps[MATCH_COUNT];

	// Tournament Variables
	int TournamentEpochNumber;

	// Requirement: When this signal is high, it is guaranteed to be turned on within the render frame
	bool RenderOnFirstFrame[MATCH_COUNT];		// First frame render after switching from off to on. Used to assign color floats
	// Requirement: When this signal is high, it is guaranteed to be turned off within the render frame
	bool RenderOffFirstFrame[MATCH_COUNT];		// First frame after render turns from on to off
	bool RenderOn[MATCH_COUNT];					// Render toggle

	bool AllDone;
};
 
struct state
{
	int Score;

	int ScoreBullet;
	int ScoreTime;
	int ScoreDistance;

	int ScoreCumulative;

	float PositionX;
	float PositionY;

	float VelocityX;
	float VelocityY;

	float AccelerationX;
	float AccelerationY;

	float Angle;
	float AngularVelocity;
	float AngularAcceleration;

	float CannonAngle;

	float EngineAngle[4];
	float EngineAngularVelocity[4];
	float EngineAngularAcceleration[4];
	float EngineThrustNormalized[4];

	float CannonCommandAngle;
	float CannonStrength;

	bool Active;

	float Neuron[NEURON_COUNT];
};

struct GraphicsObjectPointer
{
	float *Fuselage[WARP_COUNT]					= { nullptr };
	float *Wing[WARP_COUNT]						= { nullptr };
	float *Cannon[WARP_COUNT]					= { nullptr };
	float *Engine[WARP_COUNT][4]				= { nullptr };
	float *ThrustLong[WARP_COUNT][4]			= { nullptr };
	float *ThrustShort[WARP_COUNT][4]			= { nullptr };
	float *Bullet[WARP_COUNT][BULLET_COUNT_MAX] = { nullptr };
};
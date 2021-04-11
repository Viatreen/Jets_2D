#pragma once

// CUDA
#include "curand_kernel.h"

// Project Headers
#include "Config.h"

struct Vec2
{
	float X[2 * CRAFT_COUNT];
	float Y[2 * CRAFT_COUNT];
};

struct engine
{
	float Angle[2 * CRAFT_COUNT];
	float AngularVelocity[2 * CRAFT_COUNT];
	float AngularAcceleration[2 * CRAFT_COUNT];
	float Thrust[2 * CRAFT_COUNT];
	float ThrustNormalized[2 * CRAFT_COUNT];
	float ThrustNormalizedTemp[2 * CRAFT_COUNT];
};

struct cannon
{
	float Angle[2 * CRAFT_COUNT];
	float AngularVelocity[2 * CRAFT_COUNT];
	float AngularAcceleration[2 * CRAFT_COUNT];
};

struct bullet
{
	Vec2 Position;
	Vec2 Velocity;
	bool Active[2 * CRAFT_COUNT];
};

struct temp
{
	float Weight[FIT_COUNT * WEIGHT_COUNT];
};

struct CraftState
{
	float Weight[CRAFT_COUNT * WEIGHT_COUNT];

	float Neuron[2 * CRAFT_COUNT * NEURON_COUNT];
	Vec2 Position;
	Vec2 Velocity;
	Vec2 Acceleration;
	float Angle[2 * CRAFT_COUNT];	// Up is 0.f
	float AngularVelocity[2 * CRAFT_COUNT];
	float AngularAcceleration[2 * CRAFT_COUNT];
	engine Engine[4];
	
	cannon Cannon;							// Up is 0.f
	bullet Bullet[10];
	short BulletCounter[2 * CRAFT_COUNT];		// Number of active bullets
	short BulletTimer[2 * CRAFT_COUNT];		// Craft can only shoot a bullet every so often

	curandState RandState[2 * CRAFT_COUNT];
	int Score[2 * CRAFT_COUNT];
	int ScoreTime[2 * CRAFT_COUNT];
	int ScoreBullet[2 * CRAFT_COUNT];
	int ScoreDistance[2 * CRAFT_COUNT];
	
	// Tournament Variables
	int ScoreCumulative[2 * CRAFT_COUNT];
	int ScoreTemp[2 * CRAFT_COUNT];
	int Place[CRAFT_COUNT];
	int ID[CRAFT_COUNT];
	int TempID[CRAFT_COUNT];
	
	bool Active[2 * CRAFT_COUNT];	// Craft becomes inactive when colliding with wall. Other craft still active and able to shoot.

	// For output only
	float CannonCommandAngle[2 * CRAFT_COUNT];
	float CannonStrength[2 * CRAFT_COUNT];

	// TODO: More points for earlier bullet hits
	// TODO: Make score bars fixed on the window for each side
	// TODO: Move MatchState parameters to CraftState
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
	float *Fuselage					= { nullptr };
	float *Wing						= { nullptr };
	float *Cannon					= { nullptr };
	float *Engine[4]				= { nullptr };
	float *ThrustLong[4]			= { nullptr };
	float *ThrustShort[4]			= { nullptr };
	float *Bullet[BULLET_COUNT_MAX]	= { nullptr };
};
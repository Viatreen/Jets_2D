#pragma once

// CUDA
#include <curand_kernel.h>

// Project Headers
#include "Jets_2D/Config.hpp"

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
    float Weight[FIT_COUNT * WEIGHT_AMOUNT];
};

struct eval_Network
{
    float Weight[WEIGHT_AMOUNT_EVAL];
    float Neuron[NEURON_AMOUNT_EVAL];
    float Delta_Result_Over_Delta_Weight[WEIGHT_AMOUNT_EVAL]; // For backpropagation
    float Delta_Neuron[NEURON_AMOUNT_EVAL];
};

struct CraftState
{
    eval_Network Eval_Network;
    float Weight[CRAFT_COUNT * WEIGHT_AMOUNT];
    float Neuron[2 * CRAFT_COUNT * NEURON_AMOUNT];
    curandState RandState[2 * CRAFT_COUNT];

    Vec2 Position;
    Vec2 Velocity;
    Vec2 Acceleration;
    float Angle[2 * CRAFT_COUNT];   // Up is 0.f
    float AngularVelocity[2 * CRAFT_COUNT];
    float AngularAcceleration[2 * CRAFT_COUNT];
    engine Engine[4];
    
    cannon Cannon;                          // Up is 0.f
    bullet Bullet[10];
    short BulletCounter[2 * CRAFT_COUNT];       // Number of active bullets
    short BulletTimer[2 * CRAFT_COUNT];     // Craft can only shoot a bullet every so often

    float Score[2 * CRAFT_COUNT];
    float ScoreTime[2 * CRAFT_COUNT];
    float ScoreBullet[2 * CRAFT_COUNT];
    float ScoreDistance[2 * CRAFT_COUNT];
    float ScoreFuelEfficiency[2 * CRAFT_COUNT];
    
    // Tournament Variables
    float ScoreCumulative[2 * CRAFT_COUNT];
    float ScoreTemp[2 * CRAFT_COUNT];
    int Place[CRAFT_COUNT];
    int ID[CRAFT_COUNT];
    int TempID[CRAFT_COUNT];
    
    bool Active[2 * CRAFT_COUNT];   // Craft becomes inactive when colliding with wall. Other craft still active and able to shoot.

    // For output only
    float CannonCommandAngle[2 * CRAFT_COUNT];
    float CannonStrength[2 * CRAFT_COUNT];

    // TODO: More points for earlier bullet hits
    // TODO: Move MatchState parameters to CraftState
};

struct MatchState
{
    bool Done[MATCH_COUNT];
    unsigned int ElapsedTicks[MATCH_COUNT];

    // Tournament Variables
    // Requirement: When this signal is high, it is guaranteed to be turned on within the render frame
    bool RenderOnFirstFrame[MATCH_COUNT];       // First frame render after switching from off to on. Used to assign color floats
    // Requirement: When this signal is high, it is guaranteed to be turned off within the render frame
    bool RenderOffFirstFrame[MATCH_COUNT];      // First frame after render turns from on to off
    bool RenderOn[MATCH_COUNT];                 // Current render status

    bool AllDone;
};
 
struct state
{
    float Score;

    float ScoreBullet;
    float ScoreTime;
    float ScoreDistance;
    float ScoreFuelEfficiency;

    float ScoreCumulative;

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

    float Neuron[NEURON_AMOUNT];
};

struct GraphicsObjectPointer
{
    float *Fuselage                 = { nullptr };
    float *Wing                     = { nullptr };
    float *Cannon                   = { nullptr };
    float *Engine[4]                = { nullptr };
    float *ThrustLong[4]            = { nullptr };
    float *ThrustShort[4]           = { nullptr };
    float *Bullet[BULLET_COUNT_MAX] = { nullptr };
};
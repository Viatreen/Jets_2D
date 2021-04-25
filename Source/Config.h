#pragma once

// Standard Library
#include <cmath>

#define GTX_1080TI							  // Sets SM count to 28

// Constants
#define PI									  3.14159f
#define GRAVITY								  9.8f			// Meter / sec^2

#define LIFE_RADIUS							  15.f 

// Graphics
#define FRAMES_PER_SECOND					  64
#define FRAMERATE_NN						  32
#define FRAMERATE_PHYSICS					  64
#define FRAMERATE_NN_PHYSICS				( FRAMERATE_PHYSICS / FRAMERATE_NN )
#define TIME_STEP							( 1.f / float(FRAMERATE_PHYSICS) )	// Divide by a power of 2 for bit manipulation+
#define TIME_MATCH							  32.f  // Seconds

// CUDA
#define BLOCK_SIZE							  256

#ifdef GTX_1080TI							  
#define SM_COUNT							  28 
#define TIME_SPEED_FAST_DEFAULT				  512
#else
#define SM_COUNT							  2
#define TIME_SPEED_FAST_DEFAULT				  32
#endif

// Match Configuration
#define CRAFT_COUNT							( 128 * 8 * SM_COUNT  )
#define FIT_COUNT							( CRAFT_COUNT / 2 )		// Must be a factor of CRAFT_COUNT
// FIT_COUNT must be a factor of CRAFT_COUNT
#define TOURNAMENTS_PER_ROUND				  1
#define MATCH_COUNT							( CRAFT_COUNT )
#define OPPONENT_RANK_RANGE_DEFAULT			  128		// Must be equal or less than FIT_COUNT

// Dimensions and Mass (Meters, Kg)
#define CG_OFFSET_Y							  0.2f	// CG is this far below graphical center
#define CRAFT_MASS							  500.f				
#define CRAFT_MASS_INVERSE					( 1 / CRAFT_MASS)
#define CRAFT_ROTATIONAL_INERTIA			  655.f
#define CRAFT_ROTATIONAL_INERTIA_INVERSE	( 1.f / CRAFT_ROTATIONAL_INERTIA)
#define ENGINE_ROTATIONAL_INERTIA			  0.72f								// kg m^2
#define ENGINE_ROTATIONAL_INERTIA_INVERSE	( 1.f / ENGINE_ROTATIONAL_INERTIA)
#define CANNON_ROTATIONAL_INERTIA			  1.f
#define CANNON_ROTATIONAL_INERTIA_INVERSE	( 1.f / CANNON_ROTATIONAL_INERTIA)

// Craft Geometry in Meters
#define WINGSPAN							  4.f
#define WING_HEIGHT							  0.14f
#define FUSELAGE_RADIUS						  0.7f
#define FUSELAGE_VERT_COUNT					  32
#define WINDSHIELD_WIDTH					  0.528f
#define WINDSHEILD_HEIGHT					  0.8f
#define CANNON_WIDTH						  0.1f
#define CANNON_HEIGHT						( (float)sqrt(FUSELAGE_RADIUS * FUSELAGE_RADIUS - CANNON_WIDTH * CANNON_WIDTH / 4))
#define BULLET_RADIUS						  0.1f
#define BULLET_VELOCITY_INITIAL				( LIFE_RADIUS )	// m/s
#define BULLET_INTERVAL_MIN					  0.8f					// Seconds		// TODO: Experiment with this
#define BULLET_COUNT_MAX					  1
#define BULLET_VERT_COUNT					  8

#define ENGINE_WIDTH						  0.24f
#define ENGINE_HEIGHT						  0.48f
#define ENGINE_0_DISTANCE					(-WINGSPAN / 2 * 0.9f )		// Meters
#define ENGINE_1_DISTANCE					(-WINGSPAN / 2 * 0.7f )
#define ENGINE_2_DISTANCE					( WINGSPAN / 2 * 0.7f )
#define ENGINE_3_DISTANCE					( WINGSPAN / 2 * 0.9f )
#define THRUST_LENGTH_FULL					  0.76f
#define THRUST_LENGTH_FULL_SHORT			( THRUST_LENGTH_FULL / 3)
#define ENGINE_ANGULAR_ACCEL				  16.f
#define ENGINE_MAX_ANGULAR_ACCEL			( 2 * PI * 4 )
#define CANNON_ANGULAR_ACCEL				  16.f
#define CANNON_MAX_ANGULAR_ACCEL			( 2 * PI * 4 )
#define CANNON_VELOCITY_MAX					  64.f
#define ENGINE_ANGLE_MAX_IN					( 15.f / 180.f * PI )
#define ENGINE_INBOARD_ANGLE_MAX_OUT		( 60.f / 180.f * PI )
#define ENGINE_OUTBOARD_ANGLE_MAX_OUT		(-90.f / 180.f * PI )

#define THRUST_MAX							( CRAFT_MASS / 2.4f * 9.8f )									// N
#define THRUST_MIN							  0.25f														// Normalized thrust kgf. Thrust max is 1.f
#define THRUST_MIN_RAMP_TIME				  1.5f														// Seconds. Time to ramp from THRUST_MIN to 1.f thrust.
#define THRUST_RAMP_MAX						( TIME_STEP / THRUST_MIN_RAMP_TIME * (1.f - THRUST_MIN) )	// Max allowable change in thrust per timestep
#define THRUST_NORMALIZED_INITIAL			( CRAFT_MASS * GRAVITY / THRUST_MAX / 4.f )					// Initial state. Normalized

// Engine actuator


// Craft default colors
#define FUSELAGE_RED			( 100.f / 255.f)
#define FUSELAGE_GREEN			( 120.f / 255.f)
#define FUSELAGE_BLUE			( 120.f / 255.f)

#define FUSELAGE_RED_TRAINEE	( 145.f / 255.f)
#define FUSELAGE_GREEN_TRAINEE	( 129.f / 255.f)
#define FUSELAGE_BLUE_TRAINEE	( 81.f / 255.f)

#define WING_RED				( 80.f  / 255.f)
#define WING_GREEN				( 100.f / 255.f)
#define WING_BLUE				( 100.f / 255.f)

#define CANNON_RED				( 80.f  / 255.f)
#define CANNON_GREEN			( 100.f / 255.f)
#define CANNON_BLUE				( 100.f / 255.f)

#define ENGINE_RED				( 100.f / 255.f)
#define ENGINE_GREEN			( 120.f / 255.f)
#define ENGINE_BLUE				( 120.f / 255.f)

#define THRUST_BIG_RED			( 230.f / 255.f)
#define THRUST_BIG_GREEN		( 150.f / 255.f)
#define THRUST_BIG_BLUE			  0.f

#define THRUST_SMALL_RED		( 180.f / 255.f)
#define THRUST_SMALL_GREEN		( 120.f / 255.f)
#define THRUST_SMALL_BLUE		  0.f

#define BULLET_RED				( 255.f / 255.f)
#define BULLET_GREEN			  0.f
#define BULLET_BLUE				  0.f

// Neural Network Definition				 
#define SENSORS_EDGE_DISTANCE_COUNT			  4		// Must be even to allow fast compute of opposite sides

#define SENSORS_VELOCITY_START				( SENSORS_EDGE_DISTANCE_COUNT * 2)
#define SENSORS_VELOCITY_COUNT				  4
#define SENSORS_VELOCITY_SCALE				( 1.f / 8.f )

#define SENSORS_ANG_VEL_START				( SENSORS_VELOCITY_START + SENSORS_VELOCITY_COUNT)
#define SENSORS_ANG_VEL_COUNT				  2
#define SENSORS_ANG_VEL_SCALE				( 1.f )

#define SENSORS_EXTERNAL_START				( SENSORS_ANG_VEL_START + SENSORS_ANG_VEL_COUNT)
#define SENSORS_EXTERNAL_FORCE_COUNT		  8

#define SENSORS_ENGINE_ANGLE_START			( SENSORS_EXTERNAL_START + SENSORS_EXTERNAL_FORCE_COUNT * 2)
#define SENSORS_ENGINE_ANGLE_COUNT			  4

#define SENSORS_OPPONENT_ANGLE_START		( SENSORS_ENGINE_ANGLE_START + SENSORS_ENGINE_ANGLE_COUNT * 4 * 2 )
#define SENSORS_OPPONENT_ANGLE_COUNT		  4

#define SENSORS_OPPONENT_DISTANCE_START		( SENSORS_OPPONENT_ANGLE_START + SENSORS_OPPONENT_ANGLE_COUNT * 2)
#define SENSORS_OPPONENT_DISTANCE_COUNT		  2

#define SENSORS_BULLET_ANGLE_START			( SENSORS_OPPONENT_DISTANCE_START + SENSORS_OPPONENT_DISTANCE_COUNT)
#define SENSORS_BULLET_ANGLE_COUNT			  4

#define SENSORS_BULLET_DISTANCE_START		( SENSORS_BULLET_ANGLE_START + SENSORS_BULLET_ANGLE_COUNT * 2)
#define SENSORS_BULLET_DISTANCE_COUNT		  2

#define SENSORS_ANGLE_START					( SENSORS_BULLET_DISTANCE_START + SENSORS_BULLET_DISTANCE_COUNT)
#define SENSORS_ANGLE_COUNT					  2	

#define SENSORS_MEMORY_START				( SENSORS_ANGLE_START + SENSORS_ANGLE_COUNT)
#define SENSORS_MEMORY_COUNT				  0	// From 1/32 seconds to 32 seconds
#define SENSORS_BIAS_NEURON_COUNT			  1


#define LAYER_SIZE_INPUT					( SENSORS_EDGE_DISTANCE_COUNT * 2 + SENSORS_VELOCITY_COUNT + SENSORS_ANG_VEL_COUNT\
												+ SENSORS_EXTERNAL_FORCE_COUNT * 2 + SENSORS_ENGINE_ANGLE_COUNT * 4 * 2\
												 + SENSORS_OPPONENT_ANGLE_COUNT * 2 + SENSORS_OPPONENT_DISTANCE_COUNT\
												 + SENSORS_BULLET_ANGLE_COUNT * 2 + SENSORS_BULLET_DISTANCE_COUNT + SENSORS_ANGLE_COUNT\
												 + SENSORS_MEMORY_COUNT + SENSORS_BIAS_NEURON_COUNT)

#define LAYER_AMOUNT_HIDDEN					  3
#define NEURONS_PER_LAYER					  16
#define LAYER_AMOUNT						( 2 + LAYER_AMOUNT_HIDDEN )		// Input, Hidden, and Output
#define HIDDEN_NEURON_AMOUNT				( LAYER_AMOUNT_HIDDEN * NEURONS_PER_LAYER )

#define LAYER_SIZE_OUTPUT					( 25 + SENSORS_MEMORY_COUNT)
//#define LAYER_ARRAY							{ LAYER_SIZE_INPUT, LAYER_SIZE_HIDDEN, LAYER_SIZE_HIDDEN, LAYER_SIZE_OUTPUT }
//#define LAYER_BEGIN_INDEX					{ 0, LAYER_SIZE_INPUT, LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN, LAYER_SIZE_INPUT + 2 * LAYER_SIZE_HIDDEN }
#define LAYER_ARRAY							{ LAYER_SIZE_INPUT, NEURONS_PER_LAYER, NEURONS_PER_LAYER, LAYER_SIZE_OUTPUT }

#define NEURON_COUNT						( LAYER_SIZE_INPUT + LAYER_AMOUNT_HIDDEN * NEURONS_PER_LAYER + LAYER_SIZE_OUTPUT )							// Sum of layer array
#define WEIGHT_COUNT						( LAYER_SIZE_INPUT * NEURONS_PER_LAYER + ( LAYER_AMOUNT_HIDDEN - 1 ) * NEURONS_PER_LAYER * NEURONS_PER_LAYER + NEURONS_PER_LAYER * LAYER_SIZE_OUTPUT )		// Number of weights ~4500
//#define WEIGHT_BEGIN_INDEX_ARRAY			{ 0, LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN, LAYER_SIZE_INPUT * LAYER_SIZE_HIDDEN + LAYER_SIZE_HIDDEN * LAYER_SIZE_HIDDEN }
#define OUTPUT_LAYER_NEURON_BEGIN_INDEX		( LAYER_SIZE_INPUT + LAYER_AMOUNT_HIDDEN * NEURONS_PER_LAYER )

#define WEIGHTS_MULTIPLIER					  0.25f
#define NETWORK_ACTIVATION_SLOPE			  0.01f
#define NETWORK_INVERSE_ACTIVATION_SLOPE	( 1.f / NETWORK_ACTIVATION_SLOPE )

#define SAVE_COUNT_DEFAULT					( CRAFT_COUNT / 2)

#define SHRINK_COEFFICIENT_WEIGHTS			  0.9999f

// Set floating point of neural net to half2 for arch that supports it (Volta)
// Else, use standard float (32-bit)
// #define VOLTA
// #ifdef VOLTA
// #include <cuda_fp16.h>
// typedef __half fp_NN;
// #define fp_NN_To_fp32(x)	__half2float(x)
// #define fp32_To_fp_NN(x)	__float2half_rn(x)
// #else
// typedef float fp_NN;
// #define fp_NN_To_fp32(x)	(x)
// #define fp32_To_fp_NN(x)	(x)
// #endif

// TODO: Add a few rounds where the unfit are replaced by random NNs
namespace Config_
{
	// Default rendering
	extern bool RenderAllDefault;
	extern bool RenderOneDefault;
	extern bool RenderFitDefault;
	extern bool RenderNoneDefault;		// Setting this true defaults fast simulation
	
	extern int LayerSizeArray[LAYER_AMOUNT];
}

// Initial Values
struct config
{
	float MutationFlipChance	= 0.005f;	// Percent chance of each weight flipping sign
	float MutationScaleChance	= 0.08f;	// Percent chance of each weight mutating		
	float MutationScale			= 0.1f;		// Percent that each weight could possibly change
	float MutationSlideChance	= 0.08f;
	float MutationSigma			= 0.08f;	// Sigma parameter of normal distribution
	float WeightMax				= 1.f;		// Maximum magnitude of a weight
	
	float TimeLimitMatch		= TIME_MATCH;
	int TimeStepLimit			= round(TimeLimitMatch / TIME_STEP);
	int TimeSpeed				= 1;
	int TimeSpeedFastDefault	= TIME_SPEED_FAST_DEFAULT;
	int TimeSpeedFast			= TimeSpeedFastDefault;

	int IterationsPerCall		= round(float(TimeSpeed) / TIME_STEP / FRAMES_PER_SECOND);
	int RoundNumber				= 0;

	float BulletDamage			= float(TimeStepLimit / 2);
};

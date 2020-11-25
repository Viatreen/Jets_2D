#pragma once

// Project Headers
#include "GPGPU/State.h"
#include "Graphics/Axis.h"
#include "Graphics/Camera.h"
#include "Graphics/Circle.h"
#include "Graphics/CircleOfLife.h"
#include "Graphics/Component.h"
#include "Graphics/Thrust.h"
#include "GL/GLSetup.h"
#include "GL/Shader.h"

Shader CraftShader;
 
namespace Graphics
{
	void Setup()
	{
		CraftShader.Create("Load/Shaders/Craft.shader");

		//Axis[0]				= new axis(AxisVertices::X);
		//Axis[1]				= new axis(AxisVertices::Y);
		CircleOfLife		= new circleOfLife();

		for (int i = 0; i < WARP_COUNT; i++)
		{
			Craft::Fuselage[i]	= new Craft::Circle(FUSELAGE_VERT_COUNT);
			Craft::Wing[i]		= new Craft::Component();
			Craft::Cannon[i]	= new Craft::Component();
		
			for (int j = 0; j < 4; j++)
			{
				Craft::Engine[i][j]			= new Craft::Component();
				Craft::ThrustLong[i][j]	= new Craft::Thrust();
				Craft::ThrustShort[i][j]		= new Craft::Thrust();
			}
		
			for (int j = 0; j < BULLET_COUNT_MAX; j++)
				Craft::Bullet[i][j] = new Craft::Circle(BULLET_VERT_COUNT);
		}
	}

	void Shutdown()
	{
		delete CircleOfLife;

		for (int i = 0; i < WARP_COUNT; i++)
		{
			delete Craft::Fuselage[i];
			delete Craft::Wing[i];
			delete Craft::Cannon[i];
		
			for (int j = 0; j < 4; j++)
			{
				delete Craft::Engine[i][j];
				delete Craft::ThrustLong[i][j];
				delete Craft::ThrustShort[i][j];
			}
		
			for (int j = 0; j < BULLET_COUNT_MAX; j++)
				delete Craft::Bullet[i][j];
		}

		glfwDestroyWindow(window);
		glfwTerminate();
	}
}

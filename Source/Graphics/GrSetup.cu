// File Header
#include "GrSetup.h"

// Project Headers
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
		#if defined(__linux)
			CraftShader.Create("./res/Shaders/Craft.shader");
		#elif defined(_WIN32)
			CraftShader.Create("../../../res/Shaders/Craft.shader");
		#endif

		//Axis[0]				= new axis(AxisVertices::X);
		//Axis[1]				= new axis(AxisVertices::Y);
		CircleOfLife = new circleOfLife();

		Craft::Fuselage = new Craft::Circle(FUSELAGE_VERT_COUNT);
		Craft::Wing = new Craft::Component();
		Craft::Cannon = new Craft::Component();

		for (int j = 0; j < 4; j++)
		{
			Craft::Engine[j] = new Craft::Component();
			Craft::ThrustLong[j] = new Craft::Thrust();
			Craft::ThrustShort[j] = new Craft::Thrust();
		}

		for (int j = 0; j < BULLET_COUNT_MAX; j++)
			Craft::Bullet[j] = new Craft::Circle(BULLET_VERT_COUNT);
	}

	void Shutdown()
	{
		delete CircleOfLife;

		delete Craft::Fuselage;
		delete Craft::Wing;
		delete Craft::Cannon;

		for (int j = 0; j < 4; j++)
		{
			delete Craft::Engine[j];
			delete Craft::ThrustLong[j];
			delete Craft::ThrustShort[j];
		}

		for (int j = 0; j < BULLET_COUNT_MAX; j++)
			delete Craft::Bullet[j];

		glfwDestroyWindow(window);
		glfwTerminate();
	}
}

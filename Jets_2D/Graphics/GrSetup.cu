// File Header
#include "Jets_2D/Graphics/GrSetup.h"

// Project Headers
#include "Jets_2D/Graphics/Camera.h"
#include "Jets_2D/Graphics/Circle.h"
#include "Jets_2D/Graphics/CircleOfLife.h"
#include "Jets_2D/Graphics/Component.h"
#include "Jets_2D/Graphics/Thrust.h"
#include "Jets_2D/GL/GLSetup.h"
#include "Jets_2D/GL/Shader.h"

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

        //Axis[0]               = new axis(AxisVertices::X);
        //Axis[1]               = new axis(AxisVertices::Y);
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

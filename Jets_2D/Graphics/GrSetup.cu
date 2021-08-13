// File Header
#include "Jets_2D/Graphics/GrSetup.hpp"

// Project Headers
#include "Jets_2D/Graphics/Camera.hpp"
#include "Jets_2D/Graphics/Circle.hpp"
#include "Jets_2D/Graphics/CircleOfLife.hpp"
#include "Jets_2D/Graphics/Component.hpp"
#include "Jets_2D/Graphics/Thrust.hpp"
#include "Jets_2D/GL/GLSetup.hpp"
#include "Jets_2D/GL/Shader.hpp"

Shader CraftShader;

namespace Graphics
{
    void Setup()
    {
        CraftShader.Create("./res/Shaders/Craft.shader");

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

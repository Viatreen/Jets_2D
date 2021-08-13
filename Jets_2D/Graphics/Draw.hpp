#pragma once

// GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// Project Headers
#include "Jets_2D/Graphics/Axis.hpp"
#include "Jets_2D/Graphics/Circle.hpp"
#include "Jets_2D/Graphics/CircleOfLife.hpp"
#include "Jets_2D/Graphics/Component.hpp"
#include "Jets_2D/Graphics/GrSetup.hpp"
#include "Jets_2D/Graphics/Thrust.hpp"
#include "Jets_2D/GUI/GUI.hpp"
#include "Jets_2D/Graphics/Camera.hpp"

glm::mat4 Projection;
glm::mat4 View;

namespace Graphics
{
    void Draw()
    {
        Projection = glm::perspective(Camera.zoom, (float)(GL::ScreenWidth - GUI::SideBarWidth) / (float)(GL::ScreenHeight - GUI::ProgressHeight - GUI::MenuHeight), 0.1f, 1000.0f);   // Camera perspective transformation
        //Projection = glm::perspective(Camera.zoom, (float)(GL::ScreenWidth) / (float)(GL::ScreenHeight), 0.1f, 1000.0f);  // Camera perspective transformation
        //View = Camera.ViewMatrix();
        View = glm::lookAt(glm::vec3(0.f, 0.f, 2.5f * LIFE_RADIUS), glm::vec3(0.f, 0.f, 2.5f * LIFE_RADIUS) + glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 1.f, 0.f));

        GLCheck(glViewport(0, 0, ((GL::ScreenWidth > GUI::SideBarWidth) ? GL::ScreenWidth - GUI::SideBarWidth : GUI::SideBarWidth), ((GL::ScreenHeight > GUI::ProgressHeight + GUI::MenuHeight) ? GL::ScreenHeight - GUI::ProgressHeight - GUI::MenuHeight : GUI::ProgressHeight + GUI::MenuHeight)));
        //GLCheck(glViewport(0, 0, GL::ScreenWidth, GL::ScreenHeight));

        CraftShader.Bind();

        unsigned int ProjLocation = glGetUniformLocation(CraftShader.ID, "P");              // Projection
        unsigned int ViewLocation = glGetUniformLocation(CraftShader.ID, "V");              // View
        GLCheck(glUniformMatrix4fv(ProjLocation, 1, GL_FALSE, glm::value_ptr(Projection))); // Pretty sure I don't need this
        GLCheck(glUniformMatrix4fv(ViewLocation, 1, GL_FALSE, glm::value_ptr(View)));

        //Axis[X]->Draw();
        //Axis[Y]->Draw();

        CircleOfLife->Draw();

        Craft::Wing->Draw();
        Craft::Fuselage->Draw(FUSELAGE_VERT_COUNT);
        Craft::Cannon->Draw();
        
        for (int j = 0; j < 4; j++)
        {
            Craft::Engine[j]->Draw();
            Craft::ThrustLong[j]->Draw();
            Craft::ThrustShort[j]->Draw();
        }
        
        for (int j = 0; j < BULLET_COUNT_MAX; j++)
            Craft::Bullet[j]->Draw(BULLET_VERT_COUNT);
        
        CraftShader.Unbind();
    }
}
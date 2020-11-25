#pragma once

// Standard Library
#include <cmath>
#include <iostream>

// OpenGL
#include "glad/glad.h"

// Project Headers
#include "GL/Shader.h"

struct circleOfLife		// Singleton
{
	float Vertices[64 * 5];

	unsigned int VA, VB;

	circleOfLife()
	{
		for (int i = 0; i < 64; i++)
		{
			Vertices[i]				= LIFE_RADIUS * sin(float(i) / 64 * 2 * 3.14159f);	// TODO: Renormalize this
			Vertices[i + 1 * 64]	= LIFE_RADIUS * cos(float(i) / 64 * 2 * 3.14159f);
			Vertices[i + 2 * 64]	= 1.f;	// R
			Vertices[i + 3 * 64]	= 0.f;	// G
			Vertices[i + 4 * 64]	= 0.f;	// B
		}

		// Retrieve GL indices and load to GPU memory
		GLCheck(glGenVertexArrays(1, &VA));
		GLCheck(glGenBuffers(1, &VB));
		GLCheck(glBindVertexArray(VA));
		GLCheck(glBindBuffer(GL_ARRAY_BUFFER, VB));
		GLCheck(glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW));

		// Vertex Positions
		GLCheck(glEnableVertexAttribArray(0));
		GLCheck(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0));

		GLCheck(glEnableVertexAttribArray(1));
		GLCheck(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(1 * 64 * sizeof(float))));

		// Vertex Colors
		GLCheck(glEnableVertexAttribArray(2));
		GLCheck(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * 64 * sizeof(float))));

		GLCheck(glEnableVertexAttribArray(3));
		GLCheck(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(3 * 64 * sizeof(float))));

		GLCheck(glEnableVertexAttribArray(4));
		GLCheck(glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(4 * 64 * sizeof(float))));
	}
	~circleOfLife()
	{
		GLCheck(glDeleteVertexArrays(1, &VA));
		GLCheck(glDeleteBuffers(1, &VB));
	}
	void Draw()
	{
		GLCheck(glBindVertexArray(VA));
		GLCheck(glDrawArrays(GL_LINE_LOOP, 0, sizeof(Vertices) / sizeof(unsigned int) / 5));
		GLCheck(glBindVertexArray(0));
	}
};

circleOfLife* CircleOfLife;

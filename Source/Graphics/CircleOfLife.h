#pragma once

struct circleOfLife		// Singleton
{
	float Vertices[64 * 5];

	unsigned int VA, VB;

	circleOfLife();
	~circleOfLife();
	void Draw();
};

extern circleOfLife* CircleOfLife;

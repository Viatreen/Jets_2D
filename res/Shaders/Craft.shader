#shader vertex
#version 450

layout(location = 0) in float PositionX;
layout(location = 1) in float PositionY;
layout(location = 2) in float ColorR;
layout(location = 3) in float ColorG;
layout(location = 4) in float ColorB;

out vec4 ColorVertex;

uniform mat4 P; // Projection
uniform mat4 V;	// View

void main()
{
	gl_Position = P * V * vec4(PositionX, PositionY, 0.0f, 1.0f);
	ColorVertex = vec4(ColorR, ColorG, ColorB, 1.f);
}

#shader fragment
#version 450

in vec4 ColorVertex;

out vec4 ColorFragment;

void main()
{
	ColorFragment = vec4(ColorVertex);
}
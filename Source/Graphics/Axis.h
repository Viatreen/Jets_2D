#pragma once

// Project Headers
#include "Config.h"

enum
{
    X = 0,
    Y
};

namespace AxisVertices
{
    extern float X[10];
    extern float Y[10];
}

struct axis     // TODO: Finish this
{
    unsigned int VA, VB;
    axis(float* vertices);
    ~axis();
    void Draw();
};

extern axis* Axis[2];

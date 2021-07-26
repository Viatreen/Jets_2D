// File Header
#include "Vertices.h"

// CUDA
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

// Project Headers
#include "GPGPU/GPSetup.h"
#include "GPGPU/State.h"

//namespace GPGPU
//{
__device__ void Rotate(float& X, float& Y, float Theta)
{
    float X_Temp = X * cos(Theta) - Y * sin(Theta);
    Y = X * sin(Theta) + Y * cos(Theta);
    X = X_Temp;
}

__device__ void Rotate(float X_In, float Y_In, float Theta, float& X_Out, float& Y_Out)
{
    X_Out = X_In * cos(Theta) - Y_In * sin(Theta);
    Y_Out = X_In * sin(Theta) + Y_In * cos(Theta);
}

__device__ void ShowBullet(GraphicsObjectPointer* Buffer, int ID, int BulletNumber)
{
    // Edge Vertex Positions
    for (int i = 0; i < BULLET_VERT_COUNT + 1; i++)
    {
        Buffer->Bullet[BulletNumber][2 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID] = BULLET_RED;
        Buffer->Bullet[BulletNumber][3 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID] = BULLET_GREEN;
        Buffer->Bullet[BulletNumber][4 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID] = BULLET_BLUE;
    }
}

__device__ void ConcealBullet(GraphicsObjectPointer* Buffer, int ID, int BulletNumber)
{
    for (int i = 0; i < 5 * (BULLET_VERT_COUNT + 1); i++)
    {
        Buffer->Bullet[BulletNumber][CRAFT_COUNT * 2 * i + ID] = 0.f;
    }
}   // End ConcealVertices function

__device__ void ConcealVertices(GraphicsObjectPointer* Buffer, int idxLeft, int idxRight)
{
    for (int i = 0; i < 5 * (FUSELAGE_VERT_COUNT + 1); i++)
    {
        Buffer->Fuselage[CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
        Buffer->Fuselage[CRAFT_COUNT * 2 * i + idxRight] = 0.f;
    }

#pragma unroll
    for (int i = 0; i < 5 * 4; i++) // Number of Vertex Attributes
    {
        Buffer->Wing[CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
        Buffer->Wing[CRAFT_COUNT * 2 * i + idxRight] = 0.f;

        Buffer->Cannon[CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
        Buffer->Cannon[CRAFT_COUNT * 2 * i + idxRight] = 0.f;

#pragma unroll
        for (int j = 0; j < 4; j++) // Engine Count
        {
            Buffer->Engine[j][CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
            Buffer->Engine[j][CRAFT_COUNT * 2 * i + idxRight] = 0.f;
        }
    }

#pragma unroll
    for (int i = 0; i < 5 * 3; i++) // Number of Vertex Attributes
        for (int j = 0; j < 4; j++) // Engine Count
        {
            Buffer->ThrustLong[j][CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
            Buffer->ThrustLong[j][CRAFT_COUNT * 2 * i + idxRight] = 0.f;
            Buffer->ThrustShort[j][CRAFT_COUNT * 2 * i + idxLeft] = 0.f;
            Buffer->ThrustShort[j][CRAFT_COUNT * 2 * i + idxRight] = 0.f;
        }

    // Bullet
#pragma unroll
    for (int i = 0; i < BULLET_COUNT_MAX; i++)
        ConcealBullet(Buffer, idxLeft, i);

#pragma unroll
    for (int i = 0; i < BULLET_COUNT_MAX; i++)
        ConcealBullet(Buffer, idxRight, i);
}   // End ConcealVertices function

__device__ void ShowVertices(CraftState* C, GraphicsObjectPointer* Buffer, int ID1, int ID2)
{
    // Edge Vertex Positions
    for (int i = 0; i < FUSELAGE_VERT_COUNT + 1; i++)
    {
        Buffer->Fuselage[2 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID1] = FUSELAGE_RED_TRAINEE;
        Buffer->Fuselage[2 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID2] = FUSELAGE_RED;
        Buffer->Fuselage[3 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID1] = FUSELAGE_GREEN_TRAINEE;
        Buffer->Fuselage[3 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID2] = FUSELAGE_GREEN;
        Buffer->Fuselage[4 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID1] = FUSELAGE_BLUE_TRAINEE;
        Buffer->Fuselage[4 * CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID2] = FUSELAGE_BLUE;
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        Buffer->Wing[2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = WING_RED;
        Buffer->Wing[2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = WING_RED;
        Buffer->Wing[3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = WING_GREEN;
        Buffer->Wing[3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = WING_GREEN;
        Buffer->Wing[4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = WING_BLUE;
        Buffer->Wing[4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = WING_BLUE;


        Buffer->Cannon[2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = CANNON_RED;
        Buffer->Cannon[2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = CANNON_RED;
        Buffer->Cannon[3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = CANNON_GREEN;
        Buffer->Cannon[3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = CANNON_GREEN;
        Buffer->Cannon[4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = CANNON_BLUE;
        Buffer->Cannon[4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = CANNON_BLUE;

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            Buffer->Engine[j][2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = ENGINE_RED;
            Buffer->Engine[j][2 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = ENGINE_RED;
            Buffer->Engine[j][3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = ENGINE_GREEN;
            Buffer->Engine[j][3 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = ENGINE_GREEN;
            Buffer->Engine[j][4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = ENGINE_BLUE;
            Buffer->Engine[j][4 * CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = ENGINE_BLUE;
        }
    }

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            Buffer->ThrustShort[j][2 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_SMALL_RED;
            Buffer->ThrustShort[j][2 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_SMALL_RED;
            Buffer->ThrustShort[j][3 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_SMALL_GREEN;
            Buffer->ThrustShort[j][3 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_SMALL_GREEN;
            Buffer->ThrustShort[j][4 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_SMALL_BLUE;
            Buffer->ThrustShort[j][4 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_SMALL_BLUE;

            Buffer->ThrustLong[j][2 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_BIG_RED;
            Buffer->ThrustLong[j][2 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_BIG_RED;
            Buffer->ThrustLong[j][3 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_BIG_GREEN;
            Buffer->ThrustLong[j][3 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_BIG_GREEN;
            Buffer->ThrustLong[j][4 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID1] = THRUST_BIG_BLUE;
            Buffer->ThrustLong[j][4 * CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * i + ID2] = THRUST_BIG_BLUE;
        }
    }

    // Bullet
#pragma unroll
    for (int i = 0; i < BULLET_COUNT_MAX; i++)
    {
        if (C->Bullet->Active[ID1])
            for (int j = 0; j < BULLET_VERT_COUNT + 1; j++)
            {
                Buffer->Bullet[i][2 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID1] = BULLET_RED;
                Buffer->Bullet[i][3 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID1] = BULLET_GREEN;
                Buffer->Bullet[i][4 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID1] = BULLET_BLUE;
            }

        if (C->Bullet->Active[ID2])
            for (int j = 0; j < BULLET_VERT_COUNT + 1; j++)
            {
                Buffer->Bullet[i][2 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID2] = BULLET_RED;
                Buffer->Bullet[i][3 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID2] = BULLET_GREEN;
                Buffer->Bullet[i][4 * CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * j + ID2] = BULLET_BLUE;
            }
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// OpenGL Processing
__device__ void GraphicsProcessing(CraftState* C, GraphicsObjectPointer* Buffer, int ID1, int ID2)
{
    // Fuslage center Position
    Buffer->Fuselage[ID1] = C->Position.X[ID1]; // X Position
    Buffer->Fuselage[ID2] = C->Position.X[ID2]; // X Position

    Buffer->Fuselage[CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + ID1] = C->Position.Y[ID1];   // Y Position
    Buffer->Fuselage[CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + ID2] = C->Position.Y[ID2];   // Y Position

    // Edge Vertex Positions
    for (int i = 1; i < FUSELAGE_VERT_COUNT + 1; i++)
    {
        Buffer->Fuselage[CRAFT_COUNT * 2 * i + ID1] = FUSELAGE_RADIUS * cos(2.f * PI / FUSELAGE_VERT_COUNT * i) + C->Position.X[ID1];   // X Position
        Buffer->Fuselage[CRAFT_COUNT * 2 * i + ID2] = FUSELAGE_RADIUS * cos(2.f * PI / FUSELAGE_VERT_COUNT * i) + C->Position.X[ID2];   // X Position

        Buffer->Fuselage[CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID1] = FUSELAGE_RADIUS * sin(2.f * PI / FUSELAGE_VERT_COUNT * i) + C->Position.Y[ID1]; // Y Position
        Buffer->Fuselage[CRAFT_COUNT * 2 * (FUSELAGE_VERT_COUNT + 1) + CRAFT_COUNT * 2 * i + ID2] = FUSELAGE_RADIUS * sin(2.f * PI / FUSELAGE_VERT_COUNT * i) + C->Position.Y[ID2]; // Y Position
    }

    // Wing Vertices
    float WingX[4] = { WINGSPAN / 2,    -WINGSPAN / 2,      -WINGSPAN / 2,       WINGSPAN / 2 };
    float WingY[4] = { WING_HEIGHT / 2,  WING_HEIGHT / 2,   -WING_HEIGHT / 2,   -WING_HEIGHT / 2 };

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float X_Out, Y_Out;

        Rotate(WingX[i], WingY[i], C->Angle[ID1], X_Out, Y_Out);                                                        // Angle
        Rotate(WingX[i], WingY[i], C->Angle[ID2]);

        Buffer->Wing[CRAFT_COUNT * 2 * i + ID1] = X_Out + C->Position.X[ID1];   // X Position
        Buffer->Wing[CRAFT_COUNT * 2 * i + ID2] = WingX[i] + C->Position.X[ID2];

        Buffer->Wing[CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = Y_Out + C->Position.Y[ID1]; // Y Position
        Buffer->Wing[CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = WingY[i] + C->Position.Y[ID2];
    }

    // Cannon Vertices
    float CannonX[4] = { CANNON_WIDTH / 2,  -CANNON_WIDTH / 2,  -CANNON_WIDTH / 2,  CANNON_WIDTH / 2 };
    float CannonY[4] = { CANNON_HEIGHT,      CANNON_HEIGHT,      0.f,               0.f };

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float X_Out, Y_Out;

        Rotate(CannonX[i], CannonY[i], C->Angle[ID1] + C->Cannon.Angle[ID1], X_Out, Y_Out);                         // Angle
        Rotate(CannonX[i], CannonY[i], C->Angle[ID2] + C->Cannon.Angle[ID2]);

        Buffer->Cannon[CRAFT_COUNT * 2 * i + ID1] = X_Out + C->Position.X[ID1]; // X Position
        Buffer->Cannon[CRAFT_COUNT * 2 * i + ID2] = CannonX[i] + C->Position.X[ID2];

        Buffer->Cannon[CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID1] = Y_Out + C->Position.Y[ID1];   // Y Position
        Buffer->Cannon[CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * i + ID2] = CannonY[i] + C->Position.Y[ID2];
    }

    // Engine vertices
    float EngineX[4] = { ENGINE_WIDTH / 2,  -ENGINE_WIDTH / 2,  -ENGINE_WIDTH / 2,   ENGINE_WIDTH / 2 };
    float EngineY[4] = { ENGINE_HEIGHT / 2,  ENGINE_HEIGHT / 2, -ENGINE_HEIGHT / 2, -ENGINE_HEIGHT / 2 };

    float EngineInitialX[4] = { ENGINE_0_DISTANCE, ENGINE_1_DISTANCE, ENGINE_2_DISTANCE, ENGINE_3_DISTANCE };
    float EngineInitialY = 0.f;

#pragma unroll
    for (int i = 0; i < 4; i++)     // 4 Engines
    {
#pragma unroll
        for (int j = 0; j < 4; j++) // 4 Vertices
        {
            float X1, Y1, X2, Y2;

            Rotate(EngineX[j], EngineY[j], C->Engine[i].Angle[ID1], X1, Y1);                            // Rotate to engine angle relative to craft
            Rotate(EngineX[j], EngineY[j], C->Engine[i].Angle[ID2], X2, Y2);

            X1 += EngineInitialX[i];                                                                        // Move to its relative position of the craft
            X2 += EngineInitialX[i];
            Y1 += EngineInitialY;
            Y2 += EngineInitialY;

            Rotate(X1, Y1, C->Angle[ID1]);                                                          // Rotate again to craft position
            Rotate(X2, Y2, C->Angle[ID2]);

            Buffer->Engine[i][CRAFT_COUNT * 2 * j + ID1] = X1 + C->Position.X[ID1];     // Move to position of the craft
            Buffer->Engine[i][CRAFT_COUNT * 2 * j + ID2] = X2 + C->Position.X[ID2];

            Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * j + ID1] = Y1 + C->Position.Y[ID1];
            Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * j + ID2] = Y2 + C->Position.Y[ID2];
        }

        /// Thrust Processing
        // First 2 thrust vertex position will be bottom of engine
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            Buffer->ThrustLong[i][CRAFT_COUNT * 2 * j + ID1] = Buffer->Engine[i][CRAFT_COUNT * 2 * (j + 2) + ID1];                      // X
            Buffer->ThrustLong[i][CRAFT_COUNT * 2 * j + ID2] = Buffer->Engine[i][CRAFT_COUNT * 2 * (j + 2) + ID2];
            Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * j + ID1] = Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * (j + 2) + ID1];  // Y
            Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * j + ID2] = Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * (j + 2) + ID2];

            Buffer->ThrustShort[i][CRAFT_COUNT * 2 * j + ID1] = Buffer->Engine[i][CRAFT_COUNT * 2 * (j + 2) + ID1];                     // X
            Buffer->ThrustShort[i][CRAFT_COUNT * 2 * j + ID2] = Buffer->Engine[i][CRAFT_COUNT * 2 * (j + 2) + ID2];
            Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * j + ID1] = Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * (j + 2) + ID1]; // Y
            Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 3 + CRAFT_COUNT * 2 * j + ID2] = Buffer->Engine[i][CRAFT_COUNT * 2 * 4 + CRAFT_COUNT * 2 * (j + 2) + ID2];
        }

        // Calculate 3rd vertex position for big and small thrust
        float X_LongLeft, Y_LongLeft;
        float X_LongRight, Y_LongRight;
        float X_ShortLeft, Y_ShortLeft;
        float X_ShortRight, Y_ShortRight;

        X_LongLeft = 0.f;
        X_LongRight = 0.f;
        X_ShortLeft = 0.f;
        X_ShortRight = 0.f;

        Y_LongLeft = -ENGINE_HEIGHT / 2 - THRUST_LENGTH_FULL * C->Engine[i].ThrustNormalized[ID1];
        Y_LongRight = -ENGINE_HEIGHT / 2 - THRUST_LENGTH_FULL * C->Engine[i].ThrustNormalized[ID2];
        Y_ShortLeft = -ENGINE_HEIGHT / 2 - THRUST_LENGTH_FULL_SHORT * C->Engine[i].ThrustNormalized[ID1];
        Y_ShortRight = -ENGINE_HEIGHT / 2 - THRUST_LENGTH_FULL_SHORT * C->Engine[i].ThrustNormalized[ID2];

        // Initial X-position of thrust vertices is 0
        Rotate(X_LongLeft, Y_LongLeft, C->Engine[i].Angle[ID1]);    // Rotate thrust to match engine angle
        Rotate(X_LongRight, Y_LongRight, C->Engine[i].Angle[ID2]);
        Rotate(X_ShortLeft, Y_ShortLeft, C->Engine[i].Angle[ID1]);
        Rotate(X_ShortRight, Y_ShortRight, C->Engine[i].Angle[ID2]);

        X_LongLeft += EngineInitialX[i];                            // Move to its initial position on the craft
        X_LongRight += EngineInitialX[i];
        Y_LongLeft += EngineInitialY;
        Y_LongRight += EngineInitialY;

        X_ShortLeft += EngineInitialX[i];
        X_ShortRight += EngineInitialX[i];
        Y_ShortLeft += EngineInitialY;
        Y_ShortRight += EngineInitialY;

        Rotate(X_LongLeft, Y_LongLeft, C->Angle[ID1]);          // Rotate again to craft position
        Rotate(X_LongRight, Y_LongRight, C->Angle[ID2]);
        Rotate(X_ShortLeft, Y_ShortLeft, C->Angle[ID1]);
        Rotate(X_ShortRight, Y_ShortRight, C->Angle[ID2]);

        Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 2 + ID1] = X_LongLeft + C->Position.X[ID1];  // Move to position of the craft
        Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 2 + ID2] = X_LongRight + C->Position.X[ID2];
        Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 2 + CRAFT_COUNT * 2 * 3 + ID1] = Y_LongLeft + C->Position.Y[ID1];
        Buffer->ThrustLong[i][CRAFT_COUNT * 2 * 2 + CRAFT_COUNT * 2 * 3 + ID2] = Y_LongRight + C->Position.Y[ID2];

        Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 2 + ID1] = X_ShortLeft + C->Position.X[ID1]; // Move to position of the craft
        Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 2 + ID2] = X_ShortRight + C->Position.X[ID2];
        Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 2 + CRAFT_COUNT * 2 * 3 + ID1] = Y_ShortLeft + C->Position.Y[ID1];
        Buffer->ThrustShort[i][CRAFT_COUNT * 2 * 2 + CRAFT_COUNT * 2 * 3 + ID2] = Y_ShortRight + C->Position.Y[ID2];

        for (int j = 0; j < BULLET_COUNT_MAX; j++)
        {
            if (C->Bullet[j].Active[ID1])
            {
                // Center Vertex is beginning of vertex buffer
                Buffer->Bullet[j][ID1] = C->Bullet[j].Position.X[ID1];  // X Position
                Buffer->Bullet[j][CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + ID1] = C->Bullet[j].Position.Y[ID1];  // Y Position

                for (int k = 1; k < BULLET_VERT_COUNT + 1; k++)
                {
                    Buffer->Bullet[j][CRAFT_COUNT * 2 * k + ID1] = BULLET_RADIUS * cos(2.f * PI / BULLET_VERT_COUNT * k) + C->Bullet[j].Position.X[ID1];    // X Position
                    Buffer->Bullet[j][CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * k + ID1] = BULLET_RADIUS * sin(2.f * PI / BULLET_VERT_COUNT * k) + C->Bullet[j].Position.Y[ID1];    // Y Position
                }
            }
        }

        for (int j = 0; j < BULLET_COUNT_MAX; j++)
        {
            if (C->Bullet[j].Active[ID2])
            {
                // Center Vertex is beginning of vertex buffer
                Buffer->Bullet[j][ID2] = C->Bullet[j].Position.X[ID2];  // X Position
                Buffer->Bullet[j][CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + ID2] = C->Bullet[j].Position.Y[ID2];  // Y Position

                for (int k = 1; k < BULLET_VERT_COUNT + 1; k++)
                {
                    Buffer->Bullet[j][CRAFT_COUNT * 2 * k + ID2] = BULLET_RADIUS * cos(2.f * PI / BULLET_VERT_COUNT * k) + C->Bullet[j].Position.X[ID2];    // X Position
                    Buffer->Bullet[j][CRAFT_COUNT * 2 * (BULLET_VERT_COUNT + 1) + CRAFT_COUNT * 2 * k + ID2] = BULLET_RADIUS * sin(2.f * PI / BULLET_VERT_COUNT * k) + C->Bullet[j].Position.Y[ID2];    // Y Position
                }
            }
        }
    }   // End Engine1 OpenGL Processing
}   // End OpenGL Processing
//} // End GPGPU namespace

#pragma once

// CUDA
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

// Project Headers
#include "GPGPU/GPSetup.h"
#include "GPGPU/State.h"

//namespace GPGPU
//{
__device__ void Rotate(float& X, float& Y, float Theta);
__device__ void Rotate(float X_In, float Y_In, float Theta, float& X_Out, float& Y_Out);
__device__ void ShowBullet(GraphicsObjectPointer* Buffer, int ID, int BulletNumber);
__device__ void ConcealBullet(GraphicsObjectPointer* Buffer, int ID, int BulletNumber);
__device__ void ConcealVertices(GraphicsObjectPointer* Buffer, int idxLeft, int idxRight);
__device__ void ShowVertices(CraftState* C, GraphicsObjectPointer* Buffer, int ID1, int ID2);

/////////////////////////////////////////////////////////////////////////////////////
// OpenGL Processing
__device__ void GraphicsProcessing(CraftState* C, GraphicsObjectPointer* Buffer, int ID1, int ID2);

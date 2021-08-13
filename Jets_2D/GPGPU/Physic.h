#pragma once

// CUDA
#include <cuda_runtime.h>

// Project Headers
#include "Jets_2D/GPGPU/State.h"

__device__ void Physic(MatchState* Match, CraftState* CS, config* Config, bool IsOpponent);
__device__ void CollisionDetect(CraftState* C, int idx1, int idx2);
__device__ void BulletMechanics(GraphicsObjectPointer* Buffer, CraftState* CS, int ID1, int ID2, config* Config);
__device__ void ShootBullet(CraftState* CS, int ID, GraphicsObjectPointer* Buffer);

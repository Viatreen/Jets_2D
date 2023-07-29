#pragma once

// Standard Library
#include <chrono>

extern std::chrono::steady_clock::time_point Timer;
extern bool exit_round;

void Round();

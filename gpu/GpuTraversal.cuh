#pragma once

#include "GpuDeque.cuh"
#include "GpuRainbow.cuh"
#include "GpuBitVector.cuh"

struct GpuPartSol {
  GpuBitVector asg_;
  VciGpu nUnsat_;
};

struct GpuTraversal {
  // Store it in pinned memory to save GPU memory - it's not often pushed or popped.
  GpuDeque<GpuPartSol> dfsAsg_;
  // Occupies most of the device memory
  GpuRainbow rainbow_;
};

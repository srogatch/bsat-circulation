#pragma once

#include "GpuUtils.cuh"

template<bool taHashing, bool taAtomic> struct GpuBitVector {
  __uint128_t hash_ = 0;
  uint32_t *bits_ = nullptr;
  VciGpu nBits_;

  __host__ __device__ VciGpu QwordCount() const {
    return DivUp(nBits_, 32);
  }

  
};

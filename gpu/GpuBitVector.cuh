#pragma once

#include "GpuUtils.cuh"

template<bool taHashing, bool taAtomic> struct GpuBitVector {
  __uint128_t hash_ = 0;
  uint32_t *bits_ = nullptr;
  VciGpu nBits_ = 0;

  __host__ __device__ VciGpu QwordCount() const {
    return DivUp(nBits_, 32);
  }

  __host__ __device__ GpuBitVector() = default;

  __device__ void Rehash() {
    hash_ = 0;
    for(int64_t i=0; i<nBits_; i++) {
      if((*this)[i]) {
        hash_ ^= gpHashSeries[i];
      }
    }
  }

  
};

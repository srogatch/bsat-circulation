#pragma once

#include "GpuUtils.cuh"

struct GpuBitVector {
  __uint128_t hash_ = 0;
  uint32_t *bits_ = nullptr;
  VciGpu nBits_ = 0;

  __host__ __device__ VciGpu QwordCount() const {
    return DivUp(nBits_, 32);
  }

  GpuBitVector() = default;

  __device__ void Rehash() {
    hash_ = 0;
    for(VciGpu i=0; i<nBits_; i++) {
      if((*this)[i]) {
        hash_ ^= gpHashSeries[i];
      }
    }
  }

  // Note the return logic is different from CPU BitVector
  __host__ __device__ int8_t operator[](const VciGpu index) const {
    return (bits_[index/32] & (1u<<(index&31))) ? 1 : -1;
  }

  __device__ void Flip(const VciGpu index)  {
    hash_ ^= gpHashSeries[index];
    bits_[index/32] ^= (1u<<(index&31));
  }
};

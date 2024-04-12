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
    for(VciGpu i=0; i<nBits_; i++) {
      if((*this)[i]) {
        hash_ ^= gpHashSeries[i];
      }
    }
  }

  // Note the return logic is different from CPU BitVector
  int8_t operator[](const VciGpu index) const {
    return (bits_[index/32] & (1u<<(index&31))) ? 1 : -1;
  }

  void Flip(const VciGpu index)  {
    hash_ ^= gpHashSeries[i];
    bits_[index/32] ^= (1u<<(index&31));
  }
};

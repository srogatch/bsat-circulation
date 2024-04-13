#pragma once

#include "GpuUtils.cuh"

struct GpuBitVector {
  static constexpr const uint8_t cfOwned = 1;
  __uint128_t hash_ = 0;
  uint32_t *bits_ = nullptr;
  VciGpu nBits_ = 0;
  uint8_t flags_ = 0;

  __host__ __device__ VciGpu DwordCount() const {
    return DivUp(nBits_, 32);
  }

  GpuBitVector() = default;

  __host__ __device__ GpuBitVector(GpuBitVector&& src)
  : hash_(src.hash_), bits_(src.bits_), nBits_(src.nBits_), flags_(src.flags_)
  {
    src.hash_ = 0;
    src.bits_ = nullptr;
    src.nBits_ = 0;
    src.flags_ = 0;
  }

  __host__ __device__ GpuBitVector& operator=(GpuBitVector&& src)
  {
    if(this != &src) {
      if(flags_ & cfOwned) {
        free(bits_);
      }
      hash_ = src.hash_;
      bits_ = src.bits_;
      nBits_ = src.nBits_;
      flags_ = src.flags_;
      src.hash_ = 0;
      src.bits_ = nullptr;
      src.nBits_ = 0;
      src.flags_ = 0;
    }
    return *this;
  }

  __host__ __device__ GpuBitVector(const GpuBitVector& src)
  : hash_(src.hash_), nBits_(src.nBits_), flags_(src.flags_ | cfOwned)
  {
    bits_ = reinterpret_cast<uint32_t*>(malloc(DwordCount() * sizeof(uint32_t)));
    // TODO: vectorize
    for(VciGpu i=0; i<nBits_; i++) {
      bits_[i] = src.bits_[i];
    }
  }

  __host__ __device__ ~GpuBitVector() {
    if(flags_ & cfOwned) {
      free(bits_);
    }
    bits_ = nullptr;
  }

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

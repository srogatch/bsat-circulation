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

  __host__ __device__ VciGpu VectCount() const {
    return DivUp(DwordCount(), 4);
  }

  GpuBitVector() = default;

  __host__ __device__ explicit GpuBitVector(
    const VciGpu nBits, const bool setZer0) : nBits_(nBits), flags_(cfOwned)
  {
    const VciGpu nVects = VectCount();
    bits_ = reinterpret_cast<uint32_t*>(malloc( nVects * sizeof(__uint128_t) ));
    if(setZer0) {
      for(VciGpu i=0; i<nVects; i++) {
        reinterpret_cast<__uint128_t*>(bits_)[i] = 0;
      }
    } else {
      // Denote unitialized
      hash_ = __uint128_t(-1);
    }
  }

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
    const VciGpu nVects = VectCount();
    bits_ = reinterpret_cast<uint32_t*>(malloc( nVects * sizeof(__uint128_t) ));
    for(VciGpu i=0; i<nVects; i++) {
      reinterpret_cast<__uint128_t*>(bits_)[i] = reinterpret_cast<const __uint128_t*>(src.bits_)[i];
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

  template<bool doHash = true> __device__ void Flip(const VciGpu index) {
    if constexpr(doHash) {
      hash_ ^= gpHashSeries[index];
    }
    bits_[index/32] ^= (1u<<(index&31));
  }
};

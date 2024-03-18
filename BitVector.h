#pragma once

#include "Utils.h"

#include <memory>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <immintrin.h>

typedef unsigned __int128 uint128;

template<typename T, typename U> constexpr T DivUp(const T a, const U b) {
  return (a + T(b) - 1) / T(b);
}

struct BitVector {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(uint64_t); // one cache line at a time
  static constexpr const uint128 cHashBase =
    (uint128(244)  * uint128(1000*1000*1000) * uint128(1000*1000*1000) + uint128(903443422803031898ULL)) * uint128(1000*1000*1000) * uint128(1000*1000*1000)
    + uint128(471395581046679967ULL);
  static std::unique_ptr<uint128[]> hashSeries_;
  std::unique_ptr<uint64_t[]> bits_;
  int64_t nQwords_ = 0;
  int64_t nBits_ = 0;
  uint128 hash_ = 0;

  static void CalcHashSeries(const int64_t nVars) {
    hashSeries_.reset(new uint128[nVars+1]);
    hashSeries_[0] = 1;
    for(int64_t i=1; i<=nVars; i++) {
      hashSeries_[i] = hashSeries_[i-1] * cHashBase;
    }
  }

  void Rehash() {
    hash_ = 0;
    for(int64_t i=0; i<nBits_; i++) {
      if((*this)[i]) {
        hash_ ^= hashSeries_.get()[i];
      }
    }
  }

  BitVector() {}

  explicit BitVector(const int64_t nBits) {
    nBits_ = nBits;
    nQwords_ = DivUp(nBits, 64);
    bits_.reset(new uint64_t[nQwords_]);

    //memset(bits_.get(), 0, sizeof(uint64_t) * nQwords_);
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=0; i<nQwords_; i++) {
      bits_.get()[i] = 0;
    }
    hash_ = 0;
  }

  BitVector(const BitVector& fellow) {
    nBits_ = fellow.nBits_;
    nQwords_ = fellow.nQwords_;
    bits_.reset(new uint64_t[nQwords_]);
    // memcpy(bits_.get(), fellow.bits_.get(), sizeof(uint64_t) * nQwords_);
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=0; i<nQwords_; i++) {
      bits_.get()[i] = fellow.bits_.get()[i];
    }
    hash_ = fellow.hash_;
  }
  BitVector& operator=(const BitVector& fellow) {
    if(this != &fellow) {
      if(nBits_ != fellow.nBits_) {
        nBits_ = fellow.nBits_;
        nQwords_ = fellow.nQwords_;
        bits_.reset(new uint64_t[nQwords_]);
      }
      // memcpy(bits_.get(), fellow.bits_.get(), sizeof(uint64_t) * nQwords_);
      #pragma omp parallel for schedule(static, cParChunkSize)
      for(int64_t i=0; i<nQwords_; i++) {
        bits_.get()[i] = fellow.bits_.get()[i];
      }
      hash_ = fellow.hash_;
    }
    return *this;
  }
  BitVector& operator=(BitVector&& src) {
    if(this != &src) {
      nBits_ = src.nBits_;
      nQwords_ = src.nQwords_;
      bits_ = std::move(src.bits_);
      hash_ = src.hash_;
      src.nBits_ = 0;
      src.nQwords_ = 0;
      src.hash_ = 0;
    }
    return *this;
  }

  bool operator[](const int64_t index) const {
    return bits_[index/64] & (1ULL<<(index&63));
  }

  bool operator==(const BitVector& fellow) const {
    return hash_ == fellow.hash_;

    // if(nBits_ != fellow.nBits_) {
    //   return false;
    // }
    // // return memcmp(bits_.get(), fellow.bits_.get(), sizeof(uint64_t) * nQwords_) == 0;
    // std::atomic<bool> equals{true};
    // #pragma omp parallel for schedule(static, cParChunkSize)
    // for(int64_t i=0; i<nQwords_; i++) {
    //   if(bits_.get()[i] != fellow.bits_.get()[i]) {
    //     equals.store(false, std::memory_order_relaxed);
    //     #pragma omp cancel for
    //   }
    //   #pragma omp cancellation point for
    // }
    // return equals;
  }

  void Flip(const int64_t index) {
    bits_[index/64] ^= (1ULL<<(index&63));
    hash_ ^= hashSeries_.get()[index];
  }

  void Randomize() {
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=0; i<nQwords_; i++) {
      while(!_rdrand64_step(reinterpret_cast<unsigned long long*>(bits_.get()+i)));
    }
    // Ensure the dummy bit for the formula is always false
    if(bits_[0]) {
      Flip(0);
    }
    Rehash();
  }

  void SetTrue() {
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=0; i<nQwords_; i++) {
      bits_.get()[i] = -1LL;
    }
    // Ensure the dummy bit for the formula is always false
    Flip(0);
    Rehash();
  }
};

inline uint64_t hash64(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

namespace std {

template<> struct hash<BitVector> {
  inline std::size_t operator()(const BitVector &bv) const {
    std::size_t ans = 0;
    uint64_t mul = 7;
    for(int64_t i=0; i<bv.nQwords_; i++) {
      ans ^= mul * hash64(bv.bits_[i]);
      mul *= 18446744073709551557ULL;
    }
    return ans;
  }
};

} // namespace std

#pragma once

#include "Utils.h"

#include <memory>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <atomic>

struct BitVector {
  // One standard page of RAM at once
  static constexpr const uint32_t cParChunkSize = kRamPageBytes/sizeof(uint64_t);
  static std::unique_ptr<uint128[]> hashSeries_;
  std::unique_ptr<uint64_t[]> bits_;
  int64_t nQwords_ = 0;
  int64_t nBits_ = 0;
  uint128 hash_ = 0;

  static void CalcHashSeries(const int64_t nVars) {
    hashSeries_.reset(new uint128[nVars+1]);
    hashSeries_[0] = 1;
    for(int64_t i=1; i<=nVars; i++) {
      hashSeries_[i] = hashSeries_[i-1] * kHashBase;
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

  BitVector() = default;

  explicit BitVector(const int64_t nBits) {
    nBits_ = nBits;
    nQwords_ = DivUp(nBits, 64);
    bits_.reset(new uint64_t[nQwords_]);

    memset(bits_.get(), 0, nQwords_ * sizeof(bits_[0]));
    hash_ = 0;
  }

  BitVector(const BitVector& fellow) {
    nBits_ = fellow.nBits_;
    nQwords_ = fellow.nQwords_;
    bits_.reset(new uint64_t[nQwords_]);
    memcpy(bits_.get(), fellow.bits_.get(), nQwords_ * sizeof(bits_[0]));
    hash_ = fellow.hash_;
  }
  BitVector& operator=(const BitVector& fellow) {
    if(this != &fellow) {
      if(nBits_ != fellow.nBits_) {
        nBits_ = fellow.nBits_;
        nQwords_ = fellow.nQwords_;
        bits_.reset(new uint64_t[nQwords_]);
      }
      memcpy(bits_.get(), fellow.bits_.get(), nQwords_ * sizeof(*bits_.get()));
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
    return (bits_[index/64] & (1ULL<<(index&63))) != 0;
  }

  bool operator==(const BitVector& fellow) const {
    return hash_ == fellow.hash_;
  }

  void Flip(const int64_t index) {
    reinterpret_cast<std::atomic<uint64_t>*>(&bits_[index/64])->fetch_xor( 1ULL<<(index&63) );
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[0].fetch_xor(reinterpret_cast<std::atomic<uint64_t>*>(&hashSeries_.get()[index])[0]);
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[1].fetch_xor(reinterpret_cast<std::atomic<uint64_t>*>(&hashSeries_.get()[index])[1]);
  }

  void NohashSet(const int64_t index) {
    reinterpret_cast<std::atomic<uint64_t>*>(&bits_[index/64])->fetch_or(1ULL<<(index&63));
  }

  void Randomize() {
    std::mt19937_64 rng = GetSeededRandom();
    for(int64_t i=0; i<nQwords_; i++) {
      bits_[i]  = rng();
    }
    // Ensure the dummy bit for the formula is always false
    if(bits_[0]) {
      Flip(0);
    }
    Rehash();
  }

  void SetTrue() {
    memset(bits_.get(), -1, nQwords_ * sizeof(bits_[0]));
    // Ensure the dummy bit for the formula is always false
    Flip(0);
    Rehash();
  }
};

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

template<> struct hash<uint128> {
  inline std::size_t operator()(const uint128 x) const {
    return (x >> 64) * 1949 ^ (x & uint64_t(-1LL));
  }
};

} // namespace std

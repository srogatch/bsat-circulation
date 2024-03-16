#pragma once

#include <memory>
#include <cstdint>
#include <cstring>

template<typename T, typename U> constexpr T DivUp(const T a, const U b) {
  return (a + T(b) - 1) / T(b);
}

struct BitVector {
  std::unique_ptr<uint64_t[]> bits_;
  int64_t nQwords_ = 0;
  int64_t nBits_ = 0;

  BitVector() {}

  explicit BitVector(const int64_t nBits) {
    nBits_ = nBits;
    nQwords_ = DivUp(nBits, 64);
    bits_.reset(new uint64_t[nQwords_]);
    memset(bits_.get(), 0, sizeof(uint64_t) * nQwords_);
  }

  BitVector(const BitVector& fellow) {
    nBits_ = fellow.nBits_;
    nQwords_ = fellow.nQwords_;
    bits_.reset(new uint64_t[nQwords_]);
    memcpy(bits_.get(), fellow.bits_.get(), sizeof(uint64_t) * nQwords_);
  }
  BitVector& operator=(BitVector&& src) = default;

  bool operator[](const int64_t index) const {
    return bits_[index/64] & (1ULL<<(index&63));
  }

  bool operator==(const BitVector& fellow) const {
    if(nBits_ != fellow.nBits_) {
      return false;
    }
    return memcmp(bits_.get(), fellow.bits_.get(), sizeof(uint64_t) * nQwords_) == 0;
  }

  void Flip(const int64_t index) {
    bits_[index/64] ^= (1ULL<<(index&63));
  }
};

namespace std {

template<> struct hash<BitVector> {
  inline std::size_t operator()(const BitVector &bv) const {
    std::size_t ans = 0;
    uint64_t mul = 7;
    for(int64_t i=0; i<bv.nQwords_; i++) {
      ans ^= mul * bv.bits_[i];
      mul *= 18446744073709551557ULL;
    }
    return ans;
  }
};

} // namespace std

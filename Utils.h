#pragma once

#include <omp.h>
#include <cstdint>
#include <random>
#include <thread>
#include <algorithm>
#include <immintrin.h>

constexpr const uint32_t kCacheLineSize = 64;
constexpr const uint32_t kRamPageBytes = 4096;

static const uint32_t nSysCpus = std::thread::hardware_concurrency();

typedef unsigned __int128 uint128;
typedef int64_t VCIndex; // vertex or clause index

constexpr const uint128 kHashBase =
    (uint128(244)  * uint128(1000*1000*1000) * uint128(1000*1000*1000) + uint128(903443422803031898ULL)) * uint128(1000*1000*1000) * uint128(1000*1000*1000)
    + uint128(471395581046679967ULL);

template<typename T, typename U> constexpr T DivUp(const T a, const U b) {
  return (a + T(b) - 1) / T(b);
}

// Define a primary template for is_specialization_of, which defaults to false.
template<typename Test, template<typename...> class Ref>
struct is_specialization_of : std::false_type {};

// Specialize is_specialization_of for cases where the first parameter is a specialization of the template in the second parameter.
template<template<typename...> class Ref, typename... Args>
struct is_specialization_of<Ref<Args...>, Ref> : std::true_type {};

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

namespace detail {

template <typename F>
struct FinalAction {
  FinalAction(F&& f) : clean_{std::move(f)} {}
  ~FinalAction() {
    if (enabled_) clean_();
  }
  void Disable() { enabled_ = false; };

 private:
  F clean_;
  bool enabled_{true};
};

}  // namespace detail

template <typename F>
detail::FinalAction<F> Finally(F&& f) {
  return detail::FinalAction<F>(std::move(f));
}

struct SpinLock {
  std::atomic_flag *pSync_ = nullptr;

  SpinLock() = default;

  SpinLock(std::atomic_flag& sync) : pSync_(&sync) {
    while(pSync_->test_and_set(std::memory_order_acq_rel)) {
      _mm_pause();
    }
  }

  ~SpinLock() {
    if(pSync_ != nullptr) {
      pSync_->clear(std::memory_order_release);
    }
  }
};

inline std::mt19937_64 GetSeededRandom() {
  unsigned long long seed;
  while(!_rdrand64_step(&seed));
  std::mt19937_64 rng(seed);
  return rng;
}

template <typename T>
void ParallelShuffle(T* data, const size_t count) {
  if(count <= kCacheLineSize) { // Don't parallelize
    std::mt19937_64 rng = GetSeededRandom();
    std::shuffle(data, data+count, rng);
    return;
  }

  const uint32_t nThreads = std::max<int64_t>(1, std::min<int64_t>(omp_get_max_threads(), count/3));
  std::atomic_flag* syncs = static_cast<std::atomic_flag*>(malloc(count * sizeof(std::atomic_flag)));
  auto clean_syncs = Finally([&]() { free(syncs); });
  #pragma omp parallel for schedule(static, kRamPageBytes / sizeof(std::atomic_flag))
  for (size_t i = 0; i < count; i++) {
    new (syncs + i) std::atomic_flag ATOMIC_FLAG_INIT;
  }

  const size_t nPerThread = (count + nThreads - 1) / nThreads;
  // The number of threads here is important and must not default to whatever else
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < nThreads; i++) {
    std::mt19937_64 rng = GetSeededRandom();
    std::uniform_int_distribution<size_t> dist(0, count - 1);
    const size_t iFirst = nPerThread * i;
    const size_t iLimit = std::min(nPerThread + iFirst, count);
    if (iLimit <= iFirst) {
      continue;
    }
    for (size_t j = iFirst; j < iLimit; j++) {
      const size_t fellow = dist(rng);
      if (fellow == j) {
        continue;
      }
      const size_t sync1 = std::min(j, fellow);
      const size_t sync2 = j ^ fellow ^ sync1;

      SpinLock lock1(syncs[sync1]);
      SpinLock lock2(syncs[sync2]);
      std::swap(data[sync1], data[sync2]);
      syncs[sync2].clear(std::memory_order_release);
      syncs[sync1].clear(std::memory_order_release);
    }
  }
}

namespace std {

template<> struct hash<pair<uint128, uint128>> {
  uint128 operator()(const pair<uint128, uint128>& v) const {
    return v.first * 1949 + v.second * 2011;
  }
};

} // namespace std

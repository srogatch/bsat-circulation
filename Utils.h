#pragma once

#include <omp.h>
#include <cstdint>
#include <random>
#include <thread>
#include <immintrin.h>

constexpr const uint32_t kCacheLineSize = 64;
constexpr const uint32_t kRamPageBytes = 4096;

typedef unsigned __int128 uint128;

constexpr const uint128 kHashBase =
    (uint128(244)  * uint128(1000*1000*1000) * uint128(1000*1000*1000) + uint128(903443422803031898ULL)) * uint128(1000*1000*1000) * uint128(1000*1000*1000)
    + uint128(471395581046679967ULL);

template<typename T, typename U> constexpr T DivUp(const T a, const U b) {
  return (a + T(b) - 1) / T(b);
}

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

template <typename T>
void ParallelShuffle(T* data, const size_t count) {
  const uint32_t nThreads = omp_get_max_threads();

  std::atomic_flag* syncs = static_cast<std::atomic_flag*>(malloc(count * sizeof(std::atomic_flag)));
  auto clean_syncs = Finally([&]() { free(syncs); });
  #pragma omp parallel for schedule(static, kCacheLineSize / sizeof(std::atomic_flag))
  for (size_t i = 0; i < count; i++) {
    new (syncs + i) std::atomic_flag ATOMIC_FLAG_INIT;
  }

  const size_t nPerThread = (count + nThreads - 1) / nThreads;
  // The number of threads here is important and must not default to whatever else
  #pragma omp parallel for num_threads(nThreads)
  for (size_t i = 0; i < nThreads; i++) {
    unsigned long long seed;
    while(!_rdrand64_step(&seed));
    std::mt19937_64 rng(seed);
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
      const size_t sync2 = std::max(j, fellow);
      while (syncs[sync1].test_and_set(std::memory_order_acq_rel)) {
        std::this_thread::yield();
      }
      while (syncs[sync2].test_and_set(std::memory_order_acq_rel)) {
        std::this_thread::yield();
      }
      std::swap(data[sync1], data[sync2]);
      syncs[sync2].clear(std::memory_order_release);
      syncs[sync1].clear(std::memory_order_release);
    }
  }
}

namespace std {

template<> struct hash<pair<uint128, uint128>> {
  bool operator()(const pair<uint128, uint128>& v) const {
    return v.first * 1949 + v.second * 2011;
  }
};

} // namespace std

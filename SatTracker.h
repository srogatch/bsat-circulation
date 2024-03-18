#pragma once

#include "BitVector.h"

#include <cstdint>
#include <memory>
#include <atomic>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(TCounter);
  int64_t nVars_, nClauses_;
  std::unique_ptr<TCounter[]> nSat_;
  std::atomic<int64_t> totSat_ = -1;

  explicit SatTracker(const int64_t nVars, const int64_t nClauses)
  : nVars_(nVars), nClauses_(nClauses)
  {
    nSat.reset(new TCounter[nClauses+1]);
  }

  void Populate(const BitVector& varAsg) {
    totSat = 0;
    nSat_[0] = 1;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=nClauses; i++) {

    }
  }
};

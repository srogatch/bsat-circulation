#pragma once

#include "RangeVector.h"
#include "Utils.h"

struct Linkage {
  using TBackend = RangeVector<RangeVector<std::vector<VCIndex>, int8_t>, VCIndex>;
  TBackend sources_;

  Linkage() = default;

  explicit Linkage(const VCIndex maxItem) {
    sources_ = TBackend(-maxItem, maxItem);
    #pragma omp parallel for
    for(VCIndex i=-maxItem; i<=maxItem; i++) {
      if(i == 0) {
        continue;
      }
      sources_[i] = RangeVector<std::vector<VCIndex>, int8_t>(-1, 1);
    }
  }

  // Thread-safe relative to other Add() calls.
  void Add(const VCIndex from, const VCIndex to)
  {
    const int sgnTo = Signum(to);
    assert(sgnTo != 0);
    // Place the forward arc from |from| to |to|, and the backlink from |-from| to |-to|
    for(int8_t sgnBoth=-1; sgnBoth<=1; sgnBoth+=2)
    {
      auto lock1 = sources_.With(from * sgnBoth);
      RangeVector<std::vector<VCIndex>, int8_t>& source = sources_[from * sgnBoth];
      auto lock2 = source.With(sgnTo * sgnBoth);
      source[sgnTo * sgnBoth].emplace_back(to *sgnBoth);
    }
  }
};

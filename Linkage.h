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

  void Sort() {
    #pragma omp parallel for schedule(guided, kCacheLineSize)
    for(VCIndex i=sources_.minIndex_; i<=sources_.maxIndex_; i++) {
      RangeVector<std::vector<VCIndex>, int8_t>& source = sources_[i];
      for(int8_t sgn=-1; sgn<=1; sgn+=2) {
        std::sort(source[i].begin(), source[i].end());
        VCIndex newSize = 0;
        // Remove duplicate arcs
        for(VCIndex j=0; j<int64_t(source[i].size()); j++) {
          if(j == 0 || source[i][j-1] != source[i][j]) {
            source[i][newSize] = source[i][j];
            newSize++;
          }
        }
        source[i].resize(newSize);
      }
    }
  }

  // Doesn't need to be thread safe, right? If it does, we can add a concurrency template parameter like it's done for SatTracker::FlipVar
  VCIndex ArcCount(const VCIndex from) const {
    // auto lock = sources_.With(from);
    const RangeVector<std::vector<VCIndex>, int8_t>& source = sources_[from];
    // Sum positive and negative entries for this item
    const VCIndex ans = source[1].size() + source[-1].size();
    return ans;
  }

  // Doesn't need to be thread safe, right? If it does, we can add a concurrency template parameter like it's done for SatTracker::FlipVar
  VCIndex ArcCount(const VCIndex from, const int8_t sgn) const {
    // auto lock1 = sources_.With(from);
    // auto lock2 = sources_[from].With(sgn);
    return sources_[from][sgn].size();
  }

  VCIndex GetTarget(const VCIndex from, const int8_t sgn, const VCIndex at) const {
    return sources_[from][sgn][at];
  }

  bool HasArc(const VCIndex from, const VCIndex to) const {
    const std::vector<VCIndex>& arcs = sources_[from][Signum(to)];
    VCIndex l=0, r=VCIndex(arcs.size());
    // l: inclusive bound, r: exclusive bound
    while(l < r) {
      const VCIndex m = (l+r) / 2;
      const VCIndex cur = arcs[m];
      if(cur == to) {
        return true;
      }
      if(cur < to) {
        l = m+1;
      } else {
        // cur > to
        r = m;
      }
    }
    return false;
  }
};

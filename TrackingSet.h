#pragma once

#include "Utils.h"

#include <unordered_set>
#include <cstdint>

struct TrackingSet {
  std::unordered_set<int64_t> set_;
  uint128 hash_ = 0;

  void Add(const int64_t item) {
    auto it = set_.find(item);
    if(it == set_.end()) {
      set_.emplace(item);
      hash_ ^= item * kHashBase;
    }
  }

  void Remove(const int64_t item) {
    auto it = set_.find(item);
    if(it != set_.end()) {
      set_.erase(it);
      hash_ ^= item * kHashBase;
    }
  }

  void Clear() {
    set_.clear();
    hash_ = 0;
  }

  bool operator==(const TrackingSet& fellow) const {
    return hash_ == fellow.hash_ && set_ == fellow.set_;
  }

  bool operator!=(const TrackingSet& fellow) const {
    return hash_ != fellow.hash_ || set_ != fellow.set_;
  }

  TrackingSet operator-(const TrackingSet& fellow) const {
    TrackingSet ans;
    for(const int64_t iClause : set_) {
      if(fellow.set_.find(iClause) == fellow.set_.end()) {
        ans.Add(iClause);
      }
    }
    return ans;
  }

  TrackingSet operator+(const TrackingSet& fellow) const {
    if(set_.size() < fellow.set_.size()) {
      TrackingSet ans = fellow;
      for(const int64_t iClause : set_) {
        ans.Add(iClause);
      }
      return ans;
    }
    else {
      TrackingSet ans = *this;
      for(const int64_t iClause : fellow.set_) {
        ans.Add(iClause);
      }
      return ans;
    }
  }
};

namespace std {
  template<> struct hash<TrackingSet> {
    bool operator()(const TrackingSet& ts) const {
      return ts.hash_;
    }
  };
} // namespace std

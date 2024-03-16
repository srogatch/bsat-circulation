#pragma once

#include <unordered_set>
#include <cstdint>

struct TrackingSet {
  std::unordered_set<int64_t> set_;
  std::size_t hash_ = 0;

  void Add(const int64_t item) {
    auto it = set_.find(item);
    if(it == set_.end()) {
      set_.emplace(item);
      hash_ ^= std::hash<int64_t>()(item) * 18446744073709551557ULL;
    }
  }

  void Remove(const int64_t item) {
    auto it = set_.find(item);
    if(it != set_.end()) {
      set_.erase(it);
      hash_ ^= std::hash<int64_t>()(item) * 18446744073709551557ULL;
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
    return !(*this == fellow);
  }
};

namespace std {
  template<> struct hash<TrackingSet> {
    bool operator()(const TrackingSet& ts) const {
      return ts.hash_;
    }
  };
} // namespace std
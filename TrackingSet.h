#pragma once

#include "Utils.h"

#include <unordered_set>
#include <cstdint>
#include <mutex>

struct Bucket {
  std::mutex sync_;
  std::unordered_set<int64_t> set_;
  int64_t prefixSum_;
};

struct TrackingSet {
  static constexpr const int64_t kSyncContention = 37;
  static const uint32_t nCpus_;

  std::unique_ptr<Bucket[]> buckets_ = std::make_unique<Bucket[]>(kSyncContention * nCpus_);
  mutable std::mutex sync_;
  uint128 hash_ = 0;
  std::atomic<int64_t> size_ = 0;

  TrackingSet() = default;

  void UpdateHash(const int64_t item) {
    const uint128 t = item * kHashBase;
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[0].fetch_xor(t & (-1LL));
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[1].fetch_xor(t >> 64);
  }

  void CopyFrom(const TrackingSet& src) {
    hash_ = src.hash_;
    size_ = src.size_.load();
    #pragma omp parallel for
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      buckets_[i].set_ = src.buckets_[i].set_;
    }
  }

  TrackingSet(const TrackingSet& src) {
    CopyFrom(src);
  }

  TrackingSet& operator=(const TrackingSet& src) {
    if(this != &src) {
      CopyFrom(src);
    }
    return *this;
  }

  Bucket& GetBucket(const int64_t item) {
    return buckets_[(item*kHashBase) % (kSyncContention * nCpus_)];
  }

  const Bucket& GetBucket(const int64_t item) const {
    return buckets_[(item*kHashBase) % (kSyncContention * nCpus_)];
  }

  void Add(const int64_t item) {
    Bucket& b = GetBucket(item);
    bool bAdded = false;
    {
      std::unique_lock<std::mutex> lock(b.sync_);
      auto it = b.set_.find(item);
      if(it == b.set_.end()) {
        b.set_.emplace(item);
      }
    }
    if(bAdded) {
      UpdateHash(item);
    }
  }

  void Remove(const int64_t item) {
    Bucket& b = GetBucket(item);
    bool bRemoved = false;
    {
      std::unique_lock<std::mutex> lock(b.sync_);
      auto it = b.set_.find(item);
      if(it != b.set_.end()) {
        b.set_.erase(it);
        bRemoved = true;
      }
    }
    if(bRemoved) {
      UpdateHash(item);
    }
  }

  void Flip(const int64_t item) {
    Bucket& b = GetBucket(item);
    {
      auto it = b.set_.find(item);
      if(it == b.set_.end()) {
        b.set_.emplace(item);
      } else {
        b.set_.erase(it);
      }
    }
    UpdateHash(item);
  }

  // Not thread-safe: no sense in making it thread-safe because it's volatile
  bool Contains(const int64_t item) const {
    const Bucket& b = GetBucket(item);
    return b.set_.find(item) != b.set_.end();
  }

  void Clear() {
    #pragma omp parallel for
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      buckets_[i].set_.clear();
    }
    hash_ = 0;
  }

  bool operator==(const TrackingSet& fellow) const {
    if(hash_ != fellow.hash_) {
      return false;
    }
    bool isEqual = true;
    #pragma omp parallel for shared(isEqual)
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      if(buckets_[i].set_ != fellow.buckets_[i].set_) {
        isEqual = false;
        #pragma omp cancel for
      }
      #pragma omp cancellation point for
    }
    return isEqual;
  }
  bool operator!=(const TrackingSet& fellow) const {
    if(hash_ != fellow.hash_) {
      return true;
    }
    bool differ = false;
    #pragma omp parallel for
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      if(buckets_[i].set_ != fellow.buckets_[i].set_) {
        differ = true;
        #pragma omp cancel for
      }
      #pragma omp cancellation point for
    }
    return differ;
  }

  TrackingSet operator-(const TrackingSet& fellow) const {
    TrackingSet ans;
    #pragma omp parallel for
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      for(const int64_t item : buckets_[i].set_) {
        if(!fellow.Contains(item)) {
          ans.Add(item);
        }
      }
    }
    return ans;
  }

  TrackingSet operator+(const TrackingSet& fellow) const {
    if(size_.load() <= fellow.size_.load()) {
      TrackingSet ans = fellow;
      #pragma omp parallel for
      for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
        for(const int64_t item : buckets_[i].set_) {
          ans.Add(item);
        }
      }
      return ans;
    }
    else {
      return fellow + *this;
    }
  }

  std::vector<int64_t> ToVector() const {
    std::vector<int64_t> ans(size_.load());
    buckets_[0].prefixSum_ = 0;
    //TODO: use parallel prefix sum computation
    for(int64_t i=1; i<kSyncContention * nCpus_; i++) {
      buckets_[i].prefixSum_ = buckets_[i-1].prefixSum_ + buckets_[i].set_.size();
    }
    #pragma omp parallel for
    for(int64_t i=0; i<kSyncContention * nCpus_; i++) {
      int64_t j=buckets_[i].prefixSum_;
      for(const int64_t item : buckets_[i].set_) {
        ans[j] = item;
        j++;
      }
    }
    return ans;
  }
};

namespace std {
  template<> struct hash<TrackingSet> {
    bool operator()(const TrackingSet& ts) const {
      return ts.hash_;
    }
  };
} // namespace std

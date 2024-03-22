#pragma once

#include "Utils.h"

#include <unordered_set>
#include <cstdint>
#include <cassert>

template<typename TItem> struct Bucket {
  mutable std::atomic_flag sync_ = ATOMIC_FLAG_INIT;
  std::unordered_set<TItem> set_;
  int64_t prefixSum_;
};

template<typename TItem> struct MulKHashBaseWithSalt {
  uint128 operator()(const TItem& item) const {
    return item * kHashBase + 37;
  }
};

template<typename TItem, typename THasher=MulKHashBaseWithSalt<TItem>> struct TrackingSet {
  static constexpr const int64_t cSyncContention = 3;

  std::unique_ptr<Bucket<TItem>[]> buckets_ = std::make_unique<Bucket<TItem>[]>(cSyncContention * nSysCpus);
  uint128 hash_ = 0;
  std::atomic<int64_t> size_ = 0;

  TrackingSet() = default;

  void UpdateHash(const TItem& item) {
    const uint128 h = THasher()(item);
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[0].fetch_xor(h & (-1LL));
    reinterpret_cast<std::atomic<uint64_t>*>(&hash_)[1].fetch_xor(h >> 64);
  }

  void CopyFrom(const TrackingSet& src) {
    hash_ = src.hash_;
    size_ = src.Size();
    #pragma omp parallel for
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
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

  Bucket<TItem>& GetBucket(const TItem& item) {
    return buckets_[THasher()(item) % (cSyncContention * nSysCpus)];
  }

  const Bucket<TItem>& GetBucket(const TItem& item) const {
    return buckets_[THasher()(item) % (cSyncContention * nSysCpus)];
  }

  bool Add(const TItem& item) {
    Bucket<TItem>& b = GetBucket(item);
    bool bAdded = false;
    {
      SpinLock lock(b.sync_);
      auto it = b.set_.find(item);
      if(it == b.set_.end()) {
        b.set_.emplace(item);
        bAdded = true;
      }
    }
    if(bAdded) {
      UpdateHash(item);
      [[maybe_unused]] const int64_t oldSize = size_.fetch_add(1, std::memory_order_relaxed);
      assert(oldSize >= 0);
    }
    return bAdded;
  }

  void Remove(const TItem& item) {
    Bucket<TItem>& b = GetBucket(item);
    bool bRemoved = false;
    {
      SpinLock lock(b.sync_);
      auto it = b.set_.find(item);
      if(it != b.set_.end()) {
        b.set_.erase(it);
        bRemoved = true;
      }
    }
    if(bRemoved) {
      UpdateHash(item);
      [[maybe_unused]] const int64_t oldSize = size_.fetch_sub(1, std::memory_order_relaxed);
      assert(oldSize >= 1);
    }
  }

  void Flip(const TItem& item) {
    Bucket<TItem>& b = GetBucket(item);
    int sizeMod = 0;
    {
      SpinLock lock(b.sync_);
      auto it = b.set_.find(item);
      if(it == b.set_.end()) {
        b.set_.emplace(item);
        sizeMod = 1;
      } else {
        b.set_.erase(it);
        sizeMod = -1;
      }
    }
    UpdateHash(item);
    [[maybe_unused]] const int64_t oldSize = size_.fetch_add(sizeMod, std::memory_order_relaxed);
    assert(oldSize + sizeMod >= 0);
  }

  bool Contains(const TItem& item) const {
    const Bucket<TItem>& b = GetBucket(item);
    SpinLock lock(b.sync_);
    return b.set_.find(item) != b.set_.end();
  }

  void Clear() {
    #pragma omp parallel for
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
      buckets_[i].set_.clear();
    }
    hash_ = 0;
    size_ = 0;
  }

  int64_t Size() const {
    return size_.load(std::memory_order_relaxed);
  }

  bool operator==(const TrackingSet& fellow) const {
    if(hash_ != fellow.hash_) {
      return false;
    }
    std::atomic<bool> isEqual = true;
    #pragma omp parallel for shared(isEqual)
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
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
    std::atomic<bool> differ = false;
    #pragma omp parallel for
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
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
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
      for(const TItem& item : buckets_[i].set_) {
        if(!fellow.Contains(item)) {
          ans.Add(item);
        }
      }
    }
    return ans;
  }

  TrackingSet operator+(const TrackingSet& fellow) const {
    if(Size() <= fellow.Size()) {
      TrackingSet ans = fellow;
      #pragma omp parallel for
      for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
        for(const TItem& item : buckets_[i].set_) {
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
    std::vector<int64_t> ans(Size());
    int64_t prefSum = 0;
    buckets_[0].prefixSum_ = prefSum;
    prefSum += buckets_[0].set_.size();
    //TODO: use parallel prefix sum computation?
    for(int64_t i=1; i<cSyncContention * nSysCpus; i++) {
      buckets_[i].prefixSum_ = prefSum;
      prefSum += buckets_[i].set_.size();
    }
    assert( prefSum == Size() );

    #pragma omp parallel for
    for(int64_t i=0; i<cSyncContention * nSysCpus; i++) {
      int64_t j=buckets_[i].prefixSum_;
      for(const TItem& item : buckets_[i].set_) {
        ans[j] = item;
        j++;
      }
    }
    return ans;
  }
};

using VCTrackingSet = TrackingSet<int64_t>;

namespace std {
  template<> struct hash<VCTrackingSet> {
    std::size_t operator()(const VCTrackingSet& ts) const {
      return ts.hash_;
    }
  };
} // namespace std
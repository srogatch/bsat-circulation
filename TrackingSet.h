#pragma once

#include "Utils.h"

#include <unordered_set>
#include <cstdint>
#include <cassert>
#include <algorithm>

template<typename TItem, typename THasher> struct Bucket {
  mutable std::atomic_flag sync_ = ATOMIC_FLAG_INIT;
  std::unordered_set<TItem, THasher> set_;
  int64_t prefixSum_;
};

template<typename TItem> struct MulKHashBaseWithSalt {
  uint128 operator()(const TItem& item) const {
    return item * kHashBase + 37;
  }
};

template<typename TItem> struct MultiItem {
  using value_type = TItem;

  mutable int64_t nEntries_ = -1;
  TItem item_;

  MultiItem() { }

  MultiItem(const TItem& item) : nEntries_(0), item_(item) { }

  MultiItem(const MultiItem& src) = default;
  MultiItem& operator=(const MultiItem& src) = default;
  MultiItem(MultiItem&& src) = default;
  MultiItem& operator=(MultiItem&& src) = default;

  bool operator==(const MultiItem& fellow) const {
    return item_ == fellow.item_;
  }
  bool operator!=(const MultiItem& fellow) const {
    return item_ != fellow.item_;
  }
  bool operator<(const MultiItem& fellow) const {
    return nEntries_ < fellow.nEntries_;
  }
  bool operator>(const MultiItem& fellow) const {
    return nEntries_ > fellow.nEntries_;
  }
};

template<typename T> inline void SortMultiItems(std::vector<MultiItem<T>>& vec, const int sortType) {
  // sort? heap? reverse sort/heap?
  switch(sortType) {
  case 0:
    ParallelShuffle(vec.data(), vec.size());
    break;
  case -1:
    std::make_heap(vec.begin(), vec.end(), std::greater<MultiItem<T>>());
    break;
  case 1:
    std::make_heap(vec.begin(), vec.end());
    break;
  case -2:
    std::sort(vec.begin(), vec.end(), std::greater<MultiItem<T>>());
    break;
  case 2:
    std::sort(vec.begin(), vec.end());
    break;
  }
}

template<typename TItem> struct MulKHashBaseWithSalt<MultiItem<TItem>> {
  uint128 operator()(const MultiItem<TItem>& multiItem) const {
    return multiItem.item_ * kHashBase + 37;
  }
};

template<typename TItem, typename THasher=MulKHashBaseWithSalt<TItem>> struct TrackingSet {
  static constexpr const int64_t cSyncContention = 3;

  std::unique_ptr<Bucket<TItem, THasher>[]> buckets_ = std::make_unique<Bucket<TItem, THasher>[]>(cSyncContention * nSysCpus);
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
    #pragma omp parallel for schedule(guided, cSyncContention)
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

  Bucket<TItem, THasher>& GetBucket(const TItem& item) {
    return buckets_[THasher()(item) % (cSyncContention * nSysCpus)];
  }

  const Bucket<TItem, THasher>& GetBucket(const TItem& item) const {
    return buckets_[THasher()(item) % (cSyncContention * nSysCpus)];
  }

  bool Add(const TItem& item) requires(!is_specialization_of<TItem, MultiItem>::value) {
    Bucket<TItem, THasher>& b = GetBucket(item);
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

  template<typename TValue> std::enable_if<std::is_same<TItem, MultiItem<TValue>>::value, int64_t>::type
  Add(const TValue item) {
    Bucket<TItem, THasher>& b = GetBucket(item);
    MultiItem<TValue> mi(item);
    {
      SpinLock lock(b.sync_);
      typename std::unordered_set<MultiItem<TValue>, MulKHashBaseWithSalt<MultiItem<TValue>>>::iterator it = b.set_.find(mi);
      if(it == b.set_.end()) {
        mi.nEntries_ =  1;
        b.set_.emplace(mi);
      }
      else {
        mi.nEntries_ = it->nEntries_ + 1;
        it->nEntries_ = mi.nEntries_;
      }
    }
    assert(mi.nEntries_ >= 1);
    if(mi.nEntries_ == 1) {
      UpdateHash(item);
      [[maybe_unused]] const int64_t oldSize = size_.fetch_add(1, std::memory_order_relaxed);
      assert(oldSize >= 0);
    }
    return mi.nEntries_;
  }

  void Remove(const TItem& item) requires(!is_specialization_of<TItem, MultiItem>::value) {
    Bucket<TItem, THasher>& b = GetBucket(item);
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

  void Flip(const TItem& item) requires(!is_specialization_of<TItem, MultiItem>::value) {
    Bucket<TItem, THasher>& b = GetBucket(item);
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

  bool Contains(const TItem& item) const
  requires(!is_specialization_of<TItem, MultiItem>::value)
  {
    const Bucket<TItem, THasher>& b = GetBucket(item);
    SpinLock lock(b.sync_);
    return b.set_.find(item) != b.set_.end();
  }

  void Clear() {
    #pragma omp parallel for schedule(guided, cSyncContention)
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
    #pragma omp parallel for schedule(guided, cSyncContention)
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
    #pragma omp parallel for schedule(guided, cSyncContention)
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
    #pragma omp parallel for schedule(guided, cSyncContention)
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
      #pragma omp parallel for schedule(guided, cSyncContention)
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

  std::vector<TItem> ToVector() const {
    std::vector<TItem> ans(Size());
    int64_t prefSum = 0;
    buckets_[0].prefixSum_ = prefSum;
    prefSum += buckets_[0].set_.size();
    //TODO: use parallel prefix sum computation?
    for(int64_t i=1; i<cSyncContention * nSysCpus; i++) {
      buckets_[i].prefixSum_ = prefSum;
      prefSum += buckets_[i].set_.size();
    }
    assert( prefSum == Size() );

    #pragma omp parallel for schedule(guided, cSyncContention)
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

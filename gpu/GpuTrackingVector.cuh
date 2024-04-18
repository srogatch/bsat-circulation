#pragma once

#include "GpuUtils.cuh"
#include "../Utils.h"
#include "Common.h"

template<typename TItem> struct GpuTrackingVector {
  __uint128_t hash_ = 0;
  TItem* items_ = nullptr;
  VciGpu count_ = 0, capacity_ = 0;

  GpuTrackingVector() = default;

  __host__ __device__ static void Copy(TItem* dest, const TItem* src, const VciGpu nItems) {
    if(nItems == 0) [[unlikely]] {
      return;
    }
    assert(dest != nullptr);
    assert(src != nullptr);
    assert( (uintptr_t(dest) & 15) == 0 );
    assert( (uintptr_t(src) & 15) == 0 );
    const VciGpu nVects = DivUp(nItems * sizeof(TItem), sizeof(__uint128_t));
    for(VciGpu i=0; i<nVects; i++) {
      reinterpret_cast<__uint128_t*>(dest)[i] = reinterpret_cast<const __uint128_t*>(src)[i];
    }
  }

  __host__ __device__ static VciGpu AlignCap(const VciGpu count) {
    return AlignUp(count, sizeof(__uint128_t) / sizeof(TItem));
  }

  __host__ __device__ GpuTrackingVector(const GpuTrackingVector& src) {
    hash_ = src.hash_;
    count_ = src.count_;
    capacity_ = AlignCap(src.count_);
    items_ = malloc(capacity_ * sizeof(TItem));
    assert(items_ != nullptr);
    assert((capacity_ * sizeof(TItem)) % sizeof(__uint128_t) == 0);
    Copy(items_, src.items_, count_);
  }

  __host__ __device__ GpuTrackingVector& operator=(const GpuTrackingVector& src)
  {
    if(this != &src) [[likely]] {
      hash_ = src.hash_;
      count_ = src.count_;
      if(capacity_ < src.count_) {
        free(items_);
        capacity_ = AlignCap(src.count_);
        items_ = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
        assert(items_ != nullptr);
      }
      assert((capacity_ * sizeof(TItem)) % sizeof(__uint128_t) == 0);
      Copy(items_, src.items_, count_);
    }
    return *this;
  }

  // Returns whether the vector was resized
  __host__ __device__ bool Reserve(const VciGpu newCap) {
    if(newCap <= capacity_) {
      return false;
    }
    VciGpu maxCap = max(capacity_ + (capacity_>>1) + 8, newCap);
    capacity_ = AlignCap(maxCap);
    TItem* newItems = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
    assert(newItems != nullptr);
    Copy(newItems, items_, count_);
    free(items_);
    items_ = newItems;
    return true;
  }

  // Returns whether the item existed in the collection
  __host__ __device__ bool Flip(const TItem item) {
    hash_ ^= Hasher(item).hash_;
    for(VciGpu i=count_-1; i>=0; i--) {
      if(items_[i] == item) {
        items_[i] = items_[count_-1];
        count_--;
        return true;
      }
    }
    Reserve(count_+1);
    items_[count_] = item;
    count_++;
    return false;
  }

  // Returns whether a new item was added, or a duplicate existed
  template<bool checkDup> __host__ __device__ bool Add(const TItem item) {
    if constexpr(checkDup) {
      for(VciGpu i=count_-1; i>=0; i--) {
        if(items_[i] == item) {
          return false;
        }
      }
    }
    hash_ ^= Hasher(item).hash_;
    Reserve(count_+1);
    items_[count_] = item;
    count_++;
    return true;
  }

  // Returns true if the item had existed in the collection
  __host__ __device__ bool Remove(const TItem& item) {
    for(VciGpu i=count_-1; i>=0; i--) {
      if(items_[i] == item) {
        hash_ ^= Hasher(item).hash_;
        items_[i] = items_[count_-1];
        count_--;
        return true;
      }
    }
    return false;
  }

  __host__ __device__ ~GpuTrackingVector() {
    free(items_);
    items_ = nullptr;
  }

  __host__ __device__ void Clear() {
    hash_ = 0;
    count_ = 0;
  }

  __host__ __device__ void Sort() {
    // Heapify: the greatest item will move to the beginning of the array
    for(VciGpu i=1; i<count_; i++) {
      VciGpu pos = i;
      while(pos > 0 && items_[(pos-1)/2] < items_[pos]) {
        Swap(items_[pos], items_[(pos-1)/2]);
        pos = (pos-1) / 2;
      }
    }
    // Sort by popping heap
    for(VciGpu i=count_-1; i>0; i--) {
      Swap(items_[i], items_[0]);
      VciGpu pos = 0;
      while(pos*2 + 1 < i) {
        VciGpu iChild = pos*2 + 1;
        if(pos*2 + 2 < i && items_[iChild] < items_[pos*2 + 2]) {
          iChild = pos*2 + 2;
        }
        if(items_[pos] < items_[iChild]) {
          Swap(items_[pos], items_[iChild]);
          pos = iChild;
        } else {
          break;
        }
      }
    }
  }

  __host__ __device__ void DelDup() {
    Sort();
    VciGpu l=0, r=count_-1;
    while(l <= r && items_[l] < 0) {
      const VciGpu oppL = -items_[l];
      const VciGpu atR = items_[r];
      if(oppL == atR || (l+1 < r && items_[r-1] == atR)) {
        hash_ ^= Hasher(atR).hash_;
        items_[r] = 0;
        r--;
        continue;
      }
      if(oppL < atR) {
        r--;
        continue;
      }
      hash_ ^= Hasher(items_[l]).hash_;
      items_[l] = 0;
      l++;
    }
    VciGpu newCount = 0;
    for(VciGpu i=0; i<count_; i++) {
      if(items_[i] != 0) {
        items_[newCount] = items_[i];
        newCount++;
      }
    }
    count_ = newCount;
  }

  __host__ __device__ void Shrink() {
    const VciGpu newCap = AlignCap(count_);
    if(newCap >= capacity_) {
      return;
    }
    capacity_ = newCap;
    TItem* newItems = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
    assert(newItems != nullptr);
    Copy(newItems, items_, count_);
    free(items_);
    items_ = newItems;
  }
};

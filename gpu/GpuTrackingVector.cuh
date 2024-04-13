#pragma once

#include "GpuUtils.cuh"
#include "../Utils.h"
#include "Common.h"

struct Hasher {
  __uint128_t hash_;

  __host__ __device__ Hasher(const VciGpu item) {
    hash_ = item * kHashBase + 37;
  }
};

template<typename TItem> struct GpuTrackingVector {
  __uint128_t hash_ = 0;
  TItem* items_ = nullptr;
  VciGpu count_ = 0, capacity_ = 0;

  GpuTrackingVector() = default;

  __host__ __device__ GpuTrackingVector(const GpuTrackingVector& src) {
    hash_ = src.hash_;
    count_ = src.count_;
    free(items_);
    capacity_ = src.count_;
    items_ = malloc(capacity_ * sizeof(TItem));
    // TODO: vectorize
    for(VciGpu i=0; i<count_; i++) {
      items_[i] = src.items_[i];
    }
  }

  __host__ __device__ GpuTrackingVector& operator=(const GpuTrackingVector& src) {
    if(this != &src) {
      hash_ = src.hash_;
      count_ = src.count_;
      if(capacity_ < src.count_) {
        free(items_);
        capacity_ = src.count_;
        items_ = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
      }
      // TODO: vectorize
      for(VciGpu i=0; i<count_; i++) {
        items_[i] = src.items_[i];
      }
    }
    return *this;
  }

  // Returns whether the vector was resized
  __host__ __device__ bool Reserve(const VciGpu newCap) {
    if(newCap <= capacity_) {
      return false;
    }
    VciGpu maxCap = max(capacity_, newCap);
    capacity_ = maxCap + (maxCap>>1) + 16;
    TItem* newItems = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
    // TODO: vectorize
    for(VciGpu i=0; i<count_; i++) {
      newItems[i] = items_[i];
    }
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
        return true;
      }
    }
    return false;
  }

  __host__ __device__ ~GpuTrackingVector() {
    free(items_);
    #ifndef NDEBUG
    items_ = nullptr;
    #endif // NDEBUG
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
          pos = iChild;
        } else {
          break;
        }
      }
    }
  }
};

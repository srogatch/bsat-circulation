#pragma once

#include <cassert>

#include "Common.h"
#include "GpuUtils.cuh"

template<typename TItem> struct GpuDeque {
  TItem* items_ = nullptr;
  VciGpu capacity_ = 0;
  VciGpu iFirst_ = 0;
  VciGpu iLast_ = 0;

  GpuDeque() = default;

  __host__ __device__ void PushBack(TItem&& item, TItem& toRelease) {
    iLast_ = (iLast_ + 1) % capacity_;
    if(iLast_ == iFirst_) {
      toRelease = std::move(items_[iFirst_]);
      iFirst_ = (iFirst_ + 1) % capacity_;
    }
    items_[iLast_] = std::move(item);
  }

  __host__ __device__ bool PopBack(TItem& item) {
    if( IsEmpty() ) {
      return false;
    }
    item = std::move(items_[iLast_]);
    iLast_ = (iLast_ - 1 + capacity_) % capacity_;
    return true;
  }

  __host__ __device__ bool IsEmpty() const {
    return (iLast_+1) % capacity_ == iFirst_;
  }

  __host__ __device__ TItem& Back() {
    assert(!IsEmpty());
    return items_[iLast_];
  }

  __host__ __device__ const TItem& Back() const {
    assert(!IsEmpty());
    return items_[iLast_];
  }
};

template<typename TItem> struct HostDeque {
  CudaArray<TItem> items_;

  HostDeque() = default;

  void Init(const size_t capacity) {
    items_ = CudaArray<TItem>(capacity, CudaArrayType::Pinned);
  }

  GpuDeque<TItem> Marshal() {
    GpuDeque<TItem> ans;
    ans.items_ = items_.Get();
    ans.capacity_ = items_.Count();
    ans.iFirst_ = ans.iLast_ = 0;
    return ans;
  }
};

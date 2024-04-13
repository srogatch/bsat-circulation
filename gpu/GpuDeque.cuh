#pragma once

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
    if( (iLast_+1) % capacity_ == iFirst_ ) {
      return false; // empty
    }
    item = std::move(items_[iLast_]);
    iLast_ = (iLast_ - 1 + capacity_) % capacity_;
    return true;
  }
};

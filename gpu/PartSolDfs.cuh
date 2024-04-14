#pragma once

#include <cassert>

#include "Common.h"
#include "GpuUtils.cuh"
#include "GpuBitVector.cuh"

struct GpuPartSolDfs {
  __uint128_t *pVects_ = nullptr;
  VciGpu vectsPerPartSol_ = 0;
  VciGpu capacity_ = 0;
  VciGpu iFirst_ = 0;
  VciGpu iLast_ = 0;

  GpuPartSolDfs() = default;

  __host__ __device__ void PushBack(const GpuBitVector& asg, const VciGpu nUnsat) {
    iLast_ = (iLast_ + 1) % capacity_;
    if(iLast_ == iFirst_) {
      iFirst_ = (iFirst_ + 1) % capacity_;
    }
    __uint128_t* pSer = pVects_ + uint64_t(vectsPerPartSol_) * iLast_;
    *pSer = asg.hash_;
    pSer++;
    const VciGpu nToCopy = (asg.DwordCount() * sizeof(uint32_t)) / sizeof(__uint128_t);
    for(VciGpu i=0; i<nToCopy; i++) {
      pSer[i] = reinterpret_cast<const __uint128_t*>(asg.bits_)[i];
    }
    pSer += nToCopy;
    __uint128_t curVec = 0;
    const uint8_t tail = asg.DwordCount() - nToCopy * sizeof(__uint128_t) / sizeof(uint32_t);
    for(uint8_t i=0; i<tail; i++) {
      reinterpret_cast<uint32_t*>(&curVec)[i] = asg.bits_[nToCopy * sizeof(__uint128_t) / sizeof(uint32_t) + i];
    }
    if(tail*sizeof(uint32_t) + sizeof(VciGpu) <= sizeof(__uint128_t)) {
      *reinterpret_cast<VciGpu*>(reinterpret_cast<uint32_t*>(&curVec) + tail) = nUnsat;
    } else {
      *pSer = curVec;
      pSer++;
      curVec = 0;
      *reinterpret_cast<VciGpu*>(&curVec) = nUnsat;
    }
    *pSer = curVec;
    pSer++;
    assert(pSer - (pVects_ + uint64_t(vectsPerPartSol_) * iLast_) == vectsPerPartSol_);
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

struct HostPartSolDfs {
  CudaArray<__uint128_t> pVects_;
  VciGpu bitRate_;
  VciGpu vectsPerPartSol_;
  VciGpu capacityPartSols_;

  HostPartSolDfs() = default;

  void Init(const uint64_t maxRamBytes, const VciGpu bitRate) {
    const VciGpu dwordsPerAsg = DivUp(bitRate, 32);
    const VciGpu bytesPerPartSol
      = sizeof(__uint128_t) // hash
      + dwordsPerAsg * sizeof(uint32_t) // bits
      + sizeof(VciGpu); // nUnsat
    vectsPerPartSol_ = DivUp(bytesPerPartSol, sizeof(__uint128_t));
    capacityPartSols_ = std::min<uint64_t>(
      maxRamBytes / (sizeof(__uint128_t) * vectsPerPartSol_),
      std::numeric_limits<VciGpu>::max() - 2
    );
    pVects_ = CudaArray<__uint128_t>(uint64_t(capacityPartSols_) * vectsPerPartSol_, CudaArrayType::Pinned);
  }

  GpuPartSolDfs Marshal() {
    GpuPartSolDfs ans;
    ans.pVects_ = pVects_.Get();
    ans.vectsPerPartSol_ = vectsPerPartSol_;
    ans.capacity_ = capacityPartSols_;
    ans.iFirst_ = ans.iLast_ = 0;
    return ans;
  }
};

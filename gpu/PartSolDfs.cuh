#pragma once

#include <cassert>

#include "Common.h"
#include "GpuUtils.cuh"
#include "GpuBitVector.cuh"

struct GpuPartSolDfs {
  __uint128_t *pVects_ = nullptr;
  VciGpu2 *deque_ = nullptr;
  VciGpu *heads_ = nullptr;
  VciGpu vectsPerPartSol_ = 0;
  VciGpu capacity_ = 0;
  VciGpu iFirst_ = 0;
  VciGpu iLast_ = 0;
  VciGpu leftHeads_ = 0;

  GpuPartSolDfs() = default;

  __host__ __device__ void Serialize(const VciGpu iDeque, const GpuBitVector& asg, const VciGpu nUnsat) {
    const VciGpu iBlock = deque_[iDeque].x;
    __uint128_t* pSer = pVects_ + uint64_t(iBlock) * vectsPerPartSol_;
    *pSer = asg.hash_;
    pSer++;
    const VciGpu nToCopy = (asg.DwordCount() * sizeof(uint32_t)) / sizeof(__uint128_t);
    for(VciGpu i=0; i<nToCopy; i++) {
      pSer[i] = reinterpret_cast<const __uint128_t*>(asg.bits_)[i];
    }
    pSer += nToCopy;
    __uint128_t curVec = 0;
    const uint8_t tail = asg.DwordCount() - nToCopy * sizeof(__uint128_t) / sizeof(uint32_t);
    if(tail > 0) {
      for(uint8_t i=0; i<tail; i++) {
        reinterpret_cast<uint32_t*>(&curVec)[i] = asg.bits_[nToCopy * sizeof(__uint128_t) / sizeof(uint32_t) + i];
      }
      *pSer = curVec;
      pSer++;
    }
    deque_[iDeque].y = nUnsat;
    assert(pSer - (pVects_ + uint64_t(iBlock) * vectsPerPartSol_) == vectsPerPartSol_);
  }

  __host__ __device__ VciGpu PushBack() {
    if(leftHeads_ == 0) {
      return -1;
    }
    iLast_ = (iLast_ + 1) % capacity_;
    if(iLast_ == iFirst_) {
      iFirst_ = (iFirst_ + 1) % capacity_;
      // Wait for deserialization to finish
      while(atomicOr_system(&deque_[iLast_].x, 0) != -1) {
        __nanosleep(256);
      }
    }
    leftHeads_--;
    const VciGpu iBlock = heads_[leftHeads_];
    deque_[iLast_] = {iBlock, -1};
    return iLast_;
  }

  __host__ __device__ void Deserialize(const VciGpu iDeque, GpuBitVector& ans, VciGpu& nUnsat) {
    assert(ans.bits_ != nullptr);
    const VciGpu iBlock = deque_[iDeque].x;
    nUnsat = deque_[iDeque].y;
    __uint128_t* pDes = pVects_ + uint64_t() * vectsPerPartSol_;
    ans.hash_ = *pDes;
    pDes++;
    const VciGpu nToCopy = (ans.DwordCount() * sizeof(uint32_t)) / sizeof(__uint128_t);
    for(VciGpu i=0; i<nToCopy; i++) {
      reinterpret_cast<__uint128_t*>(ans.bits_)[i] = pDes[i];
    }
    pDes += nToCopy;
    __uint128_t curVec = *pDes;
    const uint8_t tail = ans.DwordCount() - nToCopy * sizeof(__uint128_t) / sizeof(uint32_t);
    if(tail > 0) {
      for(uint8_t i=0; i<tail; i++) {
        ans.bits_[nToCopy * sizeof(__uint128_t) / sizeof(uint32_t) + i] = reinterpret_cast<const uint32_t*>(&curVec)[i];
      }
      pDes++;
    }
    [[maybe_unused]] const VciGpu oldInDeque = atomicExch_system(&deque_[iDeque].x, -1);
    assert(oldInDeque == iBlock);
    assert(pDes - (pVects_ + uint64_t(iBlock) * vectsPerPartSol_) == vectsPerPartSol_);
  }

  __host__ __device__ void ReturnHead(VciGpu iBlock) {
    heads_[leftHeads_] = iBlock;
    leftHeads_++;
  }

  __host__ __device__ VciGpu2 PopBack() {
    if( IsEmpty() ) {
      return {-1, -1};
    }
    const VciGpu oldLast = iLast_;
    iLast_ = (iLast_ - 1 + capacity_) % capacity_;
    // Wait for serialization to finish
    while(atomicOr_system(&deque_[oldLast].y, 0) == -1) {
      __nanosleep(128);
    }
    return deque_[oldLast];
  }

  __host__ __device__ bool IsEmpty() const {
    return (iLast_+1) % capacity_ == iFirst_;
  }

  __host__ __device__ VciGpu TopUnsat() const {
    return deque_[iLast_].y;
  }
};

struct HostPartSolDfs {
  CudaArray<__uint128_t> pVects_;
  CudaArray<VciGpu> heads_;
  CudaArray<VciGpu2> deque_;
  VciGpu bitRate_;
  VciGpu vectsPerPartSol_;
  VciGpu capacityPartSols_;

  HostPartSolDfs() = default;

  void Init(const uint64_t maxRamBytes, const VciGpu bitRate) {
    const VciGpu dwordsPerAsg = DivUp(bitRate, 32);
    const VciGpu bytesPerPartSol
      = sizeof(__uint128_t) // hash
      + dwordsPerAsg * sizeof(uint32_t) // bits
      ;
    vectsPerPartSol_ = DivUp(bytesPerPartSol, sizeof(__uint128_t));
    capacityPartSols_ = std::min<uint64_t>(
      maxRamBytes / (sizeof(__uint128_t) * vectsPerPartSol_ + 3*sizeof(VciGpu)),
      std::numeric_limits<VciGpu>::max() - 2
    );
    pVects_ = CudaArray<__uint128_t>(uint64_t(capacityPartSols_) * vectsPerPartSol_, CudaArrayType::Pinned);
    heads_ = CudaArray<VciGpu>(capacityPartSols_, CudaArrayType::Pinned);
    deque_ = CudaArray<VciGpu2>(capacityPartSols_, CudaArrayType::Pinned);
    for(VciGpu i=0; i<capacityPartSols_; i++) {
      heads_.Get()[i] = i;
      deque_.Get()[i] = {-1, -1};
    }
  }

  GpuPartSolDfs Marshal() {
    GpuPartSolDfs ans;
    ans.pVects_ = pVects_.Get();
    ans.heads_ = heads_.Get();
    ans.deque_ = deque_.Get();
    ans.vectsPerPartSol_ = vectsPerPartSol_;
    ans.leftHeads_ = ans.capacity_ = capacityPartSols_;
    ans.iFirst_ = ans.iLast_ = 0;
    return ans;
  }
};

#pragma once

#include "PartSolDfs.cuh"
#include "GpuRainbow.cuh"
#include "GpuBitVector.cuh"
#include "GpuTrackingVector.cuh"

__device__ bool IsSatisfied(const VciGpu aClause, const GpuLinkage& linkage, const GpuBitVector& asg) {
  for(int8_t sign=-1; sign<=1; sign+=2) {
    const VciGpu nClauseArcs = linkage.ClauseArcCount(aClause, sign);
    for(VciGpu j=0; j<nClauseArcs; j++) {
      const VciGpu iVar = linkage.ClauseGetTarget(aClause, sign, j);
      const VciGpu aVar = abs(iVar);
      if(Signum(iVar) == asg[aVar]) {
        return true;
      }
    }
  }
  return false;
}

__device__ void UpdateUnsatCs(const GpuLinkage& linkage, const VciGpu aVar, const GpuBitVector& asg,
  GpuTrackingVector<VciGpu>& unsatClauses)
{
  const int8_t signSat = asg[aVar];
  const VciGpu nSatArcs = linkage.VarArcCount(aVar, signSat);
  for(VciGpu i=0; i<nSatArcs; i++) {
    const VciGpu iClause = linkage.VarGetTarget(aVar, signSat, i);
    const VciGpu aClause = abs(iClause);
    unsatClauses.Remove(aClause);
  }
  const VciGpu nUnsatArcs = linkage.VarArcCount(aVar, -signSat);
  for(VciGpu i=0; i<nUnsatArcs; i++) {
    const VciGpu iClause = linkage.VarGetTarget(aVar, -signSat, i);
    const VciGpu aClause = abs(iClause);
    if(IsSatisfied(aClause, linkage, asg)) {
      continue;
    }
    unsatClauses.Add<true>(aClause);
  }
}

struct GpuPartSol {
  GpuBitVector asg_;
  VciGpu nUnsat_ = -1;

  GpuPartSol() = default;

  __host__ __device__ explicit GpuPartSol(const GpuBitVector& asg, const VciGpu nUnsat) : asg_(asg), nUnsat_(nUnsat) { }
  
  __host__ __device__ GpuPartSol(GpuPartSol&& src) : asg_(std::move(src.asg_)), nUnsat_(src.nUnsat_)
  {
    src.nUnsat_ = -1;
  }

  __host__ __device__ GpuPartSol& operator=(GpuPartSol&& src)
  {
    if(this != &src) {
      asg_ = std::move(src.asg_);
      nUnsat_ = src.nUnsat_;
      src.nUnsat_ = -1;
    }
    return *this;
  }
};

struct GpuTraversal {
  // Store it in pinned memory to save GPU memory - it's not often pushed or popped.
  GpuDeque<GpuPartSol> dfsAsg_;
  // TODO: change from pointer back to value - we're putting the whole GpuTraversal into pinned memory anyway
  int syncDfs_ = 0;

  static __host__ __device__ bool IsSeenAsg(const GpuBitVector& asg, const GpuRainbow& rainbow) {
    return rainbow[asg.hash_];
  }

  __device__ void RecordAsg(const GpuBitVector& asg, const VciGpu nUnsat, GpuRainbow& rainbow) {
    if(!rainbow.Add(asg.hash_)) {
      return;
    }
    GpuPartSol toRelease;

    // Enter spinlock system-wide (all GPUs and CPUs)
    while(atomicCAS_system(&syncDfs_, 0, 1) == 1) {
      __nanosleep(32);
    }
    if(dfsAsg_.IsEmpty() || nUnsat <= dfsAsg_.Back().nUnsat_) {
      dfsAsg_.PushBack(GpuPartSol(asg, nUnsat), toRelease);
    }
    // Leave spinlock system-wide (all GPUs and CPUs)
    [[maybe_unused]] const int oldSync = atomicExch_system(&syncDfs_, 0);
    assert(oldSync == 1);
    if(toRelease.nUnsat_ != -1) {
      rainbow.Remove(toRelease.asg_.hash_);
    }
  }

  __device__ bool StepBack(GpuBitVector &asg, GpuTrackingVector<VciGpu>& unsatClauses, const GpuLinkage& linkage, const VciGpu maxUnsat) {
    GpuPartSol partSol;
    // Enter spinlock system-wide (all GPUs and CPUs)
    while(atomicCAS_system(&syncDfs_, 0, 1) == 1) {
      __nanosleep(32);
    }
    if(!dfsAsg_.IsEmpty()) {
      if(dfsAsg_.Back().nUnsat_ <= maxUnsat) {
        [[maybe_unused]] const bool retrieved = dfsAsg_.PopBack(partSol);
        assert(retrieved);
      }
    }
    // Leave spinlock system-wide (all GPUs and CPUs)
    [[maybe_unused]] const int oldSync = atomicExch_system(&syncDfs_, 0);
    assert(oldSync == 1);

    if(partSol.nUnsat_ == -1) {
      return false;
    }

    for(VciGpu i=0, iLim=asg.DwordCount(); i<iLim; i++) {
      uint32_t diff = asg.bits_[i] ^ partSol.asg_.bits_[i];
      while(diff != 0) {
        const int iBit = __ffs(diff) - 1;
        diff ^= 1u<<iBit;
        const VciGpu aVar = i*32 + iBit;
        asg.Flip(aVar);
        UpdateUnsatCs(linkage, aVar, asg, unsatClauses);
      }
    }
    return true;
  }
};

#pragma once

#include "PartSolDfs.cuh"
#include "GpuRainbow.cuh"
#include "GpuBitVector.cuh"
#include "GpuTrackingVector.cuh"
#include "GpuUnordSet.cuh"

__device__ bool IsSatisfied(const VciGpu aClause, const GpuBitVector& asg) {
  for(int8_t sign=-1; sign<=1; sign+=2) {
    const VciGpu nClauseArcs = gLinkage.ClauseArcCount(aClause, sign);
    for(VciGpu j=0; j<nClauseArcs; j++) {
      const VciGpu iVar = gLinkage.ClauseGetTarget(aClause, sign, j);
      const VciGpu aVar = abs(iVar);
      if(Signum(iVar) == asg[aVar]) {
        return true;
      }
    }
  }
  return false;
}

__device__ void UpdateUnsatCs(const VciGpu aVar, const GpuBitVector& asg,
  GpuUnordSet& unsatClauses)
{
  const int8_t signSat = asg[aVar];
  const VciGpu iVar = aVar * signSat;
  
  VciGpu nArcs = gLinkage.VarArcCount(iVar, -1);
  for(VciGpu j=0; j<nArcs; j++) {
    const VciGpu iClause = gLinkage.VarGetTarget(iVar, -1, j);
    assert(-gLinkage.GetClauseCount() <= iClause && iClause <= -1);
    if(!IsSatisfied(-iClause, asg)) {
      unsatClauses.Add(-iClause);
    }
  }
  nArcs = gLinkage.VarArcCount(iVar, 1);
  for(VciGpu j=0; j<nArcs; j++) {
    const VciGpu iClause = gLinkage.VarGetTarget(iVar, 1, j);
    assert(1 <= iClause && iClause <= gLinkage.GetClauseCount());
    unsatClauses.Remove(iClause);
  }
}

struct GpuTraversal {
  // Store it in pinned memory to save GPU memory - it's not often pushed or popped.
  GpuPartSolDfs dfsAsg_;
  // TODO: change from pointer back to value - we're putting the whole GpuTraversal into pinned memory anyway
  int syncDfs_ = 0;

  static __host__ __device__ bool IsSeenAsg(const GpuBitVector& asg) {
    return gSeenAsgs[asg.hash_];
  }

  __device__ void RecordAsg(const GpuBitVector& asg, const VciGpu nUnsat) {
    if(!gSeenAsgs.Add(asg.hash_)) {
      return;
    }

    __uint128_t oldHash = 0;
    VciGpu2 token = {-1, -1};

    // Enter spinlock system-wide (all GPUs and CPUs)
    while(atomicCAS_system(&syncDfs_, 0, 1) != 0) {
      __nanosleep(32);
    }
    if(dfsAsg_.IsEmpty() || nUnsat <= dfsAsg_.TopUnsat()) {
      token = dfsAsg_.PushBack(nUnsat, oldHash);
    }
    // Leave spinlock system-wide (all GPUs and CPUs)
    [[maybe_unused]] const int oldSync = atomicExch_system(&syncDfs_, 0);
    assert(oldSync == 1);

    assert(token.x >= 0 && token.y >= 0);
    dfsAsg_.Serialize(token, asg);

    if(oldHash != 0) {
      gSeenAsgs.Remove(oldHash);
    }
  }

  __device__ bool StepBack(
    GpuBitVector &asg, GpuUnordSet& unsatClauses, const VciGpu maxUnsat)
  {
    VciGpu2 retrieved{-1, -1};
    // Don't set it to zero because it will be completely overwritten
    GpuBitVector partSol(asg.nBits_, false);

    // Enter spinlock system-wide (all GPUs and CPUs)
    while(atomicCAS_system(&syncDfs_, 0, 1) != 0) {
      __nanosleep(32);
    }
    if(!dfsAsg_.IsEmpty()) {
      if(dfsAsg_.TopUnsat() <= maxUnsat) {
        retrieved = dfsAsg_.PopBack();
        assert(retrieved.x >= 0 && retrieved.y >= 0);
      }
    }
    // Leave spinlock system-wide (all GPUs and CPUs)
    [[maybe_unused]] int oldSync = atomicExch_system(&syncDfs_, 0);
    assert(oldSync == 1);

    if(retrieved.y < 0) {
      return false;
    }

    dfsAsg_.Deserialize(retrieved.x, partSol, retrieved.y);

        // Enter spinlock system-wide (all GPUs and CPUs)
    while(atomicCAS_system(&syncDfs_, 0, 1) != 0) {
      __nanosleep(32);
    }
    dfsAsg_.ReturnHead(retrieved.x);
    oldSync = atomicExch_system(&syncDfs_, 0);
    assert(oldSync == 1);

    for(VciGpu i=0, iLim=asg.DwordCount(); i<iLim; i++) {
      uint32_t diff = asg.bits_[i] ^ partSol.bits_[i];
      while(diff != 0) {
        const int iBit = __ffs(diff) - 1;
        assert(diff & (1u<<iBit) != 0);
        diff ^= 1u<<iBit;
        const VciGpu aVar = i*32 + iBit;
        asg.Flip(aVar);
        UpdateUnsatCs(aVar, asg, unsatClauses);
      }
    }
    return true;
  }
};

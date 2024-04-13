#pragma once

#include "GpuDeque.cuh"
#include "GpuRainbow.cuh"
#include "GpuBitVector.cuh"

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
  // Occupies most of the device memory
  GpuRainbow rainbow_;
  int syncDfs_ = 0;

  __host__ __device__ bool IsSeenAsg(const GpuBitVector& asg) const {
    return rainbow_[asg.hash_];
  }

  __device__ void RecordAsg(const GpuBitVector& asg, const VciGpu nUnsat) {
    if(!rainbow_.Add(asg.hash_)) {
      return;
    }
    GpuPartSol toRelease;

    // Enter spinlock
    while(atomicCAS(&syncDfs_, 0, 1) == 1) {
      __nanosleep(32);
    }
    if(dfsAsg_.IsEmpty() || nUnsat <= dfsAsg_.Back().nUnsat_) {
      dfsAsg_.PushBack(GpuPartSol(asg, nUnsat), toRelease);
    }
    // Leave spinlock
    [[maybe_unused]] const int oldSync = atomicExch(&syncDfs_, 0);
    assert(oldSync == 1);
    if(toRelease.nUnsat_ != -1) {
      rainbow_.Remove(toRelease.asg_.hash_);
    }
  }
};

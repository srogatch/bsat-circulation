#pragma once

#include "Common.h"
#include "GpuUtils.cuh"

struct GpuRainbow {
  uint32_t* bitfield_;
  // uint64_t nbfDwords_;
  uint8_t nUseBits_;
  uint8_t firstUseBit_;

  __device__ __host__ uint64_t ToIndex(const __uint128_t hash) const {
    const uint64_t index = (hash >> firstUseBit_) & ((1ULL<<nUseBits_)-1);
    return index;
  }

  __device__ __host__ bool operator[](const __uint128_t hash) const {
    const uint64_t index = ToIndex(hash);
    return bitfield_[index/32] & (1u<<(index&31));
  }

  // Returns true if item has been adder / false if item had already existed
  __device__ bool Add(const __uint128_t hash) {
    const uint64_t index = ToIndex(hash);
    const uint32_t oldVal = atomicOr(bitfield_ + index/32, 1u<<(index&31));
    return !(oldVal & (1u<<(index&31)));
  }
};

struct HostRainbow {
  CudaArray<uint32_t> bitfield_;
  // Number of uint32_t elements in the bitfield
  uint64_t nbfDwords_ = 0;
  // Number of hash bits used for the Rainbow table
  uint8_t nUseBits_ = 0;
  // The first bit of a hash to use for the Rainbow table
  uint8_t firstUseBit_ = 0;

  HostRainbow() = default;

  // This doesn't synchronize the stream
  explicit HostRainbow(const size_t maxVram, const CudaAttributes& ca) {
    nUseBits_ = uint8_t(std::log2(maxVram)) + 3;
    // TODO: shall we randomize it?
    firstUseBit_ = 0;
    nbfDwords_ = (1ULL << nUseBits_) / (8 * sizeof(uint32_t));
    bitfield_ = CudaArray<uint32_t>(nbfDwords_, CudaArrayType::Device);
    gpuErrchk(cudaMemsetAsync(bitfield_.Get(), 0, nbfDwords_ * sizeof(uint32_t), ca.cs_));
  }

  void Marshal(GpuRainbow& gr) const {
    gr.bitfield_ = bitfield_.Get();
    gr.nUseBits_ = nUseBits_;
    gr.firstUseBit_ = firstUseBit_;
  }
};

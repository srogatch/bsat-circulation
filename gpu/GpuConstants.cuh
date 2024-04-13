#pragma once

#include "../BitVector.h"
#include "Common.h"
#include "GpuUtils.cuh"

__constant__ __uint128_t *gpHashSeries;
std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

void GpuCalcHashSeries(const VciGpu maxItem, const std::vector<CudaAttributes>& cas) {
  BitVector::hashSeries_ = std::make_unique<uint128[]>(maxItem + 1);
  BitVector::hashSeries_[0] = 1;
  for(VciGpu i=1; i<=maxItem; i++) {
    BitVector::hashSeries_[i] = BitVector::hashSeries_[i-1] * kHashBase;
  }
  for(int i=0; i<int(cas.size()); i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyToSymbolAsync(gpHashSeries, BitVector::hashSeries_.get(), sizeof(__uint128_t)*(maxItem+1), 0,
      cudaMemcpyHostToDevice, cas[i].cs_));
  }
  for(int i=0; i<int(cas.size()); i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaStreamSynchronize(cas[i].cs_));
  }
}

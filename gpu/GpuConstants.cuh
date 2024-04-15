#pragma once

#include "../BitVector.h"
#include "Common.h"
#include "GpuUtils.cuh"

__constant__ const __uint128_t *gpHashSeries;
std::unique_ptr<__uint128_t[]> BitVector::hashSeries_ = nullptr;

// NOTE: This doesn't synchronize the streams
void GpuCalcHashSeries(
  const VciGpu maxItem, const std::vector<CudaAttributes>& cas,
  std::vector<CudaArray<__uint128_t>>& gpuHSes)
{
  gpuHSes.resize(cas.size());
  BitVector::hashSeries_ = std::make_unique<__uint128_t[]>(maxItem + 1);
  BitVector::hashSeries_[0] = 1;
  for(VciGpu i=1; i<=maxItem; i++) {
    BitVector::hashSeries_[i] = BitVector::hashSeries_[i-1] * kHashBase;
  }
  for(int i=0; i<int(cas.size()); i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuHSes[i] = CudaArray<__uint128_t>(maxItem+1, CudaArrayType::Device);
    // Copy data
    gpuErrchk(cudaMemcpyAsync(gpuHSes[i].Get(), BitVector::hashSeries_.get(), sizeof(__uint128_t)*(maxItem+1),
      cudaMemcpyHostToDevice, cas[i].cs_));
    const __uint128_t* pHS = gpuHSes[i].Get();
    // Copy pointer
    gpuErrchk(cudaMemcpyToSymbolAsync(gpHashSeries, &pHS, sizeof(__uint128_t*),
      0, cudaMemcpyHostToDevice, cas[i].cs_));
  }
}

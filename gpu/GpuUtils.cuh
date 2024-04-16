#pragma once

#include "Common.h"
#include "../Utils.h"

#include <cuda.h>
#include <cstdint>
#include <cassert>

using VciGpu = int32_t;
using VciGpu2 = int2;

template<typename T1, typename T2> __host__ __device__ void Swap(T1& a, T2& b) {
  T1 t = a;
  a = b;
  b = t;
}

inline __host__ __device__ void VectCopy(void* dest, const void* src, const VciGpu nBytes) {
  assert(nBytes % sizeof(__uint128_t) == 0);
  const VciGpu nVects = nBytes / sizeof(__uint128_t);
  for(VciGpu i=0; i<nVects; i++) {
    reinterpret_cast<__uint128_t*>(dest)[i] = reinterpret_cast<const __uint128_t*>(src)[i];
  }
}

inline __host__ __device__ void VectSetZero(void* dest, const VciGpu nBytes) {
  assert(nBytes % sizeof(__uint128_t) == 0);
  const VciGpu nVects = nBytes / sizeof(__uint128_t);
  for(VciGpu i=0; i<nVects; i++) {
    reinterpret_cast<__uint128_t*>(dest)[i] = 0;
  }
}

#pragma once

#include "Common.h"
#include "../Utils.h"

#include <cuda.h>
#include <cstdint>

using VciGpu = int32_t;
using VciGpu2 = int2;

template<typename T1, typename T2> __host__ __device__ void Swap(T1& a, T2& b) {
  T1 t = a;
  a = b;
  b = t;
}

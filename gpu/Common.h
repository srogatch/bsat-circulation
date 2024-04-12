#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <exception>
#include <sstream>
#include <mutex>
#include <limits>

constexpr const uint32_t kWarpSize = 32;
constexpr const uint32_t kFullWarpMask = uint32_t(-1);

// CUDA error checking helper macro.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// Check that the CUDA call is successful, and if not, exit the application.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU error %d: %s . %s:%d\n", int(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Check cuBlas
inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
  if(code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error %d: %s . %s:%d\n", int(code), cublasGetStatusString(code), file, line);
    if (abort) exit(code);
  }
}

// https://gist.github.com/alexshtf/eb5128b3e3e143187794
namespace Detail
{
	double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
	{
		return curr == prev
			? curr
			: sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
	}
}
/*
* Constexpr version of the square root
* Return value:
*	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
double constexpr csqrt(double x)
{
	return x >= 0 && x < std::numeric_limits<double>::infinity()
		? Detail::sqrtNewtonRaphson(x, x, 0)
		: std::numeric_limits<double>::quiet_NaN();
}

extern std::chrono::steady_clock::time_point gTmLast;

inline void ReportElapsed(const std::string& title) {
  std::chrono::steady_clock::time_point tmNow = std::chrono::steady_clock::now();
  std::cout << title << ": "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(tmNow - gTmLast).count() / 1e9
    << " sec." << std::endl;
  gTmLast = std::chrono::steady_clock::now();
}

struct CudaAttributes {
  cudaDeviceProp cdp_;
  uint32_t max_parallelism_;
  size_t freeBytes_;
  size_t totalBytes_;
  cudaStream_t cs_ = 0;

  void Init(const int i_gpu);
  ~CudaAttributes();
};

enum class CudaArrayType : uint8_t {
  None = 0,
  Device = 1,
  Managed = 2,
  Pinned = 3
};

template<typename T> class CudaArray {
  T* ptr_;
  uint64_t n_items_ : 62;
  uint64_t where_ : 2;

public:
  CudaArray() : ptr_(nullptr), n_items_(0) {}

  explicit CudaArray(const size_t n_items, const CudaArrayType where) : ptr_(nullptr), n_items_(n_items), where_(uint8_t(where)) {
    const size_t n_bytes = sizeof(T) * n_items;
    switch(where) {
    case CudaArrayType::Device:
      gpuErrchk(cudaMalloc(&ptr_, n_bytes));
      break;
    case CudaArrayType::Managed:
      gpuErrchk(cudaMallocManaged(&ptr_, n_bytes));
      break;
    case CudaArrayType::Pinned:
      gpuErrchk(cudaMallocHost(&ptr_, n_bytes));
      break;
    default:
      throw std::runtime_error("Unexpected CudaArrayType");
    }
  }

  CudaArray(const CudaArray&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;

  CudaArray(CudaArray&& fellow) : ptr_(fellow.ptr_), n_items_(fellow.n_items_), where_(fellow.where_) {
    fellow.ptr_ = nullptr;
    fellow.n_items_ = 0;
    fellow.where_ = uint8_t(CudaArrayType::None);
  }
  CudaArray& operator=(CudaArray&& fellow) {
    if(this != &fellow) {
      std::swap(this->ptr_, fellow.ptr_);
      //std::swap(this->n_items_, fellow.n_items_);
      // Swap n_items_
      uint64_t t = this->n_items_;
      this->n_items_ = fellow.n_items_;
      fellow.n_items_ = t;
      // Swap where_
      t = this->where_;
      this->where_ = fellow.where_;
      fellow.where_ = t;
    }
    return *this;
  }

  T* Get() const { return ptr_; }

  size_t Count() const { return n_items_; }

  ~CudaArray() {
    if(where_ != uint8_t(CudaArrayType::None)) {
      if(where_ == uint8_t(CudaArrayType::Pinned)) {
        gpuErrchk(cudaFreeHost(ptr_));
      }
      else {
        gpuErrchk(cudaFree(ptr_));
      }
    }
  }

  void SetZero(const bool on_gpu = true, const cudaStream_t cs = 0) {
    if(on_gpu) {
      gpuErrchk(cudaMemsetAsync(ptr_, 0, sizeof(T)*n_items_, cs));
    } else {
      memset(ptr_, 0, sizeof(T)*n_items_);
    }
  }

  void Prefetch(const cudaStream_t cs, const int i_gpu) {
    gpuErrchk(cudaMemPrefetchAsync(ptr_, sizeof(T)*n_items_, i_gpu, cs));
  }
};

template<typename T> constexpr  __host__ __device__ T DivUp(const T a, const T b) {
  return (a + b - 1) / b;
}

template<typename T> constexpr __host__ __device__ T ToMultiple(const T a, const T b) {
  return DivUp(a, b) * b;
}

struct Logger {
  static std::mutex sync_;

  std::stringstream buf_;

  template<typename T> Logger& operator<<(const T& arg) {
    buf_ << arg;
    return *this;
  }

  Logger();
  ~Logger();
};

constexpr uint64_t rol64(const uint64_t x, const int k)
{
	return (x << k) | (x >> (64 - k));
}

struct Xoshiro256ss {
	ulonglong4 s_;

  __host__ __device__ uint64_t Next() {
    uint64_t const result = rol64(s_.y * 5, 7) * 9;
    uint64_t const t = s_.y << 17;

    s_.z ^= s_.x;
    s_.w ^= s_.y;
    s_.y ^= s_.z;
    s_.x ^= s_.w;

    s_.z ^= t;
    s_.w = rol64(s_.w, 45);

    return result;
  }
};

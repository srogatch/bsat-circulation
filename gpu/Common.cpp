#include "Common.h"
#include <iostream>

std::chrono::steady_clock::time_point gTmLast = std::chrono::steady_clock::now();

void CudaAttributes::Init(const int i_gpu) {
  gpuErrchk(cudaGetDeviceProperties(&cdp_, i_gpu));
  max_parallelism_ = uint32_t(cdp_.maxThreadsPerMultiProcessor) * cdp_.multiProcessorCount;
  gpuErrchk(cudaSetDevice(i_gpu));

  std::cout << "Using device #" << i_gpu << ": " << cdp_.name << std::endl;

  gpuErrchk(cudaMemGetInfo(&freeBytes_, &totalBytes_));
  // Enable mergesort, that allocates device memory
  // gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, free_bytes>>1));

  gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost | cudaDeviceLmemResizeToMax));

  gpuErrchk(cudaStreamCreate(&cs_));
}

CudaAttributes::~CudaAttributes() {
  if(cs_ != 0) {
    gpuErrchk(cudaStreamDestroy(cs_)); 
  }
}

std::mutex Logger::sync_;

Logger::Logger() {
  int iGpu;
  gpuErrchk(cudaGetDevice(&iGpu));
  buf_ << "GPU#" << iGpu << ": ";
}

Logger::~Logger() {
  std::unique_lock<std::mutex> lock(sync_);
  std::cout << buf_.str() << std::endl;
}
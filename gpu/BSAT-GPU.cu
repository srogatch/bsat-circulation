#include "Common.h"

#include "GpuLinkage.cuh"

constexpr const uint32_t kThreadsPerBlock = 128;

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

// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"

struct Hasher {
  __uint128_t hash_;

  Hasher(const VciGpu item) {
    hash_ = item * kHashBase + 37;
  }
};

template<typename TItem> struct GpuTrackingVector {
  __uint128_t hash_ = 0;
  TItem* items_ = nullptr;
  VciGpu count_ = 0, capacity_ = 0;

  __host__ __device__ GpuTrackingVector() = default;

  __host__ __device__ GpuTrackingVector(const GpuTrackingVector& src) {
    hash_ = src.hash_;
    count_ = src.count_;
    free(items_);
    capacity_ = src.count_;
    items_ = malloc(capacity_ * sizeof(TItem));
    // TODO: vectorize
    for(VciGpu i=0; i<count_; i++) {
      items_[i] = src.items_[i];
    }
    return *this;
  }

  __host__ __device__ GpuTrackingVector& operator=(const GpuTrackingVector& src) {
    if(this != &src) {
      hash_ = src.hash_;
      count_ = src.count_;
      if(capacity_ < src.count_) {
        free(items_);
        capacity_ = src.count_;
        items_ = malloc(capacity_ * sizeof(TItem));
      }
      // TODO: vectorize
      for(VciGpu i=0; i<count_; i++) {
        items_[i] = src.items_[i];
      }
    }
    return *this;
  }

  // Returns whether the vector was resized
  __host__ __device__ bool Reserve(const VciGpu newCap) {
    if(newCap <= capacity_) {
      return false;
    }
    VciGpu maxCap = max(capacity_, newCap);
    capacity_ = maxCap + (maxCap>>1) + 16;
    items_ = realloc(items_, capacity_ * sizeof(TItem));
    return true;
  }

  // Returns whether the item existed in the collection
  __host__ __device__ bool Flip(const TItem item) {
    hash_ ^= Hasher(item).hash_;
    for(VciGpu i=count_-1; i>=0; i--) {
      if(items_[i] == item) {
        items_[i] = items_[count_-1];
        count_--;
        return true;
      }
    }
    Reserve(count_+1);
    items_[count_] = item;
    count_++;
    return false;
  }

  // Returns whether a new item was added, or a duplicate existed
  template<bool checkDup> __host__ __device__ bool Add(const TItem item) {
    if constexpr(checkDup) {
      for(VciGpu i=count_-1; i>=0; i--) {
        if(items_[i] == item) {
          return false;
        }
      }  
    }
    hash_ ^= Hasher(item).hash_;
    Reserve(count_+1);
    items_[count_] = item;
    count_++;
    return true;
  }

  __host__ __device__ ~GpuTrackingVector() {
    free(items_);
    #ifndef NDEBUG
    items_ = nullptr;
    #endif // NDEBUG
  }

  __host__ __device__ void Clear() {
    hash_ = 0;
    count_ = 0;
  }
};

struct GpuExec {
  Xoshiro256ss rng_; // seed it on the host
};

__global__ void StepKernel(const VciGpu nStartUnsat, VciGpu* nGlobalUnsat, const GpuLinkage linkage, GpuExec *execs) {
  constexpr const uint32_t cCombsPerStep = 1u<<11;
  const uint32_t iThread = threadIdx.x + blockIdx.x *  kThreadsPerBlock;
  const uint32_t nThreads = gridDim.x * kThreadsPerBlock;
  GpuExec& curExec = execs[iThread];

  GpuBitVector<true, false> next;
  GpuTrackingVector<VciGpu> unsatClauses;
  GpuTrackingVector<VciGpu> front;
  while(unsatClauses.count_ >= nStartUnsat) {
    // TODO: move it to the appropriate place - don't interrupt an improvement flow
    if(*nGlobalUnsat < nStartUnsat) {
      break;
    }
    // Get the variables that affect the unsatisfied clauses
    GpuTrackingVector<VciGpu> varFront;
    uint32_t totListLen = 0;
    const GpuTrackingVector<VciGpu>& combClauses = unsatClauses; // front ?
    for(VciGpu i=0; i<combClauses.count_; i++) {
      for(int8_t sign=-1; sign<=1; sign+=2) {
        const VciGpu aClause = combClauses.items_[i];
        const VciGpu varListLen = linkage.ClauseArcCount(aClause, sign);
        totListLen += varListLen;
        for(VciGpu j=0; j<varListLen; j++) {
          const VciGpu iVar = linkage.ClauseGetTarget(aClause, sign, j);
          const VciGpu aVar = llabs(iVar);
          if( next[aVar] != Signum(iVar) ) {
            varFront.Add<false>(aVar);
            next.Flip(aVar);
          }
        }
      }
    }
    // Flip back the marked vars
    for(VciGpu i=0; i<varFront.count_; i++) {
      next.Flip(varFront.items_[i]);
    }
    // Shuffle the front
    for(VciGpu i=0; i<varFront.count_; i++) {
      const VciGpu pos = i + curExec.rng_.Next() % (varFront.count_ - i);
      const VciGpu t = varFront.items_[i];
      varFront.items_[i] = varFront.items_[pos];
      varFront.items_[pos] = t;
    }

    //// Combine
    GpuTrackingVector<VciGpu> stepRevs;
    // Make sure the overhead of preparing the combinations doesn't outnumber the effort spent in combinations
    uint32_t endComb = max(cCombsPerStep, varFront.count_ + combClauses.count_ + totListLen);
    if(varFront.count_ <= 31) {
      endComb = min(endComb, (1u<<varFront.count_)-1);
    }
    // Initial assignment
    uint32_t curComb = 1;
    {
      const VciGpu aVar = varFront.items_[0];
      stepRevs.Add<false>(aVar);
      next.Flip(aVar);
    }
    while(curComb <= endComb) {
      
    }
  }
}

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::steady_clock::now();
  const auto tmVeryStart = tmStart;

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  
  // TODO: does it override the environment variable?
  omp_set_num_threads(nSysCpus);
  const uint64_t nOmpThreads = omp_get_max_threads();

  Formula formula;
  std::atomic<bool> provenUnsat = false;
  std::atomic<bool> maybeSat = formula.Load(argv[1]);
  if(!maybeSat) {
    provenUnsat = true;
    { // TODO: remove code duplication
      std::ofstream ofs(argv[2]);
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
    }
    return 0;
  }
  int64_t prevNUnsat = formula.nClauses_;

  std::cout << "Precomputing..." << std::endl;
  int nGpus = 0;
  gpuErrchk(cudaGetDeviceCount(&nGpus));
  std::vector<CudaAttributes> cas(nGpus);
  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas);

  return 0;
}

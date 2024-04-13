#include "Common.h"

#include "GpuLinkage.cuh"
#include "GpuConstants.cuh"
// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"
#include "GpuTraversal.cuh"

constexpr const uint32_t kThreadsPerBlock = 128;

struct Hasher {
  __uint128_t hash_;

  __host__ __device__ Hasher(const VciGpu item) {
    hash_ = item * kHashBase + 37;
  }
};

template<typename TItem> struct GpuTrackingVector {
  __uint128_t hash_ = 0;
  TItem* items_ = nullptr;
  VciGpu count_ = 0, capacity_ = 0;

  GpuTrackingVector() = default;

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
  }

  __host__ __device__ GpuTrackingVector& operator=(const GpuTrackingVector& src) {
    if(this != &src) {
      hash_ = src.hash_;
      count_ = src.count_;
      if(capacity_ < src.count_) {
        free(items_);
        capacity_ = src.count_;
        items_ = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
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
    TItem* newItems = reinterpret_cast<TItem*>(malloc(capacity_ * sizeof(TItem)));
    // TODO: vectorize
    for(VciGpu i=0; i<count_; i++) {
      newItems[i] = items_[i];
    }
    free(items_);
    items_ = newItems;
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

  // Returns true if the item had existed in the collection
  __host__ __device__ bool Remove(const TItem& item) {
    for(VciGpu i=count_-1; i>=0; i--) {
      if(items_[i] == item) {
        hash_ ^= Hasher(item).hash_;
        items_[i] = items_[count_-1];
        return true;
      }
    }
    return false;
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
  GpuBitVector next_;
  GpuTrackingVector<VciGpu> unsatClauses_;
  // GpuTrackingVector<VciGpu> front_;
};

__device__ void UpdateUnsatCs(const GpuLinkage& linkage, const VciGpu aVar, const GpuBitVector& next,
  GpuTrackingVector<VciGpu>& unsatClauses)
{
  const int8_t signSat = next[aVar];
  const VciGpu nSatArcs = linkage.VarArcCount(aVar, signSat);
  for(VciGpu i=0; i<nSatArcs; i++) {
    const VciGpu iClause = linkage.VarGetTarget(aVar, signSat, i);
    const VciGpu aClause = abs(iClause);
    unsatClauses.Remove(aClause);
  }
  const VciGpu nUnsatArcs = linkage.VarArcCount(aVar, -signSat);
  for(VciGpu i=0; i<nUnsatArcs; i++) {
    const VciGpu iClause = linkage.VarGetTarget(aVar, -signSat, i);
    const VciGpu aClause = abs(iClause);
    unsatClauses.Add<true>(aClause);
  }
}

__global__ void StepKernel(const VciGpu nStartUnsat, VciGpu* pnGlobalUnsat, const GpuLinkage linkage, GpuExec *execs,
  GpuTraversal* trav, const GpuBitVector maxPartial, VciGpu* pnUnsatExecs)
{
  constexpr const uint32_t cCombsPerStep = 1u<<11;
  const uint32_t iThread = threadIdx.x + blockIdx.x *  kThreadsPerBlock;
  const uint32_t nThreads = gridDim.x * kThreadsPerBlock;
  GpuExec& curExec = execs[iThread];

  while(curExec.unsatClauses_.count_ >= nStartUnsat && *pnGlobalUnsat >= nStartUnsat) {
    // Get the variables that affect the unsatisfied clauses
    GpuTrackingVector<VciGpu> varFront;
    uint32_t totListLen = 0;
    const GpuTrackingVector<VciGpu>& combClauses = curExec.unsatClauses_; // front_ ?
    for(VciGpu i=0; i<combClauses.count_; i++) {
      for(int8_t sign=-1; sign<=1; sign+=2) {
        const VciGpu aClause = combClauses.items_[i];
        const VciGpu varListLen = linkage.ClauseArcCount(aClause, sign);
        totListLen += varListLen;
        for(VciGpu j=0; j<varListLen; j++) {
          const VciGpu iVar = linkage.ClauseGetTarget(aClause, sign, j);
          const VciGpu aVar = abs(iVar);
          if( curExec.next_[aVar] != Signum(iVar) ) {
            varFront.Add<false>(aVar);
            // TODO: this is incorrect - the same variable may appear with the opposite sign in another clause
            curExec.next_.Flip(aVar);
          }
        }
      }
    }
    // Flip back the marked vars
    for(VciGpu i=0; i<varFront.count_; i++) {
      curExec.next_.Flip(varFront.items_[i]);
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
    VciGpu bestUnsat = linkage.GetClauseCount() + 1;
    GpuTrackingVector<VciGpu> bestRevVars;
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
      curExec.next_.Flip(aVar);
      UpdateUnsatCs(linkage, aVar, curExec.next_, curExec.unsatClauses_);
    }
    // The first index participating in combinations - upon success, can be shifted
    VciGpu combFirst = 0;
    while(curComb <= endComb) {
      if(!trav->IsSeenAsg(curExec.next_)) {
        if(curExec.unsatClauses_.count_ < bestUnsat) {
          bestUnsat = curExec.unsatClauses_.count_;
          bestRevVars = stepRevs;
          if(bestUnsat < nStartUnsat) {
            const VciGpu oldMin = atomicMin_system(pnGlobalUnsat, bestUnsat);
            if(oldMin > bestUnsat) {
              combFirst = combFirst + __log2f(curComb-1) + 1;
              curComb = 0;
              const VciGpu remVF = varFront.count_ - combFirst;
              if(remVF <= 31) {
                endComb = min(endComb, (1u<<remVF)-1);
              }
            }
          }
        }
        if(curExec.unsatClauses_.count_ <= *pnGlobalUnsat) {
          trav->RecordAsg(curExec.next_, bestUnsat);
        }
      }
      for(uint8_t i=0; ; i++) {
        curComb ^= 1ULL << i;
        const VCIndex aVar = varFront.items_[i+combFirst];
        stepRevs.Flip(aVar);
        curExec.next_.Flip(aVar);
        UpdateUnsatCs(linkage, aVar, curExec.next_, curExec.unsatClauses_);
        if( (curComb & (1ULL << i)) != 0 ) {
          break;
        }
      }
    }
    // Check the combinations results
    if(bestUnsat > linkage.GetClauseCount()) {
      if(trav.StepBack(curExec.next_, curExec.unsatClauses_, linkage, linkage.GetClauseCount())) {
        continue;
      }
      // Increment the unsatisfied executors counter
      atomicAdd_system(pnUnsatExecs, 1);
      // The current executor considers it unsatisfiable, but let's wait for the rest of executors
      break;
    }
    if(*pnGlobalUnsat < nStartUnsat) {
      break; // some other executor found an improvement
    }
  }
}

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::steady_clock::now();
  const auto tmVeryStart = tmStart;

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs> [<RamGBs>]" << std::endl;
    return 1;
  }

  uint64_t maxRamBytes = 0;
  if(argc >= 4) {
    maxRamBytes = std::stoull(argv[3]) * 1024 * uint64_t(1024) * 1024;
  }
  if(maxRamBytes == 0) {
    maxRamBytes = GetTotalSystemMemory() * 0.95;
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
  std::vector<HostLinkage> linkages(nGpus);
  HostDeque<GpuPartSol> dfsPartial;
  dfsPartial.Init( maxRamBytes / 2 / (DivUp(formula.nVars_, 32)*4 + sizeof(GpuPartSol)) );
  HostRainbow hRainbow; // must be one per device
  // Pinned should be better than managed here, because managed memory transfers at page granularity,
  // while Pinned - at PCIe bus granularity, which is much smaller.
  CudaArray<GpuTraversal> trav(1, CudaArrayType::Pinned);
  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
    // TODO: compute linkages on the CPU once, rather than building it again and again for every GPU
    linkages[i].Init(formula, cas[i]);
    dfses[i].Init()
    travs[i] = CudaArray<GpuTraversal>(1, CudaArrayType::Managed);
    travs[i].Get()->
    seenAsgs[i].Init(cas[i].freeBytes_, cas[i]);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas);


  return 0;
}

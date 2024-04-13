#include "Common.h"

constexpr const uint32_t kThreadsPerBlock = 128;

#include "GpuLinkage.cuh"
#include "GpuConstants.cuh"
// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"
#include "GpuTraversal.cuh"
#include "GpuTrackingVector.cuh"

struct GpuExec {
  Xoshiro256ss rng_; // seed it on the host
  GpuBitVector next_;
  GpuTrackingVector<VciGpu> unsatClauses_;
  // GpuTrackingVector<VciGpu> front_;
};

__global__ void StepKernel(const VciGpu nStartUnsat, VciGpu* pnGlobalUnsat, const GpuLinkage linkage, GpuExec *execs,
  GpuTraversal* trav, GpuRainbow seenAsg, const GpuBitVector maxPartial, VciGpu* pnUnsatExecs)
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
      if(!trav->IsSeenAsg(curExec.next_, seenAsg)) {
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
          trav->RecordAsg(curExec.next_, bestUnsat, seenAsg);
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
      if(trav->StepBack(curExec.next_, curExec.unsatClauses_, linkage, linkage.GetClauseCount())) {
        continue;
      }
      // Increment the unsatisfied executors counter
      atomicAdd_system(pnUnsatExecs, 1);
      // The current executor considers it unsatisfiable, but let's wait for the rest of executors
      break;
    }

    // Revert to the best assignment
    stepRevs.Sort();
    bestRevVars.Sort();

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
  std::vector<HostLinkage> linkages;
  std::vector<HostRainbow> seenAsgs(nGpus); // must be one per device

  // Pinned should be better than managed here, because managed memory transfers at page granularity,
  // while Pinned - at PCIe bus granularity, which is much smaller.
  CudaArray<GpuTraversal> trav(1, CudaArrayType::Pinned);
  HostDeque<GpuPartSol> dfsAsg;
  dfsAsg.Init( maxRamBytes / 2 / (DivUp(formula.nVars_, 32)*4 + sizeof(GpuPartSol)) );
  trav.Get()->dfsAsg_ = dfsAsg.Marshal();
  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
    seenAsgs[i].Init(cas[i].freeBytes_, cas[i]);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas);
  HostLinkage::Init(formula, cas, linkages);
  return 0;
}

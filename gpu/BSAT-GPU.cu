#include "Common.h"

constexpr const uint32_t kThreadsPerBlock = 128;

#include "GpuLinkage.cuh"
#include "GpuConstants.cuh"
// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"
#include "GpuTraversal.cuh"
#include "GpuTrackingVector.cuh"

struct SystemShared {
  GpuTraversal trav_;
  VciGpu nGlobalUnsat_;
  VciGpu nUnsatExecs_;
};

struct PerGpuInfo {
  uint32_t nStepBlocks_;
};

struct GpuExec {
  Xoshiro256ss rng_; // seed it on the host
  GpuBitVector nextAsg_; // nVars+1 bits
  GpuTrackingVector<VciGpu> unsatClauses_;
  // GpuTrackingVector<VciGpu> front_;
};

__global__ void StepKernel(const VciGpu nStartUnsat, SystemShared* sysShar, const GpuLinkage linkage, GpuExec *execs,
  GpuRainbow seenAsg)
{
  constexpr const uint32_t cCombsPerStep = 1u<<11;
  const uint32_t iThread = threadIdx.x + blockIdx.x *  kThreadsPerBlock;
  const uint32_t nThreads = gridDim.x * kThreadsPerBlock;
  GpuExec& curExec = execs[iThread];

  while(curExec.unsatClauses_.count_ >= nStartUnsat && sysShar->nGlobalUnsat_ >= nStartUnsat) {
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
          varFront.Add<true>(aVar);
        }
      }
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
      curExec.nextAsg_.Flip(aVar);
      UpdateUnsatCs(linkage, aVar, curExec.nextAsg_, curExec.unsatClauses_);
    }
    // The first index participating in combinations - upon success, can be shifted
    VciGpu combFirst = 0;
    while(curComb <= endComb) {
      if(!sysShar->trav_.IsSeenAsg(curExec.nextAsg_, seenAsg)) {
        if(curExec.unsatClauses_.count_ < bestUnsat) {
          bestUnsat = curExec.unsatClauses_.count_;
          bestRevVars = stepRevs;
          if(bestUnsat < nStartUnsat) {
            const VciGpu oldMin = atomicMin_system(&sysShar->nGlobalUnsat_, bestUnsat);
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
        if(curExec.unsatClauses_.count_ <= sysShar->nGlobalUnsat_) {
          sysShar->trav_.RecordAsg(curExec.nextAsg_, bestUnsat, seenAsg);
        }
      }
      for(uint8_t i=0; ; i++) {
        curComb ^= 1ULL << i;
        const VCIndex aVar = varFront.items_[i+combFirst];
        stepRevs.Flip(aVar);
        curExec.nextAsg_.Flip(aVar);
        UpdateUnsatCs(linkage, aVar, curExec.nextAsg_, curExec.unsatClauses_);
        if( (curComb & (1ULL << i)) != 0 ) {
          break;
        }
      }
    }
    // Check the combinations results
    if(bestUnsat > linkage.GetClauseCount()) {
      if(sysShar->trav_.StepBack(curExec.nextAsg_, curExec.unsatClauses_, linkage, linkage.GetClauseCount())) {
        continue;
      }
      // Increment the unsatisfied executors counter
      atomicAdd_system(&sysShar->nUnsatExecs_, 1);
      // The current executor considers it unsatisfiable, but let's wait for the rest of executors
      break;
    }

    // Revert to the best assignment
    stepRevs.Sort();
    bestRevVars.Sort();
    VciGpu iSR = 0, iBR = 0;
    while(iSR < stepRevs.count_ || iBR < bestRevVars.count_) {
      VciGpu aVar;
      if(iBR >= bestRevVars.count_ || stepRevs.items_[iSR] < bestRevVars.items_[iBR]) {
        aVar = stepRevs.items_[iSR];
        iSR++;
      } else if(iSR >= stepRevs.count_ || bestRevVars.items_[iBR] < stepRevs.items_[iSR]) {
        aVar = bestRevVars.items_[iBR];
        iBR++;
      } else {
        assert(iSR < stepRevs.count_);
        assert(iBR < bestRevVars.count_);
        assert(stepRevs.items_[iSR] == bestRevVars.items_[iBR]);
        iSR++;
        iBR++;
        continue;
      }
      curExec.nextAsg_.Flip(aVar);
      UpdateUnsatCs(linkage, aVar, curExec.nextAsg_, curExec.unsatClauses_);
    }

    if(sysShar->nGlobalUnsat_ < nStartUnsat) {
      break; // some other executor found an improvement
    }

    // TODO: Sequential Gradient Descent
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
  std::vector<HostRainbow> seenAsgs(nGpus); // must be one per device

  // Pinned should be better than managed here, because managed memory transfers at page granularity,
  // while Pinned - at PCIe bus granularity, which is much smaller.
  CudaArray<SystemShared> sysShar(1, CudaArrayType::Pinned);
  HostPartSolDfs dfsAsg;
  dfsAsg.Init( maxRamBytes / 2, formula.ans_.nBits_ );
  sysShar.Get()->trav_.dfsAsg_ = dfsAsg.Marshal();
  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas);

  std::cout << "Choosing the initial assignment..." << std::endl;
  
  BitVector bestInitAsg = formula.ans_;
  VciGpu bestInitNUnsat = formula.CountUnsat(formula.ans_);
  std::cout << "All false: " << bestInitNUnsat << ", ";
  std::cout.flush();

  formula.ans_.SetTrue();
  VciGpu altNUnsat = formula.CountUnsat(formula.ans_);
  std::cout << "All true: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInitNUnsat) {
    bestInitNUnsat = altNUnsat;
    bestInitAsg = formula.ans_;
  }

  formula.ans_.Randomize();
  altNUnsat = formula.CountUnsat(formula.ans_);
  std::cout << "Random: " << altNUnsat << std::endl;
  if(altNUnsat < bestInitNUnsat) {
    bestInitNUnsat = altNUnsat;
    bestInitAsg = formula.ans_;
  }

  std::cout << "Preparing GPU data structures" << std::endl;
  std::vector<HostLinkage> linkages;
  HostLinkage::Init(formula, cas, linkages);
  std::vector<CudaArray<GpuExec>> execs(nGpus);
  std::vector<PerGpuInfo> pgis(nGpus);
  // BPCT - Bytes Per CUDA Thread
  const uint64_t hostHeapBpct
    = sizeof(GpuExec)
    + AlignUp(DivUp(formula.nVars_, 32) * 4, 256) // nextAsg_
    + 512; // Thread stack
  const uint64_t deviceHeapBpct
    // GpuExec::unsatClauses_
    // varFront
    // stepRevs
    // bestRevVars
    = (bestInitNUnsat + (bestInitNUnsat>>1) + 16) * sizeof(VciGpu) * 4;
  
  for(int i=0; i<nGpus; i++) {
    int nBlocksPerSM = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocksPerSM, &StepKernel, kThreadsPerBlock, 0));
    // This is the upper bound for now without the correction for the actually available VRAM
    pgis[i].nStepBlocks_ = nBlocksPerSM * cas[i].cdp_.multiProcessorCount;
    const uint64_t bytesBothHeaps = pgis[i].nStepBlocks_ * (hostHeapBpct + deviceHeapBpct);
    // TODO: refresh the VRAM free bytes with a query
    seenAsgs[i].Init(cas[i].freeBytes_, cas[i]);
  }
  return 0;
}

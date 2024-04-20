//#undef NDEBUG

#include "Common.h"

#include <cassert>

constexpr const uint32_t kThreadsPerBlock = 128;
constexpr const uint8_t kL2SolRoundRobin = 13; // log2( # solutions in the round-robin )
constexpr const uint32_t kMaxVarFrontSize = 4096;

#include "CpuInit.h"

#include "GpuLinkage.cuh"
__constant__ GpuLinkage gLinkage;

#include "GpuRainbow.cuh"
__constant__ GpuRainbow gSeenAsgs;

#include "GpuConstants.cuh"
// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"
#include "GpuTraversal.cuh"
#include "GpuTrackingVector.cuh"

struct SystemShared {
  GpuTraversal trav_;
  __uint128_t *solRRasgs_;
  VciGpu *solRRnsUnsat_;
  VciGpu nGlobalUnsat_;
  VciGpu nUnsatExecs_;
  VciGpu firstSolRR_;
  VciGpu limitSolRR_;
  int syncRR_;

  __device__ void Record(const GpuBitVector& asg, const VciGpu nUnsat) {
    bool full;
    VciGpu target = -1;
    for(;;) {
      // Lock
      while(atomicCAS_system(&syncRR_, 0, 1) != 0) {
        __nanosleep(256);
      }
      const VciGpu newLimit = (limitSolRR_ + 1) & ((VciGpu(1)<<kL2SolRoundRobin)-1);
      full = (newLimit == firstSolRR_);
      if(!full) {
        target = limitSolRR_;
        limitSolRR_ = newLimit;
        // Lock
        atomicExch_system(solRRnsUnsat_ + target, -1);
      }
      // Unlock
      atomicExch_system(&syncRR_, 0);
      if(!full) {
        break;
      }
      __nanosleep(1024);
    }
    assert(!full);
    assert( 0 <= target && target < (VciGpu(1)<<kL2SolRoundRobin) );
    __uint128_t *pWrite = solRRasgs_ + uint64_t(target) * asg.VectCount();
    for(VciGpu i=0; i<asg.VectCount(); i++) {
      pWrite[i] = reinterpret_cast<__uint128_t*>(asg.bits_)[i];
    }
    // Unlock
    atomicExch_system(solRRnsUnsat_+target, nUnsat);
  }

  __host__ bool Consume(BitVector& asg, VciGpu &nUnsat) volatile {
    VciGpu iPop = -1;
    static_assert(std::atomic<int>::is_always_lock_free, "Must be same size as int");
    volatile std::atomic<int> *pSync = reinterpret_cast<volatile std::atomic<int>*>(&syncRR_);
    
    // Lock
    int state = 0;
    while(!pSync->compare_exchange_strong(state, 1)) {
      assert(state == 1);
      state = 0;
      __builtin_ia32_pause();
    }

    const VciGpu iFirst = firstSolRR_;
    if(iFirst != limitSolRR_) {
      iPop = iFirst;
      firstSolRR_ = (iFirst + 1) & ((VciGpu(1)<<kL2SolRoundRobin)-1);
    }
    
    // Unlock
    pSync->store(0);

    if(iPop == -1) {
      return false;
    }

    volatile std::atomic<VciGpu>* pCounterLock = reinterpret_cast<volatile std::atomic<VciGpu>*>(solRRnsUnsat_ + iPop);
    while( (nUnsat = pCounterLock->load()) == -1 ) {
      __builtin_ia32_pause();
    }
    memcpy(asg.bits_.get(), solRRasgs_ + uint64_t(iPop) * DivUp(asg.nBits_, 128), asg.nQwords_ * sizeof(uint64_t));
    return true;
  }
};

struct GpuExec {
  Xoshiro256ss rng_; // seed it on the host
  GpuBitVector nextAsg_; // nVars+1 bits
  GpuUnordSet unsatClauses_;
  // GpuTrackingVector<VciGpu> front_;
  VciGpu *varFrontItems_;
  VciGpu varFrontSize_;
};

struct PerGpuInfo {
  CudaArray<GpuExec> execs_;
  CudaArray<__uint128_t> bvBufs_;
  CudaArray<VciGpu> allVarFrontItems_;
  GpuLinkage gl_;
  GpuRainbow gr_;
  uint32_t nStepBlocks_;  
};

__global__ void ReplicateAssignment(GpuExec* execs, const uint32_t nExecs) {
  const uint32_t iThread = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t nThreads = blockDim.x * gridDim.x;
  for(uint32_t i=iThread + 1; i<nExecs; i+=nThreads) {
    GpuExec& curExec = execs[i];
    VectCopy(curExec.nextAsg_.bits_, execs[0].nextAsg_.bits_, curExec.nextAsg_.VectCount() * sizeof(__uint128_t));
    VciGpu iFlip = curExec.rng_.Next() % (curExec.nextAsg_.nBits_ - 1) + 1;
    curExec.nextAsg_.Flip(iFlip);
  }
}

__global__ void StepKernel(const VciGpu nStartUnsat, SystemShared* sysShar, GpuExec *execs)
{
  constexpr const uint32_t cCombsPerStep = 1u<<11;
  const uint32_t iThread = threadIdx.x + blockIdx.x * kThreadsPerBlock;
  assert(blockDim.x == kThreadsPerBlock);
  // const uint32_t nThreads = gridDim.x * kThreadsPerBlock;
  GpuExec& curExec = execs[iThread];

  if(curExec.unsatClauses_.buffer_ == nullptr) {
    assert(curExec.unsatClauses_.bitsPerPack_ == 0);
    assert(curExec.unsatClauses_.nBuckets_ == 0);
    assert(curExec.unsatClauses_.hash_ == 0);
    assert(curExec.unsatClauses_.count_ == 0);

    const VciGpu nClauses=gLinkage.GetClauseCount();
    curExec.unsatClauses_ = GpuUnordSet(nStartUnsat, nClauses);
    for(VciGpu i=1; i<=nClauses; i++) {
      if(!IsSatisfied(i, curExec.nextAsg_)) {
        curExec.unsatClauses_.Add(i);
      }
    }
  }

  assert(curExec.unsatClauses_.count_ >= sysShar->nGlobalUnsat_);

  while(curExec.unsatClauses_.count_ >= nStartUnsat && sysShar->nGlobalUnsat_ >= nStartUnsat) {
    // Save memory for varFront
    curExec.unsatClauses_.Shrink( curExec.rng_.Next() );
    // Get the variables that affect the unsatisfied clauses
    const GpuUnordSet& combClauses = curExec.unsatClauses_; // front_ ?
    curExec.varFrontSize_ = 0;
    combClauses.Visit<false>(curExec.rng_.Next(), [&](const VciGpu aClause) -> bool {
      for(int8_t sign=-1; sign<=1; sign+=2) {
        const VciGpu varListLen = gLinkage.ClauseArcCount(aClause, sign);
        for(VciGpu j=0; j<varListLen; j++) {
          const VciGpu iVar = gLinkage.ClauseGetTarget(aClause, sign, j);
          const VciGpu aVar = abs(iVar);
          // Let the duplicate variables appear multiple times in the array, and thus
          // be considered for combinations multiple times proportionally to their
          // entry numbers.
          assert(1 <= aVar && aVar <= gLinkage.GetVarCount());
          if(curExec.varFrontSize_ < kMaxVarFrontSize) {
            curExec.varFrontItems_[curExec.varFrontSize_] = aVar;
            curExec.varFrontSize_++;
          } else {
            return false;
            // Reservoir sampling
            // const uint64_t r = curExec.rng_.Next() % totVarFront;
            // if(r < kMaxVarFrontSize) {
            //   curExec.varFrontItems_[r] = aVar;
            // }
          }
        }
      }
      return true;
    });

    // Shuffle the front
    for(VciGpu i=0; i+1<curExec.varFrontSize_; i++) {
      const VciGpu pos = i + curExec.rng_.Next() % (curExec.varFrontSize_ - i);
      Swap(curExec.varFrontItems_[i], curExec.varFrontItems_[pos]);
    }

    //// Combine
    GpuTrackingVector<VciGpu> stepRevs;
    VciGpu bestUnsat = gLinkage.GetClauseCount() + 1;
    GpuTrackingVector<VciGpu> bestRevVars;
    // Make sure the overhead of preparing the combinations doesn't outnumber the effort spent in combinations
    uint32_t endComb = cCombsPerStep; //max(cCombsPerStep, totVarFront);
    if(curExec.varFrontSize_ <= 31) [[unlikely]] {
      endComb = min(endComb, (1u<<curExec.varFrontSize_)-1);
    }
    // Initial assignment
    uint32_t curComb = 1;
    {
      const VciGpu aVar = curExec.varFrontItems_[0];
      assert(1 <= aVar && aVar <= gLinkage.GetVarCount());
      stepRevs.Add<false>(aVar);
      curExec.nextAsg_.Flip(aVar);
      UpdateUnsatCs(aVar, curExec.nextAsg_, curExec.unsatClauses_);
    }
    // The first index participating in combinations - upon success, can be shifted
    VciGpu combFirst = 0;
    while(curComb <= endComb) {
      if(!sysShar->trav_.IsSeenAsg(curExec.nextAsg_)) {
        if(curExec.unsatClauses_.count_ < bestUnsat) {
          bestUnsat = curExec.unsatClauses_.count_;
          bestRevVars = stepRevs;
          // TODO: remove (DEBUG)
          // curExec.unsatClauses_.Visit([&](const VciGpu aClause) {
          //   assert( !IsSatisfied(aClause, curExec.nextAsg_) );
          // });
          // for(VciGpu i=1; i<=gLinkage.GetClauseCount(); i++) {
          //   if(IsSatisfied(i, curExec.nextAsg_)) {
          //     assert(!curExec.unsatClauses_.Contains(i));
          //   } else {
          //     assert(curExec.unsatClauses_.Contains(i));
          //   }
          // }
          if(bestUnsat < nStartUnsat) [[unlikely]] {
            const VciGpu oldMin = atomicMin_system(&sysShar->nGlobalUnsat_, bestUnsat);
            if(oldMin > bestUnsat) [[likely]] {
              sysShar->Record(curExec.nextAsg_, bestUnsat);
              combFirst = combFirst + 32 - __clz(curComb);
              curComb = 0;
              const VciGpu remVF = curExec.varFrontSize_ - combFirst;
              if(remVF <= 31) [[unlikely]] {
                endComb = min(cCombsPerStep, (1u<<remVF)-1);
              } else {
                endComb = cCombsPerStep;
              }
            }
          }
        }
        if(curExec.unsatClauses_.count_ <= sysShar->nGlobalUnsat_) [[unlikely]] {
          sysShar->trav_.RecordAsg(curExec.nextAsg_, curExec.unsatClauses_.count_);
        }
      }
      for(uint8_t i=0; ; i++) {
        curComb ^= 1u << i;
        assert(i + combFirst < curExec.varFrontSize_);
        const VCIndex aVar = curExec.varFrontItems_[i+combFirst];
        assert(1 <= aVar && aVar <= gLinkage.GetVarCount());
        [[maybe_unused]] const bool bExisted = stepRevs.Flip(aVar);
        // Variables may repeat inside varFront
        //assert( bExisted == !(curComb & (1u << i)) );
        curExec.nextAsg_.Flip(aVar);
        UpdateUnsatCs(aVar, curExec.nextAsg_, curExec.unsatClauses_);
        if( (curComb & (1u << i)) != 0u ) {
          break;
        }
      }
    }

    // Check the combinations results
    if(bestUnsat > gLinkage.GetClauseCount()) [[unlikely]] {
      if(sysShar->trav_.StepBack(curExec.nextAsg_, curExec.unsatClauses_, gLinkage.GetClauseCount())) [[likely]] {
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
      assert(iSR == 0 || iSR >= stepRevs.count_ || stepRevs.items_[iSR-1] < stepRevs.items_[iSR]);
      assert(iBR == 0 || iBR >= bestRevVars.count_ || bestRevVars.items_[iBR-1] < bestRevVars.items_[iBR]);
      VciGpu aVar;
      if( iBR >= bestRevVars.count_ || (iSR < stepRevs.count_ && stepRevs.items_[iSR] < bestRevVars.items_[iBR]) ) {
        aVar = stepRevs.items_[iSR];
        iSR++;
      } else if( iSR >= stepRevs.count_ || (iBR < bestRevVars.count_ && bestRevVars.items_[iBR] < stepRevs.items_[iSR]) ) {
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
      // if(!(1 <= aVar && aVar <= gLinkage.GetVarCount())) {
      //   printf(" %d ", aVar);
      // }
      curExec.nextAsg_.Flip(aVar);
      UpdateUnsatCs(aVar, curExec.nextAsg_, curExec.unsatClauses_);
    }
    // // TODO: remove (DEBUG)
    // for(VciGpu i=1; i<=gLinkage.GetClauseCount(); i++) {
    //   if(IsSatisfied(i, curExec.nextAsg_)) {
    //     assert(!curExec.unsatClauses_.Contains(i));
    //   } else {
    //     assert(curExec.unsatClauses_.Contains(i));
    //   }
    // }
    // // TODO: remove (DEBUG)
    // if(curExec.unsatClauses_.count_ != bestUnsat) {
    //   while(atomicCAS_system(&sysShar->syncRR_, 0, 1) != 0) {
    //     __nanosleep(256);
    //   }
    //   printf("stepRevs: ");
    //   for(VciGpu i=0; i<stepRevs.count_; i++) {
    //     printf(" %d ", stepRevs.items_[i]);
    //   }
    //   printf("\nbestRevVars: ");
    //   for(VciGpu i=0; i<bestRevVars.count_; i++) {
    //     printf(" %d ", bestRevVars.items_[i]);
    //   }
    //   printf("\n");
    //   atomicExch_system(&sysShar->syncRR_, 0);
    //   assert(false);
    // }
    assert(curExec.unsatClauses_.count_ == bestUnsat);
    assert(curExec.unsatClauses_.count_ >= sysShar->nGlobalUnsat_);

    if(sysShar->nGlobalUnsat_ < nStartUnsat) [[unlikely]] {
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
  std::vector<CudaArray<__uint128_t>> gpuHSes;
  std::vector<HostRainbow> seenAsgs(nGpus); // must be one per device

  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas, gpuHSes);

  // TODO: call CPU Init here
  const VciGpu bestInitNUnsat = CpuInit(formula);

  std::cout << "Preparing GPU data structures" << std::endl;
  std::vector<HostLinkage> linkages;
  HostLinkage::Init(formula, cas, linkages);
  std::vector<CudaArray<GpuExec>> execs(nGpus);
  std::vector<PerGpuInfo> pgis(nGpus);
  const VciGpu nVectsPerVarsBV = DivUp(formula.nVars_ + 1, 128);
  // BPCT - Bytes Per CUDA Thread
  const uint64_t hostHeapBpct
    = sizeof(GpuExec)
    + nVectsPerVarsBV * sizeof(__uint128_t) // GpuExec::nextAsg_
    + kMaxVarFrontSize * sizeof(VciGpu) // varFrontItems_
    + 256 * 3 // Alignment
    + 256; // Thread stack
  const uint64_t deviceHeapBpct
    // GpuExec::unsatClauses_
    // varFront
    // stepRevs
    // bestRevVars
    = ( (bestInitNUnsat / GpuUnordSet::cStartOccupancy + 16) * ceilf(log2f(formula.nClauses_+1)) / 8
    + kMaxVarFrontSize * 2 * sizeof(VciGpu) + 16 * 6 );
  const uint64_t overheadBpct = deviceHeapBpct/6;

    // Pinned should be better than managed here, because managed memory transfers at page granularity,
  // while Pinned - at PCIe bus granularity, which is much smaller.
  CudaArray<SystemShared> sysShar(1, CudaArrayType::Managed);
  HostPartSolDfs dfsAsg;
  dfsAsg.Init( maxRamBytes / 2, formula.ans_.nBits_ );
  CudaArray<VciGpu> solRRnsUnsat( 1u<<kL2SolRoundRobin, CudaArrayType::Managed );
  
  #pragma omp parallel for num_threads(nGpus)
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    linkages[i].Marshal(pgis[i].gl_);
    gpuErrchk(cudaMemcpyToSymbolAsync(gLinkage, &pgis[i].gl_, sizeof(pgis[i].gl_), 0, cudaMemcpyHostToDevice, cas[i].cs_));
    int nBlocksPerSM = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocksPerSM, &StepKernel, kThreadsPerBlock, 0));
    // This is the upper bound for now without the correction for the actually available VRAM
    pgis[i].nStepBlocks_ = nBlocksPerSM * cas[i].cdp_.multiProcessorCount;
    gpuErrchk(cudaMemGetInfo(&cas[i].freeBytes_, &cas[i].totalBytes_));
    uint64_t maxRainbowBytes = 1ULL << lround(ceil(log2(cas[i].freeBytes_/3)));
    for(;;) {
      uint64_t bytesBothHeaps = pgis[i].nStepBlocks_ * uint64_t(kThreadsPerBlock) * (hostHeapBpct + deviceHeapBpct + overheadBpct);
      uint64_t rainbowBytes = 1ULL << int(std::log2(maxRainbowBytes));
      uint64_t totVramReq = bytesBothHeaps + rainbowBytes;
      if(totVramReq <= cas[i].freeBytes_) {
        break;
      }
      const double reduction = std::sqrt( double(cas[i].freeBytes_) / totVramReq );
      maxRainbowBytes *= reduction;
      pgis[i].nStepBlocks_ *= reduction;
    }
    seenAsgs[i].Init(maxRainbowBytes, cas[i]);
    gpuErrchk(cudaMemGetInfo(&cas[i].freeBytes_, &cas[i].totalBytes_));
    pgis[i].nStepBlocks_ = std::min<uint64_t>(
      nBlocksPerSM * cas[i].cdp_.multiProcessorCount,
      cas[i].freeBytes_
        / (uint64_t(kThreadsPerBlock) * (hostHeapBpct + deviceHeapBpct + overheadBpct))
    );
    Logger() << "Rainbow Table: " << double(uint64_t(seenAsgs[i].nbfDwords_) * sizeof(uint32_t)) / (1ULL<<30)
      << " GB, Host heap: "
      << double(pgis[i].nStepBlocks_) *  kThreadsPerBlock * hostHeapBpct / (1ULL<<30)
      << " GB, Device heap: "
      << double(pgis[i].nStepBlocks_) *  kThreadsPerBlock * deviceHeapBpct / (1ULL<<30)
      << " GB. nStepBlocks: " << pgis[i].nStepBlocks_;

    // Enable dynamic memory allocation in the CUDA kernel
    gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
      pgis[i].nStepBlocks_ * uint64_t(kThreadsPerBlock) * deviceHeapBpct));
    pgis[i].bvBufs_ = CudaArray<__uint128_t>(
      pgis[i].nStepBlocks_ * uint64_t(kThreadsPerBlock) * nVectsPerVarsBV, CudaArrayType::Device
    );
    pgis[i].allVarFrontItems_ = CudaArray<VciGpu>(
      pgis[i].nStepBlocks_ * uint64_t(kThreadsPerBlock) * kMaxVarFrontSize, CudaArrayType::Device
    );
    pgis[i].execs_ = CudaArray<GpuExec>(
      pgis[i].nStepBlocks_ * uint64_t(kThreadsPerBlock), CudaArrayType::Device
    );
  }
  std::atomic<int64_t> totExecs = 0;
  std::vector<std::unique_ptr<GpuExec[]>> vCpuExecs(nGpus);
  #pragma omp parallel for num_threads(nGpus)
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    vCpuExecs[i] = std::make_unique<GpuExec[]>(pgis[i].execs_.Count());
    totExecs.fetch_add(pgis[i].execs_.Count());
    std::random_device rd;
    for(uint32_t j=0; j<pgis[i].execs_.Count(); j++) {
      for(int k=0; k<int(sizeof(vCpuExecs[i][j].rng_.s_)); k+=sizeof(uint32_t)) {
        reinterpret_cast<uint32_t*>(&vCpuExecs[i][j].rng_.s_)[k/sizeof(uint32_t)] = rd();
      }
      vCpuExecs[i][j].nextAsg_.hash_ = formula.ans_.hash_;
      vCpuExecs[i][j].nextAsg_.nBits_ = formula.nVars_ + 1;
      vCpuExecs[i][j].nextAsg_.bits_ = reinterpret_cast<uint32_t*>(
        pgis[i].bvBufs_.Get() + nVectsPerVarsBV * uint64_t(j));
      vCpuExecs[i][j].varFrontItems_ = pgis[i].allVarFrontItems_.Get() + uint64_t(j) * kMaxVarFrontSize;
      assert(vCpuExecs[i][j].unsatClauses_.buffer_ == nullptr);
    }
    // TODO: tail bits (beyond the last QWord, but withing the last 128-bit vector) may be corrupt
    gpuErrchk(cudaMemcpyAsync(vCpuExecs[i][0].nextAsg_.bits_, formula.ans_.bits_.get(),
      formula.ans_.nQwords_ * sizeof(uint64_t), cudaMemcpyHostToDevice, cas[i].cs_
    ));
    gpuErrchk(cudaMemcpyAsync(
      pgis[i].execs_.Get(), vCpuExecs[i].get(), pgis[i].execs_.Count() * sizeof(GpuExec),
      cudaMemcpyHostToDevice, cas[i].cs_
    ));
    seenAsgs[i].Marshal(pgis[i].gr_);
    gpuErrchk(cudaMemcpyToSymbolAsync(
      gSeenAsgs, &pgis[i].gr_, sizeof(pgis[i].gr_), 0, cudaMemcpyHostToDevice, cas[i].cs_));
  }
  #pragma omp parallel for num_threads(nGpus)
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    int nBlocksPerSM = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocksPerSM, &ReplicateAssignment, kThreadsPerBlock, 0));
    // This is the upper bound for now without the correction for the actually available VRAM
    int totBlocks = nBlocksPerSM * cas[i].cdp_.multiProcessorCount;
    ReplicateAssignment<<<totBlocks, kThreadsPerBlock, 0, cas[i].cs_>>>(
      pgis[i].execs_.Get(), pgis[i].execs_.Count());
  }
  #pragma omp parallel for num_threads(nGpus)
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaStreamSynchronize(cas[i].cs_));
    vCpuExecs[i].reset(); // release the memory
  }
  vCpuExecs.clear();

  sysShar.Get()->trav_.dfsAsg_ = dfsAsg.Marshal();
  sysShar.Get()->trav_.syncDfs_ = 0;
  sysShar.Get()->nGlobalUnsat_ = bestInitNUnsat;
  sysShar.Get()->nUnsatExecs_ = 0;
  CudaArray<__uint128_t> solRRasgs( (1u<<kL2SolRoundRobin) * uint64_t(nVectsPerVarsBV), CudaArrayType::Pinned );
  sysShar.Get()->firstSolRR_ = sysShar.Get()->limitSolRR_ = 0;
  sysShar.Get()->solRRasgs_ = solRRasgs.Get();
  sysShar.Get()->solRRnsUnsat_ = solRRnsUnsat.Get();
  sysShar.Get()->syncRR_ = 0;

  std::cout << "Running on GPU(s)" << std::endl;
  std::atomic<VciGpu> bestNUnsat = bestInitNUnsat;
  std::thread solUpdater([&] {
    BitVector asg(formula.nVars_ + 1);
    VciGpu nUnsat;
    while(bestNUnsat > 0) {
      if(!sysShar.Get()->Consume(asg, nUnsat)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        continue;
      }
      if(nUnsat < bestNUnsat) {
        bestNUnsat = nUnsat;
        formula.ans_ = asg;
      }
    }
  });

  VciGpu nStartUnsat = bestNUnsat;
  auto reportStats = [&] {
    auto tmEnd = std::chrono::steady_clock::now();
    nStartUnsat = bestNUnsat;
    double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
    double clausesPerSec = (prevNUnsat - nStartUnsat) / nSec;
    std::cout << "\n\tUnsatisfied clauses: " << nStartUnsat << " - elapsed " << nSec << " seconds, ";
    if(clausesPerSec >= 1 || clausesPerSec == 0) {
      std::cout << clausesPerSec << " clauses per second.";
    } else {
      std::cout << 1.0 / clausesPerSec << " seconds per clause.";
    }
    std::cout << " Time since very start: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmVeryStart).count() / (60 * 1e9)
      << " minutes." << std::endl;
    tmStart = tmEnd;
    prevNUnsat = nStartUnsat;
  };

  while(bestNUnsat > 0) {
    reportStats();
    #pragma omp parallel for num_threads(nGpus)
    for(int i=0; i<nGpus; i++) {
      gpuErrchk(cudaSetDevice(i));
      assert(pgis[i].nStepBlocks_ * kThreadsPerBlock == pgis[i].execs_.Count());
      StepKernel<<<pgis[i].nStepBlocks_, kThreadsPerBlock, 0, cas[i].cs_>>>(
        nStartUnsat, sysShar.Get(), pgis[i].execs_.Get() );
      gpuErrchk(cudaGetLastError());
    }
    #pragma omp parallel for num_threads(nGpus)
    for(int i=0; i<nGpus; i++) {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaStreamSynchronize(cas[i].cs_));
    }
    if(sysShar.Get()->nUnsatExecs_ == totExecs) {
      maybeSat = false;
      break;
    }
    sysShar.Get()->nUnsatExecs_ = 0;
  }
  reportStats();
  solUpdater.join();

  if(nStartUnsat == 0) {
    std::cout << "SATISFIED" << std::endl;
  } else if(maybeSat) {
    std::cout << "UNKNOWN" << std::endl;
  } else {
    std::cout << "UNSATISFIABLE" << std::endl;
  }
  return 0;
}

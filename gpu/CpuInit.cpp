#include "CpuInit.h"

#include "../SatTracker.h"
#include "../Traversal.h"

VciGpu CpuInit(Formula& formula) {
  std::cout << "Choosing the initial assignment..." << std::endl;
  std::atomic<VciGpu> bestInitNUnsat = formula.nClauses_ + 1;

  Traversal trav;
  std::mutex muGlobal;

  const uint32_t nCpusPerInit = DivUp<uint32_t>(nSysCpus, 3);
  omp_set_max_active_levels(2);
  #pragma omp parallel for schedule(dynamic, 1)
  for(int j=-1; j<=1; j++) {
    omp_set_max_active_levels(2);

    BitVector initAsg(formula.nVars_+1);
    DefaultSatTracker initSatTr(formula);
    switch(j) {
    case -1:
      break; // already all false
    case 0:
      initAsg.Randomize();
      break;
    case 1:
      initAsg.SetTrue();
      break;
    }
    VCTrackingSet initUnsatClauses = initSatTr.Populate(initAsg, nullptr);
    trav.OnSeenAssignment(initAsg, initUnsatClauses.Size());
    {
      std::unique_lock<std::mutex> lock(muGlobal);
      if(initUnsatClauses.Size() < bestInitNUnsat) {
        bestInitNUnsat = initUnsatClauses.Size();
      }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for(uint32_t i=0; i<nCpusPerInit; i++) {
      //std::mt19937_64 rng = GetSeededRandom();
      BitVector locAsg = initAsg;
      DefaultSatTracker locSatTr = initSatTr;
      VCTrackingSet locUnsatClauses = initUnsatClauses;
      VCTrackingSet locFront = locUnsatClauses;
      VCTrackingSet revVars;
      const VCTrackingSet startFront = locFront;
      int64_t nCombs = 0;
      while(nCombs < locSatTr.MaxCombs()) {
        const int8_t sortType = int(i % knSortTypes) + kMinSortType; //rng() % knSortTypes + kMinSortType;
        if(locFront.Size() == 0 || trav.IsSeenFront(locFront, locUnsatClauses)) {
          std::cout << "%";
          locFront = locUnsatClauses;
        }
        bool moved = false;
        VciGpu locBest = bestInitNUnsat.load(std::memory_order_relaxed);
        const int64_t altNUnsat = locSatTr.GradientDescend(
          trav, false, &locFront, moved, locAsg, sortType,
          //locSatTr.NextUnsatCap(nCombs, locUnsatClauses, locBest),
          locBest,
          nCombs, locSatTr.MaxCombs(), initUnsatClauses, locUnsatClauses, startFront, locFront, revVars,
          locBest
        );
        if(!moved) {
          break;
        }
        locBest = bestInitNUnsat.load(std::memory_order_relaxed);
        if(altNUnsat < locBest && locUnsatClauses != initUnsatClauses) {
          std::unique_lock<std::mutex> lock(muGlobal);
          if(altNUnsat < bestInitNUnsat) {
            bestInitNUnsat = altNUnsat;
            nCombs -= std::min<VCIndex>(locUnsatClauses.Size(), 1<<11);
          }
        }
      }
    }
  }
  formula.ans_ = trav.dfs_.back().assignment_;
  DefaultSatTracker satTr(formula);
  VCTrackingSet unsatClauses = satTr.Populate(formula.ans_, nullptr);
  std::vector<MultiItem<VCIndex>> varFront = formula.ClauseFrontToVars(unsatClauses, formula.ans_);
  VCTrackingSet affectedClauses;
  for(VCIndex i=0; i<VCIndex(varFront.size()); i++) {
    const VCIndex aVar = varFront[i].item_;
    assert(1 <= aVar && aVar <= formula.nVars_);
    for(int8_t sign=-1; sign<=1; sign+=2) {
      const VCIndex nArcs = formula.var2clause_.ArcCount(aVar, sign);
      for(VCIndex j=0; j<nArcs; j++) {
        const VCIndex aClause = formula.var2clause_.GetTarget(aVar, sign, j) * sign;
        assert(1 <= aClause && aClause <= formula.nClauses_);
        affectedClauses.Add(aClause);
      }
    }
  }
  std::cout << "|unsatClauses|=" << unsatClauses.Size() << ", |varFront|=" << varFront.size()
    << ", |affectedClauses|=" << affectedClauses.Size() << std::endl;
  return bestInitNUnsat;
}
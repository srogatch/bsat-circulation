#pragma once

#include <random>

#include "SatTracker.h"
#include "Traversal.h"

struct Exec {
  std::vector<MultiItem<VCIndex>> varFront_;
  std::mt19937_64 rng_;
  DefaultSatTracker satTr_;
  BitVector next_;
  VCTrackingSet unsatClauses_ = true;
  VCTrackingSet front_;
  const Formula* pFormula_;
  VCIndex nStartUnsat_;
  VCIndex nIncl_;

  Exec() : rng_(GetSeededRandom()) { }

  void RandomizeFront(Traversal& trav, const bool allowStepBack) {
    constexpr const int cAddLog2 = 3;
    std::vector<VCIndex> vUnsatCs = unsatClauses_.ToVector();
    if(vUnsatCs.size() <= std::log2(nSysCpus)+cAddLog2) {
      std::shuffle(vUnsatCs.begin(), vUnsatCs.end(), rng_);
      for(uint64_t iComb=0, limComb=(1ULL<<vUnsatCs.size()); iComb<limComb; iComb++) {
        front_.Clear();
        for(int j=1; j<int(vUnsatCs.size()); j++) {
          if(iComb & (1ULL<<j)) {
            front_.Add(vUnsatCs[j]);
          }
        }
        if(!trav.IsSeenFront(front_, front_)) {
          return;
        }
      }
    }
    else {
      for(uint64_t j=0; j<nSysCpus<<cAddLog2; j++) {
        front_.Clear();
        const VCIndex nUseClauses = rng_() % (vUnsatCs.size()-1) + 1;
        for(VCIndex i=0; i<nUseClauses; i++) {
          const VCIndex fellow = rng_() % (nUseClauses-i) + i;
          std::swap(vUnsatCs[i], vUnsatCs[fellow]);
          front_.Add(vUnsatCs[i]);
        }
        if(!trav.IsSeenFront(front_, front_)) {
          return;
        }
      }
    }
    if(allowStepBack && trav.StepBack(next_)) {
      unsatClauses_ = satTr_.Populate(next_, &front_);
    } else {
      front_ = unsatClauses_;
    }
  }
};

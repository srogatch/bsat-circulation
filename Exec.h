#pragma once

#include <random>

#include "SatTracker.h"

struct Exec {
  std::vector<MultiItem<VCIndex>> varFront_;
  std::mt19937_64 rng_;
  DefaultSatTracker satTr_;
  BitVector next_;
  VCTrackingSet unsatClauses_ = true;
  VCTrackingSet front_;
  const Formula* pFormula_;
  VCIndex nStartUnsat_;
  int8_t nIncl_;
  int8_t sortType_;

  Exec() : rng_(GetSeededRandom()) { }

  void RandomizeFront() {
    if(unsatClauses_.Size() <= 1) {
      front_ = unsatClauses_;
      return;
    }
    std::vector<VCIndex> vUnsatCs = unsatClauses_.ToVector();
    front_.Clear();
    const VCIndex nUseClauses = rng_() % (vUnsatCs.size()-1) + 1;
    for(VCIndex i=0; i<nUseClauses; i++) {
      const VCIndex fellow = rng_() % (nUseClauses-i) + i;
      std::swap(vUnsatCs[i], vUnsatCs[fellow]);
      front_.Add(vUnsatCs[i]);
    }
  }
};

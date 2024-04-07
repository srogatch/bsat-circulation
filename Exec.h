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

  Exec(const Formula& formula) : rng_(GetSeededRandom()), pFormula_(&formula) { }

  void RandomizeFront() {
    std::vector<MultiItem<VCIndex>> baseVarFront = pFormula_->ClauseFrontToVars(unsatClauses_, next_);
    const VCIndex nUseVars = DivUp(baseVarFront.size(), 2);
    for(VCIndex i=0; i<nUseVars; i++) {
      const VCIndex fellow = rng_() % (nUseVars-i) + i;
      std::swap(baseVarFront[i], baseVarFront[fellow]);
    }
    baseVarFront.resize(nUseVars);
    front_ = pFormula_->VarFrontToClauses(baseVarFront, next_);
  }
};

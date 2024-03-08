#pragma once

#include "Graph.h"
#include "CNF.h"

struct Reduction {
  Formula formula_;
  Graph fGraph_; // formula graph
  int64_t iRoot_;

  explicit Reduction(const Formula& src) : formula_(src) {
    iRoot_ = formula_.nVars_ + formula_.nClauses_ + 1;
    // Edges between variables and their negations
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      fGraph_.AddReplace(Arc(-i, i, 0, formula_.nClauses_));
      fGraph_.AddReplace(Arc(i, -i, 0, formula_.nClauses_));
    }
    // Edges between clauses and the root
    for(int64_t i=1; i<=formula_.nClauses_; i++) {
      fGraph_.AddReplace(Arc(formula_.nVars_ + i, iRoot_, 1, formula_.nClauses_));
      fGraph_.AddReplace(Arc(iRoot_, formula_.nVars_ + i, 1, formula_.nClauses_));
    }
    // Edges between variables/negations, and the clauses
    for(const auto& clause : formula_.clause2var_) {
      for(const int64_t iVar : clause.second) {
        fGraph_.AddReplace(Arc(clause.first + formula_.nVars_, iVar, 0, formula_.nClauses_));
        fGraph_.AddReplace(Arc(iVar, clause.first + formula_.nVars_, 0, formula_.nClauses_));
      }
    }
  }
};

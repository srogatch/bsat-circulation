#pragma once

#include "Graph.h"
#include "CNF.h"

struct Reduction {
  Formula formula_;
  Graph fGraph_; // formula graph
  int64_t halfNVertices_;

  explicit Reduction(const Formula& src) : formula_(src) {
    halfNVertices_ = formula_.nVars_ + formula_.nClauses_;

    // Undirected edges between variables and their negations
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      fGraph_.AddReplace(Arc(-i, i, 0, formula_.nClauses_));
      fGraph_.AddReplace(Arc(i, -i, 0, formula_.nClauses_));
    }

    // Directed clause edges
    for(int64_t i=1; i<=formula_.nClauses_; i++) {
      fGraph_.AddReplace(Arc(-formula_.nVars_ - i, formula_.nVars_ + i, 1, formula_.nVars_));
    }

    // Edges between variables/negations, and the clauses
    for(const auto& clause : formula_.clause2var_) {
      for(const int64_t iVar : clause.second) {
        fGraph_.AddReplace(Arc(iVar, -formula_.nVars_ - clause.first, 1, formula_.nVars_));
        fGraph_.AddReplace(Arc(formula_.nVars_ + clause.first, -iVar, 1, formula_.nVars_));
      }
    }
  }
};

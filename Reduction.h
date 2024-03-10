#pragma once

#include "Graph.h"
#include "CNF.h"
#include "MaxFlow.h"

struct Reduction {
  Formula formula_;
  Graph fGraph_; // formula graph
  int64_t iST_; // -iST_=s: source vertex; +iST_=t: sink vertex
  int64_t necessaryFlow_;

  explicit Reduction(const Formula& src) : formula_(src) {
    iST_ = formula_.nVars_ + formula_.nClauses_ + 1;

    // Undirected edges between variables and their negations
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      fGraph_.AddMerge(Arc(-i, i, 0, formula_.nClauses_));
      fGraph_.AddMerge(Arc(i, -i, 0, formula_.nClauses_));
    }

    // Directed clause edges
    for(int64_t i=1; i<=formula_.nClauses_; i++) {
      fGraph_.AddMerge(Arc(-formula_.nVars_ - i, formula_.nVars_ + i, 1, formula_.nVars_));
    }

    // Edges between variables/negations, and the clauses
    for(const auto& clause : formula_.clause2var_) {
      for(const int64_t iVar : clause.second) {
        fGraph_.AddMerge(Arc(iVar, -formula_.nVars_ - clause.first, 0, 1));
        fGraph_.AddMerge(Arc(formula_.nVars_ + clause.first, -iVar, 0, 1));
      }
    }
  }

  int64_t GetVSource() const { return -iST_; }
  int64_t GetVSink() const { return iST_; }

  bool Circulate() {
    // Reduce the circulation problem to the max flow problem
    necessaryFlow_ = 0;
    for(const auto& src : fGraph_.links_) {
      for(const auto& dst : src.second) {
        necessaryFlow_ += dst.second->low_;
        fGraph_.AddMerge(Arc(GetVSource(), dst.first, 0, dst.second->low_));
        fGraph_.AddMerge(Arc(src.first, GetVSink(), 0, dst.second->low_));
        dst.second->high_ -= dst.second->low_;
        dst.second->low_ = 0;
      }
    }
    MaxFlow mf(fGraph_, GetVSource(), GetVSink());
    if(mf.result_ < necessaryFlow_) {
      return false; // unsatisfiable
    }
    assert(mf.result_ == necessaryFlow_);
    
  }
};

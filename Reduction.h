#pragma once

#include <cassert>

#include "Graph.h"
#include "CNF.h"
#include "MaxFlow.h"

struct Reduction {
  Formula& formula_;
  Graph fGraph_; // formula graph
  int64_t iRoot_;
  int64_t iST_; // -iST_=s: source vertex; +iST_=t: sink vertex
  int64_t necessaryFlow_;

  explicit Reduction(Formula& src) : formula_(src) {
    iRoot_ = formula_.nVars_ + formula_.nClauses_ + 1;
    iST_ = iRoot_ + 1;

    for(auto& clause : formula_.clause2var_) {
      if(src.dummySat_[clause.first]) {
        continue; // don't add arcs for dummy-satisfied clauses
      }
      assert(1 <= int64_t(clause.first) && int64_t(clause.first) <= formula_.nClauses_);
      for(int64_t iVar : clause.second) {
        assert(1 <= llabs(iVar) && llabs(iVar) <= formula_.nVars_);
        if( (iVar < 0 && !formula_.ans_[-iVar]) || (iVar > 0 && formula_.ans_[iVar]) ) {
          // The value assigned to this variable satisfies the clause:
          // An arc from the variable to the clause.
          fGraph_.AddMerge(Arc(llabs(iVar), formula_.nVars_+clause.first, 0, 1));
        } else {
          // Unsatisfying assignment:
          // An arc from the clause to the variable.
          fGraph_.AddMerge(Arc(formula_.nVars_+clause.first, llabs(iVar), 0, 1));
        }
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
        if(dst.second->low_ > 0) {
          necessaryFlow_ += dst.second->low_;
          fGraph_.AddMerge(Arc(GetVSource(), dst.first, 0, dst.second->low_));
          fGraph_.AddMerge(Arc(src.first, GetVSink(), 0, dst.second->low_));
          dst.second->high_ -= dst.second->low_;
          // Use it later when restoring the flow: dst.second->low_ = 0;
        }
      }
    }
    MaxFlow mf(fGraph_, GetVSource(), GetVSink());
    if(mf.result_ < necessaryFlow_) {
      std::cerr << "No circulation" << std::endl;
      return false; // unsatisfiable
    }
    assert(mf.result_ == necessaryFlow_);
    assert(fGraph_.CheckFlow(formula_.nVars_ + formula_.nClauses_));

    // Fix the circulation - add minimal flows and remove source and sink
    for(const auto& src : fGraph_.links_) {
      for(const auto& dst : src.second) {
        dst.second->flow_ += dst.second->low_;
        dst.second->high_ += dst.second->low_;
      }
    }
    std::vector<std::pair<int64_t, int64_t>> toRemove;
    for(const auto& src : fGraph_.links_) {
      for(const auto& dst : src.second) {
        if(src.first == GetVSource() || src.first == GetVSink()
          || dst.first == GetVSource() || dst.first == GetVSink())
        {
          toRemove.emplace_back(src.first, dst.first);
        }
      }
    }
    for(const auto& curArc : toRemove) {
      fGraph_.Remove(curArc.first, curArc.second);
    }
    assert(fGraph_.CheckFlow(formula_.nVars_ + formula_.nClauses_));
    return true; // there is a circulation, but maybe the formula is still unsatisfiable if there are contradictions
  }

  bool AssignVars() {
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      for(const auto& dst : fGraph_.links_[i]) {
        if(dst.second->flow_ > 0) {
          formula_.ans_.Flip(i);
          break;
        }
      }
    }
    return true;
  }
};

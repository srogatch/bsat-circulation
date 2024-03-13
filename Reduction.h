#pragma once

#include "Graph.h"
#include "CNF.h"
#include "MaxFlow.h"

struct Reduction {
  Formula& formula_;
  Graph fGraph_; // formula graph
  int64_t iST_; // new -iST_=s: source vertex; +iST_=t: sink vertex (for the graph without minimal flows)
  int64_t iSoTo_; // old source (-iSoTo_) and sink (+iSoTo_) (for the graph with minimal flows)
  int64_t necessaryFlow_;

  explicit Reduction(Formula& src) : formula_(src) {
    iSoTo_ = formula_.nVars_ + formula_.nClauses_ + 1;
    iST_ = iSoTo_ + 1;

    necessaryFlow_ = formula_.nClauses_;
    // for(int64_t i=1; i<=formula_.nVars_; i++) {
    //   fGraph_.AddMerge(Arc(-i, i, 0, 1));
    // }
    for(auto& clause : formula_.clause2var_) {
      int64_t nSat = 0;
      for(int64_t iVar : clause.second) {
        if( (iVar < 0 && !formula_.ans_[-iVar]) || (iVar > 0 && formula_.ans_[iVar]) ) {
          nSat++;
          // TODO: the high flow can be lower here - equal to the number of unsatisfied clauses for this vertex
          fGraph_.AddMerge(Arc(formula_.nVars_+clause.first, llabs(iVar), 0, formula_.nClauses_));
        } else {
          fGraph_.AddMerge(Arc(llabs(iVar), formula_.nVars_+clause.first, 0, 1));
        }
      }
      if(nSat >= 1) {
        fGraph_.AddMerge(Arc(-iSoTo_, formula_.nVars_+clause.first, 1, clause.second.size()));
        for(int64_t iVar : clause.second) {
          fGraph_.AddMerge(Arc(llabs(iVar), -formula_.nVars_-clause.first));
        }
        fGraph_.AddMerge(Arc(-formula_.nVars_-clause.first, iSoTo_, 1, formula_.nClauses_));
      } else {
        fGraph_.AddMerge(Arc(formula_.nVars_+clause.first, iSoTo_, 1, clause.second.size()));
      }
    }
  }

  int64_t GetVSource() const { return -iST_; }
  int64_t GetVSink() const { return iST_; }

  bool SendFlow() {
    // Reduce to the max flow problem
    for(const auto& src : fGraph_.links_) {
      for(const auto& dst : src.second) {
        if(dst.second->low_ > 0) {
          fGraph_.AddMerge(Arc(GetVSource(), dst.first, 0, dst.second->low_));
          fGraph_.AddMerge(Arc(src.first, GetVSink(), 0, dst.second->low_));
          dst.second->high_ -= dst.second->low_;
        }
      }
    }
    // An arc from the old sink to the old source
    fGraph_.AddMerge(Arc(iSoTo_, -iSoTo_));

    MaxFlow mf(fGraph_, GetVSource(), GetVSink());
    if(mf.result_ < necessaryFlow_) {
      std::cerr << "Not enough flow." << std::endl;
      return false; // unsatisfiable
    }
    assert(fGraph_.CheckFlow(formula_.nVars_ + formula_.nClauses_ + 1));

    // Fix the graph - add minimal flows and remove new source and sink
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
    fGraph_.Remove(iSoTo_, -iSoTo_);
    assert(fGraph_.CheckFlow(formula_.nVars_ + formula_.nClauses_));
    return true; // there is a circulation, but maybe the formula is still unsatisfiable if there are contradictions
  }

  bool AssignVars(int64_t& nAssigned) {

    for(int64_t i=1; i<=formula_.nVars_; i++) {
      for(const auto& dst : fGraph_.links_[i]) {
        if(dst.second->flow_ > 0) {
          formula_.ans_[i] = !formula_.ans_[i];
          break;
        }
      }
    }
    return true;
  }
};

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
    // for(int64_t i=1; i<=formula_.nVars_; i++) {
    //   fGraph_.AddMerge(Arc(-i, i, 0, formula_.nClauses_));
    //   fGraph_.AddMerge(Arc(i, -i, 0, formula_.nClauses_));
    // }

    // for(int64_t i=1; i<=formula_.nClauses_; i++) {
    //   fGraph_.AddMerge(Arc(-formula_.nVars_ - i, formula_.nVars_ + i, 1, formula_.nVars_));
    // }

    for(const auto& clause : formula_.clause2var_) {
      // Directed clause edges
      fGraph_.AddMerge(Arc(
        -formula_.nVars_ - clause.first, formula_.nVars_ + clause.first,
        clause.second.size(), formula_.nVars_));
      for(const int64_t iVar : clause.second) {
        // Edges between variables/negations, and the clauses
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
        if(dst.second->low_ > 0) {
          necessaryFlow_ += dst.second->low_;
          fGraph_.AddMerge(Arc(GetVSource(), dst.first, 0, dst.second->low_));
          fGraph_.AddMerge(Arc(src.first, GetVSink(), 0, dst.second->low_));
          dst.second->high_ -= dst.second->low_;
          dst.second->low_ = 0;
        }
      }
    }
    MaxFlow mf(fGraph_, GetVSource(), GetVSink());
    if(mf.result_ < necessaryFlow_) {
      return false; // unsatisfiable
    }
    assert(mf.result_ == necessaryFlow_);

    // Fix the circulation - add minimal flows and remove source and sink
    for(const auto& clause : formula_.clause2var_) {
      // Directed clause edges
      fGraph_.Get(
        -formula_.nVars_ - clause.first, formula_.nVars_ + clause.first)
        ->flow_ += clause.second.size();
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
    return true; // there is a circulation, but maybe the formula is still unsatisfiable if there are contradictions
  }

  int64_t vStart_;
  
  void DfsRemoveFlow(const int64_t eSrc, const int64_t eDst) {
    std::shared_ptr<Arc> arc = fGraph_.Get(eSrc, eDst);
    assert(arc->flow_ > 0);
    arc->flow_--;
    if(arc->flow_ == 0) {
      // Reduce the computational complexity of the further loop searches
      fGraph_.Remove(eSrc, eDst);
    }
    if(eDst == vStart_) { // closed loop
      return;
    }
    for(const auto& vAfter : fGraph_.links_[eDst]) {
      if(vAfter.second->flow_ == 0) {
        continue;
      }
      DfsRemoveFlow(eDst, vAfter.first);
      return;
    }
    assert(false); // loop not closed
  }

  std::vector<bool> AssignVars() {
    std::vector<bool> ans(formula_.nVars_ + 1);
    std::vector<bool> known(formula_.nVars_ + 1);
    ans[0] = true;
    for(const auto& clause : formula_.clause2var_) {
      for(const auto& iVar : clause.second) {
        //std::shared_ptr<Arc> aTrue = fGraph_.Get(iVar, -clause.first-formula_.nVars_);
        std::shared_ptr<Arc> aFalse = fGraph_.Get(formula_.nVars_ + clause.first, -iVar);
        //assert( (aTrue == nullptr && aFalse == nullptr) || (aTrue != nullptr && aFalse != nullptr) );
        if(aFalse == nullptr) {
          continue; // Removed in a previous DFS
        }
        if(aFalse->flow_ > 0) {
          const bool varVal = (iVar < 0) ? false : true;
          if(known[llabs(iVar)]) {
            if(ans[llabs(iVar)] != varVal) {
              // Unsatisfiable? Broken invariant?
              ans.resize(1);
              return ans;
            }
          } else {
            known[llabs(iVar)] = true;
            ans[llabs(iVar)] = varVal;
          }
          vStart_ = formula_.nVars_ + clause.first;
          DfsRemoveFlow(formula_.nVars_ + clause.first, -iVar);
          break;
        }
      }
    }
    for(int64_t i=1; i<known.size(); i++) {
      if(!known[i]) {
        ans[0] = false; // Uncertain
      }
    }
    assert(formula_.SolWorks(ans));
    return ans;
  }
};

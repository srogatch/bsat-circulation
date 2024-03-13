#pragma once

#include "Graph.h"
#include "CNF.h"
#include "MaxFlow.h"

struct Reduction {
  Formula& formula_;
  Graph fGraph_; // formula graph
  int64_t iST_; // -iST_=s: source vertex; +iST_=t: sink vertex
  int64_t necessaryFlow_;

  explicit Reduction(Formula& src) : formula_(src) {
    iST_ = formula_.nVars_ + formula_.nClauses_ + 1;

    // Undirected edges between variables and their negations
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      fGraph_.AddMerge(Arc(-i, i, 0, formula_.nClauses_));
      fGraph_.AddMerge(Arc(i, -i, 0, formula_.nClauses_));
    }

    // for(int64_t i=1; i<=formula_.nClauses_; i++) {
    //   fGraph_.AddMerge(Arc(-formula_.nVars_ - i, formula_.nVars_ + i, 1, formula_.nVars_));
    // }

    for(const auto& clause : formula_.clause2var_) {
      // Directed clause edges
      fGraph_.AddMerge(Arc(
        -formula_.nVars_ - clause.first, formula_.nVars_ + clause.first, clause.second.size(), clause.second.size()));
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
    for(const auto& clause : formula_.clause2var_) {
      // Directed clause edges
      auto pArc = fGraph_.Get(
        -formula_.nVars_ - clause.first, formula_.nVars_ + clause.first);
      pArc->flow_ += pArc->low_;
      pArc->high_ += pArc->low_;
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

  bool AssignVars(int64_t& nAssigned) {
    // Reflow through the variable<->negation edges
    for(const auto& clause : formula_.clause2var_) {
      std::shared_ptr<Arc> aClause = fGraph_.Get(-clause.first-formula_.nVars_, clause.first+formula_.nVars_);
      for(const auto& iVar : clause.second) {
        if(aClause->flow_ <= 1) {
          break; // Everything is assigned for this clause
        }
        std::shared_ptr<Arc> aVarTrue = fGraph_.Get(-iVar, iVar);
        std::shared_ptr<Arc> aVarFalse = fGraph_.Get(iVar, -iVar);
        std::shared_ptr<Arc> aTrue = fGraph_.Get(iVar, -clause.first-formula_.nVars_);
        std::shared_ptr<Arc> aFalse = fGraph_.Get(clause.first+formula_.nVars_, -iVar);
        if(aTrue->flow_ > 0 && aFalse->flow_ > 0) {
          aTrue->flow_--;
          aFalse->flow_--;
          aClause->flow_--;
          if( aVarTrue->flow_ > 0 ) {
            aVarTrue->flow_--;
          } else {
            aVarFalse->flow_++;
          }
        }
      }
    }

    nAssigned = 0;
    // Traverse the unambiguous and contradicting flows
    for(int64_t i=1; i<=formula_.nVars_; i++) {
      std::shared_ptr<Arc> aTrue = fGraph_.Get(-i, i);
      std::shared_ptr<Arc> aFalse = fGraph_.Get(i, -i);
      const int64_t contradiction = std::min(aTrue->flow_, aFalse->flow_);
      if(contradiction > 0) {
        std::cerr << "Contradiction in within-var flows." << std::endl;
        return false;
      }
      if(aTrue->flow_) {
        formula_.known_[i] = true;
        formula_.ans_[i] = true;
        nAssigned++;
        continue;
      }
      if(aFalse->flow_) {
        formula_.known_[i] = true;
        formula_.ans_[i] = false;
        nAssigned++;
        continue;
      }
    }

    return true;
  }
};

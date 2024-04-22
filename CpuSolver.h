#pragma once

#include "SatTracker.h"
#include "Traversal.h"

struct IneqSystem {
  std::vector<std::vector<double>> A_;
  std::vector<double> c_;
  std::vector<VCIndex> clauseMap_;
  std::vector<VCIndex> varMap_;

  IneqSystem(
    const std::vector<MultiItem<VCIndex>>& varFront,
    const VCTrackingSet& affectedClauses)
  {
    clauseMap_ = affectedClauses.ToVector();
    varMap_.reserve(varFront.size());
    for(VCIndex i=0; i<VCIndex(varFront.size()); i++) {
      varMap_.push_back(varFront[i].item_);
    }

    A_.resize(clauseMap_.size());
    for(VCIndex i=0; i<VCIndex(clauseMap_.size()); i++) {
      A_[i].resize(varMap_.size(), 0);
    }

    c_.resize(clauseMap_.size(), 0);
  }

  VCIndex VarCount() const {
    return varMap_.size();
  }

  VCIndex ClauseCount() const {
    return clauseMap_.size();
  }
};

struct CpuSolver {
  Formula *pFormula_;

  explicit CpuSolver(Formula& formula) : pFormula_(&formula) { }

  bool Solve() {
    DefaultSatTracker satTr(*pFormula_);
    VCTrackingSet unsatClauses = satTr.Populate(pFormula_->ans_, nullptr);
    std::vector<MultiItem<VCIndex>> varFront = pFormula_->ClauseFrontToVars(unsatClauses, pFormula_->ans_);
    VCTrackingSet affectedClauses;
    for(VCIndex i=0; i<VCIndex(varFront.size()); i++) {
      const VCIndex aVar = varFront[i].item_;
      assert(1 <= aVar && aVar <= pFormula_->nVars_);
      for(int8_t sign=-1; sign<=1; sign+=2) {
        const VCIndex nArcs = pFormula_->var2clause_.ArcCount(aVar, sign);
        for(VCIndex j=0; j<nArcs; j++) {
          const VCIndex aClause = pFormula_->var2clause_.GetTarget(aVar, sign, j) * sign;
          assert(1 <= aClause && aClause <= pFormula_->nClauses_);
          affectedClauses.Add(aClause);
        }
      }
    }
    std::cout << "|unsatClauses|=" << unsatClauses.Size() << ", |varFront|=" << varFront.size()
      << ", |affectedClauses|=" << affectedClauses.Size() << std::endl;

    IneqSystem ieSys(varFront, affectedClauses);
    for(VCIndex i=0; i<ieSys.ClauseCount(); i++) {
      const VCIndex aClause = ieSys.clauseMap_[i];
      ieSys.c_[i] = satTr.nSat_[aClause];
      for(VCIndex j=0; j<ieSys.VarCount(); j++) {
        const VCIndex aVar = ieSys.varMap_[j];
        if(pFormula_->clause2var_.HasArc(aClause, aVar)) {
          ieSys.A_[i][j] = 1;
        } else if(pFormula_->clause2var_.HasArc(aClause, -aVar)) {
          ieSys.A_[i][j] = -1;
        }
      }
    }

    constexpr const double eps = 1e-10;

    for(VCIndex i=0; i<ieSys.VarCount(); i++) {
      VCIndex j=i;
      for(; j<ieSys.ClauseCount(); j++) {
        if(fabs(ieSys.A_[j][i]) > eps) {
          break;
        }
      }
      if(j >= ieSys.ClauseCount()) {
        continue;
      }
      if(i != j) {
        std::swap(ieSys.c_[i], ieSys.c_[j]);
        ieSys.A_[i].swap(ieSys.A_[j]);
        std::swap(ieSys.clauseMap_[i], ieSys.clauseMap_[j]);
      }
      for(j=0; j<ieSys.ClauseCount(); j++) {
        if(j == i || fabs(ieSys.A_[j][i]) <= eps) {
          continue;
        }
        if(Signum(ieSys.A_[j][i]) == Signum(ieSys.A_[i][i])) {
          // No problem to assign
          continue;
        }
        const double mul = - ieSys.A_[j][i] / ieSys.A_[i][i];
        for(VCIndex k=0; k<ieSys.VarCount(); k++) {
          if(k == i) {
            ieSys.A_[j][k] = 0; // Avoid floating-point flaws / epsilons
            continue;
          }
          ieSys.A_[j][k] += mul * ieSys.A_[i][k];
        }
        ieSys.c_[j] += mul * ieSys.c_[i];
      }
    }

    for(VCIndex i=0; i<ieSys.VarCount(); i++) {
      if(i >= ieSys.ClauseCount()) {
        break; // every variable is assigned
      }
      VCIndex j=0;
      for(; j<ieSys.ClauseCount(); j++) {
        if(fabs(ieSys.A_[j][i]) > eps) {
          break;
        }
      }
      if(j >= ieSys.ClauseCount()) {
        continue; // The assignment of this variable doesn't matter
      }
      const int8_t sign = Signum(ieSys.A_[j][i]);
      const bool setTrue = (sign > 0);
      if(pFormula_->ans_[ieSys.varMap_[i]] != setTrue) {
        pFormula_->ans_.Flip(ieSys.varMap_[i]);
      }
      for(; j<ieSys.ClauseCount(); j++) {
        ieSys.c_[j] -= ieSys.A_[j][i] * sign;
      }
    }
    for(VCIndex i=0; i<ieSys.ClauseCount(); i++) {
      if(ieSys.c_[i] > eps) {
        return false; // Unsatisfiable
      }
    }
    assert(pFormula_->SolWorks());
    return true;
  }
};

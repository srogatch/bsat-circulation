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

  explicit IneqSystem(const Formula& formula)
  {
    clauseMap_.reserve(formula.nClauses_);
    for(VCIndex i=1; i<=formula.nClauses_; i++) {
      clauseMap_.push_back(i);
    }
    varMap_.reserve(formula.nVars_);
    for(VCIndex i=1; i<=formula.nVars_; i++) {
      varMap_.push_back(i);
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

    //IneqSystem ieSys(varFront, affectedClauses);
    IneqSystem ieSys(*pFormula_);

    #pragma omp parallel for
    for(VCIndex i=0; i<ieSys.ClauseCount(); i++) {
      const VCIndex aClause = ieSys.clauseMap_[i];
      assert(1 <= aClause && aClause <= pFormula_->nClauses_);
      ieSys.c_[i] = 2 - pFormula_->clause2var_.ArcCount(aClause); // - double(satTr.nSat_[aClause]);
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
      const double rev = 1.0 / fabs(ieSys.A_[i][i]);
      for(VCIndex j=0; j<ieSys.VarCount(); j++) {
        ieSys.A_[i][j] *= rev;
      }
      ieSys.c_[i] *= rev;
      for(j=0; j<ieSys.ClauseCount(); j++) {
        if(j == i || fabs(ieSys.A_[j][i]) <= eps) {
          continue;
        }
        if(Signum(ieSys.A_[j][i]) == Signum(ieSys.A_[i][i])) {
          // No problem to assign
          continue;
        }
        const double mul = - ieSys.A_[j][i] / ieSys.A_[i][i];
        assert(mul >= 0);
        #pragma omp parallel for num_threads(nSysCpus) schedule(guided, kRamPageBytes/sizeof(double))
        for(VCIndex k=0; k<ieSys.VarCount(); k++) {
          if(k == i) {
            assert( fabs(ieSys.A_[j][k] + mul * ieSys.A_[i][k]) <= eps );
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
      #pragma omp parallel for num_threads(nSysCpus) schedule(guided, kRamPageBytes/sizeof(double))
      for(VCIndex k=0; k<ieSys.ClauseCount(); k++) {
        assert(Signum(ieSys.A_[k][i]) == sign || Signum(ieSys.A_[k][i]) == 0);
        ieSys.c_[k] -= ieSys.A_[k][i] * sign;
        ieSys.A_[k][i] = 0;
      }
    }
    for(VCIndex i=0; i<ieSys.ClauseCount(); i++) {
      if(ieSys.c_[i] > eps) {
        std::cout << "UNSATISFIABLE" << std::endl;
        return false; // Unsatisfiable
      }
    }
    std::cout << "SATISFIABLE" << std::endl;

    std::atomic<bool> allSat = true;
    #pragma omp parallel for schedule(guided, kRamPageBytes)
    for(VCIndex iClause=1; iClause<=pFormula_->nClauses_; iClause++) {
      #pragma omp cancellation point for
      if(pFormula_->dummySat_[iClause]) {
        continue; // satisfied because the clause contains a variable and its negation
      }
      bool satisfied = false;
      #pragma unroll
      for(int8_t sgnTo=-1; sgnTo<=1; sgnTo+=2) {
        const VCIndex nArcs = pFormula_->clause2var_.ArcCount(iClause, sgnTo);
        for(VCIndex at=0; at<nArcs; at++) {
          const VCIndex iVar = pFormula_->clause2var_.GetTarget(iClause, sgnTo, at);
          assert(Signum(iVar) == sgnTo);
          const int8_t sgnAsg = pFormula_->ans_[llabs(iVar)] ? 1 : -1;
          if(sgnAsg == sgnTo) {
            satisfied = true;
            break;
          }
        }
      }
      if(!satisfied)
      {
        #pragma omp critical
        {
          VCIndex i;
          for(i=0; i<ieSys.ClauseCount(); i++) {
            if(ieSys.clauseMap_[i] == iClause) {
              break;
            }
          }
          std::cout << iClause << "/" << i << ": ";
          for(VCIndex j=0; j<ieSys.VarCount(); j++) {
            if(ieSys.A_[i][j] != 0) {
              std::cout << " (" << ieSys.varMap_[j] << "/" << j << ")" << ieSys.A_[i][j];
            }
          }
          std::cout << " > " << ieSys.c_[i] << std::endl;
        }
        allSat.store(false);
        #pragma omp cancel for
      }
    }
    if(!allSat) {
      std::cout << "Verification failed." << std::endl;
    }
    return true;
  }
};

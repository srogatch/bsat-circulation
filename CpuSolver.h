#pragma once

#include "SatTracker.h"
#include "Traversal.h"
#include "extern/eigen/Eigen/Sparse"
#include <osqp/osqp.h>

struct CpuSolver {
  Formula *pFormula_;

  explicit CpuSolver(Formula& formula) : pFormula_(&formula) { }

  bool Solve() {
    for(VCIndex i=0; i<ieSys.ClauseCount(); i++) {
      const VCIndex aClause = ieSys.clauseMap_[i];
      assert(1 <= aClause && aClause <= pFormula_->nClauses_);
      ieSys.c_[i] = 2 - pFormula_->clause2var_.ArcCount(aClause); // - double(satTr.nSat_[aClause]);
      for(int8_t sign=-1; sign<=1; sign+=2) {
        const VCIndex nArcs = pFormula_->clause2var_.ArcCount(aClause, sign);
        for(VCIndex j=0; j<nArcs; j++) {
          const VCIndex iVar = pFormula_->clause2var_.GetTarget(aClause, sign, j);
          const VCIndex aVar = llabs(iVar);
          assert(ieSys.varMap_[aVar-1] == aVar);
          if(iVar < 0) {
            ieSys.A_[i][aVar-1] = -1;
          } else {
            ieSys.A_[i][aVar-1] = 1;
          }
        }
      }
      // for(VCIndex j=0; j<ieSys.VarCount(); j++) {
      //   const VCIndex aVar = ieSys.varMap_[j];
      //   if(pFormula_->clause2var_.HasArc(aClause, aVar)) {
      //     ieSys.A_[i][j] = 1;
      //   } else if(pFormula_->clause2var_.HasArc(aClause, -aVar)) {
      //     ieSys.A_[i][j] = -1;
      //   }
      // }
    }

    constexpr const double eps = 1e-10;

    std::vector<VCIndex> leadRow(ieSys.VarCount(), -1);
    VCIndex nLeads = 0;
    for(VCIndex i=0; i<ieSys.VarCount(); i++) {
      VCIndex j=nLeads;
      for(; j<ieSys.ClauseCount(); j++) {
        if( fabs(ieSys.A_[j][i]) > eps ) {
          break;
        }
      }
      if(j >= ieSys.ClauseCount()) {
        continue;
      }
      if(nLeads != j) {
        std::swap(ieSys.c_[nLeads], ieSys.c_[j]);
        ieSys.A_[nLeads].swap(ieSys.A_[j]);
        std::swap(ieSys.clauseMap_[nLeads], ieSys.clauseMap_[j]);
      }
      leadRow[i] = nLeads;
      nLeads++;
      const double rev = 1.0 / fabs(ieSys.A_[leadRow[i]][i]);
      #pragma omp parallel for num_threads(nSysCpus) schedule(guided, kRamPageBytes/sizeof(double))
      for(VCIndex j=0; j<ieSys.VarCount(); j++) {
        ieSys.A_[leadRow[i]][j] *= rev;
      }
      ieSys.c_[leadRow[i]] *= rev;
      #pragma omp parallel for num_threads(nSysCpus) schedule(guided, kRamPageBytes/sizeof(double))
      for(j=0; j<ieSys.ClauseCount(); j++) {
        if(j == leadRow[i] || fabs(ieSys.A_[j][i]) <= eps) {
          continue;
        }
        // if(Signum(ieSys.A_[j][i]) != Signum(ieSys.A_[leadRow[i]][i])) {
        //   // No problem to assign
        //   continue;
        // }
        const double mul = - ieSys.A_[j][i] / ieSys.A_[leadRow[i]][i];
        // assert(mul <= 0);
        for(VCIndex k=0; k<ieSys.VarCount(); k++) {
          if(k == i) {
            assert( fabs(ieSys.A_[j][k] + mul * ieSys.A_[leadRow[i]][k]) <= eps );
            ieSys.A_[j][k] = 0; // Avoid floating-point flaws / epsilons
            continue;
          }
          ieSys.A_[j][k] += mul * ieSys.A_[leadRow[i]][k];
        }
        ieSys.c_[j] += mul * ieSys.c_[leadRow[i]];
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
        // if(pFormula_->ans_[ieSys.varMap_[i]] != false) {
        //   pFormula_->ans_.Flip(ieSys.varMap_[i]);
        // }
        continue; // The assignment of this variable doesn't matter
      }
      const int8_t sign = Signum(ieSys.A_[j][i]);
      const bool setTrue = (sign > 0);
      if(pFormula_->ans_[ieSys.varMap_[i]] != setTrue) {
        pFormula_->ans_.Flip(ieSys.varMap_[i]);
      }
      #pragma omp parallel for num_threads(nSysCpus) schedule(guided, kRamPageBytes/sizeof(double))
      for(VCIndex k=0; k<ieSys.ClauseCount(); k++) {
        //assert(Signum(ieSys.A_[k][i]) == sign || Signum(ieSys.A_[k][i]) == 0);
        if(Signum(ieSys.A_[k][i]) == sign) {
          ieSys.c_[k] -= ieSys.A_[k][i] * sign;
          ieSys.A_[k][i] = 0;
        }
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
      }
    }
    if(!allSat) {
      std::cout << "Verification failed." << std::endl;
    }
    return true;
  }
};

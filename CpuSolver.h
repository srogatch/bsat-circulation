#pragma once

#include "SatTracker.h"
#include "Traversal.h"
#include "extern/eigen/Eigen/Sparse"
#include <osqp/osqp.h>

struct SpMatTriple {
  VCIndex row_;
  VCIndex col_;
  double val_;

  SpMatTriple(const VCIndex row, const VCIndex col, const double val) {
    row_ = row;
    col_ = col;
    val_ = val;
  }
};

struct CSCMatrix {
  std::vector<OSQPFloat> values;
  std::vector<OSQPInt> row_indices;
  std::vector<OSQPInt> col_ptrs;
};

// Comparator for sorting triples by column and then by row
bool tripleComparator(const SpMatTriple& a, const SpMatTriple& b) {
    return (a.col_ < b.col_) || (a.col_ == b.col_ && a.row_ < b.row_);
}

// Function to convert triples to CSC format
CSCMatrix triplesToCSC(std::vector<SpMatTriple> &triples, OSQPInt rows, OSQPInt cols)
{
  std::sort(triples.begin(), triples.end(), tripleComparator);

  CSCMatrix csc;
  csc.col_ptrs.resize(cols + 1, 0);

  OSQPInt current_col = 0;
  for (const SpMatTriple &t : triples)
  {
    csc.values.push_back( t.val_ );
    csc.row_indices.push_back(t.row_);

    // Update column pointers
    for (; current_col <= t.col_; ++current_col)
    {
      csc.col_ptrs[current_col] = csc.values.size() - 1;
    }
  }

  // Fill in the remaining column pointers if the last columns are empty
  for (; current_col <= cols; ++current_col)
  {
    csc.col_ptrs[current_col] = csc.values.size();
  }

  return csc;
}

struct CpuSolver {
  Formula *pFormula_;

  explicit CpuSolver(Formula& formula) : pFormula_(&formula) { }

  bool TailSolve(VCTrackingSet& unknowns) {
    std::cout << "Unknown vars: " << unknowns.Size() << std::endl;

    VCIndex nConstraints = 0;
    VCIndex nUnknowns = 0;

    std::vector<OSQPFloat> optL, optH, optQ, initX;
    CSCMatrix optA, optP;
    int64_t nUnsatClauses = 0;
    std::unordered_map<VCIndex, VCIndex> cnfToIneqVars;
    std::vector<VCIndex> vUnknowns = unknowns.ToVector();

    {
      std::vector<SpMatTriple> smtA, smtP;
      // x
      for(VCIndex i=0; i<vUnknowns.size(); i++) {
        const VCIndex aVar = vUnknowns[i];
        cnfToIneqVars[aVar] = i;
        smtP.emplace_back(nUnknowns, nUnknowns, 0);
        smtA.emplace_back(nConstraints, nUnknowns, 1);
        optL.emplace_back( -1 );
        optH.emplace_back( 1 );
        initX.emplace_back( pFormula_->ans_[aVar] ? optH.back() : optL.back() );
        optQ.emplace_back(exp2(-2*i));
        nUnknowns++;
        nConstraints++;
      }

      for(VCIndex aClause=1; aClause<=pFormula_->nClauses_; aClause++) {
        bool satisfied = false;
        int64_t nActive = 0;
        for(int8_t sign=-1; sign<=1 && !satisfied; sign+=2) {
          const VCIndex nArcs = pFormula_->clause2var_.ArcCount(aClause, sign);
          for(VCIndex j=0; j<nArcs; j++) {
            const VCIndex iVar = pFormula_->clause2var_.GetTarget(aClause, sign, j);
            assert(Signum(iVar) == sign);
            const VCIndex aVar = llabs(iVar);
            auto it = cnfToIneqVars.find(aVar);
            if(it == cnfToIneqVars.end()) {
              if( (pFormula_->ans_[aVar] ? 1 : -1) == sign ) {
                satisfied = true;
                break;
              }
            } else {
              const VCIndex iUnknown = it->second;
              smtA.emplace_back(nConstraints, iUnknown, sign);
              nActive++;
            }
          }
        }
        if(satisfied) {
          while(!smtA.empty() && smtA.back().row_ == nConstraints) {
            smtA.pop_back();
          }
        }
        else {
          optL.emplace_back(2 - nActive);
          optH.emplace_back(INFINITY);
          nConstraints++;
          nUnsatClauses++;
        }
      }

      optA = triplesToCSC(smtA, nConstraints, nUnknowns);
      optP = triplesToCSC(smtP, nUnknowns, nUnknowns);
    }

    const double eps = 1e-3; // 0.5 / nUnsatClauses;

    // Create CSC matrices for P and A
    std::unique_ptr<OSQPCscMatrix> osqpP(new OSQPCscMatrix);
    osqpP->m = nUnknowns;
    osqpP->n = nUnknowns;
    osqpP->nz = -1;
    osqpP->nzmax = optP.values.size();
    osqpP->x = optP.values.data();
    osqpP->i = optP.row_indices.data();
    osqpP->p = optP.col_ptrs.data();

    std::unique_ptr<OSQPCscMatrix> osqpA(new OSQPCscMatrix);
    osqpA->m = nConstraints;
    osqpA->n = nUnknowns;
    osqpA->nz = -1;
    osqpA->nzmax = optA.values.size();
    osqpA->x = optA.values.data();
    osqpA->i = optA.row_indices.data();
    osqpA->p = optA.col_ptrs.data();

    // Problem settings
    std::unique_ptr<OSQPSettings> settings(new OSQPSettings);
    osqp_set_default_settings(settings.get());
    settings->alpha = 1.0; // Over-relaxation parameter (you can tune this)
    //settings->time_limit = 500;
    settings->max_iter = 2 * 1000 * OSQPInt(1000) * 1000;
    settings->rho = 1.49e+2; //1.87;
    settings->eps_abs = eps / 10; //exp2(-cnVarsAtOnce);
    settings->eps_rel = eps / 10; //exp2(-cnVarsAtOnce);
    //settings->rho
    settings->polishing = 1;

    // Declare solver pointer
    OSQPSolver* solver = nullptr;
    // Initialize the solver
    OSQPInt exitflag = osqp_setup(&solver, osqpP.get(), optQ.data(), osqpA.get(), optL.data(), optH.data(),
      nConstraints, nUnknowns, settings.get());
    if (exitflag != 0) {
      std::cout << "Inequations solver failed to initialize." << std::endl;
      return false;
    }
    assert(initX.size() == nUnknowns);
    osqp_warm_start(solver, initX.data(), nullptr);
    osqp_solve(solver);

    bool maybeSat = true;
    if (solver->info->status_val != OSQP_SOLVED && solver->info->status_val != OSQP_SOLVED_INACCURATE) {
      maybeSat = false;
      goto cleanup;
    }

    // DEBUG-PRINT
    for(VCIndex i=0; i<vUnknowns.size(); i++) {
      const double val = solver->solution->x[i];
      std::cout << " v" << vUnknowns[i] << "=" << val << " ";
    }

    for(VCIndex i=0; i<vUnknowns.size(); i++) {
      const double val = solver->solution->x[i];
      bool isDef, setTrue;
      if(val >= 1.0 - eps) {
        isDef = true;
        setTrue = true;
      } else if(val <= -1.0+eps) {
        isDef = true;
        setTrue = false;
      } else {
        isDef = false;
        setTrue = val > -0;
      }
      const VCIndex aVar = vUnknowns[i];
      if(pFormula_->ans_[aVar] != setTrue) {
        pFormula_->ans_.Flip(aVar);
      }
      if(isDef) {
        unknowns.Remove(aVar);
      }
    }

cleanup:
    osqp_cleanup(solver);
    return maybeSat;
  }

  bool Solve() {
    VCTrackingSet unknowns;
    for(VCIndex i=1; i<=pFormula_->nVars_; i++) {
      unknowns.Add(i);
    }
    while(unknowns.Size() > 0) {
      VCIndex oldSize = unknowns.Size();
      if(!TailSolve(unknowns) || oldSize <= unknowns.Size()) {
        std::cout << "UNSATISFIABLE" << std::endl;
        return false; // unsatisfiable
      }
    }

    std::atomic<VCIndex> totSat = 0;
    // This is just for debugging, otherwise Formula::CountUnsat() would work
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
      if(satisfied) {
        totSat.fetch_add(1);
      }
    }
    if(totSat < pFormula_->nClauses_) {
      const VCIndex nUnsat = pFormula_->CountUnsat(pFormula_->ans_);
      std::cout << "Verification failed: " << nUnsat << " unsatisfied clauses." << std::endl;
    }

    return true;
  }
};

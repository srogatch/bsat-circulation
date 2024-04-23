#pragma once

#include "SatTracker.h"
#include "Traversal.h"
#include "extern/eigen/Eigen/Sparse"
#include <osqp/osqp.h>

struct SpMatTriple {
  VCIndex row_;
  VCIndex col_;
  float val_;

  SpMatTriple(const VCIndex aClause, const VCIndex aVar, const float val) {
    row_ = aClause;
    col_ = aVar;
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

  bool Solve() {
    std::cout << "Populating the Quadratic solver" << std::endl;

    std::vector<OSQPFloat> optL, optH, optQ(pFormula_->nVars_, 0), initX;
    CSCMatrix optA, optP;
    {
      std::vector<SpMatTriple> smtA, smtP;
      for(VCIndex aClause=1; aClause<=pFormula_->nClauses_; aClause++) {
        optL.emplace_back( 2 - pFormula_->clause2var_.ArcCount(aClause) );
        optH.emplace_back( pFormula_->clause2var_.ArcCount(aClause) );
        for(int8_t sign=-1; sign<=1; sign+=2) {
          const VCIndex nArcs = pFormula_->clause2var_.ArcCount(aClause, sign);
          for(VCIndex j=0; j<nArcs; j++) {
            const VCIndex iVar = pFormula_->clause2var_.GetTarget(aClause, sign, j);
            const VCIndex aVar = llabs(iVar);
            smtA.emplace_back(aClause-1, aVar-1, Signum(iVar));
          }
        }
      }
      for(VCIndex aVar=1; aVar<=pFormula_->nVars_; aVar++) {
        smtA.emplace_back(pFormula_->nClauses_ + aVar - 1, aVar-1, 1);
        optL.emplace_back(-1);
        optH.emplace_back(+1);
        smtP.emplace_back(aVar-1, aVar-1, 1);
        // for(VCIndex i=aVar; i<pFormula_->nVars_; i++) {
        //   smtP.emplace_back(aVar-1, i, 0);
        // }
        initX.emplace_back(pFormula_->ans_[aVar] ? 1 : -1);
      }
      optA = triplesToCSC(smtA, pFormula_->nClauses_+pFormula_->nVars_, pFormula_->nVars_);
      optP = triplesToCSC(smtP, pFormula_->nVars_, pFormula_->nVars_);
    }

    // Create CSC matrices for P and A
    std::unique_ptr<OSQPCscMatrix> osqpP(new OSQPCscMatrix);
    osqpP->m = pFormula_->nVars_;
    osqpP->n = pFormula_->nVars_;
    osqpP->nz = -1;
    osqpP->nzmax = optP.values.size();
    osqpP->x = optP.values.data();
    osqpP->i = optP.row_indices.data();
    osqpP->p = optP.col_ptrs.data();

    std::unique_ptr<OSQPCscMatrix> osqpA(new OSQPCscMatrix);
    osqpA->m = pFormula_->nClauses_+pFormula_->nVars_;
    osqpA->n = pFormula_->nVars_;
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
    settings->max_iter = 1000 * 1000 * 1000;
    settings->rho = 1.87;
    settings->eps_abs = 1.0 / pFormula_->nClauses_;
    settings->eps_rel = 1.0 / pFormula_->nClauses_;
    settings->polishing = 1;

    // Declare solver pointer
    OSQPSolver* solver = nullptr;
    // Initialize the solver
    OSQPInt exitflag = osqp_setup(&solver, osqpP.get(), optQ.data(), osqpA.get(), optL.data(), optH.data(),
      pFormula_->nClauses_ + pFormula_->nVars_, pFormula_->nVars_, settings.get());
    if (exitflag != 0) {
      std::cout << "Inequations solver failed to initialize." << std::endl;
      return false;
    }

    osqp_warm_start(solver, initX.data(), nullptr);

    osqp_solve(solver);
    constexpr const double eps = 1e-3;
    VCIndex nUnint = 0;
    std::atomic<VCIndex> totSat = 0;

    if (solver->info->status_val != OSQP_SOLVED && solver->info->status_val != OSQP_SOLVED_INACCURATE) {
      std::cout << "UNSATISFIABLE" << std::endl;
      goto cleanup;
    }
    for(VCIndex i=0; i<pFormula_->nVars_; i++) {
      const double val = solver->solution->x[i];
      if( !(-1-eps <= val && val <= 1+eps) ) {
        std::cout << " v" << i+1 << "=" << val << " ";
      }
      bool setTrue;
      if(val >= 1.0-eps) {
        setTrue = true;
      } else if(val <= -1+eps) {
        setTrue = false;
      } else {
        //std::cout << "Solution at var " << i+1 << " is not integer." << std::endl;
        setTrue = (val >= 0);
        nUnint++;
      }
      VCIndex aVar = i+1;
      if(pFormula_->ans_[aVar] != setTrue) {
        pFormula_->ans_.Flip(aVar);
      }
    }
    if(nUnint == 0) {
      std::cout << "SATISFIABLE" << std::endl;
    }
    else {
      std::cout << "nUnint=" << nUnint << std::endl;
      std::cout << "UNKNOWN" << std::endl;
    }

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
      std::cout << "Verification failed." << std::endl;
    }

cleanup:
    osqp_cleanup(solver);
    return true;
  }
};

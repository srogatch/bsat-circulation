#pragma once

#include "SatTracker.h"
#include "Traversal.h"
#include "extern/eigen/Eigen/Sparse"
#include <osqp/osqp.h>

struct SpMatTriple {
  VCIndex row_;
  VCIndex col_;
  double val_;

  SpMatTriple(const VCIndex aClause, const VCIndex aVar, const double val) {
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

    VCIndex nConstraints = 0;
    VCIndex nUnknowns = 0;

    std::vector<OSQPFloat> optL, optH, optQ, initX;

    CSCMatrix optA, optP;
    {
      std::vector<SpMatTriple> smtA, smtP;
      // x
      for(VCIndex aVar=1; aVar<=pFormula_->nVars_; aVar++) {
        smtA.emplace_back(nConstraints, nUnknowns, 1);
        optL.emplace_back(-1);
        optH.emplace_back(+1);
        //smtP.emplace_back(nUnknowns, nUnknowns, 0);
        initX.emplace_back(pFormula_->ans_[aVar] ? 1 : -1);
        optQ.emplace_back(0); // don't optimize x
        nUnknowns++;
        nConstraints++;
      }

      for(VCIndex aClause=1; aClause<=pFormula_->nClauses_; aClause++) {
        int64_t clauseSum = 0;
        for(int8_t sign=-1; sign<=1; sign+=2) {
          const VCIndex nArcs = pFormula_->clause2var_.ArcCount(aClause, sign);
          for(VCIndex j=0; j<nArcs; j++) {
            const VCIndex iVar = pFormula_->clause2var_.GetTarget(aClause, sign, j);
            const VCIndex aVar = llabs(iVar);
            smtA.emplace_back(nConstraints, aVar-1, -Signum(iVar));
            clauseSum += 1 - (pFormula_->ans_[aVar] ? 1 : -1);
          }
        }
        smtA.emplace_back(nConstraints, nUnknowns, -1); // c[i]
        optL.emplace_back(-pFormula_->clause2var_.ArcCount(aClause));
        optH.emplace_back(-pFormula_->clause2var_.ArcCount(aClause));
        nConstraints++;

        smtA.emplace_back(nConstraints, nUnknowns, 1);
        optL.emplace_back(0);
        optH.emplace_back( pFormula_->clause2var_.ArcCount(aClause)*2 - 1 );
        nConstraints++;

        optQ.emplace_back(0); // don't optimize c
        initX.emplace_back(clauseSum);
        nUnknowns++;
      }

      for(VCIndex i=0; i<pFormula_->nVars_; i++) {
        smtA.emplace_back(nConstraints, nUnknowns, 1); // t
        smtA.emplace_back(nConstraints, nUnknowns+1, -1); // u
        smtA.emplace_back(nConstraints, i, -2); // x
        optL.emplace_back(0);
        optH.emplace_back(0);
        nConstraints++;

        smtA.emplace_back(nConstraints, nUnknowns+1, 1); // u
        smtA.emplace_back(nConstraints, nUnknowns+2, 1); // v
        smtA.emplace_back(nConstraints, i, 1); // x
        optL.emplace_back(-1);
        optH.emplace_back(-1);
        nConstraints++;

        smtA.emplace_back(nConstraints, nUnknowns, 1); // t
        smtA.emplace_back(nConstraints, nUnknowns+2, 1); // v
        smtA.emplace_back(nConstraints, -i, 1); // -x
        optL.emplace_back(-1);
        optH.emplace_back(-1);
        nConstraints++;

        smtA.emplace_back(nConstraints, nUnknowns+2, 1); // v
        optL.emplace_back(-1);
        optH.emplace_back(1);
        nConstraints++;

        optQ.emplace_back(1); // t
        optQ.emplace_back(1); // u
        optQ.emplace_back(1); // v
        const double x = (pFormula_->ans_[i+1] ? 1 : -1);
        initX.emplace_back(x - 1); // t --> x-1
        initX.emplace_back(-x - 1); // u --> -x-1
        initX.emplace_back(0); // v --> 0
        nUnknowns+=3;
      }

      optA = triplesToCSC(smtA, nConstraints, nUnknowns);
      optP = triplesToCSC(smtP, nUnknowns, nUnknowns);
    }

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
    settings->max_iter = 1000 * 1000 * 1000;
    settings->rho = 1.49e+2; //1.87;
    settings->eps_abs = 1e-4; //1.0 / pFormula_->nClauses_;
    settings->eps_rel = 1e-4; //1.0 / pFormula_->nClauses_;
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
        std::cout << " Rx" << i+1 << "=" << val << " ";
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
        std::cout << " " << i+1 << ":x" << "=" << val //<< ",u=" << solver->solution->x[startU+i]
          //<< ",v=" << solver->solution->x[startV+i] // << ",s=" << solver->solution->x[startS+i];
          << " ";
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
      std::cout << "\nnUnint=" << nUnint << std::endl;
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
      const VCIndex nUnsat = pFormula_->CountUnsat(pFormula_->ans_);
      std::cout << "Verification failed: " << nUnsat << " unsatisfied clauses." << std::endl;
    }

cleanup:
    osqp_cleanup(solver);
    return true;
  }
};

#pragma once

#include "CNF.h"
#include "TrackingSet.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <variant>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(TCounter);
  std::unique_ptr<std::atomic<TCounter>[]> nSat_;
  Formula *pFormula_;
  std::atomic<int64_t> totSat_ = -1;

  explicit SatTracker(Formula& formula)
  : pFormula_(&formula)
  {
    nSat_.reset(new std::atomic<TCounter>[pFormula_->nClauses_+1]);
  }

  void Populate(const BitVector& assignment) {
    totSat_.store(0, std::memory_order_relaxed);
    nSat_[0] = 1;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      nSat_[i] = 0;
      for(const int64_t iVar : pFormula_->clause2var_.find(i)->second) {
        if( (iVar < 0 && !assignment[-iVar]) || (iVar > 0 && assignment[iVar]) ) {
          nSat_.get()[i].fetch_add(1);
        }
      }
      if(nSat_.get()[i].load(std::memory_order_relaxed)) {
        totSat_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  // The sign of iVar must reflect the new value of the variable.
  // Returns the change in satisfiability: positive - more satisfiable, negative - more unsatisfiable.
  int64_t FlipVar(const int64_t iVar, TrackingSet* pUnsatClauses) {
    const std::vector<int64_t>& clauses = pFormula_->listVar2Clause_.find(llabs(iVar))->second;
    std::atomic<int64_t> ans(0);
    #pragma omp parallel for
    for(int64_t i=0; i<clauses.size(); i++) {
      const int64_t iClause = clauses[i];
      const int64_t aClause = llabs(iClause);
      if(iClause * iVar > 0) {
        int64_t oldVal = nSat_.get()[aClause].fetch_add(1, std::memory_order_relaxed);
        assert(oldVal >= 0);
        if(oldVal == 0) {
          ans.fetch_add(1, std::memory_order_relaxed);
          const int64_t aClause = llabs(iClause);
          if(pUnsatClauses) {
            #pragma omp critical
            pUnsatClauses->Remove(aClause);
          }
        }
      }
      else {
        int64_t oldVal = nSat_.get()[aClause].fetch_sub(1, std::memory_order_relaxed);
        assert(oldVal >= 1);
        if(oldVal == 1) {
          ans.fetch_sub(1, std::memory_order_relaxed);
          if(pUnsatClauses) {
            #pragma omp critical
            pUnsatClauses->Add(aClause);
          }
        }
      }
    }
    totSat_.fetch_add(ans.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return ans.load(std::memory_order_relaxed);
  }

  TrackingSet GetUnsat() const {
    TrackingSet ans;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_.get()[i].load(std::memory_order_relaxed) == 0) {
        #pragma omp critical
        ans.Add(i);
      }
    }
    return ans;
  }

  int64_t UnsatCount() const {
    return pFormula_->nClauses_ - totSat_.load(std::memory_order_relaxed);
  }

  int64_t GradientDescend(const bool preferMove) {
    std::vector<int64_t> vVars(pFormula_->nVars_);
    constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars[0]);

    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nVars_; i++) {
      vVars[i-1] = i;
    }
    ParallelShuffle(vVars.data(), pFormula_->nVars_);

    int64_t minUnsat = UnsatCount();
    for(int64_t k=0; k<pFormula_->nVars_; k++) {
      assert(1 <= vVars[k] && vVars[k] <= pFormula_->nVars_);
      const int64_t iVar = vVars[k] * (pFormula_->ans_[vVars[k]] ? 1 : -1);
      const int64_t nNewSat = FlipVar(-iVar, nullptr);
      if(nNewSat >= (preferMove ? 0 : 1)) {
        minUnsat -= nNewSat;
        pFormula_->ans_.Flip(vVars[k]);
      } else {
        // Flip back
        FlipVar(iVar, nullptr);
      }
    }
    return minUnsat;
  }
};

using DefaultSatTracker = SatTracker<int16_t>;

#pragma once

#include "CNF.h"
#include "TrackingSet.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <variant>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cNSatChunk = kCacheLineSize / sizeof(std::atomic<TCounter>);
  static constexpr const uint32_t cVarOrderChunk = kCacheLineSize / sizeof(int64_t);
  std::unique_ptr<std::atomic<TCounter>[]> nSat_;
  Formula *pFormula_;
  std::atomic<int64_t> totSat_ = -1;
  std::vector<int64_t> varOrder_;

  explicit SatTracker(Formula& formula)
  : pFormula_(&formula)
  {
    nSat_.reset(new std::atomic<TCounter>[pFormula_->nClauses_+1]);
    varOrder_.resize(pFormula_->nVars_);
    #pragma omp parallel for schedule(static, cVarOrderChunk)
    for(int64_t i=1; i<=pFormula_->nVars_; i++) {
      varOrder_[i-1] = i;
    }
  }

  SatTracker(const SatTracker& src) {
    pFormula_ = src.pFormula_;
    totSat_.store(src.totSat_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    nSat_.reset(new std::atomic<TCounter>[pFormula_->nClauses_+1]);
    varOrder_.resize(pFormula_->nVars_);

    // TODO: better split into memcpy() ranges?
    #pragma omp parallel for
    for(int64_t i=0; i<=pFormula_->nClauses_; i++) {
      nSat_.get()[i].store(src.nSat_.get()[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    #pragma omp parallel for schedule(static, cVarOrderChunk)
    for(int64_t i=0; i<pFormula_->nVars_; i++) {
      varOrder_[i] = src.varOrder_[i];
    }
  }

  void Swap(SatTracker& fellow) {
    std::swap(nSat_, fellow.nSat_);
    std::swap(pFormula_, fellow.pFormula_);
    TCounter t = totSat_.load(std::memory_order_relaxed);
    totSat_.store(fellow.totSat_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    fellow.totSat_.store(t, std::memory_order_relaxed);
    varOrder_.swap(fellow.varOrder_);
  }

  void Populate(const BitVector& assignment) {
    totSat_.store(0, std::memory_order_relaxed);
    nSat_[0] = 1;
    #pragma omp parallel for schedule(static, cNSatChunk)
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
  int64_t FlipVar(const int64_t iVar, TrackingSet* front = nullptr) {
    const std::vector<int64_t>& clauses = pFormula_->listVar2Clause_.find(llabs(iVar))->second;
    int64_t ans = 0;
    #pragma omp parallel for reduction(+:ans)
    for(int64_t i=0; i<clauses.size(); i++) {
      const int64_t iClause = clauses[i];
      const int64_t aClause = llabs(iClause);
      if(iClause * iVar > 0) {
        int64_t oldVal = nSat_.get()[aClause].fetch_add(1, std::memory_order_relaxed);
        assert(oldVal >= 0);
        if(oldVal == 0) {
          ans++;
          if(front != nullptr) {
            #pragma omp critical
            front->Remove(aClause);
          }
        }
      }
      else {
        int64_t oldVal = nSat_.get()[aClause].fetch_sub(1, std::memory_order_relaxed);
        assert(oldVal >= 1);
        if(oldVal == 1) {
          ans--;
          if(front != nullptr) {
            #pragma omp critical
            front->Add(aClause);
          }
        }
      }
    }
    totSat_.fetch_add(ans, std::memory_order_relaxed);
    return ans;
  }

  TrackingSet GetUnsat() const {
    TrackingSet ans;
    #pragma omp parallel for schedule(static, cNSatChunk)
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
    ParallelShuffle(varOrder_.data(), pFormula_->nVars_);
    int64_t minUnsat = UnsatCount();
    for(int64_t k=0; k<pFormula_->nVars_; k++) {
      const int64_t aVar = varOrder_[k];
      assert(1 <= aVar && aVar <= pFormula_->nVars_);
      const int64_t iVar = aVar * (pFormula_->ans_[aVar] ? 1 : -1);
      const int64_t nNewSat = FlipVar(-iVar);
      if(nNewSat >= (preferMove ? 0 : 1)) {
        minUnsat -= nNewSat;
        pFormula_->ans_.Flip(aVar);
      } else {
        // Flip back
        FlipVar(iVar);
      }
    }
    return minUnsat;
  }

  int64_t ParallelGD(const bool preferMove, const int64_t varsAtOnce,
    std::vector<std::pair<int64_t, int64_t>>& weightedVars, TrackingSet* revVertices)
  {
    int64_t minUnsat = UnsatCount();
    std::vector<int64_t> front(varsAtOnce);
    const int64_t endK = DivUp(weightedVars.size(), varsAtOnce) * varsAtOnce;
    #pragma omp parallel num_threads(varsAtOnce) shared(front)
    for(int64_t k=omp_get_thread_num(); k<endK; k+=varsAtOnce) {
      assert(omp_get_num_threads() == varsAtOnce);
      int64_t aVar, iVar;
      if(k < weightedVars.size()) {
        aVar = weightedVars[k].first;
        assert(1 <= aVar && aVar <= pFormula_->nVars_);
        iVar = aVar * (pFormula_->ans_[aVar] ? 1 : -1);
        const int64_t nNewSat = FlipVar(-iVar);
        front[omp_get_thread_num()] = aVar;
      }

      #pragma omp barrier

      #pragma omp single
      {
        int64_t newUSC = UnsatCount();
        if(newUSC < minUnsat + (preferMove ? 1 : 0)) {
          minUnsat = newUSC;
          for(int64_t i=0; i<varsAtOnce; i++) {
            const int64_t aFV = llabs(front[i]);
            if(aFV == 0) {
              continue;
            }
            pFormula_->ans_.Flip(aFV);
            if(revVertices != nullptr) {
              if(revVertices->set_.find(aFV) == revVertices->set_.end()) {
                revVertices->Add(aFV);
              } else {
                revVertices->Remove(aFV);
              }
            }
          }
          for(int64_t i=0; i<varsAtOnce; i++) {
            front[i] = 0;
          }
        }
      }
      

      #pragma omp barrier

      if(k < weightedVars.size()) {
        if(front[omp_get_thread_num()] == aVar) {
          // Flip back
          FlipVar(iVar);
        }
        front[omp_get_thread_num()] = 0;
      }

      #pragma omp barrier
    }
    return minUnsat;
  }
};

using DefaultSatTracker = SatTracker<int>;

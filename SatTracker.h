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

  void CopyFrom(const SatTracker& src) {
    pFormula_ = src.pFormula_;
    totSat_.store(src.totSat_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    nSat_.reset(new std::atomic<TCounter>[pFormula_->nClauses_+1]);

    #pragma omp parallel for schedule(static, kRamPageBytes)
    for(int64_t i=0; i<=DivUp(pFormula_->nClauses_, cParChunkSize); i++) {
      const int64_t iFirst = i*cParChunkSize;
      const int64_t iLimit = std::min((i+1) * cParChunkSize, pFormula_->nClauses_);
      if(iFirst < iLimit) {
        memcpy(
          nSat_.get()+iFirst, src.nSat_.get()+iFirst,
          (iLimit-iFirst) * cParChunkSize * sizeof(std::atomic<TCounter>)
        );
      }
    }
  }

  SatTracker(const SatTracker& src) {
    CopyFrom(src);
  }

  SatTracker& operator=(const SatTracker& src) {
    if(this != &src) {
      CopyFrom(src);
    }
    return *this;
  }

  void Swap(SatTracker& fellow) {
    std::swap(nSat_, fellow.nSat_);
    std::swap(pFormula_, fellow.pFormula_);
    TCounter t = totSat_.load(std::memory_order_relaxed);
    totSat_.store(fellow.totSat_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    fellow.totSat_.store(t, std::memory_order_relaxed);
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
  int64_t FlipVar(const int64_t iVar, TrackingSet* unsatClauses, TrackingSet* front) {
    std::mutex muUC, muFront;
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
          if(unsatClauses != nullptr) {
            std::unique_lock<std::mutex> lock(muUC);
            unsatClauses->Remove(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(muFront);
            front->Remove(aClause);
          }
        }
      }
      else {
        int64_t oldVal = nSat_.get()[aClause].fetch_sub(1, std::memory_order_relaxed);
        assert(oldVal >= 1);
        if(oldVal == 1) {
          ans--;
          if(unsatClauses != nullptr) {
            std::unique_lock<std::mutex> lock(muUC);
            unsatClauses->Add(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(muFront);
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
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_.get()[i].load(std::memory_order_relaxed) == 0) {
        #pragma omp critical
        ans.Add(i);
      }
    }
    return ans;
  }

  TrackingSet GetUnsat(const SatTracker& oldSatTr, TrackingSet& newFront) const {
    TrackingSet ans;
    std::mutex muFront, muAns;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_.get()[i].load(std::memory_order_relaxed) == 0) {
        if(oldSatTr.nSat_.get()[i].load(std::memory_order_relaxed) > 0) {
          std::unique_lock<std::mutex> lock(muFront);
          newFront.Add(i);
        }
        std::unique_lock<std::mutex> lock(muAns);
        ans.Add(i);
      }
    }
    return ans;
  }

  int64_t UnsatCount() const {
    return pFormula_->nClauses_ - totSat_.load(std::memory_order_relaxed);
  }

  int64_t GradientDescend(const bool preferMove, TrackingSet *unsatClauses, TrackingSet *front) {
    std::vector<int64_t> vVars(pFormula_->nVars_);
    if(unsatClauses == nullptr) {
      constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars[0]);
      #pragma omp parallel for schedule(static, cParChunkSize)
      for(int64_t i=1; i<=pFormula_->nVars_; i++) {
        vVars[i-1] = i;
      }
    }
    else {
      std::vector<int64_t> vClauses(unsatClauses->set_.begin(), unsatClauses->set_.end());
      BitVector useVar(pFormula_->nVars_+1);
      #pragma omp parallel for
      for(int64_t i=0; i<vClauses.size(); i++) {
        for(const int64_t iVar : pFormula_->clause2var_.find(vClauses[i])->second) {
          useVar.NohashSet(llabs(iVar));
        }
      }
      std::atomic<int64_t> pos(0);
      #pragma omp parallel for
      for(int64_t i=1; i<=pFormula_->nVars_; i++) {
        if(useVar[i]) {
          const int64_t oldPos = pos.fetch_add(1, std::memory_order_relaxed);
          vVars[oldPos] = i;
        }
      }
      vVars.resize(pos.load(std::memory_order_relaxed));
    }
    ParallelShuffle(vVars.data(), vVars.size());

    int64_t minUnsat = UnsatCount();
    assert( unsatClauses == nullptr || minUnsat == unsatClauses->set_.size() );
    for(int64_t k=0; k<vVars.size(); k++) {
      assert(1 <= vVars[k] && vVars[k] <= pFormula_->nVars_);
      const int64_t iVar = vVars[k] * (pFormula_->ans_[vVars[k]] ? 1 : -1);
      const int64_t nNewSat = FlipVar(-iVar, unsatClauses, front);
      if(nNewSat >= (preferMove ? 0 : 1)) {
        minUnsat -= nNewSat;
        pFormula_->ans_.Flip(vVars[k]);
      } else {
        // Flip back
        FlipVar(iVar, unsatClauses, front);
      }
    }
    return minUnsat;
  }
};

using DefaultSatTracker = SatTracker<int16_t>;

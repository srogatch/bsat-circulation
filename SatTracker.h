#pragma once

#include "CNF.h"
#include "TrackingSet.h"
#include "Traversal.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <variant>
#include <execution>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(TCounter);
  static constexpr const int64_t kSyncContention = 37; // 37 per CPU
  std::vector<int64_t> vVars_;
  std::unique_ptr<TCounter[]> nSat_;
  std::unique_ptr<std::atomic_flag[]> syncs_;
  Formula *pFormula_ = nullptr;
  std::atomic<int64_t> totSat_ = 0;

  void Init(Formula* pFormula) {
    pFormula_ = pFormula;
    syncs_.reset(new std::atomic_flag[kSyncContention * nSysCpus]);
    vVars_.resize(pFormula->nVars_);
    constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars_[0]);
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nVars_; i++) {
      vVars_[i-1] = i;
    }
  }

  explicit SatTracker(Formula& formula)
  {
    Init(&formula);
    nSat_.reset(new TCounter[pFormula_->nClauses_+1]);
  }

  SatTracker(const SatTracker& src) {
    Init(src.pFormula_);
    CopyFrom(src);
  }

  void CopyFrom(const SatTracker& src) {
    if(nSat_ == nullptr || pFormula_ != src.pFormula_) {
      nSat_.reset(new TCounter[src.pFormula_->nClauses_+1]);
    }
    pFormula_ = src.pFormula_;
    totSat_.store(src.totSat_.load(std::memory_order_relaxed), std::memory_order_relaxed);

    #pragma omp parallel for schedule(static, kRamPageBytes)
    for(int64_t i=0; i<=pFormula_->nClauses_; i++) {
      nSat_[i] = src.nSat_[i];
    }

    // Ignore vVars_ here: they're randomized each time anyway
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

  VCTrackingSet Populate(const BitVector& assignment) {
    VCTrackingSet ans;
    int64_t curTot = 0;
    nSat_[0] = 1;
    #pragma omp parallel for schedule(static, cParChunkSize) reduction(+:curTot)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      nSat_[i] = 0;
      for(const int64_t iVar : pFormula_->clause2var_.find(i)->second) {
        if( (iVar < 0 && !assignment[-iVar]) || (iVar > 0 && assignment[iVar]) ) {
          nSat_[i]++;
        }
      }
      if(nSat_[i]) {
        curTot++;
      } else {
        // Unsatisfied clause
        ans.Add(i);
      }
    }
    totSat_ = curTot;
    return ans;
  }

  bool Verify(const BitVector& assignment) {
    int64_t curTot = 0;
    if(nSat_[0] != 1) {
      return false;
    }
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      int64_t curSat = 0;
      for(const int64_t iVar : pFormula_->clause2var_.find(i)->second) {
        if( (iVar < 0 && !assignment[-iVar]) || (iVar > 0 && assignment[iVar]) ) {
          curSat++;
        }
      }
      if(curSat > 0) {
        curTot++;
      }
      if(nSat_[i] != curSat) {
        return false;
      }
    }
    if(totSat_ != curTot) {
      return false;
    }
    return true;
  }

  void Lock(const int64_t iClause) {
    while(syncs_[iClause % (kSyncContention * nSysCpus)].test_and_set(std::memory_order_acq_rel)) {
      std::this_thread::yield();
    }
  }
  void Unlock(const int64_t iClause) {
    syncs_[iClause % (kSyncContention * nSysCpus)].clear(std::memory_order_release);
  }

  // The sign of iVar must reflect the new value of the variable.
  // Returns the change in satisfiability: positive - more satisfiable, negative - more unsatisfiable.
  int64_t FlipVar(const int64_t iVar, VCTrackingSet* unsatClauses, VCTrackingSet* front) {
    const std::vector<int64_t>& clauses = pFormula_->listVar2Clause_.find(llabs(iVar))->second;
    int64_t ans = 0;
    #pragma omp parallel for reduction(+:ans)
    for(int64_t i=0; i<int64_t(clauses.size()); i++) {
      const int64_t iClause = clauses[i];
      const int64_t aClause = llabs(iClause);
      if(iClause * iVar > 0) {
        Lock(aClause);
        nSat_[aClause]++;
        assert(nSat_[aClause] >= 1);
        if(nSat_[aClause] == 1) { // just became satisfied
          ans++;
          if(unsatClauses != nullptr) {
            unsatClauses->Remove(aClause);
          }
          if(front != nullptr && front != unsatClauses) {
            front->Remove(aClause);
          }
        }
        Unlock(aClause);
      }
      else {
        Lock(aClause);
        nSat_[aClause]--;
        assert(nSat_[aClause] >= 0);
        if(nSat_[aClause] == 0) { // just became unsatisfied
          ans--;
          if(unsatClauses != nullptr) {
            unsatClauses->Add(aClause);
          }
          if(front != nullptr && front != unsatClauses) {
            front->Add(aClause);
          }
        }
        Unlock(aClause);
      }
    }
    totSat_.fetch_add(ans, std::memory_order_relaxed);
    return ans;
  }

  VCTrackingSet GetUnsat() const {
    VCTrackingSet ans;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_[i] == 0) {
        #pragma omp critical
        ans.Add(i);
      }
    }
    return ans;
  }

  VCTrackingSet GetUnsat(const SatTracker& oldSatTr, VCTrackingSet& newFront) const {
    VCTrackingSet ans;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_.get()[i].load(std::memory_order_relaxed) == 0) {
        if(oldSatTr.nSat_.get()[i].load(std::memory_order_relaxed) > 0) {
          newFront.Add(i);
        }
        ans.Add(i);
      }
    }
    return ans;
  }

  int64_t UnsatCount() const {
    return pFormula_->nClauses_ - totSat_.load(std::memory_order_relaxed);
  }

  // int64_t Divergence(const bool preferMove, Traversal& trav,
  //   const VCTrackingSet* considerClauses, VCTrackingSet& unsatClauses,
  //   int64_t minUnsat)
  // {
  //   std::vector<int64_t> subsetVars, *pvVars = nullptr;
  //   if(considerClauses == nullptr) {
  //     ParallelShuffle(vVars_.data(), vVars_.size());
  //     pvVars = &vVars_;
  //   }
  //   else {
  //     subsetVars = pFormula_->ClauseFrontToVars(*considerClauses, pFormula_->ans_);
  //     pvVars = &subsetVars;
  //   }
  //   VCTrackingSet revVars;
  //   std::mutex muState;
  //   #pragma omp parallel for schedule(dynamic, 1)
  //   for(int64_t i=0; i<pvVars->size(); i++) {
  //     const int64_t aVar = (*pvVars)[k];
  //     assert(1 <= aVar && aVar <= pFormula_->nVars_);
  //     // This works in parallel as long as there are no duplicate variables in the vector
  //     const int64_t iVar = aVar * (pFormula_->ans_[aVar] ? 1 : -1);
  //     {
  //       std::unique_lock<std::mutex> lock(muState);
  //       revVars.Flip(aVar);
  //     }
  //     pFormula_->ans_.Flip(aVar);
  //     FlipVar(-iVar, nullptr, nullptr);
  //     bool bSeenMove = false;
  //     {
  //       std::unique_lock<std::mutex> lock(muState);
  //       if(trav.IsSeenMove(unsatClauses, revVars)) {
  //         revVars.Flip(aVar);
  //         bSeenMove = true;
  //       }
  //     }
  //     if(!bSeenMove) {

  //     }
  //   }
  // }

  int64_t GradientDescend(const bool preferMove, Traversal& trav,
    const VCTrackingSet* considerClauses, VCTrackingSet& unsatClauses, const VCTrackingSet& startFront,
    VCTrackingSet& front, int64_t minUnsat, bool& moved)
  {
    std::vector<int64_t> subsetVars, *pvVars = nullptr;
    if(considerClauses == nullptr) {
      ParallelShuffle(vVars_.data(), vVars_.size());
      pvVars = &vVars_;
    }
    else {
      subsetVars = pFormula_->ClauseFrontToVars(*considerClauses, pFormula_->ans_);
      pvVars = &subsetVars;
    }

    VCTrackingSet revVars;
    // TODO: flip a random number of consecutive vars in each step (i.e. new random count in each step)
    for(int64_t k=0; k<int64_t(pvVars->size()); k++) {
      const int64_t aVar = (*pvVars)[k];
      assert(1 <= aVar && aVar <= pFormula_->nVars_);
      const int64_t iVar = aVar * (pFormula_->ans_[aVar] ? 1 : -1);
      revVars.Flip(aVar);
      if( trav.IsSeenMove(startFront, revVars) ) {
        revVars.Flip(aVar);
        continue;
      }
      // TODO: instead of flipping directly the formula, shall we pass an arbitrary assignment as a parameter?
      pFormula_->ans_.Flip(aVar);
      FlipVar(-iVar, &unsatClauses, &front);

      if(!trav.IsSeenAssignment(pFormula_->ans_)) {
        trav.FoundMove(startFront, revVars, pFormula_->ans_, unsatClauses.Size());
        int64_t newUnsat = UnsatCount();
        if(newUnsat < minUnsat + (preferMove ? 1 : 0)) {
          moved = true;
          minUnsat = newUnsat;
          continue;
        }
      }

      // Flip back
      pFormula_->ans_.Flip(aVar);
      FlipVar(iVar, &unsatClauses, &front);
      revVars.Flip(aVar);
    }
    return minUnsat;
  }

  int64_t ParallelGD(const bool preferMove, const int64_t varsAtOnce,
    const std::vector<int64_t>& varFront, BitVector& next, Traversal& trav,
    VCTrackingSet* unsatClauses, const VCTrackingSet& startClauseFront,
    VCTrackingSet& revVars, int64_t minUnsat, bool& moved, int64_t level)
  {
    for(int64_t i=0; i<int64_t(varFront.size()); i+=varsAtOnce) {
      std::vector<int64_t> selVars(varsAtOnce, 0);
      const int64_t nVars = std::min<int64_t>(varsAtOnce, int64_t(varFront.size()) - i);
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        int64_t aVar, iVar;
        aVar = varFront[i+j];
        assert(1 <= aVar && aVar <= pFormula_->nVars_);
        iVar = aVar * (next[aVar] ? 1 : -1);
        selVars[j] = iVar;
        #pragma omp critical
        revVars.Flip(aVar);
      }
      if( trav.IsSeenMove(startClauseFront, revVars) ) {
        // Should be better sequential
        for(int64_t j=0; j<nVars; j++) {
          const int64_t iVar = selVars[j];
          const int64_t aVar = llabs(iVar);
          revVars.Flip(aVar);
        }
        continue;
      }
      
      VCTrackingSet newClauseFront;
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
        FlipVar(aVar * (next[aVar] ? 1 : -1), unsatClauses, &newClauseFront);
      }
      if(!trav.IsSeenAssignment(next)) {
        const int64_t newNUnsat = UnsatCount();
        trav.FoundMove(startClauseFront, revVars, next, newNUnsat);
        if(newNUnsat < minUnsat + (preferMove ? 1 : 0)) {
          moved = true;
          minUnsat = newNUnsat;
          if(minUnsat == 0) {
            break;
          }
          continue;
        }
        if(level > 0 && newClauseFront.Size() > 0) {
          std::vector<int64_t> newVarFront = pFormula_->ClauseFrontToVars(newClauseFront, next);
          bool nextMoved = false;
          const int64_t subNUnsat = ParallelGD(
            preferMove, varsAtOnce, newVarFront, next, trav, unsatClauses, startClauseFront,
            revVars, minUnsat, nextMoved, level-1
          );
          if(subNUnsat < minUnsat || (preferMove && nextMoved && subNUnsat == minUnsat)) {
            minUnsat = subNUnsat;
            moved = true;
            if(minUnsat == 0) {
              break;
            }
            continue;
          }
        }
      }

      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
        FlipVar(aVar * (next[aVar] ? 1 : -1), unsatClauses, nullptr);
        #pragma omp critical
        revVars.Flip(aVar);
      }
    }
    return minUnsat;
  }
};

using DefaultSatTracker = SatTracker<int16_t>;

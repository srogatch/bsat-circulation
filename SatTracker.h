#pragma once

#include "CNF.h"
#include "TrackingSet.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <variant>
#include <execution>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(TCounter);
  static constexpr const int64_t kSyncContention = 37; // 37 per CPU
  std::unique_ptr<TCounter[]> nSat_;
  std::unique_ptr<std::atomic_flag[]> syncs_;
  Formula *pFormula_ = nullptr;
  std::atomic<int64_t> totSat_ = 0;

  explicit SatTracker(Formula& formula)
  : pFormula_(&formula)
  {
    nSat_.reset(new TCounter[pFormula_->nClauses_+1]);
    syncs_.reset(new std::atomic_flag[kSyncContention * Formula::nCpus_]);
  }

  SatTracker(const SatTracker& src) {
    syncs_.reset(new std::atomic_flag[kSyncContention * Formula::nCpus_]);
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
      }
    }
    totSat_ = curTot;
  }

  void Lock(const int64_t iClause) {
    while(syncs_[iClause % (kSyncContention * Formula::nCpus_)].test_and_set(std::memory_order_acq_rel)) {
      std::this_thread::yield();
    }
  }
  void Unlock(const int64_t iClause) {
    syncs_[iClause % (kSyncContention * Formula::nCpus_)].clear(std::memory_order_release);
  }

  // The sign of iVar must reflect the new value of the variable.
  // Returns the change in satisfiability: positive - more satisfiable, negative - more unsatisfiable.
  int64_t FlipVar(const int64_t iVar, TrackingSet* unsatClauses, TrackingSet* front) {
    const std::vector<int64_t>& clauses = pFormula_->listVar2Clause_.find(llabs(iVar))->second;
    int64_t ans = 0;
    #pragma omp parallel for reduction(+:ans)
    for(int64_t i=0; i<clauses.size(); i++) {
      const int64_t iClause = clauses[i];
      const int64_t aClause = llabs(iClause);
      if(iClause * iVar > 0) {
        Lock(aClause);
        nSat_[aClause]++;
        assert(nSat_[aClause] >= 1);
        if(nSat_[aClause] == 1) { // just became satisfied
          ans++;
          if(unsatClauses != nullptr) {
            std::unique_lock<std::mutex> lock(unsatClauses->sync_);
            unsatClauses->Remove(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(front->sync_);
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
            std::unique_lock<std::mutex> lock(unsatClauses->sync_);
            unsatClauses->Add(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(front->sync_);
            front->Add(aClause);
          }
        }
        Unlock(aClause);
      }
    }
    totSat_.fetch_add(ans, std::memory_order_relaxed);
    return ans;
  }

  TrackingSet GetUnsat() const {
    TrackingSet ans;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_[i] == 0) {
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

  int64_t GradientDescend(const bool preferMove, TrackingSet* considerClauses, TrackingSet *unsatClauses, TrackingSet *front) {
    std::vector<int64_t> vVars(pFormula_->nVars_);
    if(considerClauses == nullptr) {
      constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars[0]);
      #pragma omp parallel for schedule(static, cParChunkSize)
      for(int64_t i=1; i<=pFormula_->nVars_; i++) {
        vVars[i-1] = i;
      }
    }
    else {
      std::vector<int64_t> vClauses(considerClauses->set_.begin(), considerClauses->set_.end());
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

  int64_t ParallelGD(const bool preferMove, const int64_t varsAtOnce,
    const std::vector<std::pair<int64_t, int64_t>>& weightedVars,
    BitVector& next, std::unordered_set<std::pair<uint128, uint128>>& seenMove,
    TrackingSet* unsatClauses, const TrackingSet& startFront,
    TrackingSet& revVertices, int64_t minUnsat, int64_t level)
  {
    for(int64_t i=0; i<int64_t(weightedVars.size()); i+=varsAtOnce) {
      std::vector<int64_t> selVars(varsAtOnce, 0);
      const int64_t nVars = std::min<int64_t>(varsAtOnce, int64_t(weightedVars.size()) - i);
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        int64_t aVar, iVar;
        aVar = weightedVars[i+j].first;
        assert(1 <= aVar && aVar <= pFormula_->nVars_);
        iVar = aVar * (next[aVar] ? 1 : -1);
        selVars[j] = iVar;
        #pragma omp critical
        revVertices.Flip(aVar);
      }
      if(seenMove.find({startFront.hash_, revVertices.hash_}) != seenMove.end()) {
        // Should be better sequential
        for(int64_t j=0; j<nVars; j++) {
          const int64_t iVar = selVars[j];
          const int64_t aVar = llabs(iVar);
          revVertices.Flip(aVar);
        }
        continue;
      }
      TrackingSet newFront;
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
        FlipVar(aVar * (next[aVar] ? 1 : -1), unsatClauses, &newFront);
      }
      const int64_t newNUnsat = UnsatCount();
      if(newNUnsat < minUnsat + (preferMove ? 1 : 0)) {
        minUnsat = newNUnsat;
        continue;
      }
      if(level > 0 && !newFront.set_.empty()) {
        // Try to combine the variable assignments in the new front
        std::unordered_map<int64_t, int64_t> candVs;
        std::vector<int64_t> vFront(newFront.set_.begin(), newFront.set_.end());
        #pragma omp parallel for
        for(int64_t j=0; j<vFront.size(); j++) {
          const int64_t originClause = vFront[j];
          for(const int64_t iVar : pFormula_->clause2var_[originClause]) {
            if( (iVar < 0 && next[-iVar]) || (iVar > 0 && !next[iVar]) ) {
              // A dissatisfying arc
              const int64_t revV = llabs(iVar);
              #pragma omp critical
              candVs[revV]++;
            }
          }
        }

        std::vector<std::pair<int64_t, int64_t>> combs(candVs.begin(), candVs.end());
        if( combs.size() >= 2 * omp_get_max_threads() ) {
          ParallelShuffle(combs.data(), combs.size());
        } else {
          unsigned long long seed;
          while(!_rdrand64_step(&seed));
          std::mt19937_64 rng(seed);
          std::shuffle(combs.begin(), combs.end(), rng);
        }
        std::stable_sort(std::execution::par, combs.begin(), combs.end(), [](const auto& a, const auto& b) {
          return a.second > b.second;
        });
        const int64_t subNUnsat = ParallelGD(
          preferMove, varsAtOnce, combs, next, seenMove, unsatClauses, startFront, revVertices, minUnsat, level-1);
        if(subNUnsat < minUnsat + (preferMove ? 1 : 0)) {
          minUnsat = subNUnsat;
          continue;
        }
      }

      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
        const int64_t nNewSat = FlipVar(aVar * (next[aVar] ? 1 : -1), unsatClauses, nullptr);
        #pragma omp critical
        revVertices.Flip(aVar);
      }
    }
    return minUnsat;
  }

};

using DefaultSatTracker = SatTracker<int16_t>;

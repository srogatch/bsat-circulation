#pragma once

#include "CNF.h"
#include "TrackingSet.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <variant>
#include <execution>

template<typename TCounter> struct SatTracker {
  static constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(std::atomic<TCounter>);
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
  int64_t FlipVar(const int64_t iVar, std::mutex *muUC, std::mutex *muFront, TrackingSet* unsatClauses, TrackingSet* front) {
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
            std::unique_lock<std::mutex> lock(*muUC);
            unsatClauses->Remove(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(*muFront);
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
            std::unique_lock<std::mutex> lock(*muUC);
            unsatClauses->Add(aClause);
          }
          if(front != nullptr) {
            std::unique_lock<std::mutex> lock(*muFront);
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
    std::mutex muUCs, muFront;
    for(int64_t k=0; k<vVars.size(); k++) {
      assert(1 <= vVars[k] && vVars[k] <= pFormula_->nVars_);
      const int64_t iVar = vVars[k] * (pFormula_->ans_[vVars[k]] ? 1 : -1);
      const int64_t nNewSat = FlipVar(-iVar, &muUCs, &muFront, unsatClauses, front);
      if(nNewSat >= (preferMove ? 0 : 1)) {
        minUnsat -= nNewSat;
        pFormula_->ans_.Flip(vVars[k]);
      } else {
        // Flip back
        FlipVar(iVar, &muUCs, &muFront, unsatClauses, front);
      }
    }
    return minUnsat;
  }

  int64_t ParallelGD(const bool preferMove, const int64_t varsAtOnce,
    const std::vector<std::pair<int64_t, int64_t>>& weightedVars,
    BitVector& next, std::unordered_set<std::pair<uint128, uint128>>& seenMove,
    TrackingSet& unsatClauses, TrackingSet& startFront, TrackingSet& revVertices,
    int64_t level = 0)
  {
    int64_t minUnsat = UnsatCount() * 2;
    std::mutex muUC, muFront;
    for(int64_t i=0; i<weightedVars.size(); i+=varsAtOnce) {
      std::vector<int64_t> selVars(varsAtOnce, 0);
      const int64_t nVars = std::min<int64_t>(varsAtOnce, weightedVars.size() - i);
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        int64_t aVar, iVar;
        aVar = weightedVars[i+j].first;
        assert(1 <= aVar && aVar <= pFormula_->nVars_);
        iVar = aVar * (next[aVar] ? 1 : -1);
        selVars[j] = iVar;
        #pragma omp critical
        {
          if(revVertices.set_.find(aVar) == revVertices.set_.end()) {
            revVertices.Add(aVar);
          } else {
            revVertices.Remove(aVar);
          }
        }
      }
      if(seenMove.find({startFront.hash_, revVertices.hash_}) != seenMove.end()) {
        // Should be better sequential
        for(int64_t j=0; j<nVars; j++) {
          const int64_t iVar = selVars[j];
          const int64_t aVar = llabs(iVar);
          if(revVertices.set_.find(aVar) == revVertices.set_.end()) {
            revVertices.Add(aVar);
          } else {
            revVertices.Remove(aVar);
          }
        }
        continue;
      }
      TrackingSet newFront;
      #pragma omp parallel for num_threads(nVars)
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        const int64_t nNewSat = FlipVar(-iVar, &muUC, &muFront, &unsatClauses, &newFront);
        next.Flip(aVar);
      }
      const int64_t newNUnsat = UnsatCount();
      if(newNUnsat < minUnsat + (preferMove ? 1 : 0)) {
        minUnsat = newNUnsat;
        continue;
      }
      if(level < 0 && !newFront.set_.empty()) {
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
          preferMove, varsAtOnce, combs, next, seenMove, unsatClauses, newFront, revVertices, level+1);
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
        const int64_t nNewSat = FlipVar(iVar, &muUC, nullptr, &unsatClauses, nullptr);
        #pragma omp critical
        {
          if(revVertices.set_.find(aVar) == revVertices.set_.end()) {
            revVertices.Add(aVar);
          } else {
            revVertices.Remove(aVar);
          }
        }
      }
    }
    return minUnsat;
  }

};

using DefaultSatTracker = SatTracker<int16_t>;

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
  static constexpr const uint32_t cParChunkSize = kRamPageBytes / sizeof(TCounter);
  static constexpr const int64_t cSyncContention = 37; // 37 per CPU
  std::vector<MultiItem<VCIndex>> vVars_;
  std::unique_ptr<TCounter[]> nSat_;
  std::unique_ptr<std::atomic_flag[]> syncs_;
  Formula *pFormula_ = nullptr;
  std::atomic<int64_t> totSat_ = 0;

  void Init(Formula* pFormula) {
    pFormula_ = pFormula;
    syncs_.reset(new std::atomic_flag[cSyncContention * nSysCpus]);
    vVars_.resize(pFormula->nVars_);
    //constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars_[0]);
    #pragma omp parallel for schedule(static, kRamPageBytes)
    for(int64_t i=1; i<=pFormula_->nVars_; i++) {
      vVars_[i-1] = i;
    }
  }

  SatTracker() = default;

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
    totSat_.store(src.totSat_.load(std::memory_order_release), std::memory_order_acquire);

    //#pragma omp parallel for schedule(static, kRamPageBytes)
    // for(int64_t i=0; i<=pFormula_->nClauses_; i++) {
    //   nSat_[i] = src.nSat_[i];
    // }
    memcpy(nSat_.get(), src.nSat_.get(), sizeof(TCounter) * (pFormula_->nClauses_+1));

    // Ignore vVars_ here: they're randomized each time anyway
  }

  SatTracker& operator=(const SatTracker& src) {
    if(this != &src) {
      if(pFormula_ == nullptr) {
        Init(src.pFormula_);
      }
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

  VCTrackingSet Populate(const BitVector& assignment, VCTrackingSet* front) {
    VCTrackingSet ans;
    int64_t curTot = 0;
    nSat_[0] = 1;
    if(front != nullptr) {
      front->Clear();
    }
    //#pragma omp parallel for schedule(guided, kRamPageBytes) reduction(+:curTot)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      VCIndex oldSat = nSat_[i];
      assert( pFormula_->clause2var_.ArcCount(i) <= std::numeric_limits<TCounter>::max() );
      // Prevent the counter flowing below 1 if it's a dummy (always satisfied) clause
      nSat_[i] = (pFormula_->dummySat_[i] ? 1 : 0);
      #pragma unroll
      for(int8_t sgnTo=-1; sgnTo<=1; sgnTo+=2) {
        const VCIndex nArcs = pFormula_->clause2var_.ArcCount(i, sgnTo);
        for(VCIndex at=0; at<nArcs; at++) {
          const int64_t iVar = pFormula_->clause2var_.GetTarget(i, sgnTo, at);
          assert(Signum(iVar) == sgnTo);
          assert(1 <= llabs(iVar) && llabs(iVar) <= pFormula_->nClauses_);
          if( (sgnTo < 0 && !assignment[-iVar]) || (sgnTo > 0 && assignment[iVar]) ) {
            assert( nSat_[i] < std::numeric_limits<TCounter>::max() );
            nSat_[i]++;
          }
        }
      }
      if(nSat_[i]) {
        curTot++;
      } else {
        // Unsatisfied clause
        ans.Add(i);
        if(front != nullptr && oldSat > 0) {
          front->Add(i);
        }
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
    std::atomic<bool> ans = true;
    #pragma omp parallel for schedule(guided, kRamPageBytes) reduction(+:curTot)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      #pragma omp cancellation point for
      int64_t curSat = (pFormula_->dummySat_[i] ? 1 : 0);
      #pragma unroll
      for(int8_t sgnTo=-1; sgnTo<=1; sgnTo+=2) {
        const VCIndex nArcs = pFormula_->clause2var_.ArcCount(i, sgnTo);
        for(VCIndex at=0; at<nArcs; at++) {
          const int64_t iVar = pFormula_->clause2var_.GetTarget(i, sgnTo, at);
          assert(Signum(iVar) == sgnTo);
          assert(1 <= llabs(iVar) && llabs(iVar) <= pFormula_->nClauses_);
          if( (sgnTo < 0 && !assignment[-iVar]) || (sgnTo > 0 && assignment[iVar]) ) {
            curSat++;
          }
        }
      }
      if(curSat > 0) {
        curTot++;
      }
      if(nSat_[i] != curSat) {
        ans = false;
        #pragma omp cancel for
      }
    }
    if(totSat_ != curTot) {
      ans = false;
    }
    return ans;
  }

  // Not thread-safe
  bool ReallyUnsat(const VCTrackingSet& unsatClauses) {
    std::vector<int64_t> vUnsat = unsatClauses.ToVector();
    std::atomic<bool> ans = true;
    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=0; i<int64_t(vUnsat.size()); i++) {
      // It's important it's not interfering with Finally
      #pragma omp cancellation point for
      const int64_t iClause = vUnsat[i];
      assert(1 <= iClause && iClause <= pFormula_->nClauses_);
      if(nSat_[iClause] != 0 || pFormula_->dummySat_[i]) {
        ans = false;
        #pragma omp cancel for
      }
    }
    return ans;
  }

  void Lock(const int64_t iClause) {
    std::atomic_flag& sync = syncs_[iClause % (cSyncContention * nSysCpus)];
    while(sync.test_and_set(std::memory_order_acq_rel)) {
      while (sync.test(std::memory_order_relaxed)); // keep it hot in cache
    }
  }

  void Unlock(const int64_t iClause) {
    std::atomic_flag& sync = syncs_[iClause % (cSyncContention * nSysCpus)];
    sync.clear(std::memory_order_release);
  }

  // TODO: ensure there are no duplicate clauses in the same vector (including negations),
  // otherwise non-concurrent version here will fail.
  // The sign of iVar must reflect the new value of the variable.
  // Returns the change in satisfiability: positive - more satisfiable, negative - more unsatisfiable.
  template<bool concurrent> int64_t FlipVar(const int64_t iVar, VCTrackingSet* unsatClauses, VCTrackingSet* front)
  {
    VCIndex balance = 0;
    //constexpr const int cChunkSize = (kRamPageBytes / sizeof(VCIndex));
    {
      const int8_t sgnTo = 1;
      const VCIndex nArcs = pFormula_->var2clause_.ArcCount(iVar, sgnTo);
      VCIndex ans = 0;
      //const int numThreads = std::min<int>(DivUp(nArcs, cChunkSize), nSysCpus);
      //#pragma omp parallel for reduction(+:ans) schedule(static, cChunkSize) num_threads(numThreads)
      for(VCIndex at=0; at<nArcs; at++) {
        const VCIndex iClause = pFormula_->var2clause_.GetTarget(iVar, sgnTo, at);
        const VCIndex aClause = llabs(iClause);
        if(pFormula_->dummySat_[aClause]) {
          continue;
        }
        assert(Signum(iClause) == 1);
        if constexpr(concurrent) {
          Lock(aClause);
        }
        assert(nSat_[aClause] < std::numeric_limits<TCounter>::max());
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
        if constexpr(concurrent) {
          Unlock(aClause);
        }
      }
      balance += ans;
    }
    {
      const int8_t sgnTo = -1;
      const VCIndex nArcs = pFormula_->var2clause_.ArcCount(iVar, sgnTo);
      VCIndex ans = 0;
      //const int numThreads = std::min<int>(DivUp(nArcs, cChunkSize), nSysCpus);
      //#pragma omp parallel for reduction(+:ans) schedule(static, cChunkSize) num_threads(numThreads)
      for(VCIndex at=0; at<nArcs; at++) {
        const VCIndex iClause = pFormula_->var2clause_.GetTarget(iVar, sgnTo, at);
        const VCIndex aClause = llabs(iClause);
        if(pFormula_->dummySat_[aClause]) {
          continue;
        }
        assert(Signum(iClause) == -1);
        if constexpr(concurrent) {
          Lock(aClause);
        }
        assert(nSat_[aClause] > 0);
        nSat_[aClause]--;
        if(nSat_[aClause] == 0) { // just became unsatisfied
          ans--;
          if(unsatClauses != nullptr) {
            unsatClauses->Add(aClause);
          }
          if(front != nullptr && front != unsatClauses) {
            front->Add(aClause);
          }
        }
        if constexpr(concurrent) {
          Unlock(aClause);
        }
      }
      balance += ans;
    }
    totSat_.fetch_add(balance, std::memory_order_relaxed);
    return balance;
  }

  VCTrackingSet GetUnsat() const {
    VCTrackingSet ans;
    #pragma omp parallel for schedule(static, kRamPageBytes)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_[i] == 0) {
        ans.Add(i);
      }
    }
    return ans;
  }

  VCTrackingSet GetUnsat(const SatTracker& oldSatTr, VCTrackingSet& newFront) const {
    VCTrackingSet ans;
    #pragma omp parallel for schedule(guided, kRamPageBytes)
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

  VCTrackingSet GetFront(const SatTracker& oldSatTr) {
    VCTrackingSet ans;
    #pragma omp parallel for schedule(guided, kRamPageBytes)
    for(int64_t i=1; i<=pFormula_->nClauses_; i++) {
      if(nSat_.get()[i].load(std::memory_order_relaxed) == 0) {
        if(oldSatTr.nSat_.get()[i].load(std::memory_order_relaxed) > 0) {
          ans.Add(i);
        }
      }
    }
    return ans;
  }

  int64_t UnsatCount() const {
    return pFormula_->nClauses_ - totSat_.load(std::memory_order_relaxed);
  }

  int64_t GradientDescend(Traversal& trav,
    const VCTrackingSet* considerClauses,
    bool& moved, BitVector& next, const int sortType,
    const VCIndex unsatCap, int64_t& nCombs, const int64_t maxCombs,
    VCTrackingSet& unsatClauses, VCTrackingSet& front,
    VCTrackingSet& origRevVars, const VCIndex nStartUnsat)
  {
    std::vector<MultiItem<VCIndex>> subsetVars, *pvVars = nullptr;
    if(considerClauses == nullptr) {
      pvVars = &vVars_;
    }
    else {
      subsetVars = pFormula_->ClauseFrontToVars(*considerClauses, next);
      pvVars = &subsetVars;
    }
    SortMultiItems(*pvVars, sortType);

    VCIndex minUnsat = unsatCap+1;
    VCTrackingSet bestRevVars;

    const VCTrackingSet startFront = front;
    VCTrackingSet revVars;
    // TODO: flip a random number of consecutive vars in each step (i.e. new random count in each step)
    for(int64_t k=0; k<int64_t(pvVars->size()) && nCombs < maxCombs; k++) {
      VCTrackingSet curRevVars;
      const int64_t aVar = (*pvVars)[k].item_;
      assert(1 <= aVar && aVar <= pFormula_->nVars_);
      const int64_t iVar = aVar * (next[aVar] ? 1 : -1);
      curRevVars.Add(aVar);
      const VCTrackingSet viableFront = (front.Size() == 0) ? unsatClauses : front;
      if( trav.IsSeenMove(viableFront, curRevVars) ) {
        continue;
      }
      revVars.Flip(aVar);
      if(trav.IsSeenMove(startFront, revVars)) {
        goto sgd_unflip_0;
      }
      next.Flip(aVar);
      if(trav.IsSeenAssignment(next)) {
        goto sgd_unflip_1;
      }
      FlipVar<false>(-iVar, &unsatClauses, &front);
      if( front.Size() != 0 && trav.IsSeenFront(front) ) {
        goto sgd_unflip_2;
      }

      {
        nCombs++;
        const VCIndex newUnsat = UnsatCount();
        if(k != 0) {
          trav.FoundMove(startFront, revVars);
        }
        trav.FoundMove(viableFront, curRevVars, next, newUnsat);
        if(newUnsat < minUnsat) {
          minUnsat = newUnsat;
          bestRevVars = revVars;
          if(newUnsat == 0) {
            revVars.Flip(aVar);
            // Return immediately the assignment satisfying the formula
            break;
          }
        }
        if(newUnsat < nStartUnsat) {
          revVars.Flip(aVar);
          // Don't unflip
          continue;
        }
      }

      // Flip back
sgd_unflip_2:
      FlipVar<false>(iVar, &unsatClauses, &front);
sgd_unflip_1:
      next.Flip(aVar);
sgd_unflip_0:
      revVars.Flip(aVar);
    }
    if(minUnsat <= unsatCap) {
      // Move to a good next assignment
      moved = true;
      // TODO: a dedicated method for computing this subset operation
      const VCTrackingSet toFlip = (revVars - bestRevVars) + (bestRevVars - revVars);
      std::vector<VCIndex> vRevVars = toFlip.ToVector();
      for(VCIndex i=0; i<VCIndex(vRevVars.size()); i++) {
        const VCIndex revV = vRevVars[i];
        next.Flip(revV);
        FlipVar<false>(revV * (next[revV] ? 1 : -1), &unsatClauses, &front);
      }
      assert(UnsatCount() == minUnsat);
      vRevVars = bestRevVars.ToVector();
      // TODO: FlipAll method
      for(VCIndex i=0; i<VCIndex(vRevVars.size()); i++) {
        origRevVars.Flip(vRevVars[i]);
      }
    } else {
      // There's no tolerable assignment.
      // Don't restore the initial parameters of |next|, |unsatClauses| and |front|
      moved = false;
    }
    return minUnsat;
  }

  int64_t ParallelGD(const bool preferMove, const int64_t varsAtOnce,
    std::vector<MultiItem<VCIndex>>& varFront, const int sortType,
    BitVector& next, Traversal& trav, VCTrackingSet& unsatClauses,
    VCTrackingSet& front, VCTrackingSet& origRevVars, int64_t minUnsat,
    int64_t& nCombs, bool& moved)
  {
    const VCTrackingSet startFront = front;
    SortMultiItems(varFront, sortType);
    VCTrackingSet revVars;
    for(int64_t i=0; i<int64_t(varFront.size()); i+=varsAtOnce) {
      std::vector<int64_t> selVars(varsAtOnce, 0);
      const int64_t nVars = std::min<int64_t>(varsAtOnce, int64_t(varFront.size()) - i);
      VCTrackingSet curRevVars;
      for(int64_t j=0; j<nVars; j++) {
        int64_t aVar, iVar;
        aVar = varFront[i+j].item_;
        assert(1 <= aVar && aVar <= pFormula_->nVars_);
        iVar = aVar * (next[aVar] ? 1 : -1);
        selVars[j] = iVar;
        curRevVars.Flip(aVar);
      }
      const VCTrackingSet viableFront = (front.Size() == 0) ? unsatClauses : front;
      if( trav.IsSeenMove(viableFront, curRevVars) ) {
        continue;
      }
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        revVars.Flip(aVar);
      }
      if( trav.IsSeenMove(startFront, revVars) ) {
        goto pgd_unflip_0;
      }
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
      }
      if(trav.IsSeenAssignment(next)) {
        goto pgd_unflip_1;
      }
      nCombs++;
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        FlipVar<false>(-iVar, &unsatClauses, &front);
      }
      if(front.Size() != 0 && trav.IsSeenFront(front)) {
        goto pgd_unflip_2;
      }
      {
        const int64_t newNUnsat = UnsatCount();
        if(i != 0) {
          trav.FoundMove(startFront, revVars);
        }
        trav.FoundMove(viableFront, curRevVars, next, newNUnsat);
        if(newNUnsat < minUnsat + (preferMove ? 1 : 0)) {
          moved = true;
          minUnsat = newNUnsat;
          for(int64_t j=0; j<nVars; j++) {
            const int64_t iVar = selVars[j];
            const int64_t aVar = llabs(iVar);
            origRevVars.Flip(aVar);
          }
          if(minUnsat == 0) {
            break;
          }
          continue;
        }
      }
pgd_unflip_2:
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        FlipVar<false>(iVar, &unsatClauses, &front);
      }
pgd_unflip_1:
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        next.Flip(aVar);
      }
pgd_unflip_0:
      for(int64_t j=0; j<nVars; j++) {
        const int64_t iVar = selVars[j];
        const int64_t aVar = llabs(iVar);
        revVars.Flip(aVar);
      }
    }
    return minUnsat;
  }

  int64_t NextUnsatCap(const int64_t nCombs, const VCTrackingSet& unsatClauses, const int64_t nStartUnsat) const {
    return std::max<int64_t>(nStartUnsat - 1, 
      std::max<int64_t>(
        unsatClauses.Size() * 2,
        DivUp(pFormula_->nVars_, nStartUnsat+1) + unsatClauses.Size()
      )
      - std::sqrt(nCombs) // *std::log2(nCombs+1)
    );
  }
};

using DefaultSatTracker = SatTracker<uint16_t>;

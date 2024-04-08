#include "Reduction.h"
#include "TrackingSet.h"
#include "Utils.h"
#include "SatTracker.h"
#include "Traversal.h"
#include "Exec.h"

#include <iostream>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <map>
#include <chrono>
#include <csignal>
#include <unistd.h>
#include <dlfcn.h>

std::vector<std::vector<uint64_t>> memComb;
uint64_t Comb(const int64_t n, const int64_t k) {
  if(k < 0 || k > n) {
    return 0;
  }
  if(n <= 0) {
    return 0;
  }
  if(n == k) {
    return 1;
  }
  if(k == 1) {
    return n;
  }
  while(int64_t(memComb.size())+1 < n) {
    memComb.emplace_back(memComb.size()+1);
  }
  uint64_t& mc = memComb[n-2][k-2];
  if(mc == 0) {
    mc = Comb(n, k-1) * (n-k+1);
  }
  return mc;
}

void signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  // TODO: save the maximally satisfying assignment here

  void (*_mcleanup)(void);
  _mcleanup = (void (*)(void))dlsym(RTLD_DEFAULT, "_mcleanup");
  if (_mcleanup == NULL) {
    std::cerr << "Unable to find gprof exit hook" << std::endl;
  }
  else _mcleanup();

  _exit(signum);
}

std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::steady_clock::now();
  const auto tmVeryStart = tmStart;

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }

  signal(SIGINT, signalHandler);
  // TODO: does it override the environment variable?
  omp_set_num_threads(nSysCpus);

  Formula formula;
  std::atomic<bool> provenUnsat = false;
  std::atomic<bool> maybeSat = formula.Load(argv[1]);
  if(!maybeSat) {
    provenUnsat = true;
    { // TODO: remove code duplication
      std::ofstream ofs(argv[2]);
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
    }
    return 0;
  }
  int64_t prevNUnsat = formula.nClauses_;

  std::cout << "Precomputing..." << std::endl;
  // Now there are both variable and clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );
  const uint64_t maxThreads = omp_get_max_threads();
  // TODO: shall it depend on the formula size? nVars_ or nClauses_
  const int64_t maxCombs = 1ULL << 11;
  Traversal trav;

  std::cout << "Choosing the best initial variable assignment..." << std::endl;

  std::atomic<uint64_t> nSequentialGD = 0, nWalk = 0, totCombs = 0;
  std::atomic<VCIndex> nGlobalUnsat = formula.nClauses_ + 1;
  std::mutex muGlobal;

  const uint32_t nCpusPerInit = DivUp<uint32_t>(nSysCpus, 3);
  omp_set_max_active_levels(2);
  #pragma omp parallel for schedule(dynamic, 1)
  for(int j=-1; j<=1; j++) {
    omp_set_max_active_levels(2);

    BitVector initAsg(formula.nVars_+1);
    DefaultSatTracker initSatTr(formula);
    switch(j) {
    case -1:
      break; // already all false
    case 0:
      initAsg.Randomize();
      break;
    case 1:
      initAsg.SetTrue();
      break;
    }
    VCTrackingSet initUnsatClauses = initSatTr.Populate(initAsg, nullptr);
    trav.OnSeenAssignment(initAsg, initUnsatClauses.Size());
    {
      std::unique_lock<std::mutex> lock(muGlobal);
      if(initUnsatClauses.Size() < nGlobalUnsat) {
        nGlobalUnsat = initUnsatClauses.Size();
      }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for(uint32_t i=0; i<nCpusPerInit; i++) {
      //std::mt19937_64 rng = GetSeededRandom();
      BitVector locAsg = initAsg;
      DefaultSatTracker locSatTr = initSatTr;
      VCTrackingSet locUnsatClauses = initUnsatClauses;
      VCTrackingSet locFront = locUnsatClauses;
      VCTrackingSet revVars;
      const VCTrackingSet startFront = locFront;
      int64_t nCombs = 0;
      while(nCombs < maxCombs) {
        const int8_t sortType = i % knSortTypes + kMinSortType; //rng() % knSortTypes + kMinSortType;
        if(locFront.Size() == 0 || trav.IsSeenFront(locFront)) {
          std::cout << "%";
          locFront = locUnsatClauses;
        }
        bool moved = false;
        VCIndex locBest = nGlobalUnsat.load(std::memory_order_relaxed);
        const int64_t altNUnsat = locSatTr.GradientDescend(
          trav, false, &locFront, moved, locAsg, sortType,
          //locSatTr.NextUnsatCap(nCombs, locUnsatClauses, locBest),
          locBest,
          nCombs, maxCombs, initUnsatClauses, locUnsatClauses, startFront, locFront, revVars,
          locBest
        );
        nSequentialGD.fetch_add(1);
        if(!moved) {
          break;
        }
        locBest = nGlobalUnsat.load(std::memory_order_relaxed);
        if(altNUnsat < locBest && locUnsatClauses != initUnsatClauses) {
          std::unique_lock<std::mutex> lock(muGlobal);
          if(altNUnsat < nGlobalUnsat) {
            nGlobalUnsat = altNUnsat;
          }
        }
      }
    }
  }
  formula.ans_ = trav.dfs_.back().assignment_;

  {
    auto tmEnd = std::chrono::steady_clock::now();
    double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
    double clausesPerSec = (prevNUnsat - nGlobalUnsat) / nSec;
    std::cout << "\tUnsatisfied clauses: " << nGlobalUnsat << " - elapsed " << nSec << " seconds, ";
    if(clausesPerSec >= 1 || clausesPerSec == 0) {
      std::cout << clausesPerSec << " clauses per second.";
    } else {
      std::cout << 1.0 / clausesPerSec << " seconds per clause.";
    }
    std::cout << std::endl;
    tmStart = tmEnd;
    prevNUnsat = nGlobalUnsat;
  }

  std::atomic<int> nMaybeSat = maxThreads;
  std::vector<Exec> execs(maxThreads);
  omp_set_max_active_levels(1);
  #pragma omp parallel num_threads(maxThreads)
  {
    const uint32_t iExec = omp_get_thread_num();
    assert(iExec < maxThreads);
    Exec& curExec = execs[iExec];
    curExec.pFormula_ = &formula;
    bool usedFA = false;
    if(!trav.PopIfNotWorse(curExec.next_, nGlobalUnsat)) {
      curExec.next_ = formula.ans_;
      usedFA = true;
    }
    curExec.satTr_ = DefaultSatTracker(formula);
    curExec.unsatClauses_ = curExec.satTr_.Populate(curExec.next_, usedFA ? nullptr : &curExec.front_);
    curExec.nStartUnsat_ = nGlobalUnsat;
    #pragma omp barrier
    while(nGlobalUnsat > 0) {
      bool allowDuplicateFront = false;
      while(nGlobalUnsat >= curExec.nStartUnsat_) {
        if(curExec.front_.Size() == 0 || trav.IsSeenFront(curExec.front_)) {
          curExec.RandomizeFront();
          std::cout << "%";
        }

        VCIndex bestUnsat = formula.nClauses_+1;
        VCTrackingSet bestRevVars;
        // TODO: shall we get only vars for the front here, or for all the unsatisfied clauses?
        curExec.varFront_ = formula.ClauseFrontToVars(curExec.front_, curExec.next_);
        const VCIndex startNIncl = 2, endNIncl=std::min<VCIndex>(curExec.varFront_.size(), 5);
        const VCIndex rangeNIncl = endNIncl - startNIncl + 1;
        bool allCombs;
        if(curExec.varFront_.size() < std::log2(maxCombs+1)) {
          // Consider all combinations without different methods of sorting
          allCombs = true;
        } else {
          // Consider some combinations for different methods of sorting and number of included vars
          allCombs = false;
        }

        VCTrackingSet stepRevs;
        VCTrackingSet oldFront = curExec.front_, oldUnsatCs = curExec.unsatClauses_;
        if(allCombs) {
          uint64_t curComb = 0;
          for(int i=0; i<int(curExec.varFront_.size()); i++) {
            if(curComb & (1ULL<<i)) {
              const VCIndex iVar = curExec.varFront_[i].item_;
              stepRevs.Add(iVar);
              curExec.next_.Flip(iVar);
              curExec.satTr_.FlipVar<false>(
                iVar * (curExec.next_[iVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_);
            }
          }
          const uint64_t limitComb = 1ULL<<curExec.varFront_.size();
          int64_t nCombs = 0;
          while(curComb < limitComb) {
            const VCIndex curNUnsat = curExec.unsatClauses_.Size();
            if( (curNUnsat == 0)
              || ( (allowDuplicateFront || curExec.front_.Size() == 0 || !trav.IsSeenFront(curExec.front_))
                && !trav.IsSeenMove(oldUnsatCs, oldFront, stepRevs) && !trav.IsSeenAssignment(curExec.next_) ) )
            {
              nCombs++;
              trav.FoundMove(oldFront, stepRevs, curExec.next_, curNUnsat);
              if( curExec.unsatClauses_.Size() < nGlobalUnsat || allowDuplicateFront || !trav.IsSeenFront(curExec.unsatClauses_) )
              {
                if(curNUnsat < bestUnsat) {
                  bestUnsat = curNUnsat;
                  bestRevVars = stepRevs;
                }
              }
            }
            int i=0;
            for(; i<int(curExec.varFront_.size()); i++) {
              curComb ^= 1ULL << i;
              const VCIndex aVar = curExec.varFront_[i].item_;
              stepRevs.Flip(aVar);
              curExec.next_.Flip(aVar);
              curExec.satTr_.FlipVar<false>(
                aVar * (curExec.next_[aVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_);
              if( (curComb & (1ULL << i)) != 0 ) {
                break;
              }
            }
            if(i >= int(curExec.varFront_.size())) {
              break;
            }
          }
          totCombs.fetch_add( nCombs );
        } else { // !allCombs
          curExec.sortType_ = curExec.rng_() % knSortTypes + kMinSortType;
          curExec.nIncl_ = curExec.rng_() % rangeNIncl + startNIncl;
          SortMultiItems(curExec.varFront_, curExec.sortType_);
          VCTrackingSet stepRevs;
          std::vector<VCIndex> incl(curExec.nIncl_, 0);
          uint64_t curComb;
          if(VCIndex(curExec.varFront_.size()) >= 64 + curExec.nIncl_) {
            curComb = curExec.rng_();
          } else if(VCIndex(curExec.varFront_.size()) <= curExec.nIncl_) {
            curComb = 0;
          } else {
            curComb = curExec.rng_() % (1ULL<<(VCIndex(curExec.varFront_.size()) - curExec.nIncl_));
          }
          while(__builtin_popcountll(curComb) > curExec.nIncl_) {
            curComb &= curComb-1;
          }
          VCIndex i=0;
          while(curComb != 0) {
            incl[i] = __builtin_ctzll(curComb);
            curComb &= curComb-1;
            i++;
          }
          if(i == 0) {
            incl[0] = 0;
            i++;
          }
          for(; i<curExec.nIncl_; i++) {
            incl[i] = incl[i-1] + 1;
            assert(0 <= incl[i] && incl[i] < VCIndex(curExec.varFront_.size()));
          }
          for(i=0; i<curExec.nIncl_; i++) {
            const VCIndex aVar = curExec.varFront_[incl[i]].item_;
            assert(0 < aVar && aVar <= formula.nVars_);
            stepRevs.Flip(aVar);
            curExec.next_.Flip(aVar);
            curExec.satTr_.FlipVar<false>( aVar * (curExec.next_[aVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_ );
          }
          int64_t nCombs = 0;
          for(;;) {
            //assert(execSatTr[iExec].Verify(execNext[iExec])); // TODO: very heavy
            if(nCombs >= maxCombs) {
              for(i=0; i<curExec.nIncl_; i++) {
                const VCIndex aVar = curExec.varFront_[incl[i]].item_;
                assert(0 < aVar && aVar <= formula.nVars_);
                stepRevs.Flip(aVar);
                curExec.next_.Flip(aVar);
                curExec.satTr_.FlipVar<false>( aVar * (curExec.next_[aVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_ );
              }
              break;
            }
            const VCIndex curNUnsat = curExec.unsatClauses_.Size();
            bool bFlipBack = true;
            if( (curNUnsat == 0)
              || ( (allowDuplicateFront || curExec.front_.Size() == 0 || !trav.IsSeenFront(curExec.front_))
                && !trav.IsSeenMove(oldUnsatCs, oldFront, stepRevs) && !trav.IsSeenAssignment(curExec.next_) ) )
            {
              nCombs++;
              trav.FoundMove(oldFront, stepRevs, curExec.next_, curNUnsat);
              if( curExec.unsatClauses_.Size() < nGlobalUnsat || allowDuplicateFront || !trav.IsSeenFront(curExec.unsatClauses_) )
              {
                if(curNUnsat < bestUnsat) {
                  bestUnsat = curNUnsat;
                  bestRevVars = stepRevs;
                }
                if(curNUnsat < curExec.nStartUnsat_) {
                  // Maybe we'll find an even better assignment with small modifications based on the current assignment
                  bFlipBack = false;
                }
              }
            }
            VCIndex i=curExec.nIncl_-1;
            for(; i>=0; i--) {
              if(bFlipBack) {
                const VCIndex aVar = curExec.varFront_[incl[i]].item_;
                stepRevs.Flip(aVar);
                curExec.next_.Flip(aVar);
                curExec.satTr_.FlipVar<false>(
                  aVar * (curExec.next_[aVar] ? 1 : -1),
                  &curExec.unsatClauses_, &curExec.front_
                );
              }
              if(incl[i] + curExec.nIncl_ - i < VCIndex(curExec.varFront_.size())) {
                break;
              }
            }
            if(i < 0) {
              break;
            }

            incl[i]++;
            {
              const VCIndex aVar = curExec.varFront_[incl[i]].item_;
              stepRevs.Flip(aVar);
              curExec.next_.Flip(aVar);
              curExec.satTr_.FlipVar<false>(
                aVar * (curExec.next_[aVar] ? 1 : -1),
                &curExec.unsatClauses_, &curExec.front_
              );
            }
            i++;
            for(; i<curExec.nIncl_; i++) {
              incl[i] = incl[i-1]+1;
              assert(incl[i] < VCIndex(curExec.varFront_.size()));
              const VCIndex aVar = curExec.varFront_[incl[i]].item_;
              stepRevs.Flip(aVar);
              curExec.next_.Flip(aVar);
              curExec.satTr_.FlipVar<false>(
                aVar * (curExec.next_[aVar] ? 1 : -1),
                &curExec.unsatClauses_, &curExec.front_
              );
            }
          }
          totCombs.fetch_add(nCombs);
        }

        trav.OnFrontExhausted(oldFront);
        trav.OnFrontExhausted(oldUnsatCs);

        if(bestUnsat >= formula.nClauses_) {
          std::cout << "#";

          if(oldFront != oldUnsatCs) {
            // Retry with full front
            curExec.front_ = curExec.unsatClauses_ = oldUnsatCs;
            continue;
          }

          if(trav.StepBack(curExec.next_)) {
            curExec.unsatClauses_ = curExec.satTr_.Populate(curExec.next_, &curExec.front_);
            assert(curExec.satTr_.UnsatCount() == curExec.unsatClauses_.Size());
            std::cout << "@";
            continue;
          }

          if(!allowDuplicateFront) {
            allowDuplicateFront = true;
            curExec.unsatClauses_ = oldUnsatCs;
            curExec.front_.Clear();
            std::cout << "X";
            continue;
          }

          // The current executor considers it unsatisfiable, but let's wait for the rest of executors
          nMaybeSat.fetch_sub(1);
          break;
        }

        std::cout << ">";
        nWalk++;
        
        VCTrackingSet toFlip = (stepRevs - bestRevVars) + (bestRevVars - stepRevs);
        std::vector<int64_t> vToFlip = toFlip.ToVector();
        for(int64_t i=0; i<int64_t(vToFlip.size()); i++) {
          const int64_t revV = vToFlip[i];
          curExec.next_.Flip(revV);
          curExec.satTr_.FlipVar<false>(revV * (curExec.next_[revV] ? 1 : -1), &curExec.unsatClauses_, nullptr);
        }
        assert(curExec.satTr_.UnsatCount() == bestUnsat);
        assert(curExec.unsatClauses_.Size() == bestUnsat);
        curExec.front_ = curExec.unsatClauses_ - oldUnsatCs;

        if(curExec.unsatClauses_.Size() < nGlobalUnsat) {
          std::unique_lock<std::mutex> lock(muGlobal);
          if(curExec.unsatClauses_.Size() < nGlobalUnsat) {
            nGlobalUnsat = curExec.unsatClauses_.Size();
            // Save the best partial assignment
            formula.ans_ = curExec.next_;
          }
          break;
        }

        if( curExec.front_.Size() == 0
          || (!allowDuplicateFront && curExec.unsatClauses_.Size() >= curExec.nStartUnsat_ && trav.IsSeenFront(curExec.front_)) )
        {
          // curExec.front_ = curExec.unsatClauses_; // full front
          curExec.RandomizeFront(); // random front
          std::cout << "%";
        }

        bestUnsat = formula.nClauses_ + 1;
        //bestRevVars.Clear();
        stepRevs.Clear();
        oldFront = curExec.front_;
        oldUnsatCs = curExec.unsatClauses_;
        
        int64_t nCombs = 0;
        bool moved;
        while( nCombs < maxCombs ) {
          if( curExec.front_.Size() == 0
            || (!allowDuplicateFront && curExec.unsatClauses_.Size() >= curExec.nStartUnsat_ && trav.IsSeenFront(curExec.front_)) )
          {
            curExec.RandomizeFront();
            std::cout << "%";
          }
          moved = false;
          const int8_t sortType = int8_t(curExec.rng_() % knSortTypes) + kMinSortType;
          curExec.satTr_.GradientDescend(
            trav, allowDuplicateFront, &curExec.front_, moved, curExec.next_, sortType,
            curExec.satTr_.NextUnsatCap(nCombs, curExec.unsatClauses_, nGlobalUnsat),
            nCombs, maxCombs, oldUnsatCs, curExec.unsatClauses_, oldFront, curExec.front_, stepRevs,
            nGlobalUnsat
          );
          nSequentialGD.fetch_add(1);
          if(!moved) {
            // The data structures are corrupted already (not rolled back)
            break;
          }
          if( curExec.unsatClauses_.Size() < nGlobalUnsat || allowDuplicateFront || !trav.IsSeenFront(curExec.unsatClauses_) ) {
            if(curExec.unsatClauses_.Size() < bestUnsat) {
              bestUnsat = curExec.unsatClauses_.Size();
              bestRevVars = stepRevs;
              if(bestUnsat < nGlobalUnsat) {
                break;
              }
            }
          }
        }

        if(bestUnsat < formula.nClauses_ + 1) {
          toFlip = (stepRevs - bestRevVars) + (bestRevVars - stepRevs);
          vToFlip = toFlip.ToVector();
          for(int64_t i=0; i<int64_t(vToFlip.size()); i++) {
            const int64_t revV = vToFlip[i];
            curExec.next_.Flip(revV);
            curExec.satTr_.FlipVar<false>(revV * (curExec.next_[revV] ? 1 : -1), &curExec.unsatClauses_, nullptr);
          }
          assert(curExec.satTr_.UnsatCount() == bestUnsat);
          assert(curExec.unsatClauses_.Size() == bestUnsat);
          curExec.front_ = curExec.unsatClauses_ - oldUnsatCs;

          if(curExec.unsatClauses_.Size() < nGlobalUnsat) {
            std::unique_lock<std::mutex> lock(muGlobal);
            if(curExec.unsatClauses_.Size() < nGlobalUnsat) {
              nGlobalUnsat = curExec.unsatClauses_.Size();
              formula.ans_ = curExec.next_;
            }
            // Perhaps Gradient Descent will greatly improve over a successful combination
            // break;
          }
        } else {
          // This doesn't hold - there can be a descent into a seen seet of unsat clauses
          // assert(stepRevs.Size() == 0);
          if(trav.StepBack(curExec.next_)) {
            curExec.unsatClauses_ = curExec.satTr_.Populate(curExec.next_, &curExec.front_);
            assert(curExec.satTr_.UnsatCount() == curExec.unsatClauses_.Size());
            std::cout << "@";
          }
        }
      }
      // TODO: can we eliminate some barriers e.g. this one or in the beginning of the loop?
      #pragma omp barrier
      if(nGlobalUnsat == 0) {  
        break;
      }
      assert(nMaybeSat >= 0);
      if(nMaybeSat == 0) {
        if(iExec == 0) {
          std::cout << "Unsatisfiable in all executors." << std::endl;
        }
        maybeSat = false;
        break;
      }
      if(iExec == 0) {
        nMaybeSat = maxThreads;
      }
      #pragma omp barrier
      // Pop the currently best assignments from the stack
      if(trav.PopIfNotWorse(curExec.next_, nGlobalUnsat)) {
        curExec.unsatClauses_ = curExec.satTr_.Populate(curExec.next_, &curExec.front_);
      } else {
        nMaybeSat.fetch_sub(1);
        curExec.next_ = formula.ans_;
        const VCTrackingSet oldUnsatCs = curExec.unsatClauses_;
        curExec.unsatClauses_ = curExec.satTr_.Populate(curExec.next_, nullptr);
        curExec.front_ = curExec.unsatClauses_ - oldUnsatCs;
      }
      curExec.nStartUnsat_ = nGlobalUnsat;
      #pragma omp barrier
      if(iExec == 0) {
        std::cout << "\n\tWalks: " << nWalk << ", Seen moves: " << trav.seenMove_.Size() << ", Stack: " << trav.dfs_.size()
          << ", Known assignments: " << trav.seenAssignment_.Size()
          << ", nCombinations: " << totCombs << ", nSequentialGD: " << nSequentialGD
          << ", Current wave: " << nMaybeSat << " different assignments.";
        
        auto tmEnd = std::chrono::steady_clock::now();
        double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
        double clausesPerSec = (prevNUnsat - nGlobalUnsat) / nSec;
        std::cout << "\n\tUnsatisfied clauses: " << nGlobalUnsat << " - elapsed " << nSec << " seconds, ";
        if(clausesPerSec >= 1 || clausesPerSec == 0) {
          std::cout << clausesPerSec << " clauses per second.";
        } else {
          std::cout << 1.0 / clausesPerSec << " seconds per clause.";
        }
        std::cout << " Time since very start: "
          << std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmVeryStart).count() / (60 * 1e9)
          << " minutes." << std::endl;
        tmStart = tmEnd;
        prevNUnsat = nGlobalUnsat;
        nMaybeSat = maxThreads;
      }
      #pragma omp barrier
    }
  }

  if(nGlobalUnsat == 0) {
    std::cout << "SATISFIED" << std::endl;
    assert(formula.SolWorks());
  }

  {
    std::ofstream ofs(argv[2]);
    if(provenUnsat) {
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
      return 0;
    }

    if(maybeSat) {
      assert( nGlobalUnsat == 0 );
      ofs << "s SATISFIABLE" << std::endl;
    } else {
      ofs << "s UNKNOWN" << std::endl;
      ofs << "c Unsatisfied clause count: " << nGlobalUnsat << std::endl;
    }
    int64_t nInLine = 0;
    for(int64_t i=1; i<=formula.nVars_; i++) {
      if(nInLine == 0) {
        ofs << "v ";
      }
      ofs << (formula.ans_[i] ? i : -i);
      nInLine++;
      if(nInLine >= 200) {
        nInLine = 0;
        ofs << "\n";
      } else {
        ofs << " ";
      }
    }
    ofs << "0" << std::endl;
  }
  return 0;
}

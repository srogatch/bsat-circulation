#include "Reduction.h"
#include "TrackingSet.h"
#include "Utils.h"
#include "SatTracker.h"
#include "Traversal.h"

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
  bool provenUnsat = false;
  bool maybeSat = formula.Load(argv[1]);
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
  std::atomic<int64_t> bestInit = formula.nClauses_ + 1;
  BitVector bestAsg;
  VCTrackingSet startFront;

  std::atomic<uint64_t> nSequentialGD = 0, nWalk = 0, totCombs = 0;
  std::mutex muBestUpdate;

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
    {
      std::unique_lock<std::mutex> lock(muBestUpdate);
      if(initUnsatClauses.Size() < bestInit) {
        bestInit = initUnsatClauses.Size();
        startFront = initUnsatClauses;
        bestAsg = initAsg;
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
      int64_t nCombs = 0;
      while(nCombs < maxCombs) {
        const int8_t sortType = i % knSortTypes + kMinSortType; //rng() % knSortTypes + kMinSortType;
        if(locFront.Size() == 0 || trav.IsSeenFront(locFront)) {
          std::cout << "%";
          locFront = locUnsatClauses;
        }
        bool moved = false;
        VCTrackingSet oldUnsatCs = locUnsatClauses;
        VCIndex locBest = bestInit.load(std::memory_order_relaxed);
        const int64_t altNUnsat = locSatTr.GradientDescend(
          trav, &locFront, moved, locAsg, sortType,
          //locSatTr.NextUnsatCap(nCombs, locUnsatClauses, locBest),
          locBest,
          nCombs, maxCombs, locUnsatClauses, locFront, revVars, locBest
        );
        nSequentialGD.fetch_add(1);
        if(!moved) {
          break;
        }
        locFront = locUnsatClauses - oldUnsatCs;
        locBest = bestInit.load(std::memory_order_relaxed);
        if(altNUnsat < locBest) {
          std::unique_lock<std::mutex> lock(muBestUpdate);
          if(altNUnsat < bestInit) {
            bestInit = altNUnsat;
            startFront = locFront;
            bestAsg = locAsg;
          }
        }
      }
    }
  }
  DefaultSatTracker satTr(formula);
  formula.ans_ = bestAsg;
  VCTrackingSet front = startFront;
  VCTrackingSet unsatClauses = satTr.Populate(formula.ans_, nullptr);
  assert(unsatClauses.Size() == bestInit);
  assert( satTr.Verify(formula.ans_) );

  BitVector maxPartial;
  int64_t nStartUnsat;

  while(maybeSat) {
    maxPartial = formula.ans_;
    nStartUnsat = unsatClauses.Size();
    assert(satTr.UnsatCount() == nStartUnsat);
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    auto tmEnd = std::chrono::steady_clock::now();
    double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
    double clausesPerSec = (prevNUnsat - nStartUnsat) / nSec;
    std::cout << "\tUnsatisfied clauses: " << nStartUnsat << " - elapsed " << nSec << " seconds, ";
    if(clausesPerSec >= 1 || clausesPerSec == 0) {
      std::cout << clausesPerSec << " clauses per second.";
    } else {
      std::cout << 1.0 / clausesPerSec << " seconds per clause.";
    }
    std::cout << " Time since very start: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmVeryStart).count() / (60 * 1e9)
      << " minutes." << std::endl;
    tmStart = tmEnd;
    prevNUnsat = nStartUnsat;
    bool allowDuplicateFront = false;
    while(unsatClauses.Size() >= nStartUnsat) {
      assert(satTr.UnsatCount() == unsatClauses.Size() && satTr.ReallyUnsat(unsatClauses));
      if(front.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(front))) {
        // TODO: maybe select a random subset here?
        front = unsatClauses;
        std::cout << "%";
      }

      std::cout << "C";

      std::atomic<VCIndex> bestUnsat = formula.nClauses_+1;
      VCTrackingSet bestRevVars;
      // TODO: shall we get only vars for the |front| here, or for all the |unsatClauses| ?
      std::vector<MultiItem<VCIndex>> baseVarFront = formula.ClauseFrontToVars(unsatClauses, formula.ans_);

      bool allCombs = false;
      if(baseVarFront.size() < std::log2(uint64_t(nSysCpus)*maxCombs+1)) {
        // Consider all combinations without different methods of sorting
        allCombs = true;
        std::cout << "a";
      } else {
        // Consider some combinations for different methods of sorting and number of included 
        std::cout << "p";
      }
      std::cout.flush();
      const VCIndex startNIncl = 2, endNIncl=std::min<VCIndex>(baseVarFront.size(), 5);
      const VCIndex rangeNIncl = endNIncl - startNIncl + 1;
      struct Exec {
        std::vector<MultiItem<VCIndex>> varFront_;
        DefaultSatTracker satTr_;
        BitVector next_;
        VCTrackingSet unsatClauses_ = true;
        VCTrackingSet front_ = true;
        //VCTrackingSet improvingRevs_;
        uint64_t firstComb_;
        int8_t nIncl_;
        int8_t sortType_;
      };
      std::vector<Exec> execs(maxThreads);
      #pragma omp parallel for schedule(static, 1) num_threads(maxThreads)
      for(uint32_t iExec=0; iExec<maxThreads; iExec++) {
        Exec& curExec = execs[iExec];
        curExec.satTr_ = satTr;
        curExec.next_ = formula.ans_;
        curExec.unsatClauses_ = unsatClauses;
        curExec.front_ = front;
        if(allCombs) {
          // Don't try combination 0 - nothing flipped - because we're already at it
          curExec.firstComb_ = ((1ULL<<baseVarFront.size()) * iExec) / maxThreads + 1;
        } else {
          std::mt19937_64 rng = GetSeededRandom();
          curExec.varFront_ = baseVarFront;
          curExec.nIncl_ = rng() % rangeNIncl + startNIncl;
          curExec.sortType_ = rng() % knSortTypes + kMinSortType;
          // TODO: this produces a very skewed distributions where the same combinations are traversed multiple times for low bits
          //curExec.firstComb_ = (1ULL << (rng() % std::min<VCIndex>(64, baseVarFront.size())));
          curExec.firstComb_ = ((1ULL << std::min<VCIndex>(32, baseVarFront.size())) * iExec) / maxThreads + 1;
        }
        VCTrackingSet stepRevs;
        if(allCombs) {
          uint64_t curComb = curExec.firstComb_;
          for(int i=0; i<int(baseVarFront.size()); i++) {
            if(curComb & (1ULL<<i)) {
              const VCIndex iVar = baseVarFront[i].item_;
              stepRevs.Add(iVar);
              curExec.next_.Flip(iVar);
              curExec.satTr_.FlipVar<false>(
                iVar * (curExec.next_[iVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_);
            }
          }
          const uint64_t limitComb = (iExec+1 < maxThreads) ? execs[iExec+1].firstComb_ : 1ULL<<baseVarFront.size();
          int64_t nCombs = 0;
          while(curComb < limitComb) {
            const VCIndex curNUnsat = curExec.unsatClauses_.Size();
            const VCTrackingSet& viableFront = (curExec.front_.Size() == 0) ? curExec.unsatClauses_ : curExec.front_;
            // TODO: perhaps cutting off front here is too restrictive
            if( (curNUnsat == 0)
              || (!trav.IsSeenFront(viableFront) && !trav.IsSeenMove(front, stepRevs) && !trav.IsSeenAssignment(curExec.next_)) ) 
            {
              nCombs++;
              trav.FoundMove(front, stepRevs, curExec.next_, curNUnsat);
              if(curNUnsat < bestUnsat.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lock(muBestUpdate);
                if(curNUnsat < bestUnsat.load(std::memory_order_acquire)) {
                  bestUnsat.store(curNUnsat, std::memory_order_release);
                  bestRevVars = stepRevs;
                }
                // if(curNUnsat < nStartUnsat) {
                //   execImprovingRevs[iExec] = stepRevs;
                // }
              }
            }
            int i=0;
            for(; i<int(baseVarFront.size()); i++) {
              curComb ^= 1ULL << i;
              const VCIndex iVar = baseVarFront[i].item_;
              stepRevs.Add(iVar);
              curExec.next_.Flip(iVar);
              curExec.satTr_.FlipVar<false>(
                iVar * (curExec.next_[iVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_);
              if( (curComb & (1ULL << i)) != 0 ) {
                break;
              }
            }
            if(i >= int(baseVarFront.size()) || curComb >= limitComb) {
              break;
            }
          }
          totCombs.fetch_add( nCombs );
        } else { // !allCombs
          SortMultiItems(curExec.varFront_, curExec.sortType_);
          VCTrackingSet stepRevs;
          std::vector<VCIndex> incl(curExec.nIncl_, 0);
          uint64_t curComb = curExec.firstComb_;
          while(__builtin_popcountll(curComb) > curExec.nIncl_) {
            curComb &= curComb-1;
          }
          VCIndex i=0;
          while(curComb != 0) {
            incl[i] = __builtin_ctzll(curComb);
            curComb &= curComb-1;
            i++;
          }
          for(; i<curExec.nIncl_; i++) {
            incl[i] = incl[i-1] + 1;
          }
          for(i=0; i<curExec.nIncl_; i++) {
            const VCIndex aVar = curExec.varFront_[incl[i]].item_;
            stepRevs.Flip(aVar);
            curExec.next_.Flip(aVar);
            curExec.satTr_.FlipVar<false>( aVar * (curExec.next_[aVar] ? 1 : -1), &curExec.unsatClauses_, &curExec.front_ );
          }
          int64_t nCombs = 0;
          for(;;) {
            //assert(execSatTr[iExec].Verify(execNext[iExec])); // TODO: very heavy
            if(nCombs >= maxCombs) {
              // No need to cleanup - there is 1:1 mapping between threads and data structures
              break;
            }
            const VCTrackingSet& viableFront = (curExec.front_.Size() == 0) ? curExec.unsatClauses_ : curExec.front_;
            const VCIndex curNUnsat = curExec.unsatClauses_.Size();
            bool bFlipBack = true;
            if( (curNUnsat == 0)
              || ( (allowDuplicateFront || !trav.IsSeenFront(viableFront))
                && !trav.IsSeenMove(front, stepRevs) && !trav.IsSeenAssignment(curExec.next_) ) )
            {
              nCombs++;
              trav.FoundMove(front, stepRevs, curExec.next_, curNUnsat);
              if(curNUnsat < bestUnsat.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lock(muBestUpdate);
                if(curNUnsat < bestUnsat.load(std::memory_order_acquire)) {
                  bestUnsat.store(curNUnsat, std::memory_order_release);
                  bestRevVars = stepRevs;
                }
                if(curNUnsat < nStartUnsat) {
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
      }

      //if(allCombs) {
      trav.OnFrontExhausted(front);
      //}

      if(bestUnsat >= formula.nClauses_) {
        std::cout << "#";

        trav.OnFrontExhausted(front);

        if(front != unsatClauses) {
          // Retry with full/random front
          front = unsatClauses;
          continue;
        }

        if(trav.StepBack(formula.ans_)) {
          unsatClauses = satTr.Populate(formula.ans_, &front);
          assert(satTr.UnsatCount() == unsatClauses.Size());
          std::cout << "@";
          continue;
        }

        if(!allowDuplicateFront) {
          allowDuplicateFront = true;
          std::cout << "X";
          continue;
        }

        // Unsatisfiable
        std::cout << "...Nothing reversed - unsatisfiable..." << std::endl;
        maybeSat = false;
        break;
      }

      std::cout << ">";
      nWalk++;

      VCTrackingSet oldUnsatCs = unsatClauses;
      std::vector<int64_t> vBestRevVars = bestRevVars.ToVector();
      for(int64_t i=0; i<int64_t(vBestRevVars.size()); i++) {
        const int64_t revV = vBestRevVars[i];
        formula.ans_.Flip(revV);
        satTr.FlipVar<false>(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, nullptr);
      }
      assert(satTr.UnsatCount() == bestUnsat);
      assert(unsatClauses.Size() == bestUnsat);
      front = unsatClauses - oldUnsatCs;

      if(unsatClauses.Size() < nStartUnsat) {
        break;
      }

      if( front.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(front)) ) {
        std::cout << "%";
        front = unsatClauses;
      }

      std::cout << "S";
      std::cout.flush();
      bestUnsat = formula.nClauses_ + 1;
      bestRevVars.Clear();
      std::atomic<bool> newEpoch = false;
      // omp_set_max_active_levels(1);
      #pragma omp parallel for num_threads(nSysCpus)
      for(uint32_t i=0; i<nSysCpus; i++) {
        VCTrackingSet locUnsatClauses = unsatClauses;
        VCTrackingSet locFront = front;
        DefaultSatTracker locSatTr(satTr);
        BitVector locAsg = formula.ans_;
        VCTrackingSet stepRevs;
        //std::mt19937_64 rng = GetSeededRandom();

        int64_t nCombs = 0;
        int64_t newUnsat = locUnsatClauses.Size();
        bool moved;
        while( !newEpoch.load(std::memory_order_acquire) && nCombs < maxCombs ) {
          if(locFront.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(locFront))) {
            // TODO: maybe select a random subset of the locUnsatClauses here?
            locFront = locUnsatClauses;
          }
          moved = false;
          //const int8_t sortType = i % knSortTypes + kMinSortType;
          const int8_t sortType = i % knSortTypes + kMinSortType;
          VCTrackingSet oldUnsatCs = locUnsatClauses;
          newUnsat = locSatTr.GradientDescend(
            trav, &locFront, moved, locAsg, sortType,
            locSatTr.NextUnsatCap(nCombs, locUnsatClauses, nStartUnsat),
            //nStartUnsat-1,
            nCombs, maxCombs, locUnsatClauses, locFront, stepRevs, nStartUnsat
          );
          nSequentialGD.fetch_add(1);
          if(!moved) {
            // The data structures are corrupted already (not rolled back)
            break;
          }
          {
            std::unique_lock<std::mutex> lock(muBestUpdate);
            if(locUnsatClauses.Size() < bestUnsat) {
              bestUnsat = locUnsatClauses.Size();
              bestRevVars = stepRevs;
            }
          }
          if(newUnsat < nStartUnsat) {
            newEpoch.store(true, std::memory_order_release);
            break;
          }
          locFront = locUnsatClauses - oldUnsatCs;
        }
      }
      if(bestUnsat < formula.nClauses_) {
        oldUnsatCs = unsatClauses;
        vBestRevVars = bestRevVars.ToVector();
        //#pragma omp parallel for schedule(guided, kCacheLineSize)
        for(int64_t i=0; i<int64_t(vBestRevVars.size()); i++) {
          const int64_t revV = vBestRevVars[i];
          formula.ans_.Flip(revV);
          satTr.FlipVar<true>(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, nullptr);
        }
        front = unsatClauses - oldUnsatCs;
        assert(satTr.UnsatCount() == bestUnsat);
        assert(unsatClauses.Size() == bestUnsat);
      } else {
        trav.OnFrontExhausted(front);
        front.Clear();
      }
    }
    std::cout << "\n\tWalks: " << nWalk << ", Seen moves: " << trav.seenMove_.Size() << ", Stack: " << trav.dfs_.size()
      << ", Known assignments: " << trav.seenAssignment_.Size()
      << ", nCombinations: " << totCombs << ", nSequentialGD: " << nSequentialGD << std::endl;
  }

  {
    std::ofstream ofs(argv[2]);
    if(provenUnsat) {
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
      return 0;
    }

    if(maybeSat) {
      assert(formula.SolWorks());
      ofs << "s SATISFIABLE" << std::endl;
    } else {
      formula.ans_ = maxPartial;
      ofs << "s UNKNOWN" << std::endl;
      ofs << "c Unsatisfied clause count: " << nStartUnsat << std::endl;
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

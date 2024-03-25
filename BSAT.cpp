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
constexpr const float kMaxInitSec = 1.0;

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
  Traversal trav;

  std::cout << "Precomputing..." << std::endl;
  // Now there are both variable and clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  std::cout << "Choosing the best initial variable assignment..." << std::endl;
  std::atomic<int64_t> bestInit = formula.nClauses_ + 1;
  BitVector bestAsg;
  VCTrackingSet startFront;

  std::atomic<int64_t> nParallelGD = 0, nSequentialGD = 0, nWalk = 0;
  std::mutex muBestUpdate;

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
    const int64_t nInnerLoop = std::max<int64_t>(1, nSysCpus / 3);
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i=0; i<nInnerLoop; i++) {
      //std::mt19937_64 rng = GetSeededRandom();
      BitVector locAsg = initAsg;
      DefaultSatTracker locSatTr = initSatTr;
      VCTrackingSet locUnsatClauses = initUnsatClauses;
      VCTrackingSet locFront = locUnsatClauses;
      VCTrackingSet revVars;
      int64_t nCombs = 0;
      const auto tmInitStart = std::chrono::steady_clock::now();
      for(;;) {
        const auto tmNow = std::chrono::steady_clock::now();
        const double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmNow - tmInitStart).count() / 1e9;
        if(nSec > kMaxInitSec) {
          break; 
        }
        const int8_t sortType = i % knSortTypes + kMinSortType; //rng() % knSortTypes + kMinSortType;
        bool moved = false;
        if(locFront.Size() == 0 || trav.IsSeenFront(locFront)) {
          std::cout << "%";
          locFront = locUnsatClauses;
        }
        VCIndex locBest = bestInit.load(std::memory_order_relaxed);
        const int64_t altNUnsat = locSatTr.GradientDescend(
          trav, &locFront, moved, locAsg, sortType,
          locSatTr.NextUnsatCap(nCombs, locUnsatClauses, locBest),
          nCombs, locUnsatClauses, locFront, revVars
        );
        nSequentialGD.fetch_add(1);
        if(!moved) {
          break;
        }
        locBest = bestInit.load(std::memory_order_relaxed);
        if(altNUnsat < locBest) {
          std::unique_lock<std::mutex> lock(muBestUpdate);
          if(altNUnsat < bestInit) {
            bestInit = altNUnsat;
            startFront = locFront;
            bestAsg = locAsg;
          }
        }
        locBest = bestInit.load(std::memory_order_relaxed);
      }
    }
  }
  DefaultSatTracker satTr(formula);
  formula.ans_ = bestAsg;
  VCTrackingSet front = startFront;
  VCTrackingSet unsatClauses = satTr.Populate(formula.ans_, nullptr);
  assert(unsatClauses.Size() == bestInit);

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
        front = unsatClauses;
        std::cout << "%";
      }

      int64_t bestUnsat = formula.nClauses_+1;
      VCTrackingSet bestRevVars;

      std::vector<MultiItem<VCIndex>> varFront = formula.ClauseFrontToVars(unsatClauses, formula.ans_);
      const int64_t startNIncl = 1;
      const int64_t endNIncl = std::min<int64_t>(varFront.size(), 4);
      const int64_t nInnerLoop = nSysCpus / (endNIncl - startNIncl + 1);
      std::cout << "P"; // << varFront.size() << "," << unsatClauses.Size();
      //std::cout.flush();
      #pragma omp parallel for schedule(dynamic, 1) collapse(2)
      for(int64_t nIncl=startNIncl; nIncl<=endNIncl; nIncl++) {
        // 0: shuffle
        // -1: reversed heap
        // 1: heap
        // -2: reversed full sort
        // 2: full sort
        for(int64_t i=0; i<nInnerLoop; i++) {
          const int8_t sortType = i % knSortTypes + kMinSortType;
          BitVector next = formula.ans_;
          DefaultSatTracker newSatTr = satTr;
          VCTrackingSet stepRevs;
          std::vector<MultiItem<VCIndex>> locVarFront = varFront;
          VCTrackingSet locFront = front;
          VCTrackingSet locUnsatCs = unsatClauses;
          int64_t nCombs = 0;
          for(;;) {
            bool moved = false;
            const int64_t curNUnsat = newSatTr.ParallelGD(
              true, nIncl, locVarFront, sortType, next, trav, locUnsatCs, locFront, stepRevs, 
              newSatTr.NextUnsatCap(nCombs, locUnsatCs, nStartUnsat), nCombs, moved, 0
            );

            if(moved) {
              std::unique_lock<std::mutex> lock(muBestUpdate);
              if(curNUnsat < bestUnsat) {
                bestUnsat = curNUnsat;
                bestRevVars = stepRevs;
              }
            } else {
              break;
            }
          }
        }
      }
      nParallelGD.fetch_add((endNIncl - startNIncl + 1) * knSortTypes);

      if(bestUnsat >= formula.nClauses_) {
        std::cout << "#";

        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        if(!allowDuplicateFront) {
          trav.OnFrontExhausted(front);
        }

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
        satTr.FlipVar<true>(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, nullptr);
      }
      front = unsatClauses - oldUnsatCs;
      assert(satTr.UnsatCount() == bestUnsat);
      assert(unsatClauses.Size() == bestUnsat);

      if(unsatClauses.Size() < nStartUnsat) {
        break;
      }

      std::cout << "S";
      bestUnsat = formula.nClauses_ + 1;
      bestRevVars.Clear();

      omp_set_max_active_levels(1);
      #pragma omp parallel for num_threads(nSysCpus)
      for(uint32_t i=0; i<nSysCpus; i++) {
        VCTrackingSet locUnsatClauses = unsatClauses;
        VCTrackingSet locFront = front, nextFront;
        DefaultSatTracker locSatTr(satTr);
        BitVector locAsg = formula.ans_;
        VCTrackingSet stepRevs;
        std::mt19937_64 rng = GetSeededRandom();

        int64_t newUnsat = locUnsatClauses.Size();
        bool moved;
        int64_t nCombs = 0;
        for(;;) {
          if(locFront.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(locFront))) {
            locFront = locUnsatClauses;
          }
          moved = false;
          //const int8_t sortType = i % knSortTypes + kMinSortType;
          const int8_t sortType = rng() % knSortTypes + kMinSortType;
          newUnsat = locSatTr.GradientDescend(
            trav, &locFront, moved, locAsg, sortType,
            locSatTr.NextUnsatCap(nCombs, locUnsatClauses, nStartUnsat),
            nCombs, locUnsatClauses, locFront, stepRevs
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
          if(newUnsat == 0) {
            break;
          }
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
      << ", nParallelGD: " << nParallelGD << ", nSequentialGD: " << nSequentialGD << std::endl;
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

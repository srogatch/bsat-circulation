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

std::vector<std::vector<uint64_t>> memAccComb;
uint64_t AccComb(const int64_t n, const int64_t k) {
  if(n <= 0 || k <= 0 || k > n) {
    return 0;
  }
  if(n == k) {
    return 1;
  }
  if(k == 1) {
    return n;
  }
  if(n == 1) {
    return 0;
  }
  while(int64_t(memAccComb.size())+1 < n) {
    memAccComb.emplace_back(memAccComb.size()+1);
  }
  uint64_t& mac = memAccComb[n-2][k-2];
  if(mac == 0) {
    mac = AccComb(n, k-1) + Comb(n, k);
  }
  return mac;
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
  // Enable nested parallelism
  omp_set_max_active_levels(omp_get_supported_active_levels());

  Formula formula;
  formula.Load(argv[1]);
  int64_t prevNUnsat = formula.nClauses_;
  Traversal trav;

  std::cout << "Precomputing..." << std::endl;
  // Now there are both variable and clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  std::cout << "Choosing the best initial variable assignment..." << std::endl;
  int64_t bestInit = formula.nClauses_ + 1;
  VCTrackingSet startFront;

  DefaultSatTracker satTr(formula);
  satTr.Populate(formula.ans_);
  #pragma omp parallel for schedule(dynamic, 1)
  for(int init=-1; init<=1; init++) {
    BitVector initAsg(formula.nVars_+1);
    DefaultSatTracker initSatTr(formula);
    switch(init) {
    case -1:
      break; // already all false
    case 0:
      initAsg.Randomize();
      break;
    case 1:
      initAsg.SetTrue();
      break;
    }
    VCTrackingSet initUnsatClauses = initSatTr.Populate(initAsg);
    VCTrackingSet initFront = initUnsatClauses;
    #pragma omp parallel for schedule(dynamic, 1)
    for(int sortType=kMinSortType; sortType<=kMaxSortType; sortType++) {
      BitVector locAsg = initAsg;
      DefaultSatTracker locSatTr = initSatTr;
      VCTrackingSet locUnsatClauses = initUnsatClauses;
      VCTrackingSet locFront = locUnsatClauses, locNextFront;
      for(;;) {
        bool moved = false;
        const int64_t altNUnsat = locSatTr.GradientDescend(
          false, trav, &locUnsatClauses, locUnsatClauses, initFront,
          locNextFront, locUnsatClauses.Size(), moved, locAsg,
          sortType
        );
        if(!moved) {
          break;
        }
        // This suspends all OpenMP threads in the entire program who try to enter this synchronized section
        // Naming the such section as "updateBestInit" lets other sections execute while this one is blocked.
        #pragma omp critical(updateBestInit)
        {
          if(altNUnsat < bestInit) {
            bestInit = altNUnsat;
            startFront = locNextFront;
            formula.ans_ = locAsg;
          }
        }
        locFront = locNextFront;
        locNextFront.Clear();
      }
    }
  }
  VCTrackingSet front = startFront;
  VCTrackingSet unsatClauses = satTr.Populate(formula.ans_);
  assert(unsatClauses.Size() == bestInit);

  BitVector maxPartial;
  bool maybeSat = true;
  bool provenUnsat = false;
  int64_t nStartUnsat;
  int64_t nParallelGD = 0, nSequentialGD = 0, nWalk = 0;

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

      std::vector<MultiItem<VCIndex>> varFront = formula.ClauseFrontToVars(front, formula.ans_);
      const int64_t startNIncl = 1;
      const int64_t endNIncl = std::min<int64_t>(varFront.size(), 4);
      std::cout << "P"; // << varFront.size() << "," << unsatClauses.Size();
      //std::cout.flush();
      #pragma omp parallel for schedule(dynamic, 1)
      for(int64_t nIncl=startNIncl; nIncl<=endNIncl; nIncl++) {
        // 0: shuffle
        // -1: reversed heap
        // 1: heap
        // -2: reversed full sort
        // 2: full sort
        #pragma omp parallel for schedule(dynamic, 1)
        for(int8_t sortType=kMinSortType; sortType<=kMaxSortType; sortType++) {
          BitVector next = formula.ans_;
          DefaultSatTracker newSatTr = satTr;
          VCTrackingSet stepRevs;
          std::vector<MultiItem<VCIndex>> locVarFront = varFront;
          bool moved = false;
          const int64_t curNUnsat = newSatTr.ParallelGD(
            true, nIncl, locVarFront, sortType, next, trav, nullptr, front, stepRevs, 
            newSatTr.NextUnsatCap(unsatClauses, nStartUnsat), moved, 0
          );

          // This suspends all OpenMP threads in the entire program who try to enter this synchronized section.
          // Naming the such section as "updateBestInit" lets other sections execute while this one is blocked.
          #pragma omp critical(updateBestUnsat)
          if( moved && curNUnsat < bestUnsat ) {
            bestUnsat = curNUnsat;
            bestRevVars = stepRevs;
          }
        }
      }
      nParallelGD += (endNIncl - startNIncl + 1) * 5;

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
          unsatClauses = satTr.Populate(formula.ans_);
          front = unsatClauses;
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

      front.Clear();
      std::vector<int64_t> vBestRevVars = bestRevVars.ToVector();
      #pragma omp parallel for schedule(dynamic, 1)
      for(int64_t i=0; i<int64_t(vBestRevVars.size()); i++) {
        const int64_t revV = vBestRevVars[i];
        formula.ans_.Flip(revV);
        satTr.FlipVar<true>(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, &front);
      }
      assert(satTr.UnsatCount() == bestUnsat);
      assert(unsatClauses.Size() == bestUnsat);
      // TODO: this is too heavy
      // assert(satTr.Verify(formula.ans_));

      if(unsatClauses.Size() < nStartUnsat) {
        break;
      }

      int64_t newUnsat = unsatClauses.Size();
      std::cout << "S";
      bool moved;
      int64_t nCurIts = 0;
      for(;;) {
        VCTrackingSet oldFront;
        if(front.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(front))) {
          oldFront = unsatClauses;
          std::cout << "%";
        } else {
          oldFront = front;
        }
        VCIndex oldFrontSize = oldFront.Size();
        front.Clear();
        moved = false;
        newUnsat = satTr.GradientDescend( unsatClauses.Size() >= nStartUnsat, trav, &oldFront, unsatClauses, oldFront, front,
          satTr.NextUnsatCap(unsatClauses, nStartUnsat) - nCurIts, moved, formula.ans_,
          nSequentialGD%knSortTypes + kMinSortType
        );
        nCurIts += oldFrontSize;
        nSequentialGD++;
        if(!moved || newUnsat == 0 || satTr.NextUnsatCap(unsatClauses, nStartUnsat) - nCurIts <= 0) {
          break;
        }
        // TODO: what if newUnsat > oldUnsat? Breaking at this point is empirically inefficient (progress stops).
        // Perhaps a better solution would be restoring the previous assignment, but it's slow.
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

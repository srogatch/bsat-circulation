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
  while(memComb.size()+1 < n) {
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
  while(memAccComb.size()+1 < n) {
    memAccComb.emplace_back(memAccComb.size()+1);
  }
  uint64_t& mac = memAccComb[n-2][k-2];
  if(mac == 0) {
    mac = AccComb(n, k-1) + Comb(n, k);
  }
  return mac;
}

const uint32_t TrackingSet::nCpus_ = std::thread::hardware_concurrency();
std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::high_resolution_clock::now();

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }

  // TODO: does it override the environment variable?
  omp_set_num_threads(TrackingSet::nCpus_);
  // Enable nested parallelism
  omp_set_max_active_levels(omp_get_supported_active_levels());

  Formula formula;
  formula.Load(argv[1]);
  int64_t prevNUnsat = formula.nClauses_;
  Traversal trav;

  // Now there are some clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  DefaultSatTracker satTr(formula);
  satTr.Populate(formula.ans_);

  int64_t bestInit = satTr.UnsatCount();
  std::cout << "All false: " << bestInit << ", ";
  std::cout.flush();

  int64_t altNUnsat;
  TrackingSet initFront;

  {
    TrackingSet initUnsatClauses = satTr.GetUnsat();
    // for init it's usually better if we don't move an extra time
    altNUnsat = satTr.GradientDescend(false, trav, nullptr, initUnsatClauses, initFront, formula.nClauses_);
    std::cout << "GradientDescent: " << altNUnsat << ", ";
    std::cout.flush();
    if(altNUnsat < bestInit) {
      bestInit = altNUnsat;
    } else {
      // Revert to all false
      formula.ans_ = BitVector(formula.nVars_+1);
    }
  }

  BitVector altAsg = formula.SetGreedy1();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Greedy1: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg = formula.SetGreedy2();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Greedy2: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg.Randomize();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Random: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg.SetTrue();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "All true: " << altNUnsat << std::endl;
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  TrackingSet unsatClauses = satTr.Populate(formula.ans_);
  DefaultSatTracker origSatTr(formula);

  BitVector maxPartial;
  bool maybeSat = true;
  bool provenUnsat = false;
  std::mt19937_64 rng;
  int64_t nStartUnsat;
  BitVector next;
  int64_t nParallelGD = 0, nSequentialGD = 0;
  while(maybeSat) {
    //TODO: this is a heavy assert
    assert(unsatClauses == satTr.GetUnsat());
    nStartUnsat = unsatClauses.Size();
    maxPartial = formula.ans_;
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    auto tmEnd = std::chrono::high_resolution_clock::now();
    double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
    double clausesPerSec = (prevNUnsat - nStartUnsat) / nSec;
    std::cout << "\tUnsatisfied clauses: " << nStartUnsat << " - elapsed " << nSec << " seconds, ";
    if(clausesPerSec >= 1 || clausesPerSec == 0) {
      std::cout << clausesPerSec << " clauses per second.";
    } else {
      std::cout << 1.0 / clausesPerSec << " seconds per clause.";
    }
    std::cout << std::endl;
    tmStart = tmEnd;
    prevNUnsat = nStartUnsat;
    
    TrackingSet front;
    if(nStartUnsat == bestInit) {
      front = initFront;
    }
    bool allowDuplicateFront = false;
    while(unsatClauses.Size() >= nStartUnsat) {
      assert(formula.ComputeUnsatClauses() == unsatClauses);
      if(front.Size() == 0 || (!allowDuplicateFront && trav.IsSeenFront(front))) {
        front = unsatClauses;
        std::cout << "$";
        std::cout.flush();
      }

      origSatTr = satTr;

      std::vector<int64_t> varFront = formula.ClauseFrontToVars(front, formula.ans_);
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestRevVars;
      const int64_t endNIncl = std::min<int64_t>(varFront.size(), 3);
      std::cout << "P" << varFront.size() << "," << unsatClauses.Size();
      std::cout.flush();
      int64_t nIncl=2;
      for(; nIncl<=endNIncl; nIncl++) {
        next = formula.ans_;
        TrackingSet stepRevs;
        bool moved = false;
        const int64_t curNUnsat = satTr.ParallelGD(
          true, nIncl, varFront, next, trav, nullptr, front, stepRevs, 
          std::max<int64_t>(satTr.UnsatCount() * 2, satTr.UnsatCount() + std::log2(formula.nClauses_)),
          moved, 0);
        nParallelGD++;
        satTr = origSatTr;
        if( curNUnsat < bestUnsat ) {
          bestUnsat = curNUnsat;
          bestRevVars = stepRevs;
          if(bestUnsat < nStartUnsat)
          {
            break;
          }
        }
        ParallelShuffle(varFront.data(), varFront.size());
      }
      std::cout << "/" << bestUnsat << "] ";
      std::cout.flush();

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
      std::cout.flush();
      front.Clear();

      std::vector<int64_t> vBestRevVars = bestRevVars.ToVector();
      #pragma omp parallel for
      for(int64_t i=0; i<vBestRevVars.size(); i++) {
        const int64_t revV = vBestRevVars[i];
        formula.ans_.Flip(revV);
        satTr.FlipVar(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, &front);
      }
      assert(satTr.UnsatCount() == bestUnsat);
      assert(unsatClauses.Size() == bestUnsat);

      if(unsatClauses.Size() < nStartUnsat) {
        break;
      }

      int64_t oldUnsat, newUnsat = unsatClauses.Size();
      int64_t nInARow = 0;
      std::cout << "S";
      std::cout.flush();
      do {
        std::cout << "/" << newUnsat;
        std::cout.flush();
        if(newUnsat < nStartUnsat) {
          break;
        }
        assert(newUnsat == satTr.UnsatCount());
        oldUnsat = newUnsat;
        nInARow++;
        const uint128 oldHash = formula.ans_.hash_;
        //TrackingSet consider = unsatClauses + front;
        front.Clear();
        newUnsat = satTr.GradientDescend( true, trav, &unsatClauses, unsatClauses, front,
          std::max<int64_t>( unsatClauses.Size() * 2, unsatClauses.Size() + std::log2(formula.nClauses_) )
        );
        nSequentialGD++;
        assert(newUnsat == unsatClauses.Size());
      } while(newUnsat < oldUnsat);

      assert(newUnsat == satTr.UnsatCount());
      std::cout << "} ";
      std::cout.flush();
    }
    std::cout << "\n\tWalk length: " << trav.seenMove_.size() << ", Stack length: " << trav.dfs_.size()
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

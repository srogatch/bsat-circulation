#include "Reduction.h"
#include "TrackingSet.h"
#include "Utils.h"
#include "SatTracker.h"

#include <iostream>
#include <mutex>
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

struct MoveHash {
  bool operator()(const std::pair<TrackingSet, TrackingSet>& v) const {
    return v.first.hash_ * 1949 + v.second.hash_ * 2011;
  }
};

struct Point {
  BitVector assignment_;

  Point(const BitVector& assignment)
  : assignment_(assignment)
  { }
};

const uint32_t Formula::nCpus_ = std::thread::hardware_concurrency();
std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::high_resolution_clock::now();

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }

  // TODO: does it override the environment variable?
  omp_set_num_threads(Formula::nCpus_);
  // Enable nested parallelism
  omp_set_max_active_levels(omp_get_supported_active_levels());

  std::mutex muUnsatClauses;
  std::mutex muFront;
  Formula formula;
  formula.Load(argv[1]);

  int64_t prevNUnsat = formula.nClauses_;

  // Now there are some clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  DefaultSatTracker satTr(formula);
  satTr.Populate(formula.ans_);

  int64_t bestInit = satTr.UnsatCount();
  std::cout << "All false: " << bestInit << ", ";
  std::cout.flush();

  TrackingSet initUSC = satTr.GetUnsat();
  TrackingSet initFront;
  int64_t altNUnsat=satTr.GradientDescend(false, &initUSC, &initFront); // for init it's usually better if we don't move an extra time
  std::cout << "GradientDescent: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
  } else {
    // Revert to all false
    formula.ans_ = BitVector(formula.nVars_+1);
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

  satTr.Populate(formula.ans_);

  std::unordered_map<uint128, int64_t> seenAssignment;
  seenAssignment[formula.ans_.hash_] = bestInit;

  BitVector maxPartial;
  bool maybeSat = true;
  bool provenUnsat = false;
  std::unordered_set<TrackingSet> seenFront;
  std::mt19937_64 rng;
  std::deque<Point> dfs;
  int64_t nStartUnsat;
  // Define them here to avoid reallocations
  std::vector<std::pair<int64_t, int64_t>> combs;
  std::vector<int64_t> vFront;
  BitVector next;
  int64_t nParallelGD = 0, nSequentialGD = 0;
  while(maybeSat) {
    TrackingSet unsatClauses = satTr.GetUnsat();
    nStartUnsat = unsatClauses.set_.size();
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
    while(unsatClauses.set_.size() >= nStartUnsat) {
      assert(formula.ComputeUnsatClauses() == unsatClauses);
      if(front.set_.empty() || (!allowDuplicateFront && seenFront.find(front) != seenFront.end())) {
        front = unsatClauses;
        std::cout << "$";
        std::cout.flush();
      }

      DefaultSatTracker origSatTr(satTr);
      std::unordered_map<int64_t, int64_t> candVs;
      vFront.assign(front.set_.begin(), front.set_.end());
      #pragma omp parallel for
      for(int64_t i=0; i<vFront.size(); i++) {
        const int64_t originClause = vFront[i];
        for(const int64_t iVar : formula.clause2var_[originClause]) {
          if( (iVar < 0 && formula.ans_[-iVar]) || (iVar > 0 && !formula.ans_[iVar]) ) {
            // A dissatisfying arc
            const int64_t revV = llabs(iVar);
            #pragma omp critical
            candVs[revV]++;
          }
        }
      }
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestRevVertices;

      combs.assign(candVs.begin(), candVs.end());
      if( combs.size() >= 2 * omp_get_max_threads() ) {
        ParallelShuffle(combs.data(), combs.size());
      } else {
        std::shuffle(combs.begin(), combs.end(), rng);
      }
      std::stable_sort(std::execution::par, combs.begin(), combs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
      });

      const int64_t endNIncl = std::min<int64_t>(combs.size(), Formula::nCpus_);
      for(int64_t nIncl=1; nIncl<=endNIncl; nIncl++) {
        next = formula.ans_;
        TrackingSet stepRevs;
        std::cout << "P" << nIncl;
        std::cout.flush();
        const int64_t curNUnsat = satTr.ParallelGD(true, nIncl, combs, next, seenAssignment, &stepRevs);
        nParallelGD++;
        if( curNUnsat < bestUnsat ) {
          bestUnsat = curNUnsat;
          bestRevVertices = stepRevs;
        }
        satTr = origSatTr;
        std::cout << "]";
        std::cout.flush();
      }

      if(bestUnsat >= formula.nClauses_) {
        std::cout << "#";

        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        if(!allowDuplicateFront) {
          seenFront.emplace(front);
        }

        if(front != unsatClauses) {
          // Retry with full/random front
          satTr.Swap(origSatTr);
          front = unsatClauses;
          continue;
        }

        if(!dfs.empty()) {
          formula.ans_ = std::move(dfs.back().assignment_);
          dfs.pop_back();
          satTr.Populate(formula.ans_);
          unsatClauses = satTr.GetUnsat();
          front = unsatClauses;
          std::cout << "@";
          continue;
        }

        // Unsatisfiable
        std::cout << "...Nothing reversed - unsatisfiable..." << std::endl;
        maybeSat = false;
        break;
      }

      // Limit the size of the stack
      if(dfs.size() > formula.nVars_) {
        dfs.pop_front();
      }
      dfs.push_back(Point(formula.ans_));

      satTr.Swap(origSatTr);
      front.Clear();
      for(int64_t revV : bestRevVertices.set_) {
        formula.ans_.Flip(revV);
        satTr.FlipVar(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, &front);
      }
      seenAssignment[formula.ans_.hash_] = bestUnsat;
      // Indicate a walk step
      //std::cout << " F" << front.set_.size() << ":B" << bestFront.set_.size() << ":U" << unsatClauses.set_.size() << " ";
      std::cout << ">";

      std::cout << "S";
      std::cout.flush();
      front.Clear();
      const int64_t newUnsat = satTr.GradientDescend(true, &unsatClauses, &front);
      nSequentialGD++;
      std::cout << "}";
      std::cout.flush();
      seenAssignment[formula.ans_.hash_] = unsatClauses.set_.size();

      // Limit the size of the stack
      if(dfs.size() > formula.nVars_) {
        dfs.pop_front();
      }
      dfs.push_back(Point(formula.ans_));
    }
    std::cout << "\n\tAssignments considered: " << seenAssignment.size()
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

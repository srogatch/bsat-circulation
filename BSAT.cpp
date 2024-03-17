#include "Reduction.h"
#include "TrackingSet.h"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>

namespace detail {

template <typename F>
struct FinalAction {
  FinalAction(F f) : clean_{f} {}
  ~FinalAction() { if(enabled_) clean_(); }
  void disable() { enabled_ = false; };
private:
  F clean_;
  bool enabled_{true};
};

}
template <typename F>
detail::FinalAction<F> finally(F f) {
  return detail::FinalAction<F>(f); 
}

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
const uint32_t BitVector::nCpus_ = std::thread::hardware_concurrency();

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  std::mutex muUnsatClauses;
  std::mutex muFront;
  Formula formula;
  formula.Load(argv[1]);
  bool maybeSat = true;
  std::unordered_set<std::pair<TrackingSet, TrackingSet>, MoveHash> seenMove;
  std::unordered_set<TrackingSet> seenFront;
  std::mt19937_64 rng;
  int64_t lastFlush = formula.nClauses_ + 1;
  std::deque<Point> dfs;
  while(maybeSat) {
    TrackingSet unsatClauses = formula.ComputeUnsatClauses();
    const int64_t nStartUnsat = unsatClauses.set_.size();
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    std::cout << "Unsatisfied clauses: " << nStartUnsat << std::endl;
    TrackingSet front;
    std::vector<int64_t> vClauses;
    // avoid reallocations
    vClauses.reserve(unsatClauses.set_.size() * 4);
    bool allowDuplicateFront = false;
    while(unsatClauses.set_.size() >= nStartUnsat) {
      assert(formula.ComputeUnsatClauses() == unsatClauses);
      if(front.set_.empty() || (!allowDuplicateFront && seenFront.find(front) != seenFront.end())) {
        //std::cout << "Empty front" << std::endl;
        front = unsatClauses;
      }
      std::unordered_set<int64_t> candVs;
      std::vector<int64_t> vFront(front.set_.begin(), front.set_.end());
      #pragma omp parallel for num_threads(Formula::nCpus_)
      for(int64_t i=0; i<vFront.size(); i++) {
        const int64_t originClause = vFront[i];
        for(const int64_t iVar : formula.clause2var_[originClause]) {
          if( (iVar < 0 && formula.ans_[-iVar]) || (iVar > 0 && !formula.ans_[iVar]) ) {
            // A dissatisfying arc
            #pragma omp critical
            candVs.emplace(llabs(iVar));
          }
        }
      }
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestFront, bestUnsatClauses, bestRevVertices;

      std::vector<int64_t> combs(candVs.begin(), candVs.end());
      std::shuffle(combs.begin(), combs.end(), rng);
      std::vector<int64_t> incl;
      uint64_t nCombs = 0;
      // It may be slow to instantiate the bit vector in each combination
      BitVector next = formula.ans_;
      for(int64_t nIncl=1; nIncl<=combs.size(); nIncl++) {
        if(AccComb(combs.size(), nIncl) > 100) {
          std::cout << " C" << combs.size() << "," << nIncl << " ";
          std::flush(std::cout);
        }
        incl.clear();
        for(int64_t j=0; j<nIncl; j++) {
          incl.push_back(j);
        }
        for(;;) {
          TrackingSet stepRevs;
          nCombs++;
          assert(next == formula.ans_);
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = combs[incl[j]];
            // Avoid flipping bits more than once
            assert(stepRevs.set_.find(revV) == stepRevs.set_.end());
            stepRevs.Add(revV);
            next.Flip(revV);
          }
          auto unflip = finally([&]() {
            // Flip bits back
            for(int64_t revV : stepRevs.set_) {
              next.Flip(revV);
            }
          });

          if(seenMove.find({front, stepRevs}) == seenMove.end()) {
            TrackingSet newFront;
            TrackingSet newUnsatClauses = unsatClauses;
            vClauses.clear();
            for(const int64_t revV : stepRevs.set_) {
              const std::vector<int64_t>& srcClauses = formula.listVar2Clause_[revV];
              for(const int64_t iClause : srcClauses) {
                vClauses.emplace_back(llabs(iClause));
              }
            }
            std::sort(std::execution::par, vClauses.begin(), vClauses.end());
            #pragma omp parallel for num_threads(Formula::nCpus_)
            for(int64_t j=0; j<vClauses.size(); j++) {
              const uint64_t absClause = vClauses[j];
              if(j > 0 && absClause == vClauses[j-1]) {
                continue;
              }
              const bool oldSat = formula.IsSatisfied(absClause, formula.ans_);
              const bool newSat = formula.IsSatisfied(absClause, next);
              if(newSat) {
                if(!oldSat) {
                  std::unique_lock<std::mutex> lock(muUnsatClauses);
                  newUnsatClauses.Remove(absClause);
                }
              } else {
                if(oldSat)
                {
                  {
                    std::unique_lock<std::mutex> lock(muUnsatClauses);
                    newUnsatClauses.Add(absClause);
                  }
                  {
                    std::unique_lock<std::mutex> lock(muFront);
                    newFront.Add(absClause);
                  }
                }
              }
            }
            if(allowDuplicateFront || seenFront.find(newFront) == seenFront.end()) {
              // UNSAT counting is a heavy and parallelized operation
              const int64_t stepUnsat = newUnsatClauses.set_.size();
              if(stepUnsat < bestUnsat) {
                bestUnsat = stepUnsat;
                bestFront = std::move(newFront);
                bestUnsatClauses = std::move(newUnsatClauses);
                bestRevVertices = stepRevs;
              }
              if(bestUnsat < nStartUnsat) {
                break;
              }
            }
          }
          int64_t j;
          for(j=nIncl-1; j>=0; j--) {
            if(incl[j]+(nIncl-j) >= combs.size()) {
              continue;
            }
            break;
          }
          if(j < 0) {
            // All combinations with nIncl elements exhausted
            break;
          }
          incl[j]++;
          for(int64_t k=j+1; k<nIncl; k++) {
            incl[k] = incl[k-1] + 1;
          }
        }
        if(bestUnsat < std::min<int64_t>(
            std::max(nStartUnsat + nCombs - 1, unsatClauses.set_.size()*2),
            formula.nClauses_))
        {
          break;
        }
      }
      if(nCombs > formula.nVars_) {
        std::cout << "Combinations to next: " << nCombs << std::endl;
      }

      if(bestUnsat >= formula.nClauses_) {
        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        if(!allowDuplicateFront) {
          seenFront.emplace(front);
        }
        if(front != unsatClauses) {
          // Retry with full front
          std::cout << "$";
          front = unsatClauses;
          continue;
        }

        if(!dfs.empty()) {
          formula.ans_ = std::move(dfs.back().assignment_);
          dfs.pop_back();
          unsatClauses = formula.ComputeUnsatClauses();
          front.Clear();
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

      for(int64_t revV : bestRevVertices.set_) {
        formula.ans_.Flip(revV);
      }
      seenMove.emplace(front, bestRevVertices);
      // Indicate a DFS step
      std::cout << " F" << front.set_.size() << ":B" << bestFront.set_.size() << ":U" << unsatClauses.set_.size() << " ";
      front = std::move(bestFront);
      unsatClauses = std::move(bestUnsatClauses);
    }
    std::cout << "Search size: " << seenMove.size() << std::endl;
  }

  {
    std::ofstream ofs(argv[2]);
    if(!maybeSat) {
      ofs << "s UNSATISFIABLE" << std::endl;
      return 0;
    }

    assert(formula.SolWorks());
    ofs << "s SATISFIABLE" << std::endl;
    ofs << "v ";
    for(int64_t i=1; i<=formula.nVars_; i++) {
      ofs << (formula.ans_[i] ? i : -i) << " ";
    }
    ofs << "0" << std::endl;
  }
  return 0;
}

#include "Reduction.h"
#include "TrackingSet.h"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <execution>

struct MoveHash {
  bool operator()(const std::pair<TrackingSet, TrackingSet>& v) const {
    return v.first.hash_ * 1949 + v.second.hash_ * 2011;
  }
};

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  const uint32_t nCpus = std::thread::hardware_concurrency();
  std::mutex muUnsatClauses;
  std::mutex muFront;
  Formula formula;
  formula.Load(argv[1]);
  bool maybeSat = true;
  std::unordered_set<std::pair<TrackingSet, TrackingSet>, MoveHash> seenMove;
  std::unordered_set<TrackingSet> seenFront;
  while(maybeSat) {
    TrackingSet unsatClauses;
    #pragma omp parallel for num_threads(nCpus)
    for(int64_t i=1; i<=formula.nClauses_; i++) {
      if(!formula.IsSatisfied(i, formula.ans_)) {
        #pragma omp critical
        unsatClauses.Add(i);
      }
    }
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
    while(unsatClauses.set_.size() >= nStartUnsat) {
      if(front.set_.empty() || seenFront.find(front) != seenFront.end()) {
        //std::cout << "Empty front" << std::endl;
        front = unsatClauses;
      }
      std::unordered_set<int64_t> candVs;
      std::vector<int64_t> vFront(front.set_.begin(), front.set_.end());
      #pragma omp parallel for num_threads(nCpus)
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
      BitVector bestNext;
      TrackingSet bestFront, bestUnsatClauses, bestRevVertices;

      std::vector<int64_t> combs(candVs.begin(), candVs.end());
      std::vector<int64_t> incl;
      uint64_t nCombs = 0;
      // It may be slow to instantiate the bit vector in each combination
      BitVector next = formula.ans_;
      for(int64_t nIncl=1; nIncl<combs.size(); nIncl++) {
        if(nIncl >= 3) {
          std::cout << " C" << combs.size() << "," << nIncl << " ";
        }
        incl.clear();
        for(int64_t j=0; j<nIncl; j++) {
          incl.push_back(j);
        }
        for(;;) {
          TrackingSet stepRevs;
          nCombs++;
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = combs[incl[j]];
            // Avoid flipping bits more than once
            assert(stepRevs.set_.find(revV) == stepRevs.set_.end());
            stepRevs.Add(revV);
            next.Flip(revV);
          }

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
            #pragma omp parallel for num_threads(nCpus)
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
                {
                  std::unique_lock<std::mutex> lock(muFront);
                  newFront.Add(absClause);
                }
                if(oldSat)
                {
                  std::unique_lock<std::mutex> lock(muUnsatClauses);
                  newUnsatClauses.Add(absClause);
                }
              }
            }
            if(seenFront.find(newFront) == seenFront.end()) {
              // UNSAT counting is a heavy and parallelized operation
              const int64_t stepUnsat = newUnsatClauses.set_.size();
              if(stepUnsat < bestUnsat) {
                bestUnsat = stepUnsat;
                bestNext = next;
                bestFront = std::move(newFront);
                bestUnsatClauses = std::move(newUnsatClauses);
                bestRevVertices = stepRevs;
                if(bestUnsat < nStartUnsat) {
                  break;
                }
              }
            }
          }
          // Flip bits back
          for(int64_t revV : stepRevs.set_) {
            next.Flip(revV);
          }
          assert(next == formula.ans_);
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
        if(bestUnsat < formula.nClauses_) {
          break;
        }
      }
      if(nCombs > formula.nVars_) {
        std::cout << "Combinations to next: " << nCombs << std::endl;
      }

      if(bestUnsat >= formula.nClauses_) {
        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        seenFront.emplace(front);
        // TODO: a data structure with smaller algorithmic complexity
        std::vector<std::pair<TrackingSet, TrackingSet>> toRemove;
        for(const auto& p : seenMove) {
          if(p.first == front) {
            toRemove.emplace_back(p);
          }
        }
        // Release some memory - the moves from this front are no more needed
        for(const auto& p : toRemove) {
          seenMove.erase(p);
        }
        if(front != unsatClauses) {
          // Retry with full front
          front = unsatClauses;
          continue;
        }
        // Unsatisfiable
        std::cout << "Nothing reversed - unsatisfiable" << std::endl;
        maybeSat = false;
        break;
      }

      seenMove.emplace(front, bestRevVertices);
      front = std::move(bestFront);
      unsatClauses = std::move(bestUnsatClauses);
      formula.ans_ = std::move(bestNext);
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

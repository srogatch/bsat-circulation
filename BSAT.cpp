#include "Reduction.h"

#include <iostream>
#include <mutex>

struct seen_hash {
  inline std::size_t operator()(const BitVector &bv) const {
    std::size_t ans = 0;
    uint64_t mul = 7;
    for(int64_t i=0; i<bv.nQwords_; i++) {
      ans ^= mul * bv.bits_[i];
      mul *= 18446744073709551557ULL;
    }
    return ans;
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
  std::unordered_set<BitVector, seen_hash> seen;
  seen.emplace(formula.ans_);
  while(maybeSat) {
    std::unordered_set<int64_t> unsatClauses;
    #pragma omp parallel for num_threads(nCpus)
    for(int64_t i=1; i<=formula.nClauses_; i++) {
      if(!formula.IsSatisfied(i, formula.ans_)) {
        #pragma omp critical
        unsatClauses.emplace(i);
      }
    }
    const int64_t nStartUnsat = unsatClauses.size();
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    std::cout << "Unsatisfied clauses: " << nStartUnsat << std::endl;
    std::unordered_set<int64_t> front;
    while(unsatClauses.size() >= nStartUnsat) {
      if(front.empty()) {
        //std::cout << "Empty front" << std::endl;
        front = unsatClauses;
      }
      std::unordered_set<int64_t> candVs;
      std::vector<int64_t> vFront(front.begin(), front.end());
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
      std::vector<int64_t> revVertices;
      BitVector bestNext;
      std::vector<int64_t> combs(candVs.begin(), candVs.end());
      std::vector<int64_t> incl;
      uint64_t nCombs = 0;
      for(int64_t nIncl=1; nIncl<combs.size(); nIncl++) {
        if(nIncl >= 2) {
          std::cout << "Combining " << nIncl << " variables to reverse." << std::endl;
        }
        incl.clear();
        for(int64_t j=0; j<nIncl; j++) {
          incl.push_back(j);
        }
        for(;;) {
          BitVector next = formula.ans_;
          std::unordered_set<int64_t> stepRevs;
          nCombs++;
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = combs[incl[j]];
            stepRevs.emplace(revV);
            next.Flip(revV);
          }

          const int64_t stepUnsat = formula.CountUnsat(next);
          if(stepUnsat < bestUnsat) {
            if(seen.find(next) == seen.end()) {
              bestUnsat = stepUnsat;
              revVertices.assign(stepRevs.begin(), stepRevs.end());
              bestNext = std::move(next);
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
        if(!revVertices.empty()) {
          break;
        }
      }
      if(nCombs > formula.nVars_) {
        std::cout << "Combinations to next: " << nCombs << std::endl;
      }

      if(revVertices.empty()) {
        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
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

      seen.emplace(bestNext);
      front.clear();
      //#pragma omp parallel for num_threads(nCpus)
      for(int64_t i=0; i<revVertices.size(); i++) {
        const std::vector<int64_t>& clauses = formula.listVar2Clause_[revVertices[i]];
        #pragma omp parallel for num_threads(nCpus)
        for(int64_t j=0; j<clauses.size(); j++) {
          const uint64_t absClause = llabs(clauses[j]);
          const bool oldSat = formula.IsSatisfied(absClause, formula.ans_);
          const bool newSat = formula.IsSatisfied(absClause, bestNext);
          if(newSat) {
            if(!oldSat) {
              std::unique_lock<std::mutex> lock(muUnsatClauses);
              unsatClauses.erase(absClause);
            }
          } else {
            {
              std::unique_lock<std::mutex> lock(muFront);
              front.emplace(absClause);
            }
            if(oldSat)
            {
              std::unique_lock<std::mutex> lock(muUnsatClauses);
              unsatClauses.emplace(absClause);
            }
          }
        }
      }
      formula.ans_ = std::move(bestNext);
    }
    std::cout << "Search size: " << seen.size() << std::endl;
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

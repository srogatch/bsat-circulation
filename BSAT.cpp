#include "Reduction.h"

#include <iostream>

struct seen_hash {
  inline std::size_t operator()(const std::pair<std::pair<int64_t, int64_t>, std::unordered_set<int64_t>> & v) const {
    std::size_t ans = (std::hash<int64_t>()(v.first.first) * 37)
        ^ (std::hash<int64_t>()(v.first.second) * 17);
    for(int64_t x : v.second) {
      ans ^= x * 18446744073709551557ULL;
    }
    return ans;
  }
};

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  Formula formula;
  formula.Load(argv[1]);
  bool maybeSat = true;
  std::unordered_set<std::vector<bool>> seen;
  seen.emplace(formula.ans_);
  while(maybeSat) {
    Reduction red(formula);
    std::unordered_set<int64_t> unsatClauses;
    for(int64_t i=1; i<=formula.nClauses_; i++) {
      if(formula.dummySat_[i]) {
        continue; // the clause contains a variable and its negation
      }
      if(red.fGraph_.backlinks_[formula.nVars_ + i].empty()) {
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
      for(const int64_t originClause : front) {
        for(const auto& clauseDst : red.fGraph_.links_[formula.nVars_ + originClause]) {
          assert(clauseDst.first == clauseDst.second->to_);
          assert(formula.nVars_ + originClause == clauseDst.second->from_);
          candVs.emplace(clauseDst.first);
        }
      }
      int64_t bestUnsat = formula.nClauses_+1;
      std::unordered_set<int64_t> revVertices;
      std::vector<bool> bestNext;
      std::unordered_set<int64_t> stepRevs;
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
          std::vector<bool> next = formula.ans_;
          nCombs++;
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = combs[incl[j]];
            stepRevs.emplace(revV);
            next[revV] = !next[revV];
          }
          int64_t stepUnsat = formula.CountUnsat(next);
          if(stepUnsat < bestUnsat) {
            if(seen.find(next) == seen.end()) {
              bestUnsat = stepUnsat;
              revVertices = stepRevs;
              bestNext = next;
            }
          }
          stepRevs.clear();
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
        // Unsatisfiable
        std::cout << "Nothing reversed - unsatisfiable" << std::endl;
        maybeSat = false;
        break;
      }

      seen.emplace(bestNext);
      front.clear();
      for(const int64_t revVertex : revVertices) {
        // Reverse the incoming and outgoing arcs for this variable
        std::vector<int64_t> oldForward, oldBackward;
        for(const auto& varDst : red.fGraph_.links_[revVertex]) {
          oldForward.emplace_back(varDst.first);
        }
        for(const auto& varSrc : red.fGraph_.backlinks_[revVertex]) {
          oldBackward.emplace_back(varSrc.first);
        }
        for(const int64_t c : oldForward) {
          red.fGraph_.Remove(revVertex, c);
        }
        for(const int64_t c : oldBackward) {
          red.fGraph_.Remove(c, revVertex);
        }
        for(const int64_t c : oldBackward) {
          red.fGraph_.AddMerge(Arc(revVertex, c, 0, 1));
          unsatClauses.erase(c - formula.nVars_);
        }
        for(const int64_t c : oldForward) {
          red.fGraph_.AddMerge(Arc(c, revVertex, 0, 1));
          if(red.fGraph_.backlinks_[c].empty()) {
            unsatClauses.emplace(c - formula.nVars_);
            front.emplace(c - formula.nVars_);
          }
        }
      }
      formula.ans_ = bestNext;
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
    for(int64_t i=1; i<formula.ans_.size(); i++) {
      ofs << (formula.ans_[i] ? i : -i) << " ";
    }
    ofs << "0" << std::endl;
  }
  return 0;
}

#include "Reduction.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  Formula formula;
  formula.Load(argv[1]);
  std::unordered_set<std::vector<bool>> seen;
  seen.emplace(formula.ans_);
  bool maybeSat = true;
  while(maybeSat) {
    Reduction red(formula);
    std::unordered_set<int64_t> unsatClauses;
    for(int64_t i=1; i<=formula.nClauses_; i++) {
      if(red.fGraph_.backlinks_[formula.nVars_ + i].empty()) {
        unsatClauses.emplace(i);  
      }
    }
    const int64_t nStartUnsat = unsatClauses.size();
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    while(unsatClauses.size() >= nStartUnsat) {
      bool reversed = false;
      for(const int64_t originClause : unsatClauses) {
        for(const auto& clauseDst : red.fGraph_.links_[formula.nVars_ + originClause]) {
          const int64_t revVertex = clauseDst.first;
          std::vector<bool> next = formula.ans_;
          next[revVertex] = ! next[revVertex];
          if(seen.find(next) != seen.end()) {
            continue;
          }
          seen.emplace(next);
          // TODO: Reverse the incoming and outgoing arcs for this variable
          std::vector<int64_t> oldForward, oldBackward;
          for(const auto& varDst : red.fGraph_.links_[revVertex]) {
            oldForward.emplace_back(varDst.first);
          }
          for(const auto& varSrc : red.fGraph_.backlinks_[revVertex]) {
            oldBackward.emplace_back(varSrc.first);
          }
          red.fGraph_.links_[revVertex].clear();
          red.fGraph_.backlinks_[revVertex].clear();
          for(const int64_t c : oldBackward) {
            red.fGraph_.AddMerge(Arc(revVertex, c, 0, 1));
            if(unsatClauses.find(c - formula.nVars_) != unsatClauses.end()) {
              unsatClauses.erase(c - formula.nVars_);
            }
          }
          for(const int64_t c : oldForward) {
            red.fGraph_.AddMerge(Arc(c, revVertex, 0, 1));
            if(red.fGraph_.backlinks_[c].empty()) {
              unsatClauses.emplace(c - formula.nVars_);
            }
          }
          reversed = true;
          break;
        }
        if(reversed) {
          break;
        }
      }
      if(!reversed) {
        // Unsatisfiable
        std::cout << "Nothing to reverse - unsatisfiable" << std::endl;
        maybeSat = false;
        break;
      }
    }
  }

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
  return 0;
}

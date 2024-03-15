#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include <vector>
#include <iostream>

template <typename T> constexpr int Signum(const T val) {
  return (T(0) < val) - (val < T(0));
}

struct Formula {
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> clause2var_;
  //std::unordered_map<uint64_t, std::unordered_set<int64_t>> var2clause_;
  int64_t nVars_ = 0, nClauses_ = 0;
  std::vector<bool> ans_;
  std::vector<bool> dummySat_;

  void Add(const uint64_t iClause, const int64_t iVar) {
    clause2var_[iClause].emplace(iVar);
    //var2clause_[llabs(iVar)].emplace(int64_t(iClause) * Signum(iVar));
  }

  void Load(const std::string& filePath) {
    std::ifstream ifs(filePath);
    std::string line;
    bool probDefRead = false;
    int64_t iClause = 0;
    while(std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::string cmd;
      if( !(iss >> cmd) ) {
        continue; // empty line?
      }
      if(cmd == "c" || cmd == "C") {
        continue; // a comment
      }
      if(cmd == "p" || cmd == "P") {
        if(probDefRead) {
          throw std::runtime_error("Duplicate problem definition");
        }
        std::string type;
        iss >> type;
        if(type != "cnf") {
          throw std::runtime_error("Unsupported problem type");
        }
        iss >> nVars_ >> nClauses_;
        ans_.resize(nVars_+1);
        dummySat_.resize(nClauses_+1);
        probDefRead = true;
        continue;
      }
      if(!probDefRead) {
        throw std::runtime_error("Data starts without a problem definition coming first.");
      }
      iClause++;
      if(iClause > nClauses_) {
        std::cerr << "Too many clauses: check the input DIMACS." << std::endl;
        break;
      }
      do {
        int64_t iVar = std::stoll(cmd);
        if(iVar == 0) {
          break;
        }
        Add(iClause, iVar);
      } while(iss >> cmd);
    }
    for(int64_t i=1; i<=nClauses_; i++) {
      for(int64_t iVar : clause2var_[i]) {
        if(clause2var_[i].find(-iVar) != clause2var_[i].end()) {
          dummySat_[i] = true;
          clause2var_.erase(i);
          break;
        }
      }
    }
  }

  bool SolWorks() {
    for(const auto& dj : clause2var_) {
      bool satisfied = false;
      for(const int64_t iVar : dj.second) {
        if( (iVar > 0 && ans_[iVar]) || (iVar < 0 && !ans_[-iVar]) ) {
          satisfied = true;
          break;
        }
      }
      if(!satisfied) {
        return false;
      }
    }
    return true;
  }

  bool RemoveKnown(const std::vector<bool>& known) {
    std::vector<uint64_t> satClauses;
    for(auto& clause : clause2var_) {
      std::vector<int64_t> unsatVars;
      bool hasUnknowns = false;
      bool satisfied = false;
      for(int64_t iVar : clause.second) {
        if(!known[llabs(iVar)]) {
          hasUnknowns = true;
          continue;
        }
        if( (iVar < 0 && !ans_[-iVar]) || (iVar > 0 && ans_[iVar]) ) {
          satisfied = true;
          break;
        } else {
          unsatVars.emplace_back(iVar);
        }
      }
      if(satisfied) {
        satClauses.emplace_back(clause.first);
      } else if(!hasUnknowns) {
        // Unsatisfied clause
        return false;
      }
      for(int64_t uv : unsatVars) {
        clause.second.erase(uv);
      }
    }
    for(uint64_t iClause : satClauses) {
      clause2var_.erase(iClause);
    }
    if(clause2var_.empty()) {
      std::cout << "All clauses satisfied." << std::endl;
    }
    return true;
  }
};

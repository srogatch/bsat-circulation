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

template <typename T> constexpr int Signum(const T val) {
  return (T(0) < val) - (val < T(0));
}

struct Formula {
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> clause2var_;
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> var2clause_;
  int64_t nVars_ = 0, nClauses_ = 0;

  void Add(const uint64_t iClause, const int64_t iVar) {
    clause2var_[iClause].emplace(iVar);
    var2clause_[llabs(iVar)].emplace(int64_t(iClause) * Signum(iVar));
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
        probDefRead = true;
        continue;
      }
      if(!probDefRead) {
        throw std::runtime_error("Data starts without a problem definition coming first.");
      }
      iClause++;
      if(iClause > nClauses_) {
        throw std::runtime_error("Too many clauses");
      }
      do {
        int64_t iVar = std::stoll(cmd);
        if(iVar == 0) {
          break;
        }
        Add(iClause, iVar);
      } while(iss >> cmd);
    }
  }
};

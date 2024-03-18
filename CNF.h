#pragma once

#include "BitVector.h"
#include "TrackingSet.h"
#include "BlockingQueue.h"

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
#include <omp.h>
#include <thread>
#include <atomic>
#include <cassert>
#include <stack>
#include <random>

template <typename T> constexpr int Signum(const T val) {
  return (T(0) < val) - (val < T(0));
}

struct Formula {
  static const uint32_t nCpus_;
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> clause2var_;
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> var2clause_;
  std::unordered_map<uint64_t, std::vector<int64_t>> listVar2Clause_;
  int64_t nVars_ = 0, nClauses_ = 0;
  BitVector ans_;
  BitVector dummySat_;

  void Add(const uint64_t iClause, const int64_t iVar) {
    clause2var_[iClause].emplace(iVar);
    var2clause_[llabs(iVar)].emplace(int64_t(iClause) * Signum(iVar));
  }

  void Load(const std::string& filePath) {
    std::ifstream ifs(filePath);
    if(!ifs) {
      std::cerr << "Cannot open the file to load the formula from: " << filePath << std::endl;
      throw std::runtime_error("Cannot open input file.");
    }
    constexpr const uint32_t cBufSize = 8 * 1024 * 1024;
    std::unique_ptr<char[]> buffer(new char[cBufSize]);
    ifs.rdbuf()->pubsetbuf(buffer.get(), cBufSize);

    BlockingQueue<std::string> bqParsing;
    BlockingQueue<std::pair<int64_t, int64_t>> bqAdding;

    std::thread parsingThr([&] {
      bool probDefRead = false;
      int64_t iClause = 0;
      std::string line;
      while(bqParsing.Pop(line)) {
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
          ans_ = BitVector(nVars_+1);
          dummySat_ = BitVector(nClauses_+1);
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
          bqAdding.Push(std::make_pair(iClause, iVar));
        } while(iss >> cmd);
      }
      bqAdding.RequestShutdown();
      std::cout << "Finished parsing the input file lines." << std::endl;
    });

    std::thread addingThr([&] {
      std::pair<int64_t, int64_t> entry;
      while(bqAdding.Pop(entry)) {
        Add(entry.first, entry.second);
      }
      std::cout << "Finished linking clauses and variables." << std::endl;
    });

    std::string line;
    while(std::getline(ifs, line)) {
      bqParsing.Push(std::move(line));
    }
    bqParsing.RequestShutdown();
    std::cout << "Finished streaming the input file." << std::endl;
    parsingThr.join();
    addingThr.join();

    std::cout << "Finding dummy clauses" << std::endl;
    for(int64_t i=1; i<=nClauses_; i++) {
      auto it = clause2var_.find(i);
      if(it == clause2var_.end() || it->second.size() == 0) {
        dummySat_.Flip(i);
        continue;
      }
      for(int64_t iVar : clause2var_[i]) {
        if(clause2var_[i].find(-iVar) != clause2var_[i].end()) {
          dummySat_.Flip(i);
          clause2var_.erase(i);
          var2clause_[llabs(iVar)].erase(i);
          var2clause_[llabs(iVar)].erase(-i);
          break;
        }
      }
    }
    std::cout << "Constructing vectors of clauses for variables." << std::endl;
    for(int64_t i=1; i<=nVars_; i++) {
      listVar2Clause_[i] = {};
    }
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      listVar2Clause_.find(i)->second.assign(var2clause_[i].begin(), var2clause_[i].end());
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

  int64_t CountUnsat(const BitVector& assignment) const {
    std::atomic<int64_t> nUnsat = 0;
    #pragma omp parallel for
    for(int64_t i=1; i<=nClauses_; i++) {
      if(!IsSatisfied(i, assignment)) {
        nUnsat.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return nUnsat;
  }

  bool IsSatisfied(const uint64_t iClause, const BitVector& assignment) const {
    if(dummySat_[iClause]) {
      return true;
    }
    auto it = clause2var_.find(iClause);
    assert(it != clause2var_.end());
    for(const int64_t iVar : it->second) {
      if( (iVar < 0 && !assignment[-iVar]) || (iVar > 0 && assignment[iVar]) ) {
        return true;
      }
    }
    return false;
  }

  TrackingSet ComputeUnsatClauses() const {
    TrackingSet ans;
    #pragma omp parallel for
    for(int64_t i=1; i<=nClauses_; i++) {
      if(!IsSatisfied(i, ans_)) {
        #pragma omp critical
        ans.Add(i);
      }
    }
    return ans;
  }

  BitVector SetGreedy1() const {
    BitVector ans(nVars_); // Init to false
    std::vector<std::pair<int64_t, int64_t>> counts_;
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      auto it = var2clause_.find(i);
      if(it == var2clause_.end()) {
        continue; // this variable doesn't appear in any clause - let it stay false
      }
      int64_t nPos = 0, nNeg = 0;
      for(const int64_t j : it->second) {
        if(j < 0) {
          nNeg++;
        } else {
          nPos++;
        }
      }
      if(nPos > nNeg) {
        #pragma omp critical
        ans.Flip(i);
      }
    }
    return ans;
  }

  BitVector SetGreedy2() const {
    BitVector ans(nVars_); // Init to false
    BitVector knownClauses(nClauses_); // Init to false
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      auto it = var2clause_.find(i);
      if(it == var2clause_.end()) {
        continue;
      }
      for(const int64_t iClause : it->second) {
        bool bBreak = false;
        #pragma omp critical
        if(!knownClauses[llabs(iClause)]) {
          knownClauses.Flip(llabs(iClause));
          if(iClause > 0) {
            ans.Flip(i);
          }
          bBreak = true;
        }
        if(bBreak) {
          break;
        }
      }
    }
    return ans;
  }

  // Disable it as it gives inferior results, but is also single-threaded / slow
  BitVector SetDfs() const {
    BitVector ans(nVars_); // Init to false
    BitVector visitedVars(nVars_); // Init to false
    BitVector knownClauses(nClauses_); // Init to false
    std::vector<int64_t> trail;
    std::mt19937 rng;

    for(int64_t i=1; i<=nVars_; i++) {
      if(!visitedVars[i]) {
        visitedVars.Flip(i);
        trail.push_back(i);
      }
      while(!trail.empty()) {
        const int64_t at = rng() % trail.size();
        const int64_t iVar = trail[at];
        trail[at] = trail.back();
        trail.pop_back();
        auto jt = var2clause_.find(llabs(iVar));
        if(jt == var2clause_.end()) {
          continue;
        }
        bool assigned = false;
        for(const int64_t iClause : jt->second) {
          if(knownClauses[llabs(iClause)]) {
            continue;
          }
          knownClauses.Flip(llabs(iClause));
          if(!assigned) {
            assigned = true;
            if(iClause > 0) {
              ans.Flip(llabs(iVar));
            }
          }
          auto kt = clause2var_.find(llabs(iClause));
          if(kt == clause2var_.end()) {
            continue;
          }
          for(const int64_t k : kt->second) {
            if(visitedVars[llabs(k)]) {
              continue;
            }
            visitedVars.Flip(llabs(k));
            trail.push_back(k);
          }
        }
      }
    }
    return ans;
  }

  int64_t GetSatDiff(const uint64_t iClause, const BitVector& newAsg, const int64_t iFlipVar) const {
    if(dummySat_[iClause]) {
      return 0;
    }
    auto it = clause2var_.find(iClause);
    assert(it != clause2var_.end());
    int64_t nSatVars = 0;
    bool flippedSat = false;
    for(const int64_t iVar : it->second) {
      if( (iVar < 0 && !newAsg[-iVar]) || (iVar > 0 && newAsg[iVar]) ) {
        nSatVars++;
        if(iVar == iFlipVar) {
          flippedSat = true;
        }
      }
    }
    if(nSatVars == 0) {
      return -1;
    } else if(nSatVars == 1 && flippedSat) {
      return 1;
    } else {
      return 0;
    }
  }

  // This is very slow for now - without SatTracker data structure
  BitVector SetDescent() const {
    BitVector ans(nVars_); // Init to false
    std::vector<int64_t> vVars(nVars_);
    constexpr const uint32_t cParChunkSize = kCacheLineSize / sizeof(vVars[0]);

    #pragma omp parallel for schedule(static, cParChunkSize)
    for(int64_t i=1; i<=nVars_; i++) {
      vVars[i-1] = i;
    }
    ParallelShuffle(vVars.data(), nVars_);

    int64_t minUnsat = CountUnsat(ans);
    for(int64_t k=0; k<nVars_; k++) {
      const int64_t iVar = vVars[k];
      ans.Flip(iVar);
      const std::vector<int64_t>& clauses = listVar2Clause_.find(iVar)->second;
      std::atomic<int64_t> newUnsat(minUnsat);
      #pragma omp parallel for
      for(int64_t j=0; j<clauses.size(); j++) {
        const int64_t iClause = clauses[j];
        const int64_t satDiff = GetSatDiff(llabs(iClause), ans, iVar * Signum(iClause));
        if(satDiff != 0) {
          newUnsat.fetch_sub(satDiff);
        }
      }
      if(newUnsat > minUnsat) {
        ans.Flip(iVar);
        continue;
      }
      minUnsat = newUnsat;
    }
    return ans;
  }
};

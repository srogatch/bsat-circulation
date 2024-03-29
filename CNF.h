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
#include <algorithm>

template <typename T> constexpr int Signum(const T val) {
  return (T(0) < val) - (val < T(0));
}

struct Formula {
  // TODO: replace with vectors and allow parallel build
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> clause2var_;
  std::unordered_map<uint64_t, std::unordered_set<int64_t>> var2clause_;
  std::unordered_map<uint64_t, std::vector<int64_t>> listVar2Clause_;
  int64_t nVars_ = 0, nClauses_ = 0;
  BitVector ans_;
  BitVector dummySat_;

  void Add(const uint64_t iClause, const int64_t iVar) {
    assert(1 <= int64_t(iClause) && int64_t(iClause) <= nClauses_);
    assert(1 <= llabs(iVar) && llabs(iVar) <= nVars_);
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
    const int halfCpus = std::max<int>(1, std::thread::hardware_concurrency() / 2 - 1);
    std::vector<BlockingQueue<std::pair<int64_t, int64_t>>> bqsAdding(halfCpus);

    std::thread parsingThr([&] {
      bool probDefRead = false;
      int64_t iClause = 0;
      std::string line;
      std::istringstream iss;
      std::string cmd;
      while(bqParsing.Pop(line)) {
        iss = std::istringstream(line);
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
        if(probDefRead) {
          break;
        }
        throw std::runtime_error("Data starts without a problem definition coming first.");
      }

      iClause++;
      if(iClause > nClauses_) {
        std::cerr << "Too many clauses: check the input DIMACS." << std::endl;
        // Stop reading excesssive clauses
        goto release;
      }
      do {
        int64_t iVar = std::stoll(cmd);
        if(iVar == 0) {
          break;
        }
        if( !(1 <= llabs(iVar) && llabs(iVar) <= nVars_) ) {
          std::cerr << "Variable value out of range: " << iVar << std::endl;
          throw std::runtime_error("Incorrect input");
        }
        bqsAdding[llabs(iVar) % halfCpus].Push(std::make_pair(iClause, iVar));
      } while(iss >> cmd);

      for(;;) {
        std::vector<std::string> lines;
        while(int64_t(lines.size()) < halfCpus) {
          if(!bqParsing.Pop(line)) {
            break;
          }
          bool isComment = false;
          int64_t j=0;
          for(; j<int64_t(line.size()); j++) {
            if(line[j] == 'c') {
              isComment = true;
              break;
            }
            if(!isspace(line[j])) {
              break;
            }
          }
          if( isComment || j >= int64_t(line.size()) ) {
            continue;
          }
          lines.emplace_back(std::move(line));
        }
        #pragma omp parallel for num_threads(halfCpus)
        for(int64_t i=0; i<int64_t(lines.size()); i++) {
          const int64_t locClause = iClause + 1 + i;
          if(locClause > nClauses_) {
            std::cerr << "Too many clauses: check the input DIMACS." << std::endl;
            // Stop reading excesssive clauses
            continue;
          }
          std::istringstream locIss = std::istringstream(lines[i]);
          int64_t iVar;
          while(locIss >> iVar) {
            if(iVar == 0) {
              break;
            }
            if( !(1 <= llabs(iVar) && llabs(iVar) <= nVars_) ) {
              std::cerr << "Variable value out of range: " << iVar << std::endl;
              throw std::runtime_error("Incorrect input");
            }
            bqsAdding[omp_get_thread_num()].Push(std::make_pair(locClause, iVar));
          }
        }
        iClause += lines.size();
        if(int64_t(lines.size()) < halfCpus) {
          break;
        }
        if(iClause > nClauses_) {
          break;
        }
      }
release:
      #pragma omp parallel for num_threads(halfCpus)
      for(int i=0; i<halfCpus; i++) {
        bqsAdding[i].RequestShutdown();
      }
      std::cout << "Finished parsing the input file lines." << std::endl;
    });

    std::thread addingThr([&] {
      std::mutex muC2V, muV2C;
      #pragma omp parallel num_threads(halfCpus)
      {
        std::pair<int64_t, int64_t> entry;
        while(bqsAdding[omp_get_thread_num()].Pop(entry)) {
          const int64_t iClause = entry.first;
          const int64_t iVar = entry.second;
          assert(1 <= llabs(iClause) && llabs(iClause) <= nClauses_);
          assert(1 <= llabs(iVar) && llabs(iVar) <= nVars_);
          {
            std::unique_lock<std::mutex> lock(muC2V);
            clause2var_[iClause].emplace(iVar);
          }
          {
            std::unique_lock<std::mutex> lock(muV2C);
            var2clause_[llabs(iVar)].emplace(int64_t(iClause) * Signum(iVar));
          }
        }
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
      auto& varSet = it->second;
      for(int64_t iVar : varSet) {
        if(varSet.find(-iVar) != varSet.end()) {
          dummySat_.Flip(i);
          var2clause_[llabs(iVar)].erase(i);
          var2clause_[llabs(iVar)].erase(-i);
          break;
        }
      }
    }
    std::cout << "Constructing vectors of clauses for variables." << std::endl;
    for(int64_t i=1; i<=nVars_; i++) {
      listVar2Clause_[i] = {};
      var2clause_[i];
    }
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      std::mt19937_64 rng = GetSeededRandom();
      std::vector<int64_t>& list = listVar2Clause_.find(i)->second;
      list.assign(var2clause_[i].begin(), var2clause_[i].end());
      std::shuffle(list.begin(), list.end(), rng);
    }
  }

  bool SolWorks() {
    for(const auto& dj : clause2var_) {
      if(dummySat_[dj.first]) {
        continue; // satisfied because the clause contains a variable and its negation
      }
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

  VCTrackingSet ComputeUnsatClauses() const {
    VCTrackingSet ans;
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
    BitVector ans(nVars_+1); // Init to false
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
        ans.Flip(i);
      }
    }
    return ans;
  }

  BitVector SetGreedy2() const {
    BitVector ans(nVars_+1); // Init to false
    BitVector knownClauses(nClauses_+1); // Init to false
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      auto it = var2clause_.find(i);
      if(it == var2clause_.end()) {
        continue;
      }
      for(const int64_t iClause : it->second) {
        if(!knownClauses[llabs(iClause)]) {
          knownClauses.Flip(llabs(iClause));
          if(iClause > 0) {
            ans.Flip(i);
          }
          break;
        }
      }
    }
    return ans;
  }

  // Disable it as it gives inferior results, but is also single-threaded / slow
  BitVector SetDfs() const {
    BitVector ans(nVars_+1); // Init to false
    BitVector visitedVars(nVars_+1); // Init to false
    BitVector knownClauses(nClauses_+1); // Init to false
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

  // To avoid falsse sharing in SatTracker, don't call it - let the indices stay randomized
  void SortClauseLists() {
    #pragma omp parallel for
    for(int64_t i=1; i<=nVars_; i++) {
      std::vector<int64_t>& clauses = listVar2Clause_[i];
      std::sort(clauses.begin(), clauses.end(), [](const int64_t a, const int64_t b) {
        return llabs(a) < llabs(b);
      });
    }
  }

  // For a set of clauses, return the set of variables that dissatisfy the clauses
  std::vector<int64_t> ClauseFrontToVars(const VCTrackingSet& clauseFront, const BitVector& assignment) {
    VCTrackingSet varFront;
    std::vector<int64_t> vClauseFront = clauseFront.ToVector();
    #pragma omp parallel for
    for(int64_t i=0; i<int64_t(vClauseFront.size()); i++) {
      const int64_t originClause = vClauseFront[i];
      assert(1 <= originClause && originClause <= nClauses_);
      for(const int64_t iVar : clause2var_.find(originClause)->second) {
        assert(1 <= llabs(iVar) && llabs(iVar) <= nVars_);
        if( (iVar < 0 && assignment[-iVar]) || (iVar > 0 && !assignment[iVar]) ) {
          // A dissatisfying arc
          const int64_t revV = llabs(iVar);
          varFront.Add(revV);
        }
      }
    }
    std::vector<int64_t> vVarFront = varFront.ToVector();
    // We don't serve shuffling here!
    return vVarFront;
  }

  // For a set of variables, return the set of clauses that are dissatisfied by the variables
  std::vector<int64_t> VarFrontToClauses(const VCTrackingSet& varFront, const BitVector& assignment) {
    VCTrackingSet clauseFront;
    std::vector<int64_t> vVarFront = varFront.ToVector();
    #pragma omp parallel for
    for(int64_t i=0; i<int64_t(vVarFront.size()); i++) {
      const int64_t originVar = vVarFront[i];
      assert(1 <= originVar && originVar <= nVars_);
      const int64_t iVar = assignment[originVar] ? originVar : -originVar;
      for(const int64_t iClause : listVar2Clause_[originVar]) {
        assert(1 <= iClause && iClause <= nClauses_);
        if(iVar * iClause < 0) {
          // A dissatisfying arc
          const int64_t dissatClause = llabs(iClause);
          clauseFront.Add(dissatClause);
        }
      }
    }
    std::vector<int64_t> vClauseFront = clauseFront.ToVector();
    // We don't serve shuffling here!
    return vVarFront;
  }
};

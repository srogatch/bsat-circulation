#pragma once

#include "Linkage.h"
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

struct Formula {
  Linkage clause2var_;
  Linkage var2clause_;
  int64_t nVars_ = 0, nClauses_ = 0;
  BitVector ans_;
  BitVector dummySat_;

  void Add(const uint64_t iClause, const int64_t iVar) {
    assert(1 <= int64_t(iClause) && int64_t(iClause) <= nClauses_);
    assert(1 <= llabs(iVar) && llabs(iVar) <= nVars_);
    clause2var_.Add(iClause, iVar);
    var2clause_.Add(iVar, iClause);
  }

  // Returns |false| iff the formula is unsatisfiable.
  bool Load(const std::string& filePath) {
    std::ifstream ifs(filePath);
    if(!ifs) {
      std::cerr << "Cannot open the file to load the formula from: " << filePath << std::endl;
      throw std::runtime_error("Cannot open input file.");
    }
    constexpr const uint32_t cBufSize = 8 * 1024 * 1024;
    std::unique_ptr<char[]> buffer(new char[cBufSize]);
    ifs.rdbuf()->pubsetbuf(buffer.get(), cBufSize);

    BlockingQueue<std::string> bqParsing;
    const int nParsingCpus = std::max<int>(1, int64_t(std::thread::hardware_concurrency())-1);

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
          clause2var_ = Linkage(nClauses_);
          var2clause_ = Linkage(nVars_);
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
        std::cerr << "It seems the formula contains no clauses, please check the input DIMACS file." << std::endl;
        return;
      }
      do {
        int64_t iVar = std::stoll(cmd);
        if(iVar == 0) {
          break;
        }
        if( !(1 <= llabs(iVar) && llabs(iVar) <= nVars_) ) {
          std::cerr << "Variable value out of range: " << iVar << std::endl;
          throw std::runtime_error("Variable value out of range in the input");
        }
        Add(iClause, iVar);
      } while(iss >> cmd);

      for(;;) {
        std::vector<std::string> lines;
        while(int64_t(lines.size()) < nParsingCpus) {
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
        #pragma omp parallel for num_threads(nParsingCpus)
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
              throw std::runtime_error("Variable value out of range");
            }
            Add(locClause, iVar);
          }
        }
        iClause += lines.size();
        if(int64_t(lines.size()) < nParsingCpus) {
          break;
        }
        if(iClause > nClauses_) {
          break;
        }
      }
      std::cout << "Finished parsing the input file lines." << std::endl;
    });

    std::string line;
    while(std::getline(ifs, line)) {
      bqParsing.Push(std::move(line));
    }
    bqParsing.RequestShutdown();
    std::cout << "Finished streaming the input file." << std::endl;
    parsingThr.join();

    PrepareLinkage();
  }

  void PrepareLinkage() {
    std::cout << "Sorting the linkage data structures." << std::endl;
    clause2var_.Sort();
    var2clause_.Sort();

    std::cout << "Finding and removing dummy clauses" << std::endl;
    RangeVector<RangeVector<std::vector<int64_t>, int8_t>, VCIndex> toRemove(-nClauses_, nClauses_);
    #pragma omp parallel for schedule(guided, kCacheLineSize)
    for(int64_t i=1; i<=nClauses_; i++) {
      // No variables at all in the clause - assume it's satisfied
      if(clause2var_.ArcCount(i) == 0) {
        dummySat_.Flip(i);
        continue;
      }
      toRemove[i] = RangeVector<std::vector<int64_t>, int8_t>(-1, 1);
      bool isDummy = false;
      for(int8_t sgnFrom=-1; sgnFrom<=1; sgnFrom+=2) {
        const VCIndex iClause = sgnFrom * i;
        for(int8_t sgnTo=-1; sgnTo<=1; sgnTo+=2) {
          for(VCIndex j=0; j<clause2var_.ArcCount(iClause, sgnTo); j++) {
            const VCIndex iVar = clause2var_.GetTarget(iClause, sgnTo, j);
            if(clause2var_.HasArc(iClause, -iVar)) {
              // We can't just remove it immediately, because then we would screw up the sorted data structures
              toRemove[iClause][sgnTo].emplace_back(j);
              isDummy = true;
            }
          }
        }
      }
      if(isDummy) {
        dummySat_.Flip(i);
      }
    }
    #pragma omp parallel for schedule(guided, kCacheLineSize)
    for(int64_t i=-nClauses_; i<=nClauses_; i++) {
      if(i == 0) {
        continue;
      }
      for(int8_t sgn=-1; sgn<=1; sgn+=2) {
        std::vector<VCIndex>& targets = clause2var_.sources_[i][sgn];
        const std::vector<VCIndex>& removals = toRemove[i][sgn];
        VCIndex k = 0; // index in removals
        VCIndex newSize = 0;
        for(VCIndex j=0; j<targets.size(); j++) {
          if(j == removals[k]) {
            k++;
            continue;
          }
          targets[newSize] = targets[j];
          newSize++;
        }
        assert(k == removals.size());
        assert(newSize + removals.size() == targets.size());
        targets.resize(newSize);
        // This way the data structure remains sorted
      }
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
    #pragma omp parallel for schedule(guided, kCacheLineSize)
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
    #pragma omp parallel for schedule(guided, kCacheLineSize)
    for(int64_t i=1; i<=nClauses_; i++) {
      if(!IsSatisfied(i, ans_)) {
        ans.Add(i);
      }
    }
    return ans;
  }

  BitVector SetGreedy1() const {
    BitVector ans(nVars_+1); // Init to false
    std::vector<std::pair<int64_t, int64_t>> counts_;
    #pragma omp parallel for schedule(guided, kCacheLineSize)
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
    #pragma omp parallel for schedule(guided, kCacheLineSize)
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

  // To avoid false sharing in SatTracker, don't call it - let the indices stay randomized
  void SortClauseLists() {
    #pragma omp parallel for schedule(guided, kCacheLineSize)
    for(int64_t i=1; i<=nVars_; i++) {
      std::vector<int64_t>& clauses = listVar2Clause_[i];
      std::sort(clauses.begin(), clauses.end(), [](const int64_t a, const int64_t b) {
        return llabs(a) < llabs(b);
      });
    }
  }

  // For a set of clauses, return the set of variables that dissatisfy the clauses
  std::vector<MultiItem<VCIndex>> ClauseFrontToVars(const VCTrackingSet& clauseFront, const BitVector& assignment) {
    TrackingSet<MultiItem<VCIndex>> varFront;
    std::vector<int64_t> vClauseFront = clauseFront.ToVector();
    #pragma omp parallel for schedule(guided, kCacheLineSize)
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
    // We don't serve shuffling here!
    return varFront.ToVector();
  }

  // For a set of variables, return the set of clauses that are dissatisfied by the variables
  std::vector<int64_t> VarFrontToClauses(const VCTrackingSet& varFront, const BitVector& assignment) {
    VCTrackingSet clauseFront;
    std::vector<int64_t> vVarFront = varFront.ToVector();
    #pragma omp parallel for schedule(guided, kCacheLineSize)
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

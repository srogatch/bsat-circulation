#include "Reduction.h"
#include "TrackingSet.h"
#include "Utils.h"
#include "SatTracker.h"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <map>
#include <chrono>

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
std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::high_resolution_clock::now();

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }

  // TODO: does it override the environment variable?
  omp_set_num_threads(Formula::nCpus_);
  // Enable nested parallelism
  omp_set_max_active_levels(4);

  std::mutex muUnsatClauses;
  std::mutex muFront;
  Formula formula;
  formula.Load(argv[1]);

  int64_t prevNUnsat = formula.nClauses_;

  // Now there are some clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  DefaultSatTracker satTr(formula);
  satTr.Populate(formula.ans_);

  int64_t bestInit = satTr.UnsatCount();
  std::cout << "All false: " << bestInit << ", ";
  std::cout.flush();

  TrackingSet initUSC = satTr.GetUnsat();
  TrackingSet initFront;
  int64_t altNUnsat=satTr.GradientDescend(false, &initUSC, &initFront); // for init it's usually better if we don't move an extra time
  std::cout << "GradientDescent: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
  } else {
    // Revert to all false
    formula.ans_ = BitVector(formula.nVars_+1);
  }

  BitVector altAsg = formula.SetGreedy1();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Greedy1: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg = formula.SetGreedy2();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Greedy2: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg.Randomize();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "Random: " << altNUnsat << ", ";
  std::cout.flush();
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  altAsg.SetTrue();
  altNUnsat = formula.CountUnsat(altAsg);
  std::cout << "All true: " << altNUnsat << std::endl;
  if(altNUnsat < bestInit) {
    bestInit = altNUnsat;
    formula.ans_ = altAsg;
  }

  satTr.Populate(formula.ans_);

  BitVector maxPartial;
  bool maybeSat = true;
  bool provenUnsat = false;
  std::unordered_set<std::pair<uint128, uint128>> seenMove;
  std::unordered_set<uint128> seenFront;
  std::mt19937_64 rng;
  int64_t lastFlush = formula.nClauses_ + 1;
  std::deque<Point> dfs;
  int64_t nStartUnsat;
  std::vector<int64_t> vFront;
  uint64_t cycleOffset = 0;
  int64_t lastGD = 0;
  int64_t nGD = 0; // the number of times Gradient Descent was launched

  std::vector<DefaultSatTracker> satTrackers(size_t(omp_get_max_threads()), satTr);
  std::vector<BitVector> next(size_t(omp_get_max_threads()), formula.ans_);
  std::vector<TrackingSet> parFront{size_t(omp_get_max_threads())};
  std::vector<TrackingSet> parRevVars{size_t(omp_get_max_threads())};
  while(maybeSat) {
    TrackingSet unsatClauses = satTr.GetUnsat();
    nStartUnsat = unsatClauses.set_.size();
    maxPartial = formula.ans_;
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    auto tmEnd = std::chrono::high_resolution_clock::now();
    double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmEnd - tmStart).count() / 1e9;
    double clausesPerSec = (prevNUnsat - nStartUnsat) / nSec;
    std::cout << "\tUnsatisfied clauses: " << nStartUnsat << " - elapsed " << nSec << " seconds, ";
    if(clausesPerSec >= 1 || clausesPerSec == 0) {
      std::cout << clausesPerSec << " clauses per second.";
    } else {
      std::cout << 1.0 / clausesPerSec << " seconds per clause.";
    }
    std::cout << std::endl;
    tmStart = tmEnd;
    prevNUnsat = nStartUnsat;
    
    TrackingSet front;
    if(nStartUnsat == bestInit) {
      front = initFront;
    }
    bool allowDuplicateFront = false;
    while(unsatClauses.set_.size() >= nStartUnsat) {
      assert(formula.ComputeUnsatClauses() == unsatClauses);
      if(front.set_.empty() || (!allowDuplicateFront && seenFront.find(front.hash_) != seenFront.end())) {
        front = unsatClauses;
        std::cout << "$";
        std::cout.flush();
      }

      std::vector<std::pair<int64_t, int64_t>> varSplit(formula.nVars_);
      #pragma omp parallel for schedule(guided)
      for(int64_t i=0; i<formula.nVars_; i++) {
        varSplit[i] = {i+1, 0};
      }
      vFront.assign(front.set_.begin(), front.set_.end());
      #pragma omp parallel for schedule(guided)
      for(int64_t i=0; i<vFront.size(); i++) {
        const int64_t originClause = vFront[i];
        for(const int64_t iVar : formula.clause2var_[originClause]) {
          if( (iVar < 0 && formula.ans_[-iVar]) || (iVar > 0 && !formula.ans_[iVar]) ) {
            // A dissatisfying arc
            const int64_t revV = llabs(iVar);
            reinterpret_cast<std::atomic<int64_t>*>(&varSplit[revV-1].second)->fetch_add(1);
          }
        }
      }

      ParallelShuffle(varSplit.data(), formula.nVars_);

      std::atomic<int64_t> nZeroes(0);
      #pragma omp parallel for schedule(guided)
      for(int64_t i=0; i<formula.nVars_; i++) {
        if(varSplit[i].second == 0) {
          nZeroes.fetch_add(1);
        }
      }
      std::atomic<int64_t> vsPos(0);
      #pragma omp parallel for schedule(guided)
      for(int64_t i=nZeroes; i<formula.nVars_; i++) {
        while(varSplit[i].second == 0) {
          int64_t pos = vsPos.fetch_add(1);
          int64_t t = varSplit[i].first;
          varSplit[i] = varSplit[pos];
          varSplit[pos] = {t, 0};
        }
      }
      #pragma omp parallel for
      for(int64_t i=0; i<satTrackers.size(); i++) {
        satTrackers[i] = satTr;
        next[i] = formula.ans_;
        parFront[i] = front;
        parRevVars[i].Clear();
      }
      // now varSplit contains first variables with 0 counts (i.e. not in the front), then the variables in the front with front clause counts
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestRevVertices;
      constexpr const int64_t knCombine = 5;
      const int64_t nSources = omp_get_max_threads();
      std::mutex muSeenMove, muSeenFront, muBestUnsat;
      #pragma omp parallel for
      for(int64_t i=0; i<nSources; i++) {
        int64_t baseFront = ((i+1) * BitVector::hashSeries_[0]) % (formula.nVars_-nZeroes);
        int64_t baseOut = ((i+2) * BitVector::hashSeries_[1]) % nZeroes;
        int64_t incl[knCombine];
        for(int64_t j=2; j<knCombine && j < formula.nVars_; j++) { // number of vars to combine
          for(int64_t k=1; k<j && k<formula.nVars_-nZeroes; k++) { // number of front vars to include
            const int64_t nFrontVars = k;
            const int64_t nOutVars = j-k;
            const int64_t nToInclude = j;
            int64_t l=0;
            for(; l<nFrontVars; l++) {
              incl[l] = varSplit[nZeroes+baseFront%(formula.nVars_-nZeroes)].first;
              baseFront++;
              for(int64_t m=0; m<l; m++) {
                if(incl[m] == incl[l]) {
                  // Avoid duplicates
                  l--;
                  continue;
                }
              }
              parRevVars[omp_get_thread_num()].Flip(incl[l]);
            }
            for(; l<nToInclude; l++) {
              incl[l] = varSplit[baseOut%nZeroes].first;
              baseOut++;
              for(int64_t m=nFrontVars; m<l; m++) {
                if(incl[m] == incl[l]) {
                  // Avoid duplicates
                  l--;
                  continue;
                }
              }
              parRevVars[omp_get_thread_num()].Flip(incl[l]);
            }
            {
              std::unique_lock<std::mutex> lock(muSeenMove);
              if(seenMove.find({parFront[omp_get_thread_num()].hash_, parRevVars[omp_get_thread_num()].hash_}) != seenMove.end()) {
                // Flip back
                for(l=0; l<nToInclude; l++) {
                  parRevVars[omp_get_thread_num()].Flip(incl[l]);
                }
                continue;
              }
            }
            TrackingSet newFront;
            // Flip forward
            for(l=0; l<nToInclude; l++) {
              next[omp_get_thread_num()].Flip(incl[l]);
              satTrackers[omp_get_thread_num()].FlipVar(incl[l] * (next[omp_get_thread_num()][incl[l]] ? 1 : -1), nullptr, &newFront);
            }
            bool bSeenFront = false;
            if(!seenFront.empty() || allowDuplicateFront)
            {
              std::unique_lock<std::mutex> lock(muSeenFront);
              bSeenFront = (seenFront.find(newFront.hash_) != seenFront.end());
            }
            if(!bSeenFront) {
              const int64_t stepUnsat = satTrackers[omp_get_thread_num()].UnsatCount();
              bool betterSol = false;
              {
                std::unique_lock<std::mutex> lock(muBestUnsat);
                if(stepUnsat < bestUnsat) {
                  bestUnsat = stepUnsat;
                  bestRevVertices = parRevVars[omp_get_thread_num()];
                  betterSol = true;
                }
              }
              if(betterSol) {
                if(stepUnsat < nStartUnsat) {
                  std::cout << "+";
                  std::cout.flush();
                }
                parFront[omp_get_thread_num()] = newFront;
                continue;
              }
            }
            // Flip back
            for(l=0; l<nToInclude; l++) {
              next[omp_get_thread_num()].Flip(incl[l]);
              satTrackers[l].FlipVar(incl[l] * (formula.ans_[incl[l]] ? 1 : -1), nullptr, nullptr);
              parRevVars[omp_get_thread_num()].Flip(incl[l]);
            }
          }
        }
      }

      if(bestUnsat >= formula.nClauses_) {
        std::cout << "#";

        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        if(!allowDuplicateFront) {
          seenFront.emplace(front.hash_);
        }

        if(front != unsatClauses) {
          // Retry with full/random front
          front = unsatClauses;
          continue;
        }

        if(!dfs.empty()) {
          formula.ans_ = std::move(dfs.back().assignment_);
          dfs.pop_back();
          satTr.Populate(formula.ans_);
          unsatClauses = satTr.GetUnsat();
          front = unsatClauses;
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

      seenMove.emplace(front.hash_, bestRevVertices.hash_);
      front.Clear();
      for(int64_t revV : bestRevVertices.set_) {
        formula.ans_.Flip(revV);
        satTr.FlipVar(revV * (formula.ans_[revV] ? 1 : -1), &unsatClauses, &front);
      }

      // Indicate a walk step
      //std::cout << " F" << front.set_.size() << ":B" << bestFront.set_.size() << ":U" << unsatClauses.set_.size() << " ";
      std::cout << ">";

      if(bestRevVertices.set_.size() >= 2) {
        while(seenMove.size() - lastGD > std::sqrt(formula.nClauses_)/(nStartUnsat+1) + 1) {
          std::cout << "G";
          std::cout.flush();
          nGD++;
          front.Clear();
          const int64_t oldNUnsat = unsatClauses.set_.size();
          const int64_t newUnsat = satTr.GradientDescend(true, &unsatClauses, &front);
          std::cout << oldNUnsat << "/" << newUnsat << "D";
          std::cout.flush();
          assert(unsatClauses.set_.size() == newUnsat);
          if(newUnsat > oldNUnsat - std::sqrt(oldNUnsat+4)) {
            lastGD = seenMove.size();
          }
          // Limit the size of the stack
          if(dfs.size() > formula.nVars_) {
            dfs.pop_front();
          }
          dfs.push_back(Point(formula.ans_));
        }
      }
    }
    std::cout << "\n\tTraversal size: " << seenMove.size()
      << ", Gradient Descents: " << nGD << std::endl;
  }

  {
    std::ofstream ofs(argv[2]);
    if(provenUnsat) {
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
      return 0;
    }

    if(maybeSat) {
      assert(formula.SolWorks());
      ofs << "s SATISFIABLE" << std::endl;
    } else {
      formula.ans_ = maxPartial;
      ofs << "s UNKNOWN" << std::endl;
      ofs << "c Unsatisfied clause count: " << nStartUnsat << std::endl;
    }
    int64_t nInLine = 0;
    for(int64_t i=1; i<=formula.nVars_; i++) {
      if(nInLine == 0) {
        ofs << "v ";
      }
      ofs << (formula.ans_[i] ? i : -i);
      nInLine++;
      if(nInLine >= 200) {
        nInLine = 0;
        ofs << "\n";
      } else {
        ofs << " ";
      }
    }
    ofs << "0" << std::endl;
  }
  return 0;
}

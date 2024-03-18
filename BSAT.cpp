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
  // Now there are some clause bitvectors
  BitVector::CalcHashSeries( std::max(formula.nVars_, formula.nClauses_) );

  DefaultSatTracker satTr(formula);
  satTr.Populate(formula.ans_);

  int64_t bestInit = satTr.UnsatCount();
  std::cout << "All false: " << bestInit << ", ";
  std::cout.flush();

  int64_t altNUnsat=satTr.GradientDescend(false); // for init it's usually better if we don't move an extra time
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

  std::unordered_map<uint128, int64_t> bv2nUnsat;
  bv2nUnsat[formula.ans_.hash_] = bestInit;
  BitVector maxPartial;
  bool maybeSat = true;
  bool provenUnsat = false;
  std::unordered_set<std::pair<TrackingSet, TrackingSet>, MoveHash> seenMove;
  std::unordered_set<TrackingSet> seenFront;
  std::mt19937_64 rng;
  int64_t lastFlush = formula.nClauses_ + 1;
  std::deque<Point> dfs;
  int64_t nStartUnsat;
  // Define them here to avoid reallocations
  std::vector<std::pair<int64_t, int64_t>> combs;
  std::vector<int64_t> vFront;
  std::vector<int64_t> incl;
  BitVector next;
  uint64_t cycleOffset = 0;
  while(maybeSat) {
    TrackingSet unsatClauses = formula.ComputeUnsatClauses();
    nStartUnsat = unsatClauses.set_.size();
    maxPartial = formula.ans_;
    if(nStartUnsat == 0) {
      std::cout << "Satisfied" << std::endl;
      break;
    }
    std::cout << "Unsatisfied clauses: " << nStartUnsat << std::endl;
    TrackingSet front;
    std::vector<int64_t> vClauses;
    // avoid reallocations
    vClauses.reserve(unsatClauses.set_.size() * 4);
    bool allowDuplicateFront = false;
    while(unsatClauses.set_.size() >= nStartUnsat) {
      assert(formula.ComputeUnsatClauses() == unsatClauses);
      if(front.set_.empty() || (!allowDuplicateFront && seenFront.find(front) != seenFront.end())) {
        //std::cout << "Empty front" << std::endl;
        front = unsatClauses;
      }
      std::unordered_map<int64_t, int64_t> candVs;
      vFront.assign(front.set_.begin(), front.set_.end());
      #pragma omp parallel for
      for(int64_t i=0; i<vFront.size(); i++) {
        const int64_t originClause = vFront[i];
        for(const int64_t iVar : formula.clause2var_[originClause]) {
          if( (iVar < 0 && formula.ans_[-iVar]) || (iVar > 0 && !formula.ans_[iVar]) ) {
            // A dissatisfying arc
            const int64_t revV = llabs(iVar);
            #pragma omp critical
            candVs[revV]++;
          }
        }
      }
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestFront, bestUnsatClauses, bestRevVertices;

      combs.assign(candVs.begin(), candVs.end());

      if( combs.size() <= std::log2(formula.nClauses_) ) {
        if( combs.size() >= 2 * omp_get_max_threads() ) {
          ParallelShuffle(combs.data(), combs.size());
        } else {
          std::shuffle(combs.begin(), combs.end(), rng);
        }
        std::stable_sort(std::execution::par, combs.begin(), combs.end(), [](const auto& a, const auto& b) {
          return a.second > b.second;
        });
      } else {
        std::sort(std::execution::par, combs.begin(), combs.end(), [](const auto& a, const auto& b) {
          return a.second > b.second || (a.second == b.second && hash64(a.first) < hash64(b.first));
        });
      }
      uint64_t nCombs = 0;
      uint64_t prevBestAtCombs = 0;
      next = formula.ans_;
      TrackingSet stepRevs;
      for(int64_t nIncl=1; nIncl<=combs.size(); nIncl++) {
        if(AccComb(combs.size(), nIncl) > 100) {
          std::cout << " C" << combs.size() << "," << nIncl << " ";
          std::flush(std::cout);
        }
        incl.clear();
        for(int64_t j=0; j<nIncl; j++) {
          incl.push_back(j);
        }
        int64_t nBeforeCombs = nCombs;
        for(;;) {
          nCombs++;
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = combs[(incl[j]+cycleOffset) % combs.size()].first;
            auto it = stepRevs.set_.find(revV);
            if(it == stepRevs.set_.end()) {
              stepRevs.Add(revV);
            } else {
              stepRevs.Remove(revV);
            }
            next.Flip(revV);
          }
          {
            auto unflip = Finally([&combs, &incl, &stepRevs, &next, nIncl, cycleOffset]() {
              // Flip bits back
              for(int64_t j=0; j<nIncl; j++) {
                const int64_t revV = combs[(incl[j]+cycleOffset) % combs.size()].first;
                auto it = stepRevs.set_.find(revV);
                if(it == stepRevs.set_.end()) {
                  stepRevs.Add(revV);
                } else {
                  stepRevs.Remove(revV);
                }
                next.Flip(revV);
              }
            });

            auto it = bv2nUnsat.find(next.hash_);
            bool maybeSuperior = (it == bv2nUnsat.end() || it->second < bestUnsat);
            if(!maybeSuperior) {
              // This combination has been too lightweight to count
              prevBestAtCombs++;
            }
            if( maybeSuperior && (seenMove.find({front, stepRevs}) == seenMove.end()) ) {
              // TODO: this is the bottleneck for huge formulas, but it can be alleviated
              // by means of keeping track of the degree of satisfiability (how many variables
              // are satisfying each clause) and updating the data structure as we flip
              // variables
              TrackingSet newFront;
              TrackingSet newUnsatClauses = unsatClauses;
              // TODO: these can be parallelized if ordered maps are used, then copied to the vector in partitions
              std::unordered_set<int64_t> clauses;
              for(const int64_t sr : stepRevs.set_) {
                const std::vector<int64_t>& src = formula.listVar2Clause_[sr];
                for(int64_t i=0; i<src.size(); i++) {
                  clauses.emplace(llabs(src[i]));
                }
              }
              std::vector<int64_t> vClauses(clauses.begin(), clauses.end());
              #pragma omp parallel for
              for(int64_t j=0; j<vClauses.size(); j++) {
                const uint64_t absClause = vClauses[j];
                const bool oldSat = formula.IsSatisfied(absClause, formula.ans_);
                const bool newSat = formula.IsSatisfied(absClause, next);
                if(newSat) {
                  if(!oldSat) {
                    std::unique_lock<std::mutex> lock(muUnsatClauses);
                    newUnsatClauses.Remove(absClause);
                  }
                } else {
                  if(oldSat)
                  {
                    {
                      std::unique_lock<std::mutex> lock(muUnsatClauses);
                      newUnsatClauses.Add(absClause);
                    }
                    {
                      std::unique_lock<std::mutex> lock(muFront);
                      newFront.Add(absClause);
                    }
                  }
                }
              }
              const int64_t stepUnsat = newUnsatClauses.set_.size();
              bv2nUnsat[next.hash_] = stepUnsat;
              if(allowDuplicateFront || seenFront.find(newFront) == seenFront.end()) {
                if(stepUnsat < bestUnsat) {
                  bestUnsat = stepUnsat;
                  bestFront = std::move(newFront);
                  bestUnsatClauses = std::move(newUnsatClauses);
                  bestRevVertices = stepRevs;

                  if(bestUnsat < nStartUnsat) {
                    prevBestAtCombs = nCombs;
                    unflip.Disable();
                    std::cout << "+";
                  }
                }
              }
            }
          }
          if(bestUnsat < nStartUnsat
            // These little combinations are considered a light operation
            && nCombs - prevBestAtCombs > std::pow(combs.size(), 1.0/3))
          {
            break;
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
        // Let the next traversal start from the combination we left on
        cycleOffset += nCombs - nBeforeCombs;
        if(bestUnsat < std::min<int64_t>(
            std::max(nStartUnsat + nCombs - 1, unsatClauses.set_.size()*2),
            formula.nClauses_))
        {
          cycleOffset -= bestRevVertices.set_.size();
          break;
        }
      }
      if(nCombs > formula.nVars_) {
        std::cout << "Combinations to next: " << nCombs << std::endl;
      }

      if(bestUnsat >= formula.nClauses_) {
        //std::cout << "The front of " << front.size() << " clauses doesn't lead anywhere." << std::endl;
        if(!allowDuplicateFront) {
          seenFront.emplace(front);
        }
        if(front != unsatClauses) {
          // Retry with full front
          std::cout << "$";
          front = unsatClauses;
          continue;
        }

        if(!dfs.empty()) {
          formula.ans_ = std::move(dfs.back().assignment_);
          dfs.pop_back();
          unsatClauses = formula.ComputeUnsatClauses();
          front.Clear();
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

      for(int64_t revV : bestRevVertices.set_) {
        formula.ans_.Flip(revV);
      }
      seenMove.emplace(front, bestRevVertices);
      if(bestRevVertices.set_.size() >= 2) {
        // TODO: More than 2 variables have been changed at once - try to descend into a better solution by flipping all variables one by one
      }
      // Indicate a walk step
      //std::cout << " F" << front.set_.size() << ":B" << bestFront.set_.size() << ":U" << unsatClauses.set_.size() << " ";
      std::cout << ">";
      front = std::move(bestFront);
      unsatClauses = std::move(bestUnsatClauses);
    }
    std::cout << "Traversal size: " << seenMove.size() << ", assignments considered: " << bv2nUnsat.size() << std::endl;
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

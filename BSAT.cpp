#include "Reduction.h"
#include "TrackingSet.h"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>

namespace detail {

template <typename F>
struct FinalAction {
  FinalAction(F&& f) : clean_{std::move(f)} {}
  ~FinalAction() {
    if (enabled_) clean_();
  }
  void Disable() { enabled_ = false; };

 private:
  F clean_;
  bool enabled_{true};
};

}  // namespace detail

template <typename F>
detail::FinalAction<F> Finally(F&& f) {
  return detail::FinalAction<F>(std::move(f));
}

template <typename T>
void ParallelShuffle(T* data, const size_t count) {
  const uint32_t nThreads = Formula::nCpus_;

  std::atomic_flag* syncs = static_cast<std::atomic_flag*>(malloc(count * sizeof(std::atomic_flag)));
  auto clean_syncs = Finally([&]() { free(syncs); });
#pragma omp parallel for num_threads(nThreads)
  for (size_t i = 0; i < count; i++) {
    new (syncs + i) std::atomic_flag(ATOMIC_FLAG_INIT);
  }

  const size_t nPerThread = (count + nThreads - 1) / nThreads;
#pragma omp parallel for num_threads(nThreads)
  for (size_t i = 0; i < nThreads; i++) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, count - 1);
    const size_t iFirst = nPerThread * i;
    const size_t iLimit = std::min(nPerThread + iFirst, count);
    if (iLimit <= iFirst) {
      continue;
    }
    for (size_t j = iFirst; j < iLimit; j++) {
      const size_t fellow = dist(rng);
      if (fellow == j) {
        continue;
      }
      const size_t sync1 = std::min(j, fellow);
      const size_t sync2 = std::max(j, fellow);
      while (syncs[sync1].test_and_set(std::memory_order_acq_rel)) {
        std::this_thread::yield();
      }
      while (syncs[sync2].test_and_set(std::memory_order_acq_rel)) {
        std::this_thread::yield();
      }
      std::swap(data[sync1], data[sync2]);
      syncs[sync2].clear(std::memory_order_release);
      syncs[sync1].clear(std::memory_order_release);
    }
  }
}

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
const uint32_t BitVector::nCpus_ = std::thread::hardware_concurrency();

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  std::mutex muUnsatClauses;
  std::mutex muFront;
  Formula formula;
  formula.Load(argv[1]);
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
  std::vector<int64_t> combs;
  std::vector<int64_t> vFront;
  std::vector<int64_t> incl;
  BitVector next;
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
      vFront.assign(front.set_.begin(), front.set_.end());
      combs.clear();
      #pragma omp parallel for num_threads(Formula::nCpus_)
      for(int64_t i=0; i<vFront.size(); i++) {
        const int64_t originClause = vFront[i];
        for(const int64_t iVar : formula.clause2var_[originClause]) {
          if( (iVar < 0 && formula.ans_[-iVar]) || (iVar > 0 && !formula.ans_[iVar]) ) {
            // A dissatisfying arc
            #pragma omp critical
            combs.emplace_back(iVar);
          }
        }
      }
      int64_t bestUnsat = formula.nClauses_+1;
      TrackingSet bestFront, bestUnsatClauses, bestRevVertices, stepRevs;

      std::sort(std::execution::par, combs.begin(), combs.end(), [](const int64_t a, const int64_t b) {
        return llabs(a) < llabs(b);
      });
      next = formula.ans_;
      #pragma omp parallel for num_threads(Formula::nCpus_)
      for(int64_t i=0; i<combs.size(); i++) {
        const int64_t revV = llabs(combs[i]);
        if(i == 0 || revV != llabs(combs[i-1])) {
          unsigned short r;
          while(!_rdrand16_step(&r)); // TODO: use xoroshiro** or similar
          if(r & 1) {
            #pragma omp critical
            stepRevs.Add(revV);
            next.Flip(revV);
          }
        }
      }

      if(combs.size() > 2 * Formula::nCpus_) {
        ParallelShuffle(combs.data(), combs.size());
      } else {
        std::shuffle(combs.begin(), combs.end(), rng);
      }
      uint64_t nCombs = 0;
      for(int64_t nIncl=1; nIncl<=combs.size(); nIncl++) {
        if(AccComb(combs.size(), nIncl) > 100) {
          std::cout << " C" << combs.size() << "," << nIncl << " ";
          std::flush(std::cout);
        }
        incl.clear();
        for(int64_t j=0; j<nIncl; j++) {
          incl.push_back(j);
        }
        for(;;) {
          nCombs++;
          assert(next == formula.ans_);
          for(int64_t j=0; j<nIncl; j++) {
            const int64_t revV = llabs(combs[incl[j]]);
            auto it = stepRevs.set_.find(revV);
            if(it == stepRevs.set_.end()) {
              stepRevs.Add(revV);
            } else {
              stepRevs.Remove(revV);
            }
            next.Flip(revV);
          }
          {
            auto unflip = Finally([&]() {
              // Flip bits back
              for(int64_t j=0; j<nIncl; j++) {
                const int64_t revV = llabs(combs[incl[j]]);
                auto it = stepRevs.set_.find(revV);
                if(it == stepRevs.set_.end()) {
                  stepRevs.Add(revV);
                } else {
                  stepRevs.Remove(revV);
                }
                next.Flip(revV);
              }
            });

            if(seenMove.find({front, stepRevs}) == seenMove.end()) {
              TrackingSet newFront;
              TrackingSet newUnsatClauses = unsatClauses;
              vClauses.clear();
              for(const int64_t revV : stepRevs.set_) {
                const std::vector<int64_t>& srcClauses = formula.listVar2Clause_[revV];
                for(const int64_t iClause : srcClauses) {
                  vClauses.emplace_back(llabs(iClause));
                }
              }
              std::sort(std::execution::par, vClauses.begin(), vClauses.end());
              #pragma omp parallel for num_threads(Formula::nCpus_)
              for(int64_t j=0; j<vClauses.size(); j++) {
                const uint64_t absClause = vClauses[j];
                if(j > 0 && absClause == vClauses[j-1]) {
                  continue;
                }
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
              if(allowDuplicateFront || seenFront.find(newFront) == seenFront.end()) {
                // UNSAT counting is a heavy and parallelized operation
                const int64_t stepUnsat = newUnsatClauses.set_.size();
                if(stepUnsat < bestUnsat) {
                  bestUnsat = stepUnsat;
                  bestFront = std::move(newFront);
                  bestUnsatClauses = std::move(newUnsatClauses);
                  bestRevVertices = stepRevs;

                  if(nStartUnsat <= std::max(100.0, std::sqrt(formula.nVars_))) {
                    if(bestUnsat < nStartUnsat) {
                      break;
                    }
                  } else {
                    if(bestUnsat < nStartUnsat) {
                      unflip.Disable();
                      std::cout << " +" << nStartUnsat - bestUnsat << " ";
                      std::flush(std::cout);
                    }
                    if(bestUnsat < nStartUnsat - std::sqrt(nStartUnsat)) {
                      break;
                    }
                  }
                }
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
        if(bestUnsat < std::min<int64_t>(
            std::max(nStartUnsat + nCombs - 1, unsatClauses.set_.size()*2),
            formula.nClauses_))
        {
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
      // Indicate a DFS step
      //std::cout << " F" << front.set_.size() << ":B" << bestFront.set_.size() << ":U" << unsatClauses.set_.size() << " ";
      std::cout << ">";
      front = std::move(bestFront);
      unsatClauses = std::move(bestUnsatClauses);
    }
    std::cout << "Search size: " << seenMove.size() << std::endl;
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

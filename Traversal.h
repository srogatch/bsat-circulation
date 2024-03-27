#pragma once

#include "TrackingSet.h"
#include "BitVector.h"

#include <unordered_map>
#include <unordered_set>
#include <deque>

struct Point {
  BitVector assignment_;
  int64_t nUnsat_;

  Point(const BitVector& assignment, const int64_t nUnsat)
  : assignment_(assignment), nUnsat_(nUnsat)
  { }

  Point(Point&& src) : assignment_(std::move(src.assignment_)), nUnsat_(std::move(src.nUnsat_))
  {}

  Point& operator=(Point&& src) {
    if(this != &src) {
      assignment_ = std::move(src.assignment_);
      nUnsat_ = std::move(src.nUnsat_);
    }
    return *this;
  }
};

struct Traversal {
  static constexpr const int64_t cMaxDfsRamBytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
  TrackingSet<uint128> seenFront_;
  TrackingSet<uint128> seenAssignment_;
  TrackingSet<std::pair<uint128, uint128>, std::hash<std::pair<uint128, uint128>>> seenMove_;
  std::deque<Point> dfs_;
  mutable std::atomic_flag syncDfs_ = ATOMIC_FLAG_INIT;

  void FoundMove(const VCTrackingSet& front, const VCTrackingSet& revVars, const BitVector& assignment, const int64_t nUnsat)
  {
    seenMove_.Add(std::make_pair(front.hash_, revVars.hash_));
    if(seenAssignment_.Contains(assignment.hash_)) {
      return; // added earlier, perhaps concurrently by another thread - don't put it to DFS here thus
    }
    { // DFS
      SpinLock lock(syncDfs_);
      if(!dfs_.empty() && nUnsat > dfs_.back().nUnsat_) {
        return;
      }
    }
    {
      Point p(assignment, nUnsat);
      BitVector toRelease; // release outside of the lock
      SpinLock lock(syncDfs_);
      if(dfs_.empty() || nUnsat <= dfs_.back().nUnsat_) {
        if(dfs_.size() * assignment.nQwords_ * sizeof(uint64_t) >= cMaxDfsRamBytes) {
          // don't release here - we are holding the lock
          toRelease = std::move(dfs_.front().assignment_);
          dfs_.pop_front();
        }
        dfs_.push_back(std::move(p));
      }
    }
  }

  bool IsSeenMove(const VCTrackingSet& front, const VCTrackingSet& revVars) const {
    assert(front.Size() > 0);
    return seenMove_.Contains(std::make_pair(front.hash_, revVars.hash_));
  }

  // This is not (yet) thread-safe
  bool IsSeenFront(const VCTrackingSet& front) const {
    return seenFront_.Contains(front.hash_);
  }

  bool IsSeenAssignment(const BitVector& assignment) const {
    return seenAssignment_.Contains(assignment.hash_);
  }

  // This is not (yet) thread-safe
  void OnFrontExhausted(const VCTrackingSet& front) {
    seenFront_.Add(front.hash_);
  }

  bool StepBack(BitVector& backup) {
    SpinLock lock(syncDfs_);
    if(dfs_.empty()) {
      return false;
    }
    backup = std::move(dfs_.back().assignment_);
    dfs_.pop_back();
    return true;
  }
};

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

  void OnSeenAssignment(const BitVector& assignment, const int64_t nUnsat) {
    // if( !seenAssignment_.Add(assignment.hash_) ) {
    //   return; // added earlier, perhaps concurrently by another thread - don't put it to DFS here thus
    // }
    { // DFS
      SpinLock lock(syncDfs_);
      if(!dfs_.empty() && nUnsat > dfs_.back().nUnsat_) {
        return;
      }
    }
    if( !seenAssignment_.Add(assignment.hash_) ) {
      return; // added earlier, perhaps concurrently by another thread - don't put it to DFS here thus
    }
    {
      Point p(assignment, nUnsat);
      BitVector toRelease; // release outside of the lock
      {
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
      if(toRelease.hash_ != 0) {
        seenAssignment_.Remove(toRelease.hash_);
      }
    }
  }

  void FoundMove(const VCTrackingSet& front, const VCTrackingSet& revVars) {
    // seenMove_.Add(std::make_pair(front.hash_, revVars.hash_));
  }

  void FoundMove(const VCTrackingSet& front, const VCTrackingSet& revVars, const BitVector& assignment, const int64_t nUnsat)
  {
    FoundMove(front, revVars);
    OnSeenAssignment(assignment, nUnsat);
  }

  bool IsSeenMove(const VCTrackingSet& unsatClauses, const VCTrackingSet& front, const VCTrackingSet& revVars) const {
    return false;
    // if(front.Size() > 0) {
    //   if(seenMove_.Contains(std::make_pair(front.hash_, revVars.hash_))) {
    //     return true;
    //   }
    // }
    // return seenMove_.Contains(std::make_pair(unsatClauses.hash_, revVars.hash_));
  }

  // This is not (yet) thread-safe
  bool IsSeenFront(const VCTrackingSet& front, const VCTrackingSet& unsatClauses) const {
    return seenFront_.Contains(front.hash_) || (&front != &unsatClauses && seenFront_.Contains(unsatClauses.hash_));
  }

  bool IsSeenAssignment(const BitVector& assignment) const {
    return seenAssignment_.Contains(assignment.hash_);
    //return false;
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

  bool PopIfNotWorse(BitVector& ans, const VCIndex maxUnsat) {
    SpinLock lock(syncDfs_);
    if(dfs_.empty()) {
      return false;
    }
    if(dfs_.back().nUnsat_ > maxUnsat) {
      return false;
    }
    ans = std::move(dfs_.back().assignment_);
    dfs_.pop_back();
    return true;
  }
};

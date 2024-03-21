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
};

struct Traversal {
  std::unordered_set<uint128> seenFront_;
  std::unordered_set<std::pair<uint128, uint128>> seenMove_;
  std::deque<Point> dfs_;

  void FoundMove(const TrackingSet& front, const TrackingSet& revVars, const BitVector& assignment, const int64_t nUnsat) {
    seenMove_.emplace(front.hash_, revVars.hash_);
    if(nUnsat < dfs_.back().nUnsat_) {
      dfs_.push_back(Point(assignment, nUnsat));
    }
  }

  bool IsSeenMove(const TrackingSet& front, const TrackingSet& revVars) const {
    assert(!front.set_.empty());
    return seenMove_.find({front.hash_, revVars.hash_}) != seenMove_.end();
  }

  bool IsSeenFront(const TrackingSet& front) const {
    return seenFront_.find(front.hash_) != seenFront_.end();
  }

  void OnFrontExhausted(const TrackingSet& front) {
    seenFront_.emplace(front.hash_);
  }

  bool StepBack(BitVector& backup) {
    if(dfs_.empty()) {
      return false;
    }
    backup = std::move(dfs_.back().assignment_);
    dfs_.pop_back();
    return true;
  }
};

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

// TODO: thread safety
struct Traversal {
  std::unordered_set<uint128> seenFront_;
  std::unordered_set<uint128> seenAssignment_;
  std::unordered_set<std::pair<uint128, uint128>> seenMove_;
  std::deque<Point> dfs_;
  mutable std::mutex muSeenAsg_;
  mutable std::mutex muSeenMove_;
  mutable std::mutex muDfs_;

  void FoundMove(const VCTrackingSet& front, const VCTrackingSet& revVars, const BitVector& assignment, const int64_t nUnsat)
  {
    { // Move
      std::unique_lock<std::mutex> lock(muSeenMove_);  
      seenMove_.emplace(front.hash_, revVars.hash_);
    }
    { // Assignment
      std::unique_lock<std::mutex> lock(muSeenAsg_);
      if(seenAssignment_.find(assignment.hash_) != seenAssignment_.end()) {
        return; // don't put it to DFS
      }
      seenAssignment_.emplace(assignment.hash_);
    }
    { // DFS
      std::unique_lock<std::mutex> lock(muDfs_);
      if(dfs_.empty() || nUnsat <= dfs_.back().nUnsat_) {
        dfs_.push_back(Point(assignment, nUnsat));
      }
    }
  }

  bool IsSeenMove(const VCTrackingSet& front, const VCTrackingSet& revVars) const {
    assert(front.Size() > 0);
    std::unique_lock<std::mutex> lock(muSeenMove_);
    return seenMove_.find({front.hash_, revVars.hash_}) != seenMove_.end();
  }

  // This is not (yet) thread-safe
  bool IsSeenFront(const VCTrackingSet& front) const {
    return seenFront_.find(front.hash_) != seenFront_.end();
  }

  bool IsSeenAssignment(const BitVector& assignment) const {
    std::unique_lock<std::mutex> lock(muSeenAsg_);
    return seenAssignment_.find(assignment.hash_) != seenAssignment_.end();
  }

  // This is not (yet) thread-safe
  void OnFrontExhausted(const VCTrackingSet& front) {
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

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

static constexpr const int64_t kInfFlow = 1LL << 61;

struct Arc {
  int64_t from_ = 0;
  int64_t to_ = 0;
  int64_t low_ = 0;
  int64_t high_ = kInfFlow;
  int64_t flow_ = 0;

  Arc() {}

  Arc(const int64_t from, const int64_t to, const int64_t low=0, const int64_t high=kInfFlow)
  : from_(from), to_(to), low_(low), high_(high)
  { }
};

using Linkage = std::unordered_map<int64_t, std::unordered_map<int64_t, std::shared_ptr<Arc>>>;

struct Graph {
  static constexpr const int64_t INVALID_VERTEX = 0;
  Linkage links_;
  Linkage backlinks_;

  std::shared_ptr<Arc> AddMerge(const Arc& arc) {
    auto it = links_.find(arc.from_);
    if(it != links_.end()) {
      auto jt = it->second.find(arc.to_);
      if(jt != it->second.end()) {
        // Merge
        std::shared_ptr<Arc> ans = jt->second;
        ans->flow_ += arc.flow_;
        ans->low_ += arc.low_;
        ans->high_ += arc.high_;
        return ans;
      }
    }
    // Add
    return links_[arc.from_][arc.to_] = backlinks_[arc.to_][arc.from_] = std::make_shared<Arc>(arc);
  }
  std::shared_ptr<Arc> Get(const int64_t from, const int64_t to) {
    auto it = links_.find(from);
    if(it == links_.end()) {
      return nullptr;
    }
    auto jt = it->second.find(to);
    if(jt == it->second.end()) {
      return nullptr;
    }
    return jt->second;
  }

  std::shared_ptr<Arc> BackGet(const int64_t to, const int64_t from) {
    auto it = backlinks_.find(to);
    if(it == links_.end()) {
      return nullptr;
    }
    auto jt = it->second.find(from);
    if(jt == it->second.end()) {
      return nullptr;
    }
    return jt->second;
  }

  // Returns true if the arc (and maybe the source vertex) was removed, or false if the edge didn't exist
  bool Remove(const int64_t from, const int64_t to) {
    auto it = links_.find(from);
    if(it == links_.end()) {
      return false;
    }
    auto jt = it->second.find(to);
    if(jt == it->second.end()) {
      return false;
    }
    // Remove the arc from the graph
    it->second.erase(jt);
    if(it->second.empty()) {
      // Remove the source vertex if it doesn't have any more outgoing arcs
      links_.erase(it);
    }
    // Erase backlinks
    it = backlinks_.find(to);
    it->second.erase(from);
    if(it->second.empty()) {
      backlinks_.erase(it);
    }
    return true;
  }
};


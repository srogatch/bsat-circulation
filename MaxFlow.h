#pragma once

#include "Graph.h"

#include <queue>
#include <vector>
#include <stack>
#include <limits>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

struct MaxFlow {
  Graph* pG_;
  int64_t result_ = 0;

private: // methods

public:
  explicit MaxFlow(Graph& g, const int64_t vS, const int64_t vT) : pG_(&g) {
    // Clear flows
    for(auto src : pG_->links_) {
      for(auto dest : src.second) {
        dest.second->flow_ = 0;
      }
    }
    if(vS == vT) {
      result_ = kInfFlow;
      return;
    }
    result_ = 0;
    bool augmented = true;
    // |increment|, |nextVertex|: increment is positive for forward links and negative for back links
    std::vector<std::pair<int64_t, int64_t>> frontier;
    std::vector<int64_t> path;
    // incFlow is true means the forward arc was taken (the flow must increase over it)
    std::vector<bool> incFlow;
    std::unordered_map<int64_t, int64_t> prev;
    while(augmented) {
      augmented = false;
      prev.clear();
      std::queue<int64_t> qu;
      qu.push(vS);
      prev[vS] = Graph::INVALID_VERTEX;
      while(!qu.empty()) {
        const int64_t vCur = qu.front();
        qu.pop();
        frontier.clear();
        for(const auto& target : pG_->links_[vCur]) {
          if(prev.find(target.first) == prev.end()) {
            const int64_t incr = target.second->high_ - target.second->flow_;
            if(incr > 0) {
              frontier.emplace_back(incr, target.first);
            }
          }
        }
        for(const auto& source : pG_->backlinks_[vCur]) {
          if(prev.find(source.first) == prev.end()) {
            const int64_t incr = source.second->flow_;
            if(incr > 0) {
              frontier.emplace_back(-incr, source.first);
            }
          }
        }
        for(const auto& unit : frontier) {
          if(unit.second == vT) {
            int64_t augFlow = llabs(unit.first);
            int64_t vNext = vCur;
            path.clear();
            incFlow.clear();
            while(vNext != vS && augFlow > 0) {
              path.push_back(vNext);
              const int64_t vPrev = prev[vNext];
              // Removed from the tree due to previous augmentations
              if(vPrev == Graph::INVALID_VERTEX) {
                augFlow = 0;
                break;
              }
              std::shared_ptr<Arc> forward = pG_->Get(vPrev, vNext);
              std::shared_ptr<Arc> backward = pG_->BackGet(vPrev, vNext);
              if(forward != nullptr && backward != nullptr) {
                int64_t curFlow = 0;
                if(forward->high_ - forward->flow_ >= backward->flow_) {
                  curFlow = forward->high_ - forward->flow_;
                  incFlow.push_back(true);
                } else {
                  curFlow = backward->flow_;
                  incFlow.push_back(false);
                }
                augFlow = std::min(augFlow, curFlow);
              } else if(forward != nullptr) {
                augFlow = std::min(augFlow, forward->high_ - forward->flow_);
                incFlow.push_back(true);
              } else  if(backward != nullptr) {
                augFlow = std::min(augFlow, backward->flow_);
                incFlow.push_back(false);
              } else {
                throw std::runtime_error("Data Structures seem to be corrupt.");
              }
              vNext = vPrev;
            }
            if(augFlow == 0) {
              for(int64_t i=0; i<path.size(); i++) {
                prev.erase(path[i]);
              }
              continue;
            }
            path.push_back(vS);
            if(unit.first > 0) {
              pG_->Get(vCur, vT)->flow_ += augFlow;
            } else {
              pG_->BackGet(vCur, vT)->flow_ -= augFlow;
            }
            result_ += augFlow;
            int64_t iWeak = -1;
            for(int64_t i=0; i+1<path.size(); i++) {
              const int64_t vNext = path[i];
              const int64_t vPrev = path[i+1];
              if(incFlow[i]) {
                std::shared_ptr<Arc> forward = pG_->Get(vPrev, vNext);
                forward->flow_ += augFlow;
                if(forward->flow_ == forward->high_) {
                  iWeak = i;
                }
              } else {
                std::shared_ptr<Arc> backward = pG_->BackGet(vPrev, vNext);
                backward->flow_ -= augFlow;
                if(backward->flow_ == 0) {
                  iWeak = i;
                }
              }
            }
            for(int64_t i=0; i<=iWeak; i++) {
              prev.erase(path[i]);
            }
            augmented = true;
            continue;
          }
        }
      }
    }
  }
};

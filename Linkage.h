#pragma once

#include "RangeVector.h"
#include "Utils.h"

struct Linkage {
  using TBackend = RangeVector<std::vector<VCIndex>, VCIndex>;
  TBackend backend_;

  Linkage() = default;

  explicit Linkage(const VCIndex maxItem) {
    backend_ = TBackend(-maxItem, maxItem);
  }

  // Thread-safe relative to other Add() calls.
  void Add(const VCIndex from, const VCIndex to) 
  {
    {
      auto lock = backend_.With(from);
      backend_[from].emplace_back(to);
    }
    {
      auto lock = backend_.With(-from);
      backend_[-from].emplace_back(-to);
    }
  }


};

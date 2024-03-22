#pragma once

#include <vector>
#include <mutex>

template<typename TItem, typename TIndex> struct RangeVector {
  struct TEntry {
    TItem item_;
    std::mutex mu_;
  };
  std::vector<TEntry> entires_;
  TIndex minIndex_ = 0;
  TIndex maxIndex_ = -1;

  RangeVector() = default;

  explicit RangeVector(const TIndex minIndex, const TIndex maxIndex)
    : entires_(maxIndex-minIndex+1), minIndex_(minIndex), maxIndex_(maxIndex)
  { }

  TItem& operator[](const TIndex index) {
    return entires_[index - minIndex_].item_;
  }
  const TItem& operator[](const TIndex index) const {
    return entires_[index - minIndex_].item_;
  }
  std::unique_lock<std::mutex> With(const TIndex index) {
    return std::unique_lock<std::mutex>(entires_[index-minIndex_].mu_);
  }
};

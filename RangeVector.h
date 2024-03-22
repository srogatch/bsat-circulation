#pragma once

#include <vector>

template<typename TItem, typename TIndex> struct RangeVector {
  std::vector<TItem> backend_;
  TIndex minIndex_;
  TIndex maxIndex_;

  explicit RangeVector(const TIndex minIndex, const TIndex maxIndex)
    : minIndex_(minIndex), maxIndex(maxIndex), backend_(maxIndex-minIndex+1)
  { }

  TItem& operator[](const TIndex index) {
    return backend_[index - minIndex_];
  }
  const TItem& operator[](const TIndex index) const {
    return backend_[index - minIndex_];
  }
};

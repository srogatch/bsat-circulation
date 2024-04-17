#pragma once

#include "GpuUtils.cuh"

struct GpuUnordSet {
  static constexpr const float cShrinkOccupancy = 0.42;
  static constexpr const float cStartOccupancy = 0.66;
  static constexpr const float cGrowOccupancy = 0.9;
  static constexpr const uint32_t cHashMul = 2147483647u;
  __uint128_t hash_ = 0;
  uint32_t* buffer_ = nullptr;
  VciGpu nBuckets_ = 0;
  VciGpu count_ = 0;
  int16_t bitsPerPack_ = 0;

  static constexpr const int16_t cBufElmBits = 8 * sizeof(buffer_[0]);
  static constexpr const int16_t cVectBits = sizeof(__uint128_t) * 8;

  GpuUnordSet() = default;

  static __host__ __device__ uint32_t GetPack(const VciGpu at, const uint32_t* buffer, const int16_t bitsPerPack) {
    const uint32_t iLowBit = at * bitsPerPack;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerPack - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    uint32_t val;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    if(iLowVect == iHighVect) {
      val = (buffer[iLowVect] >> lowOffs) & ((1u<<bitsPerPack)-1);
    } else {
      const uint32_t bitsInLow = cBufElmBits-lowOffs;
      const uint32_t bitsInHigh = bitsPerPack-bitsInLow;
      const uint32_t lowMask = (1u<<bitsInLow)-1;
      const uint32_t highMask = (1u<<bitsInHigh)-1;
      val = ((buffer[iLowVect]>>lowOffs) & lowMask) | ((buffer[iHighVect] & highMask)<<bitsInLow);
    }
    return val;
  }

  __host__ __device__ uint32_t GetPack(const VciGpu at) const {
    return GetPack(at, buffer_, bitsPerPack_);
  }

  static __host__ __device__ void SetPack(const VciGpu at, const uint32_t val, uint32_t* buffer, const int16_t bitsPerPack) {
    const uint32_t iLowBit = at * bitsPerPack;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerPack - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    const uint32_t valFullBits = val; // uint32_t(val) & ((1u<<bitsPerIndex_)-1);
    uint32_t lowAnd, lowOr = valFullBits<<lowOffs;
    if(iLowVect == iHighVect) {
      lowAnd = ~(((1u<<bitsPerPack)-1)<<lowOffs);
    } else {
      lowAnd = (1u<<lowOffs)-1;
      const uint32_t highOffs = (iHighBit+1) % cBufElmBits;
      buffer[iHighVect] = (buffer[iHighVect] & ~((1u<<highOffs)-1)) | (valFullBits>>(bitsPerPack-highOffs));
    }
    buffer[iLowVect] = (buffer[iLowVect] & lowAnd) | lowOr;
    assert( GetPack(at, buffer, bitsPerPack) == val );
  }

  __host__ __device__ void SetPack(const VciGpu at, const uint32_t val) {
    SetPack(at, val, buffer_, bitsPerPack_);
  }

  static __host__ __device__ VciGpu CalcBufBytes(VciGpu& nBuckets, const VciGpu bitsPerPack) {
    nBuckets = AlignUp(nBuckets, cVectBits);
    return (nBuckets * bitsPerPack) / 8;
  }

  __device__ GpuUnordSet(const VciGpu capacity, const VciGpu maxVal) {
    if(capacity == 0) {
      return;
    }
    bitsPerPack_ = VciGpu(ceilf(__log2f( maxVal+1 )));
    nBuckets_ = ceilf(capacity / cStartOccupancy);
    const VciGpu nBufBytes = CalcBufBytes(nBuckets_, bitsPerPack_);
    buffer_ = static_cast<uint32_t*>( malloc( nBufBytes ) );
    assert(buffer_ != nullptr);
    VectSetZero(buffer_, nBufBytes);
  }

  __host__ __device__ GpuUnordSet(GpuUnordSet&& src) {
    bitsPerPack_ = src.bitsPerPack_;
    buffer_ = src.buffer_;
    hash_ = src.hash_;
    count_ = src.count_;
    nBuckets_ = src.nBuckets_;

    src.bitsPerPack_ = 0;
    src.buffer_ = nullptr;
    src.hash_ = 0;
    src.count_ = 0;
    src.nBuckets_ = 0;
  }

  __host__ __device__ GpuUnordSet& operator=(GpuUnordSet&& src) {
    if(&src != this) [[likely]] {
      free(buffer_);

      bitsPerPack_ = src.bitsPerPack_;
      buffer_ = src.buffer_;
      hash_ = src.hash_;
      count_ = src.count_;
      nBuckets_ = src.nBuckets_;

      src.bitsPerPack_ = 0;
      src.buffer_ = nullptr;
      src.hash_ = 0;
      src.count_ = 0;
      src.nBuckets_ = 0;
    }
    return *this;
  }

  // Return true if it grew, false if not.
  __host__ __device__  bool CheckGrow() {
    if(count_ <= nBuckets_ * cGrowOccupancy) [[likely]] {
      return false;
    }

    VciGpu newNBuckets = ceilf(count_ / cStartOccupancy);
    const VciGpu newBufBytes = CalcBufBytes(newNBuckets, bitsPerPack_);
    uint32_t* newBuf = static_cast<uint32_t*>( malloc( newBufBytes ) );
    assert(newBuf != nullptr);
    VectSetZero(newBuf, newBufBytes);

    if(buffer_ != nullptr) {
      Visit([&](const VciGpu item) {
        assert(item != 0);
        uint32_t pos=(item*cHashMul) % newNBuckets;
        for(;;) {
          const VciGpu valAt = GetPack(pos, newBuf, bitsPerPack_);
          if(valAt == 0) {
            SetPack(pos, item, newBuf, bitsPerPack_);
            break;
          }
          assert(valAt != item);
          pos = (pos+1) % newNBuckets;
        }
      });
      free(buffer_);
    }
    buffer_ = newBuf;
    nBuckets_ = newNBuckets;
    return true;
  }

  // Returns true if item is added, false if already exists
  __host__ __device__ bool Add(const VciGpu item) {
    assert(1 <= item && item < (VciGpu(1) << bitsPerPack_));
    uint32_t pos=(item*cHashMul) % nBuckets_;
    for(;;) {
      const VciGpu valAt = GetPack(pos);
      if(valAt == 0) {
        SetPack(pos, item);
        hash_ ^= Hasher(item).hash_;
        count_++;
        CheckGrow();
        return true;
      }
      if(valAt == item) {
        return false;
      }
      pos = (pos+1) % nBuckets_;
    }
  }

  __host__ __device__ void ShiftAfterRemove(VciGpu pos) {
    assert(GetPack(pos) == 0);
    VciGpu rangeStart = pos;
    for(;;) {
      const uint32_t nextPos = (pos+1) % nBuckets_;
      const VciGpu valNext = GetPack(nextPos);
      if(valNext == 0) {
        break;
      }
      const uint32_t hashNext = (valNext * cHashMul) % nBuckets_;
      if(hashNext == rangeStart) {
        SetPack(rangeStart, valNext);
        SetPack(nextPos, 0);
        rangeStart = nextPos;
      }
      pos = nextPos;
    }
  }

  // Returns true if the item has been added to the trie, false if removed.
  __host__ __device__ bool Flip(const VciGpu item) {
    assert(1 <= item && item < (VciGpu(1) << bitsPerPack_));
    uint32_t pos=(item*cHashMul) % nBuckets_;
    for(;;) {
      const VciGpu valAt = GetPack(pos);
      if(valAt == 0 || valAt == item) {
        hash_ ^= Hasher(item).hash_;
        SetPack(pos, item ^ valAt);
        count_ += (valAt ? -1 : 1);
        if(valAt == 0) {
          CheckGrow();
        } else {
          ShiftAfterRemove(pos);
        }
        return !valAt;
      }
      pos = (pos+1) % nBuckets_;
    }
  }

  // Returns true if the item existed in the trie, false if it didn't exist.
  __host__ __device__ bool Remove(const VciGpu item) {
    assert(1 <= item && item < (VciGpu(1) << bitsPerPack_));
    uint32_t pos=(item*cHashMul) % nBuckets_;
    for(;;) {
      const VciGpu valAt = GetPack(pos);
      if(valAt == 0) {
        return false;
      }
      if(valAt == item) {
        hash_ ^= Hasher(item).hash_;
        count_--;
        SetPack(pos, 0);
        ShiftAfterRemove(pos);
        return true;
      }
      pos = (pos+1) % nBuckets_;
    }
  }

  template<typename F> __host__ __device__  void Visit(const F& f) const {
    if(count_ == 0) {
      return;
    }
    for(VciGpu i=0; i<nBuckets_; i++) {
      const VciGpu valAt = GetPack(i);
      if(valAt != 0) {
        f(valAt);
      }
    }
  }

  __device__ void Shrink() {
    if(count_ == 0) [[unlikely]] {
      free(buffer_);
      buffer_ = nullptr;
      nBuckets_ = 0;
      return;
    }
    if(count_ >= nBuckets_ * cShrinkOccupancy) {
      return;
    }
    GpuUnordSet t(count_, (VciGpu(1)<<bitsPerPack_) - 1);
    assert(t.bitsPerPack_ == bitsPerPack_);
    Visit([&](const VciGpu item) {
      t.Add(item);
    });
    assert(t.count_ == count_);
    assert(t.hash_ == hash_);
    *this = std::move(t);
  }

  __device__ ~GpuUnordSet() {
    count_ = 0;
    Shrink();
  }
};

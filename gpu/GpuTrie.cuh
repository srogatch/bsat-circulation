#pragma once

#include "GpuUtils.cuh"

struct GpuTrie {
  __uint128_t hash_ = 0;
  uint32_t* buffer_ = nullptr;
  uint32_t* nodeHasNum_ = nullptr;
  VciGpu nNodes_ = 0;
  VciGpu count_ = 0;
  int16_t bitsPerIndex_ = 0;

  static constexpr const int16_t cBufElmBits = 8 * sizeof(buffer_[0]);
  static constexpr const int16_t cVectBits = sizeof(__uint128_t) * 8;

  GpuTrie() = default;

  static __host__ __device__ uint32_t GetIndex(const VciGpu at, const uint32_t* buffer, const int16_t bitsPerIndex) {
    const uint32_t iLowBit = at * bitsPerIndex;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerIndex - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    uint32_t val;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    if(iLowVect == iHighVect) {
      val = (buffer[iLowVect] >> lowOffs) & ((1u<<bitsPerIndex)-1);
    } else {
      const uint32_t bitsInLow = cBufElmBits-lowOffs;
      const uint32_t bitsInHigh = bitsPerIndex-bitsInLow;
      const uint32_t lowMask = (1u<<bitsInLow)-1;
      const uint32_t highMask = (1u<<bitsInHigh)-1;
      val = ((buffer[iLowVect]>>lowOffs) & lowMask) | ((buffer[iHighVect] & highMask)<<bitsInLow);
    }
    return val;
  }

  __host__ __device__ uint32_t GetIndex(const VciGpu at) const {
    return GetIndex(at, buffer_, bitsPerIndex_);
  }

  static __host__ __device__ void SetIndex(const VciGpu at, const uint32_t val, uint32_t* buffer, const int16_t bitsPerIndex) {
    const uint32_t iLowBit = at * bitsPerIndex;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerIndex - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    const uint32_t valFullBits = val; // uint32_t(val) & ((1u<<bitsPerIndex_)-1);
    uint32_t lowAnd, lowOr = valFullBits<<lowOffs;
    if(iLowVect == iHighVect) {
      lowAnd = ~(((1u<<bitsPerIndex)-1)<<lowOffs);
    } else {
      lowAnd = (1u<<lowOffs)-1;
      const uint32_t highOffs = (iHighBit+1) % cBufElmBits;
      buffer[iHighVect] = (buffer[iHighVect] & ~((1u<<highOffs)-1)) | (valFullBits>>(bitsPerIndex-highOffs));
    }
    buffer[iLowVect] = (buffer[iLowVect] & lowAnd) | lowOr;
    assert( GetIndex(at, buffer, bitsPerIndex) == val );
  }

  __host__ __device__ void SetIndex(const VciGpu at, const uint32_t val) {
    SetIndex(at, val, buffer_, bitsPerIndex_);
  }

  // bitChild is 0 for left child and 1 for right child
  __host__ __device__ VciGpu GetChild(const VciGpu iParent, const int8_t bitChild) const {
    return GetIndex(iParent * 2 + bitChild);
  }

  __host__ __device__ VciGpu GetChild(const VciGpu iParent, const VciGpu item, const int16_t iBit) const {
    return GetChild(iParent, (item>>iBit) & 1);
  }

  static __host__ __device__ VciGpu CalcBufDwords(const VciGpu bitsPerIndex) {
    return DivUp( bitsPerIndex * VciGpu(2) * (VciGpu(1) << bitsPerIndex), cBufElmBits );
  }

  static __host__ __device__ VciGpu CalcNhnDwords(const VciGpu bitsPerIndex) {
    return DivUp(VciGpu(1) << bitsPerIndex, cBufElmBits);
  }

  static __host__ __device__ VciGpu CalcBufBytes(const VciGpu bitsPerIndex) {
    return DivUp( bitsPerIndex * VciGpu(2) * (VciGpu(1) << bitsPerIndex), cVectBits ) * sizeof(__uint128_t);
  }

  static __host__ __device__ VciGpu CalcNhnBytes(const VciGpu bitsPerIndex) {
    return DivUp(VciGpu(1) << bitsPerIndex, cVectBits) * sizeof(__uint128_t);
  }

  __device__ GpuTrie(const VciGpu capacity) {
    bitsPerIndex_ = max( VciGpu(ceilf(__logf( capacity ))), 7 );
    buffer_ = static_cast<uint32_t*>( malloc( CalcBufBytes(bitsPerIndex_) ) );
    nodeHasNum_ = static_cast<uint32_t*>( malloc( CalcNhnBytes(bitsPerIndex_) ) );
  }

  __host__ __device__ GpuTrie(GpuTrie&& src) {
    bitsPerIndex_ = src.bitsPerIndex_;
    buffer_ = src.buffer_;
    nodeHasNum_ = src.nodeHasNum_;
    hash_ = src.hash_;
    count_ = src.count_;
    nNodes_ = src.nNodes_;

    src.bitsPerIndex_ = 0;
    src.buffer_ = nullptr;
    src.nodeHasNum_ = nullptr;
    src.hash_ = 0;
    src.count_ = 0;
    src.nNodes_ = 0;
  }

  __host__ __device__ GpuTrie& operator=(GpuTrie&& src) {
    if(&src != this) {
      free(buffer_);
      free(nodeHasNum_);

      bitsPerIndex_ = src.bitsPerIndex_;
      buffer_ = src.buffer_;
      nodeHasNum_ = src.nodeHasNum_;
      hash_ = src.hash_;
      count_ = src.count_;
      nNodes_ = src.nNodes_;

      src.bitsPerIndex_ = 0;
      src.buffer_ = nullptr;
      src.nodeHasNum_ = nullptr;
      src.hash_ = 0;
      src.count_ = 0;
      src.nNodes_ = 0;
    }
    return *this;
  }

  // Return true if it grew, false if not.
  __host__ __device__  bool CheckGrow(const VciGpu iNode) {
    if( 2*iNode+1 < VciGpu(1)<<bitsPerIndex_ ) {
      return false;
    }

    const VciGpu newBpi = max(bitsPerIndex_ + 1, 7);

    const VciGpu newBufBytes = CalcBufBytes( newBpi );
    uint32_t *newBuf = static_cast<uint32_t*>( malloc( newBufBytes ) );
    assert( newBuf != nullptr );
    if(buffer_ != nullptr) {
      for(VciGpu i=0; i<nNodes_*2; i++) {
        SetIndex(i, GetIndex(i), newBuf, newBpi);
      }
      free(buffer_);
    }
    buffer_ = newBuf;

    const VciGpu newNhnBytes = CalcNhnBytes( newBpi );
    uint32_t *newNhn = static_cast<uint32_t*>( malloc( newNhnBytes ) );
    assert(newNhn != nullptr);
    const VciGpu oldNhnBytes = (nodeHasNum_ == nullptr ? 0 : CalcNhnBytes(bitsPerIndex_));
    if(nodeHasNum_ != nullptr) {
      VectCopy( newNhn, nodeHasNum_, oldNhnBytes );
      free(nodeHasNum_);
    }
    VectSetZero( newNhn+oldNhnBytes, newNhnBytes - oldNhnBytes );
    nodeHasNum_ = newNhn;

    bitsPerIndex_ = newBpi;
    return true;
  }

  __host__ __device__ VciGpu Traverse(const VciGpu item) {
    assert(item > 0);
    VciGpu iNode = 0;
    for(int16_t iBit=0; (VciGpu(1)<<iBit) <= item; iBit++) {
      assert(0 <= iNode && iNode <= nNodes_);
      if(iNode >= nNodes_) {
        CheckGrow(iNode);
        assert( iNode < (VciGpu(1)<<bitsPerIndex_) );
        SetIndex(2*iNode + ((item>>iBit) & 1), iNode+1);
        SetIndex(2*iNode + (((item>>iBit) & 1) ^ 1), 0);
        iNode = iNode+1;
        nNodes_++;
      } else {
        const VciGpu iChild = GetChild( iNode, item, iBit );
        if(iChild == 0) {
          SetIndex(2*iNode + ((item>>iBit) & 1), nNodes_);
          iNode = nNodes_;
        } else {
          iNode = iChild;
        }
      }
    }
    if(iNode >= nNodes_) {
      assert(0 <= iNode && iNode <= nNodes_);
      CheckGrow(iNode);
      SetIndex(2*iNode + 0, 0);
      SetIndex(2*iNode + 1, 0);
      nNodes_++;
    }
    return iNode;
  }

  // Returns true if item is added, false if already exists
  __host__ __device__ bool Add(const VciGpu item) {
    const VciGpu iNode = Traverse(item);
    if( !(nodeHasNum_[iNode / 32] & (1u<<(iNode&31))) ) {
      hash_ ^= Hasher(item).hash_;
      nodeHasNum_[iNode / 32] |= 1u<<(iNode&31);
      count_++;
      return true;
    }
    return false;
  }

  // Returns true if the item has been added to the trie, false if removed.
  __host__ __device__ bool Flip(const VciGpu item) {
    const VciGpu iNode = Traverse(item);
    hash_ ^= Hasher(item).hash_;
    nodeHasNum_[iNode / 32] ^= 1u<<(iNode&31);
    if( (nodeHasNum_[iNode / 32] & (1u<<(iNode&31))) ) {
      count_++;
      return true;
    } else {
      count_--;
      return false;
    }
  }

  // Returns true if the item existed in the trie, false if it didn't exist.
  __host__ __device__ bool Remove(const VciGpu item) {
    assert(item > 0);
    VciGpu iNode = 0;
    for(int16_t iBit=0; (VciGpu(1)<<iBit) <= item; iBit++) {
      if(iNode >= nNodes_) {
        return false;
      }
      const VciGpu iChild = GetChild( iNode, item, iBit );
      if(iChild == 0) {
        return false;
      }
      iNode = iChild;
    }
    assert(iNode < nNodes_);
    if( nodeHasNum_[iNode / 32] & (1u<<(iNode&31)) ) {
      hash_ ^= Hasher(item).hash_;
      nodeHasNum_[iNode / 32] ^= 1u<<(iNode&31);
      count_--;
      return true;
    }
    return false;
  }

  template<typename F> __host__ __device__  void Visit(const F& f) const {
    if(count_ == 0) {
      return;
    }
    constexpr const int16_t cMaxBits = sizeof(VciGpu) * 8;
    VciGpu stInds[cMaxBits];
    int16_t iBit;
    stInds[0] = 0;
    for(iBit=1; iBit<cMaxBits; iBit++) {
      const VciGpu iNode = stInds[iBit-1];
      const VciGpu leftChild = GetChild(iNode, int16_t(0));
      if(leftChild == 0) {
        break;
      }
      stInds[iBit] = leftChild;
    }
    VciGpu path = 0;
    while(iBit > 0) {
      iBit--;
      const VciGpu iNode = stInds[iBit];
      if(iNode < 0) {
        assert(path & (VciGpu(1) << iBit));
        path ^= VciGpu(1) << iBit;
        continue;
      }
      if( nodeHasNum_[iNode/32] & (1u<<(iNode&31)) ) {
        f(path);
      }
      const VciGpu iChild = GetChild(iNode, int16_t(1));
      if( iChild == 0 ) {
        continue;
      }
      stInds[iBit] = -iNode;
      path |= VciGpu(1) << iBit;
      iBit++;
      stInds[iBit] = iChild;
      for(;;) {
        iBit++;
        const VciGpu iNode = stInds[iBit-1];
        const VciGpu leftChild = GetChild(iNode, int16_t(0));
        if(leftChild == 0) {
          break;
        }
        stInds[iBit] = leftChild;
      }
    }
  }

  __device__ void Shrink() {
    if(count_ == 0) {
      free(buffer_);
      buffer_ = nullptr;
      free(nodeHasNum_);
      nodeHasNum_ = nullptr;
      return;
    }
    GpuTrie t(count_);
    Visit([&](const VciGpu item) {
      t.Add(item);
    });
    assert(t.count_ == count_);
    *this = std::move(t);
  }

  __device__ ~GpuTrie() {
    count_ = 0;
    Shrink();
  }
};

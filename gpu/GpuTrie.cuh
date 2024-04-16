#pragma once

#include "GpuUtils.cuh"

struct GpuTrie {
  uint32_t* buffer_ = nullptr;
  uint32_t* nodeHasNum_ = nullptr;
  VciGpu nNodes_ = 0;
  int16_t bitsPerIndex_ = 0;

  static constexpr const int16_t cBufElmBits = 8 * sizeof(buffer_[0]);
  static constexpr const int16_t cVectBits = sizeof(__uint128_t) * 8;

  GpuTrie() = default;

  __host__ __device__ uint32_t GetIndex(const VciGpu at) const {
    const uint32_t iLowBit = at * bitsPerIndex_;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerIndex_ - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    uint32_t val;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    if(iLowVect == iHighVect) {
      val = (buffer_[iLowVect] >> lowOffs) & ((1u<<bitsPerIndex_)-1);
    } else {
      const uint32_t bitsInLow = cBufElmBits-lowOffs;
      const uint32_t bitsInHigh = bitsPerIndex_-bitsInLow;
      const uint32_t lowMask = (1u<<bitsInLow)-1;
      const uint32_t highMask = (1u<<bitsInHigh)-1;
      val = ((buffer_[iLowVect]>>lowOffs) & lowMask) | ((buffer_[iHighVect] & highMask)<<bitsInLow);
    }
    return val;
  }

  __host__ __device__ void SetIndex(const VciGpu at, const uint32_t val) {
    const uint32_t iLowBit = at * bitsPerIndex_;
    const uint32_t iLowVect = iLowBit / cBufElmBits;
    const uint32_t iHighBit = iLowBit + bitsPerIndex_ - 1;
    const uint32_t iHighVect = iHighBit / cBufElmBits;
    const uint32_t lowOffs = iLowBit % cBufElmBits;
    const uint32_t valFullBits = val; // uint32_t(val) & ((1u<<bitsPerIndex_)-1);
    uint32_t lowAnd, lowOr = valFullBits<<lowOffs;
    if(iLowVect == iHighVect) {
      lowAnd = ~(((1u<<bitsPerIndex_)-1)<<lowOffs);
    } else {
      lowAnd = (1u<<lowOffs)-1;
      const uint32_t highOffs = (iHighBit+1) % cBufElmBits;
      buffer_[iHighVect] = (buffer_[iHighVect] & ~((1u<<highOffs)-1)) | (valFullBits>>(bitsPerIndex_-highOffs));
    }
    buffer_[iLowVect] = (buffer_[iLowVect] & lowAnd) | lowOr;
  }

  // bitChild is 0 for left child and 1 for right child
  __host__ __device__ VciGpu GetChild(const VciGpu iParent, const int8_t bitChild) {
    return GetIndex(iParent * 2 + bitChild);
  }

  __host__ __device__ VciGpu GetChild(const VciGpu iParent, const VciGpu item, const int16_t iBit) {
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

  __host__ __device__ GpuTrie(const VciGpu capacity) {
    bitsPerIndex_ = __ceilf(__logf( capacity+1 ));
    buffer_ = static_cast<uint32_t*>( malloc( CalcBufBytes(bitsPerIndex_) ) );
    nodeHasNum_ = static_cast<uint32_t*>( malloc( CalcNhnBytes(bitsPerIndex_) ) );
  }

  // Return true if it grew, false if not.
  __host__ __device__  bool CheckGrow(const VciGpu iNode) {
    if( 2*iNode+1 < VciGpu(1)<<bitsPerIndex_ ) {
      return false;
    }

    const VciGpu newBpi = min(bitsPerIndex_ + 1, 7);

    const VciGpu newBufBytes = CalcBufBytes( newBpi );
    uint32_t *newBuf = static_cast<uint32_t*>( malloc( newBufBytes ) );
    VectCopy( newBuf, buffer_, CalcBufBytes(bitsPerIndex_) );
    free(buffer_);
    buffer_ = newBuf;

    const VciGpu newNhnBytes = CalcNhnBytes( newBpi );
    uint32_t *newNhn = static_cast<uint32_t*>( malloc( newNhnBytes ) );
    const VciGpu oldNhnBytes = CalcNhnBytes(bitsPerIndex_);
    VectCopy( newNhn, nodeHasNum_, oldNhnBytes );
    free(nodeHasNum_);
    VectSetZero( newNhn+oldNhnBytes, newNhnBytes - oldNhnBytes );
    nodeHasNum_ = newNhn;
    
    return true;
  }

  bool Add(const VciGpu item) {
    VciGpu iNode = 0;
    for(int16_t iBit=0; (VciGpu(1)<<iBit) <= item; iBit++) {
      if(iNode >= nNodes_) {
        CheckGrow(iNode);
        
      } else {
        iNode = GetChild( iNode, item, iBit );
      }
    }
  }
};

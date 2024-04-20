#pragma once

#include "GpuUtils.cuh"
#include "../CNF.h"

using GpuPerSignHead = VciGpu[2];

class GpuLinkage {
  friend class HostLinkage;

  // -(nClauses_)..(nClauses_+1): the edge items are stubs for convenient arc counting
  GpuPerSignHead *headsClause2Var_;
  // -(nVars_)..(nVars_+1): the edge items are stubs for convenient arc counting
  GpuPerSignHead *headsVar2Clause_;
  VciGpu *targetsClause2Var_;
  VciGpu *targetsVar2Clause_;
  VciGpu nVars_, nClauses_;

  static __host__ __device__ int8_t SignToHead(const int8_t sign) {
    return (sign + 1) >> 1;
  }

  __device__ VciGpu ArcCount(const VciGpu from, const int8_t sign, const VciGpu nItems, GpuPerSignHead *heads, VciGpu *targets) const {
    assert(from != 0);
    const int8_t hSign = SignToHead(sign);
    const VciGpu *pHead = &heads[from + nItems][hSign];
    return *(pHead+1) - *pHead;
  }

  __device__ VciGpu ArcCount(const VciGpu from, const VciGpu nItems, GpuPerSignHead *heads, VciGpu *targets) const {
    return ArcCount(from, -1, nItems, heads, targets) + ArcCount(from, +1, nItems, heads, targets);
  }

  __device__ VciGpu GetTarget(
    const VciGpu from, const int8_t sign, const VciGpu ordinal, const VciGpu nItems, GpuPerSignHead *heads, VciGpu *targets) const
  {
    return targets[ordinal + heads[from + nItems][SignToHead(sign)]];
  }

public:
  __device__ VciGpu ClauseArcCount(const VciGpu fromClause) const {
    return ArcCount(fromClause, nClauses_, headsClause2Var_, targetsClause2Var_);
  }
  __device__ VciGpu ClauseArcCount(const VciGpu fromClause, const int8_t sign) const {
    return ArcCount(fromClause, sign, nClauses_, headsClause2Var_, targetsClause2Var_);
  }
  __device__ VciGpu ClauseGetTarget(const VciGpu fromClause, const int8_t sign, const VciGpu ordinal) const {
    return GetTarget(fromClause, sign, ordinal, nClauses_, headsClause2Var_, targetsClause2Var_);
  }

  __device__ VciGpu VarArcCount(const VciGpu fromVar) const {
    return ArcCount(fromVar, nVars_, headsVar2Clause_, targetsVar2Clause_);
  }
  __device__ VciGpu VarArcCount(const VciGpu fromVar, const int8_t sign) const {
    return ArcCount(fromVar, sign, nVars_, headsVar2Clause_, targetsVar2Clause_);
  }
  __device__ VciGpu VarGetTarget(const VciGpu fromVar, const int8_t sign, const VciGpu ordinal) const {
    return GetTarget(fromVar, sign, ordinal, nVars_, headsVar2Clause_, targetsVar2Clause_);
  }
  __device__ VciGpu GetVarCount() const { return nVars_; }
  __device__ VciGpu GetClauseCount() const { return nClauses_; }
};

struct HostLinkage {
    // -(nClauses_)..(nClauses_+1): the edge items are stubs for convenient arc counting
  CudaArray<GpuPerSignHead> headsClause2Var_;
  // -(nVars_)..(nVars_+1): the edge items are stubs for convenient arc counting
  CudaArray<GpuPerSignHead> headsVar2Clause_;
  CudaArray<VciGpu> targetsClause2Var_;
  CudaArray<VciGpu> targetsVar2Clause_;
  VciGpu nVars_, nClauses_;

  const Formula *pFormula_;
  const CudaAttributes* pCa_;

  HostLinkage() = default;

  bool Marshal(GpuLinkage& gl);


  static void Init(
    const Formula& formula, const std::vector<CudaAttributes>& cas, std::vector<HostLinkage>& linkages);
};
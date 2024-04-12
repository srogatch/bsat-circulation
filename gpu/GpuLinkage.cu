#include "GpuLinkage.cuh"

void HostLinkage::Init(const Formula& formula, const CudaAttributes& ca) {
  pFormula_ = &formula;
  pCa_ = &ca;
  nVars_ = pFormula_->nVars_;
  nClauses_ = pFormula_->nClauses_;
  headsClause2Var_ = CudaArray<GpuPerSignHead>(nClauses_*2 + 2, CudaArrayType::Device);
  headsVar2Clause_ = CudaArray<GpuPerSignHead>(nVars_*2 + 2, CudaArrayType::Device);
  std::unique_ptr<GpuPerSignHead[]> h_headsClause2Var = std::make_unique<GpuPerSignHead[]>(nClauses_ * 2 + 2);
  std::unique_ptr<GpuPerSignHead[]> h_headsVar2Clause = std::make_unique<GpuPerSignHead[]>(nVars_ * 2 + 2);

  VciGpu total;

  // Populate heads for clauses
  total = 0;
  for(VciGpu i=-pFormula_->nClauses_; i<=pFormula_->nClauses_; i++) {
    if(i == 0) {
      h_headsClause2Var[i+pFormula_->nClauses_][GpuLinkage::SignToHead(-1)] = total;  
      h_headsClause2Var[i+pFormula_->nClauses_][GpuLinkage::SignToHead(+1)] = total;
      continue;
    }
    h_headsClause2Var[i+pFormula_->nClauses_][GpuLinkage::SignToHead(-1)] = total;
    total += pFormula_->clause2var_.ArcCount(i, -1);
    h_headsClause2Var[i+pFormula_->nClauses_][GpuLinkage::SignToHead(+1)] = total;
    total += pFormula_->clause2var_.ArcCount(i, +1);
  }
  h_headsClause2Var[2*pFormula_->nClauses_+1][GpuLinkage::SignToHead(-1)] = total;  
  h_headsClause2Var[2*pFormula_->nClauses_+1][GpuLinkage::SignToHead(+1)] = total;
  gpuErrchk(cudaMemcpyAsync(
    headsClause2Var_.Get(), h_headsClause2Var.get(), sizeof(GpuPerSignHead) * size_t(2*nClauses_+2),
    cudaMemcpyHostToDevice, pCa_->cs_
  ));
  targetsClause2Var_ = CudaArray<VciGpu>(total, CudaArrayType::Device);
  std::unique_ptr<VciGpu[]> h_targetsClause2Var = std::make_unique<VciGpu[]>(total);

  // Populate targets for clauses
  total = 0;
  for(VciGpu i=-pFormula_->nClauses_; i<=pFormula_->nClauses_; i++) {
    if(i == 0) {
      continue;
    }
    for(int8_t sign=-1; sign<=1; sign+=2) {
      const VciGpu enJ = pFormula_->clause2var_.ArcCount(i, sign);
      for(VciGpu j=0; j<enJ; j++) {
        h_targetsClause2Var[total] = pFormula_->clause2var_.GetTarget(i, sign, j);
        total++;
      }
    }
  }
  gpuErrchk(cudaMemcpyAsync(
    targetsClause2Var_.Get(), h_targetsClause2Var.get(), sizeof(VciGpu) * total,
    cudaMemcpyHostToDevice, pCa_->cs_
  ));

  // Populate heads for vars
  total = 0;
  for(VciGpu i=-pFormula_->nVars_; i<=pFormula_->nVars_; i++) {
    if(i == 0) {
      h_headsVar2Clause[i+pFormula_->nVars_][GpuLinkage::SignToHead(-1)] = total;  
      h_headsVar2Clause[i+pFormula_->nVars_][GpuLinkage::SignToHead(+1)] = total;
      continue;
    }
    h_headsVar2Clause[i+pFormula_->nVars_][GpuLinkage::SignToHead(-1)] = total;
    total += pFormula_->var2clause_.ArcCount(i, -1);
    h_headsVar2Clause[i+pFormula_->nVars_][GpuLinkage::SignToHead(+1)] = total;
    total += pFormula_->var2clause_.ArcCount(i, +1);
  }
  h_headsVar2Clause[2*pFormula_->nVars_+1][GpuLinkage::SignToHead(-1)] = total;  
  h_headsVar2Clause[2*pFormula_->nVars_+1][GpuLinkage::SignToHead(+1)] = total;
  gpuErrchk(cudaMemcpyAsync(
    headsVar2Clause_.Get(), h_headsVar2Clause.get(), sizeof(GpuPerSignHead) * size_t(nVars_ * 2 + 2),
    cudaMemcpyHostToDevice, pCa_->cs_
  ));
  targetsVar2Clause_ = CudaArray<VciGpu>(total, CudaArrayType::Device);
  std::unique_ptr<VciGpu[]> h_targetsVar2Clause = std::make_unique<VciGpu[]>(total);

  // Populate targets for vars
  total = 0;
  for(VciGpu i=-pFormula_->nVars_; i<=pFormula_->nVars_; i++) {
    if(i == 0) {
      continue;
    }
    for(int8_t sign=-1; sign<=1; sign+=2) {
      const VciGpu enJ = pFormula_->var2clause_.ArcCount(i, sign);
      for(VciGpu j=0; j<enJ; j++) {
        h_targetsVar2Clause[total] = pFormula_->var2clause_.GetTarget(i, sign, j);
        total++;
      }
    }
  }
  gpuErrchk(cudaMemcpyAsync(
    targetsVar2Clause_.Get(), h_targetsVar2Clause.get(), sizeof(VciGpu) * total,
    cudaMemcpyHostToDevice, pCa_->cs_
  ));

  // We must synchronize here because the host memory on the stack of this function will be released upon function's exit,
  // while the stream is still copying this host memory into GPU.
  gpuErrchk((cudaStreamSynchronize(pCa_->cs_)));
}

bool HostLinkage::Marshal(GpuLinkage& gl) {
  gl.headsClause2Var_ = headsClause2Var_.Get();
  gl.headsVar2Clause_ = headsVar2Clause_.Get();
  gl.nClauses_ = nClauses_;
  gl.nVars_ = nVars_;
  gl.targetsClause2Var_ = targetsClause2Var_.Get();
  gl.targetsVar2Clause_ = targetsVar2Clause_.Get();
  return true;
}

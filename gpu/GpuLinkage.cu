#include "GpuLinkage.cuh"

void HostLinkage::Init(const Formula& formula, const std::vector<CudaAttributes>& cas,
  std::vector<HostLinkage>& linkages)
{
  const int nGpus = cas.size();
  linkages.resize(nGpus);
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    linkages[i].pFormula_ = &formula;
    linkages[i].pCa_ = &cas[i];
    linkages[i].nVars_ = formula.nVars_;
    linkages[i].nClauses_ = formula.nClauses_;
    linkages[i].headsClause2Var_ = CudaArray<GpuPerSignHead>(formula.nClauses_*2 + 2, CudaArrayType::Device);
    linkages[i].headsVar2Clause_ = CudaArray<GpuPerSignHead>(formula.nVars_*2 + 2, CudaArrayType::Device);
  }
  std::unique_ptr<GpuPerSignHead[]> h_headsClause2Var = std::make_unique<GpuPerSignHead[]>(formula.nClauses_ * 2 + 2);
  std::unique_ptr<GpuPerSignHead[]> h_headsVar2Clause = std::make_unique<GpuPerSignHead[]>(formula.nVars_ * 2 + 2);

  VciGpu total;

  // Populate heads for clauses
  total = 0;
  for(VciGpu i=-formula.nClauses_; i<=formula.nClauses_; i++) {
    if(i == 0) {
      h_headsClause2Var[i+formula.nClauses_][GpuLinkage::SignToHead(-1)] = total;  
      h_headsClause2Var[i+formula.nClauses_][GpuLinkage::SignToHead(+1)] = total;
      continue;
    }
    h_headsClause2Var[i+formula.nClauses_][GpuLinkage::SignToHead(-1)] = total;
    total += formula.clause2var_.ArcCount(i, -1);
    h_headsClause2Var[i+formula.nClauses_][GpuLinkage::SignToHead(+1)] = total;
    total += formula.clause2var_.ArcCount(i, +1);
  }
  h_headsClause2Var[2*formula.nClauses_+1][GpuLinkage::SignToHead(-1)] = total;  
  h_headsClause2Var[2*formula.nClauses_+1][GpuLinkage::SignToHead(+1)] = total;
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyAsync(
      linkages[i].headsClause2Var_.Get(), h_headsClause2Var.get(), sizeof(GpuPerSignHead) * size_t(2*formula.nClauses_+2),
      cudaMemcpyHostToDevice, linkages[i].pCa_->cs_
    ));
    linkages[i].targetsClause2Var_ = CudaArray<VciGpu>(total, CudaArrayType::Device);
  }
  std::unique_ptr<VciGpu[]> h_targetsClause2Var = std::make_unique<VciGpu[]>(total);

  // Populate targets for clauses
  total = 0;
  for(VciGpu i=-formula.nClauses_; i<=formula.nClauses_; i++) {
    if(i == 0) {
      continue;
    }
    for(int8_t sign=-1; sign<=1; sign+=2) {
      const VciGpu enJ = formula.clause2var_.ArcCount(i, sign);
      for(VciGpu j=0; j<enJ; j++) {
        h_targetsClause2Var[total] = formula.clause2var_.GetTarget(i, sign, j);
        total++;
      }
    }
  }
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyAsync(
      linkages[i].targetsClause2Var_.Get(), h_targetsClause2Var.get(), sizeof(VciGpu) * total,
      cudaMemcpyHostToDevice, linkages[i].pCa_->cs_
    ));
  }

  // Populate heads for vars
  total = 0;
  for(VciGpu i=-formula.nVars_; i<=formula.nVars_; i++) {
    if(i == 0) {
      h_headsVar2Clause[i+formula.nVars_][GpuLinkage::SignToHead(-1)] = total;  
      h_headsVar2Clause[i+formula.nVars_][GpuLinkage::SignToHead(+1)] = total;
      continue;
    }
    h_headsVar2Clause[i+formula.nVars_][GpuLinkage::SignToHead(-1)] = total;
    total += formula.var2clause_.ArcCount(i, -1);
    h_headsVar2Clause[i+formula.nVars_][GpuLinkage::SignToHead(+1)] = total;
    total += formula.var2clause_.ArcCount(i, +1);
  }
  h_headsVar2Clause[2*formula.nVars_+1][GpuLinkage::SignToHead(-1)] = total;  
  h_headsVar2Clause[2*formula.nVars_+1][GpuLinkage::SignToHead(+1)] = total;
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyAsync(
      linkages[i].headsVar2Clause_.Get(), h_headsVar2Clause.get(), sizeof(GpuPerSignHead) * size_t(formula.nVars_ * 2 + 2),
      cudaMemcpyHostToDevice, linkages[i].pCa_->cs_
    ));
    linkages[i].targetsVar2Clause_ = CudaArray<VciGpu>(total, CudaArrayType::Device);
  }
  std::unique_ptr<VciGpu[]> h_targetsVar2Clause = std::make_unique<VciGpu[]>(total);

  // Populate targets for vars
  total = 0;
  for(VciGpu i=-formula.nVars_; i<=formula.nVars_; i++) {
    if(i == 0) {
      continue;
    }
    for(int8_t sign=-1; sign<=1; sign+=2) {
      const VciGpu enJ = formula.var2clause_.ArcCount(i, sign);
      for(VciGpu j=0; j<enJ; j++) {
        h_targetsVar2Clause[total] = formula.var2clause_.GetTarget(i, sign, j);
        total++;
      }
    }
  }
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyAsync(
      linkages[i].targetsVar2Clause_.Get(), h_targetsVar2Clause.get(), sizeof(VciGpu) * total,
      cudaMemcpyHostToDevice, linkages[i].pCa_->cs_
    ));
  }

  // We must synchronize here because the host memory on the stack of this function will be released upon function's exit,
  // while the stream is still copying this host memory into GPU.
  for(int i=0; i<nGpus; i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk((cudaStreamSynchronize(linkages[i].pCa_->cs_)));
  }
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

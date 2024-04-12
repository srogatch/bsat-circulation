#include "Common.h"

#include "GpuLinkage.cuh"

__constant__ __uint128_t *gpHashSeries;
std::unique_ptr<uint128[]> BitVector::hashSeries_ = nullptr;

void GpuCalcHashSeries(const VciGpu maxItem, const std::vector<CudaAttributes>& cas) {
  BitVector::hashSeries_ = std::make_unique<uint128[]>(maxItem + 1);
  BitVector::hashSeries_[0] = 1;
  for(VciGpu i=1; i<=maxItem; i++) {
    BitVector::hashSeries_[i] = BitVector::hashSeries_[i-1] * kHashBase;
  }
  for(int i=0; i<int(cas.size()); i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpyToSymbolAsync(gpHashSeries, BitVector::hashSeries_.get(), sizeof(__uint128_t)*(maxItem+1), 0,
      cudaMemcpyHostToDevice, cas[i].cs_));
  }
  for(int i=0; i<int(cas.size()); i++) {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaStreamSynchronize(cas[i].cs_));
  }
}

// This must be included after gpHashSeries is defined
#include "GpuBitVector.cuh"

int main(int argc, char* argv[]) {
  auto tmStart = std::chrono::steady_clock::now();
  const auto tmVeryStart = tmStart;

  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  
  // TODO: does it override the environment variable?
  omp_set_num_threads(nSysCpus);
  const uint64_t nOmpThreads = omp_get_max_threads();

  Formula formula;
  std::atomic<bool> provenUnsat = false;
  std::atomic<bool> maybeSat = formula.Load(argv[1]);
  if(!maybeSat) {
    provenUnsat = true;
    { // TODO: remove code duplication
      std::ofstream ofs(argv[2]);
      ofs << "s UNSATISFIABLE" << std::endl;
      // TODO: output the proof: proof.out, https://satcompetition.github.io/2024/output.html
    }
    return 0;
  }
  int64_t prevNUnsat = formula.nClauses_;

  std::cout << "Precomputing..." << std::endl;
  int nGpus = 0;
  gpuErrchk(cudaGetDeviceCount(&nGpus));
  std::vector<CudaAttributes> cas(nGpus);
  for(int i=0; i<nGpus; i++) {
    cas[i].Init(i);
  }
  GpuCalcHashSeries(std::max(formula.nVars_, formula.nClauses_), cas);



  return 0;
}

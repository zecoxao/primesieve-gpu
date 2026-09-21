///
/// @file  CudaBackend.cpp
/// @brief CUDA implementation of the GPU segmented sieve of Eratosthenes.
///
///        Uses the CUDA driver API plus NVRTC, both loaded at runtime, so
///        this backend builds with a plain C++ compiler and needs no CUDA
///        toolkit at build time. The kernel (src/gpu/cuda/sieve.cu) is
///        compiled for the device's own compute capability on first use.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "../GpuBackend.hpp"
#include "../GpuTables.hpp"
#include "cuda_loader.hpp"
#include "sieve_cu.hpp"

#include <primesieve.hpp>
#include <primesieve/primesieve_error.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace primesieve::gpu;

uint32_t isqrt64(uint64_t x)
{
  if (x == 0)
    return 0;
  uint64_t r = (uint64_t) std::sqrt((double) x);
  while (r > 0 && r > x / r)
    r--;
  while (r + 1 <= 0xffffffffull && (r + 1) <= x / (r + 1))
    r++;
  return (uint32_t) r;
}

int deviceAttr(const CudaApi* api, CUdevice dev, int attr, int fallback)
{
  int v = fallback;
  if (api->DeviceGetAttribute(&v, attr, dev) != CUDA_SUCCESS)
    return fallback;
  return v;
}

class CudaBackend : public GpuBackend
{
public:
  CudaBackend() {}
  ~CudaBackend();

  bool init(const DeviceInfo& info, const GpuConfig& config, std::string& error);

  const DeviceInfo& device() const override { return info_; }
  uint32_t sieveBytes() const override { return sieveBytes_; }
  uint32_t workGroupSize() const override { return blockSize_; }

  uint64_t count(uint64_t lo, uint64_t hi, int countType) override;

private:
  bool compileKernel(int ccMajor, int ccMinor, std::string& error);
  void releaseBuffers();
  void ensureSievingPrimes(uint32_t sqrtHi);
  void ensureKTable(int countType);
  CUdeviceptr upload(const void* host, std::size_t bytes, const char* what);
  uint64_t runLaunch(uint64_t segLowBase, uint32_t numSegments,
                     uint64_t lo, uint64_t hi, int countType);

  const CudaApi* api_ = nullptr;
  DeviceInfo  info_;
  CUdevice    dev_ = 0;
  CUcontext   ctx_ = nullptr;
  CUmodule    module_ = nullptr;
  CUfunction  kernel_ = nullptr;

  uint32_t sieveBytes_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t segmentsPerLaunch_ = 0;
  uint32_t maxSegmentsPerLaunch_ = 0;
  uint32_t sharedBytes_ = 0;
  bool     usePreSieve_ = true;

  PreSieveTables preSieve_;
  CUdeviceptr dPreSieve_ = 0;
  CUdeviceptr dPreLen_ = 0;
  CUdeviceptr dPreOff_ = 0;
  CUdeviceptr dRestore_ = 0;

  std::vector<uint32_t> primes_;
  std::vector<uint64_t> magics_;
  uint32_t primesUpTo_ = 0;
  uint32_t firstPrimeIdx_ = 0;
  uint32_t tier1End_ = 0;
  uint32_t tier2End_ = 0;
  CUdeviceptr dPrimes_ = 0;
  CUdeviceptr dMagics_ = 0;

  int         kTableType_ = -1;
  CUdeviceptr dKTable_ = 0;

  CUdeviceptr dCounts_ = 0;
  std::vector<uint64_t> hostCounts_;
};

CudaBackend::~CudaBackend()
{
  if (!api_)
    return;
  if (ctx_)
    api_->CtxSetCurrent(ctx_);
  releaseBuffers();
  if (module_) api_->ModuleUnload(module_);
  if (ctx_)    api_->CtxDestroy(ctx_);
}

void CudaBackend::releaseBuffers()
{
  CUdeviceptr* all[] = { &dPreSieve_, &dPreLen_, &dPreOff_, &dRestore_,
                         &dPrimes_, &dMagics_, &dKTable_, &dCounts_ };
  for (std::size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
    if (*all[i]) { api_->MemFree(*all[i]); *all[i] = 0; }
}

CUdeviceptr CudaBackend::upload(const void* host, std::size_t bytes, const char* what)
{
  // A zero sized allocation is not useful; keep one byte so the kernel
  // argument stays a valid pointer.
  std::size_t n = bytes ? bytes : 1;
  CUdeviceptr p = 0;
  CUresult err = api_->MemAlloc(&p, n);
  if (err != CUDA_SUCCESS)
    throw primesieve::primesieve_error(std::string("CUDA: failed to allocate ")
                                       + what + ": " + cudaErrorString(api_, err));
  if (host && bytes)
  {
    err = api_->MemcpyHtoD(p, host, bytes);
    if (err != CUDA_SUCCESS)
      throw primesieve::primesieve_error(std::string("CUDA: failed to upload ")
                                         + what + ": " + cudaErrorString(api_, err));
  }
  return p;
}

bool CudaBackend::compileKernel(int ccMajor, int ccMinor, std::string& error)
{
  // The wheel tables are emitted from GpuWheel.hpp so that the device code
  // and the host driver can never disagree about them.
  std::string source = wheelTableSource(/* openclConstant */ false);
  source += kSieveCuSource;

  nvrtcProgram prog = nullptr;
  if (api_->nvrtcCreateProgram(&prog, source.c_str(), "primesieve_sieve.cu",
                               0, nullptr, nullptr) != NVRTC_SUCCESS)
  {
    error = "CUDA: nvrtcCreateProgram failed";
    return false;
  }

  char arch[64];
  std::snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", ccMajor, ccMinor);
  const char* options[] = { arch, "--std=c++11" };

  CUnvrtcResult cres = api_->nvrtcCompileProgram(prog, 2, options);

  if (cres != NVRTC_SUCCESS)
  {
    std::size_t logSize = 0;
    api_->nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize + 1, 0);
    api_->nvrtcGetProgramLog(prog, log.data());
    error = std::string("CUDA: kernel compilation failed: ")
          + api_->nvrtcGetErrorString(cres) + "\n" + log.data();
    api_->nvrtcDestroyProgram(&prog);
    return false;
  }

  std::size_t ptxSize = 0;
  api_->nvrtcGetPTXSize(prog, &ptxSize);
  std::vector<char> ptx(ptxSize + 1, 0);
  api_->nvrtcGetPTX(prog, ptx.data());
  api_->nvrtcDestroyProgram(&prog);

  CUresult err = api_->ModuleLoadData(&module_, ptx.data());
  if (err != CUDA_SUCCESS)
  {
    error = "CUDA: cuModuleLoadData failed: " + cudaErrorString(api_, err);

    // By far the most common cause: an NVRTC newer than the installed
    // driver. NVRTC then emits a PTX ISA version the driver cannot JIT.
    int major = 0, minor = 0;
    if (api_->nvrtcVersion && api_->nvrtcVersion(&major, &minor) == NVRTC_SUCCESS)
    {
      char hint[256];
      std::snprintf(hint, sizeof hint,
                    "\nNVRTC is version %d.%d; it must not be newer than the "
                    "installed NVIDIA driver's CUDA version. Install a matching "
                    "NVRTC, or point PRIMESIEVE_NVRTC_PATH at one.",
                    major, minor);
      error += hint;
    }
    return false;
  }

  err = api_->ModuleGetFunction(&kernel_, module_, "ps_count");
  if (err != CUDA_SUCCESS)
  {
    error = "CUDA: cuModuleGetFunction failed: " + cudaErrorString(api_, err);
    return false;
  }
  return true;
}

bool CudaBackend::init(const DeviceInfo& info, const GpuConfig& config, std::string& error)
{
  api_ = loadCuda();
  if (!api_)
  {
    error = cudaLoadError();
    return false;
  }
  if (!api_->haveNvrtc)
  {
    error = cudaLoadError();
    return false;
  }

  info_ = info;

  CUresult err = api_->DeviceGet(&dev_, info.deviceIndex);
  if (err != CUDA_SUCCESS)
  {
    error = "CUDA: cuDeviceGet failed: " + cudaErrorString(api_, err);
    return false;
  }

  err = api_->CtxCreate(&ctx_, 0, dev_);
  if (err != CUDA_SUCCESS)
  {
    error = "CUDA: cuCtxCreate failed: " + cudaErrorString(api_, err);
    return false;
  }
  api_->CtxSetCurrent(ctx_);

  int ccMajor = deviceAttr(api_, dev_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 5);
  int ccMinor = deviceAttr(api_, dev_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

  if (!compileKernel(ccMajor, ccMinor, error))
    return false;

  // ---- pick the sieve geometry -----------------------------------------
  uint32_t block = config.workGroupSize ? config.workGroupSize : 256;
  block = std::min<uint32_t>(block, (uint32_t) info_.maxWorkGroupSize);
  // The in-block reduction halves the block each step.
  while (block & (block - 1))
    block &= block - 1;
  if (block == 0)
  {
    error = "CUDA: device reports an unusable maximum block size";
    return false;
  }
  blockSize_ = block;

  // Volta and newer can give a block more than the default 48 KiB of shared
  // memory, but only after opting in per kernel.
  uint64_t sharedLimit = info_.localMemBytes;
  int optin = deviceAttr(api_, dev_, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 0);
  if ((uint64_t) optin > sharedLimit)
    sharedLimit = (uint64_t) optin;

  // scratch + wheel tables + sieve, laid out as sieve.cu expects.
  const uint64_t tableBytes = (64 + 30 + 30) * sizeof(uint32_t);
  uint64_t fixed = (uint64_t) block * sizeof(uint64_t) + tableBytes;
  if (sharedLimit <= fixed + 256)
  {
    error = "CUDA: device has too little shared memory";
    return false;
  }

  uint32_t sieveBytes = config.sieveBytes ? config.sieveBytes : 16384;
  sieveBytes = (uint32_t) std::min<uint64_t>(sieveBytes, sharedLimit - fixed);

  // The kernel slices a segment across a whole block and across a single
  // warp, so the segment must divide evenly by both (and by 4).
  uint32_t granularity = std::max<uint32_t>(block, 32u);
  sieveBytes = (sieveBytes / granularity) * granularity;
  if (sieveBytes < granularity || sieveBytes < 256)
  {
    error = "CUDA: device has too little shared memory for a useful segment";
    return false;
  }
  sieveBytes_ = sieveBytes;
  sharedBytes_ = (uint32_t)(fixed + sieveBytes);

  if ((uint64_t) sharedBytes_ > 48u * 1024)
  {
    err = api_->FuncSetAttribute(kernel_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 (int) sharedBytes_);
    if (err != CUDA_SUCCESS)
    {
      error = "CUDA: could not raise the dynamic shared memory limit: "
            + cudaErrorString(api_, err);
      return false;
    }
  }

  segmentsPerLaunch_ = config.segmentsPerLaunch;
  if (segmentsPerLaunch_ == 0)
    segmentsPerLaunch_ = std::max<uint32_t>(info_.computeUnits, 1) * 256;
  segmentsPerLaunch_ = std::max<uint32_t>(segmentsPerLaunch_, 1);
  maxSegmentsPerLaunch_ = segmentsPerLaunch_;

  usePreSieve_ = config.preSieve;

  if (usePreSieve_)
  {
    preSieve_ = buildPreSieveTables(defaultPreSieveGroups());
    dPreSieve_ = upload(preSieve_.data.data(), preSieve_.data.size() * sizeof(uint32_t), "pre-sieve tables");
    dPreLen_   = upload(preSieve_.len.data(), preSieve_.len.size() * sizeof(uint32_t), "pre-sieve lengths");
    dPreOff_   = upload(preSieve_.off.data(), preSieve_.off.size() * sizeof(uint32_t), "pre-sieve offsets");
    dRestore_  = upload(preSieve_.restore.data(), preSieve_.restore.size(), "pre-sieve restore bytes");
  }
  else
  {
    preSieve_ = PreSieveTables();
    dPreSieve_ = upload(nullptr, 0, "pre-sieve tables");
    dPreLen_   = upload(nullptr, 0, "pre-sieve lengths");
    dPreOff_   = upload(nullptr, 0, "pre-sieve offsets");
    dRestore_  = upload(nullptr, 0, "pre-sieve restore bytes");
  }

  hostCounts_.resize(segmentsPerLaunch_);
  dCounts_ = upload(nullptr, hostCounts_.size() * sizeof(uint64_t), "count buffer");

  return true;
}

void CudaBackend::ensureSievingPrimes(uint32_t sqrtHi)
{
  if (dPrimes_ && sqrtHi <= primesUpTo_)
    return;

  primes_.clear();
  if (sqrtHi >= 7)
    primesieve::generate_primes(7, sqrtHi, &primes_);
  primesUpTo_ = sqrtHi;

  // Primes already handled by the pre-sieve patterns must not be crossed off
  // again: the pattern removed their multiples, and a second pass would also
  // clear the primes themselves.
  firstPrimeIdx_ = 0;
  if (usePreSieve_)
    while (firstPrimeIdx_ < primes_.size() && primes_[firstPrimeIdx_] <= preSieve_.maxPrime)
      firstPrimeIdx_++;

  // Split the sieving primes into the three groups the kernel uses; a prime
  // has about (sliceBytes * 8 / p) multiples in a slice, and slicing only
  // pays while that stays above the one extra setup per slice.
  {
    uint32_t tier1Limit = std::max<uint32_t>(2u * sieveBytes_ / blockSize_, 1u);
    uint32_t tier2Limit = std::max<uint32_t>(2u * sieveBytes_ / 32u, tier1Limit);

    tier1End_ = firstPrimeIdx_;
    while (tier1End_ < primes_.size() && primes_[tier1End_] < tier1Limit)
      tier1End_++;
    tier2End_ = tier1End_;
    while (tier2End_ < primes_.size() && primes_[tier2End_] < tier2Limit)
      tier2End_++;
  }

  magics_.resize(primes_.size());
  for (std::size_t i = 0; i < primes_.size(); i++)
    magics_[i] = divisionMagic(primes_[i]);

  if (dPrimes_) { api_->MemFree(dPrimes_); dPrimes_ = 0; }
  if (dMagics_) { api_->MemFree(dMagics_); dMagics_ = 0; }

  dPrimes_ = upload(primes_.data(), primes_.size() * sizeof(uint32_t), "sieving primes");
  dMagics_ = upload(magics_.data(), magics_.size() * sizeof(uint64_t), "reciprocals");
}

void CudaBackend::ensureKTable(int countType)
{
  if (dKTable_ && kTableType_ == countType)
    return;
  if (dKTable_) { api_->MemFree(dKTable_); dKTable_ = 0; }

  std::vector<uint8_t> table = buildKTupletTable(countType);
  dKTable_ = upload(table.data(), table.size(), "k-tuplet table");
  kTableType_ = countType;
}

uint64_t CudaBackend::runLaunch(uint64_t segLowBase, uint32_t numSegments,
                                uint64_t lo, uint64_t hi, int countType)
{
  uint32_t numPrimes  = (uint32_t) primes_.size();
  uint32_t preTables  = usePreSieve_ ? (uint32_t) preSieve_.len.size() : 0u;
  uint32_t restoreLen = usePreSieve_ ? (uint32_t) preSieve_.restore.size() : 0u;
  uint32_t ctype      = (uint32_t) countType;

  // The reciprocal identity holds only below 2^63; above it the kernel does
  // a real 64-bit division.
  uint32_t useMagic = (hi < (1ull << 63)) ? 1u : 0u;

  void* args[] = {
    &segLowBase, &sieveBytes_, &numSegments, &lo, &hi,
    &dPrimes_, &dMagics_, &useMagic, &numPrimes, &firstPrimeIdx_,
    &tier1End_, &tier2End_,
    &dPreSieve_, &preTables, &dPreLen_, &dPreOff_,
    &dRestore_, &restoreLen,
    &ctype, &dKTable_, &dCounts_
  };

  CUresult err = api_->LaunchKernel(kernel_,
                                    numSegments, 1, 1,
                                    blockSize_, 1, 1,
                                    sharedBytes_, nullptr, args, nullptr);
  if (err != CUDA_SUCCESS)
    throw primesieve::primesieve_error("CUDA: kernel launch failed: " + cudaErrorString(api_, err));

  err = api_->CtxSynchronize();
  if (err != CUDA_SUCCESS)
    throw primesieve::primesieve_error("CUDA: kernel execution failed: " + cudaErrorString(api_, err));

  err = api_->MemcpyDtoH(hostCounts_.data(), dCounts_, (std::size_t) numSegments * sizeof(uint64_t));
  if (err != CUDA_SUCCESS)
    throw primesieve::primesieve_error("CUDA: reading the counts failed: " + cudaErrorString(api_, err));

  uint64_t sum = 0;
  for (uint32_t i = 0; i < numSegments; i++)
    sum += hostCounts_[i];
  return sum;
}

uint64_t CudaBackend::count(uint64_t lo, uint64_t hi, int countType)
{
  if (hi < 7 || lo > hi)
    return 0;
  if (lo < 7)
    lo = 7;

  api_->CtxSetCurrent(ctx_);
  ensureSievingPrimes(isqrt64(hi));
  ensureKTable(countType);

  // A value v == 1 (mod 30) is stored as segmentLow + 31, i.e. in the byte
  // *before* the one a naive division would pick. primesieve calls this
  // byteRemainder(); getting it wrong silently loses one prime per range.
  uint64_t firstSegLow = lo - ((lo - 7) % 30 + 7);
  uint64_t span = (uint64_t) sieveBytes_ * 30;

  uint64_t total = 0;
  uint64_t segLow = firstSegLow;

  for (;;)
  {
    uint64_t remaining = (hi - segLow) / span + 1;
    uint32_t n = (uint32_t) std::min<uint64_t>(remaining, segmentsPerLaunch_);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    total += runLaunch(segLow, n, lo, hi, countType);
    double seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();

    // Keep a single launch short: desktop OSes reset a GPU whose kernel runs
    // for a couple of seconds, and the work per segment grows with
    // pi(sqrt(stop)).
    if (seconds > 0.75 && segmentsPerLaunch_ > 16)
      segmentsPerLaunch_ = std::max<uint32_t>(segmentsPerLaunch_ / 2, 16);
    else if (seconds < 0.05 && segmentsPerLaunch_ < maxSegmentsPerLaunch_)
      segmentsPerLaunch_ = std::min(segmentsPerLaunch_ * 2, maxSegmentsPerLaunch_);

    if (remaining <= n)
      break;

    uint64_t advance = (uint64_t) n * span;
    if (advance > hi - segLow)
      break;
    segLow += advance;
  }

  return total;
}

} // namespace

namespace primesieve {
namespace gpu {

void listCudaDevices(std::vector<DeviceInfo>& out)
{
  const CudaApi* api = loadCuda();
  if (!api)
    return;
  if (api->Init(0) != CUDA_SUCCESS)
    return;

  int count = 0;
  if (api->DeviceGetCount(&count) != CUDA_SUCCESS)
    return;

  for (int i = 0; i < count; i++)
  {
    CUdevice dev = 0;
    if (api->DeviceGet(&dev, i) != CUDA_SUCCESS)
      continue;

    char name[256] = { 0 };
    api->DeviceGetName(name, (int) sizeof name - 1, dev);

    std::size_t total = 0;
    api->DeviceTotalMem(&total, dev);

    int ccMajor = deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    int ccMinor = deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

    char version[64];
    std::snprintf(version, sizeof version, "CUDA compute %d.%d", ccMajor, ccMinor);

    DeviceInfo d;
    d.backend     = "CUDA";
    d.deviceIndex = i;
    d.name        = name;
    d.vendor      = "NVIDIA";
    d.version     = version;
    d.computeUnits     = (uint32_t) deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 1);
    d.clockMHz         = (uint32_t) (deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0) / 1000);
    d.globalMemBytes   = (uint64_t) total;
    d.localMemBytes    = (uint64_t) deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 49152);
    d.maxAllocBytes    = (uint64_t) total;
    d.maxWorkGroupSize = (uint64_t) deviceAttr(api, dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1024);
    out.push_back(d);
  }
}

std::unique_ptr<GpuBackend> createCudaBackend(const DeviceInfo& dev,
                                              const GpuConfig& config,
                                              std::string& error)
{
  std::unique_ptr<CudaBackend> backend(new CudaBackend());
  if (!backend->init(dev, config, error))
    return std::unique_ptr<GpuBackend>();
  return std::unique_ptr<GpuBackend>(backend.release());
}

} // namespace gpu
} // namespace primesieve

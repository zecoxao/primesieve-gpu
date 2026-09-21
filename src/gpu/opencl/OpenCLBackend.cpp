///
/// @file  OpenCLBackend.cpp
/// @brief OpenCL implementation of the GPU segmented sieve of Eratosthenes.
///
///        One work-group sieves one segment into local memory and returns a
///        single count, so the sieve array never leaves the device. The
///        sieving primes are produced on the CPU by primesieve itself, which
///        is cheap: they only go up to sqrt(stop).
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "../GpuBackend.hpp"
#include "../GpuTables.hpp"
#include "cl_loader.hpp"
#include "sieve_cl.hpp"

#include <primesieve.hpp>
#include <primesieve/primesieve_error.hpp>

#include <algorithm>
#include <cmath>
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

std::string clDeviceString(const ClApi* cl, cl_device_id dev, cl_device_info info)
{
  std::size_t n = 0;
  if (cl->GetDeviceInfo(dev, info, 0, nullptr, &n) != CL_SUCCESS || n == 0)
    return std::string();
  std::vector<char> buf(n + 1, 0);
  if (cl->GetDeviceInfo(dev, info, n, buf.data(), nullptr) != CL_SUCCESS)
    return std::string();
  return std::string(buf.data());
}

template <typename T>
T clDeviceValue(const ClApi* cl, cl_device_id dev, cl_device_info info, T fallback)
{
  T v = fallback;
  if (cl->GetDeviceInfo(dev, info, sizeof(T), &v, nullptr) != CL_SUCCESS)
    return fallback;
  return v;
}

struct RawDevice
{
  cl_platform_id platform;
  cl_device_id   device;
};

/// Every GPU device of every OpenCL platform, in a stable order.
std::vector<RawDevice> enumerateRaw(const ClApi* cl)
{
  std::vector<RawDevice> out;
  cl_uint numPlatforms = 0;

  if (cl->GetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0)
    return out;

  std::vector<cl_platform_id> platforms(numPlatforms);
  if (cl->GetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS)
    return out;

  for (cl_uint p = 0; p < numPlatforms; p++)
  {
    cl_uint numDevices = 0;
    if (cl->GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) != CL_SUCCESS)
      continue;
    if (numDevices == 0)
      continue;

    std::vector<cl_device_id> devices(numDevices);
    if (cl->GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr) != CL_SUCCESS)
      continue;

    for (cl_uint d = 0; d < numDevices; d++)
    {
      RawDevice raw;
      raw.platform = platforms[p];
      raw.device = devices[d];
      out.push_back(raw);
    }
  }
  return out;
}

class OpenCLBackend : public GpuBackend
{
public:
  OpenCLBackend() {}
  ~OpenCLBackend();

  bool init(const DeviceInfo& info, const GpuConfig& config, std::string& error);

  const DeviceInfo& device() const override { return info_; }
  uint32_t sieveBytes() const override { return sieveBytes_; }
  uint32_t workGroupSize() const override { return workGroupSize_; }

  uint64_t count(uint64_t lo, uint64_t hi, int countType) override;

private:
  void releaseBuffers();
  void ensureSievingPrimes(uint32_t sqrtHi);
  void ensureKTable(int countType);
  cl_mem createBuffer(cl_mem_flags flags, std::size_t bytes, const void* host, const char* what);
  uint64_t runLaunch(uint64_t segLowBase, uint32_t numSegments,
                     uint64_t lo, uint64_t hi, int countType);

  const ClApi*     cl_ = nullptr;
  DeviceInfo       info_;
  cl_device_id     dev_ = nullptr;
  cl_context       context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program       program_ = nullptr;
  cl_kernel        kernel_ = nullptr;

  uint32_t sieveBytes_ = 0;
  uint32_t workGroupSize_ = 0;
  uint32_t segmentsPerLaunch_ = 0;
  bool     usePreSieve_ = true;

  PreSieveTables preSieve_;
  cl_mem   bufPreSieve_ = nullptr;
  cl_mem   bufPreLen_ = nullptr;
  cl_mem   bufPreOff_ = nullptr;
  cl_mem   bufRestore_ = nullptr;

  std::vector<uint32_t> primes_;
  std::vector<uint64_t> magics_;
  uint32_t primesUpTo_ = 0;
  uint32_t firstPrimeIdx_ = 0;
  cl_mem   bufPrimes_ = nullptr;
  cl_mem   bufMagics_ = nullptr;

  int      kTableType_ = -1;
  cl_mem   bufKTable_ = nullptr;

  cl_mem   bufCounts_ = nullptr;
  std::vector<uint64_t> hostCounts_;
};

OpenCLBackend::~OpenCLBackend()
{
  releaseBuffers();
  if (cl_)
  {
    if (kernel_)  cl_->ReleaseKernel(kernel_);
    if (program_) cl_->ReleaseProgram(program_);
    if (queue_)   cl_->ReleaseCommandQueue(queue_);
    if (context_) cl_->ReleaseContext(context_);
  }
}

void OpenCLBackend::releaseBuffers()
{
  if (!cl_)
    return;
  cl_mem* all[] = { &bufPreSieve_, &bufPreLen_, &bufPreOff_, &bufRestore_,
                    &bufPrimes_, &bufMagics_, &bufKTable_, &bufCounts_ };
  for (std::size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
    if (*all[i]) { cl_->ReleaseMemObject(*all[i]); *all[i] = nullptr; }
}

cl_mem OpenCLBackend::createBuffer(cl_mem_flags flags, std::size_t bytes,
                                   const void* host, const char* what)
{
  cl_int err = CL_SUCCESS;
  // A zero sized buffer is illegal in OpenCL; allocate one dummy byte so
  // that the kernel argument stays valid.
  std::size_t n = bytes ? bytes : 1;
  cl_mem m = cl_->CreateBuffer(context_,
                               flags | (host ? CL_MEM_COPY_HOST_PTR : 0),
                               n, const_cast<void*>(host), &err);
  if (!m || err != CL_SUCCESS)
    throw primesieve::primesieve_error(std::string("OpenCL: failed to allocate ")
                                       + what + ": " + clErrorString(err));
  return m;
}

bool OpenCLBackend::init(const DeviceInfo& info, const GpuConfig& config, std::string& error)
{
  cl_ = loadOpenCL();
  if (!cl_)
  {
    error = openclLoadError();
    return false;
  }

  std::vector<RawDevice> raw = enumerateRaw(cl_);
  if (info.deviceIndex < 0 || (std::size_t) info.deviceIndex >= raw.size())
  {
    error = "OpenCL: device index out of range";
    return false;
  }

  info_ = info;
  dev_ = raw[(std::size_t) info.deviceIndex].device;
  cl_platform_id platform = raw[(std::size_t) info.deviceIndex].platform;

  cl_int err = CL_SUCCESS;
  cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0 };
  context_ = cl_->CreateContext(props, 1, &dev_, nullptr, nullptr, &err);
  if (!context_ || err != CL_SUCCESS)
  {
    error = std::string("OpenCL: clCreateContext failed: ") + clErrorString(err);
    return false;
  }

  if (cl_->CreateCommandQueueWithProperties)
  {
    cl_queue_properties qprops[] = { 0 };
    queue_ = cl_->CreateCommandQueueWithProperties(context_, dev_, qprops, &err);
  }
  else
  {
    queue_ = cl_->CreateCommandQueue(context_, dev_, 0, &err);
  }
  if (!queue_ || err != CL_SUCCESS)
  {
    error = std::string("OpenCL: failed to create a command queue: ") + clErrorString(err);
    return false;
  }

  // ---- build the program ------------------------------------------------
  // The wheel tables are emitted from GpuWheel.hpp so that the device code
  // and the host driver can never disagree about them.
  std::string source = wheelTableSource(/* openclConstant */ true);
  source += kSieveClSource;

  const char* src = source.c_str();
  std::size_t srcLen = source.size();
  program_ = cl_->CreateProgramWithSource(context_, 1, &src, &srcLen, &err);
  if (!program_ || err != CL_SUCCESS)
  {
    error = std::string("OpenCL: clCreateProgramWithSource failed: ") + clErrorString(err);
    return false;
  }

  err = cl_->BuildProgram(program_, 1, &dev_, "-cl-std=CL1.2", nullptr, nullptr);
  if (err != CL_SUCCESS)
    err = cl_->BuildProgram(program_, 1, &dev_, "", nullptr, nullptr);

  if (err != CL_SUCCESS)
  {
    std::size_t logLen = 0;
    cl_->GetProgramBuildInfo(program_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
    std::vector<char> log(logLen + 1, 0);
    cl_->GetProgramBuildInfo(program_, dev_, CL_PROGRAM_BUILD_LOG, logLen, log.data(), nullptr);
    error = std::string("OpenCL: kernel build failed: ") + clErrorString(err) + "\n" + log.data();
    return false;
  }

  kernel_ = cl_->CreateKernel(program_, "ps_count", &err);
  if (!kernel_ || err != CL_SUCCESS)
  {
    error = std::string("OpenCL: clCreateKernel failed: ") + clErrorString(err);
    return false;
  }

  // ---- pick the sieve geometry -----------------------------------------
  std::size_t kernelMaxWg = 0;
  cl_->GetKernelWorkGroupInfo(kernel_, dev_, CL_KERNEL_WORK_GROUP_SIZE,
                              sizeof kernelMaxWg, &kernelMaxWg, nullptr);
  if (kernelMaxWg == 0)
    kernelMaxWg = (std::size_t) info_.maxWorkGroupSize;

  uint32_t wg = config.workGroupSize ? config.workGroupSize : 256;
  wg = std::min<uint32_t>(wg, (uint32_t) kernelMaxWg);
  wg = std::min<uint32_t>(wg, (uint32_t) info_.maxWorkGroupSize);
  // The in-kernel reduction halves the work-group each step, so the size
  // must be a power of two.
  while (wg & (wg - 1))
    wg &= wg - 1;
  if (wg == 0)
  {
    error = "OpenCL: device reports an unusable maximum work-group size";
    return false;
  }
  workGroupSize_ = wg;

  // Local memory holds the sieve plus the reduction scratch.
  uint64_t scratch = (uint64_t) wg * sizeof(uint64_t);
  if (info_.localMemBytes <= scratch + 1024)
  {
    error = "OpenCL: device has too little local memory";
    return false;
  }

  // 16 KiB keeps several work-groups resident per compute unit, which hides
  // the latency of the local-memory atomics better than one huge segment.
  uint32_t sieveBytes = config.sieveBytes ? config.sieveBytes : 16384;
  uint64_t maxSieve = info_.localMemBytes - scratch - 256;
  sieveBytes = (uint32_t) std::min<uint64_t>(sieveBytes, maxSieve);
  sieveBytes &= ~3u;                       // whole 32-bit words
  if (sieveBytes < 256)
  {
    error = "OpenCL: device has too little local memory for a useful segment";
    return false;
  }
  sieveBytes_ = sieveBytes;

  segmentsPerLaunch_ = config.segmentsPerLaunch;
  if (segmentsPerLaunch_ == 0)
    segmentsPerLaunch_ = std::max<uint32_t>(info_.computeUnits, 1) * 64;

  usePreSieve_ = config.preSieve;

  // ---- upload the pre-sieve patterns -----------------------------------
  if (usePreSieve_)
  {
    preSieve_ = buildPreSieveTables(defaultPreSieveGroups());
    bufPreSieve_ = createBuffer(CL_MEM_READ_ONLY, preSieve_.data.size(), preSieve_.data.data(), "pre-sieve tables");
    bufPreLen_   = createBuffer(CL_MEM_READ_ONLY, preSieve_.len.size() * 4, preSieve_.len.data(), "pre-sieve lengths");
    bufPreOff_   = createBuffer(CL_MEM_READ_ONLY, preSieve_.off.size() * 4, preSieve_.off.data(), "pre-sieve offsets");
    bufRestore_  = createBuffer(CL_MEM_READ_ONLY, preSieve_.restore.size(), preSieve_.restore.data(), "pre-sieve restore bytes");
  }
  else
  {
    preSieve_ = PreSieveTables();
    bufPreSieve_ = createBuffer(CL_MEM_READ_ONLY, 0, nullptr, "pre-sieve tables");
    bufPreLen_   = createBuffer(CL_MEM_READ_ONLY, 0, nullptr, "pre-sieve lengths");
    bufPreOff_   = createBuffer(CL_MEM_READ_ONLY, 0, nullptr, "pre-sieve offsets");
    bufRestore_  = createBuffer(CL_MEM_READ_ONLY, 0, nullptr, "pre-sieve restore bytes");
  }

  hostCounts_.resize(segmentsPerLaunch_);
  bufCounts_ = createBuffer(CL_MEM_WRITE_ONLY, hostCounts_.size() * sizeof(uint64_t), nullptr, "count buffer");

  return true;
}

void OpenCLBackend::ensureSievingPrimes(uint32_t sqrtHi)
{
  if (bufPrimes_ && sqrtHi <= primesUpTo_)
    return;

  // Generating the sieving primes on the CPU is cheap -- they only reach
  // sqrt(stop) -- and reuses primesieve's own highly tuned sieve.
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

  // One reciprocal per sieving prime, so that the kernel can replace its
  // hottest 64-bit division with a multiply-high plus a shift.
  magics_.resize(primes_.size());
  for (std::size_t i = 0; i < primes_.size(); i++)
    magics_[i] = divisionMagic(primes_[i]);

  if (bufPrimes_)
  {
    cl_->ReleaseMemObject(bufPrimes_);
    bufPrimes_ = nullptr;
  }
  if (bufMagics_)
  {
    cl_->ReleaseMemObject(bufMagics_);
    bufMagics_ = nullptr;
  }
  bufPrimes_ = createBuffer(CL_MEM_READ_ONLY, primes_.size() * sizeof(uint32_t),
                            primes_.empty() ? nullptr : primes_.data(), "sieving primes");
  bufMagics_ = createBuffer(CL_MEM_READ_ONLY, magics_.size() * sizeof(uint64_t),
                            magics_.empty() ? nullptr : magics_.data(), "reciprocals");
}

void OpenCLBackend::ensureKTable(int countType)
{
  if (bufKTable_ && kTableType_ == countType)
    return;
  if (bufKTable_)
  {
    cl_->ReleaseMemObject(bufKTable_);
    bufKTable_ = nullptr;
  }
  std::vector<uint8_t> table = buildKTupletTable(countType);
  bufKTable_ = createBuffer(CL_MEM_READ_ONLY, table.size(), table.data(), "k-tuplet table");
  kTableType_ = countType;
}

uint64_t OpenCLBackend::runLaunch(uint64_t segLowBase, uint32_t numSegments,
                                  uint64_t lo, uint64_t hi, int countType)
{
  cl_uint a = 0;
  cl_int err = CL_SUCCESS;

  uint32_t numPrimes  = (uint32_t) primes_.size();
  uint32_t preTables  = usePreSieve_ ? (uint32_t) preSieve_.len.size() : 0u;
  uint32_t restoreLen = usePreSieve_ ? (uint32_t) preSieve_.restore.size() : 0u;
  uint32_t ctype      = (uint32_t) countType;

  // The reciprocal identity mul_hi(x, M) >> L == x / p only holds for
  // x < 2^63. Above that the kernel falls back to a real 64-bit division;
  // it is slower, but such ranges take a very long time anyway.
  uint32_t useMagic = (hi < (1ull << 63)) ? 1u : 0u;

  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_ulong), &segLowBase);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &sieveBytes_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &numSegments);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_ulong), &lo);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_ulong), &hi);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufPrimes_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufMagics_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &useMagic);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &numPrimes);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &firstPrimeIdx_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufPreSieve_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &preTables);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufPreLen_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufPreOff_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufRestore_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &restoreLen);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_uint),  &ctype);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufKTable_);
  err |= cl_->SetKernelArg(kernel_, a++, sizeof(cl_mem),   &bufCounts_);
  err |= cl_->SetKernelArg(kernel_, a++, sieveBytes_, nullptr);
  err |= cl_->SetKernelArg(kernel_, a++, (std::size_t) workGroupSize_ * sizeof(cl_ulong), nullptr);

  if (err != CL_SUCCESS)
    throw primesieve::primesieve_error("OpenCL: clSetKernelArg failed");

  std::size_t local = workGroupSize_;
  std::size_t global = (std::size_t) numSegments * local;

  err = cl_->EnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw primesieve::primesieve_error(std::string("OpenCL: kernel launch failed: ") + clErrorString(err));

  err = cl_->EnqueueReadBuffer(queue_, bufCounts_, CL_TRUE, 0,
                               (std::size_t) numSegments * sizeof(uint64_t),
                               hostCounts_.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw primesieve::primesieve_error(std::string("OpenCL: reading the counts failed: ") + clErrorString(err));

  uint64_t sum = 0;
  for (uint32_t i = 0; i < numSegments; i++)
    sum += hostCounts_[i];
  return sum;
}

uint64_t OpenCLBackend::count(uint64_t lo, uint64_t hi, int countType)
{
  if (hi < 7 || lo > hi)
    return 0;
  if (lo < 7)
    lo = 7;

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

    total += runLaunch(segLow, n, lo, hi, countType);

    if (remaining <= n)
      break;

    uint64_t advance = (uint64_t) n * span;
    // Guard against wrapping around the top of the 64-bit range.
    if (advance > hi - segLow)
      break;
    segLow += advance;
  }

  return total;
}

} // namespace

namespace primesieve {
namespace gpu {

void listOpenCLDevices(std::vector<DeviceInfo>& out)
{
  const ClApi* cl = loadOpenCL();
  if (!cl)
    return;

  std::vector<RawDevice> raw = enumerateRaw(cl);

  for (std::size_t i = 0; i < raw.size(); i++)
  {
    DeviceInfo d;
    d.backend     = "OpenCL";
    d.deviceIndex = (int) i;
    d.name        = clDeviceString(cl, raw[i].device, CL_DEVICE_NAME);
    d.vendor      = clDeviceString(cl, raw[i].device, CL_DEVICE_VENDOR);
    d.version     = clDeviceString(cl, raw[i].device, CL_DEVICE_VERSION);
    d.computeUnits     = clDeviceValue<cl_uint>(cl, raw[i].device, CL_DEVICE_MAX_COMPUTE_UNITS, 1);
    d.clockMHz         = clDeviceValue<cl_uint>(cl, raw[i].device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0);
    d.globalMemBytes   = clDeviceValue<cl_ulong>(cl, raw[i].device, CL_DEVICE_GLOBAL_MEM_SIZE, 0);
    d.localMemBytes    = clDeviceValue<cl_ulong>(cl, raw[i].device, CL_DEVICE_LOCAL_MEM_SIZE, 0);
    d.maxAllocBytes    = clDeviceValue<cl_ulong>(cl, raw[i].device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0);
    d.maxWorkGroupSize = (uint64_t) clDeviceValue<std::size_t>(cl, raw[i].device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 1);
    out.push_back(d);
  }
}

std::unique_ptr<GpuBackend> createOpenCLBackend(const DeviceInfo& dev,
                                                const GpuConfig& config,
                                                std::string& error)
{
  std::unique_ptr<OpenCLBackend> backend(new OpenCLBackend());
  if (!backend->init(dev, config, error))
    return std::unique_ptr<GpuBackend>();
  return std::unique_ptr<GpuBackend>(backend.release());
}

} // namespace gpu
} // namespace primesieve

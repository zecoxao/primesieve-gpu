///
/// @file  GpuSieve.cpp
/// @brief Device independent half of primesieve's GPU support: device
///        enumeration, backend selection, the primes 2/3/5 that the wheel-30
///        bit array cannot represent, and the heuristic that decides whether
///        a range is actually worth sending to the GPU.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "GpuBackend.hpp"

#include <primesieve.hpp>
#include <primesieve/gpu.hpp>
#include <primesieve/primesieve_error.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex           g_mutex;
int                  g_deviceIndex = -1;
bool                 g_enabled = false;
primesieve::gpu::GpuConfig g_config;
std::unique_ptr<primesieve::gpu::GpuBackend> g_backend;
std::string          g_lastError;

/// Ranges smaller than this are not worth a kernel launch plus the sieving
/// prime generation; the CPU sieve wins outright.
const uint64_t kMinGpuRange = 100000000ull;   // 1e8

} // namespace

namespace primesieve {
namespace gpu {

std::string DeviceInfo::describe() const
{
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "[%d] %s: %s (%s)\n"
                "      %u compute units, %u MHz, %llu MiB global, %llu KiB local, max work-group %llu",
                index, backend.c_str(), name.c_str(), version.c_str(),
                computeUnits, clockMHz,
                (unsigned long long) (globalMemBytes >> 20),
                (unsigned long long) (localMemBytes >> 10),
                (unsigned long long) maxWorkGroupSize);
  return std::string(buf);
}

std::vector<DeviceInfo> listDevices()
{
  std::vector<DeviceInfo> out;

#if defined(PRIMESIEVE_ENABLE_CUDA)
  listCudaDevices(out);
#endif
#if defined(PRIMESIEVE_ENABLE_OPENCL)
  listOpenCLDevices(out);
#endif

  for (std::size_t i = 0; i < out.size(); i++)
    out[i].index = (int) i;

  return out;
}

std::unique_ptr<GpuBackend> createBackend(int deviceIndex,
                                          const GpuConfig& config,
                                          std::string& error)
{
  std::vector<DeviceInfo> devices = listDevices();

  if (devices.empty())
  {
    error = "no GPU device found";
    return std::unique_ptr<GpuBackend>();
  }

  if (deviceIndex < 0)
  {
    // Prefer the device with the most compute units; on a laptop that picks
    // the discrete GPU over the integrated one.
    std::size_t best = 0;
    for (std::size_t i = 1; i < devices.size(); i++)
      if (devices[i].computeUnits > devices[best].computeUnits)
        best = i;
    deviceIndex = (int) best;
  }

  if (deviceIndex >= (int) devices.size())
  {
    error = "GPU device index out of range";
    return std::unique_ptr<GpuBackend>();
  }

  const DeviceInfo& dev = devices[(std::size_t) deviceIndex];

#if defined(PRIMESIEVE_ENABLE_CUDA)
  if (dev.backend == "CUDA")
    return createCudaBackend(dev, config, error);
#endif
#if defined(PRIMESIEVE_ENABLE_OPENCL)
  if (dev.backend == "OpenCL")
    return createOpenCLBackend(dev, config, error);
#endif

  error = "no backend compiled in for " + dev.backend;
  return std::unique_ptr<GpuBackend>();
}

/// The backend for the currently selected device, created on first use.
/// Returns nullptr if no GPU is usable; g_lastError explains why.
GpuBackend* activeBackend()
{
  if (!g_backend)
  {
    g_backend = createBackend(g_deviceIndex, g_config, g_lastError);
    if (!g_backend)
      return nullptr;
  }
  return g_backend.get();
}

uint64_t countOnGpu(uint64_t start, uint64_t stop, int countType, bool& usedGpu)
{
  usedGpu = false;

  if (start > stop)
    return 0;

  GpuBackend* backend = activeBackend();
  if (!backend)
    return 0;

  uint64_t total = 0;

  // The wheel-30 bit array starts at 7, so neither the primes 2, 3, 5 nor
  // any k-tuplet containing them can be represented in it. This is exactly
  // the table PrimeSieve::processSmallPrimes() uses on the CPU, and it is
  // applied under the same condition (start <= 5), so that the GPU and the
  // CPU agree on which tuplets near the origin count.
  if (start <= 5)
  {
    struct SmallPrime { uint64_t first; uint64_t last; int index; };
    static const SmallPrime smallPrimes[8] =
    {
      { 2,  2, 0 },   // 2
      { 3,  3, 0 },   // 3
      { 5,  5, 0 },   // 5
      { 3,  5, 1 },   // (3, 5)
      { 5,  7, 1 },   // (5, 7)
      { 5, 11, 2 },   // (5, 7, 11)
      { 5, 13, 3 },   // (5, 7, 11, 13)
      { 5, 17, 4 }    // (5, 7, 11, 13, 17)
    };

    for (int i = 0; i < 8; i++)
      if (smallPrimes[i].index == countType &&
          smallPrimes[i].first >= start &&
          smallPrimes[i].last <= stop)
        total++;
  }

  if (stop >= 7)
    total += backend->count(start, stop, countType);

  usedGpu = true;
  return total;
}

} // namespace gpu

// ---------------------------------------------------------------------------
// Public API (include/primesieve/gpu.hpp)
// ---------------------------------------------------------------------------

bool gpu_available()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return !gpu::listDevices().empty();
}

std::vector<std::string> gpu_devices()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<gpu::DeviceInfo> devices = gpu::listDevices();
  std::vector<std::string> out;
  for (std::size_t i = 0; i < devices.size(); i++)
    out.push_back(devices[i].describe());
  return out;
}

void set_gpu_device(int deviceIndex)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (deviceIndex != g_deviceIndex)
  {
    g_deviceIndex = deviceIndex;
    g_backend.reset();
  }
}

int get_gpu_device()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_deviceIndex;
}

void set_gpu_enabled(bool enable)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_enabled = enable;
}

bool get_gpu_enabled()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_enabled;
}

void set_gpu_sieve_size(int kilobytes)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_config.sieveBytes = (kilobytes > 0) ? (uint32_t) kilobytes * 1024 : 0;
  g_backend.reset();
}

void set_gpu_work_group_size(int size)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_config.workGroupSize = (size > 0) ? (uint32_t) size : 0;
  g_backend.reset();
}

void set_gpu_segments_per_launch(int segments)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_config.segmentsPerLaunch = (segments > 0) ? (uint32_t) segments : 0;
  g_backend.reset();
}

void set_gpu_presieve(bool enable)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_config.preSieve = enable;
  g_backend.reset();
}

std::string gpu_last_error()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_lastError;
}

std::string gpu_active_device()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  gpu::GpuBackend* b = gpu::activeBackend();
  if (!b)
    return std::string();
  return b->device().describe();
}

/// True when a range is large enough that the GPU can beat the CPU sieve.
bool gpu_worthwhile(uint64_t start, uint64_t stop)
{
  if (start > stop)
    return false;
  return (stop - start) >= kMinGpuRange;
}

uint64_t gpu_count_primes(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 0, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

uint64_t gpu_count_twins(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 1, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

uint64_t gpu_count_triplets(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 2, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

uint64_t gpu_count_quadruplets(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 3, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

uint64_t gpu_count_quintuplets(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 4, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

uint64_t gpu_count_sextuplets(uint64_t start, uint64_t stop)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  bool used = false;
  uint64_t n = gpu::countOnGpu(start, stop, 5, used);
  if (!used)
    throw primesieve_error("GPU is not available: " + g_lastError);
  return n;
}

/// Used by api.cpp: count on the GPU when it is enabled, available and
/// worthwhile, otherwise report failure so the caller falls back to the CPU.
bool gpu_try_count(uint64_t start, uint64_t stop, int countType, uint64_t& result)
{
  std::lock_guard<std::mutex> lock(g_mutex);

  if (!g_enabled || !gpu_worthwhile(start, stop))
    return false;

  try
  {
    bool used = false;
    uint64_t n = gpu::countOnGpu(start, stop, countType, used);
    if (!used)
      return false;
    result = n;
    return true;
  }
  catch (const std::exception& e)
  {
    // A GPU failure must never break primesieve: remember why and let the
    // caller fall back to the CPU sieve.
    g_lastError = e.what();
    g_backend.reset();
    return false;
  }
}

} // namespace primesieve

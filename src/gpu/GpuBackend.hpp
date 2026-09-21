///
/// @file  GpuBackend.hpp
/// @brief Abstract interface implemented by the OpenCL and CUDA backends.
///
///        A backend owns one GPU device and knows how to count primes (or
///        prime k-tuplets) in an interval using the segmented sieve of
///        Eratosthenes. Everything that is not device specific -- generating
///        the sieving primes, handling the primes 2, 3, 5 and deciding
///        whether the GPU is worth using at all -- lives in GpuSieve.cpp.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_BACKEND_HPP
#define PRIMESIEVE_GPU_BACKEND_HPP

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

namespace primesieve {
namespace gpu {

struct DeviceInfo
{
  std::string backend;       // "OpenCL" or "CUDA"
  std::string name;
  std::string vendor;
  std::string version;
  int      index = -1;       // index in listDevices()
  int      platformIndex = 0;
  int      deviceIndex = 0;
  uint32_t computeUnits = 0;
  uint32_t clockMHz = 0;
  uint64_t globalMemBytes = 0;
  uint64_t localMemBytes = 0;
  uint64_t maxAllocBytes = 0;
  uint64_t maxWorkGroupSize = 0;

  std::string describe() const;
};

/// Tunables; all of them have sensible auto-derived defaults.
struct GpuConfig
{
  /// Bytes of the local/shared sieve per work-group. 0 = derive from the
  /// device's local memory size.
  uint32_t sieveBytes = 0;
  /// Work-items per work-group. Must be a power of two. 0 = auto.
  uint32_t workGroupSize = 0;
  /// Segments per kernel launch. 0 = auto. Bounded so that a single launch
  /// stays well under the OS GPU watchdog timeout.
  uint32_t segmentsPerLaunch = 0;
  /// Enable pre-sieving of small primes.
  bool preSieve = true;
};

class GpuBackend
{
public:
  virtual ~GpuBackend() {}
  virtual const DeviceInfo& device() const = 0;

  /// Count primes (countType 0) or prime k-tuplets (countType 1..5, i.e.
  /// twins, triplets, quadruplets, quintuplets, sextuplets) in [lo, hi].
  /// `lo` is >= 7; the caller deals with the primes 2, 3 and 5.
  virtual uint64_t count(uint64_t lo, uint64_t hi, int countType) = 0;

  /// Sieve parameters actually in use, for --gpu-info and tests.
  virtual uint32_t sieveBytes() const = 0;
  virtual uint32_t workGroupSize() const = 0;
};

/// All GPU devices visible to any compiled-in backend.
std::vector<DeviceInfo> listDevices();

/// Create a backend for `deviceIndex` (an index into listDevices()), or for
/// the most promising device when deviceIndex < 0. Returns nullptr on
/// failure and fills `error`.
std::unique_ptr<GpuBackend> createBackend(int deviceIndex,
                                          const GpuConfig& config,
                                          std::string& error);

#if defined(PRIMESIEVE_ENABLE_OPENCL)
void listOpenCLDevices(std::vector<DeviceInfo>& out);
std::unique_ptr<GpuBackend> createOpenCLBackend(const DeviceInfo& dev,
                                                const GpuConfig& config,
                                                std::string& error);
#endif

#if defined(PRIMESIEVE_ENABLE_CUDA)
void listCudaDevices(std::vector<DeviceInfo>& out);
std::unique_ptr<GpuBackend> createCudaBackend(const DeviceInfo& dev,
                                              const GpuConfig& config,
                                              std::string& error);
#endif

} // namespace gpu
} // namespace primesieve

#endif

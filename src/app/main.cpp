///
/// @file   main.cpp
/// @brief  Command-line option handling for the primesieve
///         command-line application. The user's command-line options
///         are first parsed in CmdOptions.cpp and stored in a
///         CmdOptions object. Afterwards we execute the function
///         corresponding to the user's command-line options in the
///         main() function in main.cpp.
///
///         How to add a new command-line option:
///
///         1) Add a new option enum in CmdOptions.hpp.
///         2) Add your option to parseOptions() in CmdOptions.cpp.
///         3) Add your option to main() in main.cpp.
///         4) Document your option in help.cpp (--help option summary)
///            and in doc/primesieve.txt (manpage).
///
/// Copyright (C) 2026 Kim Walisch, <kim.walisch@gmail.com>
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "CmdOptions.hpp"

#if defined(PRIMESIEVE_ENABLE_GPU)
  #include <primesieve/gpu.hpp>
#endif

#include <CpuInfo.hpp>

#include <chrono>
#include <ParallelSieve.hpp>
#include <RiemannR.hpp>
#include <primesieve/macros.hpp>
#include <primesieve/primesieve_error.hpp>
#include <primesieve/Vector.hpp>

#include <stdint.h>
#include <exception>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(ENABLE_MULTIARCH_ARM_SVE)

namespace primesieve {

bool has_arm_sve();

} // namespace

#endif

#if defined(ENABLE_MULTIARCH_AVX512_BW)

namespace primesieve {

bool has_avx512_bw();

} // namespace

#endif

#if defined(ENABLE_MULTIARCH_AVX512_VBMI2)

namespace primesieve {

bool has_avx512_vbmi2();

} // namespace

#endif

void help(int exitCode);
void version();
void stressTest(const CmdOptions& opts);
void test();

using primesieve::Array;
using primesieve::ParallelSieve;
using primesieve::primesieve_error;
using primesieve::PRINT_STATUS;

namespace {

void printSettings(const ParallelSieve& ps)
{
  std::cout << "Sieve size = " << ps.getSieveSize() << " KiB" << std::endl;
  std::cout << "Threads = " << ps.idealNumThreads() << std::endl;
}

void printSeconds(double sec)
{
  std::cout << "Seconds: " << std::fixed << std::setprecision(3) << sec << std::endl;
}

/// Count & print primes and prime k-tuplets

#if !defined(PRIMESIEVE_ENABLE_GPU)

/// Reported when --gpu or --gpu-info is used but no GPU backend was built
/// into this copy of primesieve.
void noGpuSupport()
{
  std::cerr << "primesieve: this build has no GPU support; rebuild with "
               "-DWITH_OPENCL=ON or -DWITH_CUDA=ON." << std::endl;
}

#endif

#if defined(PRIMESIEVE_ENABLE_GPU)

uint64_t gpuCountByType(uint64_t start, uint64_t stop, int countType)
{
  switch (countType)
  {
    case 0:  return primesieve::gpu_count_primes(start, stop);
    case 1:  return primesieve::gpu_count_twins(start, stop);
    case 2:  return primesieve::gpu_count_triplets(start, stop);
    case 3:  return primesieve::gpu_count_quadruplets(start, stop);
    case 4:  return primesieve::gpu_count_quintuplets(start, stop);
    default: return primesieve::gpu_count_sextuplets(start, stop);
  }
}

/// Count on the GPU, one pass per requested count type. Returns false when
/// the GPU cannot serve this request -- printing, nothing to count, the
/// range is too small to pay for a kernel launch, or the GPU failed -- and
/// the caller then sieves on the CPU.
bool sieveOnGpu(const CmdOptions& opts,
                const ParallelSieve& ps,
                Array<uint64_t, 6>& counts,
                double& seconds)
{
  if (!opts.gpu || ps.isPrint())
    return false;

  bool anyCount = false;
  for (int i = 0; i < 6; i++)
    if (ps.isCount(i))
      anyCount = true;
  if (!anyCount)
    return false;

  uint64_t start = ps.getStart();
  uint64_t stop = ps.getStop();
  if (!primesieve::gpu_worthwhile(start, stop))
    return false;

  auto t1 = std::chrono::steady_clock::now();

  try
  {
    // The CPU sieve can count every k-tuplet type in a single pass; the GPU
    // counts one type per pass, which is still far quicker per pass.
    for (int i = 0; i < 6; i++)
      if (ps.isCount(i))
        counts[i] = gpuCountByType(start, stop, i);
  }
  catch (const std::exception& e)
  {
    std::cerr << "primesieve: GPU counting failed (" << e.what()
              << "), using the CPU sieve." << std::endl;
    return false;
  }

  auto t2 = std::chrono::steady_clock::now();
  seconds = std::chrono::duration<double>(t2 - t1).count();
  return true;
}

#endif

void sieve(const CmdOptions& opts)
{
  if (opts.numbers.empty())
    throw primesieve_error("missing STOP number");

  INDETERMINATE ParallelSieve ps;

  if (opts.flags)
    ps.setFlags(opts.flags);
  if (opts.status)
    ps.addFlags(PRINT_STATUS);
  if (opts.sieveSize)
    ps.setSieveSize(opts.sieveSize);
  if (opts.threads)
    ps.setNumThreads(opts.threads);
  if (ps.isPrint())
    ps.setNumThreads(1);

  if (opts.numbers.size() < 2)
    ps.setStop(opts.numbers[0]);
  else
  {
    ps.setStart(opts.numbers[0]);
    ps.setStop(opts.numbers[1]);
  }

  Array<uint64_t, 6> counts;
  counts.fill(0);
  double seconds = 0;
  bool usedGpu = false;

#if defined(PRIMESIEVE_ENABLE_GPU)
  if (!opts.quiet && opts.gpu && !primesieve::gpu_active_device().empty())
    std::cout << "GPU = " << primesieve::gpu_active_device() << std::endl;

  usedGpu = sieveOnGpu(opts, ps, counts, seconds);
#endif

  if (!usedGpu)
  {
    if (!opts.quiet)
      printSettings(ps);

    ps.sieve();
    seconds = ps.getSeconds();
    for (int i = 0; i < 6; i++)
      counts[i] = ps.getCount(i);
  }

  const Array<std::string, 6> labels =
  {
    "Primes: ",
    "Twin primes: ",
    "Prime triplets: ",
    "Prime quadruplets: ",
    "Prime quintuplets: ",
    "Prime sextuplets: "
  };

  if (opts.time)
    printSeconds(seconds);

  // Did we count primes & k-tuplets simultaneously?
  int cnt = 0;
  for (int i = 0; i < 6; i++)
    if (ps.isCount(i))
      cnt++;

  for (int i = 0; i < 6; i++)
  {
    if (ps.isCount(i))
    {
      if (opts.quiet && cnt == 1)
        std::cout << counts[i] << std::endl;
      else
        std::cout << labels[i] << counts[i] << std::endl;
    }
  }
}

void nthPrime(const CmdOptions& opts)
{
  if (opts.numbers.empty())
    throw primesieve_error("missing n number");

  INDETERMINATE ParallelSieve ps;
  int64_t n = opts.numbers[0];
  uint64_t start = 0;

  if (opts.numbers.size() > 1)
    start = opts.numbers[1];
  if (opts.flags)
    ps.setFlags(opts.flags);
  if (opts.sieveSize)
    ps.setSieveSize(opts.sieveSize);
  if (opts.threads)
    ps.setNumThreads(opts.threads);

  uint64_t nthPrime = 0;
  ps.setStart(start);
  ps.setStop(start + std::abs(n * 20));

  if (!opts.quiet)
    printSettings(ps);

  nthPrime = ps.nthPrime(n, start);

  if (opts.time)
    printSeconds(ps.getSeconds());

  if (opts.quiet)
    std::cout << nthPrime << std::endl;
  else
    std::cout << "Nth prime: " << nthPrime << std::endl;
}

void RiemannR(const CmdOptions& opts)
{
  if (opts.numbers.empty())
    throw primesieve_error("missing x number");

  long double x = (long double) opts.numbers[0];
  long double Rx = primesieve::RiemannR(x);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(10) << Rx;
  std::string res = oss.str();

  // Remove trailing 0 decimal digits
  if (res.find('.') != std::string::npos)
  {
    std::reverse(res.begin(), res.end());
    res = res.substr(res.find_first_not_of('0'));
    if (res.at(0) == '.')
      res = res.substr(1);

    std::reverse(res.begin(), res.end());
  }

  std::cout << res << std::endl;
}

void RiemannR_inverse(const CmdOptions& opts)
{
  if (opts.numbers.empty())
    throw primesieve_error("missing x number");

  long double x = (long double) opts.numbers[0];
  long double R_inv_x = primesieve::RiemannR_inverse(x);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(10) << R_inv_x;
  std::string res = oss.str();

  // Remove trailing 0 decimal digits
  if (res.find('.') != std::string::npos)
  {
    std::reverse(res.begin(), res.end());
    res = res.substr(res.find_first_not_of('0'));
    if (res.at(0) == '.')
      res = res.substr(1);

    std::reverse(res.begin(), res.end());
  }

  std::cout << res << std::endl;
}

void cpuInfo()
{
  const primesieve::CpuInfo cpu;

  if (cpu.hasCpuName())
    std::cout << cpu.cpuName() << std::endl;
  else
    std::cout << "CPU name: unknown" << std::endl;

  if (cpu.hasLogicalCpuCores())
    std::cout << "Logical CPU cores: " << cpu.logicalCpuCores() << std::endl;
  else
    std::cout << "Logical CPU cores: unknown" << std::endl;

  #if defined(ENABLE_MULTIARCH_ARM_SVE)
    if (primesieve::has_arm_sve())
      std::cout << "Has ARM SVE: yes" << std::endl;
    else
      std::cout << "Has ARM SVE: no" << std::endl;
  #endif

  #if defined(ENABLE_MULTIARCH_AVX512_BW)
    if (primesieve::has_avx512_bw())
      std::cout << "Has AVX512 BW: yes" << std::endl;
    else
      std::cout << "Has AVX512 BW: no" << std::endl;
  #endif

  #if defined(ENABLE_MULTIARCH_AVX512_VBMI2)
    if (primesieve::has_avx512_vbmi2())
      std::cout << "Has AVX512 VBMI2: yes" << std::endl;
    else
      std::cout << "Has AVX512 VBMI2: no" << std::endl;
  #endif

  if (cpu.hasL1Cache())
    std::cout << "L1 cache size: " << (cpu.l1CacheBytes() >> 10) << " KiB" << std::endl;

  if (cpu.hasL2Cache())
    std::cout << "L2 cache size: " << (cpu.l2CacheBytes() >> 10) << " KiB" << std::endl;

  if (cpu.hasL3Cache())
    std::cout << "L3 cache size: " << (cpu.l3CacheBytes() >> 20) << " MiB" << std::endl;

  if (cpu.hasL1Cache())
  {
    if (!cpu.hasL1Sharing())
      std::cout << "L1 cache sharing: unknown" << std::endl;
    else
      std::cout << "L1 cache sharing: " << cpu.l1Sharing()
                << ((cpu.l1Sharing() > 1) ? " threads" : " thread") << std::endl;
  }

  if (cpu.hasL2Cache())
  {
    if (!cpu.hasL2Sharing())
      std::cout << "L2 cache sharing: unknown" << std::endl;
    else
      std::cout << "L2 cache sharing: " << cpu.l2Sharing()
                << ((cpu.l2Sharing() > 1) ? " threads" : " thread") << std::endl;
  }

  if (cpu.hasL3Cache())
  {
    if (!cpu.hasL3Sharing())
      std::cout << "L3 cache sharing: unknown" << std::endl;
    else
      std::cout << "L3 cache sharing: " << cpu.l3Sharing()
                << ((cpu.l3Sharing() > 1) ? " threads" : " thread") << std::endl;
  }

  if (!cpu.hasL1Cache() &&
      !cpu.hasL2Cache() &&
      !cpu.hasL3Cache())
  {
    std::cout << "L1 cache size: unknown" << std::endl;
    std::cout << "L2 cache size: unknown" << std::endl;
    std::cout << "L3 cache size: unknown" << std::endl;
    std::cout << "L1 cache sharing: unknown" << std::endl;
    std::cout << "L2 cache sharing: unknown" << std::endl;
    std::cout << "L3 cache sharing: unknown" << std::endl;
  }
}

} // namespace


#if defined(PRIMESIEVE_ENABLE_GPU)

/// --gpu-info: list the GPU devices primesieve can use.
void gpuInfo()
{
  std::vector<std::string> devices = primesieve::gpu_devices();

  if (devices.empty())
  {
    std::cout << "No GPU devices found." << std::endl;
    std::string err = primesieve::gpu_last_error();
    if (!err.empty())
      std::cout << err << std::endl;
    return;
  }

  std::cout << "GPU devices:" << std::endl;
  for (std::size_t i = 0; i < devices.size(); i++)
    std::cout << devices[i] << std::endl;
}

/// Turn on the GPU backend if --gpu was given. Counting then runs on the
/// GPU whenever the range is large enough to pay for a kernel launch, and
/// falls back to the CPU sieve otherwise.
void applyGpuOptions(const CmdOptions& opts)
{
  if (!opts.gpu)
    return;

  primesieve::set_gpu_device(opts.gpuDevice);
  primesieve::set_gpu_enabled(true);

  if (primesieve::gpu_active_device().empty())
  {
    std::cerr << "primesieve: could not initialise a GPU, using the CPU sieve."
              << std::endl;
    std::string err = primesieve::gpu_last_error();
    if (!err.empty())
      std::cerr << "primesieve: " << err << std::endl;
    primesieve::set_gpu_enabled(false);
  }
}

#endif

int main(int argc, char* argv[])
{
  try
  {
    CmdOptions opts = parseOptions(argc, argv);

#if defined(PRIMESIEVE_ENABLE_GPU)
    applyGpuOptions(opts);
#else
    if (opts.gpu)
      noGpuSupport();
#endif

    switch (opts.option)
    {
      case OPTION_CPU_INFO:    cpuInfo(); break;
#if defined(PRIMESIEVE_ENABLE_GPU)
      case OPTION_GPU_INFO:    gpuInfo(); break;
#else
      case OPTION_GPU_INFO:    noGpuSupport(); break;
#endif
      case OPTION_HELP:        help(/* exitCode */ 0); break;
      case OPTION_NTH_PRIME:   nthPrime(opts); break;
      case OPTION_R:           RiemannR(opts); break;
      case OPTION_R_INVERSE:   RiemannR_inverse(opts); break;
      case OPTION_STRESS_TEST: stressTest(opts); break;
      case OPTION_TEST:        test(); break;
      case OPTION_VERSION:     version(); break;
      default:                 sieve(opts); break;
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "primesieve: " << e.what() << std::endl
              << "Try 'primesieve --help' for more information." << std::endl;
    return 1;
  }

  return 0;
}

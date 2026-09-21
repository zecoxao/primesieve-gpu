///
/// @file   gpu_count.cpp
/// @brief  Check that every GPU backend counts primes and prime k-tuplets
///         exactly like the CPU sieve, which is the reference
///         implementation. The ranges deliberately include the awkward
///         cases: empty ranges, ranges below 7 (which the wheel-30 bit
///         array cannot represent), the k-tuplets that contain 3 and 5,
///         and values == 1 (mod 30), which live in the *previous* byte of
///         the bit array.
///
///         Skips (exit 0) when the machine has no usable GPU, so that the
///         test suite still passes on CI without one.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include <primesieve.hpp>
#include <primesieve/gpu.hpp>

#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

uint64_t cpuCount(uint64_t a, uint64_t b, int type)
{
  switch (type)
  {
    case 0:  return primesieve::count_primes(a, b);
    case 1:  return primesieve::count_twins(a, b);
    case 2:  return primesieve::count_triplets(a, b);
    case 3:  return primesieve::count_quadruplets(a, b);
    case 4:  return primesieve::count_quintuplets(a, b);
    default: return primesieve::count_sextuplets(a, b);
  }
}

uint64_t gpuCount(uint64_t a, uint64_t b, int type)
{
  switch (type)
  {
    case 0:  return primesieve::gpu_count_primes(a, b);
    case 1:  return primesieve::gpu_count_twins(a, b);
    case 2:  return primesieve::gpu_count_triplets(a, b);
    case 3:  return primesieve::gpu_count_quadruplets(a, b);
    case 4:  return primesieve::gpu_count_quintuplets(a, b);
    default: return primesieve::gpu_count_sextuplets(a, b);
  }
}

void check(uint64_t a, uint64_t b, int type)
{
  uint64_t want = cpuCount(a, b, type);
  uint64_t got = gpuCount(a, b, type);
  checks++;

  if (want != got)
  {
    std::cout << "   ERROR type=" << type
              << " [" << a << ", " << b << "]"
              << " gpu=" << got << " cpu=" << want << "\n";
    failures++;
  }
}

void runDevice(int deviceIndex, const std::string& name)
{
  primesieve::set_gpu_device(deviceIndex);

  if (primesieve::gpu_active_device().empty())
  {
    std::cout << "   skipping " << name << ": "
              << primesieve::gpu_last_error() << "\n";
    return;
  }

  std::cout << "   " << name << "\n";

  // Small stops, every count type. Covers the primes 2/3/5 and the
  // k-tuplets containing them, which primesieve counts from a table.
  const uint64_t smallStops[] = { 0, 1, 2, 3, 5, 6, 7, 8, 30, 31, 49,
                                  100, 163, 164, 241, 1000, 65537, 1000000 };
  for (int type = 0; type <= 5; type++)
    for (std::size_t i = 0; i < sizeof(smallStops) / sizeof(smallStops[0]); i++)
      check(0, smallStops[i], type);

  // Ranges starting at every residue mod 30, to pin down the byte boundary.
  for (uint64_t base = 0; base <= 120; base++)
    for (uint64_t len = 0; len <= 70; len += 7)
      check(base, base + len, 0);

  // Ranges that cross a segment boundary far from the origin.
  for (uint64_t start = 999999900ull; start <= 1000000000ull; start++)
    check(start, start + 331, 0);

  // Larger ranges, every count type.
  const uint64_t ranges[][2] = {
    { 0,                  10000000ull },
    { 0,                 100000000ull },
    { 1000000000ull,    1100000000ull },
    { 999999999989ull,  1000000200000ull },
    { 10000000000000ull, 10000010000000ull }
  };
  for (std::size_t i = 0; i < sizeof(ranges) / sizeof(ranges[0]); i++)
    for (int type = 0; type <= 5; type++)
      check(ranges[i][0], ranges[i][1], type);

  // Known values of pi(x).
  const struct { uint64_t stop; uint64_t pi; } known[] = {
    { 1000000ull,     78498ull },
    { 100000000ull,   5761455ull },
    { 10000000000ull, 455052511ull }
  };
  for (std::size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++)
  {
    uint64_t got = primesieve::gpu_count_primes(0, known[i].stop);
    checks++;
    if (got != known[i].pi)
    {
      std::cout << "   ERROR pi(" << known[i].stop << ") = " << got
                << ", expected " << known[i].pi << "\n";
      failures++;
    }
  }
}

} // namespace

int main()
{
  std::vector<std::string> devices = primesieve::gpu_devices();

  if (devices.empty())
  {
    std::cout << "No GPU found, skipping the GPU tests.\n";
    if (!primesieve::gpu_last_error().empty())
      std::cout << primesieve::gpu_last_error() << "\n";
    return 0;
  }

  std::cout << "Checking " << devices.size() << " GPU device(s) against the CPU sieve:\n";

  // Every device, so that a machine with both an OpenCL and a CUDA backend,
  // or a discrete and an integrated GPU, checks all of them.
  for (std::size_t i = 0; i < devices.size(); i++)
    runDevice((int) i, devices[i]);

  std::cout << checks << " checks, " << failures << " failures\n";

  if (failures)
    std::exit(1);

  std::cout << "   OK\n";
  return 0;
}

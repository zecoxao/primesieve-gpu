///
/// @file  GpuTables.cpp
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "GpuTables.hpp"
#include "GpuWheel.hpp"

#include <primesieve/primesieve_error.hpp>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

/// Value represented by bit `b` of byte `i` of a sieve starting at 0.
inline uint64_t bitValue(uint64_t byteIndex, int bit)
{
  return byteIndex * 30 + primesieve::gpu::gpuBitOffset[bit];
}

} // namespace

namespace primesieve {
namespace gpu {

const std::vector<std::vector<uint32_t>>& defaultPreSieveGroups()
{
  // Periods: 7*11*13*17 = 17017 B, 19*23*29 = 12673 B, 31*37*41 = 47027 B,
  // 43*47*53 = 107113 B. ~180 KiB in total, which stays resident in the L2
  // cache of any GPU worth using, and costs only 4 loads per sieve word
  // while removing the multiples of every prime <= 53.
  static const std::vector<std::vector<uint32_t>> groups = {
    {  7, 11, 13, 17 },
    { 19, 23, 29 },
    { 31, 37, 41 },
    { 43, 47, 53 }
  };
  return groups;
}

PreSieveTables buildPreSieveTables(const std::vector<std::vector<uint32_t>>& groups)
{
  PreSieveTables out;
  std::vector<uint32_t> allPrimes;

  for (const std::vector<uint32_t>& group : groups)
  {
    uint64_t period = 1;
    for (uint32_t p : group)
    {
      if (p < 7 || p % 2 == 0 || p % 3 == 0 || p % 5 == 0)
        throw primesieve_error("pre-sieve primes must be >= 7 and coprime to 30");
      period *= p;
      allPrimes.push_back(p);
      out.maxPrime = std::max(out.maxPrime, p);
    }

    if (period > 64ull * 1024 * 1024)
      throw primesieve_error("pre-sieve pattern is too large");

    uint32_t len = (uint32_t) period;
    uint32_t off = (uint32_t) out.data.size();
    out.len.push_back(len);
    out.off.push_back(off);

    // Build the pattern one byte per period position first, then fold it
    // into the sliding-window form the GPU reads.
    std::vector<uint8_t> pattern(len, 0xff);

    // Clear every bit whose value is divisible by one of the group's primes.
    // The pattern covers the numbers [0, len*30); because the period is a
    // multiple of each prime, applying it at any multiple-of-30 offset
    // removes exactly the multiples of those primes.
    uint8_t* table = pattern.data();
    for (uint32_t p : group)
    {
      // Walk the multiples of p that are coprime to 30.
      for (uint64_t m = p; m < (uint64_t) len * 30; m += p)
      {
        if (m % 2 == 0 || m % 3 == 0 || m % 5 == 0)
          continue;
        uint64_t byteIndex = (m - 7) / 30;
        int bit = -1;
        for (int b = 0; b < 8; b++)
          if (bitValue(byteIndex, b) == m) { bit = b; break; }
        if (bit < 0)
          throw primesieve_error("pre-sieve: value is not representable");
        table[byteIndex] &= (uint8_t) ~(1u << bit);
      }
    }

    // Fold into the sliding window: element i holds the four pattern bytes
    // starting at i, so the kernel reads a whole sieve word in one load.
    out.data.resize(out.data.size() + len);
    uint32_t* window = out.data.data() + off;
    for (uint32_t i = 0; i < len; i++)
      window[i] =  (uint32_t) pattern[i]
                | ((uint32_t) pattern[(i + 1) % len] << 8)
                | ((uint32_t) pattern[(i + 2) % len] << 16)
                | ((uint32_t) pattern[(i + 3) % len] << 24);
  }

  // Pre-sieving removed the small primes themselves. Rebuild the first few
  // bytes so that a bit is set iff its value is not a multiple of a
  // pre-sieve prime, or is a pre-sieve prime itself. (49 = 7*7 stays clear.)
  uint32_t restoreBytes = out.maxPrime / 30 + 1;
  out.restore.assign(restoreBytes, 0xff);
  for (uint32_t i = 0; i < restoreBytes; i++)
  {
    uint8_t byte = 0xff;
    for (int b = 0; b < 8; b++)
    {
      uint64_t v = bitValue(i, b);
      bool isPreSievePrime = std::find(allPrimes.begin(), allPrimes.end(), (uint32_t) v) != allPrimes.end();
      bool divisible = false;
      for (uint32_t p : allPrimes)
        if (v % p == 0) { divisible = true; break; }
      if (divisible && !isPreSievePrime)
        byte &= (uint8_t) ~(1u << b);
    }
    out.restore[i] = byte;
  }

  return out;
}

std::vector<uint8_t> buildKTupletTable(int countType)
{
  // Same bitmasks primesieve's CountPrintPrimes uses. A k-tuplet always
  // fits inside a single byte of this bit layout, so counting per byte is
  // exact.
  static const uint64_t bitmasks[6][5] =
  {
    { ~0ull },                          // primes (unused, popcount instead)
    { 0x06, 0x18, 0xc0, ~0ull },        // twins
    { 0x07, 0x0e, 0x1c, 0x38, ~0ull },  // triplets
    { 0x1e, ~0ull },                    // quadruplets
    { 0x1f, 0x3e, ~0ull },              // quintuplets
    { 0x3f, ~0ull }                     // sextuplets
  };

  std::vector<uint8_t> table(256, 0);

  for (uint64_t j = 0; j < 256; j++)
  {
    if (countType == 0)
    {
      uint8_t c = 0;
      for (int b = 0; b < 8; b++)
        c += (uint8_t) ((j >> b) & 1);
      table[(std::size_t) j] = c;
    }
    else
    {
      uint8_t c = 0;
      for (const uint64_t* b = bitmasks[countType]; *b <= j; b++)
        if ((j & *b) == *b)
          c++;
      table[(std::size_t) j] = c;
    }
  }
  return table;
}

uint64_t divisionMagic(uint32_t prime)
{
  if (prime < 3)
    throw primesieve_error("divisionMagic(): prime must be >= 3");

  // L = floor(log2(prime))
  uint32_t L = 0;
  while ((prime >> (L + 1)) != 0)
    L++;

  // Long division of 2^(64+L) by prime, done bit by bit so that no 128-bit
  // integer type is needed. The remainder stays below prime <= 2^32, so
  // (rem << 1) cannot overflow.
  const int n = 64 + (int) L;
  uint64_t quot = 0;
  uint64_t rem = 0;

  for (int i = n; i >= 0; i--)
  {
    rem = (rem << 1) | (uint64_t) ((i == n) ? 1 : 0);
    quot <<= 1;
    if (rem >= prime)
    {
      rem -= prime;
      quot |= 1;
    }
  }

  // ceil(); rem is never 0 because prime is odd and > 1.
  return quot + (rem != 0 ? 1u : 0u);
}

std::string wheelTableSource(bool openclConstant)
{
  const char* qualifier = openclConstant ? "__constant" : "__device__ const";
  char buf[256];
  std::string s;

  s += "// Wheel-30 tables, emitted from src/gpu/GpuWheel.hpp.\n";
  s += "// Each entry of psWheel packs a GpuWheelElement:\n";
  s += "//   bits  0..7  unsetBit, 8..15 nextMultipleFactor,\n";
  s += "//   bits 16..23 correct,  24..31 next.\n";

  std::snprintf(buf, sizeof buf, "%s uint psWheel[64] = {\n", qualifier);
  s += buf;
  for (int i = 0; i < 64; i++)
  {
    const GpuWheelElement& e = gpuWheel30[i];
    uint32_t packed = (uint32_t) e.unsetBit
                    | ((uint32_t) e.nextMultipleFactor << 8)
                    | ((uint32_t) e.correct << 16)
                    | ((uint32_t) e.next << 24);
    std::snprintf(buf, sizeof buf, "0x%08xu%s", packed, (i == 63) ? "" : ", ");
    s += buf;
    if (i % 8 == 7) s += "\n";
  }
  s += "};\n";

  // psWheelInit packs nextMultipleFactor in bits 0..7 and wheelIndex in 8..15
  std::snprintf(buf, sizeof buf, "%s uint psWheelInit[30] = {\n", qualifier);
  s += buf;
  for (int i = 0; i < 30; i++)
  {
    uint32_t packed = (uint32_t) gpuWheel30Init[i].nextMultipleFactor
                    | ((uint32_t) gpuWheel30Init[i].wheelIndex << 8);
    std::snprintf(buf, sizeof buf, "0x%04xu%s", packed, (i == 29) ? "" : ", ");
    s += buf;
    if (i % 10 == 9) s += "\n";
  }
  s += "};\n";

  std::snprintf(buf, sizeof buf, "%s uint psWheelOffset[30] = {\n", qualifier);
  s += buf;
  for (int i = 0; i < 30; i++)
  {
    std::snprintf(buf, sizeof buf, "%uu%s", (unsigned) gpuWheelOffsets[i], (i == 29) ? "" : ", ");
    s += buf;
  }
  s += "\n};\n";

  std::snprintf(buf, sizeof buf, "%s uint psBitOffset[8] = {", qualifier);
  s += buf;
  for (int i = 0; i < 8; i++)
  {
    std::snprintf(buf, sizeof buf, "%uu%s", (unsigned) gpuBitOffset[i], (i == 7) ? "" : ", ");
    s += buf;
  }
  s += "};\n\n";

  return s;
}

} // namespace gpu
} // namespace primesieve

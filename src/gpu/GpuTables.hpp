///
/// @file  GpuTables.hpp
/// @brief Lookup tables uploaded to the GPU: the pre-sieve patterns, the
///        per-byte prime k-tuplet count tables and the OpenCL/CUDA source
///        text for the wheel-30 tables.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_TABLES_HPP
#define PRIMESIEVE_GPU_TABLES_HPP

#include <stdint.h>
#include <string>
#include <vector>

namespace primesieve {
namespace gpu {

/// Pre-sieve patterns. Each table removes the multiples of its own set of
/// small primes. A table built from the primes {p1..pk} has a period of
/// p1*..*pk bytes (= p1*..*pk*30 numbers), so byte i of a segment starting
/// at segmentLow is table[(segmentLow/30 + i) % len].
///
/// The GPU ANDs all tables together while filling its local sieve, which is
/// much cheaper than crossing off those primes one multiple at a time.
struct PreSieveTables
{
  /// All tables concatenated, as a sliding window: element i of a table
  /// holds the four pattern bytes starting at byte i (wrapping at the end
  /// of the period). That lets the kernel fill a 32-bit sieve word with one
  /// coalesced load instead of four byte loads, at the cost of 4x memory --
  /// still small enough to stay resident in a GPU's L2 cache.
  std::vector<uint32_t> data;
  /// Period of each table, in bytes.
  std::vector<uint32_t> len;
  /// Offset of each table inside `data`, in 32-bit elements.
  std::vector<uint32_t> off;
  /// Largest prime removed by the tables.
  uint32_t maxPrime = 0;
  /// Pre-sieving also removes the small primes themselves. These bytes
  /// restore them; they overwrite the start of the very first segment
  /// exactly like primesieve's PreSieve::primeBits does on the CPU.
  std::vector<uint8_t>  restore;

  uint64_t totalBytes() const { return (uint64_t) data.size() * 4; }
};

/// Build the pre-sieve patterns for the given prime groups, e.g.
/// {{7,11,13,17},{19,23,29},{31,37,41}}. Every prime must be >= 7, coprime
/// to 30, and appear in exactly one group.
PreSieveTables buildPreSieveTables(const std::vector<std::vector<uint32_t>>& groups);

/// The default grouping: a balance between pattern size (they should stay
/// resident in the GPU's L2 cache) and the number of loads per sieve word.
const std::vector<std::vector<uint32_t>>& defaultPreSieveGroups();

/// 256-entry table: for each possible sieve byte, how many prime k-tuplets
/// of the given type start in it. countType 1..5 selects twins, triplets,
/// quadruplets, quintuplets and sextuplets, matching primesieve's
/// CountPrintPrimes bitmasks. countType 0 returns popcount per byte.
std::vector<uint8_t> buildKTupletTable(int countType);

/// Magic number for replacing the division x / prime on the GPU.
///
/// Returns M = ceil(2^(64+L) / prime) with L = floor(log2(prime)), so that
///
///     x / prime == mul_hi(x, M) >> L        for every x < 2^63
///
/// A 64-bit division costs ~70 instructions on a GPU (it is emulated in
/// software) and the sieve performs one per sieving prime per segment, so
/// this is one of the hottest constants in the whole backend. L is not
/// returned because the device recomputes it as 31 - clz(prime).
uint64_t divisionMagic(uint32_t prime);

/// Emits the wheel-30 tables from GpuWheel.hpp as OpenCL C / CUDA source,
/// so that the device code and the host driver can never disagree.
std::string wheelTableSource(bool openclConstant);

} // namespace gpu
} // namespace primesieve

#endif

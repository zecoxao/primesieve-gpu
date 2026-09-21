//
// primesieve GPU backend -- OpenCL sieve of Eratosthenes.
//
// One work-group sieves one segment into local (shared) memory using the
// same wheel-30 bit layout as primesieve's CPU sieve: 8 bits per 30 numbers,
// where bit b of byte i represents
//
//     segmentLow + i*30 + {7,11,13,17,19,23,29,31}[b]
//
// After the segment has been sieved the work-group counts the set bits and
// writes a single count per segment to global memory, so the sieve array
// itself never crosses the PCIe bus.
//
// The host prepends the wheel tables (psWheel, psWheelInit, psWheelOffset,
// psBitOffset) to this source, generated from src/gpu/GpuWheel.hpp, so that
// the host and the device can never disagree about them.
//
// This file is distributed under the BSD License. See the COPYING
// file in the top level directory.
//

// Clear one bit of the local sieve. Several work-items may target the same
// 32-bit word concurrently, so the read-modify-write must be atomic --
// a plain load/store would lose updates and leave composites marked prime.
inline void psClearBit(volatile __local uint* sieve, uint byteIdx, uint unsetBit)
{
  uint word  = byteIdx >> 2;
  uint shift = (byteIdx & 3u) << 3;
  atomic_and(&sieve[word], ~((uint)(0xffu ^ unsetBit) << shift));
}

//
// Cross off the multiples of the sieving primes in [begin, end).
//
// The work is split along two axes at once:
//
//   * primeStride / groupId  -- which work-items share a prime,
//   * numSlices / sliceId    -- which part of the segment this work-item owns.
//
// The number of multiples a prime has in a segment is proportional to 1/p,
// so the smallest sieving prime has thousands of times more work than the
// largest. Giving every work-item its own prime (numSlices == 1) therefore
// leaves the whole work-group waiting for the few items that drew the
// smallest primes. Splitting a small prime across many work-items, each
// crossing off its own slice of the segment, removes that imbalance; it
// costs one extra setup per slice, which only pays while the prime is small
// enough to have several multiples per slice. The caller picks the cutoffs.
//
// A bonus of slicing is that every work-item in a warp then walks the *same*
// prime, so the trip counts are equal and the warp does not diverge.
//
inline void psCrossOffRange(volatile __local uint* sieve,
                            __local const uint* wheel,
                            __local const uint* wheelInit,
                            __local const uint* wheelOffset,
                            __global const uint* primes,
                            __global const ulong* magics,
                            uint useMagic,
                            uint begin,
                            uint end,
                            uint primeStride,
                            uint groupId,
                            uint numSlices,
                            uint sliceId,
                            uint sieveBytes,
                            ulong segLow)
{
  uint sliceBytes = sieveBytes / numSlices;
  uint sliceStart = sliceId * sliceBytes;

  // Bit b of byte i stands for segmentLow + i*30 + {7,...,31}[b], so the
  // multiples wanted are those greater than the slice's base + 6.
  ulong low = segLow + (ulong) sliceStart * 30 + 6;

  for (uint i = begin + groupId; i < end; i += primeStride)
  {
    uint p = primes[i];

    // First multiple of p that is > low, rounded up to a multiple of p whose
    // cofactor is coprime to 30 (primesieve's Wheel::addSievingPrime).
    //
    // This division runs once per sieving prime per slice, and a 64-bit
    // division is emulated in ~70 instructions on a GPU, so use the
    // precomputed reciprocal whenever the identity holds (low < 2^63).
    ulong q;
    if (useMagic)
      q = (mul_hi(low, magics[i]) >> (31u - clz(p))) + 1;
    else
      q = low / p + 1;

    if (q < (ulong) p)
      q = (ulong) p;

    ulong m   = (ulong) p * q;
    uint  ini = wheelInit[(uint)(q % 30)];
    m += (ulong) p * (ini & 0xffu);

    if (m < low)
      continue;

    ulong byteIdx64 = (m - low) / 30;
    if (byteIdx64 >= (ulong) sliceBytes)
      continue;

    uint byteIdx = (uint) byteIdx64;
    uint state   = wheelOffset[p % 30] + (ini >> 8);
    uint pdiv30  = p / 30;

    // pdiv30 * nextMultipleFactor <= (2^32/30)*6 < 2^30, so byteIdx cannot
    // wrap around before the loop condition stops it.
    while (byteIdx < sliceBytes)
    {
      uint e = wheel[state];
      psClearBit(sieve, sliceStart + byteIdx, e & 0xffu);
      byteIdx += pdiv30 * ((e >> 8) & 0xffu) + ((e >> 16) & 0xffu);
      state    = e >> 24;
    }
  }
}

// Zero the bits of `word` whose values fall outside [rangeLo, rangeHi].
// `wordBase` is segmentLow + (wordIndex*4)*30, i.e. 7 below the smallest
// value the word can represent.
inline uint psMaskWord(uint word, ulong wordBase, ulong rangeLo, ulong rangeHi)
{
  ulong lo = wordBase + 7;         // smallest value in this word
  ulong hi = wordBase + 3*30 + 31; // largest value in this word

  if (lo >= rangeLo && hi <= rangeHi)
    return word;
  if (hi < rangeLo || lo > rangeHi)
    return 0u;

  uint out = 0u;
  for (uint b = 0; b < 32u; b++)
  {
    if (!((word >> b) & 1u))
      continue;
    ulong v = wordBase + (ulong)(b >> 3) * 30 + psBitOffset[b & 7u];
    if (v >= rangeLo && v <= rangeHi)
      out |= 1u << b;
  }
  return out;
}

// Sum `value` across the work-group; the result is valid in work-item 0.
// Requires a power-of-two work-group size (enforced by the host).
inline ulong psReduce(__local ulong* scratch, ulong value, uint lid, uint lsz)
{
  scratch[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = lsz >> 1; s > 0; s >>= 1)
  {
    if (lid < s)
      scratch[lid] += scratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return scratch[0];
}

//
// countType 0    -> count primes (popcount of the sieve)
// countType 1..5 -> count twins, triplets, quadruplets, quintuplets and
//                   sextuplets via the per-byte lookup table kTable.
//
// tier1End / tier2End split the sieving primes into the three groups
// described above psCrossOffRange():
//   [firstPrime, tier1End)  one prime at a time, sliced across the group
//   [tier1End,   tier2End)  one prime per warp, sliced across the warp
//   [tier2End,   numPrimes) one prime per work-item, whole segment
//
__kernel void ps_count(const ulong segLowBase,
                       const uint  sieveBytes,
                       const uint  numSegments,
                       const ulong rangeLo,
                       const ulong rangeHi,
                       __global const uint*  primes,
                       __global const ulong* magics,
                       const uint  useMagic,
                       const uint  numPrimes,
                       const uint  firstPrime,
                       const uint  tier1End,
                       const uint  tier2End,
                       __global const uint*  presieve,
                       const uint  presieveTables,
                       __global const uint*  presieveLen,
                       __global const uint*  presieveOff,
                       __global const uchar* restore,
                       const uint  restoreLen,
                       const uint  countType,
                       __global const uchar* kTable,
                       __global ulong* counts,
                       __local  uint*  sieve,
                       __local  ulong* scratch)
{
  uint seg = get_group_id(0);
  if (seg >= numSegments)
    return;

  uint lid   = get_local_id(0);
  uint lsz   = get_local_size(0);
  uint words = sieveBytes >> 2;

  // The cross-off loop indexes the wheel table with a per-prime state, so the
  // accesses inside a warp are divergent. NVIDIA serialises divergent
  // __constant reads up to 32 ways per step; a local-memory copy is banked
  // and handles divergent indices far better.
  __local uint wheelLocal[64];
  __local uint wheelInitLocal[30];
  __local uint wheelOffsetLocal[30];
  for (uint i = lid; i < 64; i += lsz)
    wheelLocal[i] = psWheel[i];
  for (uint i = lid; i < 30; i += lsz)
  {
    wheelInitLocal[i]   = psWheelInit[i];
    wheelOffsetLocal[i] = psWheelOffset[i];
  }

  ulong span   = (ulong) sieveBytes * 30;
  ulong segLow = segLowBase + (ulong) seg * span;

  // ---- 1. initialise the sieve ------------------------------------------
  // Without pre-sieving every bit starts set. With pre-sieving the multiples
  // of the small primes covered by the tables are already removed, which
  // saves far more work than the extra (L2 cached) loads cost.
  if (presieveTables == 0)
  {
    for (uint i = lid; i < words; i += lsz)
      sieve[i] = 0xffffffffu;
  }
  else
  {
    // One pass per pattern table, so that the expensive modulo runs once per
    // (table, work-item) instead of once per word. Each work-item owns a
    // fixed, strided subset of the words, so the read-modify-write below
    // needs no atomics.
    ulong baseByte = segLow / 30;

    for (uint t = 0; t < presieveTables; t++)
    {
      uint len  = presieveLen[t];
      uint off  = presieveOff[t];
      uint idx  = (uint)((baseByte + (ulong)(lid << 2)) % len);
      uint step = (uint)(((ulong)(lsz << 2)) % len);

      for (uint i = lid; i < words; i += lsz)
      {
        // The table is stored as a sliding window, so the four pattern bytes
        // of this sieve word are a single coalesced load.
        uint v = presieve[off + idx];

        if (t == 0)
          sieve[i] = v;
        else
          sieve[i] &= v;

        idx += step;
        if (idx >= len) idx -= len;
      }
    }
  }

  // Pre-sieving also removes the small primes themselves (7, 11, 13, ...).
  // Put them back, exactly like primesieve's PreSieve::primeBits does on the
  // CPU. Only the very first bytes of the whole range are affected, and word
  // 0 is written by work-item 0 in the loop above, so no barrier is needed.
  if (restoreLen > 0 && lid == 0)
  {
    ulong baseByte = segLow / 30;
    if (baseByte < (ulong) restoreLen)
    {
      __local uchar* sieve8 = (__local uchar*) sieve;
      uint n = (uint) ((ulong) restoreLen - baseByte);
      if (n > sieveBytes)
        n = sieveBytes;
      for (uint j = 0; j < n; j++)
        sieve8[j] = restore[(uint) baseByte + j];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- 2. cross off the sieving primes ----------------------------------
#if !defined(PS_NO_CROSSOFF)
  uint lane    = lid & 31u;
  uint warp    = lid >> 5;
  uint nwarps  = lsz >> 5;
  if (nwarps == 0)
    nwarps = 1;

  // Smallest primes: the whole work-group walks one prime at a time, each
  // work-item over its own slice of the segment.
  psCrossOffRange(sieve, wheelLocal, wheelInitLocal, wheelOffsetLocal,
                  primes, magics, useMagic,
                  firstPrime, tier1End,
                  /* primeStride */ 1u, /* groupId */ 0u,
                  /* numSlices */ lsz, /* sliceId */ lid,
                  sieveBytes, segLow);

  // Medium primes: one prime per warp, sliced across the 32 lanes.
  psCrossOffRange(sieve, wheelLocal, wheelInitLocal, wheelOffsetLocal,
                  primes, magics, useMagic,
                  tier1End, tier2End,
                  /* primeStride */ nwarps, /* groupId */ warp,
                  /* numSlices */ 32u, /* sliceId */ lane,
                  sieveBytes, segLow);

  // Large primes: too few multiples per segment to be worth slicing.
  psCrossOffRange(sieve, wheelLocal, wheelInitLocal, wheelOffsetLocal,
                  primes, magics, useMagic,
                  tier2End, numPrimes,
                  /* primeStride */ lsz, /* groupId */ lid,
                  /* numSlices */ 1u, /* sliceId */ 0u,
                  sieveBytes, segLow);
#endif

  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- 3. mask the range boundaries and count ---------------------------
  ulong mine = 0;

  if (countType == 0)
  {
    for (uint i = lid; i < words; i += lsz)
    {
      ulong wordBase = segLow + (ulong) i * 120;  // 4 bytes * 30 numbers
      mine += popcount(psMaskWord(sieve[i], wordBase, rangeLo, rangeHi));
    }
  }
  else
  {
    for (uint i = lid; i < words; i += lsz)
    {
      ulong wordBase = segLow + (ulong) i * 120;
      uint w = psMaskWord(sieve[i], wordBase, rangeLo, rangeHi);
      // k-tuplets never straddle a byte boundary in this bit layout, so a
      // per-byte lookup table is exact.
      mine += kTable[(w      ) & 0xffu];
      mine += kTable[(w >>  8) & 0xffu];
      mine += kTable[(w >> 16) & 0xffu];
      mine += kTable[(w >> 24) & 0xffu];
    }
  }

  ulong total = psReduce(scratch, mine, lid, lsz);
  if (lid == 0)
    counts[seg] = total;
}

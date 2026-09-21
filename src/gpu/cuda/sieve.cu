//
// primesieve GPU backend -- CUDA sieve of Eratosthenes.
//
// This is the same algorithm as src/gpu/opencl/sieve.cl, written against
// CUDA instead of OpenCL: one block sieves one segment into shared memory
// using primesieve's wheel-30 bit layout (8 bits per 30 numbers, bit b of
// byte i standing for segmentLow + i*30 + {7,11,13,17,19,23,29,31}[b]),
// counts it, and writes back one count per segment.
//
// The host prepends the wheel tables, generated from src/gpu/GpuWheel.hpp,
// and compiles this with NVRTC at runtime.
//
// CUDA gives a block a single dynamic shared memory allocation, so the host
// lays the pieces out end to end and this file carves them up:
//
//     [ scratch: blockDim.x * unsigned long long ]   (8-byte aligned first)
//     [ wheel: 64 ][ wheelInit: 30 ][ wheelOffset: 30 ]
//     [ sieve: sieveBytes / 4 ]
//
// This file is distributed under the BSD License. See the COPYING
// file in the top level directory.
//

typedef unsigned int  u32;
typedef unsigned long long u64;

// Clear one bit of the shared sieve. Several threads may target the same
// 32-bit word concurrently, so the read-modify-write must be atomic --
// a plain load/store would lose updates and leave composites marked prime.
__device__ __forceinline__ void psClearBit(u32* sieve, u32 byteIdx, u32 unsetBit)
{
  u32 word  = byteIdx >> 2;
  u32 shift = (byteIdx & 3u) << 3;
  atomicAnd(&sieve[word], ~((0xffu ^ unsetBit) << shift));
}

//
// Cross off the multiples of the sieving primes in [begin, end).
//
// The work is split along two axes at once: which threads share a prime
// (primeStride / groupId) and which slice of the segment each owns
// (numSlices / sliceId). A prime has about 1/p of the segment's multiples,
// so giving every thread its own prime leaves the block waiting on whoever
// drew the smallest prime. Slicing a small prime across many threads costs
// one extra setup per slice and removes that imbalance; it also makes every
// thread in a warp walk the same prime, so the warp does not diverge.
//
__device__ void psCrossOffRange(u32* sieve,
                                const u32* wheel,
                                const u32* wheelInit,
                                const u32* wheelOffset,
                                const u32* primes,
                                const u64* magics,
                                u32 useMagic,
                                u32 begin,
                                u32 end,
                                u32 primeStride,
                                u32 groupId,
                                u32 numSlices,
                                u32 sliceId,
                                u32 sieveBytes,
                                u64 segLow)
{
  u32 sliceBytes = sieveBytes / numSlices;
  u32 sliceStart = sliceId * sliceBytes;

  // Bit b of byte i stands for segmentLow + i*30 + {7,...,31}[b], so the
  // multiples wanted are those greater than the slice's base + 6.
  u64 low = segLow + (u64) sliceStart * 30 + 6;

  for (u32 i = begin + groupId; i < end; i += primeStride)
  {
    u32 p = primes[i];

    // First multiple of p that is > low, rounded up to a multiple of p whose
    // cofactor is coprime to 30 (primesieve's Wheel::addSievingPrime).
    //
    // A 64-bit division is emulated in ~70 instructions, and this one runs
    // once per sieving prime per slice, so use the precomputed reciprocal
    // whenever the identity holds (low < 2^63).
    u64 q;
    if (useMagic)
      q = (__umul64hi(low, magics[i]) >> (31u - __clz((int) p))) + 1;
    else
      q = low / p + 1;

    if (q < (u64) p)
      q = (u64) p;

    u64 m   = (u64) p * q;
    u32 ini = wheelInit[(u32)(q % 30)];
    m += (u64) p * (ini & 0xffu);

    if (m < low)
      continue;

    u64 byteIdx64 = (m - low) / 30;
    if (byteIdx64 >= (u64) sliceBytes)
      continue;

    u32 byteIdx = (u32) byteIdx64;
    u32 state   = wheelOffset[p % 30] + (ini >> 8);
    u32 pdiv30  = p / 30;

    // pdiv30 * nextMultipleFactor <= (2^32/30)*6 < 2^30, so byteIdx cannot
    // wrap around before the loop condition stops it.
    while (byteIdx < sliceBytes)
    {
      u32 e = wheel[state];
      psClearBit(sieve, sliceStart + byteIdx, e & 0xffu);
      byteIdx += pdiv30 * ((e >> 8) & 0xffu) + ((e >> 16) & 0xffu);
      state    = e >> 24;
    }
  }
}

// Zero the bits of `word` whose values fall outside [rangeLo, rangeHi].
// `wordBase` is segmentLow + (wordIndex*4)*30.
__device__ __forceinline__ u32 psMaskWord(u32 word, u64 wordBase, u64 rangeLo, u64 rangeHi)
{
  u64 lo = wordBase + 7;
  u64 hi = wordBase + 3*30 + 31;

  if (lo >= rangeLo && hi <= rangeHi)
    return word;
  if (hi < rangeLo || lo > rangeHi)
    return 0u;

  u32 out = 0u;
  for (u32 b = 0; b < 32u; b++)
  {
    if (!((word >> b) & 1u))
      continue;
    u64 v = wordBase + (u64)(b >> 3) * 30 + psBitOffset[b & 7u];
    if (v >= rangeLo && v <= rangeHi)
      out |= 1u << b;
  }
  return out;
}

// Sum `value` across the block; the result is valid in thread 0.
// Requires a power-of-two block size (enforced by the host).
__device__ u64 psReduce(u64* scratch, u64 value, u32 tid, u32 nthreads)
{
  scratch[tid] = value;
  __syncthreads();

  for (u32 s = nthreads >> 1; s > 0; s >>= 1)
  {
    if (tid < s)
      scratch[tid] += scratch[tid + s];
    __syncthreads();
  }
  return scratch[0];
}

extern "C" __global__ void ps_count(const u64 segLowBase,
                                    const u32 sieveBytes,
                                    const u32 numSegments,
                                    const u64 rangeLo,
                                    const u64 rangeHi,
                                    const u32* __restrict__ primes,
                                    const u64* __restrict__ magics,
                                    const u32 useMagic,
                                    const u32 numPrimes,
                                    const u32 firstPrime,
                                    const u32 tier1End,
                                    const u32 tier2End,
                                    const u32* __restrict__ presieve,
                                    const u32 presieveTables,
                                    const u32* __restrict__ presieveLen,
                                    const u32* __restrict__ presieveOff,
                                    const unsigned char* __restrict__ restore,
                                    const u32 restoreLen,
                                    const u32 countType,
                                    const unsigned char* __restrict__ kTable,
                                    u64* __restrict__ counts)
{
  u32 seg = blockIdx.x;
  if (seg >= numSegments)
    return;

  u32 tid      = threadIdx.x;
  u32 nthreads = blockDim.x;
  u32 words    = sieveBytes >> 2;

  extern __shared__ u64 psShared[];
  u64* scratch     = psShared;
  u32* tables      = (u32*) (psShared + nthreads);
  u32* wheel       = tables;
  u32* wheelInit   = tables + 64;
  u32* wheelOffset = tables + 64 + 30;
  u32* sieve       = tables + 64 + 30 + 30;

  // The cross-off loop indexes the wheel table with a per-prime state, so the
  // accesses inside a warp are divergent. Divergent __constant__ reads are
  // serialised up to 32 ways per step; a shared-memory copy is banked and
  // handles divergent indices far better.
  for (u32 i = tid; i < 64; i += nthreads)
    wheel[i] = psWheel[i];
  for (u32 i = tid; i < 30; i += nthreads)
  {
    wheelInit[i]   = psWheelInit[i];
    wheelOffset[i] = psWheelOffset[i];
  }

  u64 span   = (u64) sieveBytes * 30;
  u64 segLow = segLowBase + (u64) seg * span;

  // ---- 1. initialise the sieve ------------------------------------------
  if (presieveTables == 0)
  {
    for (u32 i = tid; i < words; i += nthreads)
      sieve[i] = 0xffffffffu;
  }
  else
  {
    // One pass per pattern table, so the expensive modulo runs once per
    // (table, thread) instead of once per word. Each thread owns a fixed,
    // strided subset of the words, so this needs no atomics.
    u64 baseByte = segLow / 30;

    for (u32 t = 0; t < presieveTables; t++)
    {
      u32 len  = presieveLen[t];
      u32 off  = presieveOff[t];
      u32 idx  = (u32)((baseByte + (u64)(tid << 2)) % len);
      u32 step = (u32)(((u64)(nthreads << 2)) % len);

      for (u32 i = tid; i < words; i += nthreads)
      {
        // The table is stored as a sliding window, so the four pattern bytes
        // of this sieve word are a single coalesced load.
        u32 v = presieve[off + idx];

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
  // CPU. Only the first bytes of the whole range are affected, and word 0 is
  // written by thread 0 above, so no barrier is needed here.
  if (restoreLen > 0 && tid == 0)
  {
    u64 baseByte = segLow / 30;
    if (baseByte < (u64) restoreLen)
    {
      unsigned char* sieve8 = (unsigned char*) sieve;
      u32 n = (u32)((u64) restoreLen - baseByte);
      if (n > sieveBytes)
        n = sieveBytes;
      for (u32 j = 0; j < n; j++)
        sieve8[j] = restore[(u32) baseByte + j];
    }
  }

  __syncthreads();

  // ---- 2. cross off the sieving primes ----------------------------------
  {
    u32 lane   = tid & 31u;
    u32 warp   = tid >> 5;
    u32 nwarps = nthreads >> 5;
    if (nwarps == 0)
      nwarps = 1;

    // Smallest primes: the whole block walks one prime at a time, each
    // thread over its own slice of the segment.
    psCrossOffRange(sieve, wheel, wheelInit, wheelOffset, primes, magics, useMagic,
                    firstPrime, tier1End, 1u, 0u, nthreads, tid, sieveBytes, segLow);

    // Medium primes: one prime per warp, sliced across the 32 lanes.
    psCrossOffRange(sieve, wheel, wheelInit, wheelOffset, primes, magics, useMagic,
                    tier1End, tier2End, nwarps, warp, 32u, lane, sieveBytes, segLow);

    // Large primes: too few multiples per segment to be worth slicing.
    psCrossOffRange(sieve, wheel, wheelInit, wheelOffset, primes, magics, useMagic,
                    tier2End, numPrimes, nthreads, tid, 1u, 0u, sieveBytes, segLow);
  }

  __syncthreads();

  // ---- 3. mask the range boundaries and count ---------------------------
  u64 mine = 0;

  if (countType == 0)
  {
    for (u32 i = tid; i < words; i += nthreads)
    {
      u64 wordBase = segLow + (u64) i * 120;   // 4 bytes * 30 numbers
      mine += __popc(psMaskWord(sieve[i], wordBase, rangeLo, rangeHi));
    }
  }
  else
  {
    for (u32 i = tid; i < words; i += nthreads)
    {
      u64 wordBase = segLow + (u64) i * 120;
      u32 w = psMaskWord(sieve[i], wordBase, rangeLo, rangeHi);
      // k-tuplets never straddle a byte boundary in this bit layout, so a
      // per-byte lookup table is exact.
      mine += kTable[(w      ) & 0xffu];
      mine += kTable[(w >>  8) & 0xffu];
      mine += kTable[(w >> 16) & 0xffu];
      mine += kTable[(w >> 24) & 0xffu];
    }
  }

  u64 total = psReduce(scratch, mine, tid, nthreads);
  if (tid == 0)
    counts[seg] = total;
}

// Host reference implementation of the GPU sieve algorithm.
// This runs, on the CPU, exactly the algorithm that the OpenCL/CUDA kernels
// run, so that the wheel tables and the segment arithmetic can be validated
// against known values of pi(x) before any GPU code is involved.
#include "../src/gpu/GpuWheel.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cassert>

using namespace primesieve::gpu;
typedef uint64_t u64; typedef uint32_t u32; typedef uint8_t u8;

static int popcount8(u8 x){ int c=0; while(x){c+=x&1;x>>=1;} return c; }

// Simple CPU sieve producing every prime in [2, n]; used for sieving primes.
static std::vector<u32> simplePrimes(u32 n)
{
  std::vector<u8> is(n + 1, 1);
  std::vector<u32> out;
  for (u32 i = 2; (u64)i * i <= n; i++) if (is[i]) for (u64 j = (u64)i*i; j <= n; j += i) is[j] = 0;
  for (u32 i = 2; i <= n; i++) if (is[i]) out.push_back(i);
  return out;
}

// Count primes in [start, stop] using the GPU algorithm on the CPU.
static u64 refCount(u64 start, u64 stop, u32 sieveBytes)
{
  if (stop < 2) return 0;
  u64 count = 0;
  // primes 2,3,5 are not represented in the wheel-30 bit array
  for (u64 p : { 2ull, 3ull, 5ull }) if (p >= start && p <= stop) count++;
  u64 lo = start < 7 ? 7 : start;
  if (lo > stop) return count;

  u32 sqrtStop = (u32)[](u64 x){ u64 r = (u64)sqrtl((long double)x); while (r*r > x) r--; while ((r+1)*(r+1) <= x) r++; return r; }(stop);
  std::vector<u32> sp = simplePrimes(sqrtStop < 7 ? 7 : sqrtStop);

  const u64 span = (u64)sieveBytes * 30;
  std::vector<u8> sieve(sieveBytes);

  // The bit array uses the offsets {7,11,13,17,19,23,29,31} relative to
  // segmentLow, so a value v == 1 (mod 30) lives in the *previous* byte
  // (as segmentLow + 31). segmentLow must therefore be derived with
  // primesieve's byteRemainder(), i.e. equivalence classes 7..36.
  u64 firstSegLow = lo - ((lo - 7) % 30 + 7);
  for (u64 segLow = firstSegLow; segLow <= stop; segLow += span)
  {
    u64 low = segLow + 6;
    u64 segHigh = segLow + span;            // exclusive-ish upper bound
    memset(sieve.data(), 0xff, sieveBytes);

    for (u32 p : sp)
    {
      if (p < 7) continue;
      if ((u64)p * p > stop) break;
      u64 q = low / p + 1;
      if (q < p) q = p;
      u64 m = (u64)p * q;
      GpuWheelInit ini = gpuWheel30Init[q % 30];
      m += (u64)p * ini.nextMultipleFactor;
      if (m < low) continue;
      u64 byteIdx = (m - low) / 30;
      u32 state = gpuWheelOffsets[p % 30] + ini.wheelIndex;
      u32 pdiv30 = p / 30;
      while (byteIdx < sieveBytes)
      {
        GpuWheelElement e = gpuWheel30[state];
        sieve[byteIdx] &= e.unsetBit;
        byteIdx += (u64)pdiv30 * e.nextMultipleFactor + e.correct;
        state = e.next;
      }
    }

    // mask off values outside [start, stop] and count
    for (u32 i = 0; i < sieveBytes; i++)
    {
      u8 b = sieve[i];
      while (b)
      {
        int bit = 0; u8 t = b; while (!(t & 1)) { t >>= 1; bit++; }
        b &= (u8)(b - 1);
        u64 v = segLow + (u64)i * 30 + gpuBitOffset[bit];
        if (v >= lo && v <= stop) count++;
      }
    }
    if (segHigh < segLow) break; // overflow guard
  }
  return count;
}

#include <cmath>

// Independent, deliberately naive segmented sieve used as an oracle.
static u64 oracleCount(u64 start, u64 stop)
{
  if (stop < 2) return 0;
  if (start < 2) start = 2;
  if (start > stop) return 0;
  u64 n = (u64)sqrtl((long double)stop) + 2;
  while (n * n > stop + 1) n--;
  std::vector<u32> sp = simplePrimes((u32)(n + 2));
  u64 len = stop - start + 1;
  std::vector<u8> is((size_t)len, 1);
  for (u32 p : sp) {
    if ((u64)p * p > stop) break;
    u64 first = ((start + p - 1) / p) * p;
    if (first < (u64)p * p) first = (u64)p * p;
    for (u64 m = first; m <= stop; m += p) is[(size_t)(m - start)] = 0;
  }
  u64 c = 0;
  for (u64 i = 0; i < len; i++) if (is[(size_t)i] && (start + i) >= 2) c++;
  return c;
}

static int randomizedTests()
{
  printf("\n-- randomized differential tests vs independent oracle --\n");
  unsigned long long seed = 0x9E3779B97F4A7C15ull;
  auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
  const u64 bases[] = { 0ull, 1000ull, 1000000ull, 1000000000ull, 100000000000ull, 10000000000000ull };
  int fails = 0, n = 0;
  for (u64 base : bases)
    for (int i = 0; i < 25; i++)
    {
      u64 start = base + (rnd() % 100000);
      u64 stop  = start + (rnd() % 60000);
      u64 want = oracleCount(start, stop);
      for (u32 sb : { 32u, 977u, 4096u })
      {
        u64 got = refCount(start, stop, sb);
        n++;
        if (got != want) { printf("FAIL [%llu,%llu] sb=%u got=%llu want=%llu\n",
             (unsigned long long)start,(unsigned long long)stop, sb,
             (unsigned long long)got,(unsigned long long)want); fails++; }
      }
    }
  printf("%d randomized range checks, %d failures\n", n, fails);
  return fails;
}

int main(int argc, char** argv)
{
  struct { u64 start, stop; u64 expect; const char* name; } tests[] = {
    { 0, 10, 4, "pi(10)" },
    { 0, 100, 25, "pi(100)" },
    { 0, 1000, 168, "pi(1000)" },
    { 0, 10000, 1229, "pi(1e4)" },
    { 0, 100000, 9592, "pi(1e5)" },
    { 0, 1000000, 78498, "pi(1e6)" },
    { 0, 10000000, 664579, "pi(1e7)" },
    { 0, 100000000, 5761455, "pi(1e8)" },
    { 1, 2, 1, "pi([1,2])" },
    { 2, 2, 1, "pi([2,2])" },
    { 3, 5, 2, "pi([3,5])" },
    { 5, 7, 2, "pi([5,7])" },
    { 6, 6, 0, "pi([6,6])" },
    { 7, 7, 1, "pi([7,7])" },
    { 1000000000ull, 1000001000ull, 49, "pi([1e9,1e9+1000])" },
    { 999999999989ull, 1000000000039ull, 2, "pi(1e12 nbhd)" },
    { 999999999000ull, 1000000001000ull, 75, "pi(1e12 +-1000)" },
  };
  int fails = 0;
  for (auto& t : tests)
  {
    for (u32 sb : { 32u, 1000u, 4096u, 32768u })
    {
      u64 got = refCount(t.start, t.stop, sb);
      if (got != t.expect) { printf("FAIL %-22s sieveBytes=%-6u got=%llu expect=%llu\n", t.name, sb, (unsigned long long)got, (unsigned long long)t.expect); fails++; }
    }
    printf("ok   %-22s = %llu\n", t.name, (unsigned long long)t.expect);
  }
  fails += randomizedTests();
  printf(fails ? "\n%d FAILURES\n" : "\nall reference tests passed\n", fails);
  return fails != 0;
}

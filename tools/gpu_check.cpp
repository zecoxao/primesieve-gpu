// Development driver: lists the GPU devices and checks every GPU count
// against primesieve's CPU sieve, which is the reference implementation.
//
//   gpu_check              run the correctness suite
//   gpu_check bench        run the benchmark suite
//   gpu_check dev <n>      select GPU device n

#include <primesieve.hpp>
#include <primesieve/gpu.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

double now()
{
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

const char* typeName(int t)
{
  static const char* names[] = { "primes", "twins", "triplets",
                                 "quadruplets", "quintuplets", "sextuplets" };
  return names[t];
}

uint64_t cpuCount(uint64_t a, uint64_t b, int type)
{
  switch (type)
  {
    case 0: return primesieve::count_primes(a, b);
    case 1: return primesieve::count_twins(a, b);
    case 2: return primesieve::count_triplets(a, b);
    case 3: return primesieve::count_quadruplets(a, b);
    case 4: return primesieve::count_quintuplets(a, b);
    default: return primesieve::count_sextuplets(a, b);
  }
}

uint64_t gpuCount(uint64_t a, uint64_t b, int type)
{
  switch (type)
  {
    case 0: return primesieve::gpu_count_primes(a, b);
    case 1: return primesieve::gpu_count_twins(a, b);
    case 2: return primesieve::gpu_count_triplets(a, b);
    case 3: return primesieve::gpu_count_quadruplets(a, b);
    case 4: return primesieve::gpu_count_quintuplets(a, b);
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
    std::printf("  FAIL %-12s [%llu, %llu]  gpu=%llu cpu=%llu  (diff %+lld)\n",
                typeName(type),
                (unsigned long long) a, (unsigned long long) b,
                (unsigned long long) got, (unsigned long long) want,
                (long long) (got - want));
    failures++;
  }
}

void correctness()
{
  std::printf("\n== small ranges, every count type ==\n");
  const uint64_t smallStops[] = { 0, 1, 2, 3, 5, 6, 7, 8, 30, 31, 49, 100, 163,
                                  164, 241, 1000, 65537, 1000000 };
  for (int type = 0; type <= 5; type++)
    for (std::size_t i = 0; i < sizeof(smallStops) / sizeof(smallStops[0]); i++)
      check(0, smallStops[i], type);
  std::printf("  %d checks done\n", checks);

  std::printf("\n== ranges that start at awkward offsets ==\n");
  // Values == 1 (mod 30) live in the previous byte of the bit array; these
  // ranges pin down that boundary.
  for (uint64_t base = 0; base <= 120; base++)
    for (uint64_t len = 0; len <= 70; len += 7)
      check(base, base + len, 0);
  for (uint64_t start = 999999900ull; start <= 1000000100ull; start++)
    check(start, start + 331, 0);

  std::printf("\n== larger ranges ==\n");
  const uint64_t ranges[][2] = {
    { 0,               10000000ull },
    { 0,              100000000ull },
    { 0,             1000000000ull },
    { 1000000000ull, 1100000000ull },
    { 999999999989ull, 1000000200000ull },
    { 10000000000000ull, 10000010000000ull }
  };
  for (std::size_t i = 0; i < sizeof(ranges) / sizeof(ranges[0]); i++)
    for (int type = 0; type <= 5; type++)
      check(ranges[i][0], ranges[i][1], type);

  std::printf("\n== pi(x) against known values ==\n");
  const struct { uint64_t stop; uint64_t pi; } known[] = {
    { 1000000ull,        78498ull },
    { 10000000ull,       664579ull },
    { 100000000ull,      5761455ull },
    { 1000000000ull,     50847534ull },
    { 10000000000ull,    455052511ull }
  };
  for (std::size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++)
  {
    uint64_t got = primesieve::gpu_count_primes(0, known[i].stop);
    checks++;
    std::printf("  pi(%-13llu) = %-12llu %s\n",
                (unsigned long long) known[i].stop, (unsigned long long) got,
                got == known[i].pi ? "ok" : "FAIL");
    if (got != known[i].pi)
      failures++;
  }
}

void bench()
{
  std::printf("\n== benchmark (GPU vs CPU) ==\n");
  std::printf("%-26s %12s %12s %10s\n", "range", "gpu (s)", "cpu (s)", "speedup");

  const uint64_t ranges[][2] = {
    { 0,   1000000000ull },
    { 0,  10000000000ull },
    { 0, 100000000000ull },
    { 1000000000000ull, 1010000000000ull }
  };

  for (std::size_t i = 0; i < sizeof(ranges) / sizeof(ranges[0]); i++)
  {
    uint64_t a = ranges[i][0], b = ranges[i][1];

    double t0 = now();
    uint64_t g = primesieve::gpu_count_primes(a, b);
    double tg = now() - t0;

    t0 = now();
    uint64_t c = primesieve::count_primes(a, b);
    double tc = now() - t0;

    char label[64];
    std::snprintf(label, sizeof label, "[%.3g, %.3g]", (double) a, (double) b);
    std::printf("%-26s %12.3f %12.3f %9.2fx %s\n", label, tg, tc, tc / tg,
                (g == c) ? "" : "MISMATCH!");
    if (g != c)
      failures++;
  }
}


void tune()
{
  const uint64_t a = 0, b = 10000000000ull;   // 1e10
  std::printf("\n== geometry sweep, counting primes in [0, 1e10] ==\n");
  std::printf("%10s %6s %8s %10s\n", "sieveKiB", "wg", "segs", "time (s)");

  const int sieveKiB[] = { 4, 8, 16, 32, 46 };
  const int wgs[]      = { 64, 128, 256, 512 };
  const int segs[]     = { 256, 1024, 4096 };

  double best = 1e30; int bs = 0, bw = 0, bg = 0;

  for (std::size_t i = 0; i < sizeof(sieveKiB)/sizeof(sieveKiB[0]); i++)
    for (std::size_t j = 0; j < sizeof(wgs)/sizeof(wgs[0]); j++)
      for (std::size_t k = 0; k < sizeof(segs)/sizeof(segs[0]); k++)
      {
        primesieve::set_gpu_sieve_size(sieveKiB[i]);
        primesieve::set_gpu_work_group_size(wgs[j]);
        primesieve::set_gpu_segments_per_launch(segs[k]);
        try
        {
          double t0 = now();
          uint64_t got = primesieve::gpu_count_primes(a, b);
          double dt = now() - t0;
          bool ok = (got == 455052511ull);
          std::printf("%10d %6d %8d %10.3f %s\n", sieveKiB[i], wgs[j], segs[k], dt,
                      ok ? "" : "WRONG");
          if (ok && dt < best) { best = dt; bs = sieveKiB[i]; bw = wgs[j]; bg = segs[k]; }
        }
        catch (const std::exception& e)
        {
          std::printf("%10d %6d %8d %10s %s\n", sieveKiB[i], wgs[j], segs[k], "-", e.what());
        }
      }

  std::printf("\nbest: sieve=%d KiB wg=%d segs=%d -> %.3f s\n", bs, bw, bg, best);
}

} // namespace

int main(int argc, char** argv)
{
  std::vector<std::string> devices = primesieve::gpu_devices();
  std::printf("GPU devices:\n");
  for (std::size_t i = 0; i < devices.size(); i++)
    std::printf("  %s\n", devices[i].c_str());

  if (devices.empty())
  {
    std::printf("no GPU devices found: %s\n", primesieve::gpu_last_error().c_str());
    return 1;
  }

  bool doBench = false;
  bool doTune = false;
  for (int i = 1; i < argc; i++)
  {
    if (std::strcmp(argv[i], "bench") == 0)
      doBench = true;
    else if (std::strcmp(argv[i], "tune") == 0)
      doTune = true;
    else if (std::strcmp(argv[i], "dev") == 0 && i + 1 < argc)
      primesieve::set_gpu_device(std::atoi(argv[++i]));
  }

  std::printf("\nactive: %s\n", primesieve::gpu_active_device().c_str());
  if (primesieve::gpu_active_device().empty())
  {
    std::printf("could not initialise a GPU: %s\n", primesieve::gpu_last_error().c_str());
    return 1;
  }

  if (doTune)
    tune();
  else if (doBench)
    bench();
  else
    correctness();

  std::printf("\n%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}

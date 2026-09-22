# GPU support

primesieve can count primes and prime k-tuplets on a GPU. The GPU runs the
same segmented sieve of Eratosthenes as the CPU, with the same wheel-30 bit
array, so the results are identical — only more parallel.

There are two backends, OpenCL and CUDA. Both are optional, both are on by
default, and **neither needs an SDK to build**: they resolve the GPU runtime
at load time, so `libprimesieve` still builds and runs on machines that have
no GPU runtime at all.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

| CMake option | Default | Effect |
| ------------ | ------- | ------ |
| `WITH_OPENCL` | `ON`  | Build the OpenCL backend. Needs no OpenCL SDK; `OpenCL.dll` / `libOpenCL.so` is loaded at runtime. |
| `WITH_CUDA`   | `ON`  | Build the CUDA backend. Needs no CUDA toolkit; the driver and NVRTC are loaded at runtime. |

Turn both off with `-DWITH_OPENCL=OFF -DWITH_CUDA=OFF` for a CPU-only build.

## Command line

```bash
primesieve --gpu-info               # list the GPU devices
primesieve 1e11 --gpu               # count the primes below 10^11 on a GPU
primesieve 1e11 --gpu --gpu-device=2
primesieve 1e10 --count=23 --gpu    # twin primes and prime triplets
```

`--gpu` never changes the answer. When the range is too small to pay for a
kernel launch, when the requested work is printing rather than counting, or
when no GPU can be initialised, primesieve silently uses its CPU sieve.

## C++ API

```cpp
#include <primesieve.hpp>
#include <primesieve/gpu.hpp>

// Explicit: count on the GPU, throw if there is none.
uint64_t n = primesieve::gpu_count_primes(0, 1000000000000ull);

// Implicit: let the existing API use the GPU when it is worth it.
primesieve::set_gpu_enabled(true);
uint64_t m = primesieve::count_primes(0, 1000000000000ull);  // GPU
uint64_t k = primesieve::count_primes(0, 1000);              // CPU, too small
```

`gpu_devices()` lists the devices, `set_gpu_device(i)` picks one (`-1` = the
one with the most compute units), and `gpu_last_error()` explains any
failure. `set_gpu_sieve_size()`, `set_gpu_work_group_size()`,
`set_gpu_segments_per_launch()` and `set_gpu_presieve()` are tuning knobs;
`0` restores the auto-derived default.

## Runtime requirements

* **OpenCL**: any OpenCL 1.2+ runtime. On Linux this is usually
  `ocl-icd-libopencl1` plus a vendor ICD.
* **CUDA**: the NVIDIA driver (which provides `nvcuda`/`libcuda`) **and**
  NVRTC. NVRTC ships with the CUDA toolkit; it can also be installed on its
  own, e.g. `pip install nvidia-cuda-nvrtc-cu12`.

  NVRTC **must not be newer than the driver's CUDA version**, or the driver
  cannot JIT the PTX it emits. primesieve reports this case explicitly.
  `PRIMESIEVE_NVRTC_PATH` points the loader at a specific NVRTC library;
  its directory is added to the library search path too, because NVRTC
  loads its own `nvrtc-builtins` sibling while compiling.

`PRIMESIEVE_GPU_CL_OPTIONS` is appended to the OpenCL build options, for
tuning and profiling builds.

## How it works

One work-group (CUDA: one block) sieves one segment into local/shared memory
and returns a single count, so the sieve array never crosses the PCIe bus —
only one 64-bit count per segment comes back. The sieving primes go only up
to `sqrt(stop)` and are generated on the CPU by primesieve itself.

The bit layout is primesieve's own: 8 bits per 30 numbers, where bit `b` of
byte `i` represents `segmentLow + i*30 + {7,11,13,17,19,23,29,31}[b]`. The
wheel tables are generated from `src/gpu/GpuWheel.hpp` and textually
prepended to the kernel source, so the host and the device can never
disagree about them.

Three things matter most for speed, in the order they were found to matter:

1. **Cross-off load balance.** A prime has about `1/p` of a segment's
   multiples, so one prime per work-item leaves the whole group waiting on
   whoever drew the smallest prime — measured at roughly 20:1. Small primes
   are therefore walked by the whole work-group, each item over its own
   slice of the segment; medium primes by one warp; large primes by a single
   item. This also removes warp divergence, because every item in a warp
   then walks the same prime. Worth ~2x.
2. **Divergent table reads.** The cross-off loop indexes the wheel table
   with a per-prime state, and NVIDIA serialises divergent `__constant`
   reads up to 32 ways per step. A local-memory copy is banked. Worth
   ~1.6–2.6x.
3. **The per-prime division.** `low / p` runs once per sieving prime per
   slice, and a 64-bit division is emulated in ~70 instructions on a GPU.
   It is replaced by a precomputed reciprocal,
   `x / p == mul_hi(x, M) >> L` with `M = ceil(2^(64+L)/p)`, valid for
   `x < 2^63`; above that the kernel falls back to the real division.

Pre-sieving removes the multiples of every prime ≤ 53 by AND-ing four
precomputed patterns into the sieve as it is filled, which is far cheaper
than crossing those primes off one multiple at a time. The patterns are
stored as a sliding window so each sieve word is a single coalesced load.

## Limitations

* Counting only: `--print`, `nth_prime()` and `primesieve::iterator` always
  use the CPU sieve. The iterator's sequential, small-batch access pattern is
  a poor fit for a GPU.
* The GPU sieve keeps all the sieving primes resident, so counting near
  `2^64` needs `pi(2^32) ≈ 2×10^8` primes (~2.4 GB). Ranges that large are
  impractical on a small GPU.
* Above about `10^12` the CPU sieve closes the gap, because primesieve uses
  bucket sieving (`EratBig`) for primes larger than a segment while the GPU
  still visits every sieving prime for every segment. Adding a bucket phase
  is the obvious next step.

## Measured results

Counting primes, NVIDIA RTX 3050 Laptop (16 SMs, compute 8.6) against
primesieve's CPU sieve on 12 threads of an AMD Ryzen 5 5600H:

| range | CPU (12 threads) | GPU, CUDA | GPU, OpenCL | speedup |
| ----- | ---------------- | --------- | ----------- | ------- |
| `[0, 10^9]`            | 0.018 s | 0.010 s | 0.009 s | ~1.9x |
| `[0, 10^10]`           | 0.22 s  | 0.12 s  | 0.10 s  | ~2.0x |
| `[0, 10^11]`           | 2.71 s  | 1.27 s  | 1.38 s  | ~2.1x |
| `[10^12, 10^12+10^10]` | 0.35 s  | 0.23 s  | 0.22 s  | ~1.6x |

The same kernel also runs on the machine's integrated AMD GPU (gfx90c,
32 KiB local memory, max work-group 256) and produces identical counts.

## Correctness

`test/gpu_count.cpp` checks every GPU device on the machine against the CPU
sieve — all six count types, empty ranges, ranges below 7, the k-tuplets
containing 3 and 5, every starting residue mod 30, ranges crossing segment
boundaries, and known values of `pi(x)`. It runs as part of `ctest` and
skips cleanly when no GPU is present.

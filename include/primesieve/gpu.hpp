///
/// @file  gpu.hpp
/// @brief primesieve GPU API.
///
///        primesieve can offload prime counting to a GPU using either an
///        OpenCL or a CUDA backend. The GPU runs the same segmented sieve of
///        Eratosthenes as the CPU, with the same wheel-30 bit array, so the
///        results are identical -- only much more parallel.
///
///        The GPU is never used implicitly: either call the gpu_count_*()
///        functions directly, or switch the whole library over with
///        primesieve::set_gpu_enabled(true), after which count_primes() and
///        friends use the GPU whenever the range is large enough to pay for
///        the kernel launch, and fall back to the CPU otherwise.
///
/// Copyright (C) 2026 Kim Walisch, <kim.walisch@gmail.com>
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_HPP
#define PRIMESIEVE_GPU_HPP

#include <stdint.h>
#include <string>
#include <vector>

namespace primesieve {

/// True if at least one GPU device is usable.
bool gpu_available();

/// One human readable line per GPU device; the leading [n] is the index
/// accepted by set_gpu_device().
std::vector<std::string> gpu_devices();

/// Select the GPU device by index, or -1 for "the most capable device".
/// Takes effect on the next GPU call.
void set_gpu_device(int deviceIndex);
int  get_gpu_device();

/// Route count_primes() and the k-tuplet counts through the GPU when the
/// range is large enough. Disabled by default.
void set_gpu_enabled(bool enable);
bool get_gpu_enabled();

/// Tuning knobs. 0 restores the auto-derived default. Changing any of them
/// recreates the backend on the next GPU call.
void set_gpu_sieve_size(int kilobytes);        // local sieve per work-group
void set_gpu_work_group_size(int size);        // must be a power of two
void set_gpu_segments_per_launch(int segments);
void set_gpu_presieve(bool enable);

/// Description of the device actually in use, or "" if none could be
/// initialised (see gpu_last_error()).
std::string gpu_active_device();

/// Why the last GPU attempt failed, if it did.
std::string gpu_last_error();

/// Whether a range is big enough for the GPU to be worth using.
bool gpu_worthwhile(uint64_t start, uint64_t stop);

/// Count primes and prime k-tuplets on the GPU. These throw a
/// primesieve_error if no GPU is available; use gpu_available() first, or
/// set_gpu_enabled(true) together with the ordinary count_*() API to get an
/// automatic CPU fallback instead.
uint64_t gpu_count_primes(uint64_t start, uint64_t stop);
uint64_t gpu_count_twins(uint64_t start, uint64_t stop);
uint64_t gpu_count_triplets(uint64_t start, uint64_t stop);
uint64_t gpu_count_quadruplets(uint64_t start, uint64_t stop);
uint64_t gpu_count_quintuplets(uint64_t start, uint64_t stop);
uint64_t gpu_count_sextuplets(uint64_t start, uint64_t stop);

/// Used by count_primes() and the k-tuplet counts when the GPU has been
/// enabled with set_gpu_enabled(true). Returns false when the GPU was not
/// used -- because it is disabled, unavailable, or the range is too small to
/// pay for a kernel launch -- and the caller then sieves on the CPU. A GPU
/// failure is reported the same way, so a broken GPU never breaks a count.
bool gpu_try_count(uint64_t start, uint64_t stop, int countType, uint64_t& result);

} // namespace primesieve

#endif

///
/// @file  cuda_loader.cpp
/// @brief Runtime loader for the CUDA driver and the NVRTC kernel compiler.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "cuda_loader.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace {

std::string g_error;
primesieve::gpu::CudaApi g_api;
bool g_loaded = false;
std::once_flag g_once;

#if defined(_WIN32)
  typedef HMODULE LibHandle;
  LibHandle openLib(const char* name)
  {
    // LOAD_WITH_ALTERED_SEARCH_PATH makes Windows resolve the library's own
    // dependencies from the directory it lives in. NVRTC needs it: loading
    // nvrtc64_*.dll by absolute path otherwise fails to find the
    // nvrtc-builtins DLL sitting right next to it.
    return LoadLibraryExA(name, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  void* symbol(LibHandle h, const char* n) { return (void*) GetProcAddress(h, n); }
  const char* const kDriver[] = { "nvcuda.dll" };
  // NVRTC carries its CUDA major version in the file name.
  const char* const kNvrtc[] = {
    "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll",
    "nvrtc64_111_0.dll", "nvrtc64_101_0.dll"
  };
#else
  typedef void* LibHandle;
  LibHandle openLib(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
  void* symbol(LibHandle h, const char* n) { return dlsym(h, n); }
  const char* const kDriver[] = { "libcuda.so.1", "libcuda.so" };
  const char* const kNvrtc[] = {
    "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"
  };
#endif

/// Make the directory containing `libPath` searchable for shared libraries,
/// so that a library loaded from there can find its own siblings.
void addLibraryDirectoryOf(const char* libPath)
{
  std::string path(libPath);
  std::size_t cut = path.find_last_of("/\\");
  if (cut == std::string::npos)
    return;
  std::string dir = path.substr(0, cut);
  if (dir.empty())
    return;

#if defined(_WIN32)
  std::string current;
  if (const char* env = std::getenv("PATH"))
    current = env;
  if (current.find(dir) == std::string::npos)
  {
    std::string updated = "PATH=" + dir + ";" + current;
    _putenv(updated.c_str());
  }
#else
  // On POSIX the dynamic loader reads LD_LIBRARY_PATH once at start-up, so
  // changing it here would not help; dlopen()ing the sibling by name from
  // the same directory is handled by the loader's RUNPATH instead.
  (void) dir;
#endif
}

struct Bind { const char* name; void** slot; };

bool bindDriver(LibHandle lib, primesieve::gpu::CudaApi& api)
{
  // The driver exports the versioned names for everything that ever changed
  // ABI; the unsuffixed names are the old, incompatible ones.
  const Bind required[] = {
    { "cuInit",                 (void**) &api.Init },
    { "cuDeviceGetCount",       (void**) &api.DeviceGetCount },
    { "cuDeviceGet",            (void**) &api.DeviceGet },
    { "cuDeviceGetName",        (void**) &api.DeviceGetName },
    { "cuDeviceGetAttribute",   (void**) &api.DeviceGetAttribute },
    { "cuDeviceTotalMem_v2",    (void**) &api.DeviceTotalMem },
    { "cuCtxCreate_v2",         (void**) &api.CtxCreate },
    { "cuCtxDestroy_v2",        (void**) &api.CtxDestroy },
    { "cuCtxSetCurrent",        (void**) &api.CtxSetCurrent },
    { "cuCtxSynchronize",       (void**) &api.CtxSynchronize },
    { "cuModuleLoadData",       (void**) &api.ModuleLoadData },
    { "cuModuleUnload",         (void**) &api.ModuleUnload },
    { "cuModuleGetFunction",    (void**) &api.ModuleGetFunction },
    { "cuFuncSetAttribute",     (void**) &api.FuncSetAttribute },
    { "cuMemAlloc_v2",          (void**) &api.MemAlloc },
    { "cuMemFree_v2",           (void**) &api.MemFree },
    { "cuMemcpyHtoD_v2",        (void**) &api.MemcpyHtoD },
    { "cuMemcpyDtoH_v2",        (void**) &api.MemcpyDtoH },
    { "cuLaunchKernel",         (void**) &api.LaunchKernel },
    { "cuGetErrorString",       (void**) &api.GetErrorString }
  };

  for (std::size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++)
  {
    void* fn = symbol(lib, required[i].name);
    if (!fn)
    {
      g_error = std::string("CUDA driver is missing the symbol ") + required[i].name;
      return false;
    }
    *required[i].slot = fn;
  }
  return true;
}

bool bindNvrtc(LibHandle lib, primesieve::gpu::CudaApi& api)
{
  const Bind required[] = {
    { "nvrtcCreateProgram",     (void**) &api.nvrtcCreateProgram },
    { "nvrtcDestroyProgram",    (void**) &api.nvrtcDestroyProgram },
    { "nvrtcCompileProgram",    (void**) &api.nvrtcCompileProgram },
    { "nvrtcGetPTXSize",        (void**) &api.nvrtcGetPTXSize },
    { "nvrtcGetPTX",            (void**) &api.nvrtcGetPTX },
    { "nvrtcGetProgramLogSize", (void**) &api.nvrtcGetProgramLogSize },
    { "nvrtcGetProgramLog",     (void**) &api.nvrtcGetProgramLog },
    { "nvrtcGetErrorString",    (void**) &api.nvrtcGetErrorString },
    { "nvrtcVersion",           (void**) &api.nvrtcVersion }
  };

  for (std::size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++)
  {
    void* fn = symbol(lib, required[i].name);
    if (!fn)
      return false;
    *required[i].slot = fn;
  }
  return true;
}

/// Try the names NVRTC ships under, plus anything the caller points us at
/// with PRIMESIEVE_NVRTC_PATH (useful when nvrtc came from a Python wheel
/// or a relocatable toolkit rather than a system install).
void loadNvrtc(primesieve::gpu::CudaApi& api)
{
  if (const char* custom = std::getenv("PRIMESIEVE_NVRTC_PATH"))
  {
    // NVRTC loads its own nvrtc-builtins library while compiling, using the
    // ordinary search order rather than its own directory. Loading NVRTC by
    // an absolute path therefore is not enough: put that directory on the
    // search path too, or the compile fails with
    // NVRTC_ERROR_BUILTIN_OPERATION_FAILURE.
    addLibraryDirectoryOf(custom);

    LibHandle lib = openLib(custom);
    if (lib && bindNvrtc(lib, api))
    {
      api.haveNvrtc = true;
      return;
    }
  }

  const std::size_t n = sizeof(kNvrtc) / sizeof(kNvrtc[0]);
  for (std::size_t i = 0; i < n; i++)
  {
    LibHandle lib = openLib(kNvrtc[i]);
    if (!lib)
      continue;
    if (bindNvrtc(lib, api))
    {
      api.haveNvrtc = true;
      return;
    }
  }

  g_error = "the CUDA driver was found but NVRTC was not; install the CUDA "
            "toolkit or set PRIMESIEVE_NVRTC_PATH to an nvrtc shared library";
}

void doLoad()
{
  const std::size_t n = sizeof(kDriver) / sizeof(kDriver[0]);

  for (std::size_t i = 0; i < n; i++)
  {
    LibHandle lib = openLib(kDriver[i]);
    if (!lib)
      continue;
    if (bindDriver(lib, g_api))
    {
      g_loaded = true;
      loadNvrtc(g_api);
    }
    return;
  }

  g_error = "no CUDA driver found (tried: ";
  for (std::size_t i = 0; i < n; i++)
    g_error += std::string(i ? ", " : "") + kDriver[i];
  g_error += ")";
}

} // namespace

namespace primesieve {
namespace gpu {

const CudaApi* loadCuda()
{
  std::call_once(g_once, doLoad);
  return g_loaded ? &g_api : nullptr;
}

const std::string& cudaLoadError()
{
  return g_error;
}

std::string cudaErrorString(const CudaApi* api, CUresult err)
{
  const char* text = nullptr;
  if (api && api->GetErrorString && api->GetErrorString(err, &text) == CUDA_SUCCESS && text)
    return std::string(text);

  char buf[64];
  std::snprintf(buf, sizeof buf, "CUDA error %d", (int) err);
  return std::string(buf);
}

} // namespace gpu
} // namespace primesieve

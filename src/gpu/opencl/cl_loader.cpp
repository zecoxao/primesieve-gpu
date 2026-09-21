///
/// @file  cl_loader.cpp
/// @brief Runtime loader for the OpenCL ICD loader shared library.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include "cl_loader.hpp"

#include <cstddef>
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
primesieve::gpu::ClApi g_api;
bool g_loaded = false;
std::once_flag g_once;

#if defined(_WIN32)
  typedef HMODULE LibHandle;
  LibHandle openLib(const char* name) { return LoadLibraryA(name); }
  void* symbol(LibHandle h, const char* n) { return (void*) GetProcAddress(h, n); }
  const char* const kCandidates[] = { "OpenCL.dll" };
#elif defined(__APPLE__)
  typedef void* LibHandle;
  LibHandle openLib(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
  void* symbol(LibHandle h, const char* n) { return dlsym(h, n); }
  const char* const kCandidates[] = {
    "/System/Library/Frameworks/OpenCL.framework/OpenCL",
    "libOpenCL.dylib"
  };
#else
  typedef void* LibHandle;
  LibHandle openLib(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
  void* symbol(LibHandle h, const char* n) { return dlsym(h, n); }
  const char* const kCandidates[] = {
    "libOpenCL.so.1", "libOpenCL.so", "libMesaOpenCL.so.1"
  };
#endif

struct Bind { const char* name; void** slot; };

/// Bind the entry points the backend cannot work without.
bool bindRequired(LibHandle lib, primesieve::gpu::ClApi& api)
{
  const Bind required[] = {
    { "clGetPlatformIDs",          (void**) &api.GetPlatformIDs },
    { "clGetPlatformInfo",         (void**) &api.GetPlatformInfo },
    { "clGetDeviceIDs",            (void**) &api.GetDeviceIDs },
    { "clGetDeviceInfo",           (void**) &api.GetDeviceInfo },
    { "clCreateContext",           (void**) &api.CreateContext },
    { "clCreateProgramWithSource", (void**) &api.CreateProgramWithSource },
    { "clBuildProgram",            (void**) &api.BuildProgram },
    { "clGetProgramBuildInfo",     (void**) &api.GetProgramBuildInfo },
    { "clCreateKernel",            (void**) &api.CreateKernel },
    { "clGetKernelWorkGroupInfo",  (void**) &api.GetKernelWorkGroupInfo },
    { "clSetKernelArg",            (void**) &api.SetKernelArg },
    { "clCreateBuffer",            (void**) &api.CreateBuffer },
    { "clEnqueueWriteBuffer",      (void**) &api.EnqueueWriteBuffer },
    { "clEnqueueReadBuffer",       (void**) &api.EnqueueReadBuffer },
    { "clEnqueueNDRangeKernel",    (void**) &api.EnqueueNDRangeKernel },
    { "clFinish",                  (void**) &api.Finish },
    { "clFlush",                   (void**) &api.Flush },
    { "clReleaseMemObject",        (void**) &api.ReleaseMemObject },
    { "clReleaseKernel",           (void**) &api.ReleaseKernel },
    { "clReleaseProgram",          (void**) &api.ReleaseProgram },
    { "clReleaseCommandQueue",     (void**) &api.ReleaseCommandQueue },
    { "clReleaseContext",          (void**) &api.ReleaseContext }
  };

  for (std::size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++)
  {
    void* fn = symbol(lib, required[i].name);
    if (!fn)
    {
      g_error = std::string("OpenCL runtime is missing the symbol ") + required[i].name;
      return false;
    }
    *required[i].slot = fn;
  }

  // At least one of these must exist: clCreateCommandQueueWithProperties is
  // OpenCL >= 2.0, clCreateCommandQueue is the deprecated 1.x entry point.
  api.CreateCommandQueueWithProperties =
    (ps_clCreateCommandQueueWithProperties_t) symbol(lib, "clCreateCommandQueueWithProperties");
  api.CreateCommandQueue =
    (ps_clCreateCommandQueue_t) symbol(lib, "clCreateCommandQueue");

  if (!api.CreateCommandQueueWithProperties && !api.CreateCommandQueue)
  {
    g_error = "OpenCL runtime provides no way to create a command queue";
    return false;
  }
  return true;
}

void doLoad()
{
  const std::size_t n = sizeof(kCandidates) / sizeof(kCandidates[0]);

  for (std::size_t i = 0; i < n; i++)
  {
    LibHandle lib = openLib(kCandidates[i]);
    if (!lib)
      continue;
    if (bindRequired(lib, g_api))
      g_loaded = true;
    // The library was found; report success or the reason it is unusable.
    return;
  }

  g_error = "no OpenCL runtime found (tried: ";
  for (std::size_t i = 0; i < n; i++)
    g_error += std::string(i ? ", " : "") + kCandidates[i];
  g_error += ")";
}

} // namespace

namespace primesieve {
namespace gpu {

const ClApi* loadOpenCL()
{
  std::call_once(g_once, doLoad);
  return g_loaded ? &g_api : nullptr;
}

const std::string& openclLoadError()
{
  return g_error;
}

const char* clErrorString(cl_int err)
{
  switch (err)
  {
    case 0:   return "CL_SUCCESS";
    case -1:  return "CL_DEVICE_NOT_FOUND";
    case -2:  return "CL_DEVICE_NOT_AVAILABLE";
    case -3:  return "CL_COMPILER_NOT_AVAILABLE";
    case -4:  return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:  return "CL_OUT_OF_RESOURCES";
    case -6:  return "CL_OUT_OF_HOST_MEMORY";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -30: return "CL_INVALID_VALUE";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    default:  return "CL_UNKNOWN_ERROR";
  }
}

} // namespace gpu
} // namespace primesieve

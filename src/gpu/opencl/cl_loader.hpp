///
/// @file  cl_loader.hpp
/// @brief Runtime loader for the OpenCL ICD loader shared library.
///
///        primesieve loads OpenCL lazily at runtime, so libprimesieve keeps
///        working (falling back to the CPU sieve) on machines that have no
///        OpenCL runtime installed, and building the GPU backend requires no
///        OpenCL SDK and no OpenCL import library.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_CL_LOADER_HPP
#define PRIMESIEVE_GPU_CL_LOADER_HPP

#include "cl_minimal.h"
#include <string>

namespace primesieve {
namespace gpu {

struct ClApi
{
  ps_clGetPlatformIDs_t                    GetPlatformIDs = nullptr;
  ps_clGetPlatformInfo_t                   GetPlatformInfo = nullptr;
  ps_clGetDeviceIDs_t                      GetDeviceIDs = nullptr;
  ps_clGetDeviceInfo_t                     GetDeviceInfo = nullptr;
  ps_clCreateContext_t                     CreateContext = nullptr;
  ps_clCreateCommandQueue_t                CreateCommandQueue = nullptr;
  ps_clCreateCommandQueueWithProperties_t  CreateCommandQueueWithProperties = nullptr;
  ps_clCreateProgramWithSource_t           CreateProgramWithSource = nullptr;
  ps_clBuildProgram_t                      BuildProgram = nullptr;
  ps_clGetProgramBuildInfo_t               GetProgramBuildInfo = nullptr;
  ps_clCreateKernel_t                      CreateKernel = nullptr;
  ps_clGetKernelWorkGroupInfo_t            GetKernelWorkGroupInfo = nullptr;
  ps_clSetKernelArg_t                      SetKernelArg = nullptr;
  ps_clCreateBuffer_t                      CreateBuffer = nullptr;
  ps_clEnqueueWriteBuffer_t                EnqueueWriteBuffer = nullptr;
  ps_clEnqueueReadBuffer_t                 EnqueueReadBuffer = nullptr;
  ps_clEnqueueNDRangeKernel_t              EnqueueNDRangeKernel = nullptr;
  ps_clFinish_t                            Finish = nullptr;
  ps_clFlush_t                             Flush = nullptr;
  ps_clReleaseMemObject_t                  ReleaseMemObject = nullptr;
  ps_clReleaseKernel_t                     ReleaseKernel = nullptr;
  ps_clReleaseProgram_t                    ReleaseProgram = nullptr;
  ps_clReleaseCommandQueue_t               ReleaseCommandQueue = nullptr;
  ps_clReleaseContext_t                    ReleaseContext = nullptr;
};

/// Returns the loaded OpenCL entry points, or nullptr if no OpenCL runtime
/// could be loaded. Loaded once per process. Thread safe.
const ClApi* loadOpenCL();

/// Human readable reason why loadOpenCL() returned nullptr.
const std::string& openclLoadError();

/// Human readable name for an OpenCL error code.
const char* clErrorString(cl_int err);

} // namespace gpu
} // namespace primesieve

#endif

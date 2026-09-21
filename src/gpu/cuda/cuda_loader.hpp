///
/// @file  cuda_loader.hpp
/// @brief Runtime loader for the CUDA driver and the NVRTC kernel compiler.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_CUDA_LOADER_HPP
#define PRIMESIEVE_GPU_CUDA_LOADER_HPP

#include "cuda_minimal.h"
#include <string>

namespace primesieve {
namespace gpu {

struct CudaApi
{
  ps_cuInit_t               Init = nullptr;
  ps_cuDeviceGetCount_t     DeviceGetCount = nullptr;
  ps_cuDeviceGet_t          DeviceGet = nullptr;
  ps_cuDeviceGetName_t      DeviceGetName = nullptr;
  ps_cuDeviceGetAttribute_t DeviceGetAttribute = nullptr;
  ps_cuDeviceTotalMem_t     DeviceTotalMem = nullptr;
  ps_cuCtxCreate_t          CtxCreate = nullptr;
  ps_cuCtxDestroy_t         CtxDestroy = nullptr;
  ps_cuCtxSetCurrent_t      CtxSetCurrent = nullptr;
  ps_cuCtxSynchronize_t     CtxSynchronize = nullptr;
  ps_cuModuleLoadData_t     ModuleLoadData = nullptr;
  ps_cuModuleUnload_t       ModuleUnload = nullptr;
  ps_cuModuleGetFunction_t  ModuleGetFunction = nullptr;
  ps_cuFuncSetAttribute_t   FuncSetAttribute = nullptr;
  ps_cuMemAlloc_t           MemAlloc = nullptr;
  ps_cuMemFree_t            MemFree = nullptr;
  ps_cuMemcpyHtoD_t         MemcpyHtoD = nullptr;
  ps_cuMemcpyDtoH_t         MemcpyDtoH = nullptr;
  ps_cuLaunchKernel_t       LaunchKernel = nullptr;
  ps_cuGetErrorString_t     GetErrorString = nullptr;

  ps_nvrtcCreateProgram_t     nvrtcCreateProgram = nullptr;
  ps_nvrtcDestroyProgram_t    nvrtcDestroyProgram = nullptr;
  ps_nvrtcCompileProgram_t    nvrtcCompileProgram = nullptr;
  ps_nvrtcGetPTXSize_t        nvrtcGetPTXSize = nullptr;
  ps_nvrtcGetPTX_t            nvrtcGetPTX = nullptr;
  ps_nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize = nullptr;
  ps_nvrtcGetProgramLog_t     nvrtcGetProgramLog = nullptr;
  ps_nvrtcGetErrorString_t    nvrtcGetErrorString = nullptr;
  ps_nvrtcVersion_t           nvrtcVersion = nullptr;

  /// True once nvrtc as well as the driver has been bound. Without nvrtc the
  /// devices can still be listed, but no kernel can be built.
  bool haveNvrtc = false;
};

/// Returns the loaded CUDA entry points, or nullptr if the CUDA driver could
/// not be loaded. Loaded once per process. Thread safe.
const CudaApi* loadCuda();

/// Human readable reason why loadCuda() failed, or why nvrtc is missing.
const std::string& cudaLoadError();

/// Human readable name for a CUDA driver error code.
std::string cudaErrorString(const CudaApi* api, CUresult err);

} // namespace gpu
} // namespace primesieve

#endif

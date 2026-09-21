///
/// @file  cuda_minimal.h
/// @brief Minimal, self-contained CUDA Driver API and NVRTC declarations.
///
///        Like the OpenCL backend, the CUDA backend loads everything it needs
///        at runtime: nvcuda (the driver, always installed alongside an
///        NVIDIA GPU) and nvrtc (the runtime kernel compiler). Building it
///        therefore needs no CUDA toolkit and links against no CUDA library.
///        Only the subset of the API the backend uses is declared here.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_CUDA_MINIMAL_H
#define PRIMESIEVE_GPU_CUDA_MINIMAL_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
  #define PS_CUDAAPI __stdcall
#else
  #define PS_CUDAAPI
#endif

typedef int      CUresult;
typedef int      CUdevice;
typedef uint64_t CUdeviceptr;
typedef int      CUdevice_attribute;
typedef int      CUfunction_attribute;

typedef struct CUctx_st*  CUcontext;
typedef struct CUmod_st*  CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;

#define CUDA_SUCCESS                                       0
#define CUDA_ERROR_NO_DEVICE                             100
#define CUDA_ERROR_INVALID_DEVICE                        101
#define CUDA_ERROR_OUT_OF_MEMORY                           2
#define CUDA_ERROR_LAUNCH_TIMEOUT                        702

#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK          1
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK    8
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE                     10
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE                    13
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT          16
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                    18
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR      75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR      76
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97

/// Raising this past 48 KiB is what lets a block use the larger shared
/// memory of Volta and newer.
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES    8

typedef CUresult (PS_CUDAAPI *ps_cuInit_t)(unsigned int);
typedef CUresult (PS_CUDAAPI *ps_cuDeviceGetCount_t)(int*);
typedef CUresult (PS_CUDAAPI *ps_cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (PS_CUDAAPI *ps_cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (PS_CUDAAPI *ps_cuDeviceGetAttribute_t)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (PS_CUDAAPI *ps_cuDeviceTotalMem_t)(size_t*, CUdevice);
typedef CUresult (PS_CUDAAPI *ps_cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (PS_CUDAAPI *ps_cuCtxDestroy_t)(CUcontext);
typedef CUresult (PS_CUDAAPI *ps_cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (PS_CUDAAPI *ps_cuCtxSynchronize_t)(void);
typedef CUresult (PS_CUDAAPI *ps_cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (PS_CUDAAPI *ps_cuModuleUnload_t)(CUmodule);
typedef CUresult (PS_CUDAAPI *ps_cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (PS_CUDAAPI *ps_cuFuncSetAttribute_t)(CUfunction, CUfunction_attribute, int);
typedef CUresult (PS_CUDAAPI *ps_cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (PS_CUDAAPI *ps_cuMemFree_t)(CUdeviceptr);
typedef CUresult (PS_CUDAAPI *ps_cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (PS_CUDAAPI *ps_cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (PS_CUDAAPI *ps_cuLaunchKernel_t)(CUfunction,
                                                   unsigned int, unsigned int, unsigned int,
                                                   unsigned int, unsigned int, unsigned int,
                                                   unsigned int, CUstream, void**, void**);
typedef CUresult (PS_CUDAAPI *ps_cuGetErrorString_t)(CUresult, const char**);

/* ---- NVRTC ---- */

typedef int CUnvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;

#define NVRTC_SUCCESS 0

typedef CUnvrtcResult (*ps_nvrtcCreateProgram_t)(nvrtcProgram*, const char*, const char*,
                                                 int, const char* const*, const char* const*);
typedef CUnvrtcResult (*ps_nvrtcDestroyProgram_t)(nvrtcProgram*);
typedef CUnvrtcResult (*ps_nvrtcCompileProgram_t)(nvrtcProgram, int, const char* const*);
typedef CUnvrtcResult (*ps_nvrtcGetPTXSize_t)(nvrtcProgram, size_t*);
typedef CUnvrtcResult (*ps_nvrtcGetPTX_t)(nvrtcProgram, char*);
typedef CUnvrtcResult (*ps_nvrtcGetProgramLogSize_t)(nvrtcProgram, size_t*);
typedef CUnvrtcResult (*ps_nvrtcGetProgramLog_t)(nvrtcProgram, char*);
typedef const char*   (*ps_nvrtcGetErrorString_t)(CUnvrtcResult);
typedef CUnvrtcResult (*ps_nvrtcVersion_t)(int*, int*);

#endif

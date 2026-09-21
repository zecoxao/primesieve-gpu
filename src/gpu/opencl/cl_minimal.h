///
/// @file  cl_minimal.h
/// @brief Minimal, self-contained OpenCL ABI declarations.
///
///        primesieve's GPU backend loads the OpenCL ICD loader at runtime
///        (OpenCL.dll / libOpenCL.so), so building it requires no OpenCL SDK
///        and links against no OpenCL import library. Only the subset of the
///        API that the backend actually uses is declared here. The types,
///        enum values and calling conventions follow the OpenCL 1.2
///        specification, which every OpenCL implementation is compatible
///        with.
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#ifndef PRIMESIEVE_GPU_CL_MINIMAL_H
#define PRIMESIEVE_GPU_CL_MINIMAL_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
  #define PS_CL_API __stdcall
#else
  #define PS_CL_API
#endif

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef int64_t   cl_long;
typedef uint64_t  cl_ulong;
typedef cl_ulong  cl_bitfield;
typedef cl_uint   cl_bool;

typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_queue_properties;
typedef intptr_t    cl_context_properties;
typedef cl_uint     cl_platform_info;
typedef cl_uint     cl_device_info;
typedef cl_uint     cl_program_build_info;
typedef cl_uint     cl_kernel_work_group_info;

typedef struct _cl_platform_id*   cl_platform_id;
typedef struct _cl_device_id*     cl_device_id;
typedef struct _cl_context*       cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_program*       cl_program;
typedef struct _cl_kernel*        cl_kernel;
typedef struct _cl_mem*           cl_mem;
typedef struct _cl_event*         cl_event;

#define CL_SUCCESS                                   0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_BUILD_PROGRAM_FAILURE                   -11
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_INVALID_WORK_GROUP_SIZE                  -54

#define CL_FALSE                                     0
#define CL_TRUE                                      1

#define CL_DEVICE_TYPE_CPU                  (1ull << 1)
#define CL_DEVICE_TYPE_GPU                  (1ull << 2)
#define CL_DEVICE_TYPE_ACCELERATOR          (1ull << 3)
#define CL_DEVICE_TYPE_ALL                  0xFFFFFFFFull

#define CL_PLATFORM_NAME                        0x0902
#define CL_PLATFORM_VERSION                     0x0901

#define CL_DEVICE_TYPE                          0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS             0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE           0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY           0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE            0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE               0x101F
#define CL_DEVICE_LOCAL_MEM_TYPE                0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE                0x1023
#define CL_DEVICE_NAME                          0x102B
#define CL_DEVICE_VENDOR                        0x102C
#define CL_DEVICE_VERSION                       0x102F
#define CL_DEVICE_EXTENSIONS                    0x1030

#define CL_LOCAL                                     1
#define CL_GLOBAL                                    2

#define CL_CONTEXT_PLATFORM                     0x1084

#define CL_QUEUE_PROFILING_ENABLE           (1ull << 1)

#define CL_MEM_READ_WRITE                   (1ull << 0)
#define CL_MEM_WRITE_ONLY                   (1ull << 1)
#define CL_MEM_READ_ONLY                    (1ull << 2)
#define CL_MEM_USE_HOST_PTR                 (1ull << 3)
#define CL_MEM_ALLOC_HOST_PTR               (1ull << 4)
#define CL_MEM_COPY_HOST_PTR                (1ull << 5)

#define CL_PROGRAM_BUILD_STATUS                 0x1181
#define CL_PROGRAM_BUILD_LOG                    0x1183

#define CL_KERNEL_WORK_GROUP_SIZE               0x11B0
#define CL_KERNEL_LOCAL_MEM_SIZE                0x11B2
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3

typedef cl_int (PS_CL_API *ps_clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (PS_CL_API *ps_clGetPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (PS_CL_API *ps_clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (PS_CL_API *ps_clGetDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (PS_CL_API *ps_clCreateContext_t)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                                     void (PS_CL_API*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (PS_CL_API *ps_clCreateCommandQueue_t)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_command_queue (PS_CL_API *ps_clCreateCommandQueueWithProperties_t)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
typedef cl_program (PS_CL_API *ps_clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (PS_CL_API *ps_clBuildProgram_t)(cl_program, cl_uint, const cl_device_id*, const char*,
                                                void (PS_CL_API*)(cl_program, void*), void*);
typedef cl_int (PS_CL_API *ps_clGetProgramBuildInfo_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (PS_CL_API *ps_clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_int (PS_CL_API *ps_clGetKernelWorkGroupInfo_t)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*);
typedef cl_int (PS_CL_API *ps_clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_mem (PS_CL_API *ps_clCreateBuffer_t)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (PS_CL_API *ps_clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (PS_CL_API *ps_clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (PS_CL_API *ps_clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (PS_CL_API *ps_clFinish_t)(cl_command_queue);
typedef cl_int (PS_CL_API *ps_clFlush_t)(cl_command_queue);
typedef cl_int (PS_CL_API *ps_clReleaseMemObject_t)(cl_mem);
typedef cl_int (PS_CL_API *ps_clReleaseKernel_t)(cl_kernel);
typedef cl_int (PS_CL_API *ps_clReleaseProgram_t)(cl_program);
typedef cl_int (PS_CL_API *ps_clReleaseCommandQueue_t)(cl_command_queue);
typedef cl_int (PS_CL_API *ps_clReleaseContext_t)(cl_context);

#endif

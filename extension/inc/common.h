#ifndef COMMON_HEADER_FILE_H
#define COMMON_HEADER_FILE_H

// clang-format off
#include <cublas.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
// clang-format on

using index_t = int64_t;

#define CHECK_CPU(x, d, t)                                                \
    TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")                   \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(x.scalar_type() == t, #x " must be type of " #t)          \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

#define CHECK_CUDA(x, d, t)                                               \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")                 \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(x.scalar_type() == t, #x " must be type of " #t)          \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

#define CUDA_CHECH(func)                                                      \
    {                                                                         \
        cudaError_t status = (func);                                          \
        if (status != cudaSuccess) {                                          \
            printf(                                                           \
                "CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
                cudaGetErrorString(status), status                            \
            );                                                                \
            throw std::runtime_error("cuda error");                           \
        }                                                                     \
    }

#define CUBLAS_CHECK(func)                                                   \
    {                                                                        \
        cublasStatus_t status = (func);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            printf(                                                          \
                "CUBLAS API failed at line %d with error: (%d)\n", __LINE__, \
                status                                                       \
            );                                                               \
            throw std::runtime_error("cublas error");                        \
        }                                                                    \
    }

#define CUSPARSE_CHECK(func)                                            \
    {                                                                   \
        cusparseStatus_t status = (func);                               \
        if (status != CUSPARSE_STATUS_SUCCESS) {                        \
            printf(                                                     \
                "CUSPARSE API failed at line %d with error: %s (%d)\n", \
                __LINE__, cusparseGetErrorString(status), status        \
            );                                                          \
            throw std::runtime_error("cusparse error");                 \
        }                                                               \
    }

#endif
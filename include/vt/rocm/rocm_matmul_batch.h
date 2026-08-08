// ROCm: batched MatmulBT helpers for MoE top-k fuse.
#pragma once

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {

void MatmulBTStridedBatchKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                    int batch, int M, int N, int K, DType dtype);

void MatmulBTStridedBatchFullKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                        int batch, int M, int N, int K, long long strideA,
                                        DType dtype);

// Pointer-array batch (no gather): out_ptrs[g] = a @ b_ptrs[g]^T
void MatmulBTPointerBatchKernelRocm(Queue& q, void** out_ptrs, const void* a,
                                    void** b_ptrs, int batch, int M, int N, int K,
                                    DType dtype);

// Both A and B per-batch: out_ptrs[g] = a_ptrs[g] @ b_ptrs[g]^T
void MatmulBTPointerBatchABKernelRocm(Queue& q, void** out_ptrs, void** a_ptrs,
                                      void** b_ptrs, int batch, int M, int N, int K,
                                      DType dtype);

}  // namespace vt::rocm

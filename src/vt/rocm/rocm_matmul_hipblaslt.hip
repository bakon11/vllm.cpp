// ROCm dense GEMM via hipBLAS (hipblasGemmEx) + optional hipBLASLt.
// Primary path: hipblasGemmEx with the standard row-major TN trick used by
// many ROCm LLM stacks. Falls back is not needed — GemmEx is required.
//
// MatmulBT: out[M,N] = a[M,K] @ b[N,K]^T  (b = Linear weight row-major)
// Matmul:   out[M,N] = a[M,K] @ b[K,N]
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

void CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

const char* BlasName(hipblasStatus_t st) {
  switch (st) {
    case HIPBLAS_STATUS_SUCCESS: return "SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM: return "INVALID_ENUM";
    default: return "UNKNOWN";
  }
}

void CheckBlas(hipblasStatus_t st, const char* what) {
  if (st != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what +
                             ": hipblas " + std::to_string(static_cast<int>(st)) +
                             " (" + BlasName(st) + ")");
  }
}

struct BlasCtx {
  hipblasHandle_t handle = nullptr;
};

// Per-thread handle: hipBLAS is not free-threaded; decode is single-threaded per
// engine worker. Avoid global mutex + SetStream on every GEMM (100s×/token).
BlasCtx GetBlas(int device, hipStream_t stream) {
  struct Tls {
    int dev = -1;
    hipStream_t stream = nullptr;
    hipblasHandle_t handle = nullptr;
  };
  static thread_local Tls tls;
  if (tls.handle == nullptr || tls.dev != device) {
    if (tls.handle) {
      // device switch rare — destroy old handle
      (void)hipblasDestroy(tls.handle);
      tls.handle = nullptr;
    }
    CheckBlas(hipblasCreate(&tls.handle), "hipblasCreate");
    tls.dev = device;
    tls.stream = nullptr;
  }
  if (tls.stream != stream) {
    CheckBlas(hipblasSetStream(tls.handle, stream), "hipblasSetStream");
    tls.stream = stream;
  }
  return BlasCtx{tls.handle};
}

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
}

hipDataType ToBlasType(DType dt) {
  switch (dt) {
    case DType::kF32: return HIP_R_32F;
    case DType::kBF16: return HIP_R_16BF;
    case DType::kF16: return HIP_R_16F;
    default:
      throw std::runtime_error("vt rocm: matmul: unsupported dtype");
  }
}

// BF16 default: COMPUTE_32F (FAST_16BF returns NOT_SUPPORTED on gfx1201 for our BT shapes).
// Override: VT_ROCM_GEMM_COMPUTE=32f | 16bf | 16f
hipblasComputeType_t GemmCompute(DType dt) {
  if (dt != DType::kBF16 && dt != DType::kF16) return HIPBLAS_COMPUTE_32F;
  static const hipblasComputeType_t k = [] {
    if (const char* e = std::getenv("VT_ROCM_GEMM_COMPUTE")) {
      if (std::strcmp(e, "32f") == 0) return HIPBLAS_COMPUTE_32F;
      if (std::strcmp(e, "16f") == 0) return HIPBLAS_COMPUTE_16F;
      if (std::strcmp(e, "16bf") == 0) return HIPBLAS_COMPUTE_32F_FAST_16BF;
    }
    return HIPBLAS_COMPUTE_32F;
  }();
  return k;
}

// VT_ROCM_GEMV=1 enables naive M=1 BF16 GEMV (default OFF — hipblas faster on gfx1201).
bool GemvEnabled() {
  static const bool on = [] {
    if (const char* e = std::getenv("VT_ROCM_GEMV")) return e[0] == '1';
    return false;
  }();
  return on;
}

// y[n] = alpha * dot(x[0:K], W[n,0:K]) + beta * y[n]
// W row-major [N,K]. One block per n.
__global__ void Bf16GemvBTKernel(__hip_bfloat16* __restrict__ y,
                                 const __hip_bfloat16* __restrict__ x,
                                 const __hip_bfloat16* __restrict__ W, int N, int K,
                                 float alpha, float beta) {
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  const __hip_bfloat16* wrow = W + static_cast<int64_t>(n) * K;
  float acc = 0.f;
  for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
    acc += __bfloat162float(x[k]) * __bfloat162float(wrow[k]);
  }
  // block reduce
  __shared__ float sm[256];
  const int tid = static_cast<int>(threadIdx.x);
  sm[tid] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float v = alpha * sm[0];
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

void Bf16GemvBT(hipStream_t s, void* out, const void* a, const void* b, int N, int K,
                float alpha, float beta) {
  constexpr int kBlock = 256;
  Bf16GemvBTKernel<<<N, kBlock, 0, s>>>(
      static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
      static_cast<const __hip_bfloat16*>(b), N, K, alpha, beta);
}

}  // namespace

// out[M,N] = a[M,K] @ b[K,N]  (row-major contiguous / a may have row stride)
// Col-major BLAS view: C^T = B^T @ A^T
//   gemm(OP_T, OP_T, N, M, K, B, ldb=N? wait)
// Standard row-major C = A @ B via:
//   gemm(OP_N, OP_N, N, M, K, B_col, N, A_col, K, C_col, N)
// where A_row[M,K] is treated as A_col with ld=K meaning A_col is K x M...
//
// PyTorch/rocBLAS convention for row-major C(m,n) = A(m,k) @ B(k,n):
//   hipblasGemm(handle, OP_N, OP_N, n, m, k,
//               B, n, A, k, C, n)
// with A,B,C stored row-major — i.e. pass dimensions swapped so BLAS
// writes the transpose layout that coincides with row-major C.
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "k0");
    return;
  }
  // Require contiguous B rows; A may be strided in dim0.
  if (b.stride[0] != N) {
    throw std::runtime_error("vt rocm: matmul: b must be row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul: bad a stride");
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C_rm = A_rm @ B_rm  <=>  treat as gemm(N,N) writing C with ld=N, dims n,m,k
  // A leading dim = a.stride[0] (row stride in elements)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(N), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), GemmCompute(a.dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx NN");
}

// out[M,N] = a[M,K] @ b[N,K]^T
// Row-major trick: gemm(OP_T, OP_N, N, M, K, B, K, A, a_rs, C, N)
//   BLAS: C = op(A)*op(B) with opA=T => A is KxN in col form = row B[N,K]
//   opB=N, B is KxM col = row A[M,K] with ld=a_rs
//   C is NxM col = row out[M,N] ld=N
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul_bt: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul_bt: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[0];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "bt k0");
    return;
  }
  if (b.stride[0] != K) {
    throw std::runtime_error("vt rocm: matmul_bt: b must be [N,K] row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul_bt: bad a stride");
  }

  // Decode: M=1 BF16 GEMV
  if (M == 1 && bf16 && out.dtype == DType::kBF16 && a.stride[0] == K && GemvEnabled()) {
    Bf16GemvBT(s, out.data, a.data, b.data, static_cast<int>(N), static_cast<int>(K), 1.f, 0.f);
    return;
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C = B^T @ A^T in row-major disguise:
  // gemm(OP_T, OP_N, N, M, K, B, ldb=K, A, lda=a_rs, C, ldc=N)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(K), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), GemmCompute(a.dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx BT");
}

// out = alpha * (a @ b^T) + beta * out
void MatmulBTAlphaBetaRocm(Queue& q, void* out, const void* a, const void* b, int M, int N,
                           int K, float alpha, float beta, DType dtype) {
  if (M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ab: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    if (beta == 0.f) {
      CheckHip(hipMemsetAsync(out, 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ab k0");
    }
    return;
  }
  // Decode path: custom BF16 GEMV often beats hipblasGemmEx at M=1.
  if (M == 1 && dtype == DType::kBF16 && GemvEnabled()) {
    Bf16GemvBT(s, out, a, b, N, K, alpha, beta);
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const hipDataType at = ToBlasType(dtype);
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          /*m=*/N, /*n=*/M, /*k=*/K, &alpha, b, at, /*ldb=*/K, a, at,
                          /*lda=*/K, &beta, out, at, /*ldc=*/N, GemmCompute(dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx BT alpha/beta");
}

// General strided-batch BT: a may also be batched (strideA elements between batches).
// out[g,M,N] = a_g[M,K] @ b_g[N,K]^T
void MatmulBTStridedBatchFullKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                        int batch, int M, int N, int K, long long strideA,
                                        DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_batch: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out, 0,
                            static_cast<size_t>(batch) * static_cast<size_t>(M) *
                                static_cast<size_t>(N) * SizeOf(dtype),
                            s),
             "bt_batch k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);
  const long long strideB = static_cast<long long>(N) * K;
  const long long strideC = static_cast<long long>(M) * N;
  CheckBlas(hipblasGemmStridedBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha, b, at, /*ldb=*/K, strideB, a, at,
                /*lda=*/K, strideA, &beta, out, at, /*ldc=*/N, strideC, batch,
                GemmCompute(dtype), HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmStridedBatchedEx BT");
}

// Batched BT via pointer arrays (no weight gather): out[g] = a @ b[g]^T
// a shared; b_ptrs[g] -> [N,K]; out_ptrs[g] -> [M,N]
void MatmulBTPointerBatchKernelRocm(Queue& q, void** out_ptrs, const void* a,
                                    void** b_ptrs, int batch, int M, int N, int K,
                                    DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ptr_batch: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    for (int g = 0; g < batch; ++g)
      CheckHip(hipMemsetAsync(out_ptrs[g], 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ptr k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);

  // Reuse device pointer tables (per-device, grows to max batch seen).
  struct PtrTables {
    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;
    int cap = 0;
  };
  static std::mutex mu;
  static std::unordered_map<int, PtrTables> tabs;
  PtrTables* tab = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    tab = &tabs[q.device.index];
    if (tab->cap < batch) {
      if (tab->A) CheckHip(hipFree(tab->A), "freeA");
      if (tab->B) CheckHip(hipFree(tab->B), "freeB");
      if (tab->C) CheckHip(hipFree(tab->C), "freeC");
      const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
      CheckHip(hipMalloc(&tab->A, psz), "ptrA");
      CheckHip(hipMalloc(&tab->B, psz), "ptrB");
      CheckHip(hipMalloc(&tab->C, psz), "ptrC");
      tab->cap = batch;
    }
  }
  std::vector<const void*> a_host(static_cast<size_t>(batch), a);
  const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
  CheckHip(hipMemcpyAsync(tab->A, a_host.data(), psz, hipMemcpyHostToDevice, s), "H2D A");
  CheckHip(hipMemcpyAsync(tab->B, b_ptrs, psz, hipMemcpyHostToDevice, s), "H2D B");
  CheckHip(hipMemcpyAsync(tab->C, out_ptrs, psz, hipMemcpyHostToDevice, s), "H2D C");

  CheckBlas(hipblasGemmBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                reinterpret_cast<const void**>(tab->B), at, /*ldb=*/K,
                reinterpret_cast<const void**>(tab->A), at, /*lda=*/K, &beta,
                reinterpret_cast<void**>(tab->C), at, /*ldc=*/N, batch, GemmCompute(dtype),
                HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmBatchedEx BT");
}

// Pointer-array BT with per-batch A and B: out[g] = a[g] @ b[g]^T
void MatmulBTPointerBatchABKernelRocm(Queue& q, void** out_ptrs, void** a_ptrs,
                                      void** b_ptrs, int batch, int M, int N, int K,
                                      DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ptr_ab: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    for (int g = 0; g < batch; ++g)
      CheckHip(hipMemsetAsync(out_ptrs[g], 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ptr_ab k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);

  struct PtrTables {
    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;
    int cap = 0;
  };
  static std::mutex mu;
  static std::unordered_map<int, PtrTables> tabs;
  PtrTables* tab = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    tab = &tabs[q.device.index];
    if (tab->cap < batch) {
      if (tab->A) CheckHip(hipFree(tab->A), "freeA");
      if (tab->B) CheckHip(hipFree(tab->B), "freeB");
      if (tab->C) CheckHip(hipFree(tab->C), "freeC");
      const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
      CheckHip(hipMalloc(&tab->A, psz), "ptrA");
      CheckHip(hipMalloc(&tab->B, psz), "ptrB");
      CheckHip(hipMalloc(&tab->C, psz), "ptrC");
      tab->cap = batch;
    }
  }
  const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
  CheckHip(hipMemcpyAsync(tab->A, a_ptrs, psz, hipMemcpyHostToDevice, s), "H2D A");
  CheckHip(hipMemcpyAsync(tab->B, b_ptrs, psz, hipMemcpyHostToDevice, s), "H2D B");
  CheckHip(hipMemcpyAsync(tab->C, out_ptrs, psz, hipMemcpyHostToDevice, s), "H2D C");

  CheckBlas(hipblasGemmBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                reinterpret_cast<const void**>(tab->B), at, /*ldb=*/K,
                reinterpret_cast<const void**>(tab->A), at, /*lda=*/K, &beta,
                reinterpret_cast<void**>(tab->C), at, /*ldc=*/N, batch, GemmCompute(dtype),
                HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmBatchedEx BT AB");
}

}  // namespace vt::rocm

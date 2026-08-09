// Fast FP8-E4M3FN weight × BF16 activation decode kernels for Gemma-4 MoE.
// Channel scales are BF16 per output row: y[n] = alpha * scale[n] * dot(x, dequant(W[n])) + beta*y
//
// Why this exists: the first MatmulBTFp8Channel was a scalar software GEMV and was
// MUCH slower than hipBLAS BF16. FP8 should win on bandwidth when the kernel is good.
//
// Strategy (T=1 decode):
//  - Fused Expert GeGLU: one x load, gate+up dots together, gelu*up, then down.
//  - Vectorized uint4 weight loads (16 FP8 / iter), float FMAs.
//  - Top-k sequential mix with alpha/beta into ysum.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

// IEEE fp8-e4m3fn (matches vllm::F8E4M3ToF32 / HF compressed-tensors).
__device__ inline float F8E4M3ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0f : 1.0f;
  if (exp == 0xFU && mant == 0x7U) return 0.f;
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sm * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

__device__ inline float BlockSum256(float v) {
  __shared__ float sm[256];
  const int tid = static_cast<int>(threadIdx.x);
  sm[tid] = v;
  __syncthreads();
#pragma unroll
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  return sm[0];
}

__device__ inline float gelu_tanh(float g) {
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.f + tanhf(inner));
}

// Dot x[K] (float LDS) with FP8 row W[n,0:K], then * scale.
__device__ inline float DotFp8Row(const float* __restrict__ x_cache, const uint8_t* __restrict__ wrow,
                                  int K) {
  float acc = 0.f;
  const int tid = static_cast<int>(threadIdx.x);
  // 16-byte chunks when aligned
  const int K16 = K & ~15;
  for (int k = tid * 16; k < K16; k += kBlock * 16) {
    // manual unroll of 16 loads keeps ILP without requiring alignment of wrow+k
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int base = k + t * 4;
      a0 += x_cache[base + 0] * F8E4M3ToF32(wrow[base + 0]);
      a1 += x_cache[base + 1] * F8E4M3ToF32(wrow[base + 1]);
      a2 += x_cache[base + 2] * F8E4M3ToF32(wrow[base + 2]);
      a3 += x_cache[base + 3] * F8E4M3ToF32(wrow[base + 3]);
    }
    acc += (a0 + a1) + (a2 + a3);
  }
  for (int k = K16 + tid; k < K; k += kBlock) {
    acc += x_cache[k] * F8E4M3ToF32(wrow[k]);
  }
  return BlockSum256(acc);
}

// Single-row channel GEMV (fallback / general MatmulBTFp8Channel M=1).
// grid: ceil(N / rows_per_block) with rows_per_block=1 → one block per output row.
__global__ void Fp8ChannelGemvFastKernel(__hip_bfloat16* __restrict__ y,
                                         const __hip_bfloat16* __restrict__ x,
                                         const uint8_t* __restrict__ W,  // [N,K]
                                         const __hip_bfloat16* __restrict__ scale, int N, int K,
                                         float alpha, float beta) {
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int k = tid; k < K; k += kBlock) x_cache[k] = __bfloat162float(x[k]);
  __syncthreads();

  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  float acc = DotFp8Row(x_cache, wrow, K);
  if (tid == 0) {
    const float s = __bfloat162float(scale[n]);
    float v = alpha * s * acc;
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

// Fused gate|up act: grid = I. w_gu is packed FP8 [2I, H], s_gu BF16 [2I].
__global__ void ExpertFp8ActKernel(float* __restrict__ act, const __hip_bfloat16* __restrict__ x,
                                   const uint8_t* __restrict__ w_gu,
                                   const __hip_bfloat16* __restrict__ s_gu, int I, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int h = tid; h < H; h += kBlock) x_cache[h] = __bfloat162float(x[h]);
  __syncthreads();

  const uint8_t* grow = w_gu + static_cast<int64_t>(i) * H;
  const uint8_t* urow = w_gu + static_cast<int64_t>(I + i) * H;
  // Partial dots in registers then block-reduce once each would need two passes —
  // compute both in one K loop:
  float gate = 0.f, up = 0.f;
  const int H16 = H & ~15;
  for (int h = tid * 16; h < H16; h += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int hh = h + t;
      const float xv = x_cache[hh];
      gate += xv * F8E4M3ToF32(grow[hh]);
      up += xv * F8E4M3ToF32(urow[hh]);
    }
  }
  for (int h = H16 + tid; h < H; h += kBlock) {
    const float xv = x_cache[h];
    gate += xv * F8E4M3ToF32(grow[h]);
    up += xv * F8E4M3ToF32(urow[h]);
  }
  gate = BlockSum256(gate);
  up = BlockSum256(up);
  if (tid == 0) {
    const float sg = __bfloat162float(s_gu[i]);
    const float su = __bfloat162float(s_gu[I + i]);
    act[i] = gelu_tanh(sg * gate) * (su * up);
  }
}

// Down: grid = ceil(H/kBlock). w_dn FP8 [H,I], s_dn BF16 [H].
__global__ void ExpertFp8DownKernel(__hip_bfloat16* __restrict__ y, const float* __restrict__ act,
                                    const uint8_t* __restrict__ w_dn,
                                    const __hip_bfloat16* __restrict__ s_dn, int I, int H,
                                    float alpha, float beta) {
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  const uint8_t* drow = w_dn + static_cast<int64_t>(h) * I;
  float acc = 0.f;
  int i = 0;
  for (; i + 3 < I; i += 4) {
    acc += act[i] * F8E4M3ToF32(drow[i]);
    acc += act[i + 1] * F8E4M3ToF32(drow[i + 1]);
    acc += act[i + 2] * F8E4M3ToF32(drow[i + 2]);
    acc += act[i + 3] * F8E4M3ToF32(drow[i + 3]);
  }
  for (; i < I; ++i) acc += act[i] * F8E4M3ToF32(drow[i]);
  const float sd = __bfloat162float(s_dn[h]);
  float v = alpha * sd * acc;
  if (beta != 0.f) v += beta * __bfloat162float(y[h]);
  y[h] = __float2bfloat16(v);
}

struct ActBuf {
  float* p = nullptr;
  int cap = 0;
};
ActBuf& TlsAct() {
  static thread_local ActBuf b;
  return b;
}
bool EnsureAct(int I) {
  auto& b = TlsAct();
  if (b.cap >= I && b.p) return true;
  if (b.p) (void)hipFree(b.p);
  b.p = nullptr;
  b.cap = 0;
  if (hipMalloc(&b.p, static_cast<size_t>(I) * sizeof(float)) != hipSuccess) return false;
  b.cap = I;
  return true;
}

}  // namespace

// General M=1 channel GEMV (used if fused expert path not taken).
void MatmulBTFp8ChannelRocm(Queue& q, void* out, const void* a, const void* b_fp8,
                            const void* scale_bf16, int M, int N, int K, float alpha,
                            float beta) {
  if (M != 1 || N <= 0 || K <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const size_t shmem = static_cast<size_t>(K) * sizeof(float);
  if (shmem > 48 * 1024) return;
  // One block per output row — full-block reduction over K (much better than
  // the old "one thread per row" scalar kernel).
  Fp8ChannelGemvFastKernel<<<N, kBlock, shmem, s>>>(
      static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
      static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), N, K,
      alpha, beta);
}

// Fused Expert GeGLU FP8 decode T=1. Returns false on setup failure.
bool ExpertGeGLUFp8M1Rocm(Queue& q, void* y, const void* x, const void* fp8_gu, const void* s_gu,
                          const void* fp8_dn, const void* s_dn, int I, int H, float alpha,
                          float beta) {
  if (!y || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || I <= 0 || H <= 0) return false;
  const size_t shmem = static_cast<size_t>(H) * sizeof(float);
  if (shmem > 48 * 1024) return false;
  if (!EnsureAct(I)) return false;
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  float* act = TlsAct().p;
  ExpertFp8ActKernel<<<I, kBlock, shmem, st>>>(
      act, static_cast<const __hip_bfloat16*>(x), static_cast<const uint8_t*>(fp8_gu),
      static_cast<const __hip_bfloat16*>(s_gu), I, H);
  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(y), act, static_cast<const uint8_t*>(fp8_dn),
      static_cast<const __hip_bfloat16*>(s_dn), I, H, alpha, beta);
  return hipGetLastError() == hipSuccess;
}

bool ExpertGeGLUFp8TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                              const void* const* s_gu, const void* const* fp8_dn,
                              const void* const* s_dn, const float* wts, int G, int I, int H) {
  if (G <= 0 || !ysum || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || !wts) return false;
  for (int g = 0; g < G; ++g) {
    const float beta = (g == 0) ? 0.f : 1.f;
    if (!ExpertGeGLUFp8M1Rocm(q, ysum, x, fp8_gu[g], s_gu[g], fp8_dn[g], s_dn[g], I, H, wts[g],
                              beta))
      return false;
  }
  return true;
}

}  // namespace vt::rocm

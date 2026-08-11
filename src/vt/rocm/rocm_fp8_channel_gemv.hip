// Fast FP8-E4M3FN weight × BF16 activation kernels for Gemma-4 MoE.
// Channel scales BF16 per output row: y[m,n] = alpha * scale[n] * dot(x[m], dequant(W[n])) + beta*y
//
// M=1: fused Expert GeGLU (decode).
// M>1: weight stays FP8 — one dequant of W row into LDS, then all M acts (prefill batch).
// No host/device expand of full expert packs to BF16.
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <vector>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "vt/device.h"
#include "vt/dtype.h"

// rocWMMA FP8 expert tiles — gfx12 only (same gate as paged_attn SharedK).
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#define VT_EXPERT_WMMA_OK 1
#endif

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;
// WMMA tile (gfx12 wave32 fp8×fp8→f32).
constexpr int kWmma = 16;

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

// Host twin for constant-LUT init (same E4M3FN mapping as device).
// NOTE: __constant__ FP8 LUT A/B'd 2026-08-10 and REGRESSED ~10–15% e2e
// (irregular weight-byte indices thrash const cache; shared LDS LUT wins). Keep shared.

// Shared 256-entry FP8 E4M3 LUT — much cheaper than ldexpf per weight load.
__device__ inline void FillF8Lut(float lut[256]) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 256) lut[tid] = F8E4M3ToF32(static_cast<uint8_t>(tid));
  __syncthreads();
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

// Fewer barriers than BlockSum256 — warp reduce then 8-way (blockDim=256).
__device__ inline float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
  return v;
}
__device__ inline float BlockSumFast(float v) {
  __shared__ float wsum[8];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = WarpSum(v);
  if (lane == 0) wsum[warp] = v;
  __syncthreads();
  // Only warp 0 reduces the 8 warp sums; publish to wsum[0] for all warps.
  if (warp == 0) {
    v = (lane < 8) ? wsum[lane] : 0.f;
    v = WarpSum(v);
    if (lane == 0) wsum[0] = v;
  }
  __syncthreads();
  return wsum[0];
}

// Dual reduce (gate+up) — half the barriers of two BlockSumFast calls.
// Lab 2026-08-10: ExpertGeGLU act is MoE hot path after peer-indexed fix.
// Supports 1–8 warps (TPB 32–256); zero-fills unused warp slots.
__device__ inline void BlockSumFast2(float& a, float& b) {
  __shared__ float wa[8];
  __shared__ float wb[8];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  a = WarpSum(a);
  b = WarpSum(b);
  if (lane == 0) {
    wa[warp] = a;
    wb[warp] = b;
  }
  __syncthreads();
  if (warp == 0) {
    a = (lane < nwarps) ? wa[lane] : 0.f;
    b = (lane < nwarps) ? wb[lane] : 0.f;
    a = WarpSum(a);
    b = WarpSum(b);
    if (lane == 0) {
      wa[0] = a;
      wb[0] = b;
    }
  }
  __syncthreads();
  a = wa[0];
  b = wb[0];
}

// Quad reduce for 2-I act pack (gate0,up0,gate1,up1) — one barrier tree.
__device__ inline void BlockSumFast4(float& a, float& b, float& c, float& d) {
  __shared__ float w[4][8];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  a = WarpSum(a);
  b = WarpSum(b);
  c = WarpSum(c);
  d = WarpSum(d);
  if (lane == 0) {
    w[0][warp] = a;
    w[1][warp] = b;
    w[2][warp] = c;
    w[3][warp] = d;
  }
  __syncthreads();
  if (warp == 0) {
    a = (lane < nwarps) ? w[0][lane] : 0.f;
    b = (lane < nwarps) ? w[1][lane] : 0.f;
    c = (lane < nwarps) ? w[2][lane] : 0.f;
    d = (lane < nwarps) ? w[3][lane] : 0.f;
    a = WarpSum(a);
    b = WarpSum(b);
    c = WarpSum(c);
    d = WarpSum(d);
    if (lane == 0) {
      w[0][0] = a;
      w[1][0] = b;
      w[2][0] = c;
      w[3][0] = d;
    }
  }
  __syncthreads();
  a = w[0][0];
  b = w[1][0];
  c = w[2][0];
  d = w[3][0];
}

// Vectorized gate/up accumulate: float4 x + packed u8×4 weights (aligned H chunks).
__device__ inline void AccGateUpVec16(float& gate, float& up, const float* __restrict__ x_f,
                                      const uint8_t* __restrict__ grow, const uint8_t* __restrict__ urow,
                                      const float* __restrict__ lut, int h0) {
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int base = h0 + t * 4;
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    gate += xv.x * lut[gw & 255u];
    up += xv.x * lut[uw & 255u];
    gate += xv.y * lut[(gw >> 8) & 255u];
    up += xv.y * lut[(uw >> 8) & 255u];
    gate += xv.z * lut[(gw >> 16) & 255u];
    up += xv.z * lut[(uw >> 16) & 255u];
    gate += xv.w * lut[(gw >> 24) & 255u];
    up += xv.w * lut[(uw >> 24) & 255u];
  }
}

// gfx1201/RDNA4: hardware OCP E4M3 → f32 (no LDS LUT). Word holds 4 fp8 bytes; byte_sel 0..3.
// Lab: VT_GEMMA4_FP8_HW_CVT=1. Must match software F8E4M3ToF32 for quality.
#if defined(__gfx1200__) || defined(__gfx1201__)
__device__ inline float Fp8E4M3HwFromByte(uint8_t b) {
  return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(b), 0);
}
__device__ inline void Fp8E4M3HwFromU32(uint32_t w, float& o0, float& o1, float& o2, float& o3) {
  o0 = __builtin_amdgcn_cvt_f32_fp8(w, 0);
  o1 = __builtin_amdgcn_cvt_f32_fp8(w, 1);
  o2 = __builtin_amdgcn_cvt_f32_fp8(w, 2);
  o3 = __builtin_amdgcn_cvt_f32_fp8(w, 3);
}
#else
__device__ inline float Fp8E4M3HwFromByte(uint8_t b) { return F8E4M3ToF32(b); }
__device__ inline void Fp8E4M3HwFromU32(uint32_t w, float& o0, float& o1, float& o2, float& o3) {
  o0 = F8E4M3ToF32(static_cast<uint8_t>(w & 255u));
  o1 = F8E4M3ToF32(static_cast<uint8_t>((w >> 8) & 255u));
  o2 = F8E4M3ToF32(static_cast<uint8_t>((w >> 16) & 255u));
  o3 = F8E4M3ToF32(static_cast<uint8_t>((w >> 24) & 255u));
}
#endif

__device__ inline void AccGateUpVec16Hw(float& gate, float& up, const float* __restrict__ x_f,
                                        const uint8_t* __restrict__ grow,
                                        const uint8_t* __restrict__ urow, int h0) {
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int base = h0 + t * 4;
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    float g0, g1, g2, g3, u0, u1, u2, u3;
    Fp8E4M3HwFromU32(gw, g0, g1, g2, g3);
    Fp8E4M3HwFromU32(uw, u0, u1, u2, u3);
    gate += xv.x * g0 + xv.y * g1 + xv.z * g2 + xv.w * g3;
    up += xv.x * u0 + xv.y * u1 + xv.z * u2 + xv.w * u3;
  }
}

__device__ inline float gelu_tanh(float g) {
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.f + tanhf(inner));
}

// gu pack row layout: pair=0 → [gate_0..gate_{I-1}, up_0..up_{I-1}]
//                   pair=1 → [g0,u0,g1,u1,…] (VT_GEMMA4_GU_INTERLEAVE)
__device__ inline void GuGateUpRows(const uint8_t* __restrict__ gu, int i, int I, int H, int pair,
                                    const uint8_t*& grow, const uint8_t*& urow) {
  if (pair) {
    grow = gu + static_cast<int64_t>(2 * i) * H;
    urow = grow + H;
  } else {
    grow = gu + static_cast<int64_t>(i) * H;
    urow = gu + static_cast<int64_t>(I + i) * H;
  }
}
__device__ inline void GuGateUpScales(const __hip_bfloat16* __restrict__ sg, int i, int I, int pair,
                                      float& sgate, float& sup) {
  if (pair) {
    sgate = __bfloat162float(sg[2 * i]);
    sup = __bfloat162float(sg[2 * i + 1]);
  } else {
    sgate = __bfloat162float(sg[i]);
    sup = __bfloat162float(sg[I + i]);
  }
}

__device__ inline float DotFp8Row(const float* __restrict__ x_cache, const uint8_t* __restrict__ wrow,
                                  int K) {
  float acc = 0.f;
  const int tid = static_cast<int>(threadIdx.x);
  const int K16 = K & ~15;
  for (int k = tid * 16; k < K16; k += kBlock * 16) {
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

// Dot x_bf16[K] with pre-dequantized float weight row w_f[K].
__device__ inline float DotBf16xFloat(const __hip_bfloat16* __restrict__ xrow,
                                      const float* __restrict__ w_f, int K) {
  float acc = 0.f;
  const int tid = static_cast<int>(threadIdx.x);
  const int K4 = K & ~3;
  for (int k = tid * 4; k < K4; k += kBlock * 4) {
    acc += __bfloat162float(xrow[k + 0]) * w_f[k + 0];
    acc += __bfloat162float(xrow[k + 1]) * w_f[k + 1];
    acc += __bfloat162float(xrow[k + 2]) * w_f[k + 2];
    acc += __bfloat162float(xrow[k + 3]) * w_f[k + 3];
  }
  for (int k = K4 + tid; k < K; k += kBlock) {
    acc += __bfloat162float(xrow[k]) * w_f[k];
  }
  return BlockSum256(acc);
}

// M=1 GEMV: grid N, shmem K floats for x. FP8 via shared LUT.
__global__ void Fp8ChannelGemvFastKernel(__hip_bfloat16* __restrict__ y,
                                         const __hip_bfloat16* __restrict__ x,
                                         const uint8_t* __restrict__ W,  // [N,K]
                                         const __hip_bfloat16* __restrict__ scale, int N, int K,
                                         float alpha, float beta) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int k = tid; k < K; k += kBlock) x_cache[k] = __bfloat162float(x[k]);
  __syncthreads();

  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  float acc = 0.f;
  const int K16 = K & ~15;
  for (int k = tid * 16; k < K16; k += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) acc += x_cache[k + t] * lut[wrow[k + t]];
  }
  for (int k = K16 + tid; k < K; k += kBlock) acc += x_cache[k] * lut[wrow[k]];
  acc = BlockSum256(acc);
  if (tid == 0) {
    const float s = __bfloat162float(scale[n]);
    float v = alpha * s * acc;
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

// M>=1 GEMM: y[M,N] = alpha * (x[M,K] @ W[N,K]^T) * scale[N] + beta * y
// grid = N. Dequant weight row once into LDS; stream all M activations.
// shmem = K floats (weight row). Weights stay FP8 in global memory.
__global__ void Fp8ChannelGemmMKernel(__hip_bfloat16* __restrict__ y,  // [M,N]
                                      const __hip_bfloat16* __restrict__ x,  // [M,K]
                                      const uint8_t* __restrict__ W,         // [N,K]
                                      const __hip_bfloat16* __restrict__ scale, int M, int N,
                                      int K, float alpha, float beta) {
  extern __shared__ float smem[];
  float* w_f = smem;
  const int tid = static_cast<int>(threadIdx.x);
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;

  __shared__ float lut[256];
  // Fill lut once (all threads); then dequant W row via lut into w_f.
  {
    if (tid < 256) lut[tid] = F8E4M3ToF32(static_cast<uint8_t>(tid));
    __syncthreads();
  }
  const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  for (int k = tid; k < K; k += kBlock) w_f[k] = lut[wrow[k]];
  __syncthreads();

  const float s = __bfloat162float(scale[n]);
  for (int m = 0; m < M; ++m) {
    const __hip_bfloat16* xrow = x + static_cast<int64_t>(m) * static_cast<int64_t>(K);
    float acc = DotBf16xFloat(xrow, w_f, K);
    if (tid == 0) {
      __hip_bfloat16* ymn = y + static_cast<int64_t>(m) * static_cast<int64_t>(N) + n;
      float v = alpha * s * acc;
      if (beta != 0.f) v += beta * __bfloat162float(*ymn);
      *ymn = __float2bfloat16(v);
    }
    // ensure all threads done before next m reuses reduction sm in DotBf16xFloat
    __syncthreads();
  }
}

// Fused gate|up act T=1: grid = I. LUT + optional global x_f32.
__global__ void ExpertFp8ActKernel(float* __restrict__ act, const __hip_bfloat16* __restrict__ x,
                                   const uint8_t* __restrict__ w_gu,
                                   const __hip_bfloat16* __restrict__ s_gu, int I, int H, int pair) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;
  __shared__ float lut[256];
  FillF8Lut(lut);
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int h = tid; h < H; h += kBlock) x_cache[h] = __bfloat162float(x[h]);
  __syncthreads();

  const uint8_t* grow;
  const uint8_t* urow;
  GuGateUpRows(w_gu, i, I, H, pair, grow, urow);
  float gate = 0.f, up = 0.f;
  const int H16 = H & ~15;
  for (int h = tid * 16; h < H16; h += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int hh = h + t;
      const float xv = x_cache[hh];
      gate += xv * lut[grow[hh]];
      up += xv * lut[urow[hh]];
    }
  }
  for (int h = H16 + tid; h < H; h += kBlock) {
    const float xv = x_cache[h];
    gate += xv * lut[grow[h]];
    up += xv * lut[urow[h]];
  }
  gate = BlockSum256(gate);
  up = BlockSum256(up);
  if (tid == 0) {
    float sg, su;
    GuGateUpScales(s_gu, i, I, pair, sg, su);
    act[i] = gelu_tanh(sg * gate) * (su * up);
  }
}

__global__ void ExpertFp8DownKernel(__hip_bfloat16* __restrict__ y, const float* __restrict__ act,
                                    const uint8_t* __restrict__ w_dn,
                                    const __hip_bfloat16* __restrict__ s_dn, int I, int H,
                                    float alpha, float beta) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  const uint8_t* drow = w_dn + static_cast<int64_t>(h) * I;
  float acc = 0.f;
  int i = 0;
  for (; i + 3 < I; i += 4) {
    acc += act[i] * lut[drow[i]];
    acc += act[i + 1] * lut[drow[i + 1]];
    acc += act[i + 2] * lut[drow[i + 2]];
    acc += act[i + 3] * lut[drow[i + 3]];
  }
  for (; i < I; ++i) acc += act[i] * lut[drow[i]];
  const float sd = __bfloat162float(s_dn[h]);
  float v = alpha * sd * acc;
  if (beta != 0.f) v += beta * __bfloat162float(y[h]);
  y[h] = __float2bfloat16(v);
}

__global__ void Bf16ToF32VecKernel(float* __restrict__ out, const __hip_bfloat16* __restrict__ in,
                                   int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = __bfloat162float(in[i]);
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

void MatmulBTFp8ChannelRocm(Queue& q, void* out, const void* a, const void* b_fp8,
                            const void* scale_bf16, int M, int N, int K, float alpha,
                            float beta) {
  if (M <= 0 || N <= 0 || K <= 0 || !out || !a || !b_fp8 || !scale_bf16) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const size_t shmem = static_cast<size_t>(K) * sizeof(float);
  if (shmem > 48 * 1024) return;

  if (M == 1) {
    Fp8ChannelGemvFastKernel<<<N, kBlock, shmem, s>>>(
        static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
        static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), N, K,
        alpha, beta);
    return;
  }

  // Prefill / multi-token: keep W in FP8; dequant one row at a time into LDS.
  // NOTE: serial-in-M — prefer DequantFp8ChannelBf16Rocm + hipBLAS for large M.
  Fp8ChannelGemmMKernel<<<N, kBlock, shmem, s>>>(
      static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
      static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), M, N, K,
      alpha, beta);
}

// out[n,k] = scale[n] * f8_e4m3_to_f32(w[n,k]) as bf16
__global__ void Fp8ChannelDequantBf16Kernel(__hip_bfloat16* __restrict__ out,
                                            const uint8_t* __restrict__ w,
                                            const __hip_bfloat16* __restrict__ scale, int N,
                                            int K) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(K);
  if (idx >= total) return;
  const int n = static_cast<int>(idx / K);
  const float s = __bfloat162float(scale[n]);
  const float v = s * F8E4M3ToF32(w[idx]);
  out[idx] = __float2bfloat16(v);
}

void DequantFp8ChannelBf16Rocm(Queue& q, void* out_bf16, const void* fp8, const void* scale_bf16,
                               int N, int K) {
  if (!out_bf16 || !fp8 || !scale_bf16 || N <= 0 || K <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(K);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  Fp8ChannelDequantBf16Kernel<<<blocks, threads, 0, s>>>(
      static_cast<__hip_bfloat16*>(out_bf16), static_cast<const uint8_t*>(fp8),
      static_cast<const __hip_bfloat16*>(scale_bf16), N, K);
}

bool ExpertGeGLUFp8M1Rocm(Queue& q, void* y, const void* x, const void* fp8_gu, const void* s_gu,
                          const void* fp8_dn, const void* s_dn, int I, int H, float alpha,
                          float beta) {
  if (!y || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || I <= 0 || H <= 0) return false;
  const size_t shmem = static_cast<size_t>(H) * sizeof(float);
  if (shmem > 48 * 1024) return false;
  if (!EnsureAct(I)) return false;
  static const int gu_pair = [] {
    const char* e = std::getenv("VT_GEMMA4_GU_INTERLEAVE");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  float* act = TlsAct().p;
  ExpertFp8ActKernel<<<I, kBlock, shmem, st>>>(
      act, static_cast<const __hip_bfloat16*>(x), static_cast<const uint8_t*>(fp8_gu),
      static_cast<const __hip_bfloat16*>(s_gu), I, H, gu_pair);
  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(y), act, static_cast<const uint8_t*>(fp8_dn),
      static_cast<const __hip_bfloat16*>(s_dn), I, H, alpha, beta);
  return hipGetLastError() == hipSuccess;
}

// --- True top-k fused FP8 ExpertGeGLU (decode T=1) ---------------------------
// 1 act grid (I) looping G + 1 down. (Multi-row/LDS-act tiled variants regressed
// on gfx1201 lab A/B 2026-08-09 — keep this path.)
constexpr int kMaxTopK = 8;

// grid = I — x already expanded to float in global (L2-resident); no per-block x smem.
__global__ void ExpertFp8ActTopKKernel(float* __restrict__ act,  // [G,I]
                                       const float* __restrict__ x_f,  // [H]
                                       const uint8_t* const* __restrict__ w_gu,
                                       const __hip_bfloat16* const* __restrict__ s_gu, int G,
                                       int I, int H, int pair) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);

  for (int g = 0; g < G; ++g) {
    const uint8_t* gu = w_gu[g];
    const __hip_bfloat16* sg = s_gu[g];
    const uint8_t* grow;
    const uint8_t* urow;
    GuGateUpRows(gu, i, I, H, pair, grow, urow);
    float gate = 0.f, up = 0.f;
    const int H16 = H & ~15;
    for (int h = tid * 16; h < H16; h += kBlock * 16) {
      AccGateUpVec16(gate, up, x_f, grow, urow, lut, h);
    }
    for (int h = H16 + tid; h < H; h += kBlock) {
      const float xv = x_f[h];
      gate += xv * lut[grow[h]];
      up += xv * lut[urow[h]];
    }
    BlockSumFast2(gate, up);
    if (tid == 0) {
      float sgate, sup;
      GuGateUpScales(sg, i, I, pair, sgate, sup);
      act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
    }
    // No trailing __syncthreads: next g only uses private regs + BlockSumFast2 barriers.
  }
}

__global__ void ExpertFp8DownTopKKernel(__hip_bfloat16* __restrict__ ysum,
                                        const float* __restrict__ act,  // [G,I]
                                        const uint8_t* const* __restrict__ w_dn,
                                        const __hip_bfloat16* const* __restrict__ s_dn,
                                        const float* __restrict__ wts, int G, int I, int H) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  float acc = 0.f;
  for (int g = 0; g < G; ++g) {
    const uint8_t* drow = w_dn[g] + static_cast<int64_t>(h) * I;
    const float* arow = act + static_cast<int64_t>(g) * I;
    float dot = 0.f;
    int i = 0;
    for (; i + 3 < I; i += 4) {
      dot += arow[i] * lut[drow[i]];
      dot += arow[i + 1] * lut[drow[i + 1]];
      dot += arow[i + 2] * lut[drow[i + 2]];
      dot += arow[i + 3] * lut[drow[i + 3]];
    }
    for (; i < I; ++i) dot += arow[i] * lut[drow[i]];
    acc += wts[g] * __bfloat162float(s_dn[g][h]) * dot;
  }
  ysum[h] = __float2bfloat16(acc);
}

struct TopKTls {
  const void** d_fgu = nullptr;
  const void** d_sgu = nullptr;
  const void** d_fdn = nullptr;
  const void** d_sdn = nullptr;
  float* d_wts = nullptr;
  float* d_act = nullptr;
  int act_cap = 0;
};
TopKTls& TlsTopK() {
  static thread_local TopKTls t;
  return t;
}

bool ExpertGeGLUFp8TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                              const void* const* s_gu, const void* const* fp8_dn,
                              const void* const* s_dn, const float* wts, int G, int I, int H) {
  if (G <= 0 || G > kMaxTopK || !ysum || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || !wts)
    return false;
  if (I <= 0 || H <= 0) return false;
  // Act no longer needs H*f32 dynamic smem (x expanded to global).
  if (H <= 0) return false;

  static const int gu_pair = [] {
    const char* e = std::getenv("VT_GEMMA4_GU_INTERLEAVE");
    return (e && e[0] == '1') ? 1 : 0;
  }();

  const int dev = q.device.index;
  if (dev < 0 || dev >= 8) return false;
  {
    int cur = -1;
    if (hipGetDevice(&cur) != hipSuccess || cur != dev) {
      if (hipSetDevice(dev) != hipSuccess) return false;
    }
  }

  // Per-device TLS: dual-GPU expert-run must not reuse GPU0 scratch on GPU1.
  struct TopKDevTls {
    const void** d_fgu = nullptr;
    const void** d_sgu = nullptr;
    const void** d_fdn = nullptr;
    const void** d_sdn = nullptr;
    float* d_wts = nullptr;
    float* d_act = nullptr;
    float* d_xf = nullptr;
    int act_cap = 0;
    int xf_cap = 0;
    int bound_dev = -1;
  };
  static thread_local TopKDevTls slots[8];
  auto& tls = slots[dev];
  if (tls.bound_dev != dev) {
    tls = TopKDevTls{};
    tls.bound_dev = dev;
  }

  const int act_need = G * I;
  if (tls.act_cap < act_need || !tls.d_act) {
    if (tls.d_act) (void)hipFree(tls.d_act);
    tls.d_act = nullptr;
    tls.act_cap = 0;
    if (hipMalloc(&tls.d_act, static_cast<size_t>(act_need) * sizeof(float)) != hipSuccess)
      return false;
    tls.act_cap = act_need;
  }
  if (tls.xf_cap < H || !tls.d_xf) {
    if (tls.d_xf) (void)hipFree(tls.d_xf);
    tls.d_xf = nullptr;
    tls.xf_cap = 0;
    if (hipMalloc(&tls.d_xf, static_cast<size_t>(H) * sizeof(float)) != hipSuccess) return false;
    tls.xf_cap = H;
  }
  if (!tls.d_fgu) {
    if (hipMalloc(&tls.d_fgu, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_sgu, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_fdn, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_sdn, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_wts, sizeof(float) * kMaxTopK) != hipSuccess) return false;
  }

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  if (hipMemcpyAsync(tls.d_fgu, fp8_gu, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_sgu, s_gu, sizeof(void*) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_fdn, fp8_dn, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_sdn, s_dn, sizeof(void*) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_wts, wts, sizeof(float) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;

  // Expand x once → L2; act blocks read x_f (no 704× smem fills of H).
  {
    const int xt = 256;
    const int xg = (H + xt - 1) / xt;
    Bf16ToF32VecKernel<<<xg, xt, 0, st>>>(tls.d_xf, static_cast<const __hip_bfloat16*>(x), H);
  }
  dim3 grid_act(static_cast<unsigned>(I));
  ExpertFp8ActTopKKernel<<<grid_act, kBlock, 0, st>>>(
      tls.d_act, tls.d_xf, reinterpret_cast<const uint8_t* const*>(tls.d_fgu),
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_sgu), G, I, H, gu_pair);

  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownTopKKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(ysum), tls.d_act,
      reinterpret_cast<const uint8_t* const*>(tls.d_fdn),
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_sdn), tls.d_wts, G, I, H);

  return hipGetLastError() == hipSuccess;
}

// Contiguous FP8 packs + device top-k indices (no host pointer gather / D2H).
// grid = I — one block per intermediate row; loop G experts in-CTA for x_f L1 reuse.
// (Prior grid=(I,G) A/B alternative; G-loop keeps 704 CTAs vs 5632.)
// gu_base [E,2I,H] u8, sgu [E,2I] bf16.
// x_f is float [H] expanded once per decode step (same as ExpertFp8ActTopKKernel) —
// avoids G*I redundant bf16→f32 loads of x (H=2816, I=704, G~8 → huge).
// NOTE: LDS staging of x_f was A/B'd and REGRESSED (~−5% e2e @short–30k): extra
// syncthreads + ~11KB smem/CTA cut occupancy; L2 already hits x_f across I×G CTAs.
__global__ void ExpertFp8ActTopKIndexedKernel(float* __restrict__ act,
                                              const float* __restrict__ x_f,
                                              const uint8_t* __restrict__ gu_base,
                                              const __hip_bfloat16* __restrict__ sgu_base,
                                              const int32_t* __restrict__ idx, int G, int I, int H,
                                              int pair) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I || G <= 0) return;
  __shared__ float lut[256];
  __shared__ int sidx[kMaxTopK];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const int H16 = H & ~15;

  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
    const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
    const uint8_t* grow;
    const uint8_t* urow;
    GuGateUpRows(gu, i, I, H, pair, grow, urow);

    float gate = 0.f, up = 0.f;
    for (int h = tid * 16; h < H16; h += kBlock * 16) {
      AccGateUpVec16(gate, up, x_f, grow, urow, lut, h);
    }
    for (int h = H16 + tid; h < H; h += kBlock) {
      const float xv = x_f[h];
      gate += xv * lut[grow[h]];
      up += xv * lut[urow[h]];
    }
    BlockSumFast2(gate, up);
    if (tid == 0) {
      float sgate, sup;
      GuGateUpScales(sg, i, I, pair, sgate, sup);
      act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
    }
    // No trailing barrier: BlockSumFast2 already syncs; act[g] not read by later g.
  }
}

// Hardware FP8 convert path — no 1KB LDS LUT (gfx1201 OCP).
// K-loop strides use blockDim.x (supports VT_GEMMA4_ACT_TPB=128/192/256).
// pair=1: gu rows (g0,u0,g1,u1,…) from VT_GEMMA4_GU_INTERLEAVE pack.
__global__ void ExpertFp8ActTopKIndexedKernelHw(float* __restrict__ act,
                                                const float* __restrict__ x_f,
                                                const uint8_t* __restrict__ gu_base,
                                                const __hip_bfloat16* __restrict__ sgu_base,
                                                const int32_t* __restrict__ idx, int G, int I,
                                                int H, int pair) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I || G <= 0) return;
  __shared__ int sidx[kMaxTopK];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const int H16 = H & ~15;

  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
    const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
    const uint8_t* grow;
    const uint8_t* urow;
    GuGateUpRows(gu, i, I, H, pair, grow, urow);

    float gate = 0.f, up = 0.f;
    for (int h = tid * 16; h < H16; h += tpb * 16) {
      AccGateUpVec16Hw(gate, up, x_f, grow, urow, h);
    }
    for (int h = H16 + tid; h < H; h += tpb) {
      const float xv = x_f[h];
      gate += xv * Fp8E4M3HwFromByte(grow[h]);
      up += xv * Fp8E4M3HwFromByte(urow[h]);
    }
    BlockSumFast2(gate, up);
    if (tid == 0) {
      float sgate, sup;
      GuGateUpScales(sg, i, I, pair, sgate, sup);
      act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
    }
  }
}

// Dense H-strip HW act: every thread owns float4 chunks (k=tid; k+=tpb).
// KEEP *KernelHw uses h=tid*16 stride tpb*16 → only 176/256 threads work @H=2816.
// Env: VT_GEMMA4_ACT_DENSE=1. Optional VT_GEMMA4_ACT_PAR_G=1 → grid=I*G (one expert/CTA).
__global__ void ExpertFp8ActTopKIndexedKernelHwDense(float* __restrict__ act,
                                                     const float* __restrict__ x_f,
                                                     const uint8_t* __restrict__ gu_base,
                                                     const __hip_bfloat16* __restrict__ sgu_base,
                                                     const int32_t* __restrict__ idx, int G, int I,
                                                     int H, int pair, int par_g) {
  int i, g0, g1;
  if (par_g) {
    // grid = I * G
    const int bid = static_cast<int>(blockIdx.x);
    i = bid % I;
    g0 = bid / I;
    g1 = g0 + 1;
    if (g0 >= G || i >= I) return;
  } else {
    i = static_cast<int>(blockIdx.x);
    if (i >= I || G <= 0) return;
    g0 = 0;
    g1 = G;
  }
  __shared__ int sidx[kMaxTopK];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const int H4 = H & ~3;
  const int n4 = H4 >> 2;  // float4 count

  for (int g = g0; g < g1; ++g) {
    const int e = sidx[g];
    const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
    const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
    const uint8_t* grow;
    const uint8_t* urow;
    GuGateUpRows(gu, i, I, H, pair, grow, urow);

    float gate = 0.f, up = 0.f;
    for (int k = tid; k < n4; k += tpb) {
      const int base = k * 4;
      const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
      const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
      const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
      float gg0, gg1, gg2, gg3, uu0, uu1, uu2, uu3;
      Fp8E4M3HwFromU32(gw, gg0, gg1, gg2, gg3);
      Fp8E4M3HwFromU32(uw, uu0, uu1, uu2, uu3);
      gate += xv.x * gg0 + xv.y * gg1 + xv.z * gg2 + xv.w * gg3;
      up += xv.x * uu0 + xv.y * uu1 + xv.z * uu2 + xv.w * uu3;
    }
    for (int h = H4 + tid; h < H; h += tpb) {
      const float xv = x_f[h];
      gate += xv * Fp8E4M3HwFromByte(grow[h]);
      up += xv * Fp8E4M3HwFromByte(urow[h]);
    }
    BlockSumFast2(gate, up);
    if (tid == 0) {
      float sgate, sup;
      GuGateUpScales(sg, i, I, pair, sgate, sup);
      act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
    }
  }
}

// One-warp act per (i,g): grid=I*G, tpb=32. Avoids 256-thread BlockSum + idle lanes.
// Env: VT_GEMMA4_ACT_WARP_I=1 (implies par-g geometry).
__global__ void ExpertFp8ActTopKIndexedKernelHwWarpI(float* __restrict__ act,
                                                     const float* __restrict__ x_f,
                                                     const uint8_t* __restrict__ gu_base,
                                                     const __hip_bfloat16* __restrict__ sgu_base,
                                                     const int32_t* __restrict__ idx, int G, int I,
                                                     int H, int pair) {
  const int bid = static_cast<int>(blockIdx.x);
  const int i = bid % I;
  const int g = bid / I;
  if (g >= G || i >= I) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  // Broadcast expert id (only lane 0 loads idx — full warp needs same e).
  __shared__ int sidx[kMaxTopK];
  if (lane < G) sidx[lane] = idx[lane];
  __syncthreads();
  const int e = sidx[g];

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
  const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
  const uint8_t* grow;
  const uint8_t* urow;
  GuGateUpRows(gu, i, I, H, pair, grow, urow);

  float gate = 0.f, up = 0.f;
  const int H4 = H & ~3;
  // 32 lanes × float4 = 128 elems/round
  const int n_vec = H4 / 128;
  for (int t = 0; t < n_vec; ++t) {
    const int base = lane * 4 + t * 128;
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    float g0, g1, g2, g3, u0, u1, u2, u3;
    Fp8E4M3HwFromU32(gw, g0, g1, g2, g3);
    Fp8E4M3HwFromU32(uw, u0, u1, u2, u3);
    gate += xv.x * g0 + xv.y * g1 + xv.z * g2 + xv.w * g3;
    up += xv.x * u0 + xv.y * u1 + xv.z * u2 + xv.w * u3;
  }
  for (int base = n_vec * 128 + lane * 4; base < H4; base += 128) {
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    float g0, g1, g2, g3, u0, u1, u2, u3;
    Fp8E4M3HwFromU32(gw, g0, g1, g2, g3);
    Fp8E4M3HwFromU32(uw, u0, u1, u2, u3);
    gate += xv.x * g0 + xv.y * g1 + xv.z * g2 + xv.w * g3;
    up += xv.x * u0 + xv.y * u1 + xv.z * u2 + xv.w * u3;
  }
  for (int h = H4 + lane; h < H; h += 32) {
    const float xv = x_f[h];
    gate += xv * Fp8E4M3HwFromByte(grow[h]);
    up += xv * Fp8E4M3HwFromByte(urow[h]);
  }
  gate = WarpSum(gate);
  up = WarpSum(up);
  if (lane == 0) {
    float sgate, sup;
    GuGateUpScales(sg, i, I, pair, sgate, sup);
    act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
  }
}

// Warp-per-expert HW act (lab 2026-08-11).
// Geometry: H=2816 → only 176 of 256 threads work in *KernelHw (h=tid*16, stride 4096).
// G experts were serial → wall = G × GEMV. Here: CTA = G warps (G≤8), each warp one expert;
// all 32*G threads busy; 2816/4/32 = 22 float4 steps/lane exactly. Env: VT_GEMMA4_ACT_WARPG=1.
__global__ void ExpertFp8ActTopKIndexedKernelHwWarpG(float* __restrict__ act,
                                                     const float* __restrict__ x_f,
                                                     const uint8_t* __restrict__ gu_base,
                                                     const __hip_bfloat16* __restrict__ sgu_base,
                                                     const int32_t* __restrict__ idx, int G, int I,
                                                     int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I || G <= 0) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ int sidx[kMaxTopK];
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();
  if (warp >= G) return;

  const int g = warp;
  const int e = sidx[g];
  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
  const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
  const uint8_t* grow = gu + static_cast<int64_t>(i) * H;
  const uint8_t* urow = gu + static_cast<int64_t>(I + i) * H;

  float gate = 0.f, up = 0.f;
  // Prefer aligned float4 path when H is multiple of 128 (32 lanes × 4).
  const int H4 = H & ~3;
  const int n_vec = H4 / 128;  // full rounds where every lane has a float4
  // Round t: base = lane*4 + t*128 covers [0, n_vec*128).
  for (int t = 0; t < n_vec; ++t) {
    const int base = lane * 4 + t * 128;
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    float g0, g1, g2, g3, u0, u1, u2, u3;
    Fp8E4M3HwFromU32(gw, g0, g1, g2, g3);
    Fp8E4M3HwFromU32(uw, u0, u1, u2, u3);
    gate += xv.x * g0 + xv.y * g1 + xv.z * g2 + xv.w * g3;
    up += xv.x * u0 + xv.y * u1 + xv.z * u2 + xv.w * u3;
  }
  // Tail: remaining float4s with base >= n_vec*128, assigned by lane if in range.
  for (int base = n_vec * 128 + lane * 4; base < H4; base += 128) {
    const float4 xv = *reinterpret_cast<const float4*>(x_f + base);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + base);
    const uint32_t uw = *reinterpret_cast<const uint32_t*>(urow + base);
    float g0, g1, g2, g3, u0, u1, u2, u3;
    Fp8E4M3HwFromU32(gw, g0, g1, g2, g3);
    Fp8E4M3HwFromU32(uw, u0, u1, u2, u3);
    gate += xv.x * g0 + xv.y * g1 + xv.z * g2 + xv.w * g3;
    up += xv.x * u0 + xv.y * u1 + xv.z * u2 + xv.w * u3;
  }
  for (int h = H4 + lane; h < H; h += 32) {
    const float xv = x_f[h];
    gate += xv * Fp8E4M3HwFromByte(grow[h]);
    up += xv * Fp8E4M3HwFromByte(urow[h]);
  }
  gate = WarpSum(gate);
  up = WarpSum(up);
  if (lane == 0) {
    const float sgate = __bfloat162float(sg[i]);
    const float sup = __bfloat162float(sg[I + i]);
    act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
  }
}

// HW + 2-I pack: half launches, dual-issue dots, x_f L2 reuse across two intermediate rows.
// Prior I2 was LUT-only and flat; HW path may differ. Env uses ACT_I_PACK=2 + HW.
__global__ void ExpertFp8ActTopKIndexedKernelHwI2(float* __restrict__ act,
                                                  const float* __restrict__ x_f,
                                                  const uint8_t* __restrict__ gu_base,
                                                  const __hip_bfloat16* __restrict__ sgu_base,
                                                  const int32_t* __restrict__ idx, int G, int I,
                                                  int H) {
  const int i0 = static_cast<int>(blockIdx.x) * 2;
  if (i0 >= I || G <= 0) return;
  const int i1 = i0 + 1;
  const bool has1 = i1 < I;
  __shared__ int sidx[kMaxTopK];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const int H16 = H & ~15;

  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
    const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
    const uint8_t* grow0 = gu + static_cast<int64_t>(i0) * H;
    const uint8_t* urow0 = gu + static_cast<int64_t>(I + i0) * H;
    const uint8_t* grow1 = has1 ? gu + static_cast<int64_t>(i1) * H : grow0;
    const uint8_t* urow1 = has1 ? gu + static_cast<int64_t>(I + i1) * H : urow0;

    float g0 = 0.f, u0 = 0.f, g1 = 0.f, u1 = 0.f;
    for (int h = tid * 16; h < H16; h += tpb * 16) {
      AccGateUpVec16Hw(g0, u0, x_f, grow0, urow0, h);
      if (has1) AccGateUpVec16Hw(g1, u1, x_f, grow1, urow1, h);
    }
    for (int h = H16 + tid; h < H; h += tpb) {
      const float xv = x_f[h];
      g0 += xv * Fp8E4M3HwFromByte(grow0[h]);
      u0 += xv * Fp8E4M3HwFromByte(urow0[h]);
      if (has1) {
        g1 += xv * Fp8E4M3HwFromByte(grow1[h]);
        u1 += xv * Fp8E4M3HwFromByte(urow1[h]);
      }
    }
    if (has1) {
      BlockSumFast4(g0, u0, g1, u1);
    } else {
      BlockSumFast2(g0, u0);
    }
    if (tid == 0) {
      const float sg0 = __bfloat162float(sg[i0]);
      const float su0 = __bfloat162float(sg[I + i0]);
      act[static_cast<int64_t>(g) * I + i0] = gelu_tanh(sg0 * g0) * (su0 * u0);
      if (has1) {
        const float sg1 = __bfloat162float(sg[i1]);
        const float su1 = __bfloat162float(sg[I + i1]);
        act[static_cast<int64_t>(g) * I + i1] = gelu_tanh(sg1 * g1) * (su1 * u1);
      }
    }
  }
}

// Pack 2 intermediate rows per CTA: half launches, shared LUT/sidx, dual-issue dots on x_f.
// Env VT_GEMMA4_ACT_I_PACK=1 forces legacy 1-I grid.
__global__ void ExpertFp8ActTopKIndexedKernelI2(float* __restrict__ act,
                                                const float* __restrict__ x_f,
                                                const uint8_t* __restrict__ gu_base,
                                                const __hip_bfloat16* __restrict__ sgu_base,
                                                const int32_t* __restrict__ idx, int G, int I,
                                                int H) {
  const int i0 = static_cast<int>(blockIdx.x) * 2;
  if (i0 >= I || G <= 0) return;
  const int i1 = i0 + 1;
  const bool has1 = i1 < I;
  __shared__ float lut[256];
  __shared__ int sidx[kMaxTopK];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < G) sidx[tid] = idx[tid];
  __syncthreads();

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const int H16 = H & ~15;

  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
    const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
    const uint8_t* grow0 = gu + static_cast<int64_t>(i0) * H;
    const uint8_t* urow0 = gu + static_cast<int64_t>(I + i0) * H;
    const uint8_t* grow1 = has1 ? gu + static_cast<int64_t>(i1) * H : grow0;
    const uint8_t* urow1 = has1 ? gu + static_cast<int64_t>(I + i1) * H : urow0;

    float g0 = 0.f, u0 = 0.f, g1 = 0.f, u1 = 0.f;
    for (int h = tid * 16; h < H16; h += kBlock * 16) {
      AccGateUpVec16(g0, u0, x_f, grow0, urow0, lut, h);
      if (has1) AccGateUpVec16(g1, u1, x_f, grow1, urow1, lut, h);
    }
    for (int h = H16 + tid; h < H; h += kBlock) {
      const float xv = x_f[h];
      g0 += xv * lut[grow0[h]];
      u0 += xv * lut[urow0[h]];
      if (has1) {
        g1 += xv * lut[grow1[h]];
        u1 += xv * lut[urow1[h]];
      }
    }
    if (has1) {
      BlockSumFast4(g0, u0, g1, u1);
    } else {
      BlockSumFast2(g0, u0);
    }
    if (tid == 0) {
      const float sg0 = __bfloat162float(sg[i0]);
      const float su0 = __bfloat162float(sg[I + i0]);
      act[static_cast<int64_t>(g) * I + i0] = gelu_tanh(sg0 * g0) * (su0 * u0);
      if (has1) {
        const float sg1 = __bfloat162float(sg[i1]);
        const float su1 = __bfloat162float(sg[I + i1]);
        act[static_cast<int64_t>(g) * I + i1] = gelu_tanh(sg1 * g1) * (su1 * u1);
      }
    }
  }
}

__global__ void ExpertFp8DownTopKIndexedKernel(__hip_bfloat16* __restrict__ ysum,
                                               const float* __restrict__ act,
                                               const uint8_t* __restrict__ dn_base,
                                               const __hip_bfloat16* __restrict__ sdn_base,
                                               const int32_t* __restrict__ idx,
                                               const float* __restrict__ wts, int G, int I, int H) {
  __shared__ float lut[256];
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  // One h per thread (legacy). Grid-stride kept for VT_GEMMA4_DOWN_GRID experiments.
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); h < H; h += stride) {
    float acc = 0.f;
    for (int g = 0; g < G; ++g) {
      const int e = sidx[g];
      const uint8_t* drow = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
      const float* arow = act + static_cast<int64_t>(g) * I;
      float dot = 0.f;
      int i = 0;
      const int I4 = I & ~3;
      for (; i < I4; i += 4) {
        const float4 av = *reinterpret_cast<const float4*>(arow + i);
        const uint32_t wp = *reinterpret_cast<const uint32_t*>(drow + i);
        dot += av.x * lut[wp & 255u];
        dot += av.y * lut[(wp >> 8) & 255u];
        dot += av.z * lut[(wp >> 16) & 255u];
        dot += av.w * lut[(wp >> 24) & 255u];
      }
      for (; i < I; ++i) dot += arow[i] * lut[drow[i]];
      const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
      acc += swts[g] * __bfloat162float(sd[h]) * dot;
    }
    ysum[h] = __float2bfloat16(acc);
  }
}

__global__ void ExpertFp8DownTopKIndexedKernelHw(__hip_bfloat16* __restrict__ ysum,
                                                 const float* __restrict__ act,
                                                 const uint8_t* __restrict__ dn_base,
                                                 const __hip_bfloat16* __restrict__ sdn_base,
                                                 const int32_t* __restrict__ idx,
                                                 const float* __restrict__ wts, int G, int I,
                                                 int H) {
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  // Stage act[G,I] once/CTA — without this every h-thread reloads full act from global
  // (~H·G·I·4 B redundant). I=704 G=8 → ~22KB LDS.
  extern __shared__ float sact[];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const int act_n = G * I;
  for (int i = tid; i < act_n; i += tpb) sact[i] = act[i];
  __syncthreads();

  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  const int I16 = I & ~15;
  for (int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); h < H; h += stride) {
    float acc = 0.f;
    for (int g = 0; g < G; ++g) {
      const int e = sidx[g];
      const uint8_t* drow = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
      const float* arow = sact + g * I;
      float dot = 0.f;
      int i = 0;
      for (; i < I16; i += 16) {
        const float4 a0 = *reinterpret_cast<const float4*>(arow + i);
        const float4 a1 = *reinterpret_cast<const float4*>(arow + i + 4);
        const float4 a2 = *reinterpret_cast<const float4*>(arow + i + 8);
        const float4 a3 = *reinterpret_cast<const float4*>(arow + i + 12);
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(drow + i);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(drow + i + 4);
        const uint32_t w2 = *reinterpret_cast<const uint32_t*>(drow + i + 8);
        const uint32_t w3 = *reinterpret_cast<const uint32_t*>(drow + i + 12);
        float p0, p1, p2, p3;
        Fp8E4M3HwFromU32(w0, p0, p1, p2, p3);
        dot += a0.x * p0 + a0.y * p1 + a0.z * p2 + a0.w * p3;
        Fp8E4M3HwFromU32(w1, p0, p1, p2, p3);
        dot += a1.x * p0 + a1.y * p1 + a1.z * p2 + a1.w * p3;
        Fp8E4M3HwFromU32(w2, p0, p1, p2, p3);
        dot += a2.x * p0 + a2.y * p1 + a2.z * p2 + a2.w * p3;
        Fp8E4M3HwFromU32(w3, p0, p1, p2, p3);
        dot += a3.x * p0 + a3.y * p1 + a3.z * p2 + a3.w * p3;
      }
      for (; i + 3 < I; i += 4) {
        const float4 av = *reinterpret_cast<const float4*>(arow + i);
        const uint32_t wp = *reinterpret_cast<const uint32_t*>(drow + i);
        float q0, q1, q2, q3;
        Fp8E4M3HwFromU32(wp, q0, q1, q2, q3);
        dot += av.x * q0 + av.y * q1 + av.z * q2 + av.w * q3;
      }
      for (; i < I; ++i) dot += arow[i] * Fp8E4M3HwFromByte(drow[i]);
      const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
      acc += swts[g] * __bfloat162float(sd[h]) * dot;
    }
    ysum[h] = __float2bfloat16(acc);
  }
}

// Warp-cooperative LDS down: one warp owns one h-row; lanes split I and warp-reduce.
// Better CU fill than 11 CTAs×256 (H=2816 → ~88 CTAs @256 TPB / 8 warps). Opt-in VT_GEMMA4_DOWN_WARP=1.
__global__ void ExpertFp8DownTopKIndexedKernelHwWarp(__hip_bfloat16* __restrict__ ysum,
                                                     const float* __restrict__ act,
                                                     const uint8_t* __restrict__ dn_base,
                                                     const __hip_bfloat16* __restrict__ sdn_base,
                                                     const int32_t* __restrict__ idx,
                                                     const float* __restrict__ wts, int G, int I,
                                                     int H) {
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  extern __shared__ float sact[];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = tpb >> 5;
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const int act_n = G * I;
  for (int i = tid; i < act_n; i += tpb) sact[i] = act[i];
  __syncthreads();

  const int h = static_cast<int>(blockIdx.x) * nwarps + warp;
  if (h >= H) return;

  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  const int I16 = I & ~15;
  float acc = 0.f;
  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* drow = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
    const float* arow = sact + g * I;
    float dot = 0.f;
    // lanes strip-mine I in 16-wide chunks (lane owns every 32nd group of 16 when possible)
    for (int base = lane * 16; base + 15 < I; base += 32 * 16) {
      const float4 a0 = *reinterpret_cast<const float4*>(arow + base);
      const float4 a1 = *reinterpret_cast<const float4*>(arow + base + 4);
      const float4 a2 = *reinterpret_cast<const float4*>(arow + base + 8);
      const float4 a3 = *reinterpret_cast<const float4*>(arow + base + 12);
      const uint32_t w0 = *reinterpret_cast<const uint32_t*>(drow + base);
      const uint32_t w1 = *reinterpret_cast<const uint32_t*>(drow + base + 4);
      const uint32_t w2 = *reinterpret_cast<const uint32_t*>(drow + base + 8);
      const uint32_t w3 = *reinterpret_cast<const uint32_t*>(drow + base + 12);
      float p0, p1, p2, p3;
      Fp8E4M3HwFromU32(w0, p0, p1, p2, p3);
      dot += a0.x * p0 + a0.y * p1 + a0.z * p2 + a0.w * p3;
      Fp8E4M3HwFromU32(w1, p0, p1, p2, p3);
      dot += a1.x * p0 + a1.y * p1 + a1.z * p2 + a1.w * p3;
      Fp8E4M3HwFromU32(w2, p0, p1, p2, p3);
      dot += a2.x * p0 + a2.y * p1 + a2.z * p2 + a2.w * p3;
      Fp8E4M3HwFromU32(w3, p0, p1, p2, p3);
      dot += a3.x * p0 + a3.y * p1 + a3.z * p2 + a3.w * p3;
    }
    // tail: scalar strip by lane
    for (int i = I16 + lane; i < I; i += 32) {
      dot += arow[i] * Fp8E4M3HwFromByte(drow[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_down(dot, off);
    }
    if (lane == 0) {
      const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
      acc += swts[g] * __bfloat162float(sd[h]) * dot;
    }
  }
  if (lane == 0) ysum[h] = __float2bfloat16(acc);
}

// Multi-h LDS down: each thread owns NH consecutive rows; reuses sact across h (ILP + better CU fill
// when paired with smaller TPB). NH=2 or 4 via template. Bounds-safe for H % NH != 0.
template <int NH>
__global__ void ExpertFp8DownTopKIndexedKernelHwNh(__hip_bfloat16* __restrict__ ysum,
                                                   const float* __restrict__ act,
                                                   const uint8_t* __restrict__ dn_base,
                                                   const __hip_bfloat16* __restrict__ sdn_base,
                                                   const int32_t* __restrict__ idx,
                                                   const float* __restrict__ wts, int G, int I,
                                                   int H) {
  static_assert(NH == 2 || NH == 4, "NH");
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  extern __shared__ float sact[];
  const int tid = static_cast<int>(threadIdx.x);
  const int tpb = static_cast<int>(blockDim.x);
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const int act_n = G * I;
  for (int i = tid; i < act_n; i += tpb) sact[i] = act[i];
  __syncthreads();

  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  // One logical "slot" per thread covers NH h-rows.
  const int nslot = (H + NH - 1) / NH;
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  const int I16 = I & ~15;
  for (int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); slot < nslot;
       slot += stride) {
    const int h0 = slot * NH;
    float acc[NH];
#pragma unroll
    for (int n = 0; n < NH; ++n) acc[n] = 0.f;
    for (int g = 0; g < G; ++g) {
      const int e = sidx[g];
      const float* arow = sact + g * I;
      const uint8_t* drow[NH];
      float sc[NH];
      const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int h = h0 + n;
        if (h < H) {
          drow[n] = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
          sc[n] = swts[g] * __bfloat162float(sd[h]);
        } else {
          drow[n] = nullptr;
          sc[n] = 0.f;
        }
      }
      float dot[NH];
#pragma unroll
      for (int n = 0; n < NH; ++n) dot[n] = 0.f;
      int i = 0;
      for (; i < I16; i += 16) {
        const float4 a0 = *reinterpret_cast<const float4*>(arow + i);
        const float4 a1 = *reinterpret_cast<const float4*>(arow + i + 4);
        const float4 a2 = *reinterpret_cast<const float4*>(arow + i + 8);
        const float4 a3 = *reinterpret_cast<const float4*>(arow + i + 12);
#pragma unroll
        for (int n = 0; n < NH; ++n) {
          if (!drow[n]) continue;
          const uint32_t w0 = *reinterpret_cast<const uint32_t*>(drow[n] + i);
          const uint32_t w1 = *reinterpret_cast<const uint32_t*>(drow[n] + i + 4);
          const uint32_t w2 = *reinterpret_cast<const uint32_t*>(drow[n] + i + 8);
          const uint32_t w3 = *reinterpret_cast<const uint32_t*>(drow[n] + i + 12);
          float p0, p1, p2, p3;
          Fp8E4M3HwFromU32(w0, p0, p1, p2, p3);
          dot[n] += a0.x * p0 + a0.y * p1 + a0.z * p2 + a0.w * p3;
          Fp8E4M3HwFromU32(w1, p0, p1, p2, p3);
          dot[n] += a1.x * p0 + a1.y * p1 + a1.z * p2 + a1.w * p3;
          Fp8E4M3HwFromU32(w2, p0, p1, p2, p3);
          dot[n] += a2.x * p0 + a2.y * p1 + a2.z * p2 + a2.w * p3;
          Fp8E4M3HwFromU32(w3, p0, p1, p2, p3);
          dot[n] += a3.x * p0 + a3.y * p1 + a3.z * p2 + a3.w * p3;
        }
      }
      for (; i + 3 < I; i += 4) {
        const float4 av = *reinterpret_cast<const float4*>(arow + i);
#pragma unroll
        for (int n = 0; n < NH; ++n) {
          if (!drow[n]) continue;
          const uint32_t wp = *reinterpret_cast<const uint32_t*>(drow[n] + i);
          float q0, q1, q2, q3;
          Fp8E4M3HwFromU32(wp, q0, q1, q2, q3);
          dot[n] += av.x * q0 + av.y * q1 + av.z * q2 + av.w * q3;
        }
      }
      for (; i < I; ++i) {
        const float av = arow[i];
#pragma unroll
        for (int n = 0; n < NH; ++n) {
          if (!drow[n]) continue;
          dot[n] += av * Fp8E4M3HwFromByte(drow[n][i]);
        }
      }
#pragma unroll
      for (int n = 0; n < NH; ++n) acc[n] += sc[n] * dot[n];
    }
#pragma unroll
    for (int n = 0; n < NH; ++n) {
      const int h = h0 + n;
      if (h < H) ysum[h] = __float2bfloat16(acc[n]);
    }
  }
}

// Legacy down (no LDS act) — A/B via VT_GEMMA4_DOWN_LDS=0.
__global__ void ExpertFp8DownTopKIndexedKernelHwNoLds(__hip_bfloat16* __restrict__ ysum,
                                                      const float* __restrict__ act,
                                                      const uint8_t* __restrict__ dn_base,
                                                      const __hip_bfloat16* __restrict__ sdn_base,
                                                      const int32_t* __restrict__ idx,
                                                      const float* __restrict__ wts, int G, int I,
                                                      int H) {
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); h < H; h += stride) {
    float acc = 0.f;
    for (int g = 0; g < G; ++g) {
      const int e = sidx[g];
      const uint8_t* drow = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
      const float* arow = act + static_cast<int64_t>(g) * I;
      float dot = 0.f;
      int i = 0;
      const int I4 = I & ~3;
      for (; i < I4; i += 4) {
        const float4 av = *reinterpret_cast<const float4*>(arow + i);
        const uint32_t wp = *reinterpret_cast<const uint32_t*>(drow + i);
        float w0, w1, w2, w3;
        Fp8E4M3HwFromU32(wp, w0, w1, w2, w3);
        dot += av.x * w0 + av.y * w1 + av.z * w2 + av.w * w3;
      }
      for (; i < I; ++i) dot += arow[i] * Fp8E4M3HwFromByte(drow[i]);
      const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
      acc += swts[g] * __bfloat162float(sd[h]) * dot;
    }
    ysum[h] = __float2bfloat16(acc);
  }
}

// --- FP8 WMMA ExpertGeGLU act (decode T=1, opt-in VT_GEMMA4_EXPERT_WMMA=1) -----
// Quantize x_f → E4M3 once, then 16×16×16 fp8×fp8 WMMA tiles for gate/up GEMVs.
// Down stays on ExpertFp8DownTopKIndexedKernelHw. Multi-wave mma forbidden (hang).

__device__ inline float AtomicMaxFloat(float* addr, float val) {
  int* iaddr = reinterpret_cast<int*>(addr);
  int old = *iaddr, assumed;
  do {
    assumed = old;
    const float old_f = __int_as_float(assumed);
    if (old_f >= val) break;
    old = atomicCAS(iaddr, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

// Pass 1: absmax(|x|) into *sx_scratch (caller zeros to 0).
__global__ void ExpertXAbsMaxKernel(float* __restrict__ sx_scratch, const float* __restrict__ x_f,
                                    int H) {
  __shared__ float sm[256];
  const int tid = static_cast<int>(threadIdx.x);
  float local = 0.f;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + tid); i < H;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    local = fmaxf(local, fabsf(x_f[i]));
  }
  sm[tid] = local;
  __syncthreads();
#pragma unroll
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
    __syncthreads();
  }
  if (tid == 0) AtomicMaxFloat(sx_scratch, sm[0]);
}

// Pass 2: sx = max(absmax/448, eps); x_f8 = E4M3(x/sx).
__global__ void ExpertXQuantFp8Kernel(uint8_t* __restrict__ x_f8, float* __restrict__ sx_io,
                                      const float* __restrict__ x_f, int H) {
  __shared__ float ssx;
  if (threadIdx.x == 0) {
    float amax = *sx_io;
    // OCP E4M3 finite max ≈ 448.
    float sx = amax > 0.f ? (amax / 448.f) : 1.f;
    if (sx < 1e-8f) sx = 1e-8f;
    *sx_io = sx;
    ssx = sx;
  }
  __syncthreads();
  const float inv = 1.f / ssx;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < H;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    const float v = x_f[i] * inv;
    x_f8[i] = static_cast<uint8_t>(
        __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3));
  }
}

// One warp / CTA: 16 intermediate rows × one expert. grid = (I/16)*G.
// C = A_broadcast(x_f8) @ W^T  via rocWMMA fp8×fp8; scale by sx * channel.
__global__ void ExpertFp8ActTopKIndexedKernelWmma(float* __restrict__ act,
                                                  const uint8_t* __restrict__ x_f8,
                                                  const float* __restrict__ sx_ptr,
                                                  const uint8_t* __restrict__ gu_base,
                                                  const __hip_bfloat16* __restrict__ sgu_base,
                                                  const int32_t* __restrict__ idx, int G, int I,
                                                  int H) {
#if !defined(VT_EXPERT_WMMA_OK)
  (void)act;
  (void)x_f8;
  (void)sx_ptr;
  (void)gu_base;
  (void)sgu_base;
  (void)idx;
  (void)G;
  (void)I;
  (void)H;
#else
  using namespace rocwmma;
  constexpr int WM = kWmma;
  const int n_it = I / WM;
  if (n_it <= 0 || G <= 0) return;
  const int bid = static_cast<int>(blockIdx.x);
  const int g = bid / n_it;
  const int t = bid - g * n_it;
  if (g >= G) return;
  const int i0 = t * WM;
  const int tid = static_cast<int>(threadIdx.x);
  // Single-wave CTA only.
  if (tid >= 32) return;

  __shared__ int se;
  __shared__ float ssx;
  __shared__ float s_c[WM * WM];
  __shared__ float8_t s_a[WM * WM];
  if (tid == 0) {
    se = idx[g];
    ssx = *sx_ptr;
  }
  __syncthreads();
  const int e = se;
  const float sx = ssx;
  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
  const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
  const float8_t* gu_f8 = reinterpret_cast<const float8_t*>(gu);
  const float8_t* x8 = reinterpret_cast<const float8_t*>(x_f8);

  auto run_half = [&](int row_base, float* out16) {
    fragment<matrix_a, WM, WM, WM, float8_t, row_major> a_frag;
    fragment<matrix_b, WM, WM, WM, float8_t, col_major> b_frag;
    fragment<accumulator, WM, WM, WM, float> acc;
    fill_fragment(acc, 0.f);
    const float8_t* Wrow0 = gu_f8 + static_cast<size_t>(row_base) * static_cast<size_t>(H);
    for (int k0 = 0; k0 < H; k0 += WM) {
      // Broadcast x_f8[k0:k0+16] into 16×16 A (all rows equal).
      for (int e2 = tid; e2 < WM * WM; e2 += 32) {
        const int kk = e2 & (WM - 1);
        s_a[e2] = x8[k0 + kk];
      }
      // matrix_b col_major ld=H: column n = weight row row_base+n, K along row.
      __syncthreads();
      load_matrix_sync(a_frag, s_a, WM);
      load_matrix_sync(b_frag, Wrow0 + k0, H);
      mma_sync(acc, a_frag, b_frag, acc);
      __syncthreads();
    }
    store_matrix_sync(s_c, acc, WM, mem_row_major);
    __syncthreads();
    // A broadcast ⇒ all rows of C equal; take row 0.
    if (tid < WM) out16[tid] = s_c[tid];
    __syncthreads();
  };

  __shared__ float gate16[WM];
  __shared__ float up16[WM];
  run_half(i0, gate16);          // gate rows [i0,i0+16)
  run_half(I + i0, up16);        // up rows

  if (tid < WM) {
    const int i = i0 + tid;
    const float sgate = __bfloat162float(sg[i]);
    const float sup = __bfloat162float(sg[I + i]);
    const float gate = sx * gate16[tid];
    const float up = sx * up16[tid];
    act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
  }
#endif  // VT_EXPERT_WMMA_OK
}

__global__ void ApplyExpertScaleRwKernel(float* __restrict__ rw, const int32_t* __restrict__ ri,
                                         const float* __restrict__ escale, int G, int E) {
  const int g = static_cast<int>(threadIdx.x);
  if (g >= G) return;
  const int e = ri[g];
  if (e >= 0 && e < E) rw[g] *= escale[e];
}

struct IdxTls {
  float* d_act = nullptr;
  float* d_xf = nullptr;
  uint8_t* d_xf8 = nullptr;
  float* d_sx = nullptr;
  int act_cap = 0;
  int xf_cap = 0;
  int xf8_cap = 0;
  int bound_dev = -1;
};
IdxTls& IdxTlsSlot(int dev) {
  static thread_local IdxTls slots[8];
  return slots[dev];
}

// Alloc-only (safe to call at Prepare). Graph-safe: no hipSetDevice/hipMalloc once warm.
bool EnsureIdxTls(int dev, int G, int I, int H) {
  if (dev < 0 || dev >= 8 || G <= 0 || G > kMaxTopK || I <= 0 || H <= 0) return false;
  auto& tls = IdxTlsSlot(dev);
  const int act_need = G * I;
  if (tls.bound_dev == dev && tls.d_act && tls.d_xf && tls.act_cap >= act_need &&
      tls.xf_cap >= H && tls.d_xf8 && tls.xf8_cap >= H && tls.d_sx)
    return true;  // hot path: no device API
  if (hipSetDevice(dev) != hipSuccess) return false;
  if (tls.bound_dev != dev) {
    tls = IdxTls{};
    tls.bound_dev = dev;
  }
  if (tls.act_cap < act_need || !tls.d_act) {
    if (tls.d_act) (void)hipFree(tls.d_act);
    tls.d_act = nullptr;
    tls.act_cap = 0;
    if (hipMalloc(&tls.d_act, static_cast<size_t>(act_need) * sizeof(float)) != hipSuccess)
      return false;
    tls.act_cap = act_need;
  }
  if (tls.xf_cap < H || !tls.d_xf) {
    if (tls.d_xf) (void)hipFree(tls.d_xf);
    tls.d_xf = nullptr;
    tls.xf_cap = 0;
    if (hipMalloc(&tls.d_xf, static_cast<size_t>(H) * sizeof(float)) != hipSuccess) return false;
    tls.xf_cap = H;
  }
  if (tls.xf8_cap < H || !tls.d_xf8) {
    if (tls.d_xf8) (void)hipFree(tls.d_xf8);
    tls.d_xf8 = nullptr;
    tls.xf8_cap = 0;
    if (hipMalloc(&tls.d_xf8, static_cast<size_t>(H) * sizeof(uint8_t)) != hipSuccess)
      return false;
    tls.xf8_cap = H;
  }
  if (!tls.d_sx) {
    if (hipMalloc(&tls.d_sx, sizeof(float)) != hipSuccess) return false;
  }
  return true;
}

bool ExpertGeGLUFp8TopKIndexedRocm(Queue& q, void* ysum, const void* x, const void* gu_base,
                                   const void* dn_base, const void* sgu_base, const void* sdn_base,
                                   const int32_t* idx_dev, const float* wts_dev, int G, int I,
                                   int H) {
  if (G <= 0 || G > kMaxTopK || !ysum || !x || !gu_base || !dn_base || !sgu_base || !sdn_base ||
      !idx_dev || !wts_dev)
    return false;
  if (I <= 0 || H <= 0) return false;
  const int dev = q.device.index;
  if (!EnsureIdxTls(dev, G, I, H)) return false;
  auto& tls = IdxTlsSlot(dev);

  // ROCm: <<<>>> launches use the *current* device; stream alone is not enough when
  // peer MoE runs ExpertGeGLU on expert_dev while sticky compute_dev is active.
  // (Lab 2026-08-10: peer path always failed hipGetLastError → slow host fallthrough.)
  int cur = -1;
  if (hipGetDevice(&cur) != hipSuccess || cur != dev) {
    if (hipSetDevice(dev) != hipSuccess) return false;
  }
  (void)hipGetLastError();  // drop sticky before launch check

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  // One-shot x expand → L2 (d_xf already in IdxTls; was allocated but unused).
  {
    const int xt = 256;
    const int xg = (H + xt - 1) / xt;
    Bf16ToF32VecKernel<<<xg, xt, 0, st>>>(tls.d_xf, static_cast<const __hip_bfloat16*>(x), H);
  }
  // VT_GEMMA4_ACT_I_PACK=2 → 2-I CTA pack; default 1 (legacy). Lab A/B flat/noisy.
  static const int act_pack = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_I_PACK");
    if (e && e[0] == '2') return 2;
    return 1;
  }();
  // VT_GEMMA4_FP8_HW_CVT: 1/default = gfx1201 OCP hw convert (no LDS LUT).
  // 0 = shared LUT fallback. Lab 2026-08-10 KEEP: ~+8% decode @9–30k vs LUT.
  static const int hw_cvt = [] {
    const char* e = std::getenv("VT_GEMMA4_FP8_HW_CVT");
    if (e && e[0] == '0') return 0;
    return 1;
  }();
  // VT_GEMMA4_EXPERT_WMMA=1 → FP8×FP8 WMMA act tiles (opt-in; quality+speed gated).
  static const int expert_wmma = [] {
    const char* e = std::getenv("VT_GEMMA4_EXPERT_WMMA");
    if (e && e[0] == '1') return 1;
    return 0;
  }();
  // VT_GEMMA4_ACT_WARPG=1 → one warp per top-k expert (parallel G, full lane fill).
  static const int act_warpg = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_WARPG");
    if (e && e[0] == '1') return 1;
    return 0;
  }();
  // VT_GEMMA4_ACT_TPB: threads/block for HW act (default 256). 128 fills H=2816 better
  // (256 leaves ~80 idle lanes: tid*16 >= 2816).
  static const int act_tpb = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_TPB");
    if (!e || !e[0]) return 256;
    const int v = std::atoi(e);
    if (v == 128 || v == 192 || v == 256) return v;
    return 256;
  }();
  // Dense float4 H-strip (all threads busy). Opt-in A/B.
  static const int act_dense = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_DENSE");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  // Parallel G via grid=I*G (with dense). Opt-in A/B.
  static const int act_par_g = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_PAR_G");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  // One warp per (i,g), grid=I*G, tpb=32. Opt-in A/B.
  static const int act_warp_i = [] {
    const char* e = std::getenv("VT_GEMMA4_ACT_WARP_I");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  // VT_GEMMA4_DOWN_LDS: 1/default = stage act[G,I] in LDS for down kernel; 0 = legacy global act.
  static const int down_lds = [] {
    const char* e = std::getenv("VT_GEMMA4_DOWN_LDS");
    if (e && e[0] == '0') return 0;
    return 1;
  }();
  // Down block size: 64/128/256. Smaller → more CTAs (H=2816 → 11@256, 22@128, 44@64).
  static const int down_tpb = [] {
    const char* e = std::getenv("VT_GEMMA4_DOWN_TPB");
    if (!e || !e[0]) return 256;
    const int v = std::atoi(e);
    if (v == 64 || v == 128 || v == 256) return v;
    return 256;
  }();
  // Multi-h per thread (LDS path only): 1=baseline, 2/4 = ExpertFp8DownTopKIndexedKernelHwNh.
  static const int down_nh = [] {
    const char* e = std::getenv("VT_GEMMA4_DOWN_NH");
    if (!e || !e[0]) return 1;
    const int v = std::atoi(e);
    if (v == 1 || v == 2 || v == 4) return v;
    return 1;
  }();
  // Warp-coop down (one warp per h). KEEP 2026-08-11: ~−28% down_us, ~+2.5% short gen.
  // =0 thread-per-h LDS baseline. Default ON.
  static const int down_warp = [] {
    const char* e = std::getenv("VT_GEMMA4_DOWN_WARP");
    if (e && e[0] == '0') return 0;
    return 1;
  }();
  // Must match UploadFp8NativeLayer pack (VT_GEMMA4_GU_INTERLEAVE).
  static const int gu_pair = [] {
    const char* e = std::getenv("VT_GEMMA4_GU_INTERLEAVE");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  // VT_GEMMA4_EXPERT_SPLIT=1 → hipEvent µs for act / down (KEEP path diagnostics).
  static const bool split_prof = [] {
    const char* e = std::getenv("VT_GEMMA4_EXPERT_SPLIT");
    return e && e[0] == '1';
  }();
  struct SplitProf {
    hipEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
    double us_act = 0, us_dn = 0;
    unsigned long long n = 0;
    bool ready = false;
    int bound_dev = -1;
  };
  static thread_local SplitProf sp_slot[8];
  SplitProf* sp = nullptr;
  if (split_prof && dev >= 0 && dev < 8) {
    sp = &sp_slot[dev];
    if (!sp->ready || sp->bound_dev != dev) {
      if (sp->e0) {
        (void)hipEventDestroy(sp->e0);
        (void)hipEventDestroy(sp->e1);
        (void)hipEventDestroy(sp->e2);
        (void)hipEventDestroy(sp->e3);
      }
      *sp = SplitProf{};
      sp->bound_dev = dev;
      auto mk = [](hipEvent_t* e) {
        return hipEventCreateWithFlags(e, hipEventDefault) == hipSuccess;
      };
      sp->ready = mk(&sp->e0) && mk(&sp->e1) && mk(&sp->e2) && mk(&sp->e3);
    }
  }
  static bool once_pack = false;
  if (!once_pack) {
    once_pack = true;
    std::fprintf(stderr,
                 "INFO ExpertGeGLU act_i_pack=%d fp8_hw_cvt=%d expert_wmma=%d act_warpg=%d "
                 "act_tpb=%d act_dense=%d act_par_g=%d act_warp_i=%d gu_pair=%d down_lds=%d "
                 "down_tpb=%d down_nh=%d down_warp=%d expert_split=%d\n",
                 act_pack, hw_cvt, expert_wmma, act_warpg, act_tpb, act_dense, act_par_g, act_warp_i,
                 gu_pair, down_lds, down_tpb, down_nh, down_warp, split_prof ? 1 : 0);
    std::fflush(stderr);
  }

#if defined(VT_EXPERT_WMMA_OK)
  // Device TU sees gfx12 — WMMA kernel is real. Host-only compiles still launch
  // when env is set (fat binary device side has the kernel).
#endif
  // Always allow launch path on this lab (HIP arches=gfx1201). Kernel is
  // empty-stub on non-gfx12 device objects.
  // WMMA tiles need 16 contiguous gate rows — incompatible with pair interleave.
  const bool wmma_ok = expert_wmma && hw_cvt && !gu_pair && (I % kWmma == 0) &&
                       (H % kWmma == 0) && (I / kWmma) > 0;
  // Warp-G needs G warps in one CTA (G≤8 → ≤256 threads) and HW cvt.
  // Pair layout needs GuGateUpRows in WarpG/I2 kernels — not wired; force off.
  const bool warpg_ok =
      act_warpg && hw_cvt && !gu_pair && G > 0 && G <= 8 && !wmma_ok;

  if (split_prof && sp && sp->ready) (void)hipEventRecord(sp->e0, st);

  // --- act launch ---
  if (wmma_ok) {
    (void)hipMemsetAsync(tls.d_sx, 0, sizeof(float), st);
    {
      const int xt = 256;
      const int xg = (H + xt - 1) / xt;
      ExpertXAbsMaxKernel<<<xg, xt, 0, st>>>(tls.d_sx, tls.d_xf, H);
      ExpertXQuantFp8Kernel<<<xg, xt, 0, st>>>(tls.d_xf8, tls.d_sx, tls.d_xf, H);
    }
    const unsigned n_it = static_cast<unsigned>(I / kWmma);
    const unsigned grid_act = n_it * static_cast<unsigned>(G);
    ExpertFp8ActTopKIndexedKernelWmma<<<grid_act, 32, 0, st>>>(
        tls.d_act, tls.d_xf8, tls.d_sx, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H);
  } else if (warpg_ok) {
    dim3 grid_act(static_cast<unsigned>(I));
    const int tpb = G * 32;
    ExpertFp8ActTopKIndexedKernelHwWarpG<<<grid_act, tpb, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H);
  } else if (hw_cvt && act_warp_i && !gu_pair) {
    const unsigned ng = static_cast<unsigned>(I) * static_cast<unsigned>(G);
    ExpertFp8ActTopKIndexedKernelHwWarpI<<<ng, 32, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H, gu_pair);
  } else if (hw_cvt && act_dense) {
    const unsigned ng =
        act_par_g ? static_cast<unsigned>(I) * static_cast<unsigned>(G)
                  : static_cast<unsigned>(I);
    ExpertFp8ActTopKIndexedKernelHwDense<<<ng, act_tpb, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H, gu_pair, act_par_g);
  } else if (hw_cvt && act_pack == 2 && !gu_pair) {
    const unsigned ng = static_cast<unsigned>((I + 1) / 2);
    ExpertFp8ActTopKIndexedKernelHwI2<<<ng, act_tpb, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H);
  } else if (hw_cvt) {
    dim3 grid_act(static_cast<unsigned>(I));
    ExpertFp8ActTopKIndexedKernelHw<<<grid_act, act_tpb, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H, gu_pair);
  } else if (act_pack == 2) {
    const unsigned ng = static_cast<unsigned>((I + 1) / 2);
    ExpertFp8ActTopKIndexedKernelI2<<<ng, kBlock, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H);
  } else {
    dim3 grid_act(static_cast<unsigned>(I));
    ExpertFp8ActTopKIndexedKernel<<<grid_act, kBlock, 0, st>>>(
        tls.d_act, tls.d_xf, static_cast<const uint8_t*>(gu_base),
        static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H, gu_pair);
  }

  if (split_prof && sp && sp->ready) (void)hipEventRecord(sp->e1, st);

  // --- down launch (same for all act paths) ---
  {
    const int tpb_dn = down_tpb;
    const size_t smem =
        down_lds ? static_cast<size_t>(G) * static_cast<size_t>(I) * sizeof(float) : 0;
    if (hw_cvt || wmma_ok || warpg_ok) {
      if (down_lds && down_warp) {
        const int nwarps = tpb_dn / 32;
        const int grid_dn = (H + nwarps - 1) / nwarps;
        ExpertFp8DownTopKIndexedKernelHwWarp<<<grid_dn, tpb_dn, smem, st>>>(
            static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
            static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
      } else if (down_lds && down_nh == 2) {
        const int nslot = (H + 1) / 2;
        const int grid_dn = (nslot + tpb_dn - 1) / tpb_dn;
        ExpertFp8DownTopKIndexedKernelHwNh<2><<<grid_dn, tpb_dn, smem, st>>>(
            static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
            static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
      } else if (down_lds && down_nh == 4) {
        const int nslot = (H + 3) / 4;
        const int grid_dn = (nslot + tpb_dn - 1) / tpb_dn;
        ExpertFp8DownTopKIndexedKernelHwNh<4><<<grid_dn, tpb_dn, smem, st>>>(
            static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
            static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
      } else if (down_lds) {
        const int grid_dn = (H + tpb_dn - 1) / tpb_dn;
        ExpertFp8DownTopKIndexedKernelHw<<<grid_dn, tpb_dn, smem, st>>>(
            static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
            static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
      } else {
        const int grid_dn = (H + tpb_dn - 1) / tpb_dn;
        ExpertFp8DownTopKIndexedKernelHwNoLds<<<grid_dn, tpb_dn, 0, st>>>(
            static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
            static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
      }
    } else {
      const int grid_dn = (H + kBlock - 1) / kBlock;
      ExpertFp8DownTopKIndexedKernel<<<grid_dn, kBlock, 0, st>>>(
          static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
          static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
    }
  }

  if (split_prof && sp && sp->ready) {
    (void)hipEventRecord(sp->e2, st);
    // e0→e1 = act (+ optional wmma quant); e1→e2 = down. x_expand is outside this window.
    if (hipEventSynchronize(sp->e2) == hipSuccess) {
      float ms_act = 0.f, ms_dn = 0.f;
      (void)hipEventElapsedTime(&ms_act, sp->e0, sp->e1);
      (void)hipEventElapsedTime(&ms_dn, sp->e1, sp->e2);
      sp->us_act += ms_act * 1000.0;
      sp->us_dn += ms_dn * 1000.0;
      ++sp->n;
      if (sp->n == 64 || (sp->n > 0 && (sp->n % 512) == 0)) {
        const double inv = 1.0 / static_cast<double>(sp->n);
        std::fprintf(stderr,
                     "INFO ExpertGeGLU split dev=%d n=%llu act_us=%.1f down_us=%.1f act%%=%.0f "
                     "down%%=%.0f (per call avg)\n",
                     dev, sp->n, sp->us_act * inv, sp->us_dn * inv,
                     100.0 * sp->us_act / (sp->us_act + sp->us_dn + 1e-9),
                     100.0 * sp->us_dn / (sp->us_act + sp->us_dn + 1e-9));
        std::fflush(stderr);
      }
    }
  }
  return hipGetLastError() == hipSuccess;
}

void ApplyExpertScaleRwRocm(Queue& q, float* rw_dev, const int32_t* ri_dev, const float* escale_dev,
                            int G, int E) {
  if (!rw_dev || !ri_dev || !escale_dev || G <= 0) return;
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  ApplyExpertScaleRwKernel<<<1, G, 0, st>>>(rw_dev, ri_dev, escale_dev, G, E);
}

// Pre-allocate per-device ExpertGeGLU scratch (act + x_f) so decode never hipMalloc mid-step.
bool PrewarmExpertGeGLUFp8TopKIndexedRocm(int dev, int G, int I, int H) {
  return EnsureIdxTls(dev, G, I, H);
}

// --- Prefill MoE gather / weighted scatter (device-only) ---

__global__ void MoeGatherRowsKernel(__hip_bfloat16* __restrict__ out,
                                    const __hip_bfloat16* __restrict__ in,
                                    const int32_t* __restrict__ tids, int n, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= n) return;
  const int t = tids[i];
  const __hip_bfloat16* src = in + static_cast<int64_t>(t) * H;
  __hip_bfloat16* dst = out + static_cast<int64_t>(i) * H;
  for (int h = static_cast<int>(threadIdx.x); h < H; h += static_cast<int>(blockDim.x)) {
    dst[h] = src[h];
  }
}

// One block per (row i); threads walk H. Atomic not needed if tokens unique per expert chunk
// (same token can appear once per expert with top-k — different experts write different
// increments to same acc row → need atomics on bf16 via float CAS or float acc).
// Use float staging: each thread does atomicAdd on float buffer — too heavy.
// Simpler: serialize rows with one warp per row doing non-atomic read-modify-write
// when we guarantee single-threaded updates per (token,h) within a launch.
// For top-k, same token appears in multiple experts → concurrent expert launches are
// sequential in host loop, so within one scatter launch each token appears once.
// Safe non-atomic RMW per (i) row.
__global__ void MoeWeightedScatterAddKernel(__hip_bfloat16* __restrict__ acc,
                                            const __hip_bfloat16* __restrict__ y,
                                            const int32_t* __restrict__ tids,
                                            const float* __restrict__ wts, int n, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= n) return;
  const int t = tids[i];
  const float w = wts[i];
  const __hip_bfloat16* src = y + static_cast<int64_t>(i) * H;
  __hip_bfloat16* dst = acc + static_cast<int64_t>(t) * H;
  for (int h = static_cast<int>(threadIdx.x); h < H; h += static_cast<int>(blockDim.x)) {
    const float v = w * __bfloat162float(src[h]) + __bfloat162float(dst[h]);
    dst[h] = __float2bfloat16(v);
  }
}

__global__ void MoeZeroBf16Kernel(__hip_bfloat16* __restrict__ buf, int64_t nelem) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
  if (i < nelem) buf[i] = __float2bfloat16(0.f);
}

void MoeGatherRowsRocm(Queue& q, void* out_bf16, const void* in_bf16, const int32_t* token_ids,
                       int n, int H) {
  if (n <= 0 || H <= 0 || !out_bf16 || !in_bf16 || !token_ids) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  MoeGatherRowsKernel<<<n, kBlock, 0, s>>>(
      static_cast<__hip_bfloat16*>(out_bf16), static_cast<const __hip_bfloat16*>(in_bf16),
      token_ids, n, H);
}

void MoeWeightedScatterAddRocm(Queue& q, void* acc_bf16, const void* y_bf16,
                               const int32_t* token_ids, const float* weights, int n, int H) {
  if (n <= 0 || H <= 0 || !acc_bf16 || !y_bf16 || !token_ids || !weights) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  // Threads in a block share a row — need sync between h-loop iterations? No, each h
  // unique per thread. Multiple threads write different h of same row — OK.
  MoeWeightedScatterAddKernel<<<n, kBlock, 0, s>>>(
      static_cast<__hip_bfloat16*>(acc_bf16), static_cast<const __hip_bfloat16*>(y_bf16),
      token_ids, weights, n, H);
}

void MoeZeroBf16Rocm(Queue& q, void* buf_bf16, int64_t nelem) {
  if (!buf_bf16 || nelem <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int block = 256;
  const int grid = static_cast<int>((nelem + block - 1) / block);
  MoeZeroBf16Kernel<<<grid, block, 0, s>>>(static_cast<__hip_bfloat16*>(buf_bf16), nelem);
}

}  // namespace vt::rocm

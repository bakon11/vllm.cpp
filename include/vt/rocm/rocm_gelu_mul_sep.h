#pragma once
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vt::rocm {
// out[i] = gelu_tanh(gate[i]) * up[i]  for i in [0,n)
void GeluMulSeparateRocm(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                         DType dtype);
// Pair-interleaved GeGLU on x[M,2I] columns (g0,u0,g1,u1,…).
void GeluAndMulPairRocm(Queue& q, Tensor& out, const Tensor& x);
}  // namespace vt::rocm

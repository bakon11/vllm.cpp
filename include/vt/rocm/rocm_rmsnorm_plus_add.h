#pragma once
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vt::rocm {
// out = rmsnorm(x, w) + addend  (Gemma-4 residual join)
void RmsNormPlusAddRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor& addend, const RmsNormArgs& args);
}  // namespace vt::rocm

# ROCm: launch SharedK WMMA from the host (#785)

Row: `BACKEND-ROCM`. Issue:
[#785](https://github.com/mudler/vllm.cpp/issues/785).

## Defect

`PagedAttnPrefillSharedKWmma` launches were behind
`#if defined(VT_ROCWMMA_OK)`. That macro is defined only on HIP's
**device** pass (`__gfx1200__` / `__gfx1201__`). The host pass never
defines it, so both launch sites were deleted. Every d=256/d=512
prefill silently ran scalar `PagedAttnPrefillSharedK`.

## P0 (this head)

Repairs **d=256 production dispatch only**. d=512 remains scalar.

- Host launch uses `hipDeviceProp_t.gcnArchName` prefix-match
  `gfx1200` / `gfx1201` (not substring). `gfx1201:xnack-` matches;
  `foogfx1201` and `gfx12010` do not.
- Per-device decision cached once (`std::call_once`). Not per-build.
  Not getenv.
- Device kernel-body `#if !defined(VT_ROCWMMA_OK)` stub is unchanged.
- `VT_ATTN_PREFILL_SHAREDK_WMMA` still forces scalar when `=0`.
- Host path launches only `PagedAttnPrefillSharedKWmma<2,8,16,32,false>`
  (d=256). The d=512 WMMA launch/stub is removed, not hidden.
- d=512 keeps the existing scalar `PagedAttnPrefillSharedK` fallthrough.
  Shipping f58b d=512 WMMA is VGPR 192 / spill 52 / private 212 — that
  violates the 0/0 compile gate and is owed as a separate kernel repair.
- Shipping f58b HIP compile of `rocm_paged_attn.hip` (`clang++` roc-7.2.4
  `f58b06dce1f9`, `--offload-arch=gfx1201`). Device body unchanged.

Measured KD on that object (fields after `.name`):

| Instantiation | path | vgpr | spill | private | LDS |
|---|---|---|---|---|---|
| `<2,8,16,32,false>` | d=256 | 151 | **0** | **0** | 4880 |

P1 (not this head): gfx1201 witness that the d=256 SharedKWmma
instantiation actually dispatches, plus same-build forced-scalar vs
oracle. No d=512 GPU work in this lane. GPU HOLD until Researcher
greens this repaired P0.

## Owed

- P1 d=256 dispatch witness + scalar-vs-oracle
- Separate kernel repair before any d=512 WMMA launch
- PR body (never-ran consequence + expected d=256 uplift)

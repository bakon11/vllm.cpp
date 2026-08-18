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

- Host launch uses `hipDeviceProp_t.gcnArchName` prefix-match
  `gfx1200` / `gfx1201` (not substring). `gfx1201:xnack-` matches;
  `foogfx1201` and `gfx12010` do not.
- Per-device decision cached once (`std::call_once`). Not per-build.
  Not getenv.
- Device kernel-body `#if !defined(VT_ROCWMMA_OK)` stub is unchanged.
- `VT_ATTN_PREFILL_SHAREDK_WMMA` still forces scalar when `=0`.
- Shipping f58b HIP compile of `rocm_paged_attn.hip` (`clang++` roc-7.2.4
  `f58b06dce1f9`, `--offload-arch=gfx1201`). Device body unchanged.
- Measured KD on that object (fields after `.name`):

| Instantiation | path | vgpr | spill | private | LDS |
|---|---|---|---|---|---|
| `<2,8,16,32,false>` | d=256 | 151 | **0** | **0** | 4880 |
| `<2,16,16,16,true>` | d=512 | 192 | **52** | **212** | 2576 |

d=256 meets the 0/0 compile gate. d=512 does not (pre-existing
global-Q body). P0 does not rewrite that kernel.

P1 (not this head): gfx1201 witness that SharedKWmma actually
dispatches, plus scalar fallback vs oracle. GPU HOLD until Researcher
greens P0. d=512 launch vs scalar-only is a Researcher call.

## Owed

P1 witness + PR body (never-ran consequence + expected uplift).

## Summary

Follow-up to merged #140 (ROCm gfx1201 + Gemma-4-26B MoE BF16/FP8).

Hardens dual-GPU FP8 expert residency for ~30 GB hosts, plus a full decode speed stack for **stream FP8**.

### Residency
- Pack GPU0 first, spill GPU1; stream FP8→BF16 H2D (no permanent host BF16 during bulk upload)
- Peer-copy off-device slices; optional `VT_GEMMA4_RESIDENT_*`

### Decode speed (branch tip)
- Fused expert gate+up MatmulBT (3→2 GEMMs/expert)
- Async expert H2D + device expert cache (+ optional VRAM LRU)
- DualRmsNormPlusRes / RmsNormPlusAdd residual fuses
- TLS MoE/MLP/Attn scratch; TLS hipBLAS handle; TLS router ones-weight
- Device MoE router top-k
- CLI `--repeat N` multi-run benches
- `VT_GEMMA4_PROFILE=1` MoE router vs expert timing
- Opt-in `VT_ROCM_GEMV=1` LDS-cached M=1 BF16 GEMV (**slower than hipblas on gfx1201**)

## Lab (2× R9700, ROCm 7.2.x, kernel **7.0.0-29**)

| Metric | Value |
|--------|--------|
| Correctness | Paris/Hello/arith **3/3** stream FP8 |
| Warm Paris (vllm.cpp ROCm FP8 serial) | **~33–34 tok/s** |
| **Target bar: llama.cpp Vulkan Q8** (llama-bench tg128, 1× R9700) | **~98 tok/s** |
| Gap | **~3×** slower than Q8 Vulkan |
| MoE profile (warm) | router **~2%**, experts **~98%** |

### Fused ExpertGeGLU spike (`VT_GEMMA4_FUSED_EXPERTS=1`)
- Multi-block act + down-mix HIP path
- **Correct** (Paris) but **~3.3 tok/s** (no MFMA; scalar dots) — default **off**

### Layer profile (`VT_GEMMA4_PROFILE=1`, warm decode)
Per layer-call (avg after warmup): **attn ~4% · dense MLP ~1% · MoE ~95%**

### Expert GEMM microbench (gfx1201, M=1)
| Shape | GemmEx BF16 | hipBLASLt FP8×BF16 (W8A16) | hipBLASLt FP8×FP8 (W8A8)→BF16 C |
|-------|-------------|----------------------------|----------------------------------|
| gate_up N=1408 K=2816 | **~30 µs** | NOT_SUPPORTED | **~54 µs (~1.8× slower)** |
| down N=2816 K=704 | **~17 µs** | NOT_SUPPORTED | **~18 µs (~1.2× slower)** |

⇒ On RDNA4 decode shapes, **library FP8 is available only as W8A8**, and it **does not beat BF16 GemmEx**.  
Skip local re-quant until more host RAM; Firworks FP8_DYNAMIC file is fine. Speed needs better kernels or different approach than stock hipBLASLt W8A8 @ M=1.

Logs: `~/llms/logs/vllm-cpp-fp8-lt-matrix.log`, `vllm-cpp-fp8-vs-bf16-gemm.log`

### Upstream (mudler/vllm.cpp, checked 2026-08-08)
- **#154** (ours) OPEN, mergeable
- **#140** merged (our base ROCm/Gemma4)
- Decode-graph / async (#36/#39) is **CUDA/Qwen** — ROCm `SupportsGraphCapture=false` (hipGraph not wired)
- No other open PRs targeting Gemma4 MoE ROCm speed

```bash
HIP_VISIBLE_DEVICES=0 ./build-hip/examples/vllm-cli \
  --model .../gemma-4-26B-A4B-it-fp8 \
  --prompt '<bos>The capital of France is' --max-tokens 12 --temperature 0 --repeat 4
```

Harness: `FARM_RESTORE=0 LAB_REPEATS=4 python3 ~/llms/scripts/vllm/run_fp8_lab_bench.py`

## Dead ends (do not re-open without new evidence)
| Attempt | Result |
|---------|--------|
| `VT_GEMMA4_BATCH_EXPERTS=1` pointer-batch | ~0.8 tok/s |
| `HIPBLAS_COMPUTE_32F_FAST_16BF` / `16F` | NOT_SUPPORTED on BT shapes |
| Multi-stream ExpertGeGLU (even per-stream handles) | **wrong tokens** + slower |
| Naive / LDS M=1 GEMV | correct but **~27–30** vs hipblas **~34** |

## Next real lever
**Fused/grouped MoE expert kernel** that wins on RDNA4 MFMA (not host batch glue).  
Experts are the wall; host hygiene is exhausted.

## Notes
- Daily farm: dual-Vulkan Q6 P=3 `:8010` — exclusive HIP needs farm down
- Avoid Ubuntu **7.0.0-28** (AMDGPU ROCm regression); use **29+**

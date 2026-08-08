## Summary

Follow-up to merged #140 (ROCm gfx1201 + Gemma-4-26B MoE BF16/FP8).

This PR hardens **dual-GPU FP8 expert residency** and decode mix so the path fits **~30 GB host RAM** and keeps weights on GPU — plus a stack of decode speed work for stream FP8.

### Residency (#140 follow-up)
1. **Pack GPU0 first**, spill to GPU1 (same-device fast path maximized)
2. **Stream per-expert FP8→BF16→H2D** — no permanent host BF16 cache during bulk upload
3. **10 GiB compute headroom** on GPU0 so dense/KV still allocate
4. **Peer-copy** off-device resident expert slices into compute scratch
5. Optional full resident via `VT_GEMMA4_RESIDENT_*` (avoid on 30 G host for routine benches)

### Decode speed (this branch tip)
- ExpertGeGLU: **GeluMulSeparate** (no gate|up pack), fused **alpha/beta** mix into down GEMM
- **Async expert H2D** (drop per-expert Synchronize); optional `VT_GEMMA4_EXPERT_VRAM_MB` LRU (default unlimited)
- TLS scratch: MoE layer temps, dense GeGLU gate_up, ForwardBody buffers
- **RmsNormPlusAdd** residual joins (post-attn / post-ff)
- PLE contiguous `[L,T,ple]` + GeluMulSeparate
- CLI `--repeat N` + stderr fflush for multi-run benches
- Device MoE router top-k (D2H only `[T,K]`)

## Lab evidence (2× R9700 gfx1201, ROCm 7.2.x)

### Kernel
Lab host should run **Ubuntu 7.0.0-29+** (not **7.0.0-28** — known AMDGPU/ROCm compute regression up to ~42×).

### Stream FP8, exclusive GPU0, `--repeat` (Firworks gemma-4-26B-A4B-it-fp8)

| Kernel | Run1 cold | Run2+ warm | Gate |
|--------|-----------|------------|------|
| 7.0.0-29 | ~0.46 tok/s | **~28 tok/s** stable | Paris OK |

Multi-prompt matrix historically 3/3 after device router.

| Env | Notes |
|-----|--------|
| Default | serial ExpertGeGLU, device expert cache after first use |
| `VT_GEMMA4_BATCH_EXPERTS=1` | opt-in; lab found slower than serial |
| `VT_GEMMA4_EXPERT_VRAM_MB` | 0/unset = unlimited device expert cache; set to cap LRU |

### Full resident (optional stress)
| Check | Result |
|-------|--------|
| Full resident upload | 30/30 layers, **~42.5 GiB** experts |
| Peak VRAM | ~**28.6 GB** historically |
| Avoid | routine full dual-resident on 30 G host |

## Usage

```bash
# Warm decode benchmark (recommended)
HIP_VISIBLE_DEVICES=0 \
./build-hip/examples/vllm-cli \
  --model /path/to/Firworks/gemma-4-26B-A4B-it-fp8 \
  --prompt '<bos>The capital of France is' --max-tokens 12 --temperature 0 \
  --repeat 4

# Optional full resident (heavy)
HIP_VISIBLE_DEVICES=0,1 \
VT_GEMMA4_RESIDENT_EXPERTS=1 VT_GEMMA4_RESIDENT_GPUS=2 \
./build-hip/examples/vllm-cli ...
```

Canonical lab script: `~/llms/scripts/vllm/run_fp8_lab_bench.py`

## Test plan

- [x] HIP build gfx1201
- [x] Stream FP8 Paris multi-run warm ~28 tok/s (k29)
- [x] RmsNormPlusAdd path Paris OK
- [ ] CI (no AMD GPU expected)
- [ ] Load-once server multi-req tok/s (follow-up)

## Notes

- Does **not** claim pip-vLLM oracle token-exact or Vulkan-Q6 farm parity yet.
- Warm decode still GEMM-bound (~720 expert GEMMs/token naive); next real leap is fused/grouped expert kernel that beats serial hipBLAS.
- Daily farm remains dual-Vulkan Q6 P=3 `:8010` — exclusive HIP smokes must stop farm first.

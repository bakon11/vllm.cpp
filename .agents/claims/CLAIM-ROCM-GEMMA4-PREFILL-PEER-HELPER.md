# CLAIM-ROCM-GEMMA4-PREFILL-PEER-HELPER

| Claim | Row IDs | Agent | Worktree | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-ROCM-GEMMA4-PREFILL-PEER-HELPER` | `BACKEND-ROCM` (slug `ROCM-GEMMA4-PREFILL-PEER-HELPER`, issue #839) | hermes-vllm (lab), helper | `/home/don/llms/vllm.cpp-prefill-peer` | `row/ROCM-GEMMA4-PREFILL-PEER-HELPER` | Owns ONLY: `RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice` → Launch/Finish + PeerPipeTls + DequantCache pin that persists in PeerSlot until host-observed ev_e retirement, in `src/vt/rocm/rocm_gemma4_experts.hip`. **EXCLUDED:** kPeerPipe default ON, FP8 Lt, GU_INTERLEAVE, donor unpin-before-ev_e, #838, #697. Requires #837 before or with impl as a separate head. Independent history from abandoned `row/ROCM-GEMMA4-XDEV-MOE`. | `SPIKE` spec-only | 2026-08-14 — `64cb` split repair; donor hashed; retirement-safe pin |

// Gemma-4 text backbone forward (`Gemma4ForConditionalGeneration` language_model
// stack, unsloth/gemma-4-E4B-it) — MODEL-GEMMA4 G1. Composed from the public vt::
// ops + the shared dense-attention device glue (dense_attn_block.h). Grounding:
// vllm/model_executor/models/gemma4.py (Gemma4MLP :224-254, Gemma4Attention
// :374-552, Gemma4DecoderLayer :555-767, Gemma4Model embed/PLE :845-928 + forward
// :1289-1370), gemma4_rope.py (proportional RoPE), layernorm.py::RMSNorm (PLAIN,
// x*w — NOT the Gemma (1+w) offset). See .agents/specs/gemma4-multimodal.md.
//
// Primitive-by-primitive vs the Gemma-2/3 path (gemma3.cpp):
//   - PLAIN RMSNorm everywhere (vt::RmsNormArgs{eps,false}); V-norm is
//     weight-less (a ones[Dh] weight = identity scale).
//   - standalone-norm + explicit-add residual (NOT the gemma2/3 fused sandwich).
//   - Per-Layer Embeddings: a second embed table + projection combined once
//     (:845-898), then a per-layer gate/proj/norm fused into the residual
//     after the MLP (:753-761). The gate `gelu(gate_lin(h)) * ple_l` reuses
//     vt::GeluAndMul on [gate_lin | ple_l] (no elementwise-Mul op needed).
//   - heterogeneous per-layer head_dim (256 sliding / 512 full), scaling=1.0.
//   - YOCO KV-sharing: shared layers read the target layer's K/V cache and
//     compute NO K/V of their own.
//   - dual RoPE: full layers use the "proportional" cos/sin cache (head_dim
//     denominator + zero-padded non-rotary pairs); sliding layers standard neox.
//   - GeGLU MLP, sqrt(hidden) embed normalizer, per-layer scalar, final logit
//     soft-cap 30.0, tied lm_head.
//
// G1b STATUS: the runner now allocates a PER-LAYER KV head_dim
// (KVCacheConfig::per_layer_attn_specs, consumed in runner.cpp initialize_kv_cache),
// so the per-layer VT_CHECK(kv.head_size == Dh) below is satisfied on both the
// 256 (sliding) and 512 (global) layers and the strict e2e gate RUNS. YOCO
// KV-sharing is handled in the forward (kv_idx = target for shared layers); the
// shared layers' own — unused — buffers are still allocated (memory-only G1c
// residual, not a correctness gap).
#include "vllm/model_executor/models/gemma4.h"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpGeluMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/fused_ops.h"
#if defined(VLLM_CPP_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
#endif

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev/DBuf/ResidentWeight/KvSlice/StepInputs/Reshape

double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

// The engine keeps HfConfig::raw as the FULL config.json (hf_config.cpp:414),
// while the mm-wrapper nests the language-model scalars under `text_config`.
// Gemma-4's per-arch fields (global_head_dim, hidden_size_per_layer_input,
// layer_types, rope_parameters, sliding_window, *_softcapping) therefore live in
// raw["text_config"], NOT at the top level — read them through this view.
const nlohmann::json& TextCfg(const nlohmann::json& raw) {
  const auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  return raw;
}

// Read a per-layer-type rope field (rope_parameters[type][key]) with a fallback.
double RopeField(const nlohmann::json& doc, const char* type, const char* key,
                 double fallback) {
  const auto rp = doc.find("rope_parameters");
  if (rp == doc.end() || !rp->is_object()) return fallback;
  const auto t = rp->find(type);
  if (t == rp->end() || !t->is_object()) return fallback;
  const auto k = t->find(key);
  if (k == t->end() || k->is_null() || !k->is_number()) return fallback;
  return k->get<double>();
}

// Read a bf16 scalar [1] weight to host float (layer_scalar).
float ReadBf16Scalar(const OwnedTensor& w) {
  VT_CHECK(w.HasHostBytes(), "gemma4: layer_scalar host bytes required");
  const auto* p = static_cast<const uint16_t*>(w.View().data);
  return vt::BF16ToF32(*p);
}

// Config-derived layout.
struct Gemma4Layout {
  int64_t hidden = 0;
  int64_t num_layers = 0;
  int64_t num_q_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim_sliding = 0;   // 256
  int64_t head_dim_full = 0;      // 512
  int64_t ple_dim = 0;            // hidden_size_per_layer_input (256)
  int64_t sliding_window = 0;
  double rope_theta_full = 1000000.0;
  double rope_partial_full = 0.25;  // partial_rotary_factor for full layers
  double rope_theta_sliding = 10000.0;
  float final_logit_softcap = 0.0f;
  float attn_logit_softcap = 0.0f;
  float embed_scale_ple = 0.0f;     // sqrt(ple_dim)
  double proj_scale = 0.0;          // hidden^-0.5
  double input_scale = 0.0;         // rsqrt(2)
};

Gemma4Layout MakeLayout(const HfConfig& cfg) {
  Gemma4Layout g;
  const nlohmann::json& raw = TextCfg(cfg.raw);
  g.hidden = cfg.hidden_size;
  g.num_layers = cfg.num_hidden_layers;
  g.num_q_heads = cfg.num_attention_heads;
  g.num_kv_heads = cfg.num_key_value_heads;
  g.head_dim_sliding = cfg.head_dim;
  g.head_dim_full = RawInt(raw, "global_head_dim", cfg.head_dim);
  g.ple_dim = RawInt(raw, "hidden_size_per_layer_input", 0);
  g.sliding_window = cfg.sliding_window.value_or(RawInt(raw, "sliding_window", 0));
  g.rope_theta_full = RopeField(raw, "full_attention", "rope_theta", 1000000.0);
  g.rope_partial_full =
      RopeField(raw, "full_attention", "partial_rotary_factor", 1.0);
  g.rope_theta_sliding =
      RopeField(raw, "sliding_attention", "rope_theta", 10000.0);
  g.final_logit_softcap =
      static_cast<float>(RawDouble(raw, "final_logit_softcapping", 0.0));
  g.attn_logit_softcap =
      static_cast<float>(RawDouble(raw, "attn_logit_softcapping", 0.0));
  g.embed_scale_ple = static_cast<float>(std::sqrt(static_cast<double>(g.ple_dim)));
  g.proj_scale = std::pow(static_cast<double>(g.hidden), -0.5);
  g.input_scale = 1.0 / std::sqrt(2.0);
  return g;
}

// Build the Gemma-4 "proportional" cos|sin cache for full-attention layers on
// host (gemma4_rope.py::Gemma4RotaryEmbedding). head_dim = Dh (512), rope_angles
// = rotary_dim/2 = int(Dh*partial)/2 non-zero pairs, the rest zero-padded to
// identity; angle denominator is Dh (not rotary_dim). Layout matches
// RopeCosSinCache: cos_sin[t, i]=cos, cos_sin[t, Dh/2 + i]=sin, over Dh pairs...
// stored as [P, Dh] with first Dh/2 = cos, second Dh/2 = sin. Double angle -> f32.
//
// CRITICAL (decode long-ctx): the old eager path rebuilt 0..max_pos on the host
// and H2D'd it EVERY token. At seq=23k / Dh=512 that is ~23MB of trig+upload per
// tok and dominated step-profile fwd (~50ms). Grow-only cache: rebuild only when
// max_pos exceeds capacity (prefill jump) with headroom so decode is a no-op.
void EnsureProportionalRopeCache(Dev d, const Gemma4Layout& g, int64_t Dh,
                                 int64_t max_pos, std::optional<DBuf>& slot,
                                 int64_t& cap_pos, int64_t& geom_Dh,
                                 double& geom_base, double& geom_partial) {
  const int64_t pairs = Dh / 2;
  const int64_t rope_angles =
      static_cast<int64_t>(g.rope_partial_full * static_cast<double>(Dh)) / 2;
  const double base = g.rope_theta_full;
  const double partial = g.rope_partial_full;

  const bool same_geom = slot.has_value() && geom_Dh == Dh && geom_base == base &&
                         geom_partial == partial && cap_pos >= 0;
  if (same_geom && max_pos <= cap_pos) return;

  int64_t build_to = max_pos;
  if (same_geom && max_pos > cap_pos) {
    const int64_t grow = std::max<int64_t>(2048, cap_pos / 4);
    build_to = max_pos + grow;
  } else {
    build_to = max_pos + (max_pos >= 4096 ? 2048 : 512);
  }

  std::vector<double> inv_freq(static_cast<size_t>(pairs), 0.0);
  for (int64_t j = 0; j < rope_angles && j < pairs; ++j) {
    const double exponent = static_cast<double>(2 * j) / static_cast<double>(Dh);
    inv_freq[static_cast<size_t>(j)] = 1.0 / std::pow(base, exponent);
  }

  const int64_t P = build_to + 1;
  std::vector<uint16_t> host(static_cast<size_t>(P) * static_cast<size_t>(Dh));
  // NOTE: cache dtype bf16 to match the bf16 q/k it rotates (RopeFromCache wants
  // q/k/cache same dtype). vLLM keeps f32 cos/sin — a named bf16-rounding nuance.
  for (int64_t t = 0; t < P; ++t) {
    for (int64_t i = 0; i < pairs; ++i) {
      const double angle = static_cast<double>(t) * inv_freq[static_cast<size_t>(i)];
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      host[static_cast<size_t>(t) * static_cast<size_t>(Dh) + static_cast<size_t>(i)] =
          vt::F32ToBF16(c);
      host[static_cast<size_t>(t) * static_cast<size_t>(Dh) +
           static_cast<size_t>(pairs + i)] = vt::F32ToBF16(s);
    }
  }
  slot.emplace(d, DType::kBF16, std::vector<int64_t>{P, Dh}, host.data());
  cap_pos = build_to;
  geom_Dh = Dh;
  geom_base = base;
  geom_partial = partial;
}

// (by-value BuildProportionalRopeCache removed — all call sites use Ensure*)

// Copy a strided source-row block into a contiguous destination sub-block. Used
// to assemble the [T, 2*ple] GeluAndMul input from the fresh gate GEMM (dense)
// and the per-layer slice of the [T, L, ple] PLE-input tensor (strided).
void CopyRow(Dev d, void* dst, const void* src, size_t bytes) {
  d.b.Copy(d.q, dst, src, bytes);
}

// One Gemma-4 self-attention block. `Dh` is this layer's head_dim (256/512).
// `kv` is the cache to attend (this layer's for non-shared, the target layer's
// for YOCO-shared). `rope_full` selects the proportional cache path.
void Gemma4AttnBlock(Dev d, DBuf& o_out, const Gemma4LayerWeights& w, const Gemma4Layout& g,
                     const Tensor& dhn, const StepInputs& si,
                     const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                     int64_t T, int64_t Dh, bool rope_full,
                     const Tensor& ones_dh, const Tensor* prop_cache,
                     std::optional<int64_t> sliding_window,
                     double rope_theta_sliding) {
  const int64_t H = g.hidden;
  const int64_t Hq = g.num_q_heads;
  const int64_t Hkv = w.num_kv_heads > 0 ? w.num_kv_heads : g.num_kv_heads;
  const float eps = 1e-6f;  // rms_norm_eps (E4B)
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma4: KV cache head dims mismatch this layer (heterogeneous KV — "
           "runner must allocate per-layer head_dim; see gemma4.h G1 note)");

  // TLS temps across layers (decode T=1 thrash).
  struct AttnTls {
    int dev = -1;
    int64_t T = 0, qdim = 0, kdim = 0, Hq = 0, Dh = 0;
    std::optional<DBuf> q, k, v, qkv, attn;
  };
  static thread_local AttnTls tls;
  if (tls.dev != d.q.device.index || tls.T != T || tls.qdim != qdim || tls.kdim != kdim ||
      tls.Hq != Hq || tls.Dh != Dh) {
    tls.q.emplace(d, adt, std::vector<int64_t>{T, qdim});
    tls.k.emplace(d, adt, std::vector<int64_t>{T, kdim});
    tls.v.emplace(d, adt, std::vector<int64_t>{T, kdim});
    tls.qkv.emplace(d, adt, std::vector<int64_t>{T, qdim + 2 * kdim});
    tls.attn.emplace(d, adt, std::vector<int64_t>{T, Hq, Dh});
    tls.dev = d.q.device.index;
    tls.T = T;
    tls.qdim = qdim;
    tls.kdim = kdim;
    tls.Hq = Hq;
    tls.Dh = Dh;
  }
  DBuf& q = *tls.q;
  DBuf& k = *tls.k;
  DBuf& v = *tls.v;
  DBuf& qkv = *tls.qkv;
  DBuf& attn = *tls.attn;

  // Merged QKVParallelLinear: one MatmulBT + QkvSplit (default).
  {
    Tensor wqkv = ResidentWeight(d, w.attn.qkv_proj);
    if (MergedQkvEnabled()) {
      vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
      vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
    } else {
      Tensor wq = wqkv.Slice(0, 0, qdim);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      vt::MatmulBT(d.q, q.t(), dhn, wq);
      vt::MatmulBT(d.q, k.t(), dhn, wk);
      vt::MatmulBT(d.q, v.t(), dhn, wv);
    }
  }

  // Q norm (PLAIN per-head RMSNorm over Dh). Always applied.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  Tensor wqn = ResidentWeight(d, w.attn.q_norm, {Dh});
  vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});

  if (!w.is_kv_shared) {
    // K norm (plain), then RoPE(q,k), then weight-less V norm (ones weight).
    Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
    Tensor wkn = ResidentWeight(d, w.attn.k_norm, {Dh});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
  }

  // RoPE (both q,k roped; k discarded on shared layers, matching vLLM's
  // rotary_emb(pos,q,k)[0] which still rotates k).
  if (rope_full) {
    VT_CHECK(prop_cache != nullptr, "gemma4: full layer needs proportional cache");
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(Dh);
    ra.is_neox_style = true;
    Tensor kk = k3;
    vt::RopeFromCache(d.q, q3, w.is_kv_shared ? nullptr : &kk, si.positions.t(),
                      *prop_cache, ra);
  } else {
    vt::RopeArgs ra;
    ra.base = static_cast<float>(rope_theta_sliding);
    ra.rotary_dim = static_cast<int>(Dh);
    vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);
  }

  if (!w.is_kv_shared) {
    // Weight-less V norm (ones weight = identity scale), then cache K/V.
    Tensor v2 = Reshape(v.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, v2, v2, ones_dh, vt::RmsNormArgs{eps, false});

    Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
    Tensor kw = k3;
    Tensor vw = v3;
    // Cast buffers only when dtype differs (rare for bf16 KV).
    if (kv.dtype != adt) {
      DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
      DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
      if (kv.dtype == DType::kBF16) {
        vt::CastBf16(d.q, kcast.t(), k3);
        vt::CastBf16(d.q, vcast.t(), v3);
      } else {
        vt::CastF32(d.q, kcast.t(), k3);
        vt::CastF32(d.q, vcast.t(), v3);
      }
      kw = kcast.t();
      vw = vcast.t();
      Tensor k_cache = KvSlice(kv, d.q.device, 0);
      Tensor v_cache = KvSlice(kv, d.q.device, 1);
      vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());
    } else {
      Tensor k_cache = KvSlice(kv, d.q.device, 0);
      Tensor v_cache = KvSlice(kv, d.q.device, 1);
      vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());
    }
  }

  // Paged GQA attention: scale = 1.0 (Q/K norms carry the scale).
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::PagedAttentionArgs pa{1.0f, meta.causal};
  if (g.attn_logit_softcap > 0.0f) pa.logits_soft_cap = g.attn_logit_softcap;
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  if (sliding_window.has_value() && *sliding_window > 0)
    pa.window_size = vt::AttentionWindow{static_cast<int32_t>(*sliding_window - 1), 0};
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.attn.o_proj);
  VT_CHECK(o_out.t().shape[0] == T && o_out.t().shape[1] == H, "gemma4 attn o_out shape");
  vt::MatmulBT(d.q, o_out.t(), o_in, wo);
}

// GeGLU MLP (gemma4.py::Gemma4MLP).
void Gemma4MlpBlock(Dev d, DBuf& down_out, const Gemma4MlpWeights& w, int64_t H, int64_t I,
                    const Tensor& dh2, int64_t T) {
  // Dense GeGLU: TLS reuse of large gate_up [T,2I] + act [T,I] across layers.
  // down_out is caller-owned (layer TLS) so hipGraph capture keeps stable pointers.
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);
  struct MlpTls {
    int dev = -1;
    int64_t T = 0, I = 0;
    std::optional<DBuf> gu, act;
  };
  static thread_local MlpTls tls;
  if (tls.dev != d.q.device.index || tls.T != T || tls.I != I) {
    tls.gu.emplace(d, DType::kBF16, std::vector<int64_t>{T, 2 * I});
    tls.act.emplace(d, DType::kBF16, std::vector<int64_t>{T, I});
    tls.dev = d.q.device.index;
    tls.T = T;
    tls.I = I;
  }
  DBuf& gu = *tls.gu;
  DBuf& act = *tls.act;
  vt::MatmulBT(d.q, gu.t(), dh2, wgu);
  vt::GeluAndMul(d.q, act.t(), gu.t());
  Tensor wd = ResidentWeight(d, w.down_proj);
  VT_CHECK(down_out.t().shape[0] == T && down_out.t().shape[1] == H, "gemma4 mlp out shape");
  vt::MatmulBT(d.q, down_out.t(), act.t(), wd);
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// --- Decode-graph prep: Layers + Logits as separate steps -------------------
// Embed/PLE stay in ForwardBody (host token ids / optional mm embeds). The layer
// stack and lm_head are factored so a future Gemma4DecodeGraph can:
//   EmbedInto(persistent hidden) → capture(ForwardGemma4Layers) → Logits
// Behavior is intentionally identical to the previous monolithic loop.

void ForwardGemma4Layers(Dev d, DBuf& hidden, const Gemma4Layout& g,
                         const Gemma4Weights& weights, const HfConfig& /*config*/,
                         const CommonAttentionMetadata& attn_meta,
                         const std::vector<PagedKvCache>& attn_kv, StepInputs& si,
                         DBuf& ones_sliding, DBuf& ones_full, DBuf& prop_cache,
                         DBuf& ple_by_layer, int64_t T, int64_t H, int64_t I, int64_t L,
                         int64_t ple) {
  const float eps = 1e-6f;
  const vt::RmsNormArgs plain{eps, false};
  const size_t ple_row_bytes = static_cast<size_t>(ple) * sizeof(uint16_t);
  const size_t th_bytes = static_cast<size_t>(T) * static_cast<size_t>(H) * sizeof(uint16_t);

  struct LayerTls {
    int dev = -1;
    int64_t T = 0, H = 0, I = 0, ple = 0;
    std::optional<DBuf> dhn, attn_o, mlp_o, h1, dh2, h2, moe_in;
    std::optional<DBuf> gate_lin, ple_l, gated, contrib;
  };
  static thread_local LayerTls lt;
  if (lt.dev != d.q.device.index || lt.T != T || lt.H != H || lt.I != I || lt.ple != ple) {
    auto mk = [&](int64_t a, int64_t b) {
      return DBuf(d, DType::kBF16, std::vector<int64_t>{a, b});
    };
    lt.dhn.emplace(mk(T, H));
    lt.attn_o.emplace(mk(T, H));
    lt.mlp_o.emplace(mk(T, H));
    lt.h1.emplace(mk(T, H));
    lt.dh2.emplace(mk(T, H));
    lt.h2.emplace(mk(T, H));
    lt.moe_in.emplace(mk(T, H));
    lt.contrib.emplace(mk(T, H));
    if (ple > 0) {
      lt.gate_lin.emplace(mk(T, ple));
      lt.ple_l.emplace(mk(T, ple));
      lt.gated.emplace(mk(T, ple));
    } else {
      lt.gate_lin.reset();
      lt.ple_l.reset();
      lt.gated.reset();
    }
    lt.dev = d.q.device.index;
    lt.T = T;
    lt.H = H;
    lt.I = I;
    lt.ple = ple;
  }
  DBuf& dhn = *lt.dhn;
  DBuf& attn_o = *lt.attn_o;
  DBuf& mlp_o = *lt.mlp_o;
  DBuf& h1 = *lt.h1;
  DBuf& dh2 = *lt.dh2;
  DBuf& h2 = *lt.h2;
  DBuf& moe_in = *lt.moe_in;
  DBuf& contrib = *lt.contrib;

  for (int64_t l = 0; l < L; ++l) {
    const Gemma4LayerWeights& w = weights.layers[static_cast<size_t>(l)];
    const int64_t Dh = w.head_dim;
    const bool full = w.is_full_attention;
    const Tensor ones_dh = full ? ones_full.t() : ones_sliding.t();
    std::optional<int64_t> window;
    if (!full) window = g.sliding_window;

    const int64_t kv_idx = w.is_kv_shared ? w.kv_target_layer : l;
    VT_CHECK(kv_idx >= 0 && kv_idx < L, "gemma4: bad kv target layer");
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(kv_idx)];

    Tensor w_in = ResidentWeight(d, w.input_layernorm, {H});
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, plain);

    static const int layer_trace = [] {
      const char* e = std::getenv("VT_GEMMA4_LAYER_TRACE");
      if (e && e[0] == '2') return 2;
      if (e && e[0] == '1') return 1;
      const char* p = std::getenv("VT_GEMMA4_PROFILE");
      return (p && p[0] == '1') ? 1 : 0;
    }();
    using clock = std::chrono::steady_clock;
    const bool layer_prof = layer_trace >= 1;
    const bool layer_hb = layer_trace >= 2;
    if (layer_hb) {
      std::fprintf(stderr,
                   "INFO gemma4-layer begin l=%lld/%lld T=%lld full=%d moe=%d kv=%lld\n",
                   static_cast<long long>(l), static_cast<long long>(L),
                   static_cast<long long>(T), full ? 1 : 0, w.moe.enabled ? 1 : 0,
                   static_cast<long long>(kv_idx));
      std::fflush(stderr);
    }
    const auto t0 = layer_prof ? clock::now() : clock::time_point{};

    Gemma4AttnBlock(d, attn_o, w, g, dhn.t(), si, attn_meta, kv, T, Dh, full,
                    ones_dh, full ? &prop_cache.t() : nullptr, window,
                    g.rope_theta_sliding);

    Tensor w_pa = ResidentWeight(d, w.post_attention_layernorm, {H});
    vt::RmsNormPlusAdd(d.q, h1.t(), attn_o.t(), w_pa, hidden.t(), plain);

    if (layer_prof) d.b.Synchronize(d.q);
    const auto t1 = layer_prof ? clock::now() : clock::time_point{};
    if (layer_hb) {
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::fprintf(stderr, "INFO gemma4-layer attn-done l=%lld ms=%.2f\n",
                   static_cast<long long>(l), ms);
      std::fflush(stderr);
    }

    Tensor w_pf = ResidentWeight(d, w.pre_feedforward_layernorm, {H});
        if (w.moe.enabled) {
    #if defined(VLLM_CPP_HIP)
          // MLP ∥ MoE: both only read h1. Side stream runs MLP; main runs MoE; join
          // before DualRmsNorm. hipBLAS handle is TLS single-thread — launch MLP fully
          // on side first (async), then MoE on main (concurrent GPU). Opt out:
          // VT_GEMMA4_MLP_MOE_PARALLEL=0.
          static const bool kPar = [] {
            const char* e = std::getenv("VT_GEMMA4_MLP_MOE_PARALLEL");
            // Lab A/B 2026-08-10: wall decode flat vs serial (~51.2); keep opt-in.
            return e && e[0] == '1';
          }();
          struct SideTls {
            int dev = -1;
            Queue q{};
            hipEvent_t ev_h1 = nullptr;
            hipEvent_t ev_mlp = nullptr;
            bool live = false;
          };
          static thread_local SideTls st;
          const int dev = d.q.device.index;
          const bool can_par = kPar && d.q.device.type == vt::DeviceType::kROCM && dev >= 0 &&
                               d.q.handle != nullptr;
          if (can_par) {
            if (st.dev != dev || !st.live) {
              if (st.live) {
                if (st.ev_h1) (void)hipEventDestroy(st.ev_h1);
                if (st.ev_mlp) (void)hipEventDestroy(st.ev_mlp);
                vt::DestroyQueue(st.q);
                st = SideTls{};
              }
              (void)hipSetDevice(dev);
              st.q = vt::CreateQueue(d.q.device);
              (void)hipEventCreateWithFlags(&st.ev_h1, hipEventDisableTiming);
              (void)hipEventCreateWithFlags(&st.ev_mlp, hipEventDisableTiming);
              st.dev = dev;
              st.live = true;
            }
            auto* main_st = static_cast<hipStream_t>(d.q.handle);
            auto* side_st = static_cast<hipStream_t>(st.q.handle);
            (void)hipEventRecord(st.ev_h1, main_st);
            (void)hipStreamWaitEvent(side_st, st.ev_h1, 0);

            Dev d_mlp = d;
            d_mlp.q = st.q;
            vt::RmsNorm(d_mlp.q, dh2.t(), h1.t(), w_pf, plain);
            Gemma4MlpBlock(d_mlp, mlp_o, w.mlp, H, I, dh2.t(), T);
            (void)hipEventRecord(st.ev_mlp, side_st);

            Tensor w_pf2 = ResidentWeight(d, w.moe.pre_feedforward_layernorm_2, {H});
            vt::RmsNorm(d.q, moe_in.t(), h1.t(), w_pf2, plain);
            Gemma4MoeScratch moe_out =
                RunGemma4Moe(d.q, w.moe, /*router_in=*/h1.t(), /*expert_in=*/moe_in.t(), T, H, eps);
            (void)hipStreamWaitEvent(main_st, st.ev_mlp, 0);

            Tensor w_p1 = ResidentWeight(d, w.moe.post_feedforward_layernorm_1, {H});
            Tensor w_p2 = ResidentWeight(d, w.moe.post_feedforward_layernorm_2, {H});
            Tensor w_pff = ResidentWeight(d, w.post_feedforward_layernorm, {H});
            vt::DualRmsNormPlusRes(d.q, h2.t(), mlp_o.t(), w_p1, moe_out.tensor, w_p2, w_pff, h1.t(),
                                   plain);
          } else
    #endif
          {
            vt::RmsNorm(d.q, dh2.t(), h1.t(), w_pf, plain);
            Gemma4MlpBlock(d, mlp_o, w.mlp, H, I, dh2.t(), T);

            if (layer_prof) d.b.Synchronize(d.q);
            const auto t2 = layer_prof ? clock::now() : clock::time_point{};

            Tensor w_pf2 = ResidentWeight(d, w.moe.pre_feedforward_layernorm_2, {H});
            vt::RmsNorm(d.q, moe_in.t(), h1.t(), w_pf2, plain);
            Gemma4MoeScratch moe_out =
                RunGemma4Moe(d.q, w.moe, /*router_in=*/h1.t(), /*expert_in=*/moe_in.t(), T, H, eps);
            Tensor w_p1 = ResidentWeight(d, w.moe.post_feedforward_layernorm_1, {H});
            Tensor w_p2 = ResidentWeight(d, w.moe.post_feedforward_layernorm_2, {H});
            Tensor w_pff = ResidentWeight(d, w.post_feedforward_layernorm, {H});
            // moe_out is TLS-stable for T=1 (graph-safe pointer).
            vt::DualRmsNormPlusRes(d.q, h2.t(), mlp_o.t(), w_p1, moe_out.tensor, w_p2, w_pff, h1.t(),
                                   plain);
            if (layer_prof) {
              (void)t2;
            }
          }
        } else {
          vt::RmsNorm(d.q, dh2.t(), h1.t(), w_pf, plain);
          Gemma4MlpBlock(d, mlp_o, w.mlp, H, I, dh2.t(), T);
          Tensor w_pff = ResidentWeight(d, w.post_feedforward_layernorm, {H});
          vt::RmsNormPlusAdd(d.q, h2.t(), mlp_o.t(), w_pff, h1.t(), plain);
        }

        // FF done. Optional early sync boundary for ff_us; final dump after PLE.
        clock::time_point t_ff{};
        if (layer_prof) {
          d.b.Synchronize(d.q);
          t_ff = clock::now();
          if (layer_hb) {
            const double ff_ms =
                std::chrono::duration<double, std::milli>(t_ff - t1).count();
            std::fprintf(stderr, "INFO gemma4-layer end l=%lld ff_ms=%.2f (mlp||moe possible)\n",
                         static_cast<long long>(l), ff_ms);
            std::fflush(stderr);
          }
        }

        if (ple > 0) {
          Tensor wg = ResidentWeight(d, w.per_layer_input_gate, {ple, H});
          DBuf& gate_lin = *lt.gate_lin;
          DBuf& ple_l = *lt.ple_l;
          DBuf& gated = *lt.gated;
          vt::MatmulBT(d.q, gate_lin.t(), h2.t(), wg);
          const size_t layer_bytes = static_cast<size_t>(T) * ple_row_bytes;
          const char* src = static_cast<const char*>(ple_by_layer.ptr()) +
                            static_cast<size_t>(l) * layer_bytes;
          d.b.Copy(d.q, ple_l.ptr(), src, layer_bytes);
          vt::GeluMulSeparate(d.q, gated.ptr(), gate_lin.ptr(), ple_l.ptr(), T * ple,
                              DType::kBF16);

          Tensor wp = ResidentWeight(d, w.per_layer_projection, {H, ple});
          vt::MatmulBT(d.q, contrib.t(), gated.t(), wp);
          Tensor w_pln = ResidentWeight(d, w.post_per_layer_input_norm, {H});
          vt::RmsNorm(d.q, contrib.t(), contrib.t(), w_pln, plain);
          vt::Add(d.q, h2.t(), h2.t(), contrib.t());
        }

        if (!w.layer_scalar.Empty()) {
          const double scalar = static_cast<double>(ReadBf16Scalar(w.layer_scalar));
          vt::MulScalar(d.q, h2.t(), h2.t(), scalar);
        }

        d.b.Copy(d.q, hidden.ptr(), h2.ptr(), th_bytes);

        // Pure-decode only (T==1). Includes PLE+scalar+hidden copy in ple_us.
        if (layer_prof && T == 1) {
          d.b.Synchronize(d.q);
          const auto t_end = clock::now();
          static std::atomic<uint64_t> n{0}, us_attn{0}, us_ff{0}, us_ple{0};
          static std::atomic<uint64_t> n_full{0}, us_attn_full{0};
          static std::atomic<uint64_t> n_slide{0}, us_attn_slide{0};
          auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
          };
          const auto a_us = static_cast<uint64_t>(us(t0, t1));
          const auto f_us = static_cast<uint64_t>(us(t1, t_ff));
          const auto p_us = static_cast<uint64_t>(us(t_ff, t_end));
          us_attn.fetch_add(a_us, std::memory_order_relaxed);
          us_ff.fetch_add(f_us, std::memory_order_relaxed);
          us_ple.fetch_add(p_us, std::memory_order_relaxed);
          const uint64_t c = n.fetch_add(1, std::memory_order_relaxed) + 1;
          if (full) {
            us_attn_full.fetch_add(a_us, std::memory_order_relaxed);
            n_full.fetch_add(1, std::memory_order_relaxed);
          } else {
            us_attn_slide.fetch_add(a_us, std::memory_order_relaxed);
            n_slide.fetch_add(1, std::memory_order_relaxed);
          }
          if (c == 30 || c == 240 || c % 960 == 0) {
            const double ta = static_cast<double>(us_attn.load());
            const double tf = static_cast<double>(us_ff.load());
            const double tp = static_cast<double>(us_ple.load());
            const double tot = ta + tf + tp + 1.0;
            const double nf = static_cast<double>(std::max<uint64_t>(1, n_full.load()));
            const double ns = static_cast<double>(std::max<uint64_t>(1, n_slide.load()));
            std::fprintf(stderr,
                         "gemma4 decode-T1 profile: layer_calls=%llu attn_us=%.1f ff_us=%.1f "
                         "ple_us=%.1f (attn%%=%.0f ff%%=%.0f ple%%=%.0f) full_attn_us=%.1f "
                         "slide_attn_us=%.1f est_ms/tok=%.1f\n",
                         static_cast<unsigned long long>(c), ta / c, tf / c, tp / c,
                         100.0 * ta / tot, 100.0 * tf / tot, 100.0 * tp / tot,
                         us_attn_full.load() / nf, us_attn_slide.load() / ns,
                         (ta + tf + tp) / c * 30.0 / 1000.0);
            std::fflush(stderr);
          }
        }
  }  // layers
}

// If logits_out non-null and shaped [n_out,vocab] f32, write there (graph-stable).
DBuf ForwardGemma4Logits(Dev d, DBuf& hidden, const Gemma4Weights& weights,
                         const Gemma4Layout& g, const HfConfig& /*config*/,
                         const std::vector<int32_t>& logits_indices, int64_t T, int64_t H,
                         int64_t vocab, DBuf* logits_out = nullptr) {
  const float eps = 1e-6f;
  const vt::RmsNormArgs plain{eps, false};
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  // TLS dnorm for graph stability when T fixed
  struct NormTls {
    int dev = -1;
    int64_t T = 0, H = 0;
    std::optional<DBuf> n;
  };
  static thread_local NormTls nt;
  if (nt.dev != d.q.device.index || nt.T != T || nt.H != H || !nt.n) {
    nt.n.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
    nt.dev = d.q.device.index;
    nt.T = T;
    nt.H = H;
  }
  DBuf& dnorm = *nt.n;
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, plain);

  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  if (logits_out != nullptr) {
    VT_CHECK(logits_out->t().shape[0] == n_out && logits_out->t().shape[1] == vocab,
             "gemma4 logits_out shape");
    if (tied)
      vt::MatmulBT(d.q, logits_out->t(), src, lm);
    else
      vt::Matmul(d.q, logits_out->t(), src, lm);
    if (g.final_logit_softcap > 0.0f)
      vt::SoftCap(d.q, logits_out->t(), logits_out->t(), g.final_logit_softcap);
    // Return a non-owning move-friendly empty: caller holds logits_out.
    // Use a dummy 1x1 that won't be used — callers with logits_out ignore return.
    return DBuf(d, DType::kF32, std::vector<int64_t>{1, 1});
  }
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);

  if (g.final_logit_softcap > 0.0f)
    vt::SoftCap(d.q, logits.t(), logits.t(), g.final_logit_softcap);
  return logits;
}

// CLAIM-GEMMA4-MM-E2E: `inputs_embeds_override` (host bf16 [T*H]) and
// `ple_token_ids` (host i32 [T]) are the multimodal seam. When BOTH are null the
// function is byte-identical to the text path (embed from token_ids, PLE from
// token_ids) — the two text call sites pass null so the SACRED text 32/32 gate is
// unchanged by construction. When set (the registered mm forward), the hidden
// stream STARTS from the already-merged + already-sqrt(H)-scaled inputs_embeds
// (text rows scaled, vision soft-token rows the embed_vision projector output,
// gemma4_mm.py:1319/embed_input_ids masked-scatter), and the PLE embed_tokens_
// per_layer lookup uses the mm-masked ids (gemma4_mm.py:1962-1973). ple_proj still
// projects the merged inputs_embeds (project_per_layer_inputs, gemma4.py:908-912).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Gemma4Weights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices,
                 const std::vector<uint16_t>* inputs_embeds_override = nullptr,
                 const std::vector<int32_t>* ple_token_ids = nullptr) {
  const int64_t T = inputs_embeds_override != nullptr
                        ? static_cast<int64_t>(positions.size())
                        : static_cast<int64_t>(token_ids.size());
  const Gemma4Layout g = MakeLayout(config);
  const int64_t H = g.hidden;
  const int64_t L = g.num_layers;
  const int64_t ple = g.ple_dim;
  const int64_t I = config.intermediate_size;
  const int64_t vocab = config.vocab_size;
  const float eps = 1e-6f;
  const vt::RmsNormArgs plain{eps, false};
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma4: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(L),
           "gemma4: one PagedKvCache per layer required");
  // mm seam: the merged-embeds override and the PLE ids must both match T.
  if (inputs_embeds_override != nullptr)
    VT_CHECK(static_cast<int64_t>(inputs_embeds_override->size()) == T * H,
             "gemma4 mm: inputs_embeds_override size != T*H");
  const std::vector<int32_t>& ple_ids =
      ple_token_ids != nullptr ? *ple_token_ids : token_ids;
  VT_CHECK(static_cast<int64_t>(ple_ids.size()) == T,
           "gemma4 mm: ple_token_ids length != T");

  // Ones weights for the weight-less V-norm (identity scale), per head_dim.
  // TLS: was H2D every decode token (tiny but free to kill).
  struct OnesTls {
    int dev = -1;
    int64_t dh_s = 0, dh_f = 0;
    std::optional<DBuf> sliding, full;
  };
  static thread_local OnesTls ones_tls;
  const int dev_i = d.q.device.index;
  if (ones_tls.dev != dev_i || ones_tls.dh_s != g.head_dim_sliding ||
      ones_tls.dh_f != g.head_dim_full || !ones_tls.sliding || !ones_tls.full) {
    auto make_ones = [&](int64_t Dh) {
      std::vector<uint16_t> h(static_cast<size_t>(Dh), vt::F32ToBF16(1.0f));
      return DBuf(d, DType::kBF16, {Dh}, h.data());
    };
    ones_tls.sliding.emplace(make_ones(g.head_dim_sliding));
    ones_tls.full.emplace(make_ones(g.head_dim_full));
    ones_tls.dev = dev_i;
    ones_tls.dh_s = g.head_dim_sliding;
    ones_tls.dh_f = g.head_dim_full;
  }
  DBuf& ones_sliding = *ones_tls.sliding;
  DBuf& ones_full = *ones_tls.full;

  // Proportional RoPE cache for full-attention layers (shared across them).
  // Grow-only TLS slot — MUST NOT rebuild 0..max_pos every decode token.
  int64_t max_pos = 0;
  for (int32_t p : positions) max_pos = std::max<int64_t>(max_pos, p);
  static thread_local std::optional<DBuf> prop_slot;
  static thread_local int64_t prop_cap = -1;
  static thread_local int64_t prop_geom_Dh = -1;
  static thread_local double prop_geom_base = 0.0;
  static thread_local double prop_geom_partial = 0.0;
  EnsureProportionalRopeCache(d, g, g.head_dim_full, max_pos, prop_slot, prop_cap,
                              prop_geom_Dh, prop_geom_base, prop_geom_partial);
  DBuf& prop_cache = *prop_slot;

  // --- Token embedding * sqrt(hidden) (bf16). Also the PLE-projection input. ---
  // mm seam: when inputs_embeds_override is set the hidden stream STARTS from the
  // already-merged, already-sqrt(H)-scaled embeddings (text rows scaled at merge
  // time, vision soft-token rows the raw embed_vision output). No embed lookup /
  // no MulScalar — mirrors gemma4.py forward inputs_embeds branch (:908-909).
  DBuf tok(d, DType::kBF16, {T, H});
  if (inputs_embeds_override != nullptr) {
    tok = DBuf(d, DType::kBF16, {T, H}, inputs_embeds_override->data());
  } else {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, tok.t(), dtab, dids.t());
    const float nsqrt = std::sqrt(static_cast<float>(H));
    const double normalizer =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
    vt::MulScalar(d.q, tok.t(), tok.t(), normalizer);
  }

  // --- Per-Layer Embeddings precompute (gemma4.py:845-898). ---
  // Skipped when hidden_size_per_layer_input==0 (google/gemma-4-12B-it dense).
  DBuf ple_input(d, DType::kBF16, ple > 0 ? std::vector<int64_t>{T, L, ple}
                                          : std::vector<int64_t>{1, 1, 1});
  if (ple > 0) {
    // ple_emb = embed_tokens_per_layer(ids) * sqrt(ple)         -> [T, L*ple]
    // ple_proj = (per_layer_model_projection @ tok) * hidden^-0.5 -> [T, L*ple]
    //            reshape [T,L,ple]; RMSNorm(plain) over ple.
    // ple_input = (ple_proj + ple_emb) * rsqrt(2)               -> [T, L, ple]
    const int64_t LP = L * ple;
    DBuf ple_emb(d, DType::kBF16, {T, LP});
    {
      Tensor dtab = ResidentWeight(d, weights.embed_tokens_per_layer, {vocab, LP});
      DBuf dids(d, DType::kI32, {T}, ple_ids.data());
      vt::Embedding(d.q, ple_emb.t(), dtab, dids.t());
    }
    vt::MulScalar(d.q, ple_emb.t(), ple_emb.t(),
                  static_cast<double>(g.embed_scale_ple));

    DBuf ple_proj(d, DType::kBF16, {T, LP});
    Tensor wproj = ResidentWeight(d, weights.per_layer_model_projection, {LP, H});
    vt::MatmulBT(d.q, ple_proj.t(), tok.t(), wproj);
    vt::MulScalar(d.q, ple_proj.t(), ple_proj.t(), g.proj_scale);

    // RMSNorm(plain) over the last dim ple.
    Tensor proj2 = Reshape(ple_proj.t(), {T * L, ple});
    Tensor wpn = ResidentWeight(d, weights.per_layer_projection_norm, {ple});
    vt::RmsNorm(d.q, proj2, proj2, wpn, plain);

    // (ple_proj + ple_emb) * rsqrt(2).
    vt::Add(d.q, ple_input.t(), Reshape(ple_proj.t(), {T, L, ple}),
            Reshape(ple_emb.t(), {T, L, ple}));
    vt::MulScalar(d.q, ple_input.t(), ple_input.t(), g.input_scale);
  }

  // Layout [L,T,ple] so each layer's slice is one contiguous D2D (not T row gathers).
  DBuf ple_by_layer(d, DType::kBF16,
                    ple > 0 ? std::vector<int64_t>{L, T, ple} : std::vector<int64_t>{1, 1, 1});
  if (ple > 0) {
    const size_t row = static_cast<size_t>(ple) * sizeof(uint16_t);
    const auto* src = static_cast<const char*>(ple_input.ptr());
    auto* dst = static_cast<char*>(ple_by_layer.ptr());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t li = 0; li < L; ++li) {
        const size_t s_off =
            (static_cast<size_t>(t) * static_cast<size_t>(L) + static_cast<size_t>(li)) * row;
        const size_t d_off =
            (static_cast<size_t>(li) * static_cast<size_t>(T) + static_cast<size_t>(t)) * row;
        CopyRow(d, dst + d_off, src + s_off, row);
      }
    }
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  // hidden state stream (each layer fully materializes h; no separate residual).
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), tok.ptr(),
           static_cast<size_t>(T) * static_cast<size_t>(H) * sizeof(uint16_t));

  // Layers (capturable region for future decode graph) then logits.
  ForwardGemma4Layers(d, hidden, g, weights, config, attn_meta, attn_kv, si, ones_sliding,
                      ones_full, prop_cache, ple_by_layer, T, H, I, L, ple);
  return ForwardGemma4Logits(d, hidden, weights, g, config, logits_indices, T, H, vocab);
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Gemma4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Gemma4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

std::vector<float> Gemma4Model::ForwardMm(
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& ple_token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // token_ids is unused when inputs_embeds_override is set (T comes from
  // positions); pass an empty vector so the mm seam never dereferences it.
  const std::vector<int32_t> no_tokens;
  DBuf dlogits = ForwardBody(d, no_tokens, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, &inputs_embeds_bf16,
                             &ple_token_ids);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}


// ─── Gemma4DecodeGraph (opt-in pure-decode hipGraph) ─────────────────────────
namespace {

bool FrameworkCudaGraphOff() {
  const char* e = std::getenv("VLLM_CPP_CUDAGRAPH");
  return e != nullptr && e[0] == '0';
}

void HostCopyInPlace(std::vector<int32_t>& dst, const std::vector<int32_t>& src) {
  if (dst.size() != src.size()) dst = src;
  else if (!src.empty()) std::memcpy(dst.data(), src.data(), src.size() * sizeof(int32_t));
}

template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) dst = src;
  else if (!src.empty()) std::copy(src.begin(), src.end(), dst.begin());
}

void RefreshHostAttnMeta(CommonAttentionMetadata& dst, const CommonAttentionMetadata& am) {
  CopyInPlace(dst.slot_mapping, am.slot_mapping);
  CopyInPlace(dst.block_table_tensor, am.block_table_tensor);
  CopyInPlace(dst.seq_lens, am.seq_lens);
  CopyInPlace(dst.query_start_loc, am.query_start_loc);
  dst.num_reqs = am.num_reqs;
  dst.num_actual_tokens = am.num_actual_tokens;
  dst.max_query_len = am.max_query_len;
  dst.max_seq_len = am.max_seq_len;
  dst.block_table_num_cols = am.block_table_num_cols;
  dst.causal = am.causal;
}

// Host sources MUST be process-persistent (captured H2D bakes addresses).
void RefreshStepInputsInPlace(Dev d, StepInputs& s, const std::vector<int32_t>& positions,
                              const CommonAttentionMetadata& am, const HfConfig& cfg) {
  const int64_t T = static_cast<int64_t>(positions.size());
  VT_CHECK(s.positions.t().shape[0] == T, "gemma4 graph: T mismatch");
  d.b.Copy(d.q, s.positions.ptr(), positions.data(),
           static_cast<size_t>(T) * sizeof(int32_t));
  d.b.Copy(d.q, s.slot_mapping.ptr(), am.slot_mapping.data(),
           static_cast<size_t>(T) * sizeof(int64_t));
  d.b.Copy(d.q, s.seq_lens.ptr(), am.seq_lens.data(),
           static_cast<size_t>(am.num_reqs) * sizeof(int32_t));
  d.b.Copy(d.q, s.query_start_loc.ptr(), am.query_start_loc.data(),
           static_cast<size_t>(am.num_reqs + 1) * sizeof(int32_t));
  const size_t bt_bytes = static_cast<size_t>(am.num_reqs) *
                          static_cast<size_t>(am.block_table_num_cols) * sizeof(int32_t);
  d.b.Copy(d.q, s.block_table.ptr(), am.block_table_tensor.data(), bt_bytes);
  const int rot = static_cast<int>(cfg.rotary_dim);
  if (rot > 0) {
    vt::RopeCosSinCache(d.q, s.cos_sin.t(), s.positions.t(), MakeRopeArgs(cfg));
    if (RopeCacheEnabled()) vt::CastBf16(d.q, s.cos_sin_bf16.t(), s.cos_sin.t());
  }
}

}  // namespace

struct Gemma4DecodeGraph::Impl {
  const Gemma4Weights& weights;
  HfConfig config;
  vt::Queue queue;
  Gemma4Layout layout{};
  bool enabled = false;
  bool failed = false;
  int warm_steps = 0;  // eager decode steps before first capture
  bool captured = false;
  void* graph = nullptr;
  int64_t replays = 0;
  static constexpr int kWarmBeforeCapture = 4;
  int fa_cols = -1;
  int64_t prop_cap = 0;

  std::vector<int32_t> h_tok, h_pos;
  CommonAttentionMetadata h_attn;  // persistent host meta (graph-safe H2D sources)
  // Persistent device
  std::optional<DBuf> hidden, logits, ones_sliding, ones_full, prop_cache, ple_by_layer;
  std::optional<DBuf> d_tok;  // embed ids — stable, not pool-steal after capture
  std::optional<StepInputs> si;

  Impl(const Gemma4Weights& w, const HfConfig& c, vt::Queue q)
      : weights(w), config(c), queue(q) {
    layout = MakeLayout(config);
    enabled = !FrameworkCudaGraphOff() &&
              vt::GetBackend(queue.device.type).SupportsGraphCapture();
  }
  ~Impl() {
    if (graph) {
      try {
        vt::GetBackend(queue.device.type).DestroyGraph(graph);
      } catch (...) {
      }
      graph = nullptr;
    }
  }
};

Gemma4DecodeGraph::Gemma4DecodeGraph(const Gemma4Weights& weights, const HfConfig& config,
                                     vt::Queue queue)
    : impl_(std::make_unique<Impl>(weights, config, queue)) {}
Gemma4DecodeGraph::~Gemma4DecodeGraph() = default;
bool Gemma4DecodeGraph::captured() const { return impl_ && impl_->captured; }
int64_t Gemma4DecodeGraph::replay_count() const { return impl_ ? impl_->replays : 0; }

bool Gemma4DecodeGraphEnabled() {
  const char* e = std::getenv("VLLM_CPP_GEMMA4_DECODE_GRAPH");
  return e != nullptr && e[0] == '1';  // default OFF until lab A/B green
}

ForwardLogits Gemma4DecodeGraph::Step(const std::vector<int32_t>& token_ids,
                                      const std::vector<int32_t>& positions,
                                      const CommonAttentionMetadata& attn_meta,
                                      const std::vector<PagedKvCache>& attn_kv) {
  Impl& I = *impl_;
  Dev d{vt::GetBackend(I.queue.device.type), I.queue};
  Backend& b = vt::GetBackend(I.queue.device.type);
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = I.layout.hidden;
  const int64_t L = I.layout.num_layers;
  const int64_t ple = I.layout.ple_dim;
  const int64_t vocab = I.config.vocab_size;
  const int64_t Inter = I.config.intermediate_size;
  const std::vector<int32_t> kNoGather;

  // Only pure single-token decode for v1 graph.
  // Dual-GPU peer + hipGraph: capture can succeed but replay page-faults on peer
  // memory (lab 2026-08-09). Refuse graph when experts are off compute device.
  if (!I.failed) {
    const int cdev = I.queue.device.index;
    for (const auto& layer : I.weights.layers) {
      if (layer.moe.enabled && layer.moe.experts.fp8_native_resident &&
          layer.moe.experts.dev_id >= 0 && layer.moe.experts.dev_id != cdev) {
        I.failed = true;
        static bool once = false;
        if (!once) {
          std::fprintf(stderr,
                       "[Gemma4DecodeGraph] peer MoE (gpu %d vs compute %d) — graph off "
                       "(replay page-fault; need single-dev or graph-safe peer)\n",
                       layer.moe.experts.dev_id, cdev);
          once = true;
        }
        break;
      }
    }
  }
  if (!I.enabled || I.failed || T != 1 || positions.size() != 1) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, I.weights, I.config,
                          kNoGather);
    return WrapDeviceLogits(d, std::move(lg), lg.t().shape[0], vocab);
  }

  HostCopyInPlace(I.h_tok, token_ids);
  HostCopyInPlace(I.h_pos, positions);
  RefreshHostAttnMeta(I.h_attn, attn_meta);
  const int cols = I.h_attn.block_table_num_cols;
  if (I.fa_cols != -1 && I.fa_cols != cols) {
    // block table width change invalidates captured memcpy geometry
    if (I.graph) {
      b.DestroyGraph(I.graph);
      I.graph = nullptr;
    }
    I.captured = false;
    I.warm_steps = 0;
  }
  I.fa_cols = cols;

  // Ensure persistent device buffers
  if (!I.hidden) {
    I.hidden.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
    I.logits.emplace(d, DType::kF32, std::vector<int64_t>{T, vocab});
    std::vector<uint16_t> one_s(static_cast<size_t>(I.layout.head_dim_sliding),
                                vt::F32ToBF16(1.f));
    std::vector<uint16_t> one_f(static_cast<size_t>(I.layout.head_dim_full),
                                vt::F32ToBF16(1.f));
    I.ones_sliding.emplace(d, DType::kBF16, std::vector<int64_t>{I.layout.head_dim_sliding},
                           one_s.data());
    I.ones_full.emplace(d, DType::kBF16, std::vector<int64_t>{I.layout.head_dim_full},
                        one_f.data());
    const int64_t max_pos_cap =
        std::max<int64_t>(I.config.max_position_embeddings > 0
                              ? I.config.max_position_embeddings
                              : 65536,
                          65536);
    int64_t gDh = -1;
    double gBase = 0.0, gPartial = 0.0;
    EnsureProportionalRopeCache(d, I.layout, I.layout.head_dim_full, max_pos_cap,
                                I.prop_cache, I.prop_cap, gDh, gBase, gPartial);
    I.ple_by_layer.emplace(d, DType::kBF16,
                           ple > 0 ? std::vector<int64_t>{L, T, ple}
                                   : std::vector<int64_t>{1, 1, 1});
    I.si = BuildStepInputs(d, I.h_pos, I.h_attn, I.config);
    I.d_tok.emplace(d, DType::kI32, std::vector<int64_t>{T});
  }

  // max_pos beyond prop cache → eager
  int64_t max_pos = 0;
  for (int32_t p : positions) max_pos = std::max<int64_t>(max_pos, p);
  if (max_pos > I.prop_cap) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, I.weights, I.config,
                          kNoGather);
    return WrapDeviceLogits(d, std::move(lg), lg.t().shape[0], vocab);
  }

  // --- Outside graph: embed + PLE + refresh SI ---
  {
    Tensor dtab = ResidentWeight(d, I.weights.embed_tokens, {vocab, H});
    d.b.Copy(d.q, I.d_tok->ptr(), I.h_tok.data(),
             static_cast<size_t>(T) * sizeof(int32_t));
    vt::Embedding(d.q, I.hidden->t(), dtab, I.d_tok->t());
    const float nsqrt = std::sqrt(static_cast<float>(H));
    const double normalizer =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
    vt::MulScalar(d.q, I.hidden->t(), I.hidden->t(), normalizer);
  }
  if (ple > 0) {
    // Full PLE precompute into ple_by_layer (same as ForwardBody) — outside graph.
    const int64_t LP = L * ple;
    const float eps = 1e-6f;
    const vt::RmsNormArgs plain{eps, false};
    DBuf ple_emb(d, DType::kBF16, {T, LP});
    {
      Tensor dtab = ResidentWeight(d, I.weights.embed_tokens_per_layer, {vocab, LP});
      vt::Embedding(d.q, ple_emb.t(), dtab, I.d_tok->t());
    }
    vt::MulScalar(d.q, ple_emb.t(), ple_emb.t(),
                  static_cast<double>(I.layout.embed_scale_ple));
    DBuf ple_proj(d, DType::kBF16, {T, LP});
    Tensor wproj = ResidentWeight(d, I.weights.per_layer_model_projection, {LP, H});
    vt::MatmulBT(d.q, ple_proj.t(), I.hidden->t(), wproj);
    vt::MulScalar(d.q, ple_proj.t(), ple_proj.t(), I.layout.proj_scale);
    Tensor proj2 = Reshape(ple_proj.t(), {T * L, ple});
    Tensor wpn = ResidentWeight(d, I.weights.per_layer_projection_norm, {ple});
    vt::RmsNorm(d.q, proj2, proj2, wpn, plain);
    DBuf ple_input(d, DType::kBF16, {T, L, ple});
    vt::Add(d.q, ple_input.t(), Reshape(ple_proj.t(), {T, L, ple}),
            Reshape(ple_emb.t(), {T, L, ple}));
    vt::MulScalar(d.q, ple_input.t(), ple_input.t(), I.layout.input_scale);
    const size_t row = static_cast<size_t>(ple) * sizeof(uint16_t);
    const auto* src = static_cast<const char*>(ple_input.ptr());
    auto* dst = static_cast<char*>(I.ple_by_layer->ptr());
    for (int64_t ti = 0; ti < T; ++ti) {
      for (int64_t li = 0; li < L; ++li) {
        const size_t s_off =
            (static_cast<size_t>(ti) * static_cast<size_t>(L) + static_cast<size_t>(li)) * row;
        const size_t d_off =
            (static_cast<size_t>(li) * static_cast<size_t>(T) + static_cast<size_t>(ti)) * row;
        CopyRow(d, dst + d_off, src + s_off, row);
      }
    }
  }
  RefreshStepInputsInPlace(d, *I.si, I.h_pos, I.h_attn, I.config);

  auto run_layers_logits = [&]() {
    ForwardGemma4Layers(d, *I.hidden, I.layout, I.weights, I.config, I.h_attn, attn_kv, *I.si,
                        *I.ones_sliding, *I.ones_full, *I.prop_cache, *I.ple_by_layer, T, H,
                        Inter, L, ple);
    (void)ForwardGemma4Logits(d, *I.hidden, I.weights, I.layout, I.config, kNoGather, T, H,
                              vocab, &*I.logits);
  };

  auto view_logits = [&]() -> ForwardLogits {
    ForwardLogits fl;
    fl.rows = T;
    fl.vocab = vocab;
    fl.device_tensor = I.logits->t();
    // Non-owning: logits buffer owned by graph impl for process lifetime of Step calls.
    fl.device_storage = std::shared_ptr<void>(I.logits->ptr(), [](void*) {});
    return fl;
  };

  if (I.captured && I.graph) {
    try {
      b.ReplayGraph(I.queue, I.graph);
      ++I.replays;
      return view_logits();
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "[Gemma4DecodeGraph] replay failed: %s — disabling\n", ex.what());
      I.failed = true;
      I.captured = false;
    }
  }

  if (I.warm_steps >= I.kWarmBeforeCapture && !I.captured && !I.failed) {
    try {
      b.BeginCapture(I.queue);
      run_layers_logits();
      I.graph = b.EndCaptureGraph(I.queue);
      I.captured = true;
      std::fprintf(stderr, "[Gemma4DecodeGraph] captured S=1 pure-decode graph\n");
      b.ReplayGraph(I.queue, I.graph);
      I.replays = 1;
      return view_logits();
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "[Gemma4DecodeGraph] capture failed: %s — eager forever\n",
                   ex.what());
      I.failed = true;
      if (I.graph) {
        try {
          b.DestroyGraph(I.graph);
        } catch (...) {
        }
        I.graph = nullptr;
      }
      // fall through to eager result of a fresh run
      run_layers_logits();
      return view_logits();
    }
  }

  // Eager decode steps: warm TLS / weights before capture.
  run_layers_logits();
  ++I.warm_steps;
  return view_logits();
}

std::optional<ForwardLogits> Gemma4DecodeGraphForward(
    std::unique_ptr<Gemma4DecodeGraph>& graph, const Gemma4Weights& weights,
    const ModelForwardInput& input) {
  if (!Gemma4DecodeGraphEnabled() || !input.pure_decode ||
      !platforms::GetPlatform(input.queue.device.type).support_static_graph_mode()) {
    return std::nullopt;
  }
  if (input.token_ids.size() != 1) return std::nullopt;
  if (!graph) graph = std::make_unique<Gemma4DecodeGraph>(weights, input.config, input.queue);
  return graph->Step(input.token_ids, input.positions, input.attn_meta, input.attn_kv);
}

}  // namespace vllm

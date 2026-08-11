// Ported from: vllm/v1/engine/input_processor.py @ e24d1b24
// See include/vllm/v1/engine/input_processor.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/input_processor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::v1 {
namespace {

// Wall-clock seconds since the epoch, mirroring upstream time.time().
double NowSeconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace

InputProcessor::InputProcessor(const tok::Tokenizer& tokenizer,
                               const HfConfig& config)
    : tokenizer_(tokenizer) {
  // model_config.max_model_len. HfConfig has no dedicated max_model_len (rope
  // scaling etc. are deferred), so max_position_embeddings stands in at T0.
  max_model_len_ = config.max_position_embeddings;

  // renderer.get_eos_token_id() + generation_config["eos_token_id"]: derive the
  // primary eos id and the secondary eos-id list from config.json's
  // "eos_token_id" (int OR list), falling back to the tokenizer's own eos.
  bool found = false;
  if (config.raw.is_object()) {
    auto it = config.raw.find("eos_token_id");
    if (it != config.raw.end() && !it->is_null()) {
      if (it->is_number_integer()) {
        const auto id = it->get<int32_t>();
        eos_token_id_ = id;
        generation_config_eos_ids_.push_back(id);
        found = true;
      } else if (it->is_array()) {
        for (const auto& e : *it) {
          if (e.is_number_integer()) {
            generation_config_eos_ids_.push_back(e.get<int32_t>());
          }
        }
        if (!generation_config_eos_ids_.empty()) {
          eos_token_id_ = generation_config_eos_ids_.front();
          found = true;
        }
      }
    }
  }
  if (!found && tokenizer_.EosId() >= 0) {
    eos_token_id_ = tokenizer_.EosId();
    generation_config_eos_ids_.push_back(tokenizer_.EosId());
  }

  // Upstream draws update_from_generation_config's id set from
  // generation_config.json (ModelConfig.try_get_generation_config), which is
  // usually a SUPERSET of config.json's: Gemma-4-26B ships config.json
  // [1, 106] against generation_config.json [1, 106, 50], and a port reading
  // only config.json never stops on 50. Union them, appending only ids not
  // already present so the PRIMARY eos id resolved above -- which is
  // generation_config_eos_ids_.front() on the list path -- keeps its position.
  for (const int32_t id : config.generation_config_eos_ids) {
    if (std::find(generation_config_eos_ids_.begin(),
                  generation_config_eos_ids_.end(),
                  id) == generation_config_eos_ids_.end()) {
      generation_config_eos_ids_.push_back(id);
    }
  }
}

void InputProcessor::ValidateParams(SamplingParams& params) const {
  // Upstream _validate_params calls params.verify(model_config, ...) after
  // __post_init__ already ran at construction. Our SamplingParams deferred
  // __post_init__ to this constructing unit (M1.1), so PostInit() both
  // normalizes the params AND runs Verify() — closing that carry.
  params.PostInit();
}

void InputProcessor::UpdateFromGenerationConfig(SamplingParams& params) const {
  // sampling_params.py:627-655. Sets eos_token_id, adds eos id(s) to
  // all_stop_token_ids (so the MinTokens processor masks them), and merges the
  // SECONDARY eos ids into stop_token_ids.
  if (!params.ignore_eos) {
    params.eos_token_id = eos_token_id_;
  }

  // The primary eos id feeds min_tokens masking (sampling_params.py:638-641).
  if (eos_token_id_.has_value()) {
    params.all_stop_token_ids.insert(*eos_token_id_);
  }

  if (generation_config_eos_ids_.empty()) {
    return;
  }
  std::set<int32_t> eos_ids(generation_config_eos_ids_.begin(),
                            generation_config_eos_ids_.end());
  // The primary eos id is handled separately for stopping; don't duplicate it.
  if (eos_token_id_.has_value()) {
    eos_ids.erase(*eos_token_id_);
  }
  if (!eos_ids.empty()) {
    // The full eos set contributes to min_tokens masking regardless of
    // ignore_eos (sampling_params.py:653).
    params.all_stop_token_ids.insert(eos_ids.begin(), eos_ids.end());
    if (!params.ignore_eos) {
      for (int32_t id : params.stop_token_ids) {
        eos_ids.insert(id);
      }
      params.stop_token_ids.assign(eos_ids.begin(), eos_ids.end());
    }
  }
}

void InputProcessor::UpdateFromTokenizer(SamplingParams& params) const {
  // sampling_params.py:659-698 (update_from_tokenizer): tokenize bad_words into
  // per-word token-id n-grams the sampler masks. Each word is encoded both
  // without and with a leading space (add_prefix_space) to catch it at the start
  // of and mid-text; the prefix-space variant is kept only when it produces a
  // genuinely different same-length token sequence.
  if (params.bad_words.empty()) {
    return;
  }
  std::vector<std::vector<int32_t>> bad_ids;
  for (const std::string& bad_word : params.bad_words) {
    // lstrip the word (add_prefix_space controls the leading space instead).
    size_t start = bad_word.find_first_not_of(" \t\n\r\f\v");
    const std::string stripped =
        start == std::string::npos ? std::string() : bad_word.substr(start);
    for (int variant = 0; variant < 2; ++variant) {
      const bool add_prefix_space = variant == 1;
      const std::string prompt =
          (add_prefix_space ? std::string(" ") : std::string()) + stripped;
      // Encode() adds no special tokens (== add_special_tokens=False).
      std::vector<int32_t> ids = tokenizer_.Encode(prompt);
      if (ids.empty()) continue;
      if (!add_prefix_space) {
        bad_ids.push_back(std::move(ids));
      } else if (!bad_ids.empty()) {
        const std::vector<int32_t>& last = bad_ids.back();
        if (!last.empty() && ids[0] != last[0] && ids.size() == last.size()) {
          bad_ids.push_back(std::move(ids));
        }
      }
    }
  }

  // Vocabulary-range check (sampling_params.py:683-698).
  const int32_t max_token_id = tokenizer_.VocabSize() - 1;
  for (const std::vector<int32_t>& ngram : bad_ids) {
    for (int32_t token_id : ngram) {
      if (token_id < 0 || token_id > max_token_id) {
        throw std::runtime_error(
            "The model vocabulary size is " +
            std::to_string(max_token_id + 1) +
            ", but a token specified as bad is out of range. All token id "
            "values should satisfy 0 <= token_id <= " +
            std::to_string(max_token_id) + ".");
      }
    }
  }

  params.bad_words_token_ids = std::move(bad_ids);
}

EngineCoreRequest InputProcessor::process_inputs(
    const std::string& request_id, const std::string& prompt,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // _validate_params: run PostInit()/Verify() on the (cloned) params.
  ValidateParams(params);

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  // input_preprocessor.preprocess -> tokenize (text path only). vLLM tokenizes
  // prompts with HF's default `add_special_tokens=True`, so the tokenizer's
  // post_processor template is APPLIED here. This is a no-op for every Qwen
  // tokenizer (their ByteLevel post_processor declares no bos/eos) and supplies
  // the prepended `</s>` that OPT's TemplateProcessing declares.
  std::vector<int32_t> prompt_token_ids =
      tokenizer_.EncodeWithSpecialTokens(prompt);

  ApplyContextBudget(request_id, prompt_token_ids.size(), params);

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  return request;
}

EngineCoreRequest InputProcessor::process_inputs_tokens(
    const std::string& request_id, std::vector<int32_t> prompt_token_ids,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // Identical to process_inputs EXCEPT the prompt is already tokenized (vLLM
  // TokensPrompt): no tokenizer_.EncodeWithSpecialTokens call, and no
  // post_processor template applied (the caller supplies the exact ids, special
  // tokens included). Every other step (validate, default max_tokens, eos/stop
  // wiring, request assembly) is byte-for-byte the string path.
  ValidateParams(params);

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  ApplyContextBudget(request_id, prompt_token_ids.size(), params);

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  return request;
}

EngineCoreRequest InputProcessor::process_inputs_mm(
    const std::string& request_id, std::vector<int32_t> prompt_token_ids,
    std::vector<multimodal::MultiModalFeatureSpec> mm_features,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // Multimodal path: the prompt is the ALREADY placeholder-EXPANDED id stream
  // (the serving-side mm processor ran and consumed the media), so like
  // process_inputs_tokens there is no tokenizer call. The ONLY delta vs the
  // tokens path is that mm_features is carried onto the EngineCoreRequest
  // (upstream input_processor.py:370-379 sets mm_features alongside
  // prompt_token_ids). Every other step is byte-for-byte identical.
  ValidateParams(params);

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  ApplyContextBudget(request_id, prompt_token_ids.size(), params);

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  request.mm_features = std::move(mm_features);
  return request;
}

void InputProcessor::ApplyContextBudget(const std::string& request_id,
                                        std::size_t prompt_len,
                                        SamplingParams& params) const {
  // Fail closed when the prompt alone exceeds the serve window. Previously the
  // engine could accept oversized Hermes SOUL+history and spin forever in WAITING
  // (idle GPU + SSE keepalives) until KV fail-fast aborted — if at all.
  if (max_model_len_ <= 0) {
    throw std::runtime_error("input_processor: max_model_len is not configured");
  }
  const int64_t plen = static_cast<int64_t>(prompt_len);
  if (plen > max_model_len_) {
    throw std::runtime_error(
        "prompt too long for this server: prompt_tokens=" +
        std::to_string(plen) + " max_model_len=" +
        std::to_string(max_model_len_) + " request_id=" + request_id +
        ". Shrink system/tools/history, raise --max-model-len/--num-blocks, "
        "or enable compression on the client.");
  }
  const int room = static_cast<int>(max_model_len_ - plen);
  if (room <= 0) {
    throw std::runtime_error(
        "no remaining context for generation: prompt_tokens=" +
        std::to_string(plen) + " max_model_len=" +
        std::to_string(max_model_len_) + " request_id=" + request_id);
  }
  // Unset max_tokens → fill remaining window (upstream input_processor.py).
  if (!params.max_tokens.has_value() || *params.max_tokens <= 0) {
    params.max_tokens = room;
    return;
  }
  // Positive client max_tokens: clamp to remaining so prompt+gen cannot exceed
  // max_model_len (stops near-ceiling hangs with finish never firing).
  if (*params.max_tokens > room) {
    std::fprintf(stderr,
                 "INFO input_processor: clamp max_tokens %d -> %d "
                 "(prompt_tokens=%lld max_model_len=%lld id=%s)\n",
                 *params.max_tokens, room, static_cast<long long>(plen),
                 static_cast<long long>(max_model_len_), request_id.c_str());
    std::fflush(stderr);
    params.max_tokens = room;
  }
}

}  // namespace vllm::v1

# TOOL-CALL-HISTORY-ARGS — normalize OpenAI tool arguments before rendering

Issue: [#526](https://github.com/mudler/vllm.cpp/issues/526)

Row: `TOOL-CALL-HISTORY-ARGS` (feature matrix §7, Tool calling)

Status: **GATING (CPU RED/GREEN + independent mutation review complete; live Gemma gate pending)**

## Problem

OpenAI Chat Completions represents
`assistant.tool_calls[].function.arguments` as a JSON-encoded string. vllm.cpp
keeps that wire representation in `FunctionCall::arguments` and currently copies
it verbatim into the Jinja `messages` value in
`src/vllm/entrypoints/chat_template.cpp::BuildMessages`.

That diverges from the pinned vLLM renderer input contract. Pinned vLLM
`vllm/entrypoints/chat_utils.py:1915-1951` post-processes assistant tool history
before rendering:

- a non-empty string is decoded with `json.loads`;
- an empty string, absent value, or JSON `null` becomes `{}`;
- already-structured dictionary/list values remain structured;
- malformed JSON raises instead of entering the model prompt.

Gemma 4 makes the divergence observable. Its native history syntax is:

```text
<|tool_call>call:terminal{command:<|"|>date<|"|>}<tool_call|>
```

The current vllm.cpp adapter instead renders:

```text
<|tool_call>call:terminal{"command":"date"}<tool_call|>
```

A model can imitate that non-native form on the next turn. The Gemma parser then
correctly treats the quote characters as part of the key/value and returns valid
JSON with the wrong schema, for example:

```json
{"\"command\"":"\"ls\""}
```

Retries can recursively escape the malformed history. The client executor is not
the repair surface: the malformed syntax originates in the provider's rendered
prompt.

## Evidence

- Repeated live failure:
  `/home/don/llms/logs/agentic_tool_loop_20260811-212603.json` and
  `agentic_tool_loop_20260811-202653.json`. Turns 1-2 are valid; the first
  malformed call appears after completed tool history.
- Live render after the 2026-08-12 incident recovery logged:
  `<|tool_call>call:terminal{"command":"printf FIRST_OK"}<tool_call|>`.
- The isolated live probe is
  `/home/don/llms/logs/issue-526/repro_issue_526.py`. Its short second turn can
  still recover, which proves this is an emission-quality poison rather than a
  deterministic parser failure; the rendered history remains wrong.
- Local source:
  `src/vllm/entrypoints/chat_template.cpp:55-70` stores the OpenAI string.
- Pinned oracle source:
  `/home/don/llms/vllm-upstream/vllm/entrypoints/chat_utils.py:1915-1951`
  (`_postprocess_messages`).
- Google Gemma 4 canonical template (2026-07-09) requires a mapping and raises
  on string arguments with the explicit instruction to deserialize before
  rendering. vLLM main's `examples/tool_chat_template_gemma4.jinja` mirrors it.

## Scope

In scope:

1. Mirror pinned vLLM's assistant tool-history post-processing at the
   protocol-to-renderer boundary.
2. Preserve the OpenAI wire type (`FunctionCall::arguments` remains a string) so
   response serialization and parser output remain API-compatible.
3. Build a separate render-context JSON value for historical arguments:
   - empty string / JSON `null` -> `{}`;
   - valid JSON object/list/scalar -> decoded JSON value, matching pinned vLLM;
   - malformed JSON -> `ChatTemplateError` before generation.
4. Add exact CPU regression coverage for Gemma native DSL history, empty/null,
   nested values, malformed JSON, and a non-Gemma template.
5. Add a live multi-turn Gemma gate only after CPU RED/GREEN and a coordinated
   `:8010` handoff.

Out of scope:

- Any Hermes Agent source, client executor, or tool-schema repair.
- Repairing quote-wrapped keys after model emission.
- Changing parser extraction, tool schemas, tool names, or OpenAI response wire
  format.
- Updating model sidecar templates as the primary fix.
- GPU kernels, scheduler behavior, or the separate stale-request wedge incident.

## Design

Keep `FunctionCall::arguments` as the OpenAI JSON string. In
`BuildMessages`, parse that string into `nlohmann::ordered_json` only for the
Jinja render context.

Suggested helper:

```cpp
nlohmann::ordered_json DecodeHistoricalToolArguments(
    const std::string& encoded);
```

Contract:

- `encoded.empty()` returns `ordered_json::object()`.
- Parse with exceptions enabled so malformed history fails closed.
- Parsed `null` returns an empty object.
- Any other valid JSON value is returned unchanged.
- A parse failure is wrapped in `ChatTemplateError` with the tool-call index or
  function name, but never includes arbitrary argument contents in server logs.

`BuildMessages` assigns this decoded value to
`function["arguments"]`. `to_json(FunctionCall)` remains untouched and continues
to emit the OpenAI string.

This is not template-capability inference. Pinned vLLM normalizes the
conversation before every HF chat-template render, and templates that require a
string are expected to serialize the structured value themselves. Mirroring the
oracle avoids a model-specific heuristic.

## Tests (RED before implementation)

Add a committed Gemma 4 template fixture from the canonical syntax, trimmed only
if the test keeps the exact tool-history branch and records that adaptation.
Extend `tests/vllm/entrypoints/test_chat_template.cpp`:

1. **Gemma object history:** OpenAI `{"command":"date"}` renders exactly
   `call:terminal{command:<|"|>date<|"|>}` and does not contain
   `call:terminal{"command"`.
2. **Nested values:** strings, booleans, integers, arrays, nested objects and
   null render through `format_argument` with canonical Gemma delimiters/types.
3. **Empty arguments:** `""` and `"null"` both render `{}`.
4. **Malformed JSON:** `{"command":` throws `ChatTemplateError`; no prompt is
   returned.
5. **Wire inertness:** serializing the original `FunctionCall` still yields its
   JSON-encoded string.
6. **Non-Gemma compatibility:** the existing real Qwen3.5 tool-history fixture
   remains green with the normalized mapping and emits the same semantic tool
   name/arguments/result.
7. **No-tool inertness:** plain messages render byte-identically.

The focused target must demonstrate the right-reason RED on case 1 before the
production change, then GREEN after it. A mutation replacing the decoded value
with the original string must re-fail case 1.

## Gates

CPU/spec gate:

```bash
cmake --build <cpu-build> --target test_chat_template -j$(nproc)
ctest --test-dir <cpu-build> -R '^test_chat_template$' --output-on-failure
scripts/check-agent-record.py
python3 tests/scripts/test_doc_checkpoint.py
```

Also run the existing OpenAI protocol/serving tests that cover
`ChatMessage::from_json`, non-streaming chat, streaming chat, and C ABI chat.
Run `git diff --check` and a clean `-Werror` build of the touched target.

Live gate (after coordinated exclusive handoff):

1. Start the existing Gemma FP8 recipe with `--tool-call-parser gemma4`.
2. Fresh required terminal call returns `{"command":"printf FIRST_OK"}`.
3. Append that exact OpenAI assistant call plus a tool result; the next request
   returns `{"command":"printf SECOND_OK"}`.
4. Inspect the logged rendered prompt: historical call must be canonical Gemma
   DSL without JSON-quoted keys.
5. Run the 3-tool `agentic_tool_loop_probe.py`; all expected tools execute, no
   quote-wrapped key appears, and the terminal verification succeeds.
6. Paris / arithmetic `4` / short decode quality gates pass.
7. Restore the production recipe and verify `/health`, `/v1/models`, scheduler
   running/waiting zero, and one fresh tool call.

## Risks and stop conditions

- **Risk: templates that expected raw strings.** The pinned vLLM contract already
  supplies decoded values. The existing Qwen3.5 fixture is the regression gate;
  if it genuinely requires a string, stop and compare against the same template
  under pinned vLLM rather than adding a local heuristic.
- **Risk: malformed client history.** Fail closed before generation. Do not
  preserve malformed text in the model prompt and do not silently coerce it.
- **Risk: broad protocol refactor.** Stop if the fix requires changing
  `FunctionCall::arguments` wire type; use render-context normalization only.
- **Risk: false live pass.** A short model can recover despite poisoned history.
  The binding proof is exact rendered prompt plus the multi-turn execution gate.
- **Stop:** no implementation before this spec-only checkpoint is committed.
- **Stop:** no live daemon bounce without the single `:8010` owner handoff.
- **Stop:** no Hermes source changes under any circumstance.

## Work breakdown

1. Commit this issue/spec/roadmap checkpoint only.
2. Add exact RED tests and canonical fixture.
3. Implement the minimum renderer-boundary normalization.
4. Run focused and protocol/serving CPU gates plus mutation review.
5. Obtain fresh independent review on the immutable head.
6. Coordinate the live Gemma gate and restore `:8010`.

## Now

Issue #526 is open. The renderer-boundary normalization and canonical Gemma 4
fixture are implemented on the row branch. The focused test failed before the
production change because completed history rendered JSON-quoted keys
(`test-red-live-fixture.log`), then passed after the change
(`test-green-expanded-2.log`). The expanded CPU protocol/serving set also passes.
The OpenAI wire representation remains a string and malformed JSON fails closed
without echoing argument contents. Independent review at immutable `d2feeb8c`
passed the implementation and proved the original-string mutation fails. Its
role-predicate mutation exposed one test gap; a new non-assistant malformed-JSON
regression now fails that mutation exactly, and the Qwen fixture locks the decoded
`city=Rome` parameter shape. Next: coordinate the exclusive live Gemma multi-turn
gate and restore `:8010`.

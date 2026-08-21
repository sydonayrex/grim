# WI-NOOA-Derived-Plan

Derivation from `old/repos/labs-OO-Agents-main/` (NVIDIA NOOA, Apache 2.0). Only items that are real gaps in grim and not already covered better by existing grim code.

---

## 1. Session event ring buffer (`session_events.rs`)

**Source:** `src/nooa/runtime/event_manager.py` — `EventManager.add()`, `collapse()`, string-tag access, middleware `on()`/`intercept()`.

**Gap:** `grim-core/src/session.rs:1-336` has zero event/trace symbols. No per-session trace store.

**Implement:**
- New file `crates/grim-core/src/session_events.rs`
- `SessionEvent` enum: `Prompt{tokens: Vec<u32>, token_ids: Vec<u32>}`, `Completion{token_id: u32, logprob: Option<f32>}`, `Error{title: String, detail: String}`, `ToolCall{index: u32, name: String, args: String}`, `ToolResult{index: u32, result: String}`
- `SessionEvents` struct: `Vec<SessionEvent>`, `current_tag: u64`, `collapse_threshold: u64` (events before shed)
- Methods: `push(&mut self, event)`, `get(&self, tag) -> Option<&SessionEvent>`, `tags(&self) -> Vec<u64>`, `collapse_oldest(&mut self, n: usize, summary: &str) -> Vec<SessionEvent>` (sheds oldest n events, returns them for logging)
- Wire into `SessionT` as optional: add `fn events(&self) -> Option<&SessionEvents>` / `fn events_mut(&mut self) -> Option<&mut SessionEvents>` to `SessionT` with default `None`
- `Inner` gets `Option<SessionEvents>` field, `SessionT` impls return `self.events.as_ref()` / `self.events.as_mut()`
- Consumers: `grim-garage` reads events for trace UI; `grim-engine` pushes `Completion` per token in `streaming_forward.rs`

**Not included:** full middleware chain, `on()` observer pattern, `intercept()` — those are LLM-orchestration features grim doesn't need. grim's events are append-only trace, not a callable pipeline.

---

## 2. `ArchHyperparameters::merge_with` cascade

**Source:** `src/nooa/agent.py:337-369` `_resolve_truncation` (default → class → instance, merge semantics); `src/nooa/config/strategy_config.py:126` `CodeActConfig.merge_with` (frozen Pydantic, `model_copy(update={k for k in other.model_fields_set})`); `src/nooa/agent.py:50-61` `_InheritSentinel` distinguishes "omitted" from "None".

**Gap:** `grim-core/src/hyperparams.rs` — `ArchHyperparameters` is a flat struct, single `Default` impl, single `extract()` path, no cascade, no override field-level semantics.

**Implement:**
- Add `HyperparameterOverrides` struct (all fields `Option<T>`, `None` = inherit):
  ```rust
  pub struct HyperparameterOverrides {
      pub max_seq_len: Option<usize>,
      pub rope_theta: Option<f32>,
      pub rms_norm_eps: Option<f32>,
      pub num_kv_heads: Option<usize>,
      pub head_dim: Option<usize>,
      pub expert_used_count: Option<usize>,
      pub routed_scaling_factor: Option<f32>,
      pub norm_topk_prob: Option<bool>,
      pub full_attention_interval: Option<usize>,
  }
  ```
  Only fields that make sense to override at runtime. Architecture, vocab_size, hidden_size, num_layers are model-fixed — not in overrides.
- Add `impl ArchHyperparameters { pub fn merge_with(&self, overrides: &HyperparameterOverrides) -> Self { ... } }` — clones self, applies each `Some(v)` field.
- Add `impl Default for HyperparameterOverrides { fn default() -> Self { Self { field: None, ... } } }`
- Cascade order (documented, not new code path): model-default (from GGUF extraction, `extract()`) → user overrides (`HyperparameterOverrides`) → runtime overrides (future, not yet needed). Today: `extract()` → `merge_with(user_overrides)`.
- Audit existing `Option<usize>` fields (ssm_d_state, expert_count, etc.): these are "not present for this arch" semantics, not "inherit" semantics. Document the distinction. Don't change them.

**Not included:** `_InheritSentinel` pattern — grim uses `Option<T>` natively, Rust doesn't need a sentinel to distinguish "omitted" from "None" since `Option::None` already means "not set" in the overrides struct.

---

## 3. Request rejection error format in `grim-server/src/lib.rs`

**Source:** `src/nooa/strategies/generated_code.py:636-797` `ArgumentValidator.validate()` — arity check + Pydantic type check + formatted error: `"Invalid call to method():\n  Argument 'x' has wrong type: expected T, got U\n  Value: ...\n  Signature: method(...)"`.

**Gap:** `grim-server/src/lib.rs:1285-1338` enforces `context_length()` but error body is generic `request_error(InvalidRequest, "...")` with extra fields tacked on. Error shape is functional but not structured like the source pattern.

**Implement:**
- In the existing context-length enforcement block (`lib.rs:1305-1328`), restructure error body to match the "parameter vs constraint" format:
  ```json
  {
    "error": {
      "code": "context_length_exceeded",
      "message": "request exceeds model context window",
      "parameter": "prompt + max_tokens",
      "requested": <total_requested>,
      "constraint": <context_limit>,
      "prompt_tokens": <n>,
      "max_tokens": <n>
    }
  }
  ```
- The existing code already sets `prompt_tokens`, `max_tokens`, `total_requested`, `context_length` — reorder into `parameter`/`requested`/`constraint` framing.
- Add similar structured error for the `total_requested > 1_000_000` warning path (currently only `eprintln!`, no client-visible error): when 0-context model + obviously excessive request, return same shape with `code: "request_too_large"`.
- No new validation logic — grim-server:1285-1338 already computes the right values. Just restructure the error body.

**Not included:** arity validation, type validation — those are for LLM-generated function calls. grim's HTTP API has its own schema (OpenAI-compatible), validated by serde at the boundary. Different concern.

---

## Files created/modified

| File | Action |
|---|---|
| `crates/grim-core/src/session_events.rs` | New — SessionEvents ring buffer |
| `crates/grim-core/src/session.rs` | Modify — add `events()` to `SessionT`, `events` field to `Inner` |
| `crates/grim-core/src/hyperparams.rs` | Modify — add `HyperparameterOverrides`, `merge_with()` |
| `crates/grim-server/src/lib.rs` | Modify — restructure context-length error body |

## Out of scope (confirmed redundant)

- OutputValidator trait — `grim-constrain/src/json_fsm.rs` + `schema.rs` already do per-token logit masking during generation, which is strictly more capable than post-hoc validation.
- InferenceStrategy trait — `grim-speculative`, `grim-disagg`, `grim-scheduler` are concrete subsystems; a generic trait would be a thin wrapper.
- Sandbox require semantics — `grim-backend-rocm/src/device/helpers.rs:87-98` already fails closed on hiprtcCompileProgram failure, returns compile log.
- Full NOOA middleware chain, intercept(), on() observer pattern — grim doesn't need LLM-call wrapping hooks.

## Verification

```bash
cargo check -p grim-core
cargo check -p grim-server
```

No runtime behavior change expected — session_events is opt-in (SessionT returns None by default), merge_with is additive on existing extract() path, error format change is cosmetic on existing enforcement.

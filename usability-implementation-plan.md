# Usability Failure Remediation Plan

Source: `usability-test.md` and the notional results in `u-test.md`  
Date: 2026-08-13  
Status: implementation plan — written for step-by-step execution

## How to use this document

Follow the work items in order. Do not start WI-U3 until WI-U1 and WI-U2 pass their gates.
Do not mark a work item complete because a function compiles. Mark it complete only when its
listed command passes and its failure-path test passes.

For every code change:

1. Read the named files before editing them.
2. Make the smallest change that satisfies the acceptance criteria.
3. Run `cargo fmt --all` on changed Rust files.
4. Run the work item's exact test command.
5. If a test fails, inspect the function named in the failure before changing the test.
6. Run `git diff --check` and `cargo test -p <changed-crate> --lib`.

Do not invent a new server, router, status type, or metrics system. The current server already
owns these behaviors in `crates/grim-server/src/lib.rs`. Extend that code first.

### Current baseline after the first P0 pass

Already present:

- `/health`, `/healthz`, `/status`, `/v1/status`, `/metrics` routes.
- `/v1/models`, `/v1/models/load`, `/v1/models/unload` routes.
- Lazy model resolution through `grim_core::catalog::resolve_model_preferring_grim`.
- Backend field in the status response and `GRIM_BACKEND` override.
- `cargo test -p grim-server --lib`: 43 tests passed.

Still open and required below:

- One hermetic model-resolution/completion acceptance test.
- One structured unknown-model error test.
- Real queue metrics instead of omitted/placeholder queue state.
- Removal of placeholder timing values.
- Metrics bind-policy enforcement and dashboard smoke coverage.

## What the workflow review found

The usability findings are not all the same kind of problem. The code review separates them into
three categories:

1. **Actionable wiring/verification gaps.** The repository already contains the relevant crates,
   handlers, routes, or configuration, but the user path is not connected, discoverable, or covered
   by an end-to-end test. These should be implemented.
2. **Explicit capability gaps.** The code and documentation explicitly say the capability is a
   stub or unimplemented. These require implementation work; changing wording alone would be
   misleading.
3. **Environment-dependent validation.** Vulkan, Metal, CUDA, containers, and the full workspace
   CI need representative hardware or CI jobs. A local ROCm result cannot be generalized to those
   backends.

The “workflow” wording is therefore actionable when it names a missing transition between existing
components. It is only a complaint when it asks the current checkout to satisfy a capability that
the source explicitly marks as out of scope or unimplemented without first adding that capability.

## Priority 0 — establish one honest vertical slice

### WI-U1: Serve one model end to end

**Findings:** Persona 1.1, 2.1–2.2, 7.1, 8.1, 15.1. `docs/cli.md` documents `pull → serve →
POST /v1/chat/completions`; `grim-server` and the CLI contain the pieces, but the usability run
had no executable acceptance path proving model resolution, backend reporting, schema response,
and first completion together.

**Implementation:**

- Add a hermetic server integration fixture using a tiny deterministic model or mock engine.
- Exercise `grim pull`/catalog resolution, `grim serve`, `/v1/models`, `/v1/chat/completions`, and
  `/v1/status` in one test.
- Return the resolved model, active backend, load state, and first-token/total timing in the
  status/log path.
- Make model-not-loaded, unknown-model, and unsupported-backend errors structured and actionable.
- Add a five-minute quick-start command block to `docs/howto/run-inference.md` and link it from
  the CLI `--help` output.

**Acceptance:** one test starts the server, obtains a completion in the fixture, validates the
OpenAI response shape, confirms the model/backend, and proves a clean failure for an unknown model.

**Owners:** `grim-cli`, `grim-server`, `grim-core`, `docs/howto/run-inference.md`.

### WI-U2: Make backend and resource state visible

**Findings:** Persona 1.3, 2.3, 9.1, 11.2, 16.2, 18.2. `docs/observability.md` documents
`/metrics`; `grim-garage` has `/api/backends`, `/api/rocm/devices`, training metrics, and a health
route. This is actionable wiring: the surfaces exist, but they are not one coherent serving
status contract and the notional run did not verify them.

**Implementation:**

- Define a versioned status payload containing selected backend, device/arch, model, scheduler
  active/waiting/admitted counts, KV allocation by tier, and token timing.
- Expose the same snapshot through `/v1/status`, `/metrics`, and the dashboard API; do not make
  users infer state from logs.
- Add `GET /healthz` and readiness semantics to the serving layer, not only the garage router.
- Add a loopback-by-default bind test and an explicit opt-in for non-loopback metrics exposure.
- Add a dashboard smoke test for backend, VRAM, queue state, and training status.

**Acceptance:** a running fixture server reports backend/model/queue/KV state through HTTP; a
dashboard API test reads the same values; a bind test rejects accidental public metrics exposure.

**Owners:** `grim-server`, `grim-scheduler`, `grim-memory`, `grim-garage`, `grim-cli`.

## Priority 1 — complete the existing API workflows

### WI-U3: OpenAI/Ollama compatibility contract

**Findings:** Persona 7.2–7.3 and Persona 8. The integration docs claim SSE and tool calls, and
the server route inventory includes them. This is actionable, not a documentation complaint.

**Implementation:**

- Add black-box tests for non-streaming and streaming `/v1/chat/completions`, including SSE
  chunk framing and `[DONE]`.
- Add tool-call fixtures covering assistant `tool_calls`, client tool result messages, malformed
  tools, and unknown tool names.
- Add black-box tests for Ollama `/api/chat` streaming/non-streaming and model listing.
- Normalize errors to the documented OpenAI/Ollama shapes and document the exact model-name mapping.

**Acceptance:** an OpenAI SDK-style fixture and an Ollama-style fixture run without adapter code
changes; malformed requests return stable 4xx JSON rather than generic 500 responses.

**Owners:** `grim-server`, `docs/integrations.md`, API integration tests.

### WI-U4: Adapter lifecycle and fine-tuning path

**Findings:** Personas 1.4, 2.4, and 4. The repository has CLI training, garage training jobs,
   bolt-on routes, sidecar loading, and merge logic. This is partially wired and needs a single
   supported lifecycle rather than more adapter names in the UI.

**Implementation:**

- Define one adapter artifact contract: training output, checksum, base-model identity, rank,
  target modules, and supported runtime backend.
- Add a server API to list/load/unload adapters and select an adapter per request, with atomic
  replacement and an in-flight request lifetime guard.
- Connect CLI training output to the server/garage loader and reject base-model/checksum mismatch
  before activation.
- Expose loss, step, checkpoint, adapter checksum, and active-adapter state through status/SSE.
- Label unsupported adapter methods explicitly; do not present every enum as an implemented
  training mode.

**Acceptance:** train or load one LoRA fixture, serve it, switch between two adapters without a
   restart, observe the selected adapter in the response metadata/status, and recover cleanly from
   a corrupt or incompatible sidecar.

**Owners:** `grim-cli`, `grim-garage`, `grim-server`, `grim-core`, `grim-format`.

## Priority 1 — turn explicit capability gaps into scoped work

### WI-U5: Quantization workflow

**Finding:** Persona 10. `grim quantize` is explicitly a stub in `docs/cli.md` and
`crates/grim-cli/src/main.rs`; the real oxidizer pipeline exists. This is an actionable CLI
workflow gap, not merely a complaint.

**Implementation:**

- Either remove `grim quantize` from the advertised happy path or make it a real front end to
  `oxidizer calibrate → search → convert`.
- Validate input/output paths, target bits-per-weight, calibration data, and backend capability.
- Emit a machine-readable result with output path, byte size, format, and quality metrics.
- Add a smoke test that converts a tiny fixture, verifies it, loads it, and compares a reference
  perplexity/loss measurement.

**Acceptance:** `grim quantize --help` is truthful; the documented command creates a loadable
artifact and reports its fidelity.

### WI-U6: Speculative decoding and KV spill observability

**Findings:** Persona 11. `grim-speculative` exists, but `NOTES.md` records the missing hidden-state
capture prerequisite; `grim-memory` documents spill tiers. The missing acceptance path is partly
wiring and partly a real algorithmic gap.

**Implementation:**

- First add an explicit `CausalLm` hidden-state/draft contract, keeping the existing logits API
  compatible through a new method or result type.
- Connect `SpeculativeCausalLm` to that contract and emit accepted/rejected draft counts,
  acceptance rate, and fallback reason.
- Connect KV transport events to the status/metrics snapshot with GPU/RAM/NVMe bytes and spill
  thresholds.
- Add deterministic CPU tests for accept/reject/rollback and a bounded integration test for one
  spill transition.

**Acceptance:** a benchmark identifies the active speculation strategy and acceptance rate; a
long-context fixture shows a documented KV tier transition without silent fallback.

**Owners:** `grim-core`, `grim-models-*`, `grim-speculative`, `grim-memory`, `grim-kvtransport`,
`grim-server`.

### WI-U7: Multimodal capability boundaries

**Finding:** Personas 5 and 6 are explicit failures. `docs/architecture.md` and
`docs/integrations.md` say vision, audio transcription, and image generation are not implemented
or stubs. This is not a workflow complaint; it is a product capability gap.

**Implementation order:**

1. Define typed model/task capability metadata and return `501 Not Implemented` with a stable
   capability identifier for current audio/image routes.
2. Implement one vertical slice (Whisper-style audio transcription or CLIP image encoding) with
   CPU reference first, then ROCm dispatch and a golden fixture.
3. Add diffusion/image generation only after the model and scheduler contracts are defined.

**Acceptance:** unsupported routes are explicit and discoverable; the first delivered modality has
an end-to-end upload/request/output test and backend disclosure.

## Priority 2 — platform and security workflows

### WI-U8: Vulkan, Metal, and CI matrix

**Finding:** Persona 12 is an environment-dependent validation failure. The crates and docs exist,
but no local run proves the backends.

**Implementation:**

- Add backend capability smoke tests that compile and run the smallest GEMM on CPU, ROCm, Vulkan,
  and Metal where the runner exists.
- Use conditional CI labels rather than claiming all backends pass on the ROCm host.
- Make backend selection and fallback explicit in the result and logs.

**Acceptance:** each backend has either a passing runner result or a clearly marked unavailable
CI job; no silent CPU fallback when a requested backend is unavailable.

### WI-U9: Plugin isolation and provenance

**Findings:** Personas 13 and 18. `grim-plugin` and CLI loading exist; `doctor.rs` explicitly
admits plugin-grant enforcement is currently a shallow check. This is actionable security work.

**Implementation:**

- Add integration tests for WASM filesystem/network denial and explicit capability grants.
- Define a separate trust boundary for native dynamic libraries; do not describe dylib loading as
  sandboxed unless it is process-isolated.
- Make plugin load errors include manifest, ABI, checksum, and required grants.
- Add model checksum/config provenance to `grim pull`, `status`, and `verify` output.

**Acceptance:** a denied WASM file/network access is observable and tested; native plugins run in
an isolated helper process or are clearly marked trusted-only; model verification emits a stable
checksum/config trace.

## Priority 2 — maintainer and deployment completion

### WI-U10: Full verification and deployment smoke

**Findings:** Personas 16 and 17. Targeted ROCm tests are green, but the usability protocol asks
for full workspace test/clippy/mutation, OCI health, and dashboard smoke.

**Implementation:**

- Add CI jobs for `cargo test --workspace`, clippy with warnings denied, ROCm targeted tests, and
  backend-specific jobs.
- Add a minimal OCI image smoke test: start server, mount a fixture model, call health/status,
  and scrape metrics.
- Add garage browser/API smoke coverage for model listing, training start/status/cancel, and
  adapter attach.
- Keep `u-test.md` as a release-gate report generated from these commands, not a claim of a human
  session.

**Acceptance:** CI publishes a matrix with explicit skipped/unavailable reasons; the container and
dashboard smoke tests complete against a deterministic fixture.

## Documentation corrections required with implementation

- Change `usability-test.md` from “feature-complete Grim” to a capability-matrix protocol until
  WI-U1–U10 land.
- Mark audio, image generation, multimodal embeddings, gRPC, and `grim quantize` as unavailable or
  experimental wherever they are currently described as supported.
- Add a “verified on gfx1036 iGPU” evidence note only to tests that actually pass; keep hardware
  claims scoped to the tested path.
- Replace generic “workflow” complaints in future reports with a concrete transition:
  command/request, expected state, observed state, owning crate, and acceptance test.

## Execution sequence

1. WI-U1 and WI-U2: serving vertical slice, status, health, metrics, backend disclosure.
2. WI-U3 and WI-U4: API compatibility and adapter lifecycle.
3. WI-U5 and WI-U6: quantization command and speculative/KV observability.
4. WI-U7: one multimodal vertical slice, then additional modalities.
5. WI-U8–U10: platform matrix, plugin security, deployment, and release verification.

Every work item must add a focused unit test plus one boundary/integration test. A work item is
complete only when the corresponding persona task can be rerun from a documented command and its
result is recorded as PASS, not merely when an internal function exists.

## Detailed execution specification for P0

This section is intentionally repetitive. It removes decisions that an implementer should not
have to make while executing the plan.

### P0-A — Hermetic serve/completion acceptance test

**Goal:** prove the documented path without downloading a real model or requiring a GPU.

**Files to inspect first:**

- `crates/grim-server/src/lib.rs`: `AppState`, `build_router`, `chat_completions`,
  `load_model_for_server`, and the existing `#[cfg(test)]` module.
- `crates/grim-core/src/catalog.rs`: model-name/path resolution.
- Existing `grim-server` test helpers that construct `Engine` and `Llama::random`.

**Steps:**

1. Add a test helper called `test_app_with_model(model_name: &str)` in the server test module.
2. Build the same small random model used by the existing server end-to-end tests.
3. Register it under `model_name` in an `Engine`.
4. Construct `AppState` with that engine, `tokenizer: Mutex::new(None)`,
   `model_path: None`, and no plugins.
5. Build the router with `build_router(Arc::new(state))`.
6. Send `GET /v1/models`. Assert HTTP 200 and that the JSON `data` array contains the exact
   registered model id.
7. Send `POST /v1/chat/completions` with:
   ```json
   {
     "model": "fixture-model",
     "messages": [{"role": "user", "content": "hello"}],
     "max_tokens": 2,
     "stream": false
   }
   ```
8. Assert HTTP 200, `choices[0].message.role == "assistant"`, and a non-empty content or
   token representation. Do not assert a particular random-model sentence.
9. Send `GET /v1/status`. Assert the response contains `status`, `engine_state`, `backend`,
   `loaded_models`, and `kv_cache`.
10. Send `POST /v1/chat/completions` with `model: "missing-model"`. Assert HTTP 4xx, JSON
    `error.type` is stable, and the message includes `Run 'grim pull missing-model'`.

**Required test name:**
`acceptance_model_catalog_chat_status_and_unknown_model_error`.

**Required command:**
```bash
cargo test -p grim-server acceptance_model_catalog_chat_status_and_unknown_model_error -- --nocapture
```

**Do not:** download a model, call the network, depend on `/tmp` state, or weaken the unknown-model
assertion to “request did not panic.”

### P0-B — Make status truthful and complete

**Goal:** every status consumer sees the same snapshot and no fabricated measurements.

**Files:** `crates/grim-server/src/lib.rs`, `crates/grim-scheduler/src/*.rs`,
`crates/grim-engine/src/*.rs`, `docs/observability.md`.

**Steps:**

1. Search for the scheduler's authoritative queue fields and identify the existing read-only
   accessors. Do not count requests by scanning logs.
2. Add one read-only engine method that returns:
   `active_requests`, `waiting_requests`, `admitted_requests`, and `paused_requests`.
3. If the scheduler has no accessor, add a `SchedulerSnapshot` struct next to the scheduler
   state. Keep it read-only and derive `Clone + Serialize` only if the existing crate pattern
   allows it.
4. Add `scheduler` to the JSON returned by `get_status` with those four integer fields.
5. Add the same `scheduler` object to `/metrics` by making `/metrics` reuse `get_status`; do not
   duplicate the calculation.
6. Replace fixed timing values such as `"ttft_ms": 820.0` and `"prefill_tps": 12.3` with actual
   engine values. If no measurement exists, return JSON `null`, not a guessed number.
7. Keep units explicit: bytes are integers, rates are tokens per second, durations are
   milliseconds.
8. Add a unit test with an empty engine. Assert all queue counts are zero and timing values are
   null or zero according to the chosen documented contract.

**Required test name:** `status_reports_scheduler_counts_and_no_fake_timings`.

**Required command:**
```bash
cargo test -p grim-server status_reports_scheduler_counts_and_no_fake_timings -- --nocapture
```

**Failure rule:** if the scheduler cannot expose a safe snapshot without taking a lock that would
block generation, stop and document the lock ordering before changing it. Do not read private
fields from another crate.

### P0-C — Backend and model identity contract

**Goal:** a user can tell what is running and which model path supplied it.

**Steps:**

1. Keep `GRIM_BACKEND` as the requested backend value.
2. Add `requested_backend` and `active_backend` separately if the current `backend` field mixes
   them. `requested_backend` is the environment/config value; `active_backend` is the backend
   actually selected.
3. For CPU fallback, return `active_backend: "cpu"` and `fallback_reason` with a short reason.
4. For a loaded model, return its catalog id and resolved path. For no model, return `null`.
5. Add a test with `GRIM_BACKEND=cpu` that proves the response says CPU. Restore the environment
   variable in a guard so tests remain independent.

**Required test name:** `status_distinguishes_requested_and_active_backend`.

### P0-D — Health and metrics exposure policy

**Goal:** deployment probes work, while metrics do not become public accidentally.

**Files:** `crates/grim-server/src/lib.rs`, `crates/grim-cli/src/main.rs`, configuration docs.

**Steps:**

1. Keep `/healthz` as a cheap liveness response that does not load a model.
2. Add `/readyz` that returns HTTP 200 only when the server can accept inference; return HTTP
   503 with JSON `{ "status": "not_ready", "reason": ... }` when no model is loaded and the
   configured mode requires one.
3. Preserve `/health` for compatibility.
4. Make the metrics bind address explicit in configuration. Default it to loopback.
5. Reject a public metrics bind unless an explicit opt-in setting is true. Return a configuration
   error before starting the listener.
6. Add tests for default loopback, explicit public opt-in, and rejection without opt-in.

**Required test names:**

- `healthz_does_not_require_loaded_model`
- `readyz_reports_503_without_model`
- `metrics_public_bind_requires_explicit_opt_in`

### P0-E — Dashboard smoke test

**Goal:** prove the dashboard reads the same status values as the API.

**Files:** `crates/grim-server/src/lib.rs`, `crates/grim-garage/src/routes.rs`,
`crates/grim-garage/src/web/app.js`.

**Steps:**

1. Use the existing `/api/stats` route and compare its backend, model, VRAM, KV, and GPU fields
   with `/v1/status` from the same fixture state.
2. Add a Rust route test that requests `/` and asserts the HTML references `/api/stats`.
3. Add a route test for `/api/stats` that asserts the JSON has backend, model, `kv_cache`, and
   GPU fields.
4. If a browser test framework is already configured, add one smoke test that loads `/`, waits
   for the stats request, and checks the connection indicator becomes live. If no browser
   framework exists, do not add a new one for this P0; the HTTP route test is the minimum gate.

**Required command:**
```bash
cargo test -p grim-garage --lib
```

### P0 completion checklist

P0 is complete only when all of the following are true:

- [ ] P0-A acceptance test passes without network or GPU.
- [ ] Unknown model returns a stable 4xx JSON error with a recovery command.
- [ ] `/v1/status`, `/status`, and `/metrics` expose the same backend/model/KV/scheduler snapshot.
- [ ] No fixed fake timing values remain in the status response.
- [ ] Backend requested vs. active values are distinguishable.
- [ ] `/healthz` and `/readyz` have tested status semantics.
- [ ] Public metrics binding requires explicit opt-in.
- [ ] Dashboard route/API smoke tests pass.
- [ ] `cargo fmt --all -- --check` passes for the changed files.
- [ ] `cargo test -p grim-server --lib` and `cargo test -p grim-garage --lib` pass.
- [ ] `cargo clippy -p grim-server -p grim-garage --all-targets -- -D warnings` passes.

When a checkbox fails, leave P0 open and record the exact failing command and error in the task
notes. Do not change `u-test.md` to PASS until the checklist is complete.

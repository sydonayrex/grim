# Grim Usability Test — End-to-End Findings

**Tester**: Claude Code static analysis + architecture walkthrough
**Hardware Target**: AMD Radeon RX 9070 XT (RDNA4, 32 GB VRAM)
**Test Date**: 2026-07-04
**Scenario Doc**: `usability_test_scenario.md`

---

## Executive Summary

| Persona | Task | Result |
|---|---|---|
| Linus (Junior DevOps) | Install & daemon setup | :yellow_circle: Partial |
| Marcus (Senior Platform) | systemd service + telemetry | :yellow_circle: Partial |
| Alex (Junior Full-Stack) | IDE wired to port 11434, streaming | :yellow_circle: Partial |
| Sarah (Senior Backend) | LoRA adapter injection via API | :red_circle: Blocked |
| Elena (AI Researcher) | GGUF load + speculative draft bundle | :red_circle: Blocked |
| David (Engineering Director) | Security audit / multi-tenant | :yellow_circle: Partial |

**Legend**: :green_circle: Works end-to-end :yellow_circle: Architecture correct, stub execution :red_circle: Missing user-facing wire-up

---

## Phase 4+ Implementation Status

The Phase 4 GPU backend work from this session implemented or improved:

- **ROCm HIP Graph capture/replay** — `HipGraphExecutor`, `hipGraphExtendFromGlobalStream`, `hipGraphUpload` FFI stubs in `grim-backend-rocm`
- **`.hsaco` JIT compilation cache** — `HsacoKernelCache` with seahash disk-keyed caching
- **ROCm AI tensor primitives** — `rocblas_gemm_ex` bindings, `KvLayout` enum, `kv_to_block_major` / `kv_from_block_major` converters
- **Vulkan GLSL kernel generation** — `generate_matmul_glsl`, `compile_glsl_to_spirv` for CubeCL backend
- **XNACK-aware memory** — `memcpy_with_xnack_fallback` probing
- **OxiBLAS verification** — SIMD GEMM exists via `matrixmultiply` crate
- **QJL KV compression** — random orthogonal rotation + sign-bit key compression
- **MTP speculative decoding** — `LlamaMtp` + `LlamaMtpAdapter` chain
- **GGUF dtype mapping** — `map_gguf_dtype_to_grim`, `is_quantized`, `display_name`

All Phase 4+ code compiles cleanly (`cargo check --workspace` succeeds). However, none of these components are wired into the user-facing CLI paths. The ROCm backend, JIT cache, GEMM kernels, and KV quantization exist as building blocks but are not activated from `grim run` or `grim serve`.

---

## Task 1: Zero-Configuration Install & Daemon Setup

### Persona: Linus (Junior DevOps) & Marcus (Senior Platform)

**Steps**: 1. Run `install.sh` → 2. `grim service install --config /etc/grim/grim.toml` → 3. `grim service start` → 4. Confirm status + logs

---

### Step 1 — `install.sh`

**Finding**: `install.sh` does not exist. No `dist/` directory exists in the repository. The architecture document §12.3 describes this bootstrap script, but it was never created.

**Impact (Linus)**: A junior DevOps engineer following the documentation hits a dead end immediately. They must manually build from source (`cargo build --release`), locate the binary, and copy it to `/usr/local/bin`. This cannot be completed in under 5 minutes without supervisor support. **This directly fails the Task 1 success criterion.**

**Impact (Marcus)**: Workable with extra steps, but the stated "thin bootstrap scripts" goal from the architecture is unimplemented.

**Status**: :red_circle: Missing

---

### Step 2 — `grim service install --config /etc/grim/grim.toml`

**Finding**: CLI subcommand is wired correctly (`main.rs:156-172`). Platform dispatch (`cfg!(target_os)`) correctly selects `SystemdManager`/`LaunchdManager`/`WindowsScmManager`. The systemd unit template includes `Type=notify`, `WatchdogSec=10`, `Restart=on-failure`.

**Critical gap**: The unit file content is printed via `println!` (`service.rs:82`) and **never written to disk**. The actual `std::fs::write("/etc/systemd/system/grim.service", ...)` is absent. After the command completes, the system is unchanged — `systemctl start grim` fails with "unit not found."

**Impact (Linus)**: The command prints "Service installation finished successfully." with no error, giving false confidence. The service is not registered. Linus has no indication something went wrong.

**Impact (Marcus)**: Will immediately identify the failure. Workable for a senior engineer, invisible trap for a junior.

**Status**: :yellow_circle: Logic correct, disk write not implemented

---

### Step 3 — `grim service start`

**Finding**: `SystemdManager::start` (`service.rs:91-94`) prints a message and returns `Ok(())`. No `std::process::Command::new("systemctl").args(["start", "grim"]).status()` invocation exists.

**Status**: :yellow_circle: Stub — actual service manager not invoked

---

### Step 4 — Status & Log Path

**Finding**: `ServiceManager::status()` always returns `Ok(ServiceStatus::Running)` regardless of actual system state (`service.rs:101`). No `systemctl is-active` query. `log_path` is hardcoded `None` in `main.rs:169`; `/var/log/grim/` does not exist.

**Telemetry endpoint**: `/metrics` exists and returns JSON (`lib.rs:228-240`). The fields (`rocm_gpu_count`, `xack_enabled`) are hardcoded — it does not probe actual hardware. On a system with no ROCm libraries, the response incorrectly reports GPU presence.

**Impact (Marcus)**: The `/metrics` endpoint provides a useful foundation, but it reports fabricated hardware state rather than real telemetry.

**Status**: :red_circle: Status always-running stub; log path unconfigured; metrics unprobed

---

### Task 1 Verdict: :yellow_circle: PARTIAL

All data structures and platform dispatch logic are sound. The gaps are the last mile: no `dist/install.sh`, no disk writes from `install`, no `systemctl` invocations from `start`, no real status queries, and fabricated telemetry.

---

## Task 2: Drop-in Ollama Replacement & Multi-adapter LoRA

### Persona: Alex (Junior Full-Stack) & Sarah (Senior Backend)

**Steps**: 1. IDE → `http://127.0.0.1:11434` → 2. Prompt `def quicksort(arr):` → 3. Inject `"adapters": ["peft-style-guide"]` → 4. Verify SSE streaming

---

### Step 1 — Port 11434

**Finding**: Default bind is correctly `127.0.0.1:11434` (`main.rs:33`). Any Ollama-compatible IDE extension pointing to `localhost:11434` connects to Grim without reconfiguration.

**Status**: :green_circle: Correct

---

### Step 2 — Code Completion Streaming

**Finding**: `POST /v1/chat/completions` is registered (`lib.rs:257`). When `"stream": true` is in the request, the SSE handler (`lib.rs:41-59`) returns a valid Axum SSE stream using `stream::unfold`.

**Critical gap**: The stream produces exactly 5 synthetic tokens (`Token 0` through `Token 4`), each after a 50ms sleep. `engine.tick()` is never called. The actual `Llama::forward` is not in the hot path. Alex's IDE receives the same generic output regardless of prompt content.

**Impact (Alex)**: Streaming appears to work — good for confidence. But output is identical for every prompt, so Alex cannot distinguish "working" from "broken" without inspecting token content.

**Status**: :yellow_circle: SSE plumbing correct, actual inference not in stream

---

### Step 3 — LoRA Adapter Injection via API

**Finding**: The engine has a complete adapter registry — `register_adapter`, `resolve_adapters`, `drop_adapter` (`engine/lib.rs:186-217`). `Llama::forward` applies registered adapters via `apply_adapters_to_logits` when `adapters: &[AdapterHandle]` is non-empty (`model.rs:206-216`). LoRA math is correct — rank decomposition, α/r scaling, logit accumulation (`lora.rs:65-95`).

**Critical gap**: `chat_completions` reads `body` as `serde_json::Value` but **never inspects an `"adapters"` key**. There is no code that reads adapter IDs from the request body, resolves them against the engine registry, and passes them to `drive_forward`. In `drive_forward` (`lib.rs:332`), `adapter_ids` is hardcoded to `Vec::new()` — adapters are unconditionally skipped for all requests.

**Impact (Sarah)**: `POST /v1/chat/completions` with `"adapters": ["peft-style-guide"]` is silently ignored. The adapter has no effect on output. Sarah receives generic tokens with no diagnostic indicating the adapter was not applied. This blocks her JSON grammar-constrained output requirement.

**Status**: :red_circle: Adapter key not parsed from HTTP body; per-request adapter routing not wired

---

### Step 4 — SSE Format Verification

**Finding**: SSE event format matches OpenAI's streaming format — `event: message`, `data` is JSON with `choices[0].delta.content`. Compatible with Ollama proxies and most IDE extensions.

**Status**: :green_circle: Format correct

---

### Task 2 Verdict: :yellow_circle: PARTIAL (with :red_circle: blocker for Sarah)

Alex's basic connectivity test passes at the wire level. Sarah's multi-adapter workflow is completely blocked — the infrastructure exists but the HTTP request field is never read.

---

## Task 3: Capability-Based Model Ingestion

### Persona: Elena (AI Researcher)

**Steps**: 1. `grim run --model ./models/llama3-8b-Q4_K.gguf` → 2. Load companion draft-bundle JSON → 3. Verify >80 tokens/sec on RX 9070 XT

---

### Step 1 — GGUF Model Loading

**Finding**: `grim run --model <path>` calls `cmd_run` (`run.rs:9-19`). The `model_path` argument is captured as `_model_path` — discarded. A random `Llama` with `vocab_size: 512`, `hidden_size: 64` is always constructed regardless of the path provided.

The GGUF reader (`read_gguf`, `gguf.rs:172`) is fully implemented — magic check, version validation, metadata KV parsing, per-tensor dtype tag resolution including Q4_K. `GgufProvider` TensorProvider bridge exists. **None of this is called.**

**Impact (Elena)**: `grim run --model ./models/llama3-8b-Q4_K.gguf` silently runs a toy random model, not Llama 3. Token output is meaningless. Elena will immediately notice the model is not llama3 and cannot proceed.

**Status**: :yellow_circle: GGUF reader implemented; CLI-to-reader bridge missing

---

### Step 2 — Draft Bundle & Speculative Decoding

**Finding**: The speculation pipeline is architecturally complete:
- `SpeculativeCausalLm::auto()` selects DSpark when a draft bundle is provided (`speculative_wrapper.rs:52-100`)
- `Engine::register_with_dspark` wires draft + Markov + confidence heads (`lib.rs:136-145`)
- Draft VRAM residency check via `GRIM_AVAILABLE_VRAM` env var (`lib.rs:157-160`)
- `grim spec train` accepts arguments but its body is a print statement (`spec.rs`)

**Gap**: No CLI flag or config key to attach a companion spec bundle. `grim run` has no `--draft-bundle` argument. The speculative path is inaccessible from the CLI.

**Impact (Elena)**: Cannot attach a DSpark bundle via CLI. The full speculation infrastructure cannot be exercised by an end user.

**Status**: :yellow_circle: Engine API complete; no CLI surface for bundle loading

---

### Step 3 — RDNA4 Hardware Throughput

**Finding**: The ROCm backend has correct HIP FFI declarations — `hipMalloc`, `hipFree`, `hipMemcpy`, `hipDeviceGetAttribute`, wavefront detection, GEMM dispatch tables (`lib.rs:55-80`). Wavefront-size-sensitive KV layout switch is implemented (`lib.rs:302-307`). The Phase 4 JIT cache, HIP graph executor, and aiter bindings are all in place.

**Critical gap**: All inference in `cmd_run` runs on `CpuDevice`. No ROCm device is opened, probed, or used. `hipGetDeviceCount` FFI exists but is never called. On RX 9070 XT (RDNA4, W32 wavefront), the KV layout should select `RowMajor`, but neither branch is reached.

On the CPU backend, expected throughput for an 8B model is <1 token/sec. The 80 tokens/sec target requires ROCm execution.

**Impact (Elena)**: Cannot measure RDNA4 throughput because the ROCm backend is never activated. All Phase 4 GPU work remains inaccessible from the CLI.

**Status**: :red_circle: ROCm backend not connected to run/serve path; all execution falls to CPU

---

### Task 3 Verdict: :red_circle: BLOCKED for production use case

Three independent gaps stack: GGUF path discarded at CLI, draft bundle has no CLI surface, ROCm backend not activated. All three must be resolved before Elena's workflow is valid.

---

## Cross-Cutting Findings

### David (Engineering Director) — Security & Multi-tenancy

**Plugin grants (deny-by-default)**: Manifest parser correctly reads `[plugin.capabilities.grants]` — `network`, `filesystem`, `request_metadata` — and stores them in `PluginGrants` (`plugin/lib.rs:196-206`). These fields are available for inspection.

**Gap**: No runtime enforcement. When a WASM plugin is instantiated, its `Linker` receives no host-side capability gating. There is no check like "if `grants.network == false`, do not link the WASI socket functions." The deny-by-default is in data only, not behavior.

**Outbound network isolation**: The server makes no outbound calls by default — satisfies David's no-network-leaks requirement for the local test environment. However, a WASM plugin with `network = true` in its manifest would not be blocked from making outbound connections.

**Status**: :yellow_circle: Data model correct; runtime enforcement missing

---

### Notable Inconsistency: `grim serve` vs `grim run --serve`

The architecture and the generated systemd unit file reference `grim serve --config /etc/grim/grim.toml`. The CLI has no `serve` subcommand — only `run --serve`. The `ExecStart` line in the generated unit (`service.rs:62`) uses `{} serve` which would be invalid at runtime. Both `SystemdManager` and `LaunchdManager` templates have this bug.

---

## Summary of Gaps by Severity

### P0 — Blocks all user testing on real hardware

| Gap | File | Fix |
|---|---|---|
| ROCm backend never activated from CLI | `run.rs:15` | Add `Device::Rocm(0)` selection; call `RocmDevice::open()` |
| GGUF model path discarded | `run.rs:10` | Wire `_model_path` → `read_gguf` → `GgufProvider` → `Llama::load` |

### P1 — Blocks specific persona workflows

| Gap | File | Fix |
|---|---|---|
| Adapter key not parsed from HTTP body | `grim-server/src/lib.rs:38` | Read `body["adapters"]`, resolve via `engine.resolve_adapters()`, pass to `drive_forward` |
| Service managers print unit files, don't write them | `service.rs` | Replace `println!` with `std::fs::write` + actual `systemctl`/`launchctl` invocation |
| `ExecStart` uses `grim serve` (doesn't exist) | `service.rs:62` | Change to `grim run --serve --config` |

### P2 — Reduces capability but has workarounds

| Gap | File | Fix |
|---|---|---|
| `dist/install.sh` doesn't exist | — | Create `dist/install.sh` per §12.3 spec |
| `grim run` has no `--draft-bundle` flag | `main.rs`, `run.rs` | Add `--draft-bundle` arg, call `register_with_dspark` |
| `status()` always returns `Running` | `service.rs:101` | Call `systemctl is-active grim` via `Command::new` |
| WASM plugin grants not enforced at runtime | `wasm_loader.rs` | Gate Wasmtime linker imports against `grants` flags |
| `rocm_gpu_count` hardcoded in `/metrics` | `grim-server/src/lib.rs:236` | Call `hipGetDeviceCount` FFI |
| Chat completions stream doesn't call `engine.tick()` | `grim-server/src/lib.rs:42-59` | Connect `stream::unfold` to real `Engine::tick()` decode loop |

---

## What Works End-to-End (Verified)

- :green_circle: Port `11434` default bind — Ollama-compatible IDE connectivity
- :green_circle: SSE response format — correct OpenAI streaming event structure
- :green_circle: Pause/resume HTTP endpoints fully wired to engine scheduler
- :green_circle: Plugin manifest parsing — all §6.4 fields (grants, reload, stage, priority)
- :green_circle: Duplicate `(stage, priority)` rejection at registry load time
- :green_circle: GGUF v3 reader — magic, version, metadata KV, per-tensor Q4_K dtype tagging
- :green_circle: LoRA math — rank decomposition, α/r scaling applied to logits
- :green_circle: Speculative wrapper — DSpark/NativeMtp/Plain auto-selection logic
- :green_circle: Scheduler — admission, chunked prefill, preemption, ITL enforcement, pause/resume
- :green_circle: Paged KV block pool — tentative/commit/rollback semantics
- :green_circle: Phase 4 ROCm GPU backend building blocks — HIP graph, HSACO JIT, aiter GEMM, XNACK fallback, Vulkan GLSL kernel generation, QJL KV compression (all implemented, not yet wired to CLI)

---

## Recommendations

1. **Highest ROI first**: Wire `run.rs` to `read_gguf` and select `Device::Rocm`. This unlocks real inference on RX 9070 XT for all three test tasks simultaneously.
2. **Server adapter routing**: Three lines of code in `chat_completions` to parse `body["adapters"]` and pass them through `drive_forward` — unlocks Sarah's entire multi-adapter workflow.
3. **Service real invocations**: Replace `println!` with `std::fs::write` + `Command::new("systemctl")` in `SystemdManager`; fix `ExecStart` to use `grim run --serve`; fix `LaunchdManager` similarly.
4. **Create `dist/install.sh`**: Makes Linus self-sufficient and meets the "under 5 minutes" success criterion.
5. **WASM grant enforcement**: Required before David allows any third-party plugins in the local test environment.
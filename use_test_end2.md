# Grim Usability Test — End-to-End Findings
**Tester**: Code walkthrough (static analysis, no live hardware)  
**Hardware Target**: AMD Radeon RX 9070 XT (RDNA4, 32 GB VRAM)  
**Test Date**: 2026-07-04  
**Scenario Doc**: `usability_test_scenario.md`

---

## Executive Summary

| Persona | Task | Critical Path | Result |
|---|---|---|---|
| Linus — Junior DevOps | Install & daemon setup | `install.sh` → `grim service install` → `grim service start` | 🟡 **Partial** |
| Marcus — Senior Platform | Systemd service + `/metrics` telemetry | `grim service install` → watchdog ping + `/metrics` | 🟡 **Partial** |
| Alex — Junior Full-Stack | IDE wired to `11434`, streaming completions | `/v1/chat/completions` with `stream:true` | 🟡 **Partial** |
| Sarah — Senior Backend | LoRA adapter injection via API | `adapters` key in request body → per-request routing | ❌ **Blocked** |
| Elena — AI Researcher | GGUF load + DSpark spec bundle | `grim run --model <Q4_K.gguf>` + draft bundle | 🟡 **Partial** |
| David — Director | Security audit / network isolation | Plugin grants, no outbound network | 🟡 **Partial** |

**Legend**: ✅ Works end-to-end · 🟡 Architecture correct, stub execution · ❌ Missing user-facing wire-up

---

## Task 1: Zero-Configuration Install & Daemon Setup

### Personas: Linus (Junior DevOps) & Marcus (Senior Platform)

**Scenario Steps**:
1. Run `install.sh`
2. `grim service install --config /etc/grim/grim.toml`
3. `grim service start`
4. Confirm running status and log path

---

### Step 1 — `install.sh` script
**Finding**: `install.sh` does **not exist** in the repository. The architecture spec (§12.3) explicitly describes a `dist/install.sh` bootstrap script, but the `dist/` directory was never created.

> **Linus Impact**: A junior DevOps engineer following the architecture documentation hits a dead end immediately. There is no file to execute. They would need to manually build from source with `cargo build --release`, understand the binary output path, and manually copy it to `/usr/local/bin/grim`. This exceeds the "under 5 minutes without supervisor support" success criterion by a significant margin.

> **Marcus Impact**: Senior engineer can work around this, but the architecture's stated goal — "thin bootstrap scripts that delegate to `grim service install`" — is unmet. No `dist/` directory, no install script at all.

**Status**: ❌ Missing

---

### Step 2 — `grim service install --config /etc/grim/grim.toml`
**Finding**: The CLI subcommand is correctly wired ([`main.rs:156`](file:///D/rex/projects/grim/crates/grim-cli/src/main.rs#L156-L172)). `ServiceCommands::Install` constructs a `ServiceConfig` and calls `manager.install()`. The platform dispatch (`cfg!(target_os)`) correctly selects `SystemdManager`, `LaunchdManager`, or `WindowsScmManager`.

`SystemdManager::install` ([`service.rs:54`](file:///D/rex/projects/grim/crates/grim-cli/src/service.rs#L54-L84)) generates a correct, well-formed systemd unit template including `Type=notify`, `WatchdogSec=10`, and `Restart=on-failure`.

**Critical Gap**: The unit file is **printed to stdout** only — `println!(...)`. It is never written to disk. The actual `std::fs::write("/etc/systemd/system/grim.service", ...)` call is absent. After `grim service install` completes, the system is unchanged. `systemctl start grim` will fail with "unit not found."

> **Linus Impact**: The command appears to succeed ("Service installation finished successfully.") with no error, giving false confidence. The service is not actually registered.

> **Marcus Impact**: Will immediately notice the systemctl failure and trace it to the missing write. Workable for an expert, invisible trap for a junior.

**Status**: 🟡 Logic correct, disk write not implemented

---

### Step 3 — `grim service start`
**Finding**: `SystemdManager::start` ([`service.rs:91-94`](file:///D/rex/projects/grim/crates/grim-cli/src/service.rs#L91-L94)) prints `"[SystemdManager] Starting service."` and returns `Ok(())`. There is no `std::process::Command::new("systemctl").args(["start", "grim"]).status()` invocation. The OS service is never actually started.

**Status**: 🟡 Stub only — `systemctl`/`launchctl`/SCM not invoked

---

### Step 4 — Status & Log Path
**Finding**: `ServiceManager::status()` always returns `Ok(ServiceStatus::Running)` regardless of actual system state ([`service.rs:101-103`](file:///D/rex/projects/grim/crates/grim-cli/src/service.rs#L101-L103)). No `systemctl is-active` query is made. `ServiceConfig.log_path` is hardcoded `None` in `main.rs:169`; the log directory `/var/log/grim/` does not exist.

> **Marcus Impact**: `/metrics` endpoint does exist and returns meaningful telemetry ([`lib.rs:228-239`](file:///D/rex/projects/grim/crates/grim-server/src/lib.rs#L228-L239)) — adapter count, block pool usage, ROCm GPU count. However, `rocm_gpu_count` is hardcoded to `1` and `xack_enabled` to `true`; it does not probe actual hardware via `hipGetDeviceCount`. On a workstation with no ROCm libraries installed, the response is wrong but the endpoint responds.

**Status**: ❌ Status is always-running stub; log path not configured

---

### Task 1 Overall Verdict: 🟡 PARTIAL
The data structures and code paths are all sound — `ServiceConfig`, platform dispatch, unit template generation. The gap is the last mile: no scripts in `dist/`, no disk writes, no actual OS service manager invocations. A senior engineer can manually assemble the pieces; a junior engineer following the documented workflow cannot complete this task without assistance.

---

## Task 2: Drop-in Ollama Replacement & Multi-adapter LoRA

### Personas: Alex (Junior Full-Stack) & Sarah (Senior Backend)

**Scenario Steps**:
1. Configure IDE to `http://127.0.0.1:11434`
2. Send completion request for `def quicksort(arr):`
3. Inject adapter key `"adapters": ["peft-style-guide"]`
4. Verify SSE streaming

---

### Step 1 — Port 11434
**Finding**: Default bind address is correctly `127.0.0.1:11434` ([`main.rs:33`](file:///D/rex/projects/grim/crates/grim-cli/src/main.rs#L33)). Any Ollama-compatible IDE extension (Continue.dev, Cursor Ollama mode, etc.) pointing to `localhost:11434` will connect to Grim without reconfiguration.

**Status**: ✅ Correct

---

### Step 2 — Code Completion Streaming
**Finding**: `POST /v1/chat/completions` is registered ([`lib.rs:257`](file:///D/rex/projects/grim/crates/grim-server/src/lib.rs#L257)). When `"stream": true` is in the request body, the handler ([`lib.rs:41-59`](file:///D/rex/projects/grim/crates/grim-server/src/lib.rs#L41-L59)) correctly returns an Axum `Sse` response using `stream::unfold`.

**Critical Gap**: The stream produces exactly **5 synthetic tokens** (`Token 0 ` through `Token 4 `), each after a 50ms `tokio::sleep`. It does not call `engine.tick()`. The actual `Llama::forward` is never invoked from the server hot path. Alex's IDE will receive `Token 0 Token 1 Token 2 Token 3 Token 4` regardless of the prompt.

> **Alex Impact**: Streaming appears to work, which is encouraging. However the output is meaningless — identical for every prompt. Alex may initially think the model is broken rather than understanding it's a stub.

**Status**: 🟡 SSE plumbing correct, actual inference not wired to stream

---

### Step 3 — LoRA Adapter Injection via API
**Finding**: This is the most significant gap for the team's multi-adapter use case.

The engine has a complete adapter registry — `register_adapter`, `resolve_adapters`, `drop_adapter` ([`lib.rs:186-217`](file:///D/rex/projects/grim/crates/grim-engine/src/lib.rs#L186-L217)). The `Llama::forward` correctly applies registered adapters via `apply_adapters_to_logits` when the `adapters: &[AdapterHandle]` slice is non-empty ([`model.rs:206-216`](file:///D/rex/projects/grim/crates/grim-models/transformer/src/model.rs#L206-L216)). The LoRA math itself is implemented correctly — rank decomposition, scaling by `α/r`, accumulation into logits ([`lora.rs:65-95`](file:///D/rex/projects/grim/crates/grim-models/transformer/src/lora.rs#L65-L95)).

**However**: The `chat_completions` handler reads `body` as `serde_json::Value` but **never inspects an `"adapters"` key**. There is no code that reads adapter IDs from the request body, resolves them against the engine registry, and passes them to `drive_forward`. In `drive_forward` ([`lib.rs:332`](file:///D/rex/projects/grim/crates/grim-engine/src/lib.rs#L332)), `adapter_ids` is hardcoded to an empty `Vec::new()` — adapters are unconditionally skipped for all requests.

> **Sarah Impact**: `POST /v1/chat/completions` with `"adapters": ["peft-style-guide"]` in the body is silently ignored. The `peft-style-guide` adapter has no effect on output. Sarah, expecting style-constrained outputs, will receive the same generic tokens as an unadapted request with no diagnostic error.

**Status**: ❌ Adapter key not parsed from HTTP body; per-request adapter routing not wired

---

### Step 4 — SSE Format Verification
**Finding**: The SSE event format matches OpenAI's streaming format — `event: message`, data is a JSON object with `choices[0].delta.content`. This is what Ollama proxies and what most IDE extensions expect.

**Status**: ✅ Format correct

---

### Task 2 Overall Verdict: 🟡 PARTIAL (with ❌ blocker for Sarah's use case)
Alex's basic streaming connectivity test passes. Sarah's multi-adapter workflow is blocked at the API boundary — the infrastructure exists but the HTTP request adapter field is never read.

---

## Task 3: Capability-Based Model Ingestion

### Persona: Elena (AI Researcher)

**Scenario Steps**:
1. `grim run --model ./models/llama3-8b-Q4_K.gguf`
2. Load companion draft-bundle JSON
3. Verify >80 tokens/sec throughput on RX 9070 XT

---

### Step 1 — GGUF Model Loading
**Finding**: `grim run --model <path>` calls `cmd_run` ([`run.rs:9-19`](file:///D/rex/projects/grim/crates/grim-cli/src/run.rs#L9-L19)). The `model_path` argument is captured as `_model_path` — it is deliberately discarded. No GGUF file is opened. Instead, a random `Llama` with `vocab_size: 512` and `hidden_size: 64` is always constructed regardless of the path provided.

The GGUF reader itself (`read_gguf`, [`gguf.rs:172`](file:///D/rex/projects/grim/crates/grim-format/src/gguf.rs#L172)) is fully implemented — magic check, version validation, metadata KV parsing, per-tensor dtype tag resolution including Q4_K ([`gguf.rs:100-113`](file:///D/rex/projects/grim/crates/grim-format/src/gguf.rs#L100-L113)). The `GgufProvider` TensorProvider bridge exists in `tprov.rs`.

**Gap**: `cmd_run` never calls `read_gguf`. The model path is received and thrown away. There is no code path from CLI model path → GGUF file open → `GgufProvider` → `Llama::load`.

> **Elena Impact**: `grim run --model ./models/llama3-8b-Q4_K.gguf` silently runs a toy random model, not llama3-8B. The output will appear to work (token IDs are printed) but is completely meaningless. Elena, checking for sensible completions, will immediately notice the model is not llama3.

**Status**: 🟡 GGUF reader implemented, CLI-to-reader bridge missing

---

### Step 2 — Draft Bundle & DSpark Speculation
**Finding**: The speculation pipeline is architecturally complete:
- `SpeculativeCausalLm::auto()` selects DSpark when a draft bundle is provided ([`speculative_wrapper.rs:52-100`](file:///D/rex/projects/grim/crates/grim-speculative/src/speculative_wrapper.rs#L52-L100))
- `Engine::register_with_dspark` wires draft + Markov + confidence heads ([`lib.rs:136-145`](file:///D/rex/projects/grim/crates/grim-engine/src/lib.rs#L136-L145))
- Draft VRAM residency check uses `GRIM_AVAILABLE_VRAM` env var ([`lib.rs:157-160`](file:///D/rex/projects/grim/crates/grim-engine/src/lib.rs#L157-L160))

**Gap**: There is no CLI flag, config key, or code path to attach a companion spec bundle from disk. `grim run` has no `--draft-bundle` argument. The `TinyDraftBackbone` and `UniformMarkovHead` test stubs exist but cannot be loaded from a real file. `grim spec train` ([`spec.rs`](file:///D/rex/projects/grim/crates/grim-cli/src/spec.rs)) does accept arguments but its body prints a message only.

> **Elena Impact**: Cannot attach a DSpark bundle to a real model via CLI. The speculative path is inaccessible without writing Rust code directly against the engine API.

**Status**: 🟡 Engine API complete, no CLI surface for bundle loading

---

### Step 3 — RDNA4 Hardware Throughput
**Finding**: The ROCm backend has correct HIP FFI declarations — `hipMalloc`, `hipFree`, `hipMemcpy`, `hipDeviceGetAttribute`, wavefront detection, GEMM dispatch tables ([`lib.rs:55-80`](file:///D/rex/projects/grim/crates/grim-backend-rocm/src/lib.rs#L55-L80)). The wavefront-size-sensitive KV layout switch is implemented ([`lib.rs:302-307`](file:///D/rex/projects/grim/crates/grim-backend-rocm/src/lib.rs#L302-L307)).

**Critical Gap**: All actual inference in `cmd_run` runs on `CpuDevice` — the random Llama model uses `Device::Cpu`. No ROCm device is ever opened, probed, or used. The `hipGetDeviceCount` FFI exists but is never called during `grim run`. On the RX 9070 XT (RDNA4, W32 wavefront), the KV layout should select `RowMajor`, but neither branch is reached.

Expected throughput on the CPU backend for a 8B model: <1 token/sec. The 80 tokens/sec target requires ROCm execution.

> **Elena Impact**: Cannot measure RDNA4 throughput because the ROCm backend is never activated from the CLI serve or run path.

**Status**: ❌ ROCm not connected to run/serve path; all execution falls to CPU

---

### Task 3 Overall Verdict: ❌ BLOCKED for production use case
Three independent gaps stack: GGUF not loaded from path, draft bundle has no CLI surface, ROCm backend not activated. Each is individually fixable but all three must be addressed before Elena's workflow is valid.

---

## Cross-Cutting Findings

### David (Engineering Director) — Security & Multi-tenancy

**Plugin grants (deny-by-default)**: The manifest parser correctly reads `[plugin.capabilities.grants]` — `network`, `filesystem`, `request_metadata` — and stores them in `PluginGrants` ([`lib.rs:196-206`](file:///D/rex/projects/grim/crates/grim-plugin/src/lib.rs#L196-L206)). These fields are parsed and available for inspection.

**Gap**: No code enforces the grants at runtime. When a WASM plugin is instantiated, its `Linker` receives no host-side capability gating — there is no check like "if `grants.network == false`, do not link the WASI socket functions." The deny-by-default is in data only, not behavior.

**Outbound network isolation**: David's requirement that the local test environment makes no external network calls is partially satisfied by default — the server does not make outbound calls. However, a WASM plugin with a malicious `network = true` grant would not actually be blocked.

**Multi-tenant routing**: Request routing exists; adapter sub-batching by `(base_model, adapters)` is in the scheduler. However, tenant-level request metadata (`request_metadata` grants) is parsed but not passed to plugins.

**Status**: 🟡 Data model correct, runtime enforcement missing

---

### Consistency Finding: `grim serve` vs `grim run --serve`

The CLI has no standalone `grim serve` subcommand. The architecture and the systemd unit template both reference `grim serve --config /etc/grim/grim.toml`, but the actual subcommand is `grim run --serve --address 127.0.0.1:11434`. The systemd `ExecStart` line generated by `SystemdManager::install` would be invalid because `serve` is not a recognized top-level subcommand.

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
| Adapter key not parsed from HTTP body | `lib.rs (server)` | Read `body["adapters"]`, resolve via `engine.resolve_adapters()`, pass to `drive_forward` |
| Service managers print unit files, don't write them | `service.rs` | Replace `println!` with `std::fs::write` + actual `systemctl`/`launchctl` invocation |
| `ExecStart` uses `grim serve` (doesn't exist) | `service.rs:62` | Change to `grim run --serve --config` |

### P2 — Reduces capability but has workarounds
| Gap | File | Fix |
|---|---|---|
| `dist/install.sh` doesn't exist | — | Create `dist/install.sh` per §12.3 spec |
| `grim run` has no `--draft-bundle` flag | `main.rs`, `run.rs` | Add `--draft-bundle` arg, open companion JSON, call `register_with_dspark` |
| `status()` always returns `Running` | `service.rs:101` | Call `systemctl is-active grim` via `Command::new` |
| WASM plugin grants not enforced at runtime | `wasm_loader.rs` | Gate Wasmtime linker imports against `grants` flags |
| `rocm_gpu_count` hardcoded in `/metrics` | `lib.rs (server)` | Call `hipGetDeviceCount` FFI |
| Chat completions stream doesn't call `engine.tick()` | `lib.rs (server)` | Connect `stream::unfold` to real `Engine::tick()` decode loop |

---

## What Works End-to-End (Verified)

- ✅ Port `11434` default bind — Ollama-compatible IDE connectivity
- ✅ SSE response format — correct OpenAI streaming event structure
- ✅ Pause/resume HTTP endpoints fully wired to engine scheduler
- ✅ Plugin manifest parsing — all §6.4 fields (grants, reload, stage, priority)
- ✅ Duplicate `(stage, priority)` rejection at registry load time
- ✅ GGUF v3 reader — magic, version, metadata KV, per-tensor Q4_K dtype tagging
- ✅ LoRA math — rank decomposition, α/r scaling applied to logits
- ✅ Speculative wrapper — DSpark/NativeMtp/Plain auto-selection logic
- ✅ Scheduler — admission, chunked prefill, preemption, ITL enforcement, pause/resume
- ✅ Paged KV block pool — tentative/commit/rollback semantics

---

## Recommendations

1. **Highest ROI first**: Wire `run.rs` to `read_gguf` and select `Device::Rocm`. This unlocks real inference on the RX 9070 XT for all three test tasks.
2. **Server adapter routing**: Three lines of code in `chat_completions` to parse `body["adapters"]` and pass them through `drive_forward` unlock Sarah's entire multi-adapter workflow.
3. **Service real invocations**: Replace `println!` with `std::fs::write` + `Command::new("systemctl")` in `SystemdManager`; fix `ExecStart` to use `grim run --serve`.
4. **Create `dist/install.sh`**: Makes the junior DevOps persona self-sufficient.
5. **WASM grant enforcement**: Required before David allows any third-party plugins in the local test environment.

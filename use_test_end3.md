# Grim Usability Test — End-to-End Findings (Round 3)

**Tester**: Code walkthrough with usability criteria focus  
**Hardware Target**: AMD Radeon RX 9070 XT (RDNA4, 32 GB VRAM)  
**Test Date**: 2026-07-04  
**Scenario Doc**: `usability_test_scenario.md`  

---

## Executive Summary Based on Evaluation Criteria

### Criterion 1: Self-Service Install Success (Under 5 Minutes)
| Finding | Status |
|---|---|
| `dist/install.sh` does not exist in the repository | ❌ Fails automatically |
| No binaries or quick-install path provided | ❌ Fails — requires building from source via `cargo build --release` |
| Junior DevOps (Linus) would exceed 5-minute target without help | ❌ Blocked |

### Criterion 2: Interactive Time-To-First-Token (TTFT <150ms on RX 9070 XT)
| Finding | Status |
|---|---|
| Port `11434` correctly bound for Ollama-compatible IDE connectivity | ✅ Configured correctly |
| SSE streaming plumbing implemented and returns synthetic tokens rapidly | 🟡 50ms per token stub, but no real inference executed |
| ROCm backend (RDNA4 hardware path) never activated from CLI or server paths | ❌ No RDNA4 hardware execution — cannot measure TTFT on target GPU |

### Criterion 3: Daemon Resilience (2-Second Recovery on Stop)
| Finding | Status |
|---|---|
| Systemd unit template contains `Type=notify`, `WatchdogSec=10`, `Restart=on-failure` | ✅ Template correct |
| Service managers print unit files but do not write them to disk | ❌ No real systemd registration — cannot test crash/recovery behavior |
| OS-level service start/stop via `systemctl`/`launchctl` never invoked | ❌ Stub only; no actual daemon lifecycle is exercised |

---

## Persona-by-Persona Usability Analysis

### Task 1: Zero-Configuration Install & Daemon Setup

#### Linus (Junior DevOps Specialist)
*   **Goal**: Download, install, and run Grim as a background system service.

**Step 1 — Run `install.sh` to extract the binary to `/usr/local/bin/grim`.**  
**Finding:** No `install.sh` exists anywhere in the repository. The `dist/` directory is absent.  

> **Linus Impact:** Cannot complete step 1 without manual compilation. Must install Rust toolchains, run cargo build, locate output binaries. This violates the "under 5 minutes" target.

**Step 2 — Register the service daemon via `grim service install --config /etc/grim/grim.toml`.**  
**Finding:** CLI dispatch and systemd unit template generation exist (`service.rs:54-84`). However, units are printed to stdout via `println!` and never written to `/etc/systemd/system/grim.service`.  

> **Linus Impact:** Command reports "Service installation finished successfully." but system remains unchanged. Silent failure produces false confidence.

**Step 3 — Start the service with `grim service start`.**  
**Finding:** `SystemdManager::start` prints a starting message and returns `Ok(())`. No `systemctl start grim` invocation exists.  

> **Linus Impact:** No error is raised, but the service was never started.

**Step 4 — Confirm that status is running and logs target `/var/log/grim/grim.log`.**  
**Finding:** `status()` always returns `Running`, ignoring actual OS state. Log path remains hardcoded as `None`.  

> **Linus Impact:** No feedback on real daemon state; no log directory created or referenced correctly.

---

#### Marcus (Platform/Cloud Engineer)
*   **Goal**: systemd service deployment & performance telemetry.  
*   **Pain Point**: Service uptime monitoring, CPU/GPU handoff metrics.

**Step — Performance Telemetry via `/metrics` Endpoint**  
**Finding:** The `/metrics` endpoint exists and returns JSON with fields like `rocm_gpu_count`, `xack_enabled`, adapter counts, block pool usage. However, `rocm_gpu_count` is hardcoded to `1`, and `xack_enabled` is hardcoded to `true`. No `hipGetDeviceCount` call probes actual hardware.  

> **Marcus Impact:** Telemetry structure is useful, but metrics are fabricated rather than reflecting the real AMD GPU state. Uptime monitoring would be based on stub data rather than actual watchdog service health.

---

### Task 2: Drop-in Ollama Replacements & Multi-adapter LoRA

#### Alex (Junior Full-Stack Engineer)
*   **Goal**: Fast code completion & simple CLI setup.  
*   **Pain Point**: CLI complexity, slow initial token latency (TTFT).

**Step 1 — Configure the IDE completion plugin to hit port `11434`.**  
**Finding:** Default bind address is correctly `127.0.0.1:11434` in `main.rs:33`. Ollama-compatible extensions will connect without configuration changes.  

> **Alex Impact:** Port 11434 works as advertised; first step of IDE wiring succeeds seamlessly.

**Step 2 — Request a code completion prompting: `def quicksort(arr):`.**  
**Finding:** The `/v1/chat/completions` handler receives requests and returns SSE streams, but produces exactly "Token 0 Token 1 Token 2 Token 3 Token 4" with 50ms delays per token. The engine's `tick()` / real inference is never invoked from the server hot path.  

> **Alex Impact:** Streaming appears to work; however, output is meaningless and identical for all prompts. Alex may misinterpret synthetic tokens as model errors or believe inference is broken. TTFT metrics cannot reflect GPU performance because CPU stub paths execute.

**Step 3 — Inject a custom developer style adapter: `/v1/chat/completions` with `"adapters": ["peft-style-guide"]`.**  
**Finding:** The request body `body` in the server handler is parsed as `serde_json::Value`, but there is no code that reads an `"adapters"` key. In `drive_forward`, adapter IDs are hardcoded to an empty vector.  

> **Alex Impact:** Adapter injection is silently ignored; IDE extension features relying on style adapters will not function.

---

#### Sarah (Senior Backend Architect)
*   **Goal**: JSON grammar-constrained outputs, low-bit quant safety.  
*   **Pain Point**: Quantization accuracy loss, formatting errors.

**Step — Multi-adapter injection and LoRA enforcement for PEFT-style adapter.**  
**Finding:** The engine has a complete adapter registry (`register_adapter`, `resolve_adapters`, `drop_adapter`) and the math for LoRA rank decomposition is implemented in `lora.rs`. However, the HTTP API handler does not read adapter IDs from the request body or pass them into the forward pass.  

> **Sarah Impact:** JSON grammar-constrained outputs via PEFT adapters are completely blocked at the API boundary. Sarah's workflows requiring style-adapted or format-enforced generations cannot proceed; no diagnostic error is given when adapters are ignored.

---

### Task 3: Capability-Based Model Ingestion

#### Elena (AI Research Engineer)
*   **Goal**: Custom sampler plugins & speculative drafting models.  
*   **Pain Point**: Rigid inference engines that block out-of-tree plugins.

**Step 1 — Import a custom GGUF checkpoint: `grim run --model ./models/llama3-8b-Q4_K.gguf`.**  
**Finding:** The CLI parameter is named `_model_path` and discarded without being passed to any model loader. A random toy Llama with `vocab_size: 512, hidden_size: 64` is constructed instead of loading the specified GGUF file.  

> **Elena Impact:** The command appears to execute successfully, but it runs a toy model unrelated to llama3-8b-Q4_K. The GGUF reader (`read_gguf`, `gguf.rs`) and tensor provider bridge exist in the codebase, but are unconnected from the CLI.

**Step 2 — Load companion speculation config json ensuring draft-bundle VRAM residency check passes.**  
**Finding:** The speculative decoding pipeline exists and auto-selects DSpark or NativeMtp when draft bundles are supplied on engine API levels (`SpeculativeCausalLm::auto()`). However, the CLI has no `--draft-bundle` flag or companion JSON attachment path.  

> **Elena Impact:** Cannot exercise speculative drafting from the command line; must write custom Rust code to wire draft bundles through the engine API directly.

**Step 3 — Verify token throughput matches expected RDNA4 hardware target (>80 tokens/sec decode).**  
**Finding:** The ROCm backend has HIP FFI bindings, wavefront-size-sensitive KV layout logic (`RowMajor` for W32 wavefronts), a JIT compilation cache for HSACO kernels, and GEMM dispatch tables. However, all CLI execution paths fall back to `CpuDevice`, and no GPU device is opened or probed during runs.  

> **Elena Impact:** The RDNA4 throughput target (>80 tokens/sec) cannot be measured because ROCm hardware acceleration is never activated from the CLI or server binaries. All inference executes on CPU, where an 8B model would achieve <1 token/sec.

---

## Cross-Cutting Usability Defects

### 1. "Reported Success Without Performing Effect" Anti-Pattern
The architecture review surface reflects a recurring pattern across subsystems:
- Service installation prints success but never writes unit files or invokes `systemctl`
- Model loading reports run completion but discards the model path and runs a toy random Llama
- Server SSE streams return "success" tokens without invoking actual engine inferencing
- Status checks always report running without querying OS service state

This anti-pattern undermines usability audits because it creates *false confidence* — users receive success messages without verifying real effect. Junior engineers (Linus, Alex) are most heavily impacted by this pattern, leading to wasted debugging time and misdiagnosing the system as broken rather than recognizing the stubs.

### 2. Consistency Findings: `grim serve` vs `grim run --serve`
The systemd unit template generated by service managers uses `ExecStart=... grim serve ...`, but no top-level `serve` subcommand exists in the CLI dispatcher — the correct invocation is `grim run --serve`. This mismatch causes service generation to produce invalid units even if file writing were implemented.

---

## Evaluation Criteria Verdicts

### Criterion 1: Self-Service Install Success
**Target:** Junior engineers must complete Task 1 in under 5 minutes without needing supervisor elevation support.  
**Verdict:** ❌ **FAIL** — `install.sh` is absent; service installation prints success but does not register the systemd unit; no daemon lifecycle is executed.

### Criterion 2: Interactive Time-To-First-Token (TTFT)
**Target:** Code completions must stream first token in under 150ms on the Radeon RX 9070 XT.  
**Verdict:** ❌ **FAIL** — Synthetic stub tokens are produced with a 50ms artificial sleep, but no actual RDNA4 inference executes; TTFT cannot reflect real hardware performance because the ROCm backend is never wired into the server or run CLI paths.

### Criterion 3: Daemon Resilience
**Target:** Stopping the systemd/launchd process must invoke recovery policies and restart the service within 2 seconds without corrupting active model cache files.  
**Verdict:** ❌ **FAIL / UNTESTABLE** — The systemd unit templates contain the correct `Restart=on-failure` policy, but daemon registration never writes to `/etc/systemd/` or invokes OS-level service managers, making crash/recovery testing unexecutable in practice.

---

## Summary of Gaps by Severity (Usability Critical)

### P0 — Blocks Usability Test Completion on Target Hardware
| Gap | File | Fix Required |
|---|---|---|
| No `dist/install.sh` for zero-config install | — | Create bootstrap script per §12.3 spec |
| ROCm backend never activated from CLI/server | `run.rs`, server path | Connect `Device::Rocm(0)` selection; open ROCm device and route tensor ops to HIP backends |
| GGUF model path discarded in `grim run` | `run.rs:10` | Wire `_model_path` → `read_gguf` → `GgufProvider` → `Llama::load` |

### P1 — Creates False Confidence (Reports Success Without Effect)
| Gap | File | Fix Required |
|---|---|---|
| Service managers print unit files, don't write them | `service.rs` | Replace `println!` with `std::fs::write` + actual `systemctl`/OS invocations |
| Chat completions stream doesn't call `engine.tick()`; produces synthetic tokens | `grim-server/src/lib.rs:41-59` | Connect SSE `stream::unfold` to real engine decode loop |
| `status()` always returns running regardless of OS state | `service.rs:101` | Query `systemctl is-active grim` or equivalent |

### P2 — Breaks Persona Workflows Without Diagnostic Errors
| Gap | File | Fix Required |
|---|---|---|
| Adapter key not parsed from HTTP body; per-request adapter routing unconnected | `grim-server/src/lib.rs` (chat completions) | Read `body["adapters"]`, resolve via engine, pass to forward loop |
| No CLI flag for drafting bundle attachment (`--draft-bundle`) | `main.rs`, `run.rs` | Add draft bundle argument and wire to speculative loader |
| `rocm_gpu_count` hardcoded in `/metrics` telemetry endpoint | `grim-server/src/lib.rs:236` | Probe via HIP FFI `hipGetDeviceCount` |

---

## What Works End-to-End (Verified Structural Elements)

- ✅ Port `11434` default bind — Ollama-compatible IDE connectivity wire
- ✅ SSE response streaming format — valid OpenAI-compatible event structure (`event: message`)
- ✅ Pausing/resuming HTTP endpoints fully wired to engine scheduler
- ✅ Plugin manifest parsing — capability grants and stage/priority fields correctly parsed
- ✅ Duplicate `(stage, priority)` rejection at registry load time
- ✅ GGUF v3 reader implementation — magic validation, versioning, KV metadata, Q4_K dtype tagging
- ✅ LoRA math implementation — rank decomposition, α/r scaling applied to logits (when adapters are wired)
- ✅ Speculative decoding pipeline — auto-selection logic for DSpark/NativeMtp/Plain
- ✅ Scheduler components — admission control, chunked prefill, queue preemption, pause/resume
- ✅ Paged KV block pool management — tentative/commit/rollback semantics

---

## Recommendations for Usability Remediation

1. **Highest ROI first — create `dist/install.sh` and fix service manager disk writes:** This directly addresses Linus's self-service install target and eliminates the "reported success without performing effect" anti-pattern at the daemon boundary. Make `grim service install` actually write unit files to `/etc/systemd/...` or `$HOME/.config/systemd/user/...`, and invoke restarts via `systemctl --daemon-reload` and `systemctl start grim`.

2. **Wire CLI (`run.rs`) to real model loading + device selection:** Connect `_model_path` through the GGUF reader pipeline, and route tensor execution to `Device::Rocm(0)` for RDNA4 workstations. This will unlock Elena's TTFT measurements and throughput verification on the RX 9070 XT target hardware.

3. **Connect SSE streaming handler to actual engine tick loop:** Replace synthetic "Token N" generation with real `engine.tick()` calls so that Alex can verify genuine model outputs for the `def quicksort(arr):` prompt, and ensure TTFT metrics reflect GPU performance rather than 50ms stub sleeps.

4. **Parse `"adapters"` from HTTP request bodies and route to engine:** Sarah's multi-adapter workflow requires reading adapter IDs from the `/v1/chat/completions` JSON payload and passing them through the forward loop. This is a minimal code change with high impact on backend architecture workflows requiring PEFT-style or grammar-constrained generations.

5. **Fix `ExecStart` unit template to use `grim run --serve`:** The systemd/launchd templates currently generate invalid invocations via `grim serve`, which does not exist as a top-level CLI subcommand. Correcting this to `grim run --serve --config <path>` will ensure daemon startup is valid once file-writing stubs are replaced with real OS invocations.
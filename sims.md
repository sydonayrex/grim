# Code Review: Hardcoded Simulation Bugs

## Critical Simulation Bugs (Production Code)

### 1. `grim-disagg/src/lib.rs` — `DisaggRouter` is entirely simulated
**Severity: CRITICAL**
**Skills: `code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`**

Three of four trait methods are pure simulations — they print and return `Ok(())` without doing any actual work:

```rust
// dispatch_prefill (line 58-63): just prints, returns Ok(())
fn dispatch_prefill(&self, request_id: u64, _tokens: &[u32]) -> Result<()> {
    println!("[DisaggRouter] Dispatching prefill task...");
    Ok(())  // ← No actual dispatch happens
}

// dispatch_decode (line 91-96): just prints, returns Ok(())
fn dispatch_decode(&self, request_id: u64, _last_token: u32, assignment: PoolAssignment) -> Result<()> {
    println!("[DisaggRouter] Dispatching decode task...");
    Ok(())  // ← No actual decode happens
}
```

Only `transfer_kv_cache` does *some* work — it calls `send_block_remote` and `fetch_block_remote`, but those are also simulations (see below).

### 2. `grim-kvtransport/src/lib.rs` — `NetworkKvClient` is entirely simulated
**Severity: CRITICAL**
**Skills: `code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`, `rust-ffi`, `performance`**

```rust
// send_block_remote (line 262-273): just prints, returns Ok(())
pub fn send_block_remote(&self, block_id: BlockId, k: &[f32], _v: &[u32], target_ip: &str) -> Result<()> {
    println!("[NetworkKvClient] Sending KV block {}...");
    Ok(())  // ← No data is actually sent
}

// fetch_block_remote (line 277-288): returns hardcoded values
pub fn fetch_block_remote(&self, block_id: BlockId, target_ip: &str, block_elems: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    println!("[NetworkKvClient] Fetching KV block {}...");
    Ok((vec![1.0; block_elems], vec![2.0; block_elems]))  // ← Hardcoded 1.0/2.0 values
}
```

The `fetch_block_remote` function returns hardcoded `vec![1.0; block_elems]` and `vec![2.0; block_elems]` regardless of what was sent. This means:
- Data integrity is impossible — sent data ≠ fetched data
- The simulation masks real network errors
- Any test using this client passes regardless of correctness

### 3. `grim-kvtransport/src/lib.rs` — `NvmeWeightStreamer` uses mock weights
**Severity: HIGH**
**Skills: `code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`, `rocm`, `rocm-kernels`, `rust-ffi`, `rust-ml-llm-architecture`, `kernel-review`, `gpu`, `performance`**

```rust
// prefetch_layer_async (line 357): uses mock weights instead of reading from disk
let mock_weights = vec![0.5f32; 1024];
cache.insert(layer_id, mock_weights.clone());
```

The function is supposed to read layer weights from NVMe, but instead inserts hardcoded `0.5f32` values. This means:
- The weight cache contains garbage data
- Any computation using these weights produces incorrect results
- The LRU eviction and bandwidth tracking work, but the data is fake

### 4. `grim-disagg/src/lib.rs` — `transfer_kv_cache` uses mock data
**Severity: HIGH**
**Skills: `code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`, `rust-ffi`, `rust-ml-llm-architecture`**

```rust
// transfer_kv_cache (line 81): uses mock data instead of real KV blocks
let mock_data = vec![0.5f32; 1024];
self.kv_client.send_block_remote(block_idx, &mock_data, &mock_data, &self.decode_node_addr)?;
```

The function transfers hardcoded `0.5f32` values instead of actual KV cache blocks. This means:
- The "transfer" doesn't move real data
- The receive side gets fake data
- The entire KV handoff protocol is simulated, not real

## Medium Severity Simulation Patterns

### 5. `grim-cli/src/doctor.rs` — Hardcoded GPU backend value
**Severity: MEDIUM**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`**

```rust
report.gpu_backend_actual = Some("cpu (hardcoded 0 in /metrics)".into());
```

The GPU backend is hardcoded to "cpu" with a comment acknowledging it's hardcoded. This means the doctor command always reports CPU backend regardless of actual GPU availability.

### 6. `grim-cli/src/service.rs` — Hardcoded localhost addresses
**Severity: MEDIUM**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`**

```rust
let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
```

SSL certificate SANs are hardcoded to localhost only, preventing production deployment with custom domains.

### 7. `grim-engine/src/model_loader.rs` — Debug `eprintln!` in production
**Severity: LOW**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`**

```rust
eprintln!("[meta-get-u32] {key} = {u} (u32)");
eprintln!("[meta-get-u32] {key} = MISSING");
```

Debug logging in production code that prints metadata lookups to stderr. Not a simulation bug, but noisy in production.

## Already Fixed (from prior session)
**Skills: `code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`**

| Bug | Location | Status |
|-----|----------|--------|
| Shadow `load_model_from_gguf` in `run.rs` | `grim-cli/src/run.rs:471` | Fixed — removed, uses engine loader |
| Hardcoded `Device::Cpu` for Mamba/Bert | `grim-cli/src/run.rs` (old shadow loader) | Fixed — engine loader uses `device.clone()` |
| `use_gguf` computed but unused | `grim-cli/src/run.rs` | Fixed — engine loader handles GGUF |

## Summary

| Severity | Count | Impact |
|----------|-------|--------|
| **CRITICAL** | 2 | `NetworkKvClient` and `DisaggRouter` are pure simulations — no actual network or dispatch happens |
| **HIGH** | 2 | `NvmeWeightStreamer` and `transfer_kv_cache` use mock data instead of real data |
| **MEDIUM** | 2 | Hardcoded GPU backend and SSL SANs |
| **LOW** | 1 | Debug `eprintln!` in production |
| **Already Fixed** | 3 | Shadow loader, hardcoded Device::Cpu, unused use_gguf |

**The most critical issue is that `NetworkKvClient` and `DisaggRouter` are entirely simulated** — they don't perform any actual network communication or dispatch. The `fetch_block_remote` function returns hardcoded `1.0`/`2.0` values regardless of what was sent, making data integrity impossible. The `NvmeWeightStreamer` inserts mock `0.5f32` weights instead of reading from disk, making all weight-dependent computations incorrect.

## Why ML/AI Skills Were Not Categorized

The following skills were considered but deemed **not directly relevant** to these specific bugs:

| Skill | Why Not Relevant |
|-------|-----------------|
| `rust-ffi` | No FFI boundaries are involved — all bugs are in pure Rust logic (network dispatch, mock data, weight caching). No C interop, no extern calls. |
| `rocm` | None of the bugs are ROCm-specific. The GPU backend detection bug was already fixed. These bugs are in the infrastructure layer, not GPU compute. |
| `rocm-kernels` | No GPU kernel code is involved. The bugs are in dispatch/router/transport, not compute kernels. |
| `kernel-review` | No GPU kernel code is involved. The bugs are in the distributed routing and network transport layer. |
| `rust-ml-llm-architecture` | While the project is an ML inference engine, the bugs are in the infrastructure layer (network transport, KV cache transfer, weight streaming, device detection), not in model architecture. Model loading bugs were already fixed. |
| `llama-cpp` | No llama.cpp integration is involved in these bugs. |
| `llm-training` / `model-tuning` | No training or tuning code is involved — these are inference-time infrastructure bugs. |
| `gpu` | No GPU compute code is involved. The bugs are in I/O simulation (network, disk, dispatch). |

**Key distinction:** These bugs are in the **systems infrastructure layer** (network transport, distributed routing, weight caching, device detection) — not in ML model logic, GPU kernels, or FFI boundaries. The selected skills (`code-reviewer`, `clean-code-guard`, `ponytail-review`, `caveman`) focus on code quality and review methodology, which is exactly what's needed to identify and fix simulation bugs.

**Exception:** `rust-ffi` and `rocm` skills *would* be relevant if the `NetworkKvClient` or `NvmeWeightStreamer` used FFI to call C networking libraries or ROCm APIs for GPU memory transfers. Currently they're pure Rust stubs, but if real implementations are added that cross FFI/ROCm boundaries, those skills would become directly applicable.

## Production Use Case: 118B MoE Model with NVMe Expert Offloading

**Requirement:** Load a 118B parameter MoE model with 8B experts, where most experts reside on VRAM and the remaining stream from NVMe storage on-demand.

This use case **immediately activates** the ML/AI skills that were previously not relevant:

| Skill | Now Relevant Because |
|-------|---------------------|
| `rocm` | GPU memory management for VRAM allocation, expert placement, and GPU-CPU-NVMe data transfers via ROCm/HIP APIs |
| `rocm-kernels` | Custom GPU kernels for NVMe-to-GPU direct transfers, expert loading pipelines, and async copy operations |
| `rust-ffi` | FFI to NVMe driver libraries (libnvme), ROCm/HIP runtime APIs for GPU memory management, and potentially kernel-bypass networking for `NetworkKvClient` |
| `rust-ml-llm-architecture` | Understanding MoE model structure, expert routing, and the implications of expert offloading on inference correctness |
| `kernel-review` | Custom GPU kernels for streaming expert weights from NVMe to VRAM without blocking inference |
| `gpu` | GPU memory management, VRAM allocation strategies, and PCIe/NVLink bandwidth optimization |
| `m06-error-handling` | I/O error handling for NVMe failures, GPU OOM recovery, and network partition tolerance in distributed KV transport |
| `performance` | Bandwidth optimization for NVMe reads, GPU memory transfer optimization, and latency hiding for expert loading |

### Implementation Requirements

1. **`NvmeWeightStreamer` must be replaced with real NVMe I/O:**
   - FFI to `libnvme` or direct `/dev/nvme*` access for reading expert weights
   - Async I/O with prefetching to hide latency
   - LRU cache for VRAM-resident experts with eviction to NVMe
   - Error handling for I/O failures (corruption, device removal)

2. **`NetworkKvClient` must implement real network transport:**
   - TCP/UDP or RDMA for KV block transfer between prefill/decode nodes
   - Potentially FFI to `libibverbs` for kernel-bypass networking
   - Data integrity checks (checksums) to verify transferred KV blocks

3. **Expert scheduling and placement:**
   - Hot/cold expert identification based on routing frequency
   - VRAM budget management for 118B parameter model
   - Async expert loading during inference to avoid blocking

4. **GPU memory management:**
   - ROCm/HIP API integration for VRAM allocation/deallocation
   - Unified memory or explicit GPU memory management
   - Potential FFI to `hipMalloc`/`hipMemcpy`/`hipMemcpyHtoD`

**This transforms the bugs from "simulation stubs" to "production blockers"** — the current mock implementations would produce completely incorrect results for a 118B MoE model, as the mock `0.5f32` weights would be used instead of real expert weights streamed from NVMe.

## Additional Simulation Stubs Found

Beyond the 7 issues already documented, the following simulation stubs were found across the project:

### 8. `grim-server/src/lib.rs` — Mock model fallback when model file not found
**Severity: HIGH**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`, `rust-ml-llm-architecture`**

```rust
// line 681-713: falls back to Llama::random() when model file not found
eprintln!("[grim-server] Model file not found for '{}', using mock model", req.name);
let mock = Box::new(grim_models_transformer::Llama::random(Device::Cpu, LlamaConfig { ... }));
engine.register_model(&req.name, mock);
```

The server silently loads a random-weight model when the requested model file doesn't exist, returning `"loaded_kind": "mock"` in the response. This means:
- Users get garbage output without realizing the model wasn't loaded
- No error is returned to the client — the response says "success"
- The mock model uses hardcoded config (32000 vocab, 512 hidden, 4 layers) regardless of what was requested

### 9. `grim-server/src/lib.rs` — Hardcoded API endpoint stubs
**Severity: MEDIUM**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`**

```rust
// line 583-592: embeddings endpoint returns hardcoded [0.01, 0.02, 0.03]
async fn embeddings() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "data": [{ "embedding": [0.01, 0.02, 0.03] }],
        "model": "grim"
    }))
}

// line 595-599: audio transcription returns hardcoded string
async fn audio_transcriptions() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "text": "Simulated audio transcription output." }))
}

// line 602-609: image generation returns hardcoded URL
async fn images_generations() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "data": [{ "url": "http://localhost:8080/image.png" }] }))
}
```

Three OpenAI-compatible API endpoints return hardcoded responses instead of actual computation. The embeddings endpoint returns `[0.01, 0.02, 0.03]` regardless of input, the audio endpoint returns a fixed string, and the image endpoint returns a localhost URL that doesn't exist.

### 10. `grim-core/src/session.rs` — `Graph::replay` is a no-op
**Severity: MEDIUM**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`, `rust-ml-llm-architecture`**

```rust
// line 212-214: just prints, returns Ok(())
pub fn replay(&self, _session: &mut dyn SessionT) -> Result<()> {
    println!("[Graph] Replaying captured static computation graph with {} nodes", self.nodes.len());
    Ok(())
}
```

The computation graph replay function is a pure simulation — it prints the node count and returns `Ok(())` without executing any graph nodes. This means:
- Shape-specialized computation paths are never actually executed
- The `GraphBuilder` trait can build graphs, but they can't be replayed
- Any optimization based on graph replay is dead code

### 11. `grim-backend-metal/src/lib.rs` — Dummy Metal command buffer
**Severity: LOW**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`, `rust-ffi`**

```rust
// line 618-620: creates a dummy command buffer instead of a real one
let dummy_cmd = ctx.command_queue.commandBuffer()
    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create dummy command buffer".into())))?;
return Ok((out_storage, Box::new(MetalHandle { command_buffer: dummy_cmd })));
```

The Metal backend creates a dummy command buffer for certain code paths instead of properly encoding compute commands. While this may be intentional for fallback paths, the name "dummy" suggests it's a placeholder that was never properly implemented.

### 12. `grim-server/src/lib.rs` — Mock models in integration tests used as production fallback
**Severity: LOW**
**Skills: `code-reviewer`, `clean-code-guard`, `caveman`**

```rust
// lines 1597, 1659, 1713, 1766, 1819, 1870, 1966, 2024: Llama::random() used as mock
let mock_model = Box::new(grim_models_transformer::Llama::random(Device::Cpu, ...));
engine.register_model("default", mock_model);
```

Multiple integration tests use `Llama::random()` as mock models. While these are in test code, the pattern of using `Llama::random()` as a "mock model" has leaked into production code (see issue #8 above).

## Updated Summary

| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 2 | `NetworkKvClient` and `DisaggRouter` are pure simulations |
| **HIGH** | 3 | `NvmeWeightStreamer` mock weights, `transfer_kv_cache` mock data, `grim-server` mock model fallback |
| **MEDIUM** | 3 | Hardcoded GPU backend, SSL SANs, `Graph::replay` no-op, hardcoded API endpoints |
| **LOW** | 2 | Debug `eprintln!`, dummy Metal command buffer, mock models in tests |
| **Already Fixed** | 3 | Shadow loader, hardcoded Device::Cpu, unused use_gguf |

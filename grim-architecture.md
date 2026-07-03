# Grim: A Rust Inference Engine Architecture

## 0. What Grim borrows from prior art

Grim isn't a clone of any single project — it's a synthesis of the three approaches that currently define the state of the art, each solving a different layer of the problem:

| Project | What it does best | What Grim takes from it |
|---|---|---|
| **Candle** (huggingface/candle) | Clean Rust-native `Tensor`/`Device`/`DType` core, `VarBuilder` for hierarchical weight loading from safetensors, pluggable `BackendDevice`/`BackendStorage` traits per hardware target | The **tensor + device abstraction layer** and the **VarBuilder-style weight-loading pattern** — Grim's `grim-tensor` and model-construction code follow this shape almost directly, since it's the cleanest solution in Rust for "load a hierarchical checkpoint into typed structs." |
| **vLLM** | PagedAttention (block-table KV cache, OS-virtual-memory-style paging), iteration-level continuous batching, a scheduler with waiting/running/swapped queues | The **memory manager and scheduler design** for Grim's serving path — paged KV cache blocks, per-step admission of new sequences, and preemption when memory is tight. |
| **llama.cpp / ggml** | GGUF as a single self-describing file (metadata + tensors + tokenizer), a computation graph (`ggml_cgraph`) that's backend-agnostic, block-wise quantization (Q4_K, Q5_K, etc.) with per-block scale/min | Grim's **model packaging format** (`.grim`/GGUF-compatible), the **graph-based executor** so a model is "build a graph once, run it many times," and the **quantization scheme** (block quant with per-block scale, dequant-on-load for sensitive tensors like embeddings/norms).
| **DSpark** (DeepSeek-AI & Peking University, 2026 — *"Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"*) | A drafter that couples a parallel backbone (O(1) block drafting) with a lightweight sequential "Markov head" to fix suffix decay, plus a confidence head and a load-aware scheduler that dynamically trims verification depth under traffic pressure, while staying exact via rejection sampling | Grim's **default decode path** — every autoregressive generation request runs through confidence-scheduled semi-autoregressive speculation unless explicitly disabled. See §5.3. |
| **vLLM MTP** (Multi-Token Prediction — `docs/features/speculative_decoding/mtp.md`) | Model-native speculative decoding: the target checkpoint itself carries extra prediction head(s) trained alongside the trunk (DeepSeek-V3/Gemma-4-assistant style) and shares KV cache with the target, so no separate draft model or distillation step is required — just a `num_speculative_tokens` depth knob | Grim's **zero-config speculation path** — when a loaded model exposes native MTP heads, `grim-speculative` uses them automatically instead of requiring an attached DSpark draft bundle. See §5.3. |
| **TurboQuant** (0xSero/turboquant, KV cache compression, ICLR 2026) | Runtime KV-cache compression — random orthogonal rotation + Lloyd-Max scalar quantization + QJL residual sign bits for keys (down to 3-bit), group quantization with bit-packing for values (2–4 bit) — applied only to full-attention layers, with an unbiased inner-product estimator so attention scores stay statistically correct | Grim's **KV block compression tier**, sitting inside `grim-memory`'s block pool rather than the weight-quantization path — see §5.4. |

The rest of this document treats these as *inspirations for subsystems*, not a monolithic design to imitate — Grim's job is to unify transformer, Mamba/SSM, vision, audio, and diffusion models under one execution model, which none of the three fully do today (Candle covers the model zoo breadth but not serving; vLLM covers serving but is transformer/attention-shaped; llama.cpp covers packaging/quantization but is CPU/edge-first and text-centric; DSpark and MTP cover decode-loop acceleration but assume a single dense/MoE transformer target rather than Grim's multi-family scope; TurboQuant covers KV memory footprint but explicitly leaves linear-attention/Mamba state uncompressed).

---

## 1. Requirements

**Functional**
- Load and run: dense transformers (Llama/Mistral-style), Mamba/SSM & hybrid (Mamba2, Jamba-style), vision encoders (ViT/CLIP/SAM-style), audio models (Whisper-style encoder-decoder), diffusion models (UNet/DiT + noise schedulers).
- Single request API surface regardless of model family (text-in/text-out, text-in/image-out, audio-in/text-out, etc.), with family-specific extensions.
- Plugin system: third parties can add new model architectures, new tokenizers/processors, new samplers, new backends (custom kernels/hardware), and pre/post-processing steps — without forking Grim.
- Batched, concurrent serving with streaming output.
- Confidence-scheduled, semi-autoregressive speculative decoding **on by default** for every autoregressive text request, transparent to callers and lossless with respect to the target model's output distribution.

**Non-functional**
- Native Rust, no mandatory Python at runtime (build tooling — converters, evals — may use it).
- Multi-backend: ROCm (primary GPU via hip/rocBLAS), Vulkan (platform-agnostic via shader-based compute), CPU (SIMD via packed_simd / std::simd) — ordered primary → fallback. CUDA (cudarc) and Metal (metal-rs) kept as optional targets, not blocking delivery.
- Memory-safe by construction where possible; `unsafe` isolated to kernel/FFI boundaries.
- Predictable latency under load (continuous batching, backpressure, no unbounded queues).
- A plugin can crash or misbehave without taking down the whole engine (isolation boundary).

**Constraints**
- Single cohesive workspace, but with crate boundaries that mirror deployment boundaries (you should be able to ship `grim-runtime` as a library with zero server dependencies).

---

## 2. High-level architecture

```
                              ┌────────────────────────────────────────┐
                              │              grim-server                │
                              │   HTTP/gRPC API · Auth · Streaming      │
                              └───────────────────┬──────────────────────┘
                                                  │  requests
                              ┌───────────────────▼──────────────────────┐
                              │               grim-engine                 │
                              │  Scheduler │ Request Queue │ Batcher      │
                              │  (waiting / running / swapped, per vLLM)  │
                              └───────┬───────────────────┬──────────────┘
                                     │                   │
                     ┌───────────────▼──────┐   ┌────────▼─────────────┐   ┌──────────────────────┐
                     │   grim-memory          │   │   grim-plugin         │   │   grim-speculative     │
                     │  Paged KV cache        │   │  Plugin registry,     │   │  Draft backbone +      │
                     │  Block allocator       │   │  ABI, sandboxing       │   │  Markov head +         │
                     │  Prefix cache          │   └────────┬─────────────┘   │  confidence scheduler   │
                     └───────────────┬────────┘             │ extends       │  (default-on, §5.3)     │
                                     │                      │               └──────────┬────────────┘
                                     │                      │                          │ wraps CausalLm
                     ┌───────────────▼──────────────────────▼───────────┐
                     │                  grim-core                        │
                     │   Model trait family │ Graph executor │ Session   │
                     └───────┬─────────────────────────────┬─────────────┘
                             │                             │
       ┌─────────────────────▼──────────┐    ┌─────────────▼─────────────────┐
       │          grim-models             │    │          grim-tensor           │
       │  transformer / mamba / vision /  │    │  Tensor, DType, Shape, Device   │
       │  audio / diffusion adapters      │    │  autograd-free inference ops    │
       └─────────────────────┬──────────┘    └─────────────┬─────────────────┘
                             │                             │
                     ┌───────▼─────────────────────────────▼───────┐
                     │              grim-backend-*                  │
                     │   cpu (SIMD) │ rocm (hip) │ vulkan │ cuda/metal │
                     └────────────────────────────────────────────┘

                     ┌────────────────────────────────────────────┐
                     │        grim-quant · grim-format (GGUF-ish)   │
                     │   Loads/saves model files, weight quant,     │
                     │   per-weight-tensor QAT provenance (§7.2)    │
                     └────────────────────────────────────────────┘

                               ┌─────────────────────────────────────┐
                               │ grim-kvquant (TurboQuant-like, §5.4) │
                               │  KV block compression: rotation,    │
                               │  Lloyd-Max, QJL, bit-packing.       │
                               │  Sits inside grim-memory's pool     │
                               └─────────────────────────────────────┘
```

Data flows top-down for control (requests → scheduling → execution) and bottom-up for capability discovery (backends and model families register themselves into `grim-core` at startup).

---

## 3. Workspace layout

```
grim/
├── Cargo.toml                     # workspace root
├── crates/
│   ├── grim-tensor/                # Tensor, Shape, DType, Device — Candle-shaped core
│   ├── grim-backend-cpu/           # SIMD kernels (packed_simd / std::simd), rayon
│   ├── grim-backend-rocm/          # hip/rocBLAS-based kernels, ROCm (primary GPU)
│   ├── grim-backend-vulkan/        # shader-based compute, Vulkan/SPIR-V (platform-agnostic)
│   ├── grim-backend-cuda/          # cudarc-based kernels, PTX/cubin loading (optional)
│   ├── grim-backend-metal/         # metal-rs kernels, MSL (optional)
│   ├── grim-nn/                    # Linear, Embedding, LayerNorm, RMSNorm, Attention,
│   │                                # SSM scan, Conv1d, RoPE, VarBuilder-equivalent
│   ├── grim-format/                # GGUF-compatible reader/writer, safetensors bridge,
│   │                                # tokenizer + processor metadata embedding
│   ├── grim-quant/                 # block quantizers/dequantizers (Q4_K/Q5_K/Q8_0-style),
│   │                                # importance-matrix support
│   ├── grim-core/                  # Model trait family, Graph, Session, KV cache trait,
│   │                                # sampler trait, error types
│   ├── grim-models/
│   │   ├── transformer/            # Llama/Mistral/Qwen/Phi-style dense + MoE
│   │   ├── mamba/                  # Mamba1/Mamba2/hybrid SSM+attention
│   │   ├── vision/                 # ViT, CLIP, SAM-style, DiT backbone
│   │   ├── audio/                  # Whisper-style encoder-decoder, streaming ASR
│   │   └── diffusion/              # UNet + DiT diffusion, schedulers/samplers
│   ├── grim-memory/                # Paged KV cache, block allocator, prefix cache,
│   │                                # SSM state cache (fixed-size, not paged the same way)
│   ├── grim-kvquant/                # TurboQuant-style runtime KV cache compression:
│   │                                # rotation + Lloyd-Max + QJL for keys, group quant
│   │                                # for values, bit-packing, fused dequant-attention
│   │                                # kernels (see §5.4) — separate from grim-quant,
│   │                                # which quantizes weights, not runtime KV
│   ├── grim-speculative/           # DSpark-style semi-autoregressive drafter, Markov head,
│   │                                # confidence head, confidence-scheduled verifier, plus
│   │                                # native MTP support — default-on decode acceleration,
│   │                                # see §5.3
│   ├── grim-scheduler/             # Continuous batching scheduler, admission control
│   ├── grim-plugin/                # Plugin trait ABI, dynamic loading (dylib) + WASM host,
│   │                                # capability negotiation, manifest schema
│   ├── grim-engine/                # Wires scheduler + memory + core into a runtime;
│   │                                # this is the library entry point (`grim-runtime`)
│   ├── grim-server/                # axum HTTP/gRPC server, OpenAI-compatible + native API
│   └── grim-cli/                   # `grim run`, `grim quantize`, `grim bench`, `grim plugin`
├── plugins/                        # first-party example plugins (out-of-tree by design)
│   ├── example-json-grammar/
│   └── example-custom-sampler/
└── xtask/                          # cargo-xtask for kernel codegen, GGUF conversion scripts
```

Crate boundary rule of thumb: **`grim-tensor` through `grim-memory` know nothing about HTTP or scheduling policy; `grim-engine` and above know nothing about kernel internals.** This is what lets you embed `grim-engine` in something other than `grim-server` (e.g. an in-process desktop app) later.

---

## 4. Core abstractions (`grim-core`)

### 4.1 Tensor & device layer (`grim-tensor`, Candle-inspired)

```rust
pub enum Device {
    Cpu,
    Rocm(usize),       // primary GPU target
    Vulkan,            // platform-agnostic compute
    Cuda(usize),       // optional
    Metal(usize),      // optional
}

/// The arithmetic type used for computation (what the hardware computes in).
/// Most backends work in F32 or F16 internally regardless of weight storage.
pub enum ArithType {
    F32, F16, BF16, I64, U32, U8,
}

/// Physical storage encoding of tensor data.
/// When storage differs from the arithmetic type, dequantization is needed
/// before compute. This split means adding a new low-bit format (MXFP4,
/// NVFP4, etc.) adds one Storage variant instead of N new DType variants
/// that each require match arms across the entire dispatch surface.
pub enum Storage {
    /// Stored in native ArithType encoding — no dequant needed.
    Native,
    /// Block-quantized K-quant format (Grim's own PTQ, llama.cpp-compatible).
    KQuant(KQuantScheme),
    /// Grouped INT weights from an external QAT pipeline (EfficientQAT, GPTQ).
    /// Always asymmetric (scale + zero-point) unless explicitly symmetric;
    /// see §7.2 for per-weight-tensor provenance handling.
    GroupInt(GpuIntConfig),
}

pub enum KQuantScheme {
    Q4K, Q5K, Q6K, Q8_0,
}

pub struct GpuIntConfig {
    pub bits: u8,           // 2, 3, 4, or 8
    pub group_size: usize,  // 32, 64, or 128
    pub scheme: GroupQuantScheme,
    pub desc_act: bool,     // false for EfficientQAT (sequential g_idx)
}

pub struct DType {
    pub arith: ArithType,
    pub storage: Storage,
}

/// Every tensor knows its quantization provenance — this stops Grim from
/// re-quantizing a tensor that was already quantization-aware trained,
/// and tells the dequant/matmul kernel which layout to expect.
pub enum QuantProvenance {
    /// Not quantized, or produced by grim-quant's own post-training pass.
    GrimNative,
    /// Produced by an external QAT pipeline. Never re-quantized by grim-quant.
    ExternalQat { bits: u8, group_size: usize, scheme: GroupQuantScheme, desc_act: bool },
}

pub struct Tensor {
    storage: Arc<dyn BackendStorage>,
    layout: Layout,          // shape + strides, row-major by default
    dtype: DType,
    provenance: QuantProvenance,  // per-weight-tensor, resolved at load time
    device: Device,
}

/// A handle to an asynchronous compute operation.
/// CPU backends resolve immediately (synchronize returns Ok(())).
/// GPU backends represent a stream/queue; synchronize() blocks until the
/// operation this handle tracks has completed. Operations on the same
/// device that consume a storage buffer as input implicitly wait for any
/// outstanding handle on that buffer — callers only need to synchronize
/// before reading results back to the CPU.
pub trait ComputeHandle: Send {
    fn synchronize(&self) -> Result<()>;
    fn is_ready(&self) -> bool;
}

/// Every hardware target implements this trait; grim-tensor dispatches
/// through it and never contains device-specific code itself.
/// Operations return both the result storage and a ComputeHandle that
/// tracks the operation's completion. The CPU backend returns handles
/// that are immediately ready; GPU backends (ROCm, Vulkan, CUDA, Metal)
/// return handles backed by stream/queue state.
pub trait BackendDevice: Send + Sync {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>>;
    fn matmul(&self, a: &dyn BackendStorage, b: &dyn BackendStorage, out: &Shape)
        -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;
    // ... elementwise ops, reductions, gather/scatter, conv1d (for SSM),
    // softmax, rope, etc. — each returns (result, handle).
}

pub trait BackendStorage: Send + Sync {
    fn dtype(&self) -> DType;
    fn provenance(&self) -> QuantProvenance;
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>>;
}
```

Kernel registration mirrors Candle: `grim-backend-cpu`, `grim-backend-rocm`, `grim-backend-vulkan`, `grim-backend-cuda`, and `grim-backend-metal` each implement `BackendDevice`/`BackendStorage`, and are selected via Cargo feature flags (`--features rocm`) plus runtime device probing, similar to Candle's `Device::cuda_if_available`. The ROCm backend is the primary GPU target; Vulkan serves as the platform-agnostic fallback; CUDA and Metal are optional.

### 4.2 Weight loading — `VarBuilder`-equivalent (`grim-nn`)

```rust
pub struct WeightSource<'a> {
    tensors: &'a dyn TensorProvider,   // backed by GGUF, safetensors, or in-memory map
    prefix: Vec<String>,
    default_dtype: DType,              // per-tensor override takes priority (see §7.2)
    default_provenance: QuantProvenance,
    device: Device,
}

impl<'a> WeightSource<'a> {
    pub fn pp(&self, name: &str) -> WeightSource<'a> { /* push_prefix, as in Candle */ }
    /// Get a weight tensor. The dtype and provenance are resolved per-tensor:
    /// first from the checkpoint's per-tensor metadata (GGUF key-value map
    /// or safetensors metadata), falling back to `default_dtype` and
    /// `default_provenance` if the checkpoint doesn't specify. This lets
    /// models mix e.g. FP16 embeddings + w4g128 attention + w3g128 MLP
    /// in a single `WeightSource`, with each tensor carrying its own
    /// `QuantProvenance` — see §7.2 for the provenance semantics.
    pub fn get(&self, shape: impl Into<Shape>, name: &str) -> Result<Tensor> { /* ... */ }
}
```

Every model's `load(ws: WeightSource, cfg: &Config) -> Result<Self>` constructor walks the config-defined layer hierarchy and pulls tensors by name — the same pattern Candle uses for `Mistral`/`Whisper`/etc., which is also compatible with mmapped GGUF loading (zero-copy where dtype matches, on-the-fly dequant otherwise, following llama.cpp's model-loading flow).

### 4.3 The graph executor

Rather than re-tracing Python-style eager execution for every token (expensive for KV-cache-heavy decode loops), Grim builds a **static computation graph once per model shape class** (à la `ggml_cgraph`) and replays it with new inputs bound:

```rust
pub struct Graph {
    nodes: Vec<GraphNode>,
    inputs: HashMap<&'static str, NodeId>,
    outputs: Vec<NodeId>,
}

pub trait GraphBuilder {
    /// Called once at model load / shape-change time.
    fn build(&self, cfg: &Config, batch_shape: BatchShape) -> Result<Graph>;
}

pub struct Session {
    graph: Graph,
    device: Device,
    kv_cache: Option<Box<dyn KvCache>>,
    ssm_state: Option<Box<dyn SsmState>>,
}
```

Prefill (full prompt) and decode (single new token / chunk) get distinct graphs, cached by `(model_id, batch_shape_bucket)` so common shapes (e.g. batch sizes rounded to powers of two, à la vLLM's CUDA-graph bucketing) reuse compiled graphs instead of rebuilding per request.

### 4.4 Unified `Model` trait family

The hard design problem: transformers, Mamba, vision, audio, and diffusion models have genuinely different call shapes (token-in/token-out autoregressive vs. full-frame-in/full-frame-out denoising vs. encoder-decoder). Grim doesn't force them into one interface — it defines a **small core trait** plus **capability traits** that a model implements as applicable:

```rust
/// Every model implements this. It says nothing about modality.
pub trait Model: Send + Sync {
    fn config(&self) -> &dyn ModelConfig;
    fn device(&self) -> &Device;
    fn param_arith(&self) -> ArithType;  // arithmetic type for computation
}

/// Autoregressive, token-level generation — dense transformers, Mamba, hybrids.
pub trait CausalLm: Model {
    fn new_session(&self) -> Box<dyn Session>;
    fn forward(&self, session: &mut dyn Session, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor>; // logits
}

/// Sequence state models need an explicit state cache instead of KV blocks.
pub trait StatefulSequence: Model {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState>;
    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor>;
}

/// Non-autoregressive encoders — vision towers, CLIP, audio encoders.
pub trait Encoder: Model {
    fn encode(&self, input: &Tensor) -> Result<Tensor>; // e.g. patch embeddings, mel features
}

/// Encoder-decoder, cross-attention conditioned generation — Whisper-style ASR.
pub trait EncoderDecoderLm: Model {
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
    fn decode_step(&self, session: &mut dyn Session, encoder_out: &Tensor, input_ids: &Tensor) -> Result<Tensor>;
}

/// Iterative denoising models — UNet/DiT diffusion.
pub trait DiffusionModel: Model {
    fn denoise_step(&self, latents: &Tensor, timestep: &Tensor, cond: &Tensor) -> Result<Tensor>; // predicted noise/velocity
    fn scheduler(&self) -> &dyn NoiseScheduler;
}
```

A hybrid architecture (e.g. a Jamba-style Mamba+attention model) just implements both `CausalLm` and internally mixes `StatefulSequence` layers with attention layers inside its own `forward` — the trait boundary is at the *request* level, not forced down into every layer.

Every `CausalLm` served by Grim is, by default, actually a `SpeculativeCausalLm` — a transparent wrapper adding speculative decoding via whichever strategy applies (native MTP heads or an attached DSpark draft bundle, §5.3). Callers of `CausalLm::forward` never see this; it's chosen at model-load time based on what the model supports and what's attached, and can be disabled per-model or per-request.

`grim-scheduler` and `grim-server` branch on **which capability traits a loaded model implements** to decide what request types it can serve, rather than hard-coding a model-family enum — so a new modality plugin just needs to implement one of these traits (or its own, via the plugin extension point in §6).

---

## 5. Memory management & scheduling (vLLM-inspired)

### 5.1 Paged KV cache

```rust
pub struct BlockId(u32);
pub const BLOCK_SIZE: usize = 16; // tokens per physical block, matches vLLM default

pub struct KvBlockPool {
    physical_blocks: Vec<KvBlockStorage>,   // pre-allocated GPU/CPU buffers
    free_list: VecDeque<BlockId>,
    ref_counts: HashMap<BlockId, u32>,      // >1 under prefix caching / beam search
}

pub struct BlockTable {
    logical_to_physical: Vec<BlockId>,
}

pub trait KvCache: Send {
    fn append_slot(&mut self, pool: &mut KvBlockPool) -> Result<()>;
    fn block_table(&self) -> &BlockTable;

    /// Speculative-decoding support (see §5.3): write draft-token KV entries
    /// provisionally, then either commit the accepted prefix or roll the
    /// tail back off before the next iteration. Blocks holding only rejected
    /// tail entries return to `KvBlockPool.free_list` untouched by the target
    /// model's own accounting.
    fn tentative_append(&mut self, pool: &mut KvBlockPool, n: usize) -> Result<()>;
    fn commit(&mut self, accepted_len: usize) -> Result<()>;
    fn rollback_to(&mut self, len: usize) -> Result<()>;
}
```

This is a direct structural analogue of vLLM's block-table-over-physical-blocks design: sequences address KV memory through a logical table, physical blocks are allocated/freed from a shared pool, and identical prompt prefixes across requests can share ref-counted blocks (prefix caching).

**SSM/Mamba state is not paged the same way** — Mamba's recurrent state is a small, fixed-size tensor per sequence (not O(sequence length) like KV cache), so `grim-memory` gives it a separate, much simpler `SsmStatePool` that just allocates/frees fixed-size slots. This is one of the reasons Grim doesn't force a single cache abstraction across model families. It also means SSM rollback for speculative decoding (§5.3) is a cheap state-snapshot/restore rather than a block-table truncation — see the caveat in §5.3 on why Mamba speculation ships later than transformer speculation.

Physical blocks in `KvBlockPool` don't have to hold full-precision tensors — §5.4 covers compressing block contents in place to buy back capacity, and it's this same pool/block-table structure that compression operates on.

### 5.2 Scheduler (`grim-scheduler`) — with admission control for latency guarantees

Iteration-level / continuous batching, three queues, matching vLLM's model, plus a latency-aware admission gate:

```rust
/// Tells the scheduler whether to admit, defer, or reject an incoming
/// request based on its predicted impact on time-to-first-token (TTFT).
/// This is what prevents a single long-prompt request from silently
/// degrading every other user's response time — even on single-GPU
/// hobbyist setups where there's no SLA contract, just a user who
/// closes the app when it feels slow.
pub struct AdmissionController {
    /// Maximum acceptable TTFT in milliseconds.
    /// Default: 2000 — most users perceive >2 s as sluggish.
    /// Set to 0 to disable admission control entirely.
    target_ttft_ms: u64,
    /// Maximum acceptable per-token decode latency in milliseconds.
    /// Default: 100 — keeps streaming output feeling fluid.
    /// Set to 0 to skip per-token checks during decode.
    target_itl_ms: u64,
    /// Running throughput estimate, in tokens/second, measured from
    /// actual iteration timings — self-calibrating, no manual config.
    throughput_estimate: Arc<AtomicF64>,
}

impl AdmissionController {
    /// Predicts prefill TTFT for a request given its prompt length and
    /// the current batch's already-committed backlog of prefill tokens.
    ///
    /// Model: `predicted_time = (batch_backlog + prompt_tokens) / throughput`
    /// This is deliberately simple — linear in token count, no quadratic
    /// attention-term correction. On GPU backends, prefill is memory-bound
    /// enough that the linear approximation is within ~30% of wall-clock
    /// for common prompt sizes (up to ~32k tokens). The source of truth
    /// is not the prediction model, but the correction: after each completed
    /// prefill, the controller compares predicted vs. actual wall time and
    /// updates the throughput estimate via an exponential moving average.
    fn predict_ttft(&self,
        prompt_tokens: usize,
        batch_token_backlog: usize,
    ) -> Duration {
        let total = batch_token_backlog + prompt_tokens;
        let rate = self.throughput_estimate.load(Ordering::Relaxed);
        Duration::from_secs_f64(total as f64 / rate.max(1.0))
    }

    /// Admission decision for an incoming request, evaluated once per
    /// engine tick before `schedule()` admits from the waiting queue.
    fn admit(&self,
        request: &Request,
        backlog: &BatchTokenBacklog,
    ) -> AdmissionDecision {
        if self.target_ttft_ms == 0 {
            return AdmissionDecision::Admit;   // gated feature, no-op when off
        }
        let predicted = self.predict_ttft(request.prompt_tokens(), backlog.total());
        if predicted.as_millis() as u64 <= self.target_ttft_ms {
            AdmissionDecision::Admit
        } else {
            AdmissionDecision::Defer  // stay in waiting queue, check next tick
        }
    }

    /// Called after each completed prefill to record actual wall time.
    /// Updates the throughput EMA so the model self-calibrates without
    /// operator tuning.
    fn observe_prefill(&self,
        prompt_tokens: usize,
        wall_duration: Duration,
    ) {
        let measured_tps = prompt_tokens as f64 / wall_duration.as_secs_f64();
        const EMA_ALPHA: f64 = 0.3;
        self.throughput_estimate
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed,
                |current| Some(current * (1.0 - EMA_ALPHA) + measured_tps * EMA_ALPHA))
            .ok();
    }
}

pub struct Scheduler {
    waiting: VecDeque<Request>,   // not yet admitted (prefill pending)
    running: Vec<Request>,        // actively decoding this iteration
    swapped: VecDeque<Request>,   // preempted, KV evicted to host memory or dropped
    block_pool: Arc<Mutex<KvBlockPool>>,
    max_batched_tokens: usize,
    max_num_seqs: usize,
    admission: AdmissionController,  // latency-aware admission gate
}

impl Scheduler {
    /// Called once per engine iteration. Decides what runs this step.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // 0. Pre-admission: for each request in `waiting`, ask the
        //    AdmissionController whether admitting it this tick would
        //    violate the TTFT budget. Deferred requests stay in
        //    `waiting` and are re-evaluated next tick — the controller
        //    may admit them once the current backlog drains.
        let backlog = self.compute_token_backlog();
        self.waiting.retain(|r| {
            match self.admission.admit(r, &backlog) {
                AdmissionDecision::Admit => true,   // move on to step 1
                AdmissionDecision::Defer => false,  // stays in waiting
            }
        });
        // 1. Try to give every `running` sequence its next KV slot.
        //    If the pool is full, preempt lowest-priority running seqs
        //    (swap or drop).
        // 2. Admit from `waiting` up to max_batched_tokens / max_num_seqs
        //    budget, preferring chunked prefill for very long prompts
        //    (Sarathi-Serve style stall-free scheduling).
        // 3. Return the batch descriptor for the execution engine to run.
    }

    fn compute_token_backlog(&self) -> BatchTokenBacklog {
        // Sum of remaining prefill tokens for all waiting requests,
        // plus unprocessed prefill tokens for requests currently in
        // chunked-prefill state.
    }
}
```

Design notes carried over deliberately from vLLM:
- **Iteration-level scheduling**: batch membership is re-decided every forward pass, not per-request — this is what keeps GPU utilization high under mixed-length traffic.
- **Preemption** under memory pressure, lowest-priority-first, with a swap-to-host or recompute-from-scratch fallback.
- **Chunked prefill** so a single huge prompt doesn't starve decode-latency-sensitive requests sharing the batch.

New differences driven by the latency-aware admission gate:
- **Self-calibrating throughput model**: the `AdmissionController` learns the backend's actual tokens/second from real prefill timings — no manual benchmark, no hardware-specific config. A hobbyist on a laptop ROCm iGPU gets the same latency behavior as a datacenter MI300X, just slower throughput, because admission scales to the hardware's actual capability.
- **Default TTFT target of 2 seconds**: tuned for human perception of responsiveness. A user pasting a 10k prompt into a chat interface sees results start within 2 seconds, or their prompt was explicitly deferred (the request doesn't silently queue). This is the same latency bar web apps set for page loads, applied to LLM inference.
- **Requests are deferred, not rejected**: admission control never drops work — it defers requests whose estimated TTFT exceed the budget. The request stays in `waiting` and is re-evaluated each tick. Under steady load, the controller automatically spaces admissions so no individual request starves.
- **Per-token ITL check during decode**: once a request is admitted and enters decode phase, the controller also caps inter-token latency. If a dense decode batch slows per-token throughput below the `target_itl_ms` budget, the controller defers new prefill admissions for the next tick, protecting the streaming feel of already-running requests — the user who's watching tokens appear doesn't get a stall because a new request arrived.

The admission controller is **always active** (including minimal threshold defaults) but its penalty is low: on an idle engine the model predicts zero backlog so every request is admitted immediately. The gate only engages when the engine is under enough load that admitting one more request would degrade everyone else's responsiveness — which is exactly when hobbyists start noticing lag and leave.

For diffusion requests, "scheduling" means something different — there's no growing KV cache, but there is a fixed number of denoising steps and a request occupies a latent buffer for its full duration. `grim-scheduler` treats these as a separate request class with a simpler capacity-based admission policy (bounded by concurrent latent-buffer memory, not token budget). The admission controller applies its prefill TTFT prediction to the first denoising step (the "prefill" equivalent), and the per-token ITL check to the intermediate steps (the "decode" equivalent).

### 5.3 Speculative decoding (`grim-speculative`) — **default-on**

Grim bakes speculative decoding into the decode path by default for every `CausalLm` request: not an opt-in feature flag hunted down in docs, but the standard way tokens get generated unless a request or deployment explicitly turns it off. It supports **two strategies behind one abstraction**, chosen automatically per model, because they solve different deployment situations:

| Strategy | Source | Requires | Chosen when |
|---|---|---|---|
| **MTP** (Multi-Token Prediction) | vLLM's model-native speculation path (`docs/features/speculative_decoding/mtp.md`) | Nothing extra — the target checkpoint already carries trained prediction head(s) (DeepSeek-V3/Gemma-4-assistant style) and shares KV cache/trunk compute with the target | The loaded model exposes native MTP heads. Zero-config, so this is the **default fallback whenever it's available**. |
| **Confidence-scheduled semi-autoregressive (DSpark)** | DeepSeek-AI & Peking University, 2026 | A trained, attached draft bundle (draft backbone + Markov head + confidence head) | A draft bundle is attached to the target's manifest. Deeper drafts and load-adaptive verification generally beat MTP's fixed shallow depth, at the cost of needing a bundle trained for that specific target. |

Both are exact — rejection sampling guarantees the emitted token distribution is identical to what the target model would have produced decoding on its own, so choosing between them (or falling back to plain decode) never changes output quality, only speed.

**Selection priority at model load time:** attached DSpark bundle → native MTP heads → plain autoregressive decode. A model can have both; DSpark wins because its confidence scheduling generally yields a longer accepted length, but MTP remains available as an instant fallback if the bundle fails to load or is disabled.

#### 5.3.1 MTP — the zero-config path

```rust
/// Implemented directly by a target model that was trained with its own
/// multi-token-prediction head(s) — no separate draft model, no distillation
/// step. The MTP head(s) share the target's trunk computation and KV cache,
/// so speculating costs far less than running an independent draft model's
/// forward pass.
pub trait NativeMtp: CausalLm {
    /// How many extra tokens the model can natively predict ahead in one
    /// pass — vLLM's `num_speculative_tokens`, typically small (1 is a
    /// reasonable default to start with).
    fn mtp_depth(&self) -> usize;

    /// Runs the trunk once and returns predictions for the next
    /// `mtp_depth()` positions, reusing the same KV cache entries the
    /// target forward pass would have written anyway.
    fn predict_multi(&self, session: &mut dyn Session, input_ids: &Tensor, positions: &Tensor) -> Result<DraftBlock>;
}
```

Because MTP heads are trained jointly with the trunk and share its KV cache, there's no separate draft-model forward pass to schedule or route — `grim-scheduler` just asks the model for `mtp_depth()` extra speculative positions on every decode step it already runs. This is the simplest possible speculative path and the reason it's the default for any model that supports it: it needs no bundle, no distillation run, and no extra memory pool.

#### 5.3.2 DSpark — the enhanced, bundle-based path

**Why reach for this over MTP when a bundle is available:** ordinary parallel drafters (propose a whole block in one forward pass, à la DFlash-style drafters) are cheap but suffer "suffix decay" — later positions in the block are guessed without knowing what was actually chosen for earlier positions, so acceptance falls off fast toward the tail. Fully sequential drafters avoid that but lose the parallel speed advantage. DSpark's semi-autoregressive design keeps both: an O(1) parallel backbone for the whole block, corrected by a lightweight sequential module that's just expensive enough to fix intra-block dependencies and not more — and its confidence-scheduled verifier adapts block depth to load, which MTP's fixed `mtp_depth()` doesn't do.

```rust
/// Parallel backbone: one forward pass over a block-sized draft window,
/// producing base logits for every position simultaneously — this is what
/// keeps drafting O(1) instead of O(block_len).
pub trait DraftBackbone: Send + Sync {
    fn draft_block(&self, session: &mut dyn Session, context: &Tensor, block_len: usize) -> Result<DraftBlock>;
}

pub struct DraftBlock {
    pub tokens: Vec<u32>,
    pub base_logits: Tensor,   // parallel-backbone logits per position
    pub confidence: Vec<f32>,  // per-position acceptance-probability estimate
}

/// Lightweight sequential correction — a low-rank (rank-256 in DSpark's own
/// config) prefix-conditioned bias applied to the base logits before each
/// in-block token is sampled. This is the "semi" in semi-autoregressive:
/// still one backbone pass, but each position now depends on the tokens
/// already chosen earlier in the same block, which is what suppresses
/// suffix decay.
pub trait MarkovHead: Send + Sync {
    fn bias(&self, prefix_within_block: &[u32], base_logits: &Tensor) -> Result<Tensor>;
}

/// Trained jointly with the drafter against the target model's own
/// acceptance statistics — predicts, per position, the probability that a
/// drafted token survives verification against the target.
pub trait ConfidenceHead: Send + Sync {
    fn score(&self, draft_block: &DraftBlock) -> Vec<f32>;
}

pub struct SpeculationConfig {
    pub block_len: usize,           // production default: 5 (DSpark-5)
    pub min_verify_len: usize,      // floor — always verify at least this many
    pub confidence_floor: f32,      // positions below this score aren't offered to the target
}
```

**Confidence-scheduled verification** is the second half of DSpark's contribution, and it's a serving-system concern, not just a modeling one: verifying every drafted position indiscriminately wastes target-model batch capacity on tail tokens that were unlikely to be accepted anyway, and that waste gets worse exactly when the engine is already under load.

```rust
pub struct ConfidenceScheduler {
    throughput_profile: ThroughputProfile,  // profiled verify-cost curve for this backend/hardware
}

impl ConfidenceScheduler {
    /// Called once per engine iteration, per drafted sequence, before the
    /// batched target-verification pass. This is the load-adaptive knob:
    /// under light load it extends verification toward the full block to
    /// maximize accepted length; under heavy load it verifies only the
    /// high-confidence prefix and drops the low-confidence tail *before*
    /// it ever reaches the target model, protecting other requests' share
    /// of this iteration's batch.
    pub fn choose_verify_len(
        &self,
        draft: &DraftBlock,
        live_gpu_utilization: f32,
        batch_pressure: usize,
    ) -> usize {
        // Walk the confidence-ranked prefix; keep extending verification
        // while marginal survival probability still clears the throughput
        // headroom implied by `throughput_profile` at the current
        // `live_gpu_utilization`. Never drop below `min_verify_len`.
    }
}
```

**Verification** itself is standard speculative-sampling rejection sampling against the target model's exact softmax: accepted tokens are free (no extra target forward pass consumed beyond the one batched verification step); the first rejected position is resampled from the corrected residual distribution, and everything after it is discarded and re-drafted next iteration. `grim-scheduler` runs this as an extra phase per iteration — draft, score, choose verify length, batch-verify against the target, commit-or-rollback KV — rather than as a separate code path bolted onto decode.

**Integration points:**
- **`grim-scheduler`**: the iteration loop gains a draft→score→verify phase ahead of the existing admit/preempt logic in §5.2, regardless of which strategy is active; `choose_verify_len` consumes the same load signals (`batch_pressure`, live utilization) the scheduler already tracks for preemption decisions, so the two are co-tuned rather than independent. For MTP, `verify_len` is simply capped at `mtp_depth()` since there's no confidence-ranked tail to trim — the load-adaptive scheduler still decides *whether* to spend the verification pass at all under extreme pressure, just not how deep.
- **`grim-memory`**: draft-token KV entries are written via `KvCache::tentative_append`, then either `commit`-ed up to the accepted length or `rollback_to`-ed — see §5.1's `KvCache` trait. This is identical for both strategies; MTP's KV writes just happen to be the same writes the target forward pass needed anyway, so there's no separate draft-cache pool to manage.
- **`grim-core`**: a `SpeculativeCausalLm` wrapper composes a target `CausalLm` with whichever strategy applies — either the model's own `NativeMtp` impl, or an external `DraftBackbone` + `MarkovHead` + `ConfidenceHead` bundle — and exposes the *same* `CausalLm` interface outward, so nothing above `grim-core` (scheduler, server) needs to know speculation is happening at all, let alone which strategy, it's an implementation detail of how a `CausalLm` fulfills `forward()`.
- **`grim-format`**: a DSpark draft model, its Markov head, and its confidence head ship as a companion component file attached to the target model's manifest — the same "small companion artifact next to the big model" pattern already used for vision/audio `mmproj`-style components in §7, just for the decode loop instead of multimodal encoders. MTP heads need no companion file at all — they're already part of the target checkpoint's own weights.
- **`grim-cli`**: `grim spec train --target <model> --draft <arch>` drives a DeepSpec-style distillation pipeline to produce a DSpark draft/Markov/confidence bundle for a target checkpoint that doesn't ship with native MTP heads.

**Caveats worth carrying into the roadmap, not glossing over:**
- Every DSpark number above (26.7–30.9% higher accepted length vs. Eagle3, 16.3–18.4% vs. DFlash, 60–85% per-user latency improvement on V4-Flash) is DeepSeek's own reported benchmark on their own infrastructure as of the June 2026 release; there's no independent third-party reproduction yet, so Grim should treat these as directional, not as guaranteed numbers on Grim's own kernels/hardware.
- **A DSpark draft bundle has to be trained per target checkpoint** — this is not a zero-shot technique. `grim-speculative` needs a clean "no bundle, no native MTP" fallback (skip speculation, decode normally) rather than assuming one always exists, especially for community/plugin-supplied models. MTP doesn't have this problem, but only covers the handful of model families that ship MTP-trained checkpoints (per vLLM's docs: model families with native MTP support only — everything else needs EAGLE-, draft-model-, or DSpark-style speculation instead).
- Both techniques as published target dense/MoE **transformer** decode. Applying either to `StatefulSequence` (Mamba/hybrid) targets requires a different drafting strategy — the state-recurrence pattern doesn't support the KV-block-rollback that transformer speculation uses, and neither DSpark's tree-attention nor MTP's shared-trunk approach has been validated for SSMs. However, a simpler and well-understood path exists: **train a small draft *transformer* model for the Mamba target** (Medusa/EAGLE-style), where the draft is a separate model, not the SSM itself speculating. The draft model (a tiny transformer, 10-20% of the target's parameter count) learns to predict the target Mamba's output distribution via distillation — a known process with existing tooling. It doesn't need to understand SSM state; it just needs to be right often enough to be accepted at the usual speculative-sampling rate. This trades the research risk (making DSpark work on SSM states) for engineering cost (training a transformer draft for each Mamba target), which is a solvable pipeline problem rather than an open research question. **Ship transformer speculation in phase 5; commit Mamba (SSM) speculation to phase 7.5** — immediately after model breadth lands in phase 7 — starting with the draft-model approach and reassessing if DSpark-on-SSM research matures.
- Reported acceptance can degrade over long multi-turn context as a DSpark draft model's approximation of the target's distribution drifts further from its training distribution — worth a monitored fallback (auto-shrink `block_len`, drop to MTP if available, or disable speculation entirely) rather than a fixed config assumed to hold forever.

### 5.4 Runtime KV cache compression (`grim-kvquant`, TurboQuant-inspired)

This is a different quantization axis from §7's weight quantization: `grim-quant` shrinks the *model's* weights on disk; `grim-kvquant` shrinks the *cache* built up during serving, at runtime, inside `grim-memory`'s block pool. It follows TurboQuant (0xSero/turboquant, ICLR 2026): random orthogonal rotation to spread information across dimensions, Lloyd-Max optimal scalar quantization on the rotated (Beta-distributed) values, QJL projection for residual sign bits on keys, and group quantization with per-group scale/zero and bit-packing for values — with an unbiased inner-product estimator, so attention scores computed against compressed KV stay statistically correct rather than silently biased.

```rust
/// Compresses/decompresses the contents of physical KV blocks in place.
/// Sits inside grim-memory's KvBlockPool (§5.1) as an optional storage
/// tier, not a replacement for it — the block table / free list /
/// ref-counting machinery is unchanged, only what a physical block
/// actually holds changes.
pub trait KvCompressor: Send + Sync {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock>;
    fn dequantize_for_attention(&self, block: &CompressedKvBlock) -> Result<(Tensor, Tensor)>;
}

pub struct KvQuantConfig {
    pub key_bits: u8,     // production default: 3
    pub value_bits: u8,   // 2-bit for max capacity, 4-bit for quality-sensitive workloads
    pub group_size: usize,
}
```

**Applicability is per attention layer, not global**, and this maps cleanly onto a distinction Grim already makes: `grim-kvquant` only compresses `KvBlockPool` entries for genuine full-attention layers. Linear-attention and Mamba/SSM layers already live in the separate, fixed-size `SsmStatePool` (§5.1) and aren't touched by this at all — which is exactly TurboQuant's own reported limitation (linear-attention/Mamba state isn't compressible by this method), so Grim's existing pool separation is a good fit rather than something that needs bridging.

**What this buys, and what it doesn't** — carrying TurboQuant's own adversarial self-audit forward rather than only the headline numbers:
- **Capacity, not raw compute speed**: freed KV memory converts into more concurrent sequences or longer max context (TurboQuant reports roughly 2x token capacity on a pure dense model, less on hybrid/MoE architectures where full-attention layers are a minority of total KV). Prefill/decode throughput deltas were reported as within noise on the source project's own benchmarks — this is a memory-capacity feature first, a raw-speed feature second.
- **Value quantization is the quality bottleneck, not key quantization**: 3-bit key compression measured near-lossless (cosine similarity ≈1.0) against full precision, but 2-bit value quantization measurably degrades output (cosine similarity ≈0.94), while 4-bit values stay close to lossless (≈0.997). `grim-kvquant`'s default config should bias toward 4-bit values unless a deployment explicitly opts into 2-bit for maximum capacity and has validated quality on its own workload.
- **No free lunch on compute during decode**: dequantizing compressed KV back to full precision for the attention computation itself costs real work per decode step; the win is memory capacity, and realizing a *compute* win too requires the fused compressed-attention kernels (grim-backend-cuda's equivalent of TurboQuant's Triton kernels) actually being wired into the decode path rather than compress-then-dequantize-to-float32-then-attend, which only saves memory, not compute.
- **This is genuinely new capacity, not automatically "free"**: like TurboQuant's own honest audit of its headline compression-ratio claims, Grim's docs/benchmarks for `grim-kvquant` should report measured, workload-specific numbers rather than a single generic compression ratio, since the achievable ratio depends heavily on how much of a given model's KV is full-attention vs. linear-attention/SSM.

**Interaction with §5.3 speculative decoding**: compressed KV blocks still support `tentative_append`/`commit`/`rollback_to` — speculative draft tokens get written into the same compressed representation as committed tokens, so the two features compose rather than requiring separate code paths. The dequantize-for-attention step simply runs once per verification batch either way.

---

## 6. Plugin system (`grim-plugin`)

Goals: let third parties add model architectures, tokenizers/processors, samplers, and backends, **without recompiling Grim**, and **without one bad plugin taking down the engine**.

### 6.1 Two loading strategies, chosen per plugin

| Strategy | Use for | Isolation | Cost |
|---|---|---|---|
| **Dynamic library (`libloading` + stable C ABI)** | Performance-critical extensions: new backend kernels, new model forward passes | Process-shared memory — a crash takes the engine down, so this tier is for trusted/first-party-reviewed plugins | Near-native speed |
| **WASM component (`wasmtime`)** | Samplers, grammars/constrained decoding, pre/post-processors, tokenizers, moderation filters | Sandboxed — fuel-limited, memory-limited, cannot touch host memory or make syscalls outside a granted capability set | Slower (fine for logic that runs once per token or once per request, not per-matmul) |

The rule: **anything on the hot path that operates on raw tensors (a new kernel, a new architecture's forward pass) is a dylib plugin; anything on the control path (a sampler decision, a grammar constraint, a post-processing step) is a WASM plugin by default**, with dylib available as an opt-in for performance if a developer accepts the isolation trade-off.

### 6.2 Plugin ABI

```rust
/// Stable, `#[repr(C)]` ABI boundary — this is what dylib plugins implement.
/// (Rust trait objects aren't ABI-stable across compiler versions, so the FFI
/// boundary uses a C-compatible vtable, similar to how `abi_stable`/`stabby`
/// crates solve this problem.)
#[repr(C)]
pub struct GrimPluginVTable {
    pub abi_version: u32,
    pub name: extern "C" fn() -> *const c_char,
    pub capabilities: extern "C" fn() -> PluginCapabilities,
    pub init: extern "C" fn(ctx: *mut EngineContext) -> i32,
    pub model_factory: Option<extern "C" fn(cfg: *const c_char) -> *mut c_void>,
    pub sampler_factory: Option<extern "C" fn() -> *mut c_void>,
    pub teardown: extern "C" fn(),
}

bitflags! {
    pub struct PluginCapabilities: u32 {
        const MODEL_ARCHITECTURE = 1 << 0;
        const BACKEND            = 1 << 1;
        const SAMPLER            = 1 << 2;
        const TOKENIZER          = 1 << 3;
        const PRE_POST_PROCESSOR = 1 << 4;
    }
}
```

WASM plugins implement an equivalent surface via a WIT interface (component model), e.g.:

```wit
package grim:plugin@1.0.0;

interface sampler {
  record sampler-config { params: list<tuple<string, string>> }
  resource sampler {
    constructor(config: sampler-config);
    sample: func(logits: list<f32>, history: list<u32>) -> u32;
  }
}
```

### 6.3 Manifest & discovery

```toml
# plugin.grim.toml
[plugin]
name = "grammar-constrained-json"
abi_version = 1
kind = "wasm"                     # or "dylib"
capabilities = ["sampler"]
entry = "grammar_json.wasm"

[plugin.limits]                   # only meaningful for kind = "wasm"
fuel = 5_000_000
max_memory_mb = 64
```

`grim-plugin` scans a plugins directory at startup, validates `abi_version` compatibility, and registers each plugin's declared capabilities into `grim-core`'s registries (model factory registry, sampler registry, etc.) so the rest of the engine looks up "give me the factory for architecture `mamba2`" without knowing whether it's built-in or plugin-provided.

### 6.4 Failure isolation

- WASM plugins: fuel metering + memory limits enforced by wasmtime; a panic or fuel exhaustion returns an error to the calling request, not a process crash.
- Dylib plugins: run their `forward`/`sample` calls behind `catch_unwind` at minimum; longer-term, performance-critical dylib plugins that need real fault isolation can be run in a supervised subprocess with a shared-memory tensor handoff (opt-in, since it costs a copy or a `memfd`/IPC round trip).

---

## 7. Quantization & model format (`grim-quant`, `grim-format`)

Grim has to handle two genuinely different provenances of "quantized weights," and conflating them is a real correctness risk, not just a style choice — a checkpoint that was already quantization-aware-trained should never get silently re-quantized by Grim's own post-training pipeline, since stacking two rounds of quantization error compounds loss for no benefit.

### 7.1 Grim-produced post-training quantization

- **On-disk format**: GGUF-compatible container — single file, metadata (architecture, hyperparameters, tokenizer, special tokens) + tensor data, memory-mappable for zero-copy loading, exactly as llama.cpp does it. Grim reads GGUF directly (interop with the existing quantized-model ecosystem) and also reads safetensors (interop with the HF/Candle ecosystem) via `grim-format`'s two loaders behind one `TensorProvider` trait.
- **Quantization**: block-wise quantization, blocks of 32 values, per-block scale (+ min for asymmetric formats), mirroring Q4_K/Q5_K/Q8_0. Sensitive tensors (embeddings, final norm) default to higher precision or full dequantization on load, same reasoning llama.cpp uses.
- **`grim-quant` also carries importance-matrix (imatrix) support**: an optional calibration pass that weights quantization error by activation magnitude, improving low-bit quality — same idea as llama.cpp's `--imatrix`. This is still round-to-nearest-style post-training quantization: fast, no training loop, but it hits a quality wall at 2–3 bits that training-aware methods don't.

### 7.2 Ingesting quantization-aware-trained (QAT) checkpoints — EfficientQAT-compatible

A separate, growing class of models are shipped already quantized *through training*, not after it — EfficientQAT (OpenGVLab, ACL 2025) is the reference case: a two-phase pipeline (block-wise training of all parameters, then end-to-end training of just the quantization parameters) that pushes uniform INT weight-only quantization down to 2-bit while staying meaningfully closer to full-precision accuracy than round-to-nearest quantization does at the same bit-width. These arrive as **grouped INT weights** — `wNgM` naming (e.g. `w4g128`, `w3g128`, `w2g64`): N-bit weights, per-group scale and per-group zero-point (EfficientQAT is always **asymmetric**: uint `[0, 2^N - 1]` range, never symmetric), group size M — a different tensor layout from Grim's own block-32 K-quant scheme, and one Grim should read natively rather than force through its own quantizer.

**Format version distinction**: EfficientQAT checkpoints come in two forms. The *native training format* saves `QuantLinear` modules with trainable `nn.Parameter` scales. The **GPTQ v2 transfer format** (what EfficientQAT publishes on HuggingFace as `*‑GPTQ`) freezes scales and zero-points into inference-ready buffers with the `qweight`/`qzeros`/`scales`/`g_idx` layout. Grim ingests the **GPTQ v2 transfer format** only — training-format checkpoint loading is not a target.

**Per-weight-tensor provenance**: The `QuantProvenance` enum lives in `grim-tensor` (§4.1) and every `Tensor` carries its own `provenance` field, resolved from the checkpoint's per-tensor metadata at load time by `WeightSource::get()` (§4.2). This means a single model can mix e.g. FP16 embeddings + w4g128 attention + w3g128 MLP layers, each with an independent `QuantProvenance::ExternalQat` config, without any model-level override. The `Storage::GroupInt` variant in the `DType` struct mirrors this: the dequant kernel is selected per tensor based on `(tensor.provenance, tensor.dtype.storage)`, not from a model-global flag. This design was chosen to avoid a painful refactor when mixed-precision QAT models — already common in the EfficientQAT model zoo — become the norm.

- **Loader**: `grim-format` gains a GPTQ-tensor-layout reader (`qweight`/`qzeros`/`scales`/`g_idx` naming, the layout EfficientQAT itself transfers checkpoints into for downstream compatibility via GPTQModel) alongside the existing GGUF and safetensors loaders — same `TensorProvider` trait, so `WeightSource` (§4.2) doesn't care which of the three produced the file it's reading from.
    - **`g_idx` and `desc_act`**: EfficientQAT-transferred models have `desc_act=False`, meaning `g_idx` is a sequential group index (`[i // group_size for i in range(in_features)]`), not a column-permutation map as in classic GPTQ with activation ordering. The loader must read a `desc_act` flag from the checkpoint metadata (or infer it from the format version). If `desc_act=False`, `g_idx` is a group-assignment index only; if `desc_act=True`, `g_idx` is a permutation. The `ExternalQat` type's carry-over field should include a `desc_act: bool` discriminant, defaulting to `false` for EfficientQAT-ingested checkpoints.
- **Dequant/matmul kernel**: grouped INT weights need their own dequantize-and-matmul kernel per backend (distinct from the block-32 K-quant kernel), following the same shape GPTQ/BitBLAS kernels use — packed low-bit weights unpacked against a per-group scale/zero-point immediately before the matmul. This is a defined, bounded kernel surface (`grim-backend-*` gains one more op alongside `matmul`/`conv1d`/etc.), not a new execution model.
    - **Bit-widths**: 2, 3, 4, 8 (matching EfficientQAT's shipped `w2`, `w3`, `w4`). 3-bit packing uses 10 weights per int32 with 2 unused bits — a non-power-of-two stride — so the dequant kernel must handle irregular unpacking, unlike the 2-, 4-, and 8-bit paths which align to 16, 8, and 4 weights per int32 respectively.
- **Group size compatibility**: `g` values of 64 and 128 (EfficientQAT's shipped configurations) become supported group sizes in `grim-quant`'s reader; the group-size field is read from checkpoint metadata rather than assumed, so future group sizes from other QAT tools aren't a breaking change.
- **What Grim does *not* do**: retrain or fine-tune quantization parameters itself — Block-AP/E2E-QP-style QAT is a training-time technique that lives upstream of Grim (in a training framework, not an inference engine); Grim's job is correct, efficient ingestion and serving of the result, not reproducing the training pipeline.
- **Interaction with §5.3 speculative decoding**: EfficientQAT's asymmetric quantization (per-group zero-point, not round-to-nearest-symmetric) shifts the target's output distribution *more* than symmetric quant at the same bit-width. A DSpark draft/confidence bundle trained against a full-precision target will drift further from a w2g64 EfficientQAT target than from a symmetric w2g64 model. `grim spec train` must accept the exact quantized checkpoint (QAT or not) it will run against, not a proxy full-precision version. Draft bundles should be validated against the deployment target before shipping.

### 7.3 Shared packaging conventions

- **Multimodal components** (vision encoder, audio encoder, diffusion UNet) are packaged as separate GGUF-style files with a `component` metadata tag (mirroring llama.cpp's `mmproj` pattern) rather than crammed into one giant file — keeps quantization policy independent per component, since vision/audio towers are far more precision-sensitive than the LLM backbone.
- **Speculative decoding bundles** (draft backbone + Markov head + confidence head, §5.3) use the same `component`-tagged companion-file pattern as multimodal components — small, separately quantized, and optional: a target model manifest with no attached bundle simply runs without speculation.

---

## 8. Serving layer (`grim-server`)

- `axum`-based HTTP server, OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/images/generations`) plus a native streaming API (`grim-server` speaks both JSON-over-SSE and a binary gRPC/protobuf path for lower overhead).
- Each endpoint maps to a request type that `grim-engine` dispatches based on which `Model` capability traits the target model implements — a `/v1/chat/completions` request against a model that only implements `Encoder` returns a clear "model does not support this operation" error rather than a panic.
- Streaming: token-level SSE for LLMs, chunked partial-transcript streaming for ASR, and progress-callback streaming (denoise step N/T) for diffusion.

---

## 9. Request lifecycle (end-to-end)

```text
1. grim-server receives request → validates → builds Request { modality, params, priority, prompt_len }
2. grim-engine enqueues into grim-scheduler.waiting
2.5. Each tick, before schedule() admits from waiting:
       AdmissionController::admit() evaluates each queued request against
       the predicted TTFT budget (backlog + this request's prompt).
       Requests are deferred (stay in waiting) if they'd bust the budget
       — re-evaluated next tick. See §5.2 for the prediction model and
       self-calibrating throughput estimate.
3. Each engine tick:
...
     for CausalLm sequences with a speculation bundle attached (default path, §5.3):
       DraftBackbone + MarkovHead propose a block → ConfidenceHead scores it →
       ConfidenceScheduler picks a verify length under current load →
       target model batch-verifies → accepted tokens commit, rejected tail rolls back
     grim-core::Session executes the appropriate Graph on the batch via the backend
     grim-memory updates block tables / SSM state / diffusion latent buffers
       (blocks may be compressed in place via grim-kvquant if enabled, §5.4)
     sampled tokens (or denoised latents, or transcript chunks) flow back out
4. grim-server streams partial results to the client as they're produced
5. On completion: KV blocks released back to the pool (or retained if prefix-cacheable),
   session torn down, plugin teardown hooks fired if applicable
```

---

## 10. Build phases (suggested roadmap)

1. **Foundation**: `grim-tensor` + `grim-backend-cpu` + `grim-nn` — get a dense transformer running single-request, unbatched, F32, CPU only. Validate against a known-good Candle output for numerical parity.
2. **Format & quant**: `grim-format` (GGUF + safetensors) + `grim-quant` (Q8_0, Q4_K) — load real checkpoints, quantized and not.
3. **Serving core**: `grim-memory` (paged KV) + `grim-scheduler` with `AdmissionController` (latency-aware admission, §5.2) + `grim-server` minimal HTTP — multi-request throughput on transformers with bounded TTFT. Admission control ships as part of phase 3, not deferred — without it, the scheduler has no mechanism to prevent a long prefill from silently degrading every other request's latency.
4. **GPU backends**: `grim-backend-rocm` (primary GPU target via hip/rocBLAS), then `grim-backend-vulkan` (platform-agnostic shader-based compute) — same model code, new device. `grim-backend-cuda` and `grim-backend-metal` built if their respective feature flags are set, but they are not blocking delivery.
5. **Speculative decoding for transformers**: `grim-speculative` — start with `NativeMtp` support (zero-config, no distillation needed, immediately useful for MTP-trained checkpoints), then add the DSpark path (`DraftBackbone`/`MarkovHead`/`ConfidenceHead`, `ConfidenceScheduler`, the `grim spec train` distillation CLI) as the enhanced option. Ships as soon as dense-transformer serving is solid, since it's the default decode path, not a later add-on — a model with neither native MTP nor an attached bundle just falls back to normal decoding.
6. **KV cache compression**: `grim-kvquant` — TurboQuant-style compression for full-attention-layer KV blocks, defaulting to 4-bit values for quality, with fused compressed-attention kernels tracked as a follow-up once the compress/dequantize path is correct.
7. **Model breadth**: Mamba/SSM (`StatefulSequence` + `SsmStatePool`), vision (`Encoder`), audio (`EncoderDecoderLm`), diffusion (`DiffusionModel` + scheduler).
8. **Mamba speculative decoding**: small draft transformer model trained per Mamba target (Medusa/EAGLE-style distillation). Ships immediately after model breadth — without it, Mamba users decode at 1× while transformer users get 2-3× speculation speedup, creating a UX deficit that the architecture cannot otherwise close. Start with the draft-model approach (well-understood, existing tooling); reassess if DSpark-on-SSM research matures.
9. **Plugin system**: WASM sampler/processor plugins first (lower risk), then dylib model-architecture plugins.
10. **Hardening**: fault isolation, chunked prefill, prefix caching, multi-GPU tensor parallelism.
11. **Research track**: KV compression for whatever fraction of a hybrid model's state turns out to be compressible beyond today's full-attention-only scope — state-snapshot rollback is mechanically straightforward but compression has not been validated for SSMs; treat as exploratory rather than a committed default.

---

## 11. Trade-offs to revisit as Grim grows

- **Static ABI vtable vs. a full component-model boundary everywhere**: starting with C-ABI dylibs for performance plugins is simpler now but means Grim owns compatibility guarantees across its own version bumps; migrating hot-path plugins to WASM+SIMD once wasmtime's numerical performance closes the gap is worth revisiting.
- **One graph executor for all modalities**: diffusion's iterative denoising and Mamba's recurrent state stress the "build once, replay" graph model differently than transformer decode; if this abstraction leaks too much complexity, splitting into a `TokenGraph` vs `StepGraph` executor is a reasonable fallback. Note: per-step analysis of the single executor confirms it does NOT cause a Mamba/transformer latency deficit — both modalities use the same build-once pattern with external state buffers (KV cache vs. SSM state), and the executor's fusion passes are op-pattern-driven, not modality-driven. The Mamba deficit, if it exists, comes from the speculation timeline (phase 5 for transformers, phase 8 for Mamba), not from executor overhead.
- **Paged KV for everything vs. per-family memory strategy**: already split (KV pool vs. SSM state pool vs. diffusion latent buffers) — worth watching whether a fourth pattern emerges (e.g. video models) that needs its own pool type rather than being forced into one of the three.
- **Tensor-parallel / pipeline-parallel** multi-GPU is deliberately out of scope for the initial design above; it changes `Session` and the scheduler's placement logic non-trivially and deserves its own design pass once single-GPU serving is solid.
- **Default-on speculative decoding vs. simplicity**: making `SpeculativeCausalLm` the default wrapper for every `CausalLm` buys real latency wins but adds a mandatory draft→score→verify phase to the hot loop and a rollback contract every `KvCache` implementation must honor correctly — worth re-examining if it turns out to complicate backend or plugin development more than the throughput gain justifies for smaller deployments (an escape hatch — load a model with no attached draft bundle and no native MTP heads, or set `speculation: off` — should always exist). It's also worth re-benchmarking DSpark's own reported gains against Grim's kernels directly once phase 5 lands, rather than assuming DeepSeek's numbers transfer unchanged.
- **KV compression as a default-off, opt-in tier**: unlike speculative decoding, `grim-kvquant` is deliberately *not* default-on — it's a memory/quality trade-off a deployment should choose deliberately (particularly the 2-bit-vs-4-bit value question), not one Grim should silently make on someone's behalf. Revisit this stance once fused compressed-attention kernels close the compute gap and the quality trade-off at 4-bit values is validated broadly enough to be a safe default.

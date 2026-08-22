# grim vs. Unsloth, Ollama, vLLM, SGLang, LlamaFactory, Axolotl

**Method:** Every claim below was verified against source at grim commit `de52ff0d` (2026-08-22, "add vulkan and metal numerical tests and harden training and endpoints") and against the competitor snapshots in `old/repos/`. Nothing is asserted from README marketing copy; claims that could not be confirmed from source are marked `UNVERIFIED`. Per review scope, maturity and user-base size are excluded as evaluation criteria — this compares capability, design, and code only. This revision replaces all earlier revisions of this document, including their superseded findings.

---

## 1. What each tool actually is

| Tool | Core identity | Engine origin |
|---|---|---|
| **grim** | Pure-Rust, from-scratch inference **and** fine-tuning engine, ROCm-primary | Native, written from scratch (no PyTorch, no PyO3) |
| **Ollama** | Model management/serving product | Go orchestration that builds **upstream llama.cpp** (`LLAMA_CPP_VERSION=b10091`) via FetchContent into a `llama-server` binary and proxies HTTP to one subprocess per loaded model (`llm/llama_server.go`), plus a 105-line compat hook patch. The old vendored ggml fork is gone. Two first-party stacks exist: a Go MLX runner (`x/mlxrunner`, own prefix-cache trie + MTP drafting) and an experimental Flux-architecture image generator (`x/imagegen`) |
| **vLLM** | Production serving engine | Python orchestration + C++/CUDA/HIP kernels + a **first-party Rust frontend** (`rust/`: server, tokenizer, parser, chat — 306 `.rs` files bridged via PyO3). The original PagedAttention CUDA kernel no longer exists; its own design doc is self-flagged historical. Block-table paged KV management remains the substrate |
| **SGLang** | Production serving engine + generator DSL | Native (Python/C++/CUDA/HIP + substantial Rust gateway/grpc components). RadixAttention is central |
| **Unsloth** | Fine-tuning acceleration library | Import-time monkey-patching of HF `transformers`/`peft`/`trl`/bitsandbytes with hand-written Triton kernels and manual autograd; requires `unsloth_zoo` as a hard runtime dependency |
| **LlamaFactory** | Unified fine-tuning framework | Wraps HF `transformers`/`peft`/`trl`; typed config over six stages, huge model/template registry |
| **Axolotl** | Config-driven fine-tuning framework | Wraps HF stack; YAML schema-driven with plugin/integration system |

grim is the only tool here that is simultaneously a from-scratch serving engine *and* a from-scratch training stack with no PyTorch dependency anywhere. That position is genuinely unique in this set — and it is also why grim trails each specialist on that specialist's home turf (§7).

---

## 2. AMD positioning (RDNA vs. CDNA), checked

Counting files referencing consumer-RDNA targets (`gfx1030/1036/1100/1101/1102/1103/1150-1153/1200/1201`) vs datacenter-CDNA targets (`gfx906/90a/942/950/1250`):

| Engine | RDNA evidence | CDNA evidence | Reading |
|---|---|---|---|
| **grim** | **72 files** reference RDNA targets (up from 54 earlier in 2026); e.g. `fp8_gemm_rdna4.rs` has real `#if defined(__gfx1200__)`/`(__gfx1100__)` HIPRTC branches selecting distinct GEMM tiles; device-gated tests annotated "Verified via gfx1036 iGPU" | 18 files | RDNA-majority design center |
| **vLLM** | ~138 mentions across 16 files, including purpose-built RDNA3 W4A16 WMMA kernels (`csrc/rocm/q_gemm_rdna3_wmma.cu`, `rdna3_w4a16.py`) with dedicated compile-guard tests | ~524 mentions / 101 files (gfx950 alone: 363) | CDNA-majority, real but thinner RDNA investment |
| **SGLang** | **Zero** RDNA-specific code (0 files for any consumer target) | 73 files (gfx90a/942/950/1250) | CDNA-only |

Fair framing: Ollama consumers on Radeon are well served *inherited* support — llama.cpp's ggml HIP/Vulkan backends cover consumer GPUs broadly without Ollama writing RDNA-specific code; that is a different mechanism than grim's per-generation kernel work, not an absence. LlamaFactory's entire ROCm footprint is one Dockerfile (functionality rides on CUDA-compatible PyTorch, `UNVERIFIED` beyond that). Unsloth's ROCm support is now substantial (radeon.com-pinned wheel matrices including Windows ROCm, HIP branches throughout). Axolotl is NVIDIA-first with several perf paths (FP8 attention SM90+, FA3, NVFP4) CUDA-bound. LlamaFactory is the standout for a different non-CUDA platform: Ascend NPU support is real (`torch-npu` pins, fused NPU MoE/RMSNorm/RoPE kernels).

The ratio inversion is legitimate evidence for grim's "AMD-consumer-first design center" positioning. It is **not** evidence about relative quality or completeness of anyone's kernels.

---

## 3. Serving-engine comparison

### 3.1 KV-cache management and prefix caching

- **vLLM**: `KVCacheBlocks`/`BlockPool` with hash→block maps for prefix reuse + eviction (`v1/core/kv_cache_manager.py`, `block_pool.py`); 42 attention-backend files consuming updatable block tables (CUDA-graph compatible).
- **SGLang**: the deepest prefix-cache stack in the set — 30+ modules under `srt/mem_cache/`: the 863-line `RadixCache` plus SWA, hierarchical (`HiRadixCache` spilling to lmcache/mooncake/hf3fs/nixl/file/mmap/shm/aibrix/flexkv/umbp storage), Mamba-hybrid, bigram-keyed and salted variants, and a native C++ tree.
- **grim**: `grim-memory` implements a fixed-size block pool (`BLOCK_SIZE = 16`, refcounted free-list), per-layer page tensors, GPU↔HostRAM↔NVMe tiering (`SharedSpillManager`, `demote_cold_prefix`), and a block-granular radix trie modeled on RadixAttention. The live forward pass writes KV through `append_kv_layer`, which mirrors into the shared pool, so disaggregated transfers read real data. Wired end-to-end: prefill matches/promotes/inserts prefixes, decode consumes block tables via `paged_kv_handles`. **Prefix reuse is now default-ON** (opt out with `GRIM_RADIX=0|false|off`). Caveats: the KV pool itself remains host-resident f32 (device pages are staged copies); `paged_kv_handles` uses a device-resident cache fed by `append_kv_layer`, eliminating the former read-path re-upload, but the upload cost moved to the write path (once per layer per step).

Net: grim has a real, tested, wired paged-KV-plus-prefix-cache system in the same functional category as its competitors'. SGLang's stack is far broader; vLLM's is deeply integrated with CUDA graphs; grim's is narrower but complete for its feature set.

### 3.2 Schedulers

- **vLLM**: unified token-budget scheduler covering chunked prefill (default on), prefix caching, and speculation in one algorithm; piecewise cudagraphs/torch.compile.
- **SGLang**: zero-overhead overlap scheduler (default on), cache-aware scheduling policies (longest-prefix-match), two-batch/single-batch overlap.
- **Ollama**: inherits llama.cpp batching; VRAM-aware multi-model placement with eviction, but `OLLAMA_NUM_PARALLEL` defaults to 1, loads serialize through one active-loading slot, and VRAM estimation is file-size heuristics refined by regex-parsing llama-server logs (brittle by their own README's admission).
- **grim** (`grim-scheduler`): four queues (waiting/running/swapped/paused), TTFT/ITL-driven admission control with livelock bypass, Sarathi-style chunked prefill (default 512), priority preemption, pause/resume that keeps KV alive (exposed as HTTP endpoints), strict-determinism ordering, LoRA sub-batched output. No overlap scheduler / CPU-GPU micro-batch overlap equivalent.

### 3.3 Speculative decoding

- **SGLang**: eight builtins (EAGLE, EAGLE3, NEXTN/MTP, STANDALONE, NGRAM, DFLASH, DSPARK, FROZEN_KV_MTP) plus a runtime plugin registry with contract enforcement.
- **vLLM**: nine proposer implementations (EAGLE/EAGLE3, Medusa, GPU n-gram, suffix, draft-model, MLP speculator, DFlash, custom class).
- **Ollama**: exposes llama.cpp speculative drafting (Modelfile `DRAFT`, MTP auto-detection from GGUF metadata) without owning the machinery.
- **grim** (`grim-speculative`): speculative decoding is **default-ON** — `Engine::register_model` wraps every registered model in `SpeculativeCausalLm::auto` with strategy priority DSpark (draft backbone + Markov head + confidence head) > NativeMtp > Plain, VRAM-aware fallback under weight streaming, dynamic verification-length scheduling, Mamba state save/rollback for draft rejection, and acceptance-rate telemetry surfaced to Prometheus. Method breadth is much narrower than vLLM/SGLang: an `Eagle3Drafter` exists (with margin-based top1−top2 confidence) and `Engine::register_native_mtp_model` exists, but **neither has any production caller** — EAGLE3-as-drafter and native-MTP registration are implemented-but-unreachable seams awaiting wiring. A `lookahead_literal()` jump-forward helper was also added in `grim-constrain` with zero callers.

### 3.4 Structured/constrained output

- **vLLM**: xgrammar, guidance/llguidance, outlines, lm-format-enforcer backends (one per server); platform-gated grammar wheels.
- **SGLang**: xgrammar/outlines/llguidance plus jump-forward decoding (compressed-FSM `JumpEdge`) — the performance optimization grim's approach lacks.
- **Ollama**: JSON-schema forwarded to llama-server grammar engine; bare `format=json` via embedded BNF.
- **grim** (`grim-constrain`): `response_format` accepts `text | json_object | json_schema`. JSON-object mode uses a per-FSM-state memoized token mask (O(1) on revisits) — production-reasonable. JSON-schema mode combines a cached PDA structural mask with a prefix-keyed memoized schema-validity mask (`mask_for`, 1024-entry cap): the former full-vocab parse storm per structurally-valid token is gone, but **each novel output prefix still costs one O(vocabulary) serde parse+validate pass**, so per-step asymptotics are unchanged for straight-line generation; there is no FSM-precomputed mask or jump-forward. Supported keyword subset grew to `type, properties, required, enum, items, nested object/array, $ref` (internal `#/...` pointers, depth-limited, circular-ref detection), `oneOf/anyOf/allOf`, and `pattern` — though pattern matching is heuristic (special-cases `^[A-Z]{3}$`, otherwise anchored-prefix/suffix/substring approximations, not a regex engine). Only `format` is rejected outright. Honest limitation: grim constrains a subset of JSON Schema; vLLM/SGLang delegate to dedicated grammar engines.

### 3.5 Disaggregated (prefill/decode split) serving

- **vLLM**: pluggable KV-connector ecosystem (NIXL, Mooncake, LMCache, HF3FS, Moriio, offloading, FlexKV, multi-connector); docs admit disagg prefill improves tail latency, not throughput.
- **SGLang**: PD-disaggregation with mooncake/nixl/mori/ascend connectors plus single-instance PD multiplexing.
- **grim** (`grim-disagg` + `grim-kvtransport`): single-connector system with a checksummed V2 wire protocol, router, push and pull paths, and receiver servers. The formerly traced bug (transfers reading storage the live forward never populated) is **fixed**: `append_kv_layer` mirrors KV into the pool, receivers now start for **all** roles including prefill, the decode fetch loop skips per-block via an explicit received bit (set on the layer-0 write), and byte-exact loopback tests drive two real engines over TCP. Residual rough edges: if any layer >0 fetch fails the error is logged-and-skipped while the block stays marked received (that layer attends stale pages silently); the same data is sent twice per handoff (per-layer slices + pool-level transfer); the loopback test's decode side also runs a small local prefill, so pure transferred-KV decode is not yet proven. Scope remains smaller than the competitors' pluggable systems.

### 3.6 Quantization breadth

- **vLLM**: 25+ integrated method names (GPTQ/AWQ + Marlin variants, FP8, compressed-tensors, TorchAO, Quark, INC, MXFP4/MXFP8, NVFP4, modelopt family, online shorthands) — mostly maintained integrations of external projects.
- **SGLang**: 28+ registered methods with hardware-dispatch tables, Marlin FP8/FP4 utilities, DeepGEMM wrapper, ROCm MXFP8 kernels, FP4/FP8 KV-cache quant.
- **Ollama**: consumes anything llama.cpp reads (K-quants, i-quants, MXFP4); creates only F32/F16/Q8_0/Q4_K_S/Q4_K_M via shelling out to `llama-quantize`.
- **LlamaFactory**: quantized *export* to GPTQ (gptqmodel)/AWQ with calibration; no GGUF path.
- **Axolotl**: post-training quantize CLI via llm-compressor; MoE expert NVFP4/MXFP4 monkeypatch.
- **Unsloth**: GGUF export pipeline that builds llama.cpp itself, imatrix IQ quants, TorchAO, compressed-tensors/NVFP4 with calibration; "Dynamic" quant recipes live in companion infrastructure (`UNVERIFIED` in this snapshot).
- **grim**: a 16-variant `QuantFormat` enum (Q8_0, Q4K–Q6K, FP4/NF4/FP8 + block-16 variants, and the i-quant family Iq4Nl → Iq2S down to ~2 bits), all hand-implemented with round-trip tests. The previously-open **IQ2S production gap is closed**: `quant_iq2s` is now a real grid-code/sign-pack quantizer (82 bytes/block) matched by an exact-layout dequant test. Plus GPTQ GEMM + correction kernels (ROCm), SpQR residual format consumed by a fused dequant GEMM, and MXFP4 fake-quant for QAT (see §4.2 for its scope caveat). Separately, `grim-kvquant` provides Lloyd-Max-trained KV compression attached to the live engine (env `GRIM_KV_QUANT=int8|int4`) with compress-on-spill semantics, plus OmniKV-style eviction.

grim's count is comparable to the big engines'; the difference is provenance — grim hand-implements and golden-tests its formats, the big engines integrate external quant projects.

### 3.7 API surface and endpoint honesty

- **SGLang** serves the widest protocol set: OpenAI + Anthropic Messages + **Ollama-compatible** routes + native `/generate` + gRPC, behind a Rust gateway/router with cache-aware load balancing; 38 function-call format detectors.
- **vLLM**: OpenAI + Anthropic + Cohere protocols, batch endpoints, audio realtime WebSocket, extensive ops/admin routes (sleep/wake, pause/resume, weight updates, elastic-EP scaling).
- **Ollama**: native API + OpenAI compat (including images/audio) + Anthropic Messages; registry pull/push UX is best-in-class (content-addressed, resumable, digest-verified).
- **grim** (`grim-server`): broad OpenAI-shaped surface (`/v1/chat/completions`, `/v1/completions`, models + load/unload, adapters, tokenize/detokenize, rerank/score, cache resets, request pause/resume/cancel/stream SSE) plus Ollama-shaped `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull` (streaming real HF downloads), Prometheus metrics, health/readiness, dashboard. Streaming SSE throughout with `[DONE]` sentinels.

Endpoint honesty audit (the notable difference between grim and the specialists):
- `/v1/embeddings`: returns an explicit **501** with guidance — restored after briefly shipping a synthetic hash projection; the honest choice.
- `/v1/audio/transcriptions|translations`: run Whisper on an all-zero mel with constant token ids and return status strings ("audio sequence decoded (N token steps)") — the handlers do not even read the uploaded audio. Functional stubs with better wording than before.
- `/v1/images/generations`: now returns **real base64 pixels**, but from a Flux2 forward over constant latents with a zero prompt context decoded through a freshly `random()`-initialized VAE per request — unconditioned output, not prompt-driven generation.
- The grim-garage "Diffusion/Audio Studio" UI tabs are wired end-to-end at the transport level but demo-grade: random-init default configs, synthetic prompt embeddings, `char % 256` TTS tokens, sine-synthesized mel input; no checkpoint loading.
- By contrast, the core chat/completion/speculative paths are backed by real model forwards, and failures error rather than fabricate.

### 3.8 Local-API security posture

grim ships **zero authentication** on any HTTP route (verified by exhaustive grep); protection is loopback-default binding, CLI refusal of `0.0.0.0` without `--allow-public`, warning on non-loopback without TLS, and optional rustls. This matches Ollama's posture (its auth machinery is registry-only; local API is open by design) and general llama.cpp-style local servers. It remains a real constraint for anyone exposing these engines past localhost, for all three projects.

### 3.9 Model architecture coverage

- **LlamaFactory** registers ~634 named checkpoints across 131 model groups (triple-hub mirrors) and inherits everything else in HF transformers; ~124 chat templates; 24 multimodal plugins including audio and video.
- **vLLM**: 310 architecture entries in its registry, including dozens of VL/OCR/VLA families and Whisper ASR.
- **SGLang**: ~188 registered classes; diffusion generation lives in-repo alongside LLM serving.
- **Axolotl/Ollama/Unsloth**: effectively inherit HF transformers or llama.cpp coverage respectively (Unsloth maintains ~15 dedicated fast-path patchers plus a generic compiler path).
- **grim**: 152 declared architecture identifiers, 140 referenced by the loader, ~105 dedicated constructor sites — dense transformer families (LLaMA through Llama-4-class, Qwen2/3/3.5 + VL/MoE/Next, DeepSeek V2/V3/V4 + OCR, GLM4/Moe/Dsa, Gemma2/3/3n/4, Phi-3, CommandR/Cohere2, DBRX, Ernie4.5, Hunyuan incl. VL, Granite hybrid, MiniMax-M2/M3, Kimi Linear/K3, Apertus, Falcon/Bloom via generic arm…), non-transformers (Mamba/Mamba2/Jamba/Falcon-H1/Nemotron-H, RWKV-6/7, DeltaNet, BitNet, T5), vision (ViT/CLIP/BERT), audio (Whisper, Kokoro, StyleTTS2, Vocos, WavTokenizer), and diffusion (Flux2, UNet/VAE, flow-match). Multimodal fusion blocks exist at the model layer (`merge_multimodal_embeddings`, VL architectures), but image+text joint inference through the served chat path was not verifiable end-to-end — treat served multimodality as implemented-at-model-layer, `UNVERIFIED`.

The qualitative difference: everyone else rents architecture breadth from an upstream (HF or llama.cpp); grim maintains its own zoo, which means no inheritance — every addition is first-party work, and day-of-release model support (routine for HF-backed tools) is not achievable at grim's scale.

### 3.10 Evaluation

- **grim CLI**: windowed perplexity (wikitext2 sample, committed baseline) and GSM8K exact-match against a running server at temperature 0, plus an 18-format quant accuracy/perplexity regression gate. In-training held-out evaluation prints perplexity derived from a token-equality pseudo-NLL heuristic (`0.4` if consecutive tokens match else `2.1`) that is **independent of any model output** — a fabricated metric; use `grim eval` for real numbers.
- **Axolotl**: lm-eval integration plugin (plus its own evaluate CLI).
- **LlamaFactory**: benchmark evaluator orphaned — `llamafactory-cli eval` raises `NotImplementedError`.
- **Unsloth**: delegates entirely to HF Trainer/TRL.

---

## 4. Fine-tuning comparison

### 4.1 Method matrix

| Tool | LoRA family | Full FT | Preference/RL stages | Memory-efficient optimizers | Multi-GPU training |
|---|---|---|---|---|---|
| **grim** | LoRA/QLoRA + PiSSA/OLoRA/LoRA+/ReLoRA/OFT flags; SoulEater adapter math exists but unreachable | **Not truly available** — `full-bf16/full-fp16` emulate full FT by raising adapter rank to `min(hidden,1024)`; base weights stay frozen | Garage worker: real DPO/KTO/SimPO/ORPO/GRPO (four-forward, frozen reference, analytic grads). CLI modes: improved mechanics but still synthetic (§4.2) | AdamW/8-bit/paged/Lion/Lion-8bit/Adafactor/Muon/MAdam/LionVote/QGaLore-8bit genuine; `galore` silently aliases to plain AdamW; `lomo/adalomo/came/sophia/adamw-bnb` fail loudly | Garage data-parallel real (ROCm/RCCL, per-rank replicas, deterministic sharding, integration-tested on gfx1036). CLI `--num-gpus` all-reduces a single replica with itself. FSDP module is shape math with no executor |
| **Unsloth** | Core strength: LoRA/QLoRA/rsLoRA, FP8 LoRA, DoRA (slow path); manual-autograd Triton kernels | Yes (`full_finetuning=True`) | Via rewritten TRL (GRPO with Dr-GRPO corrections, vLLM colocate); DPO/KTO logic moved to zoo | Q-GaLore optimizer in-tree | **Hard-blocked**: RuntimeError when >1 GPU is visible |
| **LlamaFactory** | LoRA/OFT/freeze/full + LoRA+/rsLoRA/DoRA/PiSSA/LoftQ; QLoRA across bnb/gptq/awq/aqlm/quanto/eetq/hqq/mxfp4/fp8 | Yes | Stages pt/sft/rm/ppo/dpo/kto with ipo/orpo/simpo/hinge as DPO-stage losses; **no GRPO stage** (spun out to EasyR1) | GaLore/APOLLO/BAdam/Muon/Adam-mini | Deepest in set: DeepSpeed ZeRO 0–3 (+fp8/offload/auto-TP), FSDP1/2, Megatron-Core Adapter, HyperParallel TP/CP, Ulysses SP, Ray, elastic multi-node torchrun |
| **Axolotl** | LoRA/QLoRA/DoRA + ReLoRA in-tree + sample packing | Yes | DPO/IPO/ORPO/KTO/SimPO/GRPO/GDPO/EBFT + reward/PRM; async GRPO with vLLM weight sync and replay buffers; KD and diffusion-LM plugins | 8-bit/paged bnb, adopt, came, muon, dion, sinkgd, flash-adam/lion, q-galore, grokfast | FSDP1/2, DeepSpeed, N-D parallelism (FSDP×TP×CP×EP via DeviceMesh, ring attention, SSM state passing), Ray, multi-node |
| **vLLM / SGLang / Ollama** | n/a (inference products; vLLM serves LoRAs, none train) | n/a | Rollout-side plumbing only (weight-update/pause/resume endpoints designed for external RL frameworks) | n/a | Serving-side parallelism only |

### 4.2 What `grim train` actually does today (verified at `de52ff0d`)

Improvements landed since the last audit:

- **Dataset loader widened**: Alpaca, ShareGPT, OpenAI-messages, preference pairs, and now plain-text arrays and JSONL `{text}` / `{instruction, output}` formats — `docs/howto/train-adapter.md`'s documented format now works.
- **Optimizer help corrected**: `lomo/adalomo/came/sophia/adamw-bnb` were removed from help and return explicit `Unimplemented` errors with alternatives instead of silently aliasing to AdamW; Muon/MAdam/LionVote advertised.
- **CLI preference modes partially de-faked**: DPO/KTO/ORPO/SimPO/GRPO now split the sequence's logps into chosen/rejected halves, synthesize reference logps as −0.05 offsets, and (for DPO/KTO/GRPO) backprop the loss function's actual analytic derivative rather than a hardcoded scalar. This produces a *non-constant* loss — but it is still synthetic preference training: both halves come from one forward pass over the same sequence, no preference-pair dataset is consumed by the CLI, and ORPO/SimPO still use fixed gradient scalars. The **garage worker remains the only correct end-to-end preference trainer** (real chosen/rejected pairs from JSONL, four forwards including a frozen reference policy, distinct per-input gradients, sharded RCCL execution).
- **Vulkan/Metal parity tests added**: `parity_cpu_vulkan_metal.rs` covers kernel-registry/SPIR-V and MSL-manifest parity plus CPU-referenced numerical parity for RMSNorm/RoPE/SwiGLU/softmax and FP8/MXFP4 dequant — closing the "Vulkan/Metal absent from parity crate" gap (host-referenced checks; hardware-launch validation `UNVERIFIED`).

Still true and unresolved:

- `--mode` selects hyperparameters, not algorithms: `lora | full-bf16 | full-fp16 | soul-eater | oft` all inject the same QLoRA-style adapters; full-parameter training does not exist (base weights are never optimized), `soul-eater` maps onto the spectral-QLoRA flag rather than the SoulEater adapter/optimizer, while `oft` genuinely toggles OFT forward/backward ops.
- `--optimizer galore` now silently constructs **plain AdamW** (no GaLore projector) — a re-aliasing, different target, same silent-under-delivery pattern. `qgalore`/`galore-8bit` map to the genuine QGaLore implementation.
- In-training "eval" is the fabricated pseudo-NLL described in §3.10.
- `--qat-mxfp4` fake-quantizes only `lm_head.weight`; the flag's doc claiming Linear-weight coverage is false.
- Advanced research trainers (SoulEater/Scythe1 FIM preconditioning, TurboFinetune, OmniGrad, ContrastOmni, OmniloPrune, distillation, multimodal GRPO modalities) are implemented and tested but reachable from no production entry point; corresponding garage labels fall through to generic SFT.
- The autograd engine is deliberately narrow: 9 taped op kinds with hand-written VJPs, gradient checkpointing plus a new segment-replay mode proven gradient-parity-equal to uncheckpointed backward, finite-difference checks for DoRA/OFT — a transformer-training op set, not a general library. No second-order gradients, no distributed autograd.
- SCYTHE-2 (online per-layer GPU-placement controller) is now closed-loop within garage *training*: real per-layer indices/shapes, P2P link probing, measured-latency feedback routing gradient sync. It does not touch inference.

### 4.3 Framework character

LlamaFactory and Axolotl buy enormous method/architecture breadth by standing on HF; their risk is upstream coupling (both pin exact library versions; Axolotl carries 40+ monkeypatch modules and a separately-licensed `integrations/` directory inside its Apache repo; LlamaFactory is visibly mid-refactor with a dual classic/`v1` framework). Unsloth buys speed/memory on supported architectures with hand-derived kernels and pays in fragility (version-exclusion lists, `exec`-based source rewriting, silent fast-path fallbacks when dropout/bias settings disqualify them, single-GPU training lock). grim owns its whole stack — no upstream breakage, full determinism control, honest numerics cores pinned by golden tests — and pays with breadth: no full fine-tuning, no real RLHF stage orchestration outside the garage worker, and a fraction of the architecture coverage.

---

## 5. Deployment model differences

- **Ollama, grim**: single-binary local daemon with model-pull UX; scaling story ends at one machine (grim adds intra-host multi-GPU and prefill/decode split; Ollama adds a signed cloud-proxy passthrough and an embedded agent runtime, drifting toward platform territory).
- **vLLM, SGLang**: cluster-scale by design — tensor/pipeline/data/expert parallelism, disaggregation connectors, load-balancing gateways, elastic rescaling.
- **Unsloth, LlamaFactory, Axolotl**: training-only; output checkpoints get served by the engines above (Unsloth notably exports directly into vLLM/Ollama formats; LlamaFactory embeds vLLM/SGLang engines behind its API server).
- **grim** again occupies the odd slot: serving-grade features (continuous batching, paging, prefix cache, speculation, constrained decoding, disagg) *and* a training stack in one dependency-free binary, at single-node scope.

---

## 6. Backend/kernel depth (grim-specific axis)

ROCm is the primary backend by a wide margin (85 files / ~43.6k lines): MLA decode, FlashDecode split-K, SageAttention, Marlin GEMM, MXFP4/IQ/GPTQ dequant-GEMM families, the Charon fused-MoE kernel suite with runtime variant selection and WMMA/backward paths, selective scan for SSM models, RCCL/P2P/peer-access, graph capture, and a persistent JIT HSACO disk cache. CUDA (~7.4k lines) carries 42 fused kernels and an nvcc→PTX disk cache but no CUDA-graph capture yet; Vulkan (~6k lines, SPIR-V, cooperative-matrix GEMM, paged dequant attention, speculative acceptor kernel) and Metal (~5.8k lines, split-K matmul, MLA decode, Marlin, Sage, M-RoPE) are real implementations trailing the ROCm surface. Numerical-parity discipline: known-answer CPU/ROCm/CUDA tests for five quant formats, plus the new CPU↔Vulkan↔Metal parity suite. grim's cross-backend uniformity (one model code, five backends) has no analog in this comparison set — the Python engines vendor vendor-libraries, and Ollama delegates entirely to llama.cpp.

---

## 7. Summary table

| Axis | Standing |
|---|---|
| Consumer-AMD (RDNA) design center | **grim leads** (72 files, per-generation kernels, hardware-verified tests) > vLLM (real rdna3 kernels, thin) > SGLang (zero) ; Ollama serves consumers well via inherited llama.cpp backends |
| Datacenter-AMD (CDNA) depth | vLLM and SGLang ahead; grim present but secondary |
| Paged KV + prefix caching | Present and wired in all three engines; SGLang broadest (radix variants + tiered storage), vLLM deepest CUDA-graph integration, grim complete-but-narrower with host-resident-pool caveat; radix reuse now **default-on** |
| Scheduler sophistication | vLLM/SGLang ahead (overlap schedulers, micro-batch overlap); grim competitive on admission control/chunked prefill/preemption; Ollama inherits defaults |
| Speculative decoding | SGLang (8 builtins + plugin ABI) ≥ vLLM (9 proposers) > grim (3 strategies, **default-on**, EAGLE3/native-MTP implemented-but-unwired) > Ollama (exposure only) |
| Structured output | vLLM/SGLang ahead (grammar engines, jump-forward); grim functional on a JSON subset ($ref/composition supported, heuristic patterns, O(V)-per-novel-prefix cost, no jump-forward) |
| Disaggregated serving | vLLM/SGLang pluggable ecosystems; grim single-connector, transport now correct with documented residual edges; others none |
| Quantization | Comparable breadth by count (vLLM 25+/SGLang 28+ integrated vs grim 16 hand-implemented + GPTQ/SpQR/MXFP4-QAT + KV-spill compression); IQ2S production gap closed |
| API protocols | SGLang widest (incl. Ollama-compat + gRPC); vLLM multi-dialect; Ollama strongest registry UX; grim broad OpenAI+Ollama shapes |
| Endpoint integrity | grim mixed: honest failures on embeddings, status-string stubs on audio, unconditioned pixels on images; competitors' endpoints are engine-backed |
| Local-API auth | All of grim/Ollama/llama.cpp-style servers: none by default |
| Architecture coverage | LlamaFactory ≫ vLLM > SGLang > grim (self-maintained zoo, 152 identifiers) ; HF/llama.cpp-backed tools inherit the rest |
| Fine-tuning breadth | Axolotl ≈ LlamaFactory > grim > Unsloth(breadth) ; grim uniquely dependency-free, with real RLHF-stage training only via the garage worker |
| Training correctness discipline | grim stands out: golden-value optimizer/loss tests, mutation-resistant suites, gradient-parity-proven checkpoint replay, loud rejection of unimplemented optimizers — undercut by decorative CLI modes and two remaining silent aliases/heuristics |
| Full-parameter fine-tuning | Available in Unsloth/LlamaFactory/Axolotl; **not in grim** (rank-raised adapter emulation) |
| Multi-GPU training | LlamaFactory/Axolotl deep; grim garage-DP real but ROCm-only; Unsloth blocked |
| Evaluation harness | grim (ppl+GSM8K) and Axolotl (lm-eval plugin) usable; LlamaFactory deprecated theirs; in-training eval in grim is a fabricated placeholder |

---

## 8. Open weaknesses in grim (verified current at `de52ff0d`)

Serving:
1. Zero authentication on all HTTP surfaces.
2. Audio transcription/translation endpoints ignore input audio entirely and return status strings; image generation is unconditioned (zero prompt context, per-request random VAE); garage studios run random-init pipelines.
3. EAGLE3 drafter and native-MTP registration implemented with no callers; `lookahead_literal` jump-forward helper likewise unwired.
4. Schema-constraint `pattern` support is heuristic (no regex engine); novel-prefix schema masking remains O(vocab) per step; no FSM-precomputed masks or jump-forward decoding.
5. KV pool is host-resident f32; device pages are staged per-layer-per-step uploads; CUDA backend lacks graph capture (ROCm has it); Vulkan/Metal parity is host-referenced rather than hardware-launch-validated.
6. Disagg residuals: failed layer>0 fetches leave blocks marked received (silent stale pages); duplicate send redundancy; pure transferred-KV decode unproven by tests.

Training:
7. No true full-parameter fine-tuning despite `full-bf16`/`full-fp16` modes and docs implying otherwise; `soul-eater` mode relabels spectral-QLoRA.
8. CLI preference modes are synthetic (split-half chosen/rejected from one forward, fabricated reference offsets); the correct implementation lives only in the garage worker.
9. `--optimizer galore` silently yields plain AdamW.
10. In-training eval prints model-independent pseudo-perplexity.
11. QAT touches only `lm_head.weight` contrary to its documentation.
12. CLI `--num-gpus` performs a self-all-reduce (no data sharding/replication); FSDP remains shape math; advanced trainers (SoulEater/Scythe1/distillation/etc.) unreachable from any entry point.
13. `docs/howto/train-adapter.md` still describes `full-bf16/full-fp16/soul-eater/oft` as distinct behaviors.

Where grim is genuinely ahead of this comparison set: the RDNA-consumer design center with per-generation kernels and hardware-validated tests; a fully self-owned, PyTorch-free stack spanning serving and training with cross-backend numerical-parity discipline; default-on speculative decoding with acceptance telemetry; and an optimizer/loss numerics core pinned by golden values. Where it is genuinely behind: architecture breadth (no HF inheritance), RLHF orchestration outside the garage worker, grammar-engine-grade constrained decoding, cluster-scale serving machinery, and full-parameter/multi-GPU training depth.

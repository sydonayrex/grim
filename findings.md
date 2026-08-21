# grim — Findings (improvement backlog)

Sources: comparative reviews of `old/repos/` — SGLang (FIND-1..3, prior pass)
and the full repo sweep below (FIND-4 onward). Each finding is a gap to close
or an integration opportunity; this file will later be converted into an
implementation plan. Ordered by leverage within each section.

---

# Part I — SGLang-derived findings

## FIND-1 · Accuracy benchmark suite

**Gap.** Quantization and kernel correctness currently rests on ad-hoc KATs
(kernel-vs-dequant comparisons) and one broken-until-this-week smoke bench.
There is no perplexity or task-eval harness to prove a quant path, a new
kernel, or a loader change preserves model quality. SGLang ships
gsm8k/hellaswag/boolq/perplexity suites in `benchmark/`; grim has none.

**Why it's #1.** Every quant kernel, GGUF loader workaround, and autotune
winner is currently unverifiable end-to-end. This protects all existing work
and gates everything below.

**Needed:**
- Perplexity eval on a fixed corpus (wikitext-2 class) runnable per model/quant.
- Task evals (gsm8k at minimum) via the local server API.
- Golden-number capture per (model, quant) pair; CI-comparable output.
- `grim-cli eval` subcommand + persisted results readable by `bench`.

**Acceptance shape:** `grim-cli eval --model <id> --task ppl,gsm8k` emits
stable numbers; a regression beyond tolerance fails.

## FIND-2 · Attention-kernel breadth

**Gap.** SGLang delegates attention to FlashInfer/FlashAttention/CUTLASS/
DeepGEMM — MLA, DSA, preshuffled KV, dual-chunk, hybrid-linear backends come
free. Grim hand-writes every attention path; coverage centers on standard
MHA/GQA decode + prefill with hybrid conv/recurrent handled per-model.
Missing breadth: MLA-class compressed-KV attention, preshuffled/KV-block
layouts for large batch, sliding-window hybrids as a shared kernel rather than
per-model scalar loops, cross-attention batching.

**Needed:**
- Inventory which attention variants the 139 supported models actually need;
  rank by usage frequency.
- Shared sliding-window / hybrid-attention kernels extracted from per-model
  scalar loops.
- MLA-class path for DeepSeek-family loaders.
- Recorded decision per variant: delegate (vendor hsaco) vs own-kernel, with
  the same JIT-cache/autotune treatment as GEMM.

## FIND-3 · Throughput telemetry + parity numbers

**Gap.** No published tokens/sec parity vs vLLM/SGLang on identical hardware
(gfx1036); no ITL/acceptance-rate histograms surfaced to users. The
consumer-GPU niche claim is currently unmeasurable.

**Needed:**
- Serving benchmark (`grim-cli bench --mode serving`): tokens/sec + ITL
  percentiles under concurrency on real workloads (sharegpt-length).
- Speculative-decoding telemetry (acceptance rate, draft length, net speedup)
  in `/api/stats` and per-request logs.
- Head-to-head script vs vLLM/SGLang; results in `docs/benchmarks/gfx1036.md`.

---

# Part II — Full old/repos sweep (16 repos)

Repos reviewed: unsloth, vllm, axolotl, labs-OO-Agents (NOOA), peregrine,
gigatoken, burn, LlamaFactory, llama.cpp, ollama, AngelSlim,
llvm-project-amd-staging, cubek, hip-develop, bebelm, cubecl.

Quick map:

| Repo | Size | What it is | Relevance to grim |
|---|---|---|---|
| unsloth-main | ~970K py | Fine-tuning speed lib (patched transformers, custom kernels, QGaLore etc.) | Training-side techniques |
| vllm-main | ~1.5M py | Serving engine, widest quant/hardware matrix | Serving parity target |
| axolotl-main | ~204K py | Fine-tuning config framework (YAML→trainer) | Training UX |
| labs-OO-Agents | ~256K py | NVIDIA NOOA object-oriented agent framework | Agent-serving integration |
| peregrine-main | ~76K rs | From-scratch Rust DL lib (Apple Silicon), 600+ tests | Autograd reference |
| gigatoken-main | ~40K rs | ~1000x faster tokenizer, HF-compatible | Tokenizer perf |
| burn-main | ~372K rs | 33-crate Rust DL framework on CubeCL | Backend architecture |
| LlamaFactory-main | ~56K py | Unified fine-tuning (LoRA/QLoRA/full/WebUI) | Training UX |
| llama.cpp-master | ~548K c/c++ | GGML inference, 15+ backends incl ggml-hip | Quant/loader gold standard |
| ollama-main | ~296K go | Model server + new agent runtime, model parsers | Server/product shape |
| AngelSlim-main | ~82K py | Compression toolkit (quant/QAT/distill/sparsity/spec-decode) | Quant pipeline |
| llvm-project-amd-staging | huge | AMD LLVM staging: comgr, device-libs, hipcc | Toolchain understanding |
| cubek-main | ~98K rs | CubeCL kernel library (matmul/attention/quant/fft…) | Kernel algorithms |
| hip-develop | C/C++ | HIP runtime headers/runtime source | FFI ground truth |
| bebelm-main | ~7K rs | Pure-Rust CPU LFM2.5-8B-A1B inference | Minimal-model reference for same arch |
| cubecl-main | ~140K rs | Multi-platform GPU kernel language (HIP/CUDA/Metal/CPU/WGPU) | Kernel portability |

Cross-cutting strengths grim already holds vs this field: only llama.cpp,
ollama, and bebelm share grim's "own-the-whole-stack" property; none of the
Python tools do. Grim's training capability (autograd + SCYTHE + garage) is
matched only by unsloth/LlamaFactory/axolotl/AngelSlim, all Python/torch-bound.

## FIND-4 · Tokenizer throughput (from gigatoken)

**Gap.** Grim's tokenizer is a straightforward GGUF BPE/SentencePiece decode
path; `grim-cli run` and server ingest tokenize single-threaded on the hot
path. Gigatoken demonstrates GB/s tokenization via parallel pre-tokenization
and cache-friendly batch encode, drop-in HF-compatible.

**Fix / integration:**
- Profile tokenizer share of TTFT at 32K+ context; if >2%, adopt gigatoken's
  two techniques: regex pre-tokenize fan-out across threads + arena-packed
  vocab lookup.
- Optional: vendor gigatoken as a crate dependency for bulk dataset
  tokenization in garage training jobs (it has a native Rust core).

## FIND-5 · Compression pipeline depth (from AngelSlim)

**Gap.** Grim quantizes post-ho (GGUF family, MXFP4) but has no QAT, no
distillation-based compression, and no sparsity path. AngelSlim packages
quant/QAT/QAD/distill/sparsity/speculative-compression as one engine with
per-method configs. Grim's SCYTHE distill crate covers part of this but isn't
wired as a compression product.

**Fix / integration:**
- Wire `grim-speculative/src/distill.rs` into garage as a user-facing
  "compress" job (teacher→student with quantized student target), mirroring
  AngelSlim's distill flow.
- Add QAT for MXFP4 targets: forward-fake-quant during garage fine-tune so
  adapters trained on Q8_0 bases don't degrade when merged into MXFP4 packs.
- Sparsity: defer unless a consumer use-case appears (2:4 structured needs
  tensor-core support RDNA2 lacks).

## FIND-6 · Training-framework UX (from unsloth / LlamaFactory / axolotl)

**Gap.** Unsloth wins fine-tuning users via patch-free speed (custom
cross-entropy/layernorm/GEGLU/LoRA kernels, gradient checkpointing done right)
plus 26 model-family patches; LlamaFactory/axolotl win on config UX (YAML
recipes, WebUI, dataset registry). Grim's garage/train works but has neither
the speed story nor the recipe UX.

**Fix / integration:**
- Steal the unsloth kernel list selectively: fused cross-entropy over the full
  vocab (grim's output head materializes logits — a chunked CE would cut peak
  memory ~vocab×seq), fused RMSNorm backward, fused GEGLU. All are
  straightforward in grim-autograd and benefit both train and inference.
- Adopt the LlamaFactory recipe shape: versioned YAML training recipes in-repo
  (`docs/recipes/lora-lfm25.yaml`) so garage jobs are reproducible one-liners.
- Dataset registry with hash-verified local datasets (LlamaFactory's
  `data/dataset_info.json` pattern).

## FIND-7 · Kernel portability layer decision (cubecl / cubek / burn)

**Gap.** Grim maintains 45 hand-written ROCm kernel modules plus separate
CUDA/Vulkan backends with duplicated logic. CubeCL solves exactly this: one
Rust kernel dialect compiling to HIP/CUDA/Metal/WGPU/CPU, with cubek shipping
matmul/attention/reduce/quant kernels on top, and burn proving the approach
across 33 backend crates.

**Fix / integration (strategic decision required):**
- Do NOT rewrite grim's proven ROCm kernels onto CubeCL now — the autotuner
  and RDNA-specific tuning are competitive advantages.
- DO evaluate CubeCL for the *next* backend surface where duplication hurts:
  Metal and Vulkan paths currently lag; porting new algorithmic kernels there
  via CubeCL (or porting cubek's matmul/quant kernels as references for
  grim-vulkan) avoids writing SPIR-V by hand twice.
- Track cubek's attention/matmul algorithm evolution as a free R&D feed; their
  RDNA2 issue (#1365, saved PDF in old/repos) is directly relevant intel.

## FIND-8 · CPU-inference floor (from llama.cpp / bebelm)

**Gap.** Grim treats CPU as a fallback backend; llama.cpp's ggml-cpu is a
first-class highly-tuned engine (packed GEMM, quantized VPMADD paths), and
bebelm proves LFM2.5-8B-A1B runs interactively on pure CPU with minimal code.
Grim's CPU backend lacks packed low-bit GEMM (q4_K/q8_0 dot products run via
dequant-to-f32 today in several paths).

**Fix / integration:**
- Implement packed integer CPU GEMM for q4_K/q5_K/q8_0 following ggml-cpu's
  `dotprod` kernels (AVX2/AVX-VNNI/NEON variants). This makes the
  self-hoster persona genuinely usable and gives a correct reference oracle
  for GPU kernels (FIND-1 synergy).
- Bebelm is a same-architecture (LFM2.5) minimal reference — diff its conv
  state handling against grim's lfm2.rs for latent bugs; its agent.rs/chat.rs
  show a leaner session API worth reviewing.

## FIND-9 · Product surface: agents & tool ecosystem (from ollama / NOOA)

**Gap.** Ollama shipped an agent runtime (agent/app/auth/registry dirs) and
per-model parser/render pairs (deepseek3.go, cogito.go, cohere.go…); NVIDIA's
NOOA formalizes typed object-oriented agents. Grim's server has tool-calling
plumbing (WI-TOOLS) but no agent runtime, session persistence, or model-
specific output-parser registry beyond `tool_parse.rs` families.

**Fix / integration:**
- Per-model tool-call detector registry (ollama parsers pattern): map model
  family → detector instead of the current template-heuristic
  (`resolve_tool_family`). Start with the families grim actually serves
  (lfm2, llama, qwen, deepseek).
- Session/conversation persistence so multi-turn agent loops survive server
  restarts (self-hoster persona asks for this in usability-test.md T15.2).
- NOOA-style typed tool schemas can ride grim's existing OpenAI tools endpoint;
  no protocol work needed, just docs + examples.

## FIND-10 · RL-rollout readiness (from vllm / axolotl ecosystem)

**Gap.** vLLM's dominance in RL post-training comes from API surfaces for
rollout engines (weight-sync endpoints, sleep/wake for colocated trainers,
partial rollout). Grim trains natively but offers no rollout-server interface,
so it can't participate in that ecosystem, and its own SCYTHE loop can't
import external reward models.

**Fix / integration:**
- Minimal viable surface: `/v1/weights/update` (load adapter/weights from path
  without restart — mostly exists via adapters/load), sleep/resume for KV
  retention (pause_request exists), and a documented stateless generation API
  contract for trainer drivers.
- Defer full verl/slime integrations until FIND-1 exists to validate outputs.

## FIND-11 · Toolchain intelligence (llvm-project-amd-staging / hip-develop)

**Gap.** Not a feature gap — these are ground-truth sources grim under-uses.
The comgr/device-libs/hipcc trees explain exactly why hipRTC compilations fail
(name-mangling, device-lib linking, gfx-target propagation — the F-1 class of
bugs), and hip-develop documents runtime APIs grim's FFI guesses at.

**Fix / integration:**
- When any hipRTC/comgr failure occurs, grep these trees first (saved a
  round-trip on the fused-q8_0 fix).
- Extract the offline-compile recipe (hipcc → hsaco → load) as a grim build
  option to bypass JIT entirely for release builds: faster startup, no
  hiprtc dependency, deterministic binaries.

## FIND-12 · Reference-grade test discipline (from peregrine / burn)

**Gap.** Peregrine (76K LOC, one author) carries 600+ tests including
op-by-op benchmarks vs PyTorch/MLX/JAX; burn has backend-test crates
(burn-backend-tests) enforcing op equivalence across every backend. Grim's
~2K tests skew toward scheduler/server plumbing; tensor-op equivalence across
CPU/ROCm/CUDA is thin.

**Fix / integration:**
- Adopt burn's pattern: a `grim-backend-tests` crate running identical KATs on
  every backend, gate on CI (CPU always; ROCm when GRIM_RUN_GPU_TESTS=1).
- Adopt peregrine's op-benchmark suite shape: JSON-emitting per-op timing for
  regression tracking (feeds FIND-3 too).

## Explicitly reviewed, no action needed now

- **vllm quantization breadth (24 methods)**: grim's GGUF/MXFP focus is a
  deliberate differentiator; revisit AWQ/GPTQ only if a user demand appears —
  conversion tools cover those formats upstream.
- **NOOA framework adoption**: grim serves agents; it needn't be one.
- **llama.cpp backend zoo (CANN/MUSA/OpenVINO/hexagon…)**: out of scope;
  grim's four backends match its audience.
- **burn as a base**: grim's stack is further along for LLM inference than a
  migration would credit; cherry-pick patterns (FIND-12) instead.

---

*Status: findings only — no plan commitments yet. Next step: prioritize
FIND-1..12 into a WI-style implementation plan with scope, milestones, and
verification.*

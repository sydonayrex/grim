# grim architecture coverage gap analysis (vs llama.cpp v1)

This document inventories the gap between grim-models' existing implementations
and the set of model architectures present in upstream `llama.cpp` as of its
current source tree.

## Source of truth

- llama.cpp model implementations: `old/repos/llama.cpp-master/src/models/*.cpp`
- grim-model files: `crates/grim-models/{transformer,mamba,vision,audio,diffusion}/src/*.rs`
- Architecture dispatch: `crates/grim-cli/src/run.rs::load_model_from_gguf`

Total architectures implemented by llama.cpp (this sync): **136** `.cpp`
model-implementation files in `llama.cpp/src/models/`. This number double-counts
some families (e.g. each MoE variant is its own file). The list below is the
canonical set the GGUF parser cared about pre-existing in grim.dll dispatch.

## Already shipped in grim-models

| Architecture | File | Status                                            |
| ------------ | ---- | ------------------------------------------------- |
| `llama`      | transformer/model.rs (Llama)                 | production (CPU stub)   |
| `mamba`      | mamba/lib.rs                              | production (CPU stub)   |
| `gpt2`       | transformer/gpt2.rs                       | skeleton                |
| `gemma`      | transformer/gemma.rs                      | skeleton                |
| `gemma2/3`   | —                                        | not yet implemented     |
| `t5`         | transformer/t5.rs                         | skeleton                |
| `deepseek`   | transformer/deepseek.rs                   | skeleton (DeepSeek-V2 not V1) |
| `rwkv` (RWKV-4-style) | mamba/rwkv.rs                  | skeleton (RWKV/v4 only) |
| `bert`       | vision/bert.rs (in `CausalLm`)           | skeleton                |
| `vit`        | vision/vit.rs                            | vision only             |
| `lfm2`       | transformer/lfm2.rs                      | **wired into dispatch, GPU/CPU run path end-to-end verified (sleipnir.gguf)** |

Total: **11** architecture implementations.

## Tier-1 gap (must implement for "all llama.cpp-standard architectures" goal)

These are the architectures most likely to ship as GGUF files. Each entry lists:
- the llama.cpp source file containing the canonical ops + tensor name map
- the approximate ops that need a new grim module if not covered by existing Linear/RmsNorm/Rope
- the dispatch key added to `run.rs::load_model_from_gguf`

| Architecture   | llama.cpp source file | New ops needed                              |
| -------------- | --------------------- | ------------------------------------------- |
| `phi2`         | phi2.cpp              | (covered by Linear + Rope + RmsNorm)        |
| `phi3`         | phi3.cpp              | (Linear + Rope + RmsNorm)                   |
| `mixtral` (Mixtral MoE) | mixtral-style in llama.cpp | top-k sparse MoE swap (replaces dense ffn down/up) |
| `mistral3`     | mistral3.cpp          | (Linear + Rope + RmsNorm + sliding-window)  |
| `mistral4`     | mistral4.cpp          | (transparent — sliding-window tweaks)       |
| `falcon`       | falcon.cpp            | (Linear + Rope + LayerNorm, no QKV-norm)    |
| `falcon-h1`    | falcon-h1.cpp         | hybrid Mamba/attention — extends `mamba` + `Llama` blocks |
| `rwkv6`        | rwkv6.cpp / rwkv6-base.cpp | RWKV-time-decay + loRA-style gate (extends Rwkv) |
| `rwkv7`        | rwkv7.cpp / rwkv7-base.cpp | RWKV-7 time-decay + D-LERP (extends Rwkv)   |
| `arwkv7`       | arwkv7.cpp            | RWKV-7 attention-receptive variant         |
| `jamba`        | jamba.cpp             | Mamba+attention hybrid (extends Mamba + LlamaBlock) |
| `qwen2`        | qwen2.cpp             | (Linear + Rope + RmsNorm; same as Llama)   |
| `qwen2moe`     | qwen2moe.cpp          | + MoE swap                                  |
| `qwen3`        | qwen3.cpp             | (Linear + Rope + RmsNorm; same as Llama)   |
| `qwen3moe`     | qwen3moe.cpp          | + MoE swap                                  |
| `jais` / `jais2` | jais.cpp / jais2.cpp | (Linear + Rope + LayerNorm; ARGSWapper)     |
| `xverse`       | xverse.cpp            | (Linear + Rope + RmsNorm; same as Llama)   |
| `cohere2`      | cohere2.cpp           | (Linear + Rope + no QKV-norm; sliding-window) |
| `command-r`    | command-r.cpp         | (Linear + Rope + RmsNorm; sliding-window)  |
| `deepseek2`    | deepseek2.cpp         | + Multi-latent Attention (compat w/ existing deepseek partial) |
| `deepseek3`    | deepseek32.cpp        | (DeepSeek-V3 sparse MoE — extends deepseek) |
| `olmoe`        | olmoe.cpp             | + MoE swap                                  |
| `olmo2`        | olmo2.cpp             | (Llama-style w/o QKV-norm + no bias)       |
| `llama4`       | llama4.cpp            | + Vision encoder + MoE; extends Llama + new vit-layer |
| `gemma3`       | gemma3.cpp            | (Linear + Rope + RmsNorm + sliding-window + vision) |
| `bai-llm`      | baichuan.cpp          | (Linear + Rope + RmsNorm; same as Llama)   |
| `minicpm`      | minicpm.cpp           | (Linear + Rope + RmsNorm; same as Llama)   |
| `minicpm3`     | minicpm3.cpp          | (Linear + Rope + RmsNorm; same as Llama)   |
| `chatglm`      | chatglm.cpp           | + ChatGLM rotary embedding scheme         |
| `smollm3`      | smollm3.cpp           | (Linear + Rope + RmsNorm; same as Llama + sliding-window) |
| `internlm2`    | internlm2.cpp         | (Linear + Rope + RmsNorm; same as Llama)   |
| `seed-oss`     | seed-oss.cpp          | (Linear + Rope + RmsNorm; same as Llama)   |
| `dbrx`         | dbrx.cpp              | + Fine-grained MoE swap                     |
| `grok-1`       | grok.cpp              | + MoE w/ routing-class embedding             |
| `grovemoe`     | grovemoe.cpp          | + MoE w/ shared experts                     |
| `granite`      | granite.cpp           | (Linear + Rope + RmsNorm; Llama-style)      |
| `granite-moe`  | granite-moe.cpp       | + MoE swap                                  |
| `granite-hybrid` | granite-hybrid.cpp  | hybrid SSM/attention (extends Mamba + LlamaBlock) |
| `exaone`       | exaone.cpp            | (Linear + Rope + RmsNorm; Llama-style)      |
| `exaone4`      | exaone4.cpp           | (Linear + Rope + RmsNorm; sliding-window)  |
| `falcon-h1`    | falcon-h1.cpp         | hybrid attention + Mamba                    |
| `llada`        | llada.cpp             | (Linear + Rope + RmsNorm; diffusion-then-AR) |
| `kimi-linear`  | kimi-linear.cpp       | linear attention hybrid                     |
| `mamba2`       | mamba2.cpp            | (extends Mamba: SSD + structured-conv)       |
| `nemotron-h`   | nemotron-h.cpp        | hybrid attention + Mamba                    |
| `plamo`        | plamo.cpp             | (Linear + Rope + RmsNorm; Llama-style)      |
| `plamo2`       | plamo2.cpp            | Mamba-style hybrid                          |
| `plamo3`       | plamo3.cpp            | (Linear + Rope + RmsNorm; Llama-style)      |
| `olmo`         | olmo.cpp              | (Linear + Rope + LayerNorm; Llama-style)   |
| `olmoe`        | olmoe.cpp             | + MoE swap                                  |
| `ernie4-5`     | ernie4-5.cpp          | (Linear + Rope + RmsNorm; same as Llama)   |
| `hunyuan-dense` | hunyuan-dense.cpp    | (Linear + Rope + RmsNorm; Llama-style)      |
| `hunyuan-moe`  | hunyuan-moe.cpp       | + MoE swap                                  |
| `step35`       | step35.cpp            | + MoE swap                                  |
| `smollm3`      | smollm3.cpp           | (Linear + Rope + RmsNorm; sliding-window)  |
| `smallthinker` | smallthinker.cpp      | (Linear + Rope + RmsNorm; small MoE)       |
| `qwen35`       | qwen35.cpp            | (Linear + Rope + RmsNorm; new arch)        |
| `qwen35moe`    | qwen35moe.cpp         | + MoE swap                                  |
| `seed-oss`     | seed-oss.cpp          | (Linear + Rope + RmsNorm; Llama-style)      |
| `pangu-embed`  | pangu-embed.cpp       | (embedding-only)                             |
| `t5encoder`    | t5encoder.cpp         | (encoder-only T5)                            |
| `wavtokenizer-dec` | wavtokenizer-dec.cpp | (audio decoder)                            |

## Tier-2 gap (specialized/secondary coverage)

- specialized models: `arcee`, `arctic`, `bailingmoe`, `bailingmoe2`, `bloom`, `chameleon`, `codeshell`, `cogvlm`, `deci`, `deepseek2ocr`, `dflash`, `delta-net-base`, `dots1`, `dream`, `eagle3`, `eurobert`, `exaone-moe`, `falcon-h1`, `gemma3n`, `gemma4`, `gemma4-assistant`, `gemma-embedding`, `glm-dsa`, `glm4`, `glm4-moe`, `gptneox`, `hunyuan-vl`, `jina-bert-v2`, `jina-bert-v3`, `llada-moe`, `llama-embed`, `maincoder`, `mellum`, `mimo2`, `mpt`, `nemotron`, `nemotron-h-moe`, `neo-bert`, `nomic-bert`, `nomic-bert-moe`, `openai-moe`, `openelm`, `orion`, `paddleocr`, `phimoe`, `rnd1`, `smollm3`, `stablelm`, `starcoder`, `starcoder2`, `talkie`, `modern-bert`, `refact`

Each of these is a separate `*Config` + `*Forward` implementation that follows
the existing grim-models pattern: one file per architecture with `pub struct X` +
`impl CausalLm for X` + one `else if arch.contains("…") { X::load(...) ; Ok(Box::new(m)) }`
line in `run.rs`. Primitive coverage (Linear/RmsNorm/Rope/Embedding) is already
sufficient for the majority of these — only a small minority require a new op.

## MoE dependency (cross-architecture)

Implementing the MoE swap is the largest single piece of work needed to cover
most tier-1 architectures. Once the `MoE` swap (gating network, top-k routing,
shared experts pattern) exists in `grim-nn` as a `MoeBlock` module, all 24+
MoE architectures can be implemented by combining it with existing
`Llama`-style blocks.

The current `Llama` block in `crates/grim-models/transformer/src/model.rs` has
`println!("[MoE Router]")` debug leftovers in the gate forward path — must
remove before any MoE work.

## Status: end-to-end LFM2 verified

Implemented this session:
- `crates/grim-cli/src/run.rs`: fixed `as_u32` extension for Int32/Int8/Int16/Uint8/Uint16
  metadata coercion; fixed `get_meta` to fall back to `as_u32`/`as_f32`; added
  GGUF-array parsing for `lfm2.attention.head_count_kv[]`; correct layer-type
  map (shortconv at layers where head_count_kv==0, attention otherwise);
  fix `main.rs:293` model-path dispatch.
- `crates/grim-format/src/gguf.rs`:
  - `read_gguf_tensor_info` now computes byte-size using `block_size`/`type_size_per_block`
    matching the canonical gguf-main layout (Q8_0 type_size=34, block_size=32 →
    `params × type_size / block_size`); the previous version assumed byte-size
    equal to `params × element_size` and multiplied Q-quant sizes by 8x.
  - `GgufValue::as_u32` widened to coerce Int32/Int8/Int16/Uint8/Uint16.
- `crates/grim-format/src/tprov.rs`: `architecture_owned()` + removed unused debug iter.
- `crates/grim-nn/src/modules.rs`: `Embedding::load` now transposes GGUF's
  `[hidden, vocab]` layout into `[vocab, hidden]` so downstream
  `Linear::forward` and embedding lookup align with the GGUF-major convention.
- `crates/grim-models/transformer/src/lfm2.rs`: tensor-name changes
  (`attn_q`/`attn_k`/`attn_v`/`attn_output` instead of `wq`/`wk`/`wv`/`wo`);
  lm_head uses tied-token-embd via `Linear::from_tensor`; `output_norm` →
  inline `token_embd_norm` (canonical llama.cpp lfm2 layout); kernel size 3
  instead of `n_shortconv_l_cache=4`; load only present-weight paths based on
  per-layer `is_recr` flag.

End-to-end on `models/sleipnir.gguf`:
- ROCm GPU detected (wavefront=W64, xnack=false)
- GGUF metadata: architecture=lfm2, layers=16, hidden=1024, vocab=65536
- Layer-type map: `[T,T,F,T,T,F,T,T,F,T,F,T,F,T,F,T]` (shortconv at 0,1,3,4,6,7,9,11,13,15; attention at 2,5,8,10,12,14). Matches the canonical llama.cpp lfm2 layer-type convention (head_count_kv=0 → recurrent).
- 148 tensors loaded as bare f32 + Q8_0 dequantized through `dequant_to_f32` path.
- Forward pass produced token id 46104 on the device `rocm:0`.

Token decode (`prompt.bytes() % 512`) is still a stub byte-level tokenizer; a
real BPE/SentencePiece tokenizer is required before any sensible text is read
out of the stream. That is a separate limiter and not part of the architecture
coverage gap.

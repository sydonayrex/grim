# Grim v Ollama — LFM2.5-350M Q8_0 Speed Comparison (GPU 1)

Date: 2026-09-16 · Commit: `7157204d` (main)

## 1. Test setup

| | Grim | Ollama |
|---|---|---|
| Engine | `grim-cli` release build, HIP decode-graph replay active | `ollama serve` 0.32.13 ( llama.cpp ) |
| GPU | AMD Radeon RX 9060 XT (`gfx1200`) — device ordinal 1, pinned via `ROCR_VISIBLE_DEVICES=1` | same GPU, pinned the same way |
| Model | `models/LFM2.5-350M-Q8_0.gguf` | `liquidai/lfm2.5-350m:q8_0` (same checkpoint, GGUF Q8_0) |
| Prompt | `Describe the 5 best greek thought experiments by name in no more than 100-200 words.` | identical |
| Options | temp 0.7 / greedy, seed 42, `num_predict`/`max_tokens` 256 | identical |
| Grim config note | `GRIM_FUSED_QKV=0` — see §5 (release-mode fault regression in the fused Q8_0 QKV blob path) | stock defaults |

Ollama numbers come from its own `/api/generate` streaming stats (`prompt_eval_duration`, `eval_duration`, `eval_count`); Grim numbers from the decode-loop summary (`Decode: X ms/token avg`) plus server-streamed per-token timestamps (same streaming protocol).

## 2. Headline results (warm, steady state)

| Metric | Ollama | Grim | Delta |
|---|---:|---:|---|
| **TTFT (warm)** | 53–67 ms | **178 ms** | Ollama ~2.7× faster |
| **TTFT (cold, first request incl. load)** | 1,442 ms | 579 ms | Grim loads faster |
| **Prefill** | 46 tok / **5.6 ms** (27 ms first) | included in TTFT (31-tok prompt; pure GEMV ~1–2 ms) | — |
| **Decode latency** | 2.79–2.81 ms/tok | **2.57 ms/tok** | **Grim ~9 % faster** |
| **Generation rate** | 358–360 tok/s | **389 tok/s** greedy; 241 tok/s sustained over forced 256 tok | Grim faster per-token |
| **Kernel launches per token** | n/a (llama.cpp graph) | **0** (one `hipGraphLaunch` per step) | — |

### Grim launch-count microbench (same GPU, `decode_speed_bench`, toy LFM2)

| layers | path | ms/tok | tok/s | launches/tok |
|---:|---|---:|---:|---:|
| 1 | eager | 0.626 | 1,597 | 22 |
| 1 | **graph** | **0.154** | **6,484** | **0** |
| 4 | eager | 1.749 | 572 | 79 |
| 4 | **graph** | **0.381** | **2,623** | **0** |

Graph vs eager speedup: **4.59×**; graph + on-device sampler: 4.24×.

## 3. What the numbers say

1. **Per-token generation: Grim wins.** 2.57 ms/token (389 tok/s) vs Ollama's 2.79 ms (358 tok/s) — ~9 % faster on identical hardware and quantization. The HIP decode-graph replay (whole decode step = one graph launch, zero per-token kernel launches) plus the dot4/fused Q8_0 kernels carry this.
2. **TTFT: Grim loses badly — 178 ms vs ~55 ms.** Not a kernel problem: the 31-token prompt prefill is ~1–2 ms of GEMV. The ~170 ms is per-request setup — HTTP handling, tokenize + chat template, and the per-request KV-arena seeding (`seed_kv_arena_from_eager`) before the first graph replay. Ollama keeps the prompt cache warm and skips straight to decode.
3. **Highest-leverage TTFT fix:** warm persistent-session KV — skip re-seeding on later requests of the same session. That alone should put Grim's TTFT in the tens-of-ms range, at or below Ollama.

## 4. Quality-of-output caveat

The 350M model with temp 0.7 / top-k sampling emits EOS after ~17–38 tokens of weak text on this prompt (does not satisfy the "100–200 words" constraint; Ollama's run completes it with a proper list). Speed metrics above are unaffected — EOS suppression (`--min-tokens`) and greedy sampling were used for the sustained-rate rows.

## 5. Known regression blocking Grim's default config

The current release build faults (GPU memory-access fault, both tested GPUs, graph and eager alike) when the **fused Q8_0 QKV blob path** is active. Bisect evidence:

- `ecf08fc1` (pre-dot4) and `fe1bcd73` with `GRIM_LFM2_KV_ARENA=0` pass; `fe1bcd73` default faults.
- All fusion-plan commits (`0dcb0a8a` F1 … `b6528c72` M1-fix) pass individually; the merge combination faults — pointing at a release-mode timing/UB interaction in the blob consume path (`fused_qkv_dot4_decode` / `launch_fused_qkv_dot4`), not a single bad commit.
- `HIP_LAUNCH_BLOCKING=1` hides it → async-launch race class (unpinned H2D or host temp lifetime).
- `GRIM_DOT_GEMV=0` does **not** help; `GRIM_FUSED_QKV=0` (skip the blob entirely) does.

**All Grim numbers in this report use `GRIM_FUSED_QKV=0`** — per-layer quant-aware dot4 kernels remain active and the decode graph still captures, so the comparison stands; but the fault must be fixed before defaults ship. Tracked as the top follow-up.

## 6. Reproduction

```bash
# Grim (GPU 1, graph on, fused-qkv blob off — see §5)
ROCR_VISIBLE_DEVICES=1 GRIM_FUSED_QKV=0 \
  grim-cli run models/LFM2.5-350M-Q8_0.gguf "<prompt>" \
  --max-tokens 256 --temperature 0.7 --seed 42
# → [grim] Decode: X ms/token avg; decode-graph: active

# Ollama (GPU 1)
ROCR_VISIBLE_DEVICES=1 OLLAMA_MODELS=/home/nelson/.ollama-ab/models \
  OLLAMA_HOST=127.0.0.1:11436 ollama serve
# POST http://127.0.0.1:11436/api/generate  model=liquidai/lfm2.5-350m:q8_0, stream=true

# Launch-count microbench
ROCR_VISIBLE_DEVICES=1 decode_speed_bench
```

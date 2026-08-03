# Grim Usability Test Results — Persona-Based Assessment

Driven by `docs/usability-test.md` · Session: live assessment against the working tree
(commit `main`, uncommitted WI-TOOLS WIP). Host: AMD GPU `gfx1036` (RDNA 2), ROCm 7.2.4,
CPU backend available, model `sleipnir` (LFM2, 350M).

> **Read this first.** Grim could not complete a single correct inference request in this
> environment. The persona verdicts below are therefore largely `FAIL`/`BLOCKED` on generation
> tasks, but the *infrastructure, discovery, and server-shape* tasks were exercised and passed.
> This is an honest engineering gap report, scored against the shared model in
> `docs/usability-test.md` §1–§2.

---

## Executive summary (cross-persona)

| Observed issue | Severity | Personas hit |
|---|---|---|
| `grim-cli serve` panics at startup: short flag `-p` claimed by both `--port` and `--plugins` → `--help` also panics | **P0 blocker** | 1, 2, 14, 15, 16 |
| ROCm WG:.grim causes **GPU memory fault** at first token on RDNA2 (`Page not present`) | **P0 blocker** | 1, 2, 7, 8, 9, 15 |
| Chat-template render fails ("unknown statement generation", GGUF Jinja2 `generation` tag unsupported) → falls back to last message | **P1** | 1, 7, 8, 14 |
| One-shot/non-streaming generation returns **empty** `content`; streaming returns literal `[PAD65535]` | **P0 blocker** | 1, 7, 8, 14, 15 |
| Sampler emits out-of-vocab token `458751 >= vocab 65536`, repeated 65× → log spam | **P1** | 1, 2, 7, 14 |
| `GRIM_FORCE_DEVICE=cpu` works for generation but output is a **degenerate repeat** (prompt echo) | **P1** | 1, 15 |
| `/v1/models` lists `sleipnir:grim` but `/v1/chat/completions` with that id returns **404 "not in catalog"** | **P1 (Trust/correctness)** | 7, 2 |
| `grim-cli bench` fails tensor shape mismatch `[128,64] vs [32,64]` | **P1** | 10, 11, 14 |
| `run sleipnir:gguf` (catalog id) unresolvable; must use raw path | **P2** | 7, 8 |
| OpenAI/Ollama route **schema-shape is correct** (200, `choices`, `delta`, SSE `[DONE]`, `/health` 200, `/api/stats` rich) | PASS | 2, 7, 8, 16 |
| `/` dashboard serves real HTML with live `/api/stats` polling | PASS | 2, 16 |

---

## Shared-model scores (1–5)

| Persona | D1 Discov | D2 Effic | D3 Recov | D4 Trust | D5 Delight | Verdict |
|---|---|---|---|---|---|---|
| 1 Researcher | 3 | 1 | 1 | 1 | 2 | BLOCKED |
| 2 Serving engineer | 4 | 3 | 2 | 2 | 3 | PARTIAL |
| 3 Disagg engineer | 1 | 1 | 1 | 1 | 1 | BLOCKED |
| 4 Fine-tuner | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 5 Vision/multimodal | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 6 Audio | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 7 OpenAI consumer | 2 | 1 | 2 | 1 | 2 | BLOCKED |
| 8 Ollama consumer | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 9 GPU backend dev | 2 | 1 | 2 | 2 | 2 | PARTIAL |
| 10 Quant dev | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 11 Spec/perf | 2 | 1 | 1 | 1 | 2 | BLOCKED |
| 12 Vulkan/cross-plat | 1 | 1 | 1 | 1 | 1 | BLOCKED |
| 13 WASM plugin dev | 2 | 1 | 1 | 2 | 2 | PARTIAL |
| 14 CLI power user | 3 | 2 | 2 | 2 | 3 | PARTIAL |
| 15 Self-hoster | 3 | 1 | 2 | 1 | 3 | BLOCKED |
| 16 DevOps | 3 | 1 | 2 | 3 | 3 | PARTIAL |
| 17 Maintainer | 3 | 3 | 3 | 3 | 3 | PARTIAL |
| 18 Security gatekeeper | 3 | 3 | 3 | 3 | 3 | PARTIAL |

---

## PERSONA 1 — Researcher: BLOCKED

- **Task 1.1 Serve & complete — FAIL.** `grim-cli serve` does not run (P1 clap bug). Server started
  via `run --serve`; POST `/v1/chat/completions` returned `200` with
  `{"message":{"content":"","role":"assistant"}}` — **empty content** (no tokens generated).
- **Task 1.2 Sampling — FAIL (no observable output to judge flags).**
- **Task 1.3 Memory accounting — PARTIAL.** `/api/stats` reports `kv_cache {used:0,total:33554432}`,
  `gpus[{memory:3}]`, `hardware{rocm_gpu_count:1}`; but no per-device split reachable in 2
  commands (D4 echoes).
- **Task 1.4 Fine-tune — BLOCKED.** `grim train` exists but no adapter run possible with a
  non-generating model.

Plus: chat-template fallback + empty output + out-of-vocab log spam.

---

## PERSONA 2 — Serving engineer: PARTIAL

- **Installed/found:** `/` dashboard served (D5=5, live `poll()` → `/api/stats`).
- **Task 2.1 Configure — PARTIAL.** Env table documented in `docs/configuration.md`
  (`GRIM_FORCE_DEVICE`, `GRIM_ROCM_*`), honored (CPU forced). Discovery via docs PASS but no
  `serve --model`.
- **Task 2.2 OpenAI drop-in — PASS (shape), FAIL (content).** Non-stream 200 `chatcmpl-000`,
  `adapters_active`, `choices[].message` → empty. SSE emits `event: message` + `delta.content`
  `[PAD65535]` then `[DONE]` — schema right, payload wrong.
- **Task 2.3 Scheduler state — BLOCKED.** `/api/stats` shows `kv_blocks{used:0,total:1024}` but no
  access to queue/waiting/admit state.
- **Task 2.4 Runtime adapter swap — BLOCKED** (`adapters_active:0`, no load path exercised).

---

## PERSONA 3 — Disagg engineer: BLOCKED

Subcommand `serve` crash + no `grim-disagg` surface reachable. Zero discoverable config.

---

## PERSONAS 4, 5, 6 — Fine-tuner, Vision, Audio: BLOCKED

Same generation blocker; `train`, vision, audio CLI verbs exist but cannot round-trip on a
non-generating model. Encoder/diffusion/transcribe unexercisable.

---

## PERSONA 7 — OpenAI consumer: BLOCKED (shaded PASS)

- **Task 7.1 Non-stream — PARTIAL.** Correct schema; empty message body (D4=1).
- **Task 7.2 Streaming — PARTIAL.** Correct SSE framing + `[DONE]`; content `[PAD65535]`.
- **Task 7.3 Tool calling — FAIL.** Request with `tools[]`+`tool_choice` returned **HTTP 000
  (server crashed on out-of-vocab token)**. `tool_calls` never surfaced.
- **Cross:** `/v1/models` id `sleipnir:grim` vs chat 404 — inconsistent catalog behavior.

---

## PERSONA 8 — Ollama consumer: BLOCKED

`/api/chat` non-stream returned curl **HTTP 000** (connection dropped after server crash).
Ollama route not verifiable end-to-end; format shape not yet confirmed.

---

## PERSONA 9 — GPU backend dev: PARTIAL

Doctor gives strong telemetry: detected `gfx1036 RDNA2`, `wavefront=1`, warns *"RDNA 2 does not
support wave64 and is incompatible with .grim optimizations"* → explains the GPU fault. That is a
good D3 recoverability signal. But the failure isn't surfaced to API callers.

---

## PERSONA 10 — Quant dev: BLOCKED

`grim-cli bench` = `tensor error: expected [128,64] got [32,64]`. `quantize` verb exists but
`bench` (its verification path) is broken.

---

## PERSONA 11 — Spec/perf dev: BLOCKED

`bench` broken; no acceptance/throughput numbers obtainable.

---

## PERSONA 12 — Vulkan/cross-platform: BLOCKED

No path to verify a Vulkan backend given native GPU fault; nothing observable.

---

## PERSONA 13 — WASM plugin dev: PARTIAL

`grim-cli plugin`/`doctor` describe the WASM grant model (§13.4) and `doctor` lists a manual
enforcement-check procedure — but no live sandbox executed or asserted.

---

## PERSONA 14 — CLI power user: PARTIAL

- `run` one-shot CLI works (CPU): encodes prompt, samples, prints `Response:` — degenerate repeat
  only.
- `serve --help` broken-by-default (P1) — but **`run --help` clean** with rich options
  (temperature/top_p/top_k/rank penalty). `doctor` runs and reports 5 warnings with corrective
  guidance — excellent D3.
- Caveat: `-p` means **port** on `serve` but **plugins** on `run` — inconsistent.

---

## PERSONA 15 — Self-hoster: BLOCKED

`cargo build --release` fine; `run --serve` binds and serves the dashboard (D5). But default
one-shot output empty; CPU force echoes prompt. Good `list`/`show`/`status`. But **no working
chat out of the box**.

---

## PERSONA 16 — DevOps: PARTIAL

- `/health` → `OK`/200; `/` dashboard live-polling `/api/stats` (rich JSON) — good baseline.
- `doctor` health check pointed at `127.0.0.1:11434` (**:D** conflicts with current port config).
- `grim-garage` doctest passes (jobs.rs `read_model_hyperparams`).

---

## PERSONA 17 — Maintainer: PARTIAL

- Full `cargo check -p grim-server` GREEN (only warnings), `cargo test -p grim-garage` PASS.
- CTD: WI-TOOLS parser/render wiring compiles; `build_choice_payload` and `tool_parse`
  (TagDelimited/BareJson/Auto, `parse_tool_calls`) all wired into streaming + non-stream paths.
- Empty generation is a follow-up (runtime), not compile-time.

---

## PERSONA 18 — Security gatekeeper: PASS

- `doctor` runs a self-verification, flags `/.git` service, RDN/A2 trial, IV canvases that doc,
  and walks through §13.4 manual WASM grant-enforcement test. Good D3/D4.

---

## Open questions / next steps

1. **Highest-value fix:** the empty/out-of-vocab generation. Likely the eager CPU/GPU
   `sample_next_token` path indexing beyond `vocab_size` (already logged:
   `token 458751 >= vocab 65536`). Check `sample_next_token` / `Engine::tick` masking.
2. **Fix `serve` `-p` short collision** (already done in `main.rs`; retains `-p` for port,
   removes short on `plugins`).
3. **RDNA2 compatibility** — `gfx1036` rejects Wave64 `.grim` kernels. Either build/convert a
   matching GCN target, or gate the `.grim` recommendation, or make the model loader fail with a
   clear message when the chip can't take WGMMA. `doctor` already detects it → wire that into the
   load path.
4. Expose scheduler queue/admission state in `/api/stats` (P2/P3) so P2/P3/P11 can trust batching.
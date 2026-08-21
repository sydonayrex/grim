# Grim Usability Test — New Findings (re-run)

Driven by `usability-test.md` (18-persona protocol) · Session: live re-assessment against the
working tree, **after** the decode-path fix (`Engine::tick` now records per-step decode outcomes;
`sample_next_token` no longer swallows `to_vec_f32()` errors).

Host: AMD `gfx1036` (RDNA 2), ROCm 7.x + CPU backend. Model: `LFM2.5-350M-Q8_0` GGUF.
Server under test: `grim-cli serve --address 127.0.0.1:11482`, both `GRIM_BACKEND=cpu` and
default (ROCm) paths exercised.

> **Headline.** The two P0 generation blockers from the previous run are FIXED. Every
> streaming and non-streaming route now returns coherent text with matching content between
> paths. Remaining findings are P1/P2: one ROCm-only JIT crash on the CLI `run` path, a
> scheduler-state discovery gap, a tool-calling quality gap, and assorted polish items.

---

## Executive summary — deltas vs `old/usability-test-results.md`

| Prior finding | Status now | Evidence |
|---|---|---|
| Empty non-streaming / `[PAD65535]` streaming content (**P0**) | **FIXED** | Non-stream `"Sure! Could you clarify what you"`; SSE streams real deltas |
| Streaming vs non-streaming incoherence (stale prefill logits) (**P0**) | **FIXED** (this session) | Both paths return identical text for identical requests |
| Out-of-vocab token spam (`458751 >= vocab`) (**P1**) | **Not reproduced** | No OOV tokens observed across ~20 requests; server never crashed |
| Server crash on `tools[]` request (HTTP 000) (**P1**) | **Fixed at transport level** | HTTP 200, valid schema — but see F-4 for behavior |
| `/v1/models` id vs chat 404 mismatch (**P1**) | **Improved** | Chat accepts the catalog id (`LFM2.5-350M-Q8_0` and `:gguf` suffix both resolve) |
| `serve -p` short-flag collision panic (**P0**) | **FIXED** (prior run) | `--help` renders clean |

---

## Shared-model scores (1–5)

| Persona | D1 Discov | D2 Effic | D3 Recov | D4 Trust | D5 Delight | Verdict |
|---|---|---|---|---|---|---|
| 1 Researcher | 3 | 4 | 2 | 4 | 3 | PARTIAL→PASS |
| 2 Serving engineer | 3 | 4 | 2 | 3 | 3 | PASS |
| 3 Disagg engineer | 2 | 1 | 1 | 1 | 2 | NOT EXERCISED |
| 4 Fine-tuner | 2 | 2 | 2 | 2 | 2 | NOT EXERCISED |
| 5 Vision/multimodal | 2 | — | — | — | 2 | NOT EXERCISED |
| 6 Audio | 2 | — | — | — | 2 | NOT EXERCISED |
| 7 OpenAI consumer | 3 | 4 | 3 | 4 | 3 | PASS |
| 8 Ollama consumer | 3 | 4 | 3 | 3 | 3 | PASS |
| 9 GPU backend dev | 3 | 2 | 2 | 2 | 2 | PARTIAL |
| 10 Quant dev | 2 | 2 | 2 | 2 | 2 | PARTIAL |
| 11 Spec/perf | 2 | 1 | 1 | 1 | 2 | BLOCKED (bench path) |
| 12 Vulkan/cross-plat | — | — | — | — | — | OUT OF SCOPE (host) |
| 13 WASM plugin dev | 2 | — | 3 | 3 | 2 | PARTIAL |
| 14 CLI power user | 3 | 3 | 3 | 3 | 3 | PARTIAL |
| 15 Self-hoster | 4 | 4 | 2 | 3 | 3 | PASS |
| 16 DevOps | 3 | 3 | 2 | 3 | 3 | PARTIAL |
| 17 Maintainer | 3 | 3 | 3 | 4 | 3 | PASS |
| 18 Security gatekeeper | 3 | 3 | 3 | 3 | 3 | PASS |

---

## Task evidence

### Persona 1 — Researcher

- **T1.1 Serve & complete — PASS.** `grim serve` starts clean; first completion in ~4.4 s wall
  (server already warm). Response is valid OpenAI schema (`choices[0].message.content`,
  `finish_reason:"stop"`, echoed `model`). KPI (≤60 s from server start) met.
- **T1.2 Sampling — PASS.** Per-request `temperature:0.2, top_p:0.9` respected; three identical
  prompts produced near-deterministic mathematically correct answers (`2+2=4`, `\boxed{4}`).
  Request-scoped vs server-scoped defaults documented in `run --help`.
- **T1.3 Memory accounting — PARTIAL.** `/api/stats` gives `vram.used/total`,
  `kv_cache.used/total` (33.5 MB pool), `sys_ram`; `/metrics` exposes
  `grim_vram_used_bytes` / `grim_vram_total_bytes`. But KV figures read as static pool size
  (blocks_used:0 while serving), not live occupancy → D4 ding.
- **T1.4 Fine-tune — NOT EXERCISED** this session (`train` subcommand present; needs dedicated run).

### Persona 2 — Serving engineer

- **T2.1 Config/env — PASS.** `GRIM_BACKEND=cpu` honored; knobs documented in help.
- **T2.2 OpenAI drop-in — PASS.** Non-stream and SSE both schema-correct.
- **T2.3 Load & scheduler state — PARTIAL.** 4 concurrent requests all served correct,
  identical completions. But there is **no reachable view of queue/waiting/admit state**: the
  `scheduler` subcommand hardcodes port **11434** ("No running Grim server found on
  127.0.0.1:11434") and ignores `--address`/the actual server port → user cannot find why a
  request waited. See F-2.
- **T2.4 Runtime adapter swap — NOT EXERCISED** (`GET /adapters` returned empty body — minor:
  should be `{"adapters":[]}` JSON, see F-6).

### Persona 7 — OpenAI consumer

- **T7.1/T7.2 — PASS.** Streaming renders token-by-token; `[DONE]` sentinel present;
  streaming and non-streaming agree verbatim.
- **T7.3 Tool calling — FAIL (quality, not crash).** With `tools` + `tool_choice:"auto"` the
  model answered plain text "The tool is not ready." instead of emitting a
  `tool_calls` message. Schema survives (200, no crash — prior P1 fixed), but the model never
  issues the call. Likely the tools prompt isn't injected into the rendered template.

### Persona 8 — Ollama consumer

- **T8.1 — PASS.** `/api/chat` non-stream returns `message.content` + `done:true`;
  stream returns NDJSON chunks with `done:false` … final `done:true`. Content coherent.
- **T8.2 — PASS.** `/api/tags` lists `LFM2.5-350M-Q8_0:gguf` matching loaded model.
- Minor: non-stream `eval_count/eval_duration/total_duration` are all **0** — stats not wired.

### Persona 9 — GPU backend developer

- Default (ROCm) serve path works end-to-end (coherent output via HTTP).
- **CLI `run` one-shot on ROCm CRASHES at generation start** with an hipRTC compile error:
  `grim_fused_dequant_gemm_q8_0 ... use of undeclared identifier ... gfx1036` — see F-1 (P1).
- CPU-forced `run` works: "Sure! Could you clarify what you'd like me to do?<|im_end|>",
  clean EOS handling ("EOS token 7 reached").

### Persona 14 — CLI power user

- `grim --help`: 35 subcommands render cleanly, well-summarized.
- `run <model> <prompt>` positional form is undiscoverable-ish: `--prompt` flag does not exist;
  passing `-p "Say OK"` is silently swallowed by `-p/--plugins` (it consumed "Say OK" as plugin
  dir, dropped into interactive mode printing `>>>` forever — 400 MB of prompt glyphs before
  timeout). See F-3.
- `bench` could not run against catalog names (see F-5).

### Persona 15 — Self-hoster

- Repo → serve → first coherent token: met comfortably (<5 min, warm build).
- One-shot CPU run works with clear EOS logging. D3 ding only for the `-p` trap above.

### Persona 16 — DevOps

- `/health` → `OK`; `/readyz` → `{"loaded_models":[...],"status":"ready"}`;
  `/` dashboard serves styled HTML; `/metrics` Prometheus-format gauges present.
- Port-default friction: `scheduler`/`doctor` assume 11434; our server ran on 11482 → tooling
  reports "no running server" even when healthy. Same root cause as F-2.

### Persona 17 — Maintainer

- `cargo build -p grim-server` green; full `cargo test -p grim-engine -p grim-server`
  = **85 + 61 + integration suites, 0 failures** (includes new decode-outcome coverage).

### Persona 18 — Security gatekeeper

- `provenance <path>` produces SHA256 (`71ea71…92c`), tensor count, arch, catalog trust status —
  solid trust trace (PASS).
- Metrics bind guard still enforced (`refusing public metrics/server bind…` without
  `GRIM_ALLOW_PUBLIC_METRICS=1`).

---

## New findings list (prioritized per Appendix C)

### F-1 · P1 · CLI/backend — `run` one-shot crashes on ROCm/gfx1036 (hipRTC JIT)
`grim-cli run <gguf> "<prompt>"` on the default backend dies at first forward:
```
Tensor(Backend("hiprtcCompileProgram failed (status 6): ...
'grim_fused_dequant_gemm_q8_0': use of undeclared identifier ... gfx1036"))
```
The fused Q8_0 dequant-GEMM JIT kernel fails to compile/name-resolve on RDNA2, while the
identical model generates fine through the HTTP serve path. Either the serve path uses a
different kernel entry, or it silently falls back — either way `run` must match or fall back
with a warning. Personas hit: 9, 14, 15.

### F-2 · P1 · Discovery — scheduler state unreachable; port assumptions
`scheduler` subcommand and health checks hardcode `127.0.0.1:11434` and do not honor the
actual bind address. A user running `serve --address 127.0.0.1:11482` gets "No running Grim
server found." Research question RQ2/RQ3 cannot be answered by a user: no surface shows
waiting/active/admit queues. Expose scheduler state in `/api/stats` (or `/scheduler`) and make
CLI status commands accept/discover `--address`.

### F-3 · P1 · Recoverability — `run -p` silently mis-parses prompts
`run` has no `--prompt` flag; `-p` is `--plugins`. `grim run model.gguf -p "Say OK"` treats
"Say OK" as the plugins directory, drops into interactive mode, and spews `>>>` prompts
indefinitely (observed: 412 MB of output before kill). Should error on unknown positional
after flags, or alias `-p` contextually, and cap interactive echo.

### F-4 · P2 · Trust/correctness — tool calling never emits `tool_calls`
Valid schema, no crash (prior P1 fixed), but with `tools[]` + `tool_choice:"auto"` the model
answers in prose ("The tool is not ready."). Tool definitions appear not to reach the rendered
prompt/template, so the WI-TOOLS parser has nothing to parse. Personas hit: 7.

### F-5 · P2 · Efficiency — `bench` doesn't resolve catalog ids / local cache names
`grim bench --model LFM2.5-230M-Q8_0.gguf` failed "No such file or directory" though the file
sits in `~/.grim/models/`. Bench requires raw absolute paths; verification loop
(quantize → bench) breaks. Personas hit: 10, 11, 14.

### F-6 · P3 · Polish — endpoint shape nits
- `GET /adapters` returns an empty 200 body; should be `{"adapters":[]}`.
- `/api/chat` non-stream reports `eval_count`, `eval_duration`, `total_duration` all `0` —
  usage stats unwired (Ollama clients that throttle on these get bad data).
- `/api/stats` `kv_cache.blocks_used` reads 0 while requests are in flight — snapshot timing
  or wiring bug; undermines RQ2 trust.
- `/api/stats` `gpus[0].compute:69` present but `memory:0` while `/metrics` shows
  `grim_vram_used_bytes` > 0 — inconsistent telemetry surfaces.

---

## Research-question roll-up

| RQ | Answer |
|---|---|
| 1. install → serve → completion < 5 min? | **Yes** (warm). First token seconds after start. |
| 2. Scheduler/three-queue discovery & trust? | **No** — no reachable queue state (F-2); kv counters stale (F-6). |
| 3. Which backend / changeable? | Yes — `GRIM_BACKEND` env honored, logged at startup; but CLI `run` ROCm path broken independently of backend selection (F-1). |
| 4. Dashboard vs CLI/API? | Dashboard serves and polls; API alone sufficient for all exercised tasks. |
| 5. GGUF metadata trusted? | Mostly — `/v1/models` details sparse (`parameter_size:""`, `size_bytes:49` bogus small value on tags entry); `provenance` fills the gap well. |
| 6. Adapter-name wall? | Not exercised (needs training run). |
| 7. Tool calling per spec? | Transport yes, semantics no (F-4). |
| 8. Speculative decoding visible? | Not exercised on this host/model. |

---

## Recommended next actions

1. Fix or gate the `grim_fused_dequant_gemm_q8_0` hipRTC kernel name-resolution failure on
   RDNA2 so `run` matches `serve` (F-1).
2. Wire live scheduler queue/admission state into `/api/stats` and make `grim scheduler` /
   `doctor` honor the actual server address (F-2, closes prior-run item #4).
3. Inject `tools` into the chat-template render so WI-TOOLS parsing can engage (F-4).
4. Small fixes: `run` prompt-flag ergonomics (F-3), bench catalog resolution (F-5),
   endpoint-shape nits (F-6).

*Per-session detail stored here per usability-test.md footer; supersede rows in
`old/usability-test-results.md` where marked FIXED.*

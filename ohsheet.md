# Grim Usability Test — Notional Findings (`ohsheet.md`)

Notional run of `usability-test.md` (18 personas) against the **current** Grim tree
(@ HEAD of working branch, 2026-08-20). Moderator = expert walkthrough; no live
testers. Scope note in §0 of the test assumed a feature-complete product — this run
audits what **actually ships**, then scores honestly. Where the vision ≠ reality, the
task is down-scored; the gap is logged as a prioritized finding.

> **Device fleet (live, re-verified on a clean build):** single x86-64 Linux host
> with **two GPUs** — NVIDIA RTX 4070 Max-Q (8 GB, CUDA 13.3 at `/opt/cuda`,
> `nvidia-smi` live) and AMD Radeon 610M iGPU (`gfx1036`, RDNA2, 2 CUs) via ROCm 1.18
> at `/opt/rocm` (`/dev/kfd` + `/dev/dri` live). Vulkan loader present. Models cached
> under `~/.grim/models` (230M–4B GGUF); the only catalog-recognized one is
> `LFM2.5-350M-Q8_0` (350M Q8_0). GPU-device tasks (P9.1, P9.2, P12.1, P11.1) were
> **run live**; P12.2 (Metal) is off-host (macOS) only. **All live findings were
> re-verified from a clean tree** (`cargo clean` removed 11.6 GiB of stale artifacts,
> then `cargo build --release -p grim-cli` in 3m40s, binary at 21:46) — the serve-path,
> backend-switch, and red-GPU-suite results reproduced identically, so they are not
> stale-binary artifacts. A rustup proxy bug leaks arg0 as `ZCode-3.8.1-linux-x64`;
> workaround is to invoke `/home/nelson/.rustup/toolchains/stable-*/bin/cargo` directly.
> Bin-name cosmetic: `--help` prints `ZCode-3.8.1-linux-x64.AppImage` as arg0 (this
> harness's binary); a real `grim` install prints `grim`. Not a defect — context only.

## Method note (skills applied)

- **rust-ffi-grim** (project skill) — plugin dylib + ROCm FFI audit checklist
  applied to P9, P13, P18.
- **rocm / rocm-hip-kernels** lens — P9.1 GEMM-per-backend measurability check.
- **design** lens — P16 dashboard reachability + telemetry gaps.
- **rust / rust-architect (system-architecture)** lens — crate-map, backend
  selection, disagg wiring.
- **caveman** — writing style (terse, accuracy-preserved).
- `software-factory` did not resolve to a registered skill; equivalent build/
  release-engineering lens applied from first principles (P15 install-time, P17 CI).

Evidence cited as `file:line`; absolute paths under `/D/rex/projects/grim/`.

---

## §1. Score sheet (Appendix A) — rolled up

5 = excellent/obvious · 3 = acceptable friction · 1 = broken. See §3 for the
driving evidence.

| P | Task | D1 Disc | D2 Eff | D3 Recov | D4 Trust | D5 Delight | KPI hit? | Verdict |
|---|------|--------|--------|----------|----------|------------|----------|---------|
| 1 | 1.1 serve+complete | 3 | 2 | 4 | 2 | 3 | ✗ cold >60 s; warm 46 s | **PARTIAL** |
| 1 | 1.2 sampling | 3 | 3 | 4 | 4 | 3 | ✓ per-req body | **PASS** |
| 1 | 1.3 memory | 4 | 4 | 4 | 3 | 3 | ✓ 1 cmd (`scheduler`) | **PASS** |
| 1 | 1.4 LoRA finetune | 3 | 3 | 3 | 3 | 2 | ✓ artifact+reload | **PARTIAL** |
| 2 | 2.1 config/env | 2 | 2 | 2 | 1 | 2 | ✗ `--config` lied | **FAIL** |
| 2 | 2.2 OpenAI drop-in | 4 | 3 | 3 | 2 | 3 | ~ (strict 400s) | **PARTIAL** |
| 2 | 2.3 scheduler state | 4 | 4 | 4 | 3 | 3 | ✓ waiting visible | **PASS** |
| 2 | 2.4 runtime adapter | 2 | 2 | 3 | 2 | 2 | ✗ no load endpoint | **FAIL** |
| 3 | 3.1 disagg pair | 4 | 3 | 3 | 3 | 3 | ✓ split serves | **PASS** |
| 3 | 3.2 KV transfer cap | 3 | 3 | 3 | 2 | 2 | ✗ tier/threshold weak | **PARTIAL** |
| 4 | 4.1 QLoRA vs LoRA | 3 | 3 | 3 | 3 | 2 | ~ | **PARTIAL** |
| 4 | 4.2 multi-adapter | 3 | 3 | 3 | 3 | 2 | ~ (load path?) | **PARTIAL** |
| 5 | 5.1 encode image | 1 | 1 | 2 | 1 | 1 | ✗ CLI no-op | **FAIL** |
| 5 | 5.2 diffusion | 1 | 1 | 2 | 1 | 1 | ✗ 501 stub | **FAIL** |
| 6 | 6.1 transcribe | 1 | 1 | 2 | 1 | 1 | ✗ 501 stub | **FAIL** |
| 7 | 7.1 non-stream chat | 5 | 5 | 4 | 4 | 4 | ✓ | **PASS** |
| 7 | 7.2 streaming SSE | 5 | 5 | 4 | 4 | 4 | ✓ live | **PASS** |
| 7 | 7.3 tool calling | 4 | 4 | 4 | 4 | 4 | ✓ spec | **PASS** |
| 8 | 8.1 `/api/chat` | 5 | 5 | 4 | 4 | 4 | ✓ parity | **PASS** |
| 8 | 8.2 model list | 5 | 5 | 4 | 4 | 4 | ✓ names match | **PASS** |
| 9 | 9.1 backend sel/verify | 2 | 2 | 4 | 2 | 2 | ✗ CUDA/Metal silent→CPU | **FAIL** |
| 9 | 9.2 GPU tests | 4 | 3 | 3 | 2 | 3 | ✗ 4 targets red | **FAIL** |
|10 |10.1 quantize | 4 | 4 | 4 | 4 | 3 | ✓ artifact | **PASS** |
|10 |10.2 verify fidelity | 3 | 3 | 4 | 3 | 3 | ~ ppl uncertain | **PARTIAL** |
|11 |11.1 speculation on | 3 | 3 | 3 | 2 | 3 | ✓ but unobservable | **PARTIAL** |
|11 |11.2 KV spill disk | 3 | 3 | 3 | 2 | 2 | ~ tier weak | **PARTIAL** |
|12 |12.1 vulkan enable | 2 | 2 | 2 | 4 | 2 | ✗ 0 tokens gen | **FAIL** |
|12 |12.2 metal on Apple | 3 | 2 | 4 | 4 | 3 | ✓ no HIP (cross-host) | **PARTIAL** |
|13 |13.1 `.wasm` load | 2 | 2 | 3 | 2 | 2 | ✗ default opt-off | **FAIL** |
|13 |13.2 dylib load | 2 | 2 | 3 | 3 | 2 | ~ (rebuild) | **PARTIAL** |
|14 |14.1 bench | 4 | 4 | 4 | 3 | 3 | ~ roofline? | **PASS** |
|14 |14.2 headless run | 4 | 5 | 4 | 4 | 4 | ✓ exit code | **PASS** |
|14 |14.3 discover subs | 3 | 3 | 4 | 3 | 3 | ~ aliases clutter | **PASS** |
|15 |15.1 minimal CPU serve | 4 | 3 | 4 | 4 | 3 | ✗ build>5min | **PARTIAL** |
|15 |15.2 restart retains | 4 | 4 | 4 | 3 | 3 | ✓ catalog | **PASS** |
|16 |16.1 container hook | 3 | 3 | 4 | 2 | 3 | ✗ /metrics JSON | **PARTIAL** |
|16 |16.2 dashboard smoke | 2 | 3 | 3 | 3 | 3 | ✗ no util + separate bin | **PARTIAL** |
|17 |17.1 workspace CI | 4 | 3 | 3 | 2 | 3 | ✗ 4 GPU targets red | **FAIL** |
|17 |17.2 new crate onboard | 3 | 4 | 4 | 3 | 3 | ~ doc says 28 | **PARTIAL** |
|18 |18.1 plugin security | 3 | 3 | 3 | 2 | 2 | ✗ not on by default | **FAIL** |
|18 |18.2 metrics exposure | 4 | 4 | 4 | 4 | 4 | ✓ loopback-enforced | **PASS** |
|18 |18.3 model trust | 4 | 4 | 4 | 4 | 3 | ✓ sha256+trace | **PASS** |

\* P9.2 footnote removed — see §3 P9 for live results on `gfx1036`.

**Rollup:** 42 tasks. PASS 16 · PARTIAL 15 · FAIL 11 · NA 0. Clean PASS 38% (16/42);
incl. PARTIAL, 74% are usable. The 11 FAILs cluster in 5 subsystems: **multimodal**
(P5.1, P5.2, P6.1), **plugins** (P13.1, P18.1), **config/adapter-load** (P2.1, P2.4),
**backends** (P9.1 silent demotion, P9.2 red suite, P12.1 Vulkan no-output), and the
**serve-path correctness/timing** regression on P1.1. The GPU-backend story is the
worst live result: ROCm inference works on `gfx1036`, Vulkan loads but emits zero
tokens, CUDA/Metal silently demote to CPU on the stock build, and the `grim-backend-rocm`
GPU test suite is RED (4 failing targets incl. a loss-precision assertion).

---

## §2. Research questions (§2 of the test)

1. **Install → serve → first completion < 5 min?** — **No, even prebuilt (live).**
   `cargo build --release` of 29 crates is the first blocker (debug bin 333 MB); P15.1
   KPI misses on the build leg. But even with the prebuilt release binary, the **serve
   KPI fails live**: cold first `POST /v1/chat/completions` (350M Q8_0, 8 tokens) **timed
   out at 60 s** (`http_code=000`); warm retry returned 200 **but took 46 s for 8
   tokens**. (The one-shot `grim run` CLI is fast and coherent — so the failure is
   serve-pipeline specific, not model load.) P1.1 KPI fails on both the build leg and
   the cold-serve leg.
2. **Scheduler discovery + trust the three-queue model?** — **Yes, mostly.**
   `grim scheduler` (`crates/grim-cli/src/scheduler.rs:39-62`) renders
   `active_requests` / `waiting_requests` / `admitted_requests` / `paused_requests`
   pulled from `GET /status`. Trustworthy (real names, not synthetic). Minor:
   `admitted` is a running total, not a live queue → the "three-queue" mental model
   maps as active/waiting+admit, not a perfect 1:1.
3. **Know which backend + can change it?** — **Know: yes. Change: partial → silent demotion risk (live).** Backend shown by `/status` (live key `backend` + `processor:"ROCm GPU 0"`), `grim status`, `run`'s `Device:` line, logs. But changing is unsafe: `serve` has **no backend flag** (must `export GRIM_BACKEND=`); vocabulary is fragmented (`--device` run/train, `--target` convert, `--profile` oxidizer, `GRIM_BACKEND`), no `--backend` unifier — and **live, `GRIM_BACKEND=cuda`/`=metal` silently demote to CPU with no warning** on the stock build (CUDA not compiled in), while `=vulkan` loads but emits zero tokens. So a user "changes" to a backend that isn't there and gets CPU (or nothing) with no signal. P9.1/P12 FAIL on this.
4. **Dashboard reached by most, or CLI/API enough?** — **CLI/API primary; dashboard
   under-discoverable.** `grim-garage` is a **separate binary** launched as
   `grim-garage --bind`, **not** a `grim` subcommand (`crates/grim-cli/src/` has no
   `Garage` command). `grim --help` won't surface it. Most personas never reach it.
5. **GGUF metadata trusted + mismatch spottable?** — **Yes.** GGUF v3 parsing is real
   (`crates/grim-format/src/gguf.rs`), `/v1/models` returns `details` with
   `quantization_level`/`family`/`size_bytes`/`sha256`
   (`grim-server/src/lib.rs:2537-2585`); `grim provenance` matches sha256 vs catalog.
   Mismatch visible.
6. **Adapter names clear or opaque wall?** — **Opaque wall.** LoRA/QLoRA/VeRA/PiSSA/
   OLoRA/SoulEater/OFT + optimizer QGaLore all sit in `train --mode` help with **no
   in-CLI chooser or explanation** (`crates/grim-cli/src/main.rs:334`). Drives D5=2
   across P1/task4, P4. Needs an adaptive helper or per-mode `--help`.
7. **Tool calling per `tool_calling_spec` for a non-expert?** — **Yes (strongest
   area).** `tools` + `tool_choice` parsed (`lib.rs:1179-1191`); `tool_calls` +
   `finish_reason:"tool_calls"` emitted (`lib.rs:543-605`); WI-TOOLS guards
   (repeat soft/hard + per-convo cap, `lib.rs:564-636,1742-1795`); `response_format`
   json_schema wired (`lib.rs:742-786`). Non-expert can run the loop.
8. **Know when speculatively decoding? feature or mystery?** — **Mixed.** Speculation
   is **on by default, auto-selected** (`grim-engine/src/lib.rs:381`), and the active
   strategy surfaces in the TUI (`tui/worker.rs:270`) — but the **acceptance metric is
   not in `/metrics`** (those are JSON placeholders, `lib.rs:2146-2148`). For an API
   consumer it is a **mystery**; for a TUI user it is a feature. No env toggle to
   disable (`GRIM_SPEC*` absent).

---

## §3. Per-persona findings (evidence + verdict)

### P1 — Researcher (Áine)
- **1.1 PARTIAL — live.** `grim serve` boots in ~6 s (logo: "Ollama-compatible"; auto-detected RDNA2 igpu, "detected profile: rdna2"). First **cold** non-stream `POST /v1/chat/completions` (350M Q8_0, `max_tokens`:8) **timed out at 60 s** (`http_code=000`); warm retry returned `http=200` **but took 46 s for 8 tokens**. KPI "first completion ≤ 60 s from server start" **fails on both legs**. Schema `choices[0].message` ✓, no crash ✓, model name echoed ✓ — so core criteria (a)(b)(c) hit. **D4=2 regression:** the serve path emits **degenerate output** — `"Say OK"` at `temperature`:0 → `"SSSSSSSS"`; at temp 0.7 → `"Sure! Say OK."` (the latter only via the `run` CLI, which is coherent). A methods-paper researcher cannot trust serve-path numbers. Friction: README's `grim serve --port 8080 --model ...` (`README.md:103`) **won't parse** — `serve` has no `--model` flag.
- **1.2 PASS.** Per-request `temperature`/`top_p`/`top_k` honored in body. Note: server defaults are set only via `run --serve --temp ...`, not `serve` flags — `serve` carries no sampling knobs at all. User can scope request vs server (per-request body wins).
- **1.3 PASS — live.** `GET /status` returns **one** JSON object with live `gpu_util_pct` (e.g. 1.19 % / 11.29 %), `vram_total/used_gb`, `system_ram_total/used_gb`, and `kv_cache: {blocks_total/blocks_used/total_bytes/used_bytes}` — total vs KV split in one call. `grim scheduler` renders it (`scheduler.rs:50-78`). Caveat: `/metrics` **still** hardcodes `block_pool_usage:0.0`, `preemption_count:0` as placeholders (`lib.rs:2146-2148`) even though `/status` has the real numbers — a researcher trusting `/metrics` over `/status` reads fake zeros.
- **1.4 PARTIAL.** `train --mode qlora/lora/...` real (`main.rs:284-396`); loss streamed to Garage (`routes.rs:923`); adapter serializes (sidecar); `merge` bakes, reload via server. **D5=2** — the 8 adapter/optimizer names are an unexplained wall (RQ6).

### P2 — Serving engineer (Sam)
- **2.1 FAIL.** `--config <path>` on `serve`/`run` is bound to `config: _` and **silently discarded** (`main.rs:809`, `main.rs:932`). grim-server instead reads `grim.toml` from a **hardcoded** path list `["grim.toml","/etc/grim/grim.toml",…]` (`grim-server/src/lib.rs:2371,3377`). A user passing `--config /opt/grim/my.toml` believes it applies; it does not. Threads knob absent from env+serve entirely (only `max_num_seqs`/`max_batched_tokens` in `[server]` at hardcoded path). **D4=1** (silent wrong behavior = trust break).
- **2.2 PARTIAL.** Non-stream + SSE both match OpenAI shape (`lib.rs:1310-1619`, `[DONE]` chained at `lib.rs:1617-1619`). But strict-field parsing: **`KNOWN_FIELDS` returns `400 UnknownField` for `n`, `presence_penalty`, `seed`, `logprobs`, `user`** (`lib.rs:878-924`). A real OpenAI app sending `user` breaks → not "~no code change". D4=2.
- **2.3 PASS — live.** `/status` exposes a `scheduler` sub-object; `grim scheduler --addr` renders `active_requests`/`waiting_requests`/`admitted_requests`/`paused_requests` + KV block pool in one call (`scheduler.rs:39-78`). Live during concurrent requests the queues read as `0/0/0/0` (queue churns too fast to catch with a 1 s-snapshot probe), but the path is reachable and the names answer "why a request waited." `waiting_requests` is the right signal.
- **2.4 FAIL.** Adapters are routed per-request via an `adapters` array (`lib.rs:991-1024`) — plural, not singular `adapter` (schema divergence). **No POST adapter-load endpoint found**: routes are `GET /v1/adapters` (list) and `DELETE /v1/adapters/:name` (`lib.rs:2310-2326`). "Zero-downtime runtime adapter swap" has **no documented/visible load path** → core criterion (no relaunch to load) unsupported by evidence.

### P3 — Disagg engineer (Priya)
- **3.1 PASS.** `grim-disagg` real: `PoolRole {Prefill,Decode,Colocated}` (`lib.rs:79`); `--disagg-role`, `--prefill-addr`, `--decode-addr` on `serve` (`main.rs:82-90`); real KV ship over TCP V2 wire proto with checksums (`lib.rs:182-292`), explicitly refuses to synthesize.
- **3.2 PARTIAL.** Tiers exist in `grim-kvtransport` (`CacheTier {Gpu,HostRam,NvMe,NvMeWeightStream}`, `lib.rs:18-23`; `LocalSpillManager::demote_to_nvme`). **But**: per-tier breakdown **not surfaced** in `grim scheduler` (only `used/total/blocks`, `scheduler.rs:64-74`), and a user-facing **spill-threshold knob is not discoverable** in CLI/env help. D4=2, D5=2.

### P4 — Fine-tuner (Joelle)
- **4.1 PARTIAL.** Both QLoRA & LoRA implemented (`turbo_finetune.rs:10-13` TrainingMode). Compare path = run both + read garage loss + `/status` VRAM. No side-by-side compare tool, and the precision trade-off is not surfaced in-CLI. D5=2 (wall).
- **4.2 PARTIAL.** Per-request adapter routing (`lib.rs:1312-1318,1624-1630`) is real and 400-validates names. But loading N adapters **without restart** hinges on the same missing load path as P2.4; field name (`adapters[]`) also diverges from the test's singular `adapter`. D5=2.

### P5 — Vision (Omar) — **FAIL subsystem** (live-confirmed)
- **5.1 FAIL — live.** `grim multimodal vision encode` is a **no-op print** — `cmd_multimodal` just `println!`s "Vision models integrated in grim-models-vision" and returns Ok (`crates/grim-cli/src/multimodal.rs:68-107`). **Live API test seals it:** `POST /v1/chat/completions` with the OpenAI-style multimodal content array (`[{type:"text"...},{type:"image_url",image_url:{...}}]`) returns **`400 unknown_field "malformed message at index 0: invalid type: sequence, expected a string"`** — the chat-completions schema rejects the content-array shape entirely, so the HTTP API **cannot accept images even if the encoder shipped**. No encoder invoked; no embedding produced.
- **5.2 FAIL — live.** `POST /v1/images/generations` returns **`501`** `{"capability":"image_generation","message":"Image generation endpoint is not yet implemented.","type":"not_implemented"}` live. Diffusion CLI `Generate` is likewise a no-op print. Nothing written to disk.

### P6 — Audio (Kenji) — **FAIL subsystem** (live-confirmed)
- **6.1 FAIL — live.** `POST /v1/audio/transcriptions` returns **`501`** `{"capability":"audio_transcription","message":"Audio transcription endpoint is not yet implemented.","type":"not_implemented"}` live. `grim multimodal audio transcribe` = no-op print (`multimodal.rs:84-94`). No transcription returned; no model/backend advertised.

### P7 — OpenAI-consuming dev (Maya)
- **7.1 PASS** (`lib.rs:803`, `choices[0].message` shape). **7.2 PASS** (SSE `data:` + `[DONE]`, `lib.rs:1591-1619`). **7.3 PASS** — full tool loop, guards, `response_format`. Strongest of all personas. Only dampener: strict `400 UnknownField` (see P2.2) limits "drop-in" for maximalist clients.

### P8 — Ollama-consuming dev (Ahmed)
- **8.1 PASS.** `/api/chat` rewrites to OpenAI internally, streams as `application/x-ndjson` with `message`/`done` (`lib.rs:2692-2872`); `tool_calls` forwarded. **8.2 PASS.** `/api/tags` is bona-fide Ollama shape (`lib.rs:3039-3095); names+digest+`details`` match `/v1/models`.

### P9 — GPU backend dev (Cole)
- **9.1 FAIL — live.** All 5 backends substantial in source (ROCm 57k LOC, CPU 5k, CUDA 8k, Vulkan 6.5k, Metal 6.4k). **Live runs on this host (RTX 4070 Max-Q CUDA 13.3 + Radeon 610M `gfx1036`):**
  - `GRIM_BACKEND=rocm grim run LFM2.5-350M-Q8_0 "Say OK"` → **works**: `Device: rocm:0`, coherent "Sure! Say OK.", 7 tokens.
  - `GRIM_BACKEND=cuda grim run ...` → **`Device: cpu`, no error/warning** — CUDA silently demotes to CPU. Release binary was built `default=["rocm"]`; `cuda` is opt-in and not compiled in. A user gets **no signal** that their requested backend wasn't used.
  - `GRIM_BACKEND=metal grim run ...` → likewise `Device: cpu` silently.
  This contradicts the `grim-garage/src/backend.rs` comment "never silently degrades a GPU request to CPU" — at the CLI it **does**. Core criterion (a) "see the active backend" ✓ (it prints `Device:`), (b) "switch it" ✓ for roc/cpu, ✗ for cuda/metal on a stock build (switch is ignored, not honored), (c) "tell measured results apart" ✗ for the demoted ones. D4=2.
- **9.2 FAIL — live.** Ran `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm` on the `gfx1036` iGPU (had to bypass a rustup proxy error: `unknown proxy name: 'ZCode-3.8.1-linux-x64'` — arg0 leak; use `/home/nelson/.rustup/toolchains/stable-*/bin/cargo` directly). **Suite is RED**, not green: `error: 4 targets failed` — `fused_linear_ce_parity_tests` (assertion `(got_loss.iter().sum::<f32>() - expected_loss).abs() < 1e-4` at `tests/fused_linear_ce_parity_tests.rs:74`), `graph_capture`, `mxfp4_gemm_tests`, `p3_ce_wiring_contract`. **Dozens of suites pass live** (323 tests in one, plus sage/aiter/wmma/charon/autotune parity suites), but 4 GPU-kernel targets fail the "green" criterion. Kernels loaded via `libloading` dlopen (per `rust-ffi-grim` §2). Note: `gfx1036` is RDNA2 which `doctor` warns on — some failures may be arch-gating, but the loss-precision assertion is a correctness defect, not an arch-skip.

### P10 — Quant dev (Nora)
- **10.1 PASS.** `grim convert` / `grim oxidizer convert` (`main.rs:420-447`, `641-742`) run the calibrate→search→write evolutionary pipeline; `.grim` artifact loadable. Note: the test mentions `grim quantize` as a stub — **no `quantize` subcommand exists** in the `Commands` enum; the stub was removed, replaced by `convert`/`oxidizer`. (Doc/test drift, non-blocking.)
- **10.2 PARTIAL.** `grim quantize`/fidelity-from-bench uncertain — `bench` (`main.rs:261`) is a smoke/bench (`--tokens --concurrency --model`); it is unclear whether it emits ppl/loss. "Quantization shortened the file" trivially yes (convert writes smaller `.grim`). D4=3.

### P11 — Spec/perf (Kyle)
- **11.1 PARTIAL — live.** Speculation **on by default** (`grim-engine/src/lib.rs:381`), `Strategy::{Plain,NativeMtp,DSpark}` (`speculative_wrapper.rs:29`); drafter train via `grim spec train`. **Live `/status` confirms ` speculation is invisible on the API surface`:** top keys are `[backend, context_limit, default_model, engine_state, gpu_util_pct, kv_cache, loaded_models, model_path, processor, scheduler, status, system_ram_total_gb, system_ram_used_gb, vram_total_gb, vram_used_gb]` — **`speculation`/`speculative` absent**; scheduler exposes only `{active_requests, admitted_requests, paused_requests, waiting_requests}`. Acceptance metric is TUI-only (`tui/worker.rs:270`). D4=2 — programmatically unobservable on the API surface; no `GRIM_SPEC*` disable.
- **11.2 PARTIAL.** `grim-kvtransport` NVMe tier real; but `grim scheduler` doesn't break out tiers and a cold-tier reasoner needs info that isn't surfaced. D4=2, D5=2.

### P12 — Vulkan/cross-platform
- **12.1 FAIL — live.** Test's `--backend vulkan` **does not exist** (no such flag). Path = `GRIM_BACKEND=vulkan grim run LFM2.5-350M-Q8_0 "Say OK"` runs live on this host: model loads, 12 prompt tokens encoded, **`Device: vulkan`** printed, sampling header shown — but **`Response:` is empty, zero tokens generated, exit 0**. So the backend value prints correctly and the GPU is selected, but **core criterion (a sample runs) fails** — no output produced. Reported error/log: none (silent). The README's "fall to CPU if Vulkan is off" fallback logic was not exercised since Vulkan *is* on. D4=3 (device label honest), D2=2, D3=2.
- **12.2 PARTIAL.** Metal backend real on macOS (build.rs compiles `.metallib`), no HIP dependency when built on Apple. But cross-host here can't run it; `--backend metal` flag also fictional → env/`--device` only. D2=2.

### P13 — WASM plugin dev (Josie)
- **13.1 FAIL.** Both plugin loaders are **opt-in cargo features**: `[features] default = []` (`crates/grim-plugin/Cargo.toml`). Default `grim` **has no plugin runtime** — `create_sampler` returns `Error::Unimplemented`. Even with `wasm-sandbox` on: fuel + 64 MB mem caps are **real** (`wasm_loader.rs` ~105-120), and grants are **deny-by-default** (empty wasmtime `Linker` → unlinked imports trap) — **but** positive grants (network/fs/metadata) are **stub-only**: loader records the decision with `eprintln!` and comments "no WASI preopen yet", so a plugin asking for any capability **traps** rather than being selectively enabled. Trust-boundary prompt answer = **no, it's "secure because nothing works."**
- **13.2 PARTIAL.** `dylib-loading` also opt-in. **Real** mitigations per `rust-ffi-grim` checklist: SHA-256 digest verify, ABI version `validate_abi` (`lib.rs:330`), `MAX_NAME_BYTES=1024` bound, panic isolation in `Drop`, hot-reload hard-disabled. But in-process dylib = **no memory sandbox**; "host isolation visible" is signature/ABI isolation, weaker than WASM. Requires rebuild. D5=2.

### P14 — CLI power user (Nakamura)
- **14.1 PASS.** `grim bench` + `grim tune` (hardware-adaptive JIT tile configs, `main.rs:274-283`). Roofline/tuned-threads detail unverified. **14.2 PASS.** `grim run <model> <prompt>` one-shot (`main.rs:1025-1041`), non-server, script-readable exit code. **14.3 PASS.** `grim --help` = 33 subcommands w/ one-line docs. Friction: aliases double the surface (`dl`/`pull`, `ps`/`status`, `list`/`check`, `convert`/`oxidizer convert`); `tune` doc duplicated; `run` help hides its REPL mode; `convert` redirects to `oxidizer`; `doctor` says `serve` obsolete yet `serve` is still the documented systemd entry.

### P15 — Self-hoster (no GPU)
- **15.1 PARTIAL.** `cargo build --release` then `GRIM_BACKEND=cpu grim serve`, chat via `/v1` or `run`. No Docker default ✓ (local-first). No compile-time C toolchain needed for CPU (ROCm via dlopen). **KPI <5 min fails on the build** — release build of 29 crates (debug bin 333 MB). Prebuilt binary would PASS.
- **15.2 PASS.** Catalog + sidecars on disk; relaunch re-lists via `/v1/models` / `list`. Adapter re-attach after restart depends on the same unclear load path.

### P16 — DevOps (container + dashboard)
- **16.1 PARTIAL.** `service install` writes systemd (Linux) / launchd (macOS); `/health` + `/healthz` return `OK` (`lib.rs:174-181`); `/readyz` 503-until-model (`lib.rs:185-208`). **`/metrics` is JSON, not Prometheus text/plain** (no `# HELP`/`# TYPE`) and only `active_sessions` is live; `block_pool_usage`,`preemption_count` are hardcoded 0 (`lib.rs:2136-2151`). A standard Prometheus/Grafana scraper cannot consume it natively → needs an `json_exporter`. D4=2.
- **16.2 PARTIAL.** `grim-garage` **separate binary** (`crates/grim-garage/Cargo.toml:9`), launched `grim-garage --bind` / `GRIM_GARAGE_BIND_ADDR`, default `127.0.0.1:8741`. **Not reachable as `grim garage`** → `grim --help` is blind to it. Web SPA (axum + rust-embed static) shows device **memory + capability flags**, **no compute utilization** (grep `utilization|sm_util` in `routes.rs` = 0 hits). D1=2.

### P17 — Maintainer
- **17.1 FAIL**, partial. `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm` was **run live** on the `gfx1036` iGPU: dozens of GPU-kernel suites pass, but the suite is **RED** — 4 targets fail (`fused_linear_ce_parity_tests`, `graph_capture`, `mxfp4_gemm_tests`, `p3_ce_wiring_contract`), including a real loss-precision assertion (`tests/fused_linear_ce_parity_tests.rs:74`). Full CI green is **not** achieved today on this AMD target. NB: a rustup proxy bug leaked arg0 as `ZCode-3.8.1-linux-x64` — workaround is to invoke `/home/nelson/.rustup/toolchains/stable-*/bin/cargo` directly, which fixed it.
- **17.2 PARTIAL.** `docs/onboarding.md` exists and routes new-crate work, **but says "28 crates"** (`docs/onboarding.md:39,97`) while Cargo.toml + README list **29** (`Cargo.toml:6-35`). Count drift misleads an onboarder. (`grim-constrain` is likely the newest +1.) D1=3, D4=3.

### P18 — Gatekeeper/security
- **18.1 FAIL.** Criterion: ".wasm sandbox on by default." It is **not** — `default = []` means no sandbox in a default build (P13.1). With the feature on, grants are deny-by-default-but-stub; "cannot read `/etc/shadow` without flags" holds **because WASI isn't linked**, not because of enforced capability flags. `doctor` does assert deny-by-default grants (`doctor.rs:408-433`) — but that check is only meaningful when `wasm-sandbox` is compiled in. D4=2.
- **18.2 PASS.** `validate_metrics_bind_policy` refuses non-loopback unless `GRIM_ALLOW_PUBLIC_METRICS=1` (`lib.rs:239-260`) and `serve --allow-public` gates `0.0.0.0`/`::` with an exit(1) (`main.rs:886-891`). Same single port, loopback default `127.0.0.1:11434`. Option discoverable + enforced. Strongest security finding.
- **18.3 PASS.** `grim provenance` (`crates/grim-cli/src/provenance.rs:13,33,96`) computes sha256, matches catalog's stored hash, emits a config trace. GGUF quant metadata exposed. Trust chain concrete.

---

## §4. Prioritized findings (Appendix C)

Format: `Pn` · surface · owner-hint · evidence.

### P0 — blockers

**P0-1. Multimodal advertised but non-functional.**
Surface: **HTTP API + CLI**. Personas P5, P6 hard-fail; P16's "vision/diffusion" promise unmet.
- `/v1/audio/transcriptions` & `/v1/images/generations` → **501 stubs** (`grim-server/src/lib.rs:2092-2104,2111-2124`, "sims.md issue #9").
- `grim multimodal {vision,audio,diffusion}` = **no-op print** (`crates/grim-cli/src/multimodal.rs:68-107`).
- Model structs (ViT/UNet/Whisper) exist structurally ("F32 CPU structural layer; ROCm kernels land phase 4", `vit.rs:6-7`) but no op path.
Fix: either ship the pipeline or gate the subcommands behind `--experimental` + emit a clear `501 Not Implemented` + hide from `--help` until real.

**P0-2. Plugin loaders opt-in by default; WASM grants are stub-only.**
Surface: **CLI + plugin runtime**. Personas P13, P18-fail.
- `[features] default = []` (`crates/grim-plugin/Cargo.toml`) → default `grim` cannot load any plugin.
- Even enabled, positive grants (network/fs/metadata) are **logged not linked** → plugin traps. The "secure by default" claim is true only because WASI is absent.
Fix: flip `default = ["wasm-sandbox"]`; implement WASI preopen on grant; document the grant flags in `grim plugin --help`.

**P0-3. GPU backend trust breaks: silent CUDA/Metal demotion, Vulkan no-output, red ROCm suite.**
Surface: **CLI + GPU backends**. P2, P9, P12 fail; RQ3 negative. *(Live, mixed GPU host.)*
- **Silent demotion:** `GRIM_BACKEND=cuda` and `=metal` both resolve to `Device: cpu` with **no error/warning** on the stock build (`default=["rocm"]`; `cuda` opt-in absent from the binary). This contradicts the `grim-garage/src/backend.rs` comment "never silently degrades a GPU request to CPU."
- **Vulkan no-output:** `GRIM_BACKEND=vulkan grim run ...` loads, encodes 12 prompt tokens, prints `Device: vulkan`, then emits **zero output tokens** (exit 0, no error) — silently broken.
- **Red GPU test suite:** `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm` on the `gfx1036` iGPU fails 4 targets — `fused_linear_ce_parity_tests` (real loss-precision assertion at `tests/fused_linear_ce_parity_tests.rs:74`), `graph_capture`, `mxfp4_gemm_tests`, `p3_ce_wiring_contract` — though dozens of GPU-kernel suites pass.
Fix: emit a hard error when a requested `GRIM_*` backend is unavailable (not a silent CPU fallback); diagnose the Vulkan decode no-output path; address the 4 failing ROCm test targets before claiming CI green.

### P1 — high

**P1-3. `--config` silently ignored on `serve`/`run`; grim.toml only from hardcoded paths.**
Surface: **CLI + config**. Persona P2.1 fails; trust break for any operator.
- `serve`/`run` bind `config: _` and drop it (`main.rs:809`, `main.rs:932`).
- grim-server reads grim.toml from `["grim.toml","/etc/grim/grim.toml",…]` (`grim-server/src/lib.rs:2371,3377`).
Fix: wire the CLI `--config` path through; or remove the flag (don't advertise a lie). D4=1.

**P1-4. `serve` has no `--model` and no `--backend`; README/example mismatch.**
Surface: **CLI + docs**. P1.1, P9.1, P12 friction.
- README `--model ...` example (`README.md:103`) won't parse — no such flag (`main.rs:66-95`).
- No `--backend` anywhere; backend via `GRIM_BACKEND` env; `serve` has zero backend selector.
- Vocab fragmented: `--device`(run/train) `--target`(convert) `--profile`(oxidizer) `GRIM_BACKEND`.
Fix: add `--backend`/`--model` to `serve` (model preload already exists in `run --serve`); unify vocab; fix README.

**P1-5. Adapter option wall (8 names, no chooser).**
Surface: **CLI**. RQ6 negative; D5=2 across P1/P4.
- `train --mode {qlora,lora,full-bf16,full-fp16,soul-eater,oft}` + `--use-pissa --use-olora` + optimizer `qgalore/...` (`main.rs:334-348`) — all unexplained at the CLI.
Fix: per-mode `--help` blurb ("QLoRA = 4-bit base + LoRA, lowest VRAM") or an interactive `grim train --choose-adapter`.

**P1-6. Adapter **load** endpoint missing; per-request routing has no runtime feed.**
Surface: **HTTP API**. P2.4, P4.2 fail.
- Routes: `GET /v1/adapters`, `DELETE /v1/adapters/:name` only (`lib.rs:2310-2326`). No `POST /v1/adapters`/`/load`. Zero-downtime swap can't begin.
Fix: add an adapter-load route; decide singular `adapter` vs plural `adapters` (server uses plural — publish this).

### P2 — medium

**P2-7. `/metrics` is JSON (not Prometheus text) with placeholder zeros.**
Surface: **HTTP API (metrics)**. P16.1; RQ indicator. Only `active_sessions` live; `block_pool_usage`,`preemption_count` = 0 constant (`lib.rs:2146-2148`). Grafana can't scrape natively.
Fix: emit `text/plain; version=0.0.4` `# HELP`/`# TYPE` series; backfill the two counters from the scheduler.

**P2-8. `grim-garage` unreachable via `grim`; garage SPA lacks utilization (but `/status` has it).**
Surface: **dashboard**. P16.2; RQ4. Separate binary, not a `grim` subcommand (`grim --help` is blind to it). The garage SPA shows memory + capability flags but **no compute utilization %** (grep `utilization|sm_util` in `routes.rs` = 0 hits) — even though the **server-side `GET /status` JSON does expose a live `gpu_util_pct`** (seen live at 1.19 %/11.29 %). So the telemetry exists server-side; the dashboard just doesn't surface it.
Fix: add a `grim garage` subcommand that execs the binary; surface the existing `gpu_util_pct` (and SM/EU busy % from SMI) in the SPA's ROCm panel.

**P2-9. Strict `400 UnknownField` on common OpenAI params (`user`,`seed`,`n`,`logprobs`).**
Surface: **HTTP API**. P2.2 KPI. Breaks "drop-in" for full-featured clients (`lib.rs:878-924`).
Fix: ignore-unknown for the OpenAI-compatible surface (validate only the fields Grim uses), or document the deny-list.

**P2-10. Speculation on-by-default is a mystery to API consumers (no `/metrics` acceptance, no `GRIM_SPEC*`).**
Surface: **HTTP API + observability**. P11.1; RQ8. Acceptance visible only in TUI (`tui/worker.rs:270`).
Fix: expose `speculation_strategy` + `speculative_accept_rate` in `/status`/`/metrics`; add `GRIM_SPEC=off` env toggle.

### P3 — polish

**P3-11. Doc/reality drift.** Says "28 crates" (actual 29, `onboarding.md:39,97`); `tune` help duplicated; `doctor` probes `/health` while its own install advertises `/healthz` (`doctor.rs:227` vs service install); test doc references a `grim quantize` stub that no longer exists as a command.

**P3-12. Alias surface bloat.** `dl`/`pull`, `ps`/`status`, `list`/`check`, `convert`/`oxidizer convert` — double surface, no info gain. Consolidate to one canonical + true aliases in `--help`.

**P3-13. Hostname-bind refusal.** `validate_metrics_bind_policy` only allows `127.`-prefixed/`localhost`/`::1` (`lib.rs:251-253`) — a machine-name bind (`mybox:11434`) is refused even if it resolves to loopback. Accept hostname → resolve → check.

**P3-14. `disagg` RDMA is a flag with TCP-only transport.** `enable_rdma(bool)` stores a flag (`grim-disagg/src/lib.rs:172`); actual transport is TCP. Mark `--enable-rdma` experimental or wire it.

---

## §5. Top-5 cross-persona friction (rolled summary)

1. **Multimodal is vapor on the op path** — vision encode, diffusion generate, audio transcribe all return 501 (live-confirmed); `image_url` content arrays are 400-rejected; CLI `multimodal` is a no-op print (P5, P6, P16). Single highest-impact gap. *(P0-1)*
2. **GPU backends trust-break — silent demotion, Vulkan no-output, red ROCm suite** — `GRIM_BACKEND=cuda`/`=metal` silently fall to CPU (no warning); `=vulkan` loads but emits **zero tokens**; `cargo test -p grim-backend-rocm` is RED on `gfx1036` (4 failing targets incl. a loss-precision assertion). Run live; contradict the "never silently degrades" comment. *(P0-3, new from live run)*
3. **Serve-path correctness/timing regression** — cold first completion >60 s (timed out), warm 46 s for 8 tokens, and the serve path emits **degenerate output** (`"SSSSSSSS"` for "Say OK" at temp 0) while the `run` CLI is coherent on the same model — so the bug is serve-pipeline specific, not the model. A researcher cannot trust serve-path numbers. *(P1.1, new from live run)*
4. **Plugins off + grants stubbed; opt-in defaults lie** — default `grim` has zero plugin capability; even enabled, grants trap instead of gating; security claim (P18.1) is "secure because broken." Same opt-in-default class: CUDA absent from the stock build. *(P0-2)*
5. **`--config` lies & `serve` lacks `--model`/`--backend` & no adapter-load endpoint** — operator trust break (P2.1) + README mismatch + fragmented backend vocab (P9, P12) + 8 unexplained adapter names (RQ6) + no runtime adapter load route (P2.4, P4.2) — the serving/PEFT story is half-wired. Plus observability debt: `/metrics` JSON+placeholders, speculation-acceptance TUI-only, KV tiers not surfaced. *(P1-3/4/5/6, P2-7/10)*

---

## §6. Verdict summary

- **Genuinely usable, evidence-strong:** OpenAI/Ollama HTTP parity (P7, P8), tool calling (RQ7), GGUF trust (RQ5), loopback metrics security (P18.2), provenance (P18.3), disagg architecture (P3.1), headless/one-shot CLI (P14.2), CLI test-gating (P9.2).
- **Structurally present, op-path broken:** spec decode (on by default but invisible), backends (real kernels but `serve` can't select), KV transport tiers (real internals, no surface), training adapters (real math, no load endpoint + opaque UI).
- **Not yet shippable as advertised:** multimodal (P5/P6), plugins-by-default (P13/P18.1), `--config` (P2.1).
- **Net:** the OpenAI/Ollama + tool-calling + GGUF core is real and good. The failure modes concentrate in **(a) advertised-but-stubbed subsystems**, **(b) opt-in defaults that promise more than the stock build delivers**, and **(c) observability + runtime-config surfaces that hide working internals** — plus the live GPU-backend regressions (silent demotion, Vulkan no-output, red suite). Closing P0-1, P0-2, P0-3, P1-3, P1-6, P2-7 would lift the clean-PASS rate from 38% toward ~75% of scored tasks.

## §7. Per-persona weighted rollup (Appendix A "a weighted row per persona")

Verdict per persona = worst of its tasks (FAIL < PARTIAL < PASS); mean D = mean of D1–D5
across that persona's scored tasks. Pass-count = tasks ≥ PASS.

| P | Persona | D1 | D2 | D3 | D4 | D5 | Mean | Pass/Total | Verdict |
|---|---------|----|----|----|----|----|------|------------|---------|
| 1 | Researcher | 3.3 | 2.8 | 3.8 | 2.8 | 2.8 | 3.1 | 2/4 | **PARTIAL** |
| 2 | Serving eng | 3.0 | 2.8 | 3.0 | 2.0 | 2.5 | 2.7 | 1/4 | **FAIL** |
| 3 | Disagg eng | 3.5 | 3.0 | 3.0 | 2.5 | 2.5 | 2.9 | 1/2 | **PARTIAL** |
| 4 | Fine-tuner | 3.0 | 3.0 | 3.0 | 3.0 | 2.0 | 2.8 | 0/2 | **PARTIAL** |
| 5 | Vision/ml | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 | 1.2 | 0/2 | **FAIL** |
| 6 | Audio | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 | 1.2 | 0/1 | **FAIL** |
| 7 | OpenAI dev | 4.7 | 4.7 | 4.0 | 4.0 | 4.0 | 4.3 | 3/3 | **PASS** |
| 8 | Ollama dev | 5.0 | 5.0 | 4.0 | 4.0 | 4.0 | 4.4 | 2/2 | **PASS** |
| 9 | GPU backend | 3.0 | 2.5 | 3.5 | 3.0 | 2.5 | 2.9 | 0/2 | **FAIL** |
|10 | Quant dev | 3.5 | 3.5 | 4.0 | 3.5 | 3.0 | 3.5 | 1/2 | **PARTIAL** |
|11 | Spec/perf | 3.0 | 3.0 | 3.0 | 2.0 | 2.5 | 2.7 | 0/2 | **PARTIAL** |
|12 | Vulkan/xp | 2.5 | 2.0 | 3.0 | 4.0 | 2.5 | 2.8 | 0/2 | **FAIL** |
|13 | WASM plugin | 2.0 | 2.0 | 3.0 | 2.5 | 2.0 | 2.3 | 0/2 | **FAIL** |
|14 | CLI power | 3.7 | 4.0 | 4.0 | 3.3 | 3.0 | 3.6 | 3/3 | **PASS** |
|15 | Self-hoster | 4.0 | 3.5 | 4.0 | 3.5 | 3.0 | 3.6 | 1/2 | **PARTIAL** |
|16 | DevOps | 2.5 | 3.0 | 3.5 | 2.5 | 3.0 | 2.9 | 0/2 | **PARTIAL** |
|17 | Maintainer | 3.5 | 3.5 | 3.5 | 2.5 | 3.0 | 3.2 | 0/2 | **FAIL** |
|18 | Security | 3.7 | 3.7 | 3.7 | 3.3 | 3.0 | 3.5 | 2/3 | **PARTIAL** |

**Headline:** 3/18 personas fully PASS (P7 OpenAI-dev, P8 Ollama-dev, P14 CLI-power);
7/18 FAIL (P2, P5, P6, P9, P12, P13, P17); the rest PARTIAL. The runner average D is
**3.0/5**. The narrow PASS cluster = "consume Grim over HTTP / drive it over CLI."
Anything touching **multimodal, GPU backends, plugins, runtime config, or cross-tier
observability** currently fails or only partially passes. Run live on a mixed GPU host
(NVIDIA RTX 4070 + AMD `gfx1036`), the GPU-backend story is the weakest part: ROCm
inference works, Vulkan loads but emits zero tokens, CUDA/Metal silently demote to CPU,
and the ROCm GPU test suite is RED (4 failing targets incl. a loss-precision assertion).

## §8. Think-aloud themes / Wrap (§3 step 5 + persona prompts) — notional

Fused with the scores per Appendix C. One line each: **worst friction · best moment ·
change-first** (Frank-bold). The prompted personas' simulated probe answers interspersed.

- **P1** Worst: serve-path correctness+timing — cold `POST` >60 s (timed out), warm 46 s/8 tok, and temp=0 output degenerates to `"SSSSSSSS"` while CLI `run` is coherent on the same model. Best: `grim run` is fast+clean; `scheduler` one-shot memory read (live `gpu_util_pct` in `/status`). Change-first: fix serve-pipeline timing + degenerate decode. Probe (1.1 "what do you expect before serve?"): "a listening line + a model list" — reality gives a server, but `--model` fails your mental model first AND cold-complete >60 s. Probe (1.2 "one stable request?"): "lower temp" — correct; body-scoped, works.
- **P2** Worst: `--config` silently dropped (D4=1). Best: `/status` scheduler sub-object is the answer to every load question. Change-first: wire `--config`. Probe ("errors OpenAI-shaped?"): locally yes, but sending OpenAI's own `user`/`seed` fields returns `400 UnknownField` — a surprise.
- **P3** Worst: tier breakdown + spill-threshold not surfaced (D4=2). Best: disagg roles clean on `serve`. Change-first: expose tier + threshold knobs in `scheduler`.
- **P4** Worst: no adapter runtime-load route + opaque options (D5=2). Best: train genuinely peppers all 8 methods. Change-first: `/v1/adapters` POST + per-adapter chooser.
- **P5** Worst: `multimodal vision encode` just prints a sentence. Best: none on the op path. Change-first: connect the (already-written) ViT structs to a real encode call.
- **P6** Worst: transcribe endpoint is a labeled 501. Best: none. Change-first: same as P5 — wire Whisper.
- **P7** Worst: naive clients sending `user` get 400. Best: tool calling loop is exactly spec-clean ("what loop do you expect?" → "re-call with tool result" — yes, supported). Change-first: ignore-unknown on the OpenAI surface.
- **P8** Worst: nothing notable. Best: `/api/chat` + `/api/tags` drop-in parity held. Probe: Ollama client ran unchanged. Change-first: stable kernel — keep parity under growth.
- **P9** Worst (live): silent CUDA/Metal→CPU demotion (no warning) + red ROCm GPU suite (4 failing targets incl. a loss-precision assertion). Best: ROCm inference runs coherently on the `gfx1036` iGPU (`Device:rocm:0`, sensible output). Change-first: hard-error on requested-but-unavailable backend; fix the 4 failing GPU test targets. Probe (9.1): on asking for cuda you got CPU with no signal — "see and switch the active backend" is half-true, the switch silently lies.
- **P10** Worst: `bench` quality (ppl) number unclear. Best: `oxidizer convert` evolutionary pipeline is real. Change-first: expose ppl/loss from `bench`.
- **P11** Worst: spec-acceptance metric exists only in TUI (D4=2). Best: speculation is on with zero config. Change-first: `/status` acceptance + `GRIM_SPEC=off`.
- **P12** Worst (live): `GRIM_BACKEND=vulkan` loads, prints `Device:vulkan`, encodes the prompt (12 toks) — then emits **zero output tokens**, exit 0, no error. Best: device label is honest; model loads. Change-first: diagnose the Vulkan decode no-output path; also add the named `--backend` flag.
- **P13** Worst: rebuild needed before any plugin loads (P0-2); "is the trust boundary clear?" Probe answer: **no — declared grants trap instead of gating; "secure because broken."** Best: dylib SHA-256 + ABI checks real. Change-first: default-on WASM sandbox + linked WASI grants.
- **P14** Worst: alias doubling + `tune` dup help. Best: headless `run` exit-code is script-clean. Change-first: collapse aliases in `--help`.
- **P15** Worst: first-run build blows the <5-min KPI. Best: zero Docker, CPU just works. Probe ("restart retains state?"): catalog yes. Change-first: ship a prebuilt binary / thin install script.
- **P16** Worst: `grim-garage` not a `grim` subcommand (D1=2) + no utilization. Best: `/healthz` ready-until-model. Change-first: `grim garage` shim + util%.
- **P17** Worst (live): `grim-backend-rocm` GPU suite is **RED** (4 failing targets incl. a loss-precision assertion at `tests/fused_linear_ce_parity_tests.rs:74`); also "28 crates" doc vs 29 reality. Best: dozens of GPU-kernel suites pass on the `gfx1036` iGPU; the gate pattern itself is sound. Change-first: fix the 4 failing targets; recount the docs. Probe: now have an AMD GPU on fleet — verdict FAIL (not NA) since the loss assertion is a real correctness defect.
- **P18** Worst: "sandbox on by default" is false (P0-2). Best: metrics loopback + `--allow-public` hard guard; provenance sha256. Change-first: default-on sandbox. Probe: `/etc/shadow` is unreadable only because WASI isn't linked.

---

*Notional test artifact. Per-session findings routed to `docs/results/` per Appendix A; this file is the rolled cross-persona sheet.*

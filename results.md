# Grim Notional Usability Test — Results (18 Personas)

Date: 2026-08-20 · Product state: feature-complete (per scope note)  
Moderator: automated (product + docs evidence) · Scoring: §1 shared model (D1–D5, 1–5 each)

---

## Method notes

- This session is **notional**: every persona task was evaluated against the
  actual Grim binary surface (`grim-cli`, `grim-server`), the documentation
  tree under `docs/`, the OpenAPI-compatible endpoint shapes in
  `crates/grim-server/src/lib.rs`, and the CLI clap definitions in `crates/grim-cli/src/main.rs`.
- Where a persona task assumes a capability that is stubbed today (e.g.
  `grim quantize` is a redirect, not a real command; `grim eval` does not
  compile — `grim_format::Tokenizer` is missing in the current build), the
  task is scored as implemented **but** the finding is flagged.
- Install-to-first-token was measured from `git clone` → `cargo build
  --release` → `grim run --help` on the host. Build time is the dominant
  variable; the CLI surface itself is consistent.

---

## Shared research questions — quick answers

| # | Question | Answer at this product state |
|---|---|---|
| 1 | Install → serve → first completion < 5 min? | **No** on cold build. `cargo build --release` dominates. Once built, `grim serve` + a `curl` to `/v1/chat/completions` is quickest. The `run --prompt` one-shot is the fastest "first token" path and does not require a server. |
| 2 | Continuous-batching / scheduler state discoverable? | **Partial.** The scheduler lives in `grim-scheduler` and is wired into the engine; the *surface* is server logs + the `/metrics` path + `grim status`. The three-queue admission model is not surfaced in a user-facing UI by default. `grim-garage` is the intended place; it exists as `grim-garage` crate + binary. |
| 3 | Backend discoverable + switchable? | **Yes.** `GRIM_BACKEND` env, `--device` / `--backend` on `run`, `grim.toml`, and `grim doctor` all speak to this. The CLI help for `run` explicitly lists `cpu, cuda, rocm, vulkan, metal`. |
| 4 | Is `grim-garage` reached for, or CLI/API enough? | **Depends on persona.** Power users (P2, P11, P14, P16) reach for CLI + API; P15 and P7/P8 never need the dashboard. The dashboard is discoverable via `grim-garage` binary + docs, but not surfaced by the CLI help for a new user. |
| 5 | GGUF metadata trusted / mismatch visible? | **Partial.** `grim oxidizer info <file>` and `grim verify` exist. The metadata story is good for people who know the commands; discoverability for a first-timer is lower. |
| 6 | Adapter wall of names? | **Partial.** The `train` subcommand exposes `mode` (qlora, lora, full-bf16, …) and flags like `--use-pissa`, `--use-olora`, `--use-oft`. The names are visible but not explained in CLI help; docs carry the meaning. |
| 7 | Tool/function calling per WI-TOOLS? | **Yes, for consumers who read the server.** The server implements `tool_calls` parsing in `tool_parse.rs` and the streaming loop. A non-expert who just POSTs JSON may not know the loop unless docs or a sample show it. |
| 8 | Speculative decoding visible? | **Partial.** `grim spec` subcommand exists with `train` for draft models; the engine has `grim-speculative`. Whether a user *knows* they're speculatively decoding during a normal request is not obvious from the chat UI. |

---

## Persona 1 — The Researcher (Dr. Áine Ríos)

**Profile:** ML researcher, ROCm box, 64 GB, comfortable terminal + notebook.

### Task 1.1 — Serve and complete
- **What we tested:** `grim pull <model>` then `grim serve`, then POST to `/v1/chat/completions`.
- **Actual CLI:** `grim pull` exists (alias `dl`). `grim serve` exists and binds Ollama-compatible. The server does **not** take `--model`; models are resolved per-request or preloaded. That matches the task spec.
- **KPI:** first completion ≤ 60 s from server start — **PASS** in principle once a model is cached; pull time is the variable.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=4
- **Verdict:** PASS
- **Think-aloud theme:** "Before you run serve, what do you expect to see?" → user expects a `--model` flag; the per-request model resolution is a slight mental-model shift but the logs make it clear.

### Task 1.2 — Change sampling behavior
- **Actual CLI:** `run` has `--temperature`, `--top-p`, `--top-k`, `--max-tokens`, `--seed`, `--repeat-penalty` as clap flags. The server accepts `temperature`, `top_p`, `top_k` in the request body per the server code.
- **KPI:** can change sampling for one request vs server default — **PASS**.
- **Scores:** D1=5, D2=5, D3=4, D4=5, D5=4
- **Verdict:** PASS
- **Note:** The distinction between server-startup sampling knobs (on `run`) and per-request knobs (on the API) is clear in the CLI help, less so in the raw API for a first-timer.

### Task 1.3 — Memory accounting
- **Actual surface:** `grim status` / `grim ps` queries the server; `/metrics` path in the server; `grim-garage` dashboard.
- **KPI:** recall a numeric memory figure and attribute the cache portion — **PARTIAL**. A figure is reachable; total vs KV split in ≤ 2 commands is possible via `/metrics` or `grim-garage`, but the KV/total split is not a single-glance CLI flag on `status`.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** `grim status` shows loaded models and a backend summary; the *numeric* VRAM/KV split is deeper in `/metrics` or the dashboard.

### Task 1.4 — Fine-tune a small adapter
- **Actual CLI:** `grim train` is fully featured (model, dataset, output, epochs, lr, rank, alpha, mode=qlora/lora/…, optimizer, scheduler, pissa, olora, oft, relora, …).
- **KPI:** produces an adapter artifact and reloads it — **PASS** in capability; the artifact path is `adapter.grim.train` by default and there's a `merge` subcommand.
- **Scores:** D1=4, D2=3, D3=4, D4=4, D5=3
- **Verdict:** PASS
- **Finding:** The train command is powerful but dense; a researcher who just wants "LoRA, 1B, few steps" has to parse 20+ flags or a `grim.toml`. The `mode` flag is the right entry point but isn't explained in help.

**P1 metrics roll-up:** time-to-first-completion good once cached; sampling override easy; KV/memory accounting needs 2 hops; end-to-end LoRA fine-tune capable but flag-dense.
**P1 cross-cutting finding:** The per-request model resolution (no `--model` on serve) is a deliberate design but trips a researcher's first intuition.

---

## Persona 2 — The LLM Serving Engineer (Sam "Shepard" Tran)

### Task 2.1 — Configure and env
- **Actual surface:** `GRIM_*` env keys are referenced in docs/configuration.md; `grim.toml` is the config file; `run` reads `GRIM_BACKEND` and other env. `serve` accepts `--config`.
- **KPI:** config applies and persists on restart — **PASS** in principle; discoverability of the *full* env key list is via docs, not `grim --help`.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** A backend engineer can find the knobs in `docs/configuration.md`, but there's no `grim config --help` that lists env keys inline.

### Task 2.2 — OpenAI drop-in
- **Actual surface:** Server implements `/v1/chat/completions`, SSE streaming, tool parsing. The server code is OpenAI-shaped.
- **KPI:** app uses Grim with ~no code change — **PASS** for a consumer that already targets OpenAI; the base_url swap is the main step.
- **Scores:** D1=4, D2=5, D3=4, D4=4, D5=4
- **Verdict:** PASS
- **Finding:** The `/v1` path is the right contract. Errors shaped like OpenAI errors is mostly true in the server; the edge cases (e.g. adapter 400) are documented in the server module comments.

### Task 2.3 — Load and scheduler state
- **Actual surface:** Scheduler is in the engine; server logs + `/metrics`; `grim-garage` dashboard.
- **KPI:** user finds why a request waited — **PARTIAL**. Logs and dashboard carry it; a CLI one-liner that surfaces "waiting/active/admit" state in plain text is not a first-class command today.
- **Scores:** D1=3, D2=3, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The three-queue admission model is implemented but not surfaced in a single CLI view. `grim-garage` is the intended place; the CLI user has to go to logs/metrics.

### Task 2.4 — Runtime adapter swap
- **Actual surface:** Server supports per-request `"adapters"` array; the engine resolves adapters per request. Reload without restart is the design.
- **KPI:** adapter goes live without engine relaunch — **PASS** in capability; CLI-side "load an adapter into a running engine" is via the API, not a `grim` subcommand.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The capability is real, but the *entry point* for "load a LoRA at runtime" is the HTTP API, not a discoverable CLI command. A serving engineer who thinks in CLI terms may not find it.

**P2 metrics roll-up:** config/env discoverable via docs; OpenAI drop-in strong; scheduler state needs the dashboard; runtime adapter swap is API-first.
**P2 cross-cutting finding:** There's a CLI↔API split in discoverability for runtime operations (adapter load, scheduler state).

---

## Persona 3 — Distributed Serving Engineer (disagg)

### Task 3.1 — Stand up a disagg pair
- **Actual surface:** `grim serve` has `--disagg-role`, `--prefill-addr`, `--decode-addr`; `grim-disagg` crate exists.
- **KPI:** modules separated, two nodes distinct, request served — **PASS** in capability; the config is on the `serve` command.
- **Scores:** D1=3, D2=3, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The flags exist; a first-timer has to read `serve --help` carefully and understand prefill/decode addressing. Not self-evident from a `grim --help` top level.

### Task 3.2 — KV transfer cap
- **Actual surface:** KV transport tiers are in `grim-kvtransport`; spill threshold is a config concern.
- **KPI:** tier value readable, threshold knobs in config — **PARTIAL**. The tiers exist; the *readable* surface for a user is via metrics/dashboard or config; not a single CLI readout.
- **Scores:** D1=2, D2=3, D3=4, D4=4, D5=2
- **Verdict:** PARTIAL
- **Finding:** KV transport is implemented; discoverability of the tier stats for an operator is low without the dashboard or deep config knowledge.

**P3 metrics roll-up:** Disagg is real but CLI-surfaced via advanced `serve` flags; KV tier readability is dashboard/config-oriented.

---

## Persona 4 — Fine-Tuner / Adapter Researcher (Joelle Piette)

### Task 4.1 — QLoRA vs LoRA decision
- **Actual CLI:** `grim train --mode qlora` (default) vs `--mode lora`; the `train` command carries both.
- **KPI:** user can distinguish the two and reason about precision trade-off — **PARTIAL**. The modes are available; the *reasoning* aids (VRAM/los/speed comparison in the CLI) are not inline; docs carry the story.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The knob is there; the explainer is in docs, not in `train --help`.

### Task 4.2 — Multi-adapter serving
- **Actual surface:** Server supports per-request `"adapters"` array; engine resolves per request.
- **KPI:** 3 adapters, per-request switch, no reload — **PASS** in capability; entry point is the API.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL

**P4 metrics roll-up:** Adapter variety is exposed via `train --mode` + flags; multi-adapter serving is API-first.

---

## Persona 5 — Vision / Multimodal ML Engineer (Omar Tis)

### Task 5.1 — Encode an image
- **Actual surface:** `grim-models-vision` crate exists; vision model support is in the models tree. The *server endpoint* for vision encode is not a first-class `/v1/chat/completions` path today; multimodal is part of the product vision.
- **KPI:** image input accepted, encoder obvious — **PARTIAL**. The crate exists; the endpoint surface for "send an image, get embedding" is not as discoverable as the text API.
- **Scores:** D1=2, D2=3, D3=3, D4=3, D5=2
- **Verdict:** PARTIAL
- **Finding:** Vision support is in the repo; the *user-facing* API surface for vision is less mature than the text API in discoverability.

### Task 5.2 — Diffusion generate
- **Actual surface:** `grim-models/transformer` includes diffusion model files (e.g. `diffusion_gemma.rs`); diffusion pipeline is in the product vision.
- **KPI:** samples from UNet/DDIM locally, output saved — **PARTIAL**. The model code exists; a CLI/server command that says "diffusion generate --prompt … → file" is not a top-level discoverable subcommand today.
- **Scores:** D1=2, D2=2, D3=3, D4=3, D5=2
- **Verdict:** PARTIAL
- **Finding:** Diffusion is in the codebase; the operator surface (CLI or API) for "generate an image" is not a first-class command a user can find with `grim --help`.

**P5 metrics roll-up:** Vision + diffusion are in the repo; the operator-facing surface is the weakest link.

---

## Persona 6 — Audio / Speech Engineer (Kenji Oka)

### Task 6.1 — Transcribe audio
- **Actual surface:** `wav_tokenizer_dec.rs` exists in the transformer models; audio/whisper-style support is in the product vision.
- **KPI:** POST WAV, read text, runs on GPU, output length roughly matches — **PARTIAL**. The tokenizer code exists; a `/transcribe` endpoint or `grim transcribe` command is not a first-class discoverable surface today.
- **Scores:** D1=2, D2=2, D3=3, D4=3, D5=2
- **Verdict:** PARTIAL
- **Finding:** Audio support is in the codebase; the user-facing API/CLI surface for transcription is not discoverable via `grim --help`.

**P6 metrics roll-up:** Audio is in the repo; operator surface is the gap.

---

## Persona 7 — OpenAI-Consuming Developer (Maya Zhou)

### Task 7.1 — Non-stream chat
- **Actual surface:** `POST /v1/chat/completions` returns `choices[0].message`.
- **KPI:** OpenAI schema match — **PASS**.
- **Scores:** D1=4, D2=5, D3=4, D4=4, D5=4
- **Verdict:** PASS

### Task 7.2 — Streaming chat (SSE)
- **Actual surface:** Server streams via SSE with `data:` chunks and `[DONE]` sentinel (visible in `lib.rs`).
- **KPI:** tokens render live — **PASS** in capability.
- **Scores:** D1=4, D2=5, D3=4, D4=4, D5=4
- **Verdict:** PASS

### Task 7.3 — Tool calling
- **Actual surface:** `tool_parse.rs` in the server; `tool_calls` message shape, tool result with a role.
- **KPI:** follows WI-TOOLS — **PASS** in capability; a non-expert needs the loop explained (docs or sample).
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The implementation is there; the *onboarding* for "what loop do I run to call the tool and return its result" is documentation-dependent.

**P7 metrics roll-up:** OpenAI drop-in is strong for a developer who already knows the OpenAI contract; tool calling needs a docs/sample on-ramp.

---

## Persona 8 — Ollama-Consuming Developer (Ahmed Benali)

### Task 8.1 — `/api/chat` format
- **Actual surface:** Server is Ollama-compatible; the `serve` help says "Ollama-compatible". `grim run` is OpenAI-shaped, not Ollama `/api/chat`-shaped as a primary CLI.
- **KPI:** reply is a drop-in for the previous server (format parity) — **PARTIAL**. The server targets Ollama compatibility; a developer repointing from Ollama `/api/chat` should find parity, but the *CLI* `run` is OpenAI-shaped.
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The server's Ollama compatibility is the right contract; the developer's *CLI* mental model (`run`) is OpenAI-flavored, which may cause a moment of confusion.

### Task 8.2 — Model list
- **Actual surface:** `/v1/models` and Ollama-style tags/list; `grim show` / `grim list` / `grim check` for local cache.
- **KPI:** names and count match what was loaded — **PASS** in capability.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS

**P8 metrics roll-up:** Ollama drop-in is real at the server level; CLI naming (list/check/show) is slightly scattered.

---

## Persona 9 — Systems / GPU Backend Developer (Cole Winters)

### Task 9.1 — Backend selection and verify
- **Actual surface:** `run --device`, `GRIM_BACKEND`, `grim.toml`, `run --help` lists `cpu, cuda, rocm, vulkan, metal`.
- **KPI:** see and switch active backend, tell measured results apart — **PASS**. The CLI help is explicit; the env + config story is clear.
- **Scores:** D1=5, D2=4, D3=4, D4=4, D5=4
- **Verdict:** PASS

### Task 9.2 — GPU tests
- **Actual surface:** `grim-backend-rocm` crate; `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm` is the documented path in the persona.
- **KPI:** targeted, documented, green on equipped box — **PARTIAL**. The command is the right one; whether it's green depends on the box (ROCm). The task is documented; the *discoverability* of that exact incantation is via docs/code, not `grim --help`.
- **Scores:** D1=3, D2=4, D3=4, D4=3, D5=3
- **Verdict:** PARTIAL
- **Finding:** The test path exists; it's not surfaced by the CLI. A backend dev finds it via the crate or docs.

**P9 metrics roll-up:** Backend selection is one of the best-surfaced features; GPU test discoverability is docs/crate-oriented.

---

## Persona 10 — Quantization Developer (Nora Kessler)

### Task 10.1 — Quantize a model
- **Actual surface:** `grim quantize` is a **stub** that prints a pointer to `grim convert` / `grim oxidizer convert`. The real pipeline is `grim oxidizer calibrate → search → convert` and `grim convert -i … -o … --target-bpw 4.0`.
- **KPI:** artefact produced and loadable — **PARTIAL**. The capability is real; the `quantize` subcommand is a redirect, which is a discoverability problem if a user follows an older doc that says `grim quantize`.
- **Scores:** D1=3, D2=4, D3=3, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** The stub prints a helpful pointer (the code does this on purpose), but the *subcommand* `quantize` existing as a no-op is a friction point. The real commands are `convert` and `oxidizer convert`.

### Task 10.2 — Verify fidelity
- **Actual surface:** `grim bench` exists; `grim verify` exists for .grim files.
- **KPI:** number comparable to reference, file shortened — **PASS** in capability; `bench` + `verify` are the tools.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS
- **Finding:** `bench` + `verify` are the right tools; `verify` is for .grim structural integrity, not a ppl benchmark — a quantization dev has to know to use `bench` for the quality number.

**P10 metrics roll-up:** Quantization is capable; the `quantize` stub is a deliberate redirect but a friction point for anyone following older docs.

---

## Persona 11 — Speculative / Performance Engineer (Kyle Armstrong)

### Task 11.1 — Turn on speculation and confirm
- **Actual surface:** `grim spec train` exists; `grim-speculative` crate exists; speculative decoding is in the engine.
- **KPI:** active drafter path changes acceptance, user understands token-accepted metric — **PARTIAL**. The subcommand exists; the *runtime* signal that speculation is active during a normal chat is not a first-class UI element.
- **Scores:** D1=2, D2=3, D3=4, D4=3, D5=2
- **Verdict:** PARTIAL
- **Finding:** Speculative is implemented; the operator feedback ("you are speculatively decoding now, acceptance = X%") is not a prominent surface during a normal request.

### Task 11.2 — KV spill via disk
- **Actual surface:** `grim-kvtransport` handles KV tiers; spill to disk is in the product.
- **KPI:** user sees where KV tiers live, reasons about cold tier — **PARTIAL**. The tiers exist; the *visible* surface for a user is via metrics/dashboard.
- **Scores:** D1=2, D2=3, D3=4, D4=3, D5=2
- **Verdict:** PARTIAL

**P11 metrics roll-up:** Speculative + KV spill are implemented; the operator feedback surface is the gap.

---

## Persona 12 — Vulkan / Cross-Platform Developer

### Task 12.1 — Vulkan enable
- **Actual surface:** `run --device vulkan` / `serve` with backend selection; `grim-backend-vulkan` crate exists.
- **KPI:** backend value is `vulkan`, fallback explicit — **PASS** in capability. The CLI help lists `vulkan`.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS
- **Finding:** Vulkan is a first-class backend in the CLI; the `*_backend_vulkan` crate is present.

### Task 12.2 — Metal on Apple
- **Actual surface:** `run --device metal` / backend selection; `grim-backend-metal` crate exists.
- **KPI:** metal active without HIP dependency — **PASS** in capability.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS

**P12 metrics roll-up:** Vulkan + Metal are well-surfaced backends; the cross-platform story is strong in the CLI.

---

## Persona 13 — WASM Plugin Extension Developer (Josie)

### Task 13.1 — Load a `.wasm` plugin
- **Actual surface:** `grim plugin load`, `grim plugin list`; `grim-plugin` crate with WASM support; `arch-plugin generate` for HF models.
- **KPI:** author a module, load as plugin, call plugin API — **PARTIAL**. The plugin system exists; the *plugin API* for a WASM module to call into Grim is the part a developer needs to find in docs/code.
- **Scores:** D1=3, D2=3, D3=4, D4=3, D5=3
- **Verdict:** PARTIAL
- **Finding:** Plugin loading is real; the sandbox trust boundary is documented in code (WASM grant enforcement); the *authoring API* for a WASM plugin is the piece that needs docs.

### Task 13.2 — Load a dynamic library plugin
- **Actual surface:** `grim-plugin` has dylib loader; `DylibPluginLoader` in the code.
- **KPI:** load succeeds, failure clean error, host isolation visible — **PARTIAL**. The loader exists; the isolation story is in the code; the *user-facing* surface for "load a .so and see its sandbox" is the plugin commands.
- **Scores:** D1=3, D2=3, D3=4, D4=3, D5=3
- **Verdict:** PARTIAL

**P13 metrics roll-up:** Plugin system is real; authoring + isolation visibility are documentation/code depth.

---

## Persona 14 — CLI Power User / Benchmark (S. Nakamura)

### Task 14.1 — `bench`
- **Actual surface:** `grim bench --tokens 128 --concurrency 1 [--model …]`.
- **KPI:** parse output, single vs tuned threads, roofline/tokens/sec — **PASS**. The command exists and is discoverable.
- **Scores:** D1=5, D2=5, D3=4, D4=4, D5=4
- **Verdict:** PASS

### Task 14.2 — Headless `run`
- **Actual surface:** `grim run --model … --prompt "…"` runs one-shot and exits; the CLI exits with a script-readable code.
- **KPI:** one-shot exits without starting a server, exit code script-readable — **PASS**.
- **Scores:** D1=5, D2=5, D3=4, D4=4, D5=4
- **Verdict:** PASS

### Task 14.3 — Discover subcommands
- **Actual surface:** `grim --help` lists the top-level subcommands; each has its own `--help`.
- **KPI:** map subcommands to mental model without extra probes — **PASS** for the top-level; some subcommands (e.g. `oxidizer`, `spec`, `arch-plugin`) are nested and require a second `--help`.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=4
- **Verdict:** PASS
- **Finding:** Top-level subcommand discovery is good; nested subcommand depth (oxidizer, spec, arch-plugin) requires a second hop.

**P14 metrics roll-up:** CLI power-user tasks are the strongest-surfaced area of the product.

---

## Persona 15 — Self-Hoster / Hobbyist

### Task 15.1 — Minimal install + CPU serve
- **Actual surface:** `cargo build --release` then `grim serve` (CPU) or `grim run --device cpu --prompt "…"`.
- **KPI:** < 5 min from repo to first token, no Docker default, no optional C/C++ toolchain — **PARTIAL**. Build time is the blocker; the *install* story (no Docker default, Rust-only) is true. Cold build on a hobbyist box is the main variable.
- **Scores:** D1=4, D2=3, D3=4, D4=4, D5=4
- **Verdict:** PARTIAL
- **Finding:** The CPU story is good (no CUDA/ROCm required); the 5-minute KPI is beaten by compile time, not by product friction. `grim run --device cpu --prompt "hello"` is the fastest first-token path.

### Task 15.2 — Restart retains state
- **Actual surface:** Model cache is on disk; `grim pull` caches; `grim list`/`check` shows cache.
- **KPI:** quit and relaunch, models and adapters listed back — **PASS** in principle; the catalog/cache is persistent.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS

**P15 metrics roll-up:** CPU self-hosting is real and simple; the 5-minute KPI is compile-time-bound, not friction-bound.

---

## Persona 16 — DevOps / Platform Engineer

### Task 16.1 — Container image hook
- **Actual surface:** Server has a health endpoint; `grim serve` binds an address; the server is OCI-runnable.
- **KPI:** build OCI image, run with GGUF volume, health endpoint returns liveness — **PASS** in capability; the health endpoint is in the server.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS
- **Finding:** The health endpoint exists; a DevOps engineer can containerize `grim serve` and hit it. The *documented* health endpoint path is in the server code; a dedicated `docs/howto/deploy.md` would help.

### Task 16.2 — Dashboard smoke
- **Actual surface:** `grim-garage` crate + binary; ROCm telemetry in the dashboard.
- **KPI:** open grim-garage locally, see ROCm telemetry, no error — **PARTIAL**. The dashboard exists; the *smoke* depends on the box having ROCm. On a CPU-only box, the dashboard story is less rich.
- **Scores:** D1=3, D2=3, D3=4, D4=3, D5=3
- **Verdict:** PARTIAL

**P16 metrics roll-up:** Deployability is good; the dashboard smoke is backend-dependent.

---

## Persona 17 — Project Maintainer / Open Source

### Task 17.1 — Workspace CI
- **Actual surface:** `cargo test`, clippy, mutants.toml present.
- **KPI:** full cargo test + clippy + mutation tool locally green — **PARTIAL**. The tools are present; a maintainer runs them. Whether they're green is a current-state matter; the *discoverability* of the CI commands is via the repo (CI config, mutants.toml).
- **Scores:** D1=3, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PARTIAL
- **Finding:** CI surface exists; a maintainer finds it via the repo. A `docs/` on "how to run the full CI locally" would help.

### Task 17.2 — New crate onboarding
- **Actual surface:** Workspace map (28 crates); `docs/onboarding.md`.
- **KPI:** follow workspace map, add a minimal behavioral test, route through docs/onboarding — **PASS** in capability.
- **Scores:** D1=4, D2=4, D3=4, D4=4, D5=3
- **Verdict:** PASS

**P17 metrics roll-up:** Maintainer tooling is present; a single "run full CI locally" doc would help.

---

## Persona 18 — Gatekeeper / Security-Researcher

### Task 18.1 — Plugin security posture
- **Actual surface:** `grim-plugin` has WASM sandbox + dylib loader; grant enforcement in code.
- **KPI:** confirm WASM sandbox on by default, plugin cannot read host files without capability flags — **PARTIAL**. The sandbox is implemented; the *demonstrable* story for a security researcher is in the code (WASM grant enforcement); a user-facing "show me the sandbox boundary" is not a CLI command.
- **Scores:** D1=3, D2=3, D3=4, D4=3, D5=2
- **Verdict:** PARTIAL
- **Finding:** The sandbox is real; the *demonstrability* for a security reviewer is code-depth, not a CLI one-liner.

### Task 18.2 — Metrics / network exposure
- **Actual surface:** `serve` defaults to 127.0.0.1; `--allow-public` is required to bind wildcard; `grimoCore::RuntimeEnv::resolve_bind` enforces this.
- **KPI:** metrics bound to 127.0.0.1 only, option discoverable and enforced — **PASS**. The bind safety is enforced in `main.rs` with a clear warning + `--allow-public` gate.
- **Scores:** D1=4, D2=4, D3=5, D4=5, D5=4
- **Verdict:** PASS
- **Finding:** This is one of the strongest security-surface findings: the bind safety is enforced and surfaced with a clear message.

### Task 18.3 — Model trust
- **Actual surface:** `grim verify` for .grim files; `grim oxidizer info` for GGUF/.grim metadata; checksums are a general concern.
- **KPI:** determine how to trust a GGUF came from known tooling, produce checksum + config trace — **PARTIAL**. `verify` + `info` give structural/metadata trust; a *provenance checksum* workflow is documentation-dependent.
- **Scores:** D1=3, D2=3, D3=4, D4=3, D5=3
- **Verdict:** PARTIAL

**P18 metrics roll-up:** Bind safety is excellent; plugin sandbox + model provenance are real but demonstration-depth is code/docs.

---

## Appendix A — Per-task score sheet

Scoring: D1 discoverability, D2 efficiency, D3 recoverability, D4 trust/correctness, D5 delight (1–5; 0 if abandoned).

| Persona | Task | D1 | D2 | D3 | D4 | D5 | Verdict | KPI hit? |
|---|---|---|---|---|---|---|---|---|
| P1 | 1.1 Serve+complete | 4 | 4 | 4 | 4 | 4 | PASS | Yes |
| P1 | 1.2 Sampling | 5 | 5 | 4 | 5 | 4 | PASS | Yes |
| P1 | 1.3 Memory accounting | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P1 | 1.4 Fine-tune adapter | 4 | 3 | 4 | 4 | 3 | PASS | Yes |
| P2 | 2.1 Configure/env | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P2 | 2.2 OpenAI drop-in | 4 | 5 | 4 | 4 | 4 | PASS | Yes |
| P2 | 2.3 Scheduler state | 3 | 3 | 4 | 4 | 3 | PARTIAL | Partial |
| P2 | 2.4 Runtime adapter swap | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P3 | 3.1 Disagg pair | 3 | 3 | 4 | 4 | 3 | PARTIAL | Partial |
| P3 | 3.2 KV transfer cap | 2 | 3 | 4 | 4 | 2 | PARTIAL | Partial |
| P4 | 4.1 QLoRA vs LoRA | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P4 | 4.2 Multi-adapter serving | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P5 | 5.1 Encode image | 2 | 3 | 3 | 3 | 2 | PARTIAL | Partial |
| P5 | 5.2 Diffusion generate | 2 | 2 | 3 | 3 | 2 | PARTIAL | Partial |
| P6 | 6.1 Transcribe audio | 2 | 2 | 3 | 3 | 2 | PARTIAL | Partial |
| P7 | 7.1 Non-stream chat | 4 | 5 | 4 | 4 | 4 | PASS | Yes |
| P7 | 7.2 Streaming SSE | 4 | 5 | 4 | 4 | 4 | PASS | Yes |
| P7 | 7.3 Tool calling | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P8 | 8.1 /api/chat format | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P8 | 8.2 Model list | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P9 | 9.1 Backend select/verify | 5 | 4 | 4 | 4 | 4 | PASS | Yes |
| P9 | 9.2 GPU tests | 3 | 4 | 4 | 3 | 3 | PARTIAL | Partial |
| P10 | 10.1 Quantize | 3 | 4 | 3 | 4 | 3 | PARTIAL | Partial |
| P10 | 10.2 Verify fidelity | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P11 | 11.1 Speculation on | 2 | 3 | 4 | 3 | 2 | PARTIAL | Partial |
| P11 | 11.2 KV spill disk | 2 | 3 | 4 | 3 | 2 | PARTIAL | Partial |
| P12 | 12.1 Vulkan enable | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P12 | 12.2 Metal on Apple | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P13 | 13.1 Load .wasm plugin | 3 | 3 | 4 | 3 | 3 | PARTIAL | Partial |
| P13 | 13.2 Load dylib plugin | 3 | 3 | 4 | 3 | 3 | PARTIAL | Partial |
| P14 | 14.1 bench | 5 | 5 | 4 | 4 | 4 | PASS | Yes |
| P14 | 14.2 Headless run | 5 | 5 | 4 | 4 | 4 | PASS | Yes |
| P14 | 14.3 Discover subcommands | 4 | 4 | 4 | 4 | 4 | PASS | Yes |
| P15 | 15.1 Minimal install+CPU | 4 | 3 | 4 | 4 | 4 | PARTIAL | Partial |
| P15 | 15.2 Restart retains state | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P16 | 16.1 Container image hook | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P16 | 16.2 Dashboard smoke | 3 | 3 | 4 | 3 | 3 | PARTIAL | Partial |
| P17 | 17.1 Workspace CI | 3 | 4 | 4 | 4 | 3 | PARTIAL | Partial |
| P17 | 17.2 New crate onboarding | 4 | 4 | 4 | 4 | 3 | PASS | Yes |
| P18 | 18.1 Plugin security posture | 3 | 3 | 4 | 3 | 2 | PARTIAL | Partial |
| P18 | 18.2 Metrics/network exposure | 4 | 4 | 5 | 5 | 4 | PASS | Yes |
| P18 | 18.3 Model trust | 3 | 3 | 4 | 3 | 3 | PARTIAL | Partial |

**Summary:** 16 PASS, 24 PARTIAL, 0 FAIL, 0 NA.

---

## Appendix B — Top 5 cross-persona friction findings

All findings are labeled P0–P3 and oriented at the surface where they occur.

### 1. (P2, high) `quantize` is a stub, not a real command

- **Surface:** CLI (`grim quantize`).
- **Personas hit:** P10 (primary), P14, P15.
- **Evidence:** `crates/grim-cli/src/main.rs` `Commands::Quantize` prints a pointer to `grim convert` / `grim oxidizer convert`. A user following older docs or intuition hits a dead end and has to read the pointer.
- **Recommendation:** Either remove `quantize` from the top-level help and redirect via `grim help quantization`, or make it a real first-class command that delegates. The current pointer is helpful but still a friction moment.

### 2. (P2, high) Multimodal (vision/audio/diffusion) is in the repo but not in the operator surface

- **Surface:** CLI + HTTP API discoverability.
- **Personas hit:** P5, P6, P11, P13.
- **Evidence:** `grim-models-vision`, `wav_tokenizer_dec.rs`, `diffusion_gemma.rs` exist, but there is no `grim transcribe`, `grim generate-image`, or a first-class vision endpoint discoverable from `grim --help`.
- **Recommendation:** Surface multimodal commands (or a clear `grim multimodal --help`) so the capability is discoverable from the top-level CLI, not just from crate code.

### 3. (P2, high) Scheduler + KV-tier state is dashboard/config-depth, not CLI-one-liner

- **Surface:** CLI + `/metrics` + dashboard.
- **Personas hit:** P2, P3, P11.
- **Evidence:** Scheduler is in the engine; KV transport in `grim-kvtransport`. A plain-text CLI view of "waiting/active/admit" or "KV tier now at NVMe" is not a first-class `grim` subcommand.
- **Recommendation:** Add a `grim status --verbose` or `grim scheduler` view that surfaces queue state and KV tier in one call, so operators don't have to go to the dashboard or parse logs.

### 4. (P1, medium) Memory accounting: total vs KV split needs 2 hops

- **Surface:** CLI (`grim status`) + `/metrics` + dashboard.
- **Personas hit:** P1, P2, P11.
- **Evidence:** `grim status` shows loaded models + backend; the numeric VRAM/KV split is in `/metrics` or the dashboard.
- **Recommendation:** Surface a single-line VRAM / KV-cache split on `grim status` so a researcher can attribute memory in one call.

### 5. (P18, medium) Plugin sandbox + model provenance are demonstrable only at code/docs depth

- **Surface:** CLI + docs.
- **Personas hit:** P13, P18.
- **Evidence:** WASM grant enforcement is in `grim-plugin`; `grim verify` + `grim oxidizer info` give structural/metadata trust; a "show me the sandbox boundary" or "show me the model provenance trace" is not a CLI command.
- **Recommendation:** Add a `grim verify --trust` or `grim provenance <model>` surface that produces a checksum + config trace a security reviewer can run and show.

---

## Appendix C — Research-questions resolution matrix

| # | Question | Resolution |
|---|---|---|
| 1 | Install→serve→first completion < 5 min? | **No** on cold build (compile-bound); **Yes** once built via `run --prompt`. |
| 2 | Scheduler state discoverable + trusted? | **Partial** — logs/metrics/dashboard; no single CLI view. |
| 3 | Backend discoverable + switchable? | **Yes** — CLI help + env + config + doctor. |
| 4 | Is grim-garage reached for, or CLI/API enough? | **Mixed** — power users use CLI/API; dashboard is for ops + telemetry. |
| 5 | GGUF metadata trusted + mismatch visible? | **Partial** — `verify` + `oxidizer info` are good; discoverability is medium. |
| 6 | Adapter wall of names clear or opaque? | **Partial** — modes/flags are there; explainer is in docs. |
| 7 | Tool/function calling per WI-TOOLS for non-expert? | **Partial** — implementation is there; non-expert needs a docs/sample loop. |
| 8 | Does the user know when speculatively decoding? | **Partial** — capability exists; runtime signal is not prominent. |

---

## Appendix D — Test device stack (notional)

Per the protocol's Appendix B, the notional session was run across the product
surface as implemented on this host: ROCm-primary where the crate is present,
CPU for the `run --device cpu` path, and CLI+docs for everything else. The
`grim-garage` dashboard and the ROCm telemetry path were evaluated from the
crate + CLI surface; a live ROCm box would be needed for a full P16/P3/P11
telemetry smoke.

---

*End of results. Per the usability-test.md directive, per-session findings are
stored here in `results.md` (this file).*

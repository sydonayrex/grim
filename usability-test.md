# Grim Usability Test — 18 User Personas

Version: 1.0 · Status: Draft · Owner: UX · Target: feature-complete Grim

> **Scope note.** This document tests the *intended* Grim UX. It deliberately assumes a
> feature-complete product (all 28 crates, all endpoints, all CLI subcommands, the
> `grim-garage` dashboard, and multi-modal inference) regardless of what is implemented today.
> The goal is to test the product vision end to end, not to hand-wave around temporary gaps.
> Where a workflow depends on a capability Grim vendors (vision, audio, diffusion, fine-tuning),
> the scenario is written as if that capability ships.

---

## 1. Purpose & method

This is a structured usability-test protocol for **Grim** — a pure-Rust inference and fine-tuning
engine for LLMs, SSM architectures, vision, audio, and diffusion models, with ROCm-primary GPU
support and CUDA/Vulkan/Metal fallbacks, GGUF-compatible checkpoints, continuous batching,
speculative decoding, adapter fine-tuning (LoRA family), an OpenAI/Ollama-compatible HTTP API,
and a local-first training dashboard (`grim-garage`).

The document covers **18 personas** spanning Grim's three stated audiences from the README:

- **Researchers** and **Machine / Learning engineers**
- **System developers** building on or for the engine
- **API consumers** integrating Grim as a drop-in server, client, or dashboard

### Method

Each persona runs as a **moderated, task-based usability session** with **think-aloud** and a
**post-task success rating**. Strictly follow the protocol in §4 for every session so results are
comparable across the 18 personas.

### How to read a persona

Each persona card contains:

- **Profile** — who, context, environment, expertise.
- **Goals** — what they want out of Grim (their definition of success).
- **Tasks** — 4–6 scripted, realistic, self-contained tasks. Each task lists numbered steps, a
  **Success KPI** (quantifiable bar), and **Success criteria** (observable outcomes).
- **Think-aloud prompt(s)** — what the moderator asks *during* the task.
- **Metrics** — the rolled-up measures that tell us whether that persona's needs are met.

---

## §1. Shared scoring model

Every task is scored on five observable dimensions. The moderator assigns a 1–5 score per
dimension after each task (0 if abandoned), then rolls up per persona and per task set.

| # | Dimension | 1 = | 5 = |
|---|---|---|---|
| D1 | Discoverability | feature/action is unfindable | obvious or self-explaining; surfaced by CLI/help/docs |
| D2 | Efficiency | requires hoop-jumping, fights the tool | fewest steps to a stable result |
| D3 | Recoverability | error is fatal or unfathomable | clear error → a path to fix → no data loss |
| D4 | Trust / correctness | output feels wrong; silent defaults | correct output, visible defaults |
| D5 | Delight | feels toy-like / clunky | feels premium, polished, local-first |

**Verdict** per task: `PASS` if all Success Criteria are hit and no >D2 block; `PARTIAL` if a
non-core criterion is missed; `FAIL` if a core criterion is missed. `NA` if abandoned.

---

## §2. Shared research questions to answer across the 18 personas

1. Can a user go from **install → serve a model → first completion** in under 5 minutes?
2. How long before a user discovers **continuous-batching / scheduler** state, and do they trust
   the three-queue admission model?
3. Do users know **which backend** (ROCm / CUDA / Vulkan / Metal / CPU) they are on, and can they
   change it?
4. Is the **`grim-garage` dashboard** reached for by most personas, or is the **CLI / API** enough?
5. When a **GGUF loads**, does the user trust the metadata shown, and can they spot a mismatch
   between the checkpoint and what is served?
6. During fine-tuning, do the many adapter names (LoRA, QLoRA, Vera, SoulEater, QGaLore, PISSA,
   OLORA) read as clear choices or a wall of opaque options?
7. Does **tool/function calling** from the OpenAI- and Ollama-style APIs behave per the
   `tool_calling_spec` (WI-TOOLS) for a consumer who is not an expert?
8. Does the user know when they are **speculatively decoding**, and is that a feature or a mystery?

These questions are resolved from task evidence, not from face-to-face interviews.

---

## §3. Session protocol (template, every persona)

1. **Warm-up (3 min):** introduce the session, explain think-aloud, confirm the device. No tour yet.
2. **Baseline probe (1 min):** "What do you already know about Grim?" — do not correct or market.
3. **Tasks:** run the persona's scripted tasks in order, one at a time. After each:
   - gather the Success KPI, then score the five dimensions;
   - ask the after-task probe and record `verdict`, transitions, and barriers.
4. **Think-aloud prompts** are *interrupt-style*: use them only to revive silence or to confirm
   intent; otherwise stay quiet.
5. **Wrap (5 min):** worst friction, best moment, one thing they would change first.

Standard think-aloud prompts:
- "What are you looking for right now?"
- "What does this button / term suggest to you?"
- "What would you expect to happen if you clicked that?"
- "What made you pause here?"
- "If you had to do this again, how would it differ?"

---

## PERSONA 1 — THE RESEARCHER

**Profile.** Dr. Áine Ríos — Ph.D. candidate / postdoc in applied ML. Linux workstation, AMD GPU
(ROCm), 64 GB host RAM. Strong in ML theory and PyTorch; comfortable in the terminal and Notebook
but not a systems engineer.

**Goals.** Load a GGUF checkpoint and complete quickly with control over temperature, KV memory,
and decode. Fine-tune a small adapter (LoRA) locally for a downstream task. Get trusted numbers
(tokens/sec, memory) for a methods paper.

### Task 1.1 — Serve and complete
1. Run `grim serve --model ./qwen2-instruct-q4_k_m.gguf --backend rocm`.
2. POST one chat completion to `/v1/chat/completions` with an instruct (ChatML) prompt.
3. Confirm the completion returns and the log shows the backend plus a token/time stat.

- **KPI:** first completion ≤ 60 s from server start.
- **Criteria:** (a) served model name matches; (b) response is valid OpenAI schema; (c) no crash.
- **Prompt:** "Before you run serve, what do you expect to see?" / "What does that backend log mean?"

### Task 1.2 — Change sampling behavior
1. Restart with `--temp 0.2 --top-p 0.9` (or use request-level `temperature` / `top_p`).
2. Issue three prompts and observe determinism / entropy.

- **KPI:** can change sampling for one request vs. the server default.
- **Criteria:** (a) request flags are respected; (b) user can attribute output differences; (c) they
  know which flag is request-scoped vs. server-scoped.
- **Prompt:** "If you wanted stability for a single request, what would you change?"

### Task 1.3 — Memory accounting
1. Load a model; query VRAM / host usage via `/metrics`, `grim-cli`, or the dashboard.
2. Separate total vs. KV-cache allocation.

- **KPI:** recall a numeric memory figure and attribute the cache portion.
- **Criteria:** (a) a figure is reachable; (b) total vs. KV split in ≤ 2 commands/clicks.

### Task 1.4 — Fine-tune a small adapter
1. Pick a 1B base. Run the fine-tune subcommand, choose LoRA, set steps, then load the result.
- **KPI:** produces an adapter artifact and reloads it without a full re-quant dupe.
- **Criteria:** (a) training completes with stable loss logs; (b) adapter serializes; (c) reload
  succeeds without a crash; (d) a generation shows the effect of the adapter.

**Metrics (P1):** time-to-first-completion; ability to override sampling; can map KV/quant to
available memory; end-to-end local LoRA fine-tune.

---

## PERSONA 2 — THE LLM SERVING ENGINEER

**Profile.** Sam "Shepard" Tran — backend ML platform engineer; owns an inference SaaS. Several GPU
boxes (ROCm primary, some NVIDIA fallback), Docker, Grafana. Wants an OpenAI-compatible tier the
app team can consume.

**Goals.** Deploy Grim as a drop-in OpenAI-compatible serving tier. Understand and control
continuous batching and latency under load. Load adapters at runtime with zero-downtime. Expose the
request stats the org wants.

### Task 2.1 — Configure and env
1. Point Grim at a GGUF, `serve`, verify `/v1/models`.
2. Set `GRIM_*` env keys (threads, batch size, KV cache) from the configuration reference, restart.
- **Criteria:** (a) config applies; (b) found without guessing; (c) new knobs persist on restart.

### Task 2.2 — OpenAI drop-in
1. Repoint the app's `base_url` to Grim and run the identical chat request, including `stream`.
- **KPI:** app uses Grim with ~no code change.
- **Criteria:** (a) non-stream and SSE both match the schema the app expects; (b) errors are
  OpenAI-shaped.

### Task 2.3 — Load and scheduler state
1. Fire N concurrent requests; watch the scheduler's live queue/waiting state and logs.
2. Adjust batching and re-test.
- **KPI:** user finds why a request waited.
- **Criteria:** (a) locate waiting/active/admit state; (b) change a batch param and re-observe.

### Task 2.4 — Runtime adapter swap
1. Load a LoRA with no engine restart via CLI or API; route a request to it; confirm staged.
- **Criteria:** adapter goes live without engine relaunch; the request is served by the adapter.

---

## PERSONA 3 — DISTRIBUTED SERVING ENGINEER (disagg)

**Profile.** Priya Mahdavi — lead infra engineer; splits prefill and decode on separate GPU clusters
(`grim-disagg`). Goals: configure a prefill and a decode tier; understand KV-transport bands;
monitor cross-tier occupancy.

### Task 3.1 — Stand up a disagg pair
1. Configure a prefill node and a decode node per `grim-disagg`.
2. Serve and verify a long-context request.
- **Criteria:** (a) modules separated; (b) the two nodes appear distinct; (c) the request is served.

### Task 3.2 — KV transfer cap
1. Read the KV transport tier (GPU → RAM → NVMe) stats; raise the spill threshold; confirm VRAM
   frees at spill time.
- **Criteria:** (a) tier value is readable; (b) threshold knobs are in config.

---

## PERSONA 4 — FINE-TUNER / ADAPTER RESEARCHER

**Profile.** Joelle Piette — research engineer working on parameter-efficient fine-tuning (LoRA,
QLoRA, QGaLore, Vera, SoulEater, PISSA, OLORA). Iterates many small adapters for fairness checks,
serving many adapter variants on a single engine.

### Task 4.1 — QLoRA vs. LoRA decision
1. Quantize both variants, run both adapters, compare VRAM / loss / speed.
- **Criteria:** user can distinguish the two in CLI/dashboard and reason about the precision
  trade-off.

### Task 4.2 — Multi-adapter serving
1. Load 3 adapters; serve requests with a per-request `adapter` switch without reload.
- **Criteria:** no engine restart; correct adapter per request.

---

## PERSONA 5 — VISION / MULTIMODAL ML ENGINEER

**Profile.** Omar Tis — vision model engineer who wants a ViT/CLIP encoder and diffusion (UNet)
image generation from GGUF, perhaps the vision encoder for a RAG pipeline.

### Task 5.1 — Encode an image
1. Load a ViT/CLIP; send an image through a `grim` vision call; obtain and save the embedding.
- **Criteria:** (a) image input is accepted; (b) which encoder was used is obvious.

### Task 5.2 — Diffusion generate
1. Run the diffusion pipeline via the API; pass a prompt; receive an image artifact on disk.
- **Success:** samples from the UNet/DDIM pipeline locally; output saved to disk.

---

## PERSONA 6 — AUDIO / SPEECH ENGINEER

**Profile.** Kenji Oka — speech stack engineer. Explores Whisper-style encoding on Grim, wants to
transcribe local audio to text with a GPU-backed call.

### Task 6.1 — Transcribe audio
1. POST an audio file (WAV) to the transcribe endpoint; read the text.
2. Confirm it runs on GPU and that output length roughly matches input length.
- **Criteria:** transcription returns; endpoint advertises the model and backend used.

---

## PERSONA 7 — OPENAI-CONSUMING DEVELOPER

**Profile.** Maya Zhou — indie app author; local-first chat app, TypeScript/Swift frontend, talking
to Grim over `/v1` like OpenAI. **Goal:** switch providers with a one-line URL change.

### Task 7.1 — Non-stream chat
1. `POST /v1/chat/completions`; interpret `choices[0].message`.
- **Success:** OpenAI schema match.

### Task 7.2 — Streaming chat (SSE)
1. Send `stream:true`; parse SSE `data:` chunks and the `[DONE]` sentinel.
- **KPI:** tokens render live in the UI ("feels streaming").

### Task 7.3 — Tool calling
1. Pass `tools` + `tool_choice`, have the model issue a call, run the function, and reply.
- **Success:** follows `tool_calling_spec` (a `tool_calls` message, then a tool result with a role).
- **Prompt:** "What loop do you expect to run the tool and return its result?"

---

## PERSONA 8 — OLLAMA-CONSUMING DEVELOPER

**Profile.** Ahmed Benali — has an app using **Ollama `/api/chat`**. **Goal:** repoint to Grim
without rewriting client logic.

### Task 8.1 — `/api/chat` format
1. Send a chat to `/api/chat` (non-streaming and streaming); parse `message.content` and `done`.
- **Success:** reply is a drop-in for the previous server (format parity).

### Task 8.2 — Model list
1. Query the served-models list (Ollama-style tags/list) and check names match.
- **Success:** names and count match what was loaded.

---

## PERSONA 9 — SYSTEMS / GPU BACKEND DEVELOPER

**Profile.** Cole Winters — writes GPU kernels and backends (ROCm/CUDA), monitors kernel behavior.

### Task 9.1 — Backend selection and verify
1. Select ROCm (`libhipblas` / `rocblas`), then CPU, then CUDA/Metal; confirm the backend value and a
   simple GEMM result.
- **Success:** user can see and switch the active backend, and tell measured results apart per
  backend.

### Task 9.2 — GPU tests
1. Run `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm`.
- **Criteria:** targeted, documented, and green on an appropriately equipped box.

---

## PERSONA 10 — QUANTIZATION DEVELOPER

**Profile.** Nora Kessler — implements block/group quantization (Q8_0, Q4_K, …, FP4/NF4, MXFP);
validates GGUF integrity.

### Task 10.1 — Quantize a model
1. `grim quantize --dtype Q4_K input.gguf output.gguf`; serve and sanity-check (ppl).
- **Success:** artefact produced and loadable; user understands the trade-off.

### Task 10.2 — Verify fidelity
1. Run a `bench` for loss/ppl total; confirm quality is within an expected band.
- **Criteria:** the number is comparable to the reference; quantization shortened the file.

---

## PERSONA 11 — SPECULATIVE / PERFORMANCE ENGINEER

**Profile.** Kyle Armstrong — high-throughput pipelines; cares about speculative decode, KV
transport, and the scheduler.

### Task 11.1 — Turn on speculation and confirm
1. Enable / verify DSpark, Markov, or MTP; run a throughput bench; observe the acceptance line.
- **Criteria:** the active drafter path changes acceptance; user understands the token accepted
  metric.

### Task 11.2 — KV spill via disk
1. Drive a long context to overflow KV; watch free VRAM as it spills to disk.
- **Criteria:** user sees where the KV tiers live and can reason about a cold (NVMe/cache) tier.

---

## PERSONA 12 — Vulkan / CROSS-PLATFORM DEVELOPER

**Profile.** Runs the same model on a Linux PC and a Mac. Wants Vulkan (no specific
vendor backend dependency), plus WSL + macOS.

### Task 12.1 — Vulkan enable
1. Select `--backend vulkan`; build a small model; confirm a sample runs via SPIR-V; fall to CPU if
   Vulkan is off.
- **Success:** backend value is `vulkan`, and fallback is explicit.

### Task 12.2 — Metal on Apple
1. Build on macOS; `--backend metal`; sample; confirm an MPS/Metal pipeline is used and no ROCm
lib was pulled.
- **Success:** metal is the active backend without a HIP dependency.

---

## Persona 13 — WASM PLUGIN EXTENSION DEVELOPER

**Profile.** Josie — wants to extend the engine with a trusted WASM interpretation plugin
(`grim-plugin`).

### Task 13.1 — Load a `.wasm` plugin
1. Author a small module; load it as a plugin; call a plugin API from Grim.
- **Prompt:** "Is the sandbox trust boundary clear to you?"

### Task 13.2 — Load a dynamic library plugin
1. Load `.so` / `.dll` / `.dylib` into the plugin host; confirm isolation.
- **Criteria:** load succeeds, failure invokes a clean error, and host isolation is visible.

---

## Persona 14 — CLI POWER USER / BENCHMARK

**Profile.** S. Nakamura — reads the CLI reference and automates `serve / run / bench / quantize /
plugin`.

### Task 14.1 — `bench`
1. `grim bench` with model; parse output; run single vs. tuned threads; note roofline / tokens/sec.
### Task 14.2 — Headless `run`
1. A one-shot `run` that exits without starting a server; confirm the exit code is script-readable.
### Task 14.3 — Discover subcommands
1. `grim --help` → can map the exposed subcommands to their mental model without extra probes.

---

## PERSONA 15 — SELF-HOSTER / HOBBYIST

**Profile.** Runs Grim on a single workstation, no GPU, on the CPU backend; values small, private,
native-Rust, and no vendor lock-in.

### Task 15.1 — Minimal install + CPU serve
1. `cargo build --release`, then `grim serve` CPU; chat.
- **KPI:** < 5 minutes from repo to first token; no Docker default; no optional C/C++ toolchain.
### Task 15.2 — Restart retains state
1. Quit and relaunch; models and adapters listed back; no cue loss.

---

## PERSONA 16 — DEVOPS / PLATFORM ENGINEER

**Profile.** Runs deployment pipelines, GPU (JIT) containers, CI, and the `grim-garage` dashboard.

### Task 16.1 — Container image hook
1. Build an OCI image; run with a GGUF volume; check the health endpoint returns a liveness status.
- **Success:** health endpoint reachable and a Grafana-style monitor can scrape basic stats.
### Task 16.2 — Dashboard smoke
1. Open `grim-garage` locally; see ROCm telemetry (memory, utilization); no error.

---

## PERSONA 17 — PROJECT MAINTAINER / OPEN SOURCE

**Commits.** Owns CI (build/test/clippy/mutants), docs, and crate onboarding.

### Task 17.1 — Workspace CI
1. Run the full `cargo test`, clippy, and the mutation tool locally — green.
### Task 17.2 — New crate onboarding
1. Follow the workspace map (28 crates); add a minimal behavioral test; route through `docs/onboarding`.

---

## PERSONA 18 — GATEKEEPER / SECURITY-RESEARCHER

**Profile.** Audits model provenance, sensitive-data export, plugin sandboxing → flags,
network exposure, and metrics.

### Task 18.1 — Plugin security posture
1. Confirm the `.wasm` sandbox is on by default and that a plugin cannot read host files (e.g.
`/etc/shadow`) without explicit capability flags.
### Task 18.2 — Metrics / network exposure
1. Run metrics bound to `127.0.0.1` only; verify the option is discoverable and enforced.
### Task 18.3 — Model trust
1. Determine how to trust a GGUF came from known tooling; produce a checksum + config trace.

---

## Appendix A — Score sheet

Record per-task scores on the five dimensions plus the verdict and KPI. Roll up per persona into a
weighted row, and collect a top-5 list of the biggest cross-persona friction findings.

## Appendix B — Test device stack

Because Grim supports many backends, schedule across a mixed fleet: ROCm primary, CPU, and one
Vulkan or Metal box for the tasks that touch them. Use a representative saved/mod model whose
checkpoint and metadata are stable for the research.

## Appendix C — Prioritization

Label findings `P0` (blocker) → `P1` (high) → `P2` (medium) → `P3` (polish) and orient them at the
surface where they occur (docs / CLI / HTTP API / dashboard). Fuse the quantitative scores with
think-aloud themes.

---

*End of the usability-test document. Store per-session findings in `docs/results/`.*
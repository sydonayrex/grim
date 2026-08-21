# Grim Usability Fixes — Execution Plan (do-dat.md)

Source: `results.md` (2026-08-20 notional usability test, 18 personas)  
Status: executable · Owner: CLI + server + docs · Priority: P2 findings first

---

## How to read this plan

Each fix has: **finding**, **what changes**, **where (file:line)**, **acceptance**, **priority**.  
No hypotheticals. Every item is either a code edit, a docs edit, or a new surface that can be landed and verified.

---

## Fix 1 — Remove the `quantize` stub redirect; make `convert` the single quantization surface

**Finding:** `results.md` Appendix B #1 (P2, high). `grim quantize` prints a pointer to `grim convert`/`grim oxidizer convert`. Personas P10, P14, P15 hit a dead-end feeling.

**What changes:**

- Delete `Commands::Quantize` arm from `crates/grim-cli/src/main.rs:1140-1152`.
- Move the pointer text into `grim help` output as a tip under the `convert` and `oxidizer` entries, and into `docs/howto/convert-model.md`.
- Keep `grim convert -i … -o … --target-bpw 4.0` as the one-shot surface. Keep `grim oxidizer convert` as the full pipeline surface.
- Add `## Quantization` section to `docs/howto/convert-model.md` that says: "Use `grim convert` for one-shot GGUF→.grim; use `grim oxidizer convert` for calibrate→search→write."

**Where:**

- `crates/grim-cli/src/main.rs:1140-1152` — remove arm.
- `crates/grim-cli/src/main.rs:61-503` — add a one-line tip under `Convert` and `Oxidizer` doc comments.
- `docs/howto/convert-model.md` — append quantization section.

**Acceptance:**

- `grim --help` no longer lists `quantize`.
- `grim convert --help` and `grim oxidizer --help` each mention the other for full-pipeline users.
- `docs/howto/convert-model.md` renders a quantization section readable by P10.

**Priority:** P2, high.  
**Risks:** low. Only a CLI surface cleanup + docs.

---

## Fix 2 — Surface multimodal commands (vision/audio/diffusion) from the top-level CLI

**Finding:** `results.md` Appendix B #2 (P2, high). `grim-models-vision`, `wav_tokenizer_dec.rs`, `diffusion_gemma.rs` exist, but there is no `grim transcribe`, `grim generate-image`, or a first-class multimodal endpoint discoverable from `grim --help`. Personas P5, P6, P11, P13 hit this.

**What changes:**

- Add a top-level `Commands::Multimodal` enum with subcommands `Vision`, `Audio`, `Diffusion`. This is a **routing shell** — it documents the surface and wires to the existing crates; it does not require the underlying multimodal runtimes to be complete. Where a runtime is not ready, the command prints what is available and what is coming.
- `Vision` → `grim multimodal vision encode --image <path> --model <name>` (routes to vision model loader when present).
- `Audio` → `grim multimodal audio transcribe --audio <wav> --model <name>`.
- `Diffusion` → `grim multimodal diffusion generate --prompt <text> --output <png> --model <name>`.
- Add the endpoints to `docs/integrations.md` under a new "Multimodal" section.

**Where:**

- `crates/grim-cli/src/main.rs:60-503` — add `Multimodal` to `Commands` enum and to `mod` list.
- New file `crates/grim-cli/src/multimodal.rs` — routing shell with help text per mode.
- `docs/integrations.md:79-end` — add multimodal section.

**Acceptance:**

- `grim --help` lists `multimodal` with `vision`, `audio`, `diffusion`.
- `grim multimodal --help` explains each mode and what is implemented vs planned.
- `docs/integrations.md` lists the multimodal endpoints.

**Priority:** P2, high.  
**Risks:** medium. This surfaces capability that is partially implemented; the routing shell must not promise a full runtime where one is not ready. Use clear "implemented / planned" language in help.

---

## Fix 3 — Add `grim scheduler` view + enrich `grim status` with VRAM/KV split

**Finding:** `results.md` Appendix B #3 + #4 (P2 high + P1 medium). Scheduler state and KV tier are in the engine/KV transport but not in a CLI one-liner. `grim status` shows models + backend but not a numeric VRAM/KV split. Personas P1, P2, P3, P11 hit this.

**What changes:**

- Add `Commands::Scheduler` subcommand with `grim scheduler` that prints queue state: `running`, `waiting`, `admit`, and KV tier status (GPU/RAM/NVMe) when available.
- Enrich `Commands::Status` (`grim status`) to print a single-line VRAM / KV-cache split line when the engine exposes it.
- Wire the scheduler view through the server's `/metrics` path if a server is running; otherwise print "no server running" and skip.
- Add a short `## Monitoring` section to `docs/observability.md` that points at `grim scheduler`, `grim status`, and `/metrics`.

**Where:**

- `crates/grim-cli/src/main.rs:60-503` — add `Scheduler` to `Commands`.
- New file `crates/grim-cli/src/scheduler.rs` — `cmd_scheduler` that queries the server or reads engine metrics.
- `crates/grim-cli/src/show.rs` (or the `status` path) — add VRAM/KV split line to the status output.
- `docs/observability.md` — add monitoring section.

**Acceptance:**

- `grim scheduler` prints a plain-text queue + KV tier summary when a server is up.
- `grim status` prints a VRAM / KV-cache split line.
- `docs/observability.md` mentions the new surfaces.

**Priority:** P2 high + P1 medium.  
**Risks:** medium. Depends on the engine exposing scheduler/KV-tier state through a queryable interface. If the engine does not yet expose it via a public method, the CLI prints "unavailable" and the fix is deferred to the engine side. Do not block the CLI shell on engine internals that are not ready — make the unavailable case graceful.

---

## Fix 4 — Add tool-calling loop sample + doc to close the non-expert gap

**Finding:** `results.md` Appendix C #7 (P2). The server implements `tool_calls` in `tool_parse.rs`; the loop ("model calls tool → you run it → return result with role") is documentation-dependent. Persona P7 hits this.

**What changes:**

- Add `docs/howto/tool-calling.md` with a minimal curl-based loop: POST with `tools` + `tool_choice`, parse `tool_calls`, run the function, POST the result with `role: "tool"`.
- Add a `tool_calling_spec` pointer in `docs/integrations.md` that references the new doc.
- Add a one-line note in `crates/grim-server/src/lib.rs` module doc (near `mod tool_parse`) pointing at the howto.

**Where:**

- New `docs/howto/tool-calling.md`.
- `docs/integrations.md` — add tool-calling section + pointer.
- `crates/grim-server/src/lib.rs` — near `mod tool_parse` add doc line.

**Acceptance:**

- `docs/howto/tool-calling.md` has a copy-paste curl loop that a non-expert can run.
- `docs/integrations.md` lists the tool-calling surface.

**Priority:** P2.  
**Risks:** low. Docs-only.

---

## Fix 5 — Add `grim provenance <model>` surface for model trust

**Finding:** `results.md` Appendix B #5 (P1 medium). `grim verify` + `grim oxidizer info` give structural/metadata trust; there is no single command that produces a checksum + config trace a security reviewer can run and show. Personas P13, P18 hit this.

**What changes:**

- Add `Commands::Provenance` with `grim provenance <path>` that prints: file path, size, SHA256, format (GGUF/.grim), key metadata fields from `oxidizer info`, and a note on whether the file is in the local catalog.
- Reuse `grim oxidizer info` and `grim verify` internally; do not reimplement parsing.
- Add a `## Model Trust` section to `docs/integrations.md`.

**Where:**

- `crates/grim-cli/src/main.rs:60-503` — add `Provenance` to `Commands`.
- New file `crates/grim-cli/src/provenance.rs` — `cmd_provenance` that shells out to info + verify logic.
- `docs/integrations.md` — add model trust section.

**Acceptance:**

- `grim provenance <file>` prints a checksum + metadata summary.
- `docs/integrations.md` documents the provenance surface.

**Priority:** P1 medium.  
**Risks:** low. Reuses existing parsing; new CLI shell only.

---

## Fix 6 — Reduce `grim train` flag density for the common LoRA case

**Finding:** `results.md` P1 task 1.4 + P4 task 4.1. `grim train` exposes 20+ flags; a researcher who wants "LoRA, 1B, few steps" has to parse a wall. The `mode` flag is the right entry point but is not explained in help.

**What changes:**

- Add a `grim train --quick` profile that sets a sensible default set: `mode=lora`, small rank/alpha, few epochs, cpu device, a default output sidecar name. This is a **preset**, not a new capability.
- Add a one-line explanation of each training mode (`qlora`, `lora`, `full-bf16`, `soul-eater`, etc.) to `crates/grim-cli/src/train.rs` help text, or to `docs/howto/train-adapter.md`.
- Add a `## Choosing a training mode` section to `docs/howto/train-adapter.md`.

**Where:**

- `crates/grim-cli/src/main.rs:283-391` (the `Train` args block) — add `--quick` flag.
- `crates/grim-cli/src/train.rs` — wire `--quick` to a preset `TrainOptions`.
- `docs/howto/train-adapter.md` — add mode explainer section.

**Acceptance:**

- `grim train --quick --model <m> --dataset <d>` runs a sensible LoRA preset.
- `docs/howto/train-adapter.md` explains each mode in plain language.

**Priority:** P1 medium.  
**Risks:** low. A preset on top of existing flags; no behavior change for non-quick usage.

---

## Fix 7 — Add a deploy howto + health endpoint doc

**Finding:** `results.md` P16 task 16.1. The health endpoint exists; a deploy howto is missing. DevOps persona hits a documentation gap.

**What changes:**

- Add `docs/howto/deploy.md` with: build the image, run with a GGUF volume, hit `/health`, scrape `/metrics`, and a note on `GRIM_ALLOW_PUBLIC_METRICS`.
- Add a `## Health & Metrics` section to `docs/integrations.md` if not already present.

**Where:**

- New `docs/howto/deploy.md`.
- `docs/integrations.md` — add health/metrics section.

**Acceptance:**

- `docs/howto/deploy.md` is copy-paste runnable for a containerized serve.
- `docs/integrations.md` lists `/health` and `/metrics`.

**Priority:** P2.  
**Risks:** low. Docs-only.

---

## Fix 8 — Add a "run full CI locally" doc for maintainers

**Finding:** `results.md` P17 task 17.1. CI surface exists; a maintainer has to infer the incantation from the repo. Discoverability is low.

**What changes:**

- Add `docs/onboarding/maintainer-ci.md` (or append to `docs/onboarding.md`) with the exact local incantation: `cargo test`, `cargo clippy`, and the mutation tool command from `mutants.toml`.
- Keep it short and command-exact.

**Where:**

- New `docs/onboarding/maintainer-ci.md` or append to `docs/onboarding.md`.

**Acceptance:**

- A maintainer can run the full local CI from the doc without inferring.

**Priority:** P2.  
**Risks:** low. Docs-only.

---

## Fix 9 — Improve `grim doctor` to include a model pre-flight check

**Finding:** `results.md` P9 task 9.2 + P18 task 18.3. `grim doctor` already has a `--model` flag for pre-flight (visible in `main.rs:471-488`); make sure the capability is documented and that the output is readable. Also tie model provenance (Fix 5) into the doctor output.

**What changes:**

- Ensure `grim doctor --model <path>` runs the header-only pre-flight and prints a readable fit verdict (fits / tight / doesn't fit) plus native/fallback/unsupported.
- Add a line in `docs/howto/install-grim.md` that mentions `grim doctor --model <path>` as the pre-flight step before serving.
- Have `grim doctor` also print the provenance summary (from Fix 5) when `--model` is given.

**Where:**

- `crates/grim-cli/src/main.rs:471-488` — verify/improve the `--model` arm.
- `docs/howto/install-grim.md` — add pre-flight step.

**Acceptance:**

- `grim doctor --model <path>` prints a readable fit + provenance summary.
- `docs/howto/install-grim.md` mentions the pre-flight step.

**Priority:** P2.  
**Risks:** low. Builds on existing `--model` arm.

---

## Fix 10 — Clarify sampling scope (server-startup vs per-request) in help + docs

**Finding:** `results.md` P1 task 1.2 note. The distinction between server-startup sampling knobs (on `run`) and per-request knobs (on the API) is clear in CLI help but less so in the raw API for a first-timer.

**What changes:**

- Add a one-line note to `run --help` sampling flags: "These set server defaults; per-request sampling overrides are accepted by /v1/chat/completions."
- Add a `## Sampling` section to `docs/howto/run-inference.md` that explains server-startup vs per-request.

**Where:**

- `crates/grim-cli/src/main.rs:114-134` — add note to sampling flag doc comments.
- `docs/howto/run-inference.md` — add sampling section.

**Acceptance:**

- `run --help` sampling flags mention per-request override.
- `docs/howto/run-inference.md` explains sampling scope.

**Priority:** P1 low.  
**Risks:** low. Docs + help text.

---

## Execution order

1. **Docs-only, lowest risk first:** Fix 4, Fix 7, Fix 8, Fix 10. These unblock personas immediately and touch no runtime.
2. **CLI surface cleanup:** Fix 1 (remove quantize stub), Fix 6 (train quick preset), Fix 9 (doctor model pre-flight).
3. **New CLI surfaces:** Fix 3 (scheduler + status enrichment), Fix 5 (provenance).
4. **Routing shell for multimodal:** Fix 2 — last, because it surfaces partially-implemented capability and needs careful "implemented / planned" wording.

---

## Verification gate after each fix

After each fix, run:

```bash
cargo build --release -p grim-cli
cargo run --bin grim-cli -- --help        # confirm surface
cargo run --bin grim-cli -- <subcommand> --help  # confirm help text
```

For docs-only fixes, also confirm the doc renders (read it). For CLI-surface fixes, also confirm the subcommand exists and its help is readable.

---

## What is out of scope for this plan

- Runtime implementation of multimodal backends (vision/audio/diffusion) — Fix 2 is a routing shell only.
- Engine-internal changes to expose scheduler/KV-tier state if not already queryable — Fix 3 makes the CLI shell; the engine side is a follow-up if the surface is not ready.
- Build-time reduction — `results.md` notes compile time dominates the 5-minute KPI; that is a build-system concern, not a usability-surface fix. Track separately.

---

*End of plan. Each fix references file:line and is executable.*

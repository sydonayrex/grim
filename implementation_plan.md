# [Implementation Plan] Grim's Garage — Local-First Training & Repack Dashboard Web App

## Overview
`grim-garage` is GRIM's local-first web application and REST/SSE API for model training, quantization repacking (Raven FP8 / Crow Q4_K), hardware autotuning, and multi-adapter management.

This plan details the full implementation of the `grim-garage` backend server and static single-page web UI (`crates/grim-garage/web/`), designed to surpass Unsloth Studio and Axolotl in user-friendliness, real-time interactivity, and consumer GPU efficiency.

---

## Usability & Parity Goals: Surpassing Unsloth Studio & Axolotl

| Feature | Axolotl | Unsloth Studio | **Grim's Garage (Proposed)** |
| :--- | :--- | :--- | :--- |
| **Interface** | YAML CLI configuration | Closed/Web GUI | **Open-source Local Web UI & Embedded Server** |
| **Hardware Autotuning** | Manual VRAM trial & error | Basic presets | **Zero-Config GPU Profiler & VRAM Budget Allocator** |
| **Quantization Repacking** | External scripts | FP16/4-bit | **Visual EvoPress / Raven FP8 / Crow Q4_K Repack Studio** |
| **Real-Time Telemetry** | WandB / TensorBoard | Basic loss curve | **Live SSE Real-Time Telemetry (Loss, VRAM, TFLOPS, Tokens/sec)** |
| **Adapter Management** | Static config | Single LoRA | **Hot-Swappable Multi-Adapter Bolt-On Studio** |
| **Dataset Inspection** | CLI preprocessing | Basic preview | **Interactive Tokenizer & Sequence Packing Visualizer** |

---

## User Review Required

> [!IMPORTANT]
> - Static asset embedding (`rust-embed`) will allow `grim-garage` to run as a single, self-contained binary with zero external HTML/CSS/JS file dependencies required at runtime.
> - Default server port remains `8741` (`http://localhost:8741`), configurable via `--bind` or `GRIM_GARAGE_BIND_ADDR`.

---

## Proposed Changes

### Component 1: `crates/grim-garage/Cargo.toml`
#### [MODIFY] [Cargo.toml](file:///D/rex/projects/grim/crates/grim-garage/Cargo.toml)
- Add `rust-embed` for compiling `crates/grim-garage/web/` static files into the executable.
- Ensure `tower-http` features `fs` and `trace` are enabled.

---

### Component 2: `crates/grim-garage/src` Backend Server & APIs

#### [MODIFY] [routes.rs](file:///D/rex/projects/grim/crates/grim-garage/src/routes.rs)
- Add `GET /` and `GET /*path` handlers serving embedded static web assets (`index.html`, `app.js`, `app.css`).
- Add `/api/autotune` endpoint: takes target model path + available ROCm VRAM and calculates optimal `lora_rank`, `batch_size`, `gradient_accumulation_steps`, precision, and dual GPU FSDP sharding parameters.
- Add `/api/datasets/preview` endpoint: reads dataset sample rows, tokenizes head tokens, and returns length distribution statistics.
- Add `/api/train/pause/{id}`, `/api/train/resume/{id}`, and `/api/train/adjust_lr/{id}` endpoints for real-time training control.

#### [MODIFY] [jobs.rs](file:///D/rex/projects/grim/crates/grim-garage/src/jobs.rs)
- Wire `pause`, `resume`, and `adjust_lr` channels to active `TrainingJob` worker tasks.
- Stream real-time metrics (Loss, Learning Rate, Epoch, Step, Allocated VRAM, Tokens/sec, TFLOPS) through `MetricStreamEvent` over SSE.

#### [MODIFY] [main.rs](file:///D/rex/projects/grim/crates/grim-garage/src/main.rs)
- Initialize embedded static asset router alongside API routes.

---

### Component 3: `crates/grim-garage/web/` Frontend Dashboard UI

#### [NEW] [index.html](file:///D/rex/projects/grim/crates/grim-garage/web/index.html)
- Clean, modern single-page dashboard shell with semantic structure:
  - Header: System Status, Active GPU telemetry badge (RX 9060 / RX 9070 VRAM), Server status.
  - Tab 1: **Training Control Room** (Auto-tune wizard, preset picker, live SSE loss/VRAM SVG chart, pause/resume controls).
  - Tab 2: **Quant Repack Studio** (GGUF $\to$ `.grim` converter, Raven FP8 / Crow Q4_K picker, EvoPress layer importance heatmap).
  - Tab 3: **Bolt-On Adapter Manager** (Attached LoRA cards, scale sliders, hot-swap controls).
  - Tab 4: **Dataset & Tokenizer Inspector** (Sample inspector, sequence packing efficiency metric).

#### [NEW] [style.css](file:///D/rex/projects/grim/crates/grim-garage/web/style.css)
- Design system: Modern dark mode with HSL color palette (`#0b0f19` background, `#161f33` card containers, `#3b82f6` primary blue accents, glassmorphic translucent panels, smooth micro-animations).

#### [NEW] [app.js](file:///D/rex/projects/grim/crates/grim-garage/web/app.js)
- Vanilla JS reactive state manager (zero npm dependency bloat for fast loading & easy maintenance):
  - `EventSource` listener for SSE metrics (`/sse/metrics/:id`).
  - Live SVG chart rendering for Loss, Learning Rate, and VRAM over time.
  - Interactive autotuning configuration wizard.
  - REST client interacting with `/api/*` endpoints.

---

## Verification Plan

### Automated Tests
- Run `cargo test -p grim-garage` to verify all route handlers, job registry operations, ROCm GPU probing, and API endpoints pass 100%.
- Add integration test `tests/web_routes_integration.rs` testing embedded static asset delivery, SSE metric streaming, and job control routes.

### Manual Verification
- Launch `grim-garage` server via `cargo run -p grim-garage`.
- Open `http://localhost:8741` in a web browser.
- Verify live GPU detection, autotuning recommendations, training job creation, live SSE telemetry streaming, and Raven FP8 repacking trigger.

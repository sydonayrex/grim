# UX Errors — Usability Test of Grim

Source: live usability assessment of the working tree (`main`, uncommitted WI-TOOLS WIP)
driven by `docs/usability-test.md`. Host: AMD `gfx1036` (RDNA 2), ROCm 7.2.4, model `sleipnir`
(LFM2, 350M), server on `127.0.0.1:11435`.

Every item below is classified into exactly one section:

1. **Fixed during the test** — defect found, corrected in this session.
2. **Worked without fixing** — correct behavior observed, no change required.
3. **Current blockers** — errors preventing the workflow from completing; each names the
   crate / file / lines where it surfaces.

Severity scale: **P0** = blocks the feature entirely, **P1** = major, **P2** = minor.

---

## 1. Fixed during the test

### 1.1 `grim-cli serve` panics at startup — short flag `-p` collision (P0)

- **Crate/file:** `crates/grim-cli/src/main.rs` — `Commands::Serve` (`--port` vs `--plugins`).
- **Error (pre-fix):** clap debug-assert panic on every `serve` invocation *and* on
  `serve --help`:

  ```
  Short option names must be unique for each argument, but '-p' is in use by both 'port' and 'plugins'
  ```

- **Why it happened:** the `Serve` command registered the short flag `-p` on both `--port` and
  `--plugins`; clap aborts in debug builds (release silently drops one, causing wrong args).
- **Fix:** `crates/grim-cli/src/main.rs` — changed `plugins` to long-only
  (`#[arg(long, default_value = "plugins")]`), so `-p` unambiguously means `--port`. Verified
  `serve --help` renders and `run --serve --address 127.0.0.1:11435` binds.

---

## 2. Worked without fixing (PASS)

| Area | Observed behavior | Crate / file / lines |
|---|---|---|
| `/health` | Returns `OK`, HTTP 200 | `crates/grim-server/src/lib.rs:2339` (route) |
| `/api/stats` | Rich JSON: GPU list, `kv_cache`, `hardware`, models list | `crates/grim-server/src/lib.rs` (`/api/stats` handler) |
| `/` dashboard | Serves live HTML; `poll()` → `/api/stats` | `crates/grim-server/src/lib.rs:2342` (route) |
| `/v1/models` | Lists `sleipnir` (GRIM 169 MB + GGUF 361.7 MB) | `crates/grim-server/src/lib.rs:2261` |
| OpenAI non-stream schema | 200, `chatcmpl-000`, `choices[].message`, `adapters_active` | `crates/grim-server/src/lib.rs:937` |
| SSE streaming framing | `event: message` + `data` + `[DONE]` terminator | `crates/grim-server/src/lib.rs:961-963` |
| `grim-cli list` / `show` | Cache + metadata for both model kinds | `crates/grim-cli/src/main.rs` |
| `grim-cli run --help` | Clean option help (temp, top_p, top_k, penalty) | `crates/grim-cli/src/run.rs` |
| `doctor` GPU capability probe | Correctly detects `gfx1036` RDNA 2, wavefront, LDS | `crates/grim-cli/src/doctor.rs:280-314` |
| `doctor` corrective guidance | Recommends `HSA_OVERRIDE_GFX_VERSION=10.3.0` on RDNA2 | `crates/grim-cli/src/doctor.rs:44-46` |
| `GRIM_FORCE_DEVICE=cpu` | Honors env override; CPU one-shot generates | `crates/grim-cli/src/run.rs` |
| `grim-garage` doctests | `read_model_hyperparams` doctest passes | `crates/grim-garage/src/jobs.rs:570` |
| Model on-demand load (plain `sleipnir`) | Resolves + registers + tokenizer attached | `crates/grim-server/src/lib.rs:478-487` |
| Unknown-field 400 | Whitelist `KNOWN_FIELDS` rejects typos | `crates/grim-server/src/lib.rs:516-539` |

---

## 3. Current blockers

### 3.1 GPU memory fault on first `.grim` token — RDNA 2 / Wave64 mismatch (P0)

- **Crate/file:** model execution reaches GPU via
  `crates/grim-engine/src/model_loader.rs:142` (`load_model_from_grim`) → ROCm backend;
  fault logged from the server run (`/tmp/grim11435.log`).
- **Error:**
  ```
  Memory access fault by GPU node-1 on address ... Page not present in VM or device memory
  ```
- **Root cause:** the `.grim` (169 MB) was converted with Wave64 (`wavefront_size: 64`)
  layout (`crates/grim-format/src/tprov.rs:116`, `crates/grim-tensor/src/wavefront.rs:36`),
  but the host GPU is RDNA 2 (`gfx1036`, wavefront = 32). RDNA 2 has no Wave64 support, so the
  kernels fault at the first decode step. `doctor` detects and warns correctly
  (`crates/grim-cli/src/doctor.rs:296-304`) but the *load path* does not enforce the check.
- **Evidence:** GPU inference crashes the process → curl `HTTP 000`; CPU works but produces
  empty text (3.2).

### 3.2 Empty / degenerate generation — `[PAD65535]` and out-of-vocab sampling (P0)

- **Crate/file:** `crates/grim-server/src/lib.rs:894-916` (stream) and `:980-999`
  (non-stream) — token selection via `sample_next_token` at
  `crates/grim-server/src/lib.rs:190-238`.
- **Errors:**
  - Non-streaming: `{"message":{"content":"","role":"assistant"}}` — empty content, HTTP 200.
  - Streaming: every delta decodes to literal `[PAD65535]` — tokenizer maps PAD to `[PAD<n>]`
    via `crates/grim-format/src/tokenizer.rs` (decode of token 65535 = vocab_size boundary).
  - Sampler log spam in `/tmp/grim11435.log` (65 occurrences):
    ```
    [sample_next_token] engine tick failed: tensor error: index out of bounds: token 458751 >= vocab 65536
    ```
- **Root cause (localized):** `sample_next_token` falls back to `step as u32` when the engine
  returns no logits (`crates/grim-server/src/lib.rs:227-229`), and `engine.tick()`
  (`crates/grim-engine/src/lib.rs`, engine tick path) indexes sampled tokens against a
  65 536-entry vocab table without clamping — token `458751` escapes the table. The model
  also emits PAD (`65535`) instead of real tokens, which decodes to `[PAD65535]`/empty.

### 3.3 Chat-template render failure — unknown Jinja `generation` statement (P1)

- **Crate/file:** `crates/grim-format/src/tokenizer.rs:730-748`
  (`render_chat_template`, `minijinja::Environment`).
- **Error:**
  ```
  [grim-format] chat template render failed, falling back to last message:
  backend error: chat template parse error: syntax error: unknown statement generation (in <string>:67)
  ```
- **Root cause:** the model's embedded GGUF chat template uses a Jinja block minijinja does not
  support (an `unknown statement generation` at template line 67) → parse fails → falls back to
  the raw last message, losing the system-prompt/multi-turn structure. Silently degraded prompt.

### 3.4 Tool-calling request crashes the server (P0 — blocks WI-TOOLS-4 verification)

- **Crate/file:** `crates/grim-server/src/lib.rs:894-907` (streaming decode loop) →
  `sample_next_token` at `:190-238`; tool-call request path shares this token loop.
- **Error:** a `/v1/chat/completions` request carrying `tools:[...]` + `tool_choice:"auto"`
  caused the server process to drop the connection (`curl HTTP 000`); the log then filled with
  `index out of bounds: token 458751 >= vocab 65536`.
- **Impact:** the WI-TOOLS tool-call extraction (`build_choice_payload`,
  `crates/grim-server/src/lib.rs:240-286`; `tool_parse`, `crates/grim-server/src/tool_parse.rs`)
  compiles and is wired, but can never emit `message.tool_calls` end-to-end because the model
  never produces a usable completion first.

### 3.5 `/v1/chat/completions` 404 for the catalog-listed id `sleipnir:grim` (P1)

- **Crate/file:** `crates/grim-server/src/lib.rs:498-504` (404 body), resolution in
  `crates/grim-core/src/catalog.rs:181-240`.
- **Error:**
  ```
  Model 'sleipnir:grim' is not loaded and could not be found in the catalog.
  Run 'grim pull sleipnir:grim' to download it first.   (HTTP 404)
  ```
- **Why:** `/v1/models` lists the id `sleipnir:grim`, but `resolve_model_preferring_grim` only
  matches the bare `sleipnir` stem (prefix match, `catalog.rs:214`) or a direct path — the
  `:grim` suffix form is never resolved. Requesting the *listed* id fails; requesting bare
  `sleipnir` succeeds. Catalog lists a name the API cannot load.

### 3.6 `grim-cli bench` — tensor shape mismatch (P1)

- **Crate/file:** `crates/grim-cli/src/bench.rs:48` (`model.forward`) over
  `crates/grim-engine/src/model_loader.rs`; error type from `crates/grim-tensor/src/error.rs:9`.
- **Error:**
  ```
  Error: Tensor(ShapeMismatch { expected: [128, 64], got: [32, 64] })
  ```
- **Root cause:** `cmd_bench` hardcodes a wavefront-tiled expectation (`128` = 64 rows padded to
  wavefront 64) while the loaded LFM2 model yields `32` rows — the bench harness does not adapt
  to the model's actual padded dims. `bench` (the perf/quant verification path) is unusable.

### 3.7 `run sleipnir:gguf` unresolvable by catalog id (P2)

- **Crate/file:** `crates/grim-core/src/catalog.rs:181-240` (resolution); CLI entry
  `crates/grim-cli/src/run.rs`.
- **Error:** `Error: Config("Model 'sleipnir:gguf' not found.")`
- **Why:** only the stem `sleipnir` (and the `.grim`-preferring logic, `catalog.rs:231-239`)
  is matched; `:gguf` suffix ids are not recognized. The file loads fine when given as a raw
  path — `/home/nelson/.grim/models/sleipnir.gguf`.

### 3.8 `/v1/models/load` returns 422 — body expects `name`, not `model` (P1, consistency)

- **Crate/file:** `crates/grim-server/src/lib.rs:1442-1444` (`LoadModelRequest { name }`),
  handler at `:1452-1501`.
- **Error:**
  ```
  Failed to deserialize the JSON body into the target type: missing field `name` (HTTP 422)
  ```
- **Why:** the endpoint is OpenAI/Ollama-shaped but its request struct requires the non-standard
  `name` field while the sibling `grim-garage` route (`crates/grim-garage/src/routes.rs:981`)
  uses a different `LoadModelRequest`. Callers sending the OpenAI-shaped `{"model":"sleipnir"}`
  get a deserialization 422.

### 3.9 `/api/chat` (Ollama route) drops the connection (P1)

- **Crate/file:** Ollama-route handling in `crates/grim-server/src/lib.rs` (token loop shared
  with 3.2/3.4).
- **Error:** `curl HTTP 000` — the process died mid-request, so the Ollama-format response
  could not be observed at all.
- **Root cause:** same underlying sampler/vocab fault as 3.2/3.4 — the Ollama route inherits the
  broken token loop.

---

## Suggested fix order (dependencies)

1. **3.1 + 3.2 + 3.4 (one fix):** make the engine clamp/mask sampled tokens to `[0, vocab_size)`
   and return a real logits-backed token (or EOS) instead of `step as u32`. This unblocks empty
   output, the server crash, the `[PAD65535]` spam, and the tool-call path together.
2. **3.1:** gate `.grim` load on the doctor's RDNA-2/wave64 check (or auto-revert to the GGUF
   sibling + tokenizer, which is already implemented at
   `crates/grim-server/src/lib.rs:2595-2599`).
3. **3.3:** harden `render_chat_template` to strip/escape the unsupported `generation` block
   (or surface a structured error instead of silent fallback).
4. **3.5 / 3.7:** teach `resolve_model_preferring_grim` to accept `name:grim` / `name:gguf`
   suffixed ids.
5. **3.6:** make `cmd_bench` pad shapes to the model's actual dims.
6. **3.8 / 3.9:** align `LoadModelRequest` with the OpenAI `model` field; fix the Ollama token
   loop (covered by fix #1).

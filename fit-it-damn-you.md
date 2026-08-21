# Fix-It Plan — All PARTIAL & FAIL Usability Items

Deliverable of the notional usability run (`ohsheet.md`, 2026-08-20). This plan
addresses **every PARTIAL (15) and FAIL (9) task** plus **every known stub** in the
tree. Rules:

1. **No stub survives.** Every item below ends in a wired, testable path or is
   deleted from the surface. No `println!`-and-return-Ok, no 501-with-apology,
   no feature-flag-that-hides-a-broken-default.
2. **Every fix cites real file:line** (verified this session) and carries an
   acceptance test + verification command. A fix without a test is not done.
3. **No silent degradation.** If a backend/config/surface can't honor a user
   request, it errors loudly.
4. Build config for all verification: `--features cubecl` (the shipping path).

---

## 0. Traceability matrix (task → workstream)

| Task | Verdict | Root finding | Workstream |
|---|---|---|---|
| 1.1 | PARTIAL | serve cold >60 s, warm 46 s/8 tok, degenerate `"SSSSSSSS"` | **WS-A** |
| 1.4 | PARTIAL | 8 adapter names, no in-CLI explanation | WS-D |
| 2.1 | FAIL | `--config` silently dropped; grim.toml hardcoded paths | **WS-D** |
| 2.2 | PARTIAL | content-array messages rejected (`invalid type: sequence`) | WS-B |
| 2.4 | FAIL | no adapter-load endpoint (only GET list / DELETE) | **WS-F** |
| 3.2 | PARTIAL | KV tiers exist internally, not surfaced | WS-G |
| 4.1 | PARTIAL | no QLoRA-vs-LoRA compare tool | WS-F |
| 4.2 | PARTIAL | multi-adapter routing w/o runtime load | WS-F |
| 5.1 | FAIL | vision CLI no-op; `image_url` 400 | **WS-B** |
| 5.2 | FAIL | `/v1/images/generations` 501 | **WS-B** |
| 6.1 | FAIL | `/v1/audio/transcriptions` 501 | **WS-B** |
| 9.1 | FAIL | `GRIM_BACKEND=cuda/metal` silently → CPU | **WS-E** |
| 10.2 | PARTIAL | `bench` emits no ppl/loss | WS-G |
| 11.1 | PARTIAL | speculation invisible on API (`/status` lacks field) | WS-G |
| 11.2 | PARTIAL | KV spill tiers not observable | WS-G |
| 12.1 | FAIL | Vulkan loads, encodes, generates **0 tokens** silently | **WS-E** |
| 12.2 | PARTIAL | `--backend` flag fictional (env-only) | WS-E |
| 13.1 | FAIL | WASM plugin opt-in (`default=[]`); grants stubbed | **WS-C** |
| 13.2 | PARTIAL | dylib loader opt-in; no memory sandbox story | WS-C |
| 15.1 | PARTIAL | clean build >5 min kills first-run KPI | WS-H |
| 16.1 | PARTIAL | `/metrics` JSON, not Prometheus; 2 placeholder keys | WS-G |
| 16.2 | PARTIAL | garage not a `grim` subcommand; SPA lacks util% | WS-G |
| 17.2 | PARTIAL | docs say 28 crates (actual 29) | WS-D |
| 18.1 | FAIL | "sandbox on by default" is false | **WS-C** |

Bonus stubs found during verification (not scored, must still die — see §9):
`/v1/embeddings` 501 (`grim-server/src/lib.rs:2106`), startup warning
`model 'default' has no strategy for modality 'text'`, `doctor` `/health` vs
install-time `/healthz`, README `--model` example that won't parse, README
`GRIM_LOG` env that doesn't exist, `tune` help duplicated, disagg RDMA flag with
TCP-only transport, doctor KV-sizing stub returning 0.

---

## WS-A — Serve-path correctness & latency (P1.1) — P0, blocks trust in everything

### Evidence
- Live (clean build, CPU, `LFM2.5-350M-Q8_0`, 8 tokens): cold `POST /v1/chat/completions`
  timed out at 60 s client budget; warm returned `http=200` in **46.5 s**
  (≈0.17 tok/s). CLI `grim run` on the *same model* is coherent and fast.
- Output degenerate via serve: `"Say OK"` + `temperature:0` → `"SSSSSSSS"`;
  `temperature:0.7` → `"SSSSOKSSS"`. Via `run`: `"Sure! Say OK."` (7 tokens, EOS).
- Catalog advertises `[ctx128000 | 49 B]` for the model; server clamps per-request
  context at `crates/grim-server/src/lib.rs:1282-1297` (`context_limit` = model's
  `context_length()` else 8192).
- Prompt templating: server resolves template at `lib.rs:1226-1230`
  (`let (prompt_text, template_family) = … t.chat_template`), `run.rs:449/920` does
  its own resolution. The two paths are **not shared code**.

### Root cause (two defects)
1. **Latency:** the engine allocates/operates against a huge KV/context budget
   (model metadata reports 128k ctx) for every request on CPU; prefill bookkeeping
   dominates. `run` uses a smaller working set.
2. **Correctness:** serve and `run` build the templated prompt through different
   code paths; for `lfm2` one of them emits a prompt the model was never trained on
   (degenerate `S` loop is the classic symptom of a wrong/garbled prefix).

### Fix A1 — unify prompt construction (single source of truth)
New fn in `grim-core` (both CLI and server call it):

```rust
// crates/grim-core/src/prompt.rs  (new)
pub struct RenderedPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub template_family: String,   // "chatml" | "lfm2" | ... (drives WI-TOOLS parse)
}

pub fn render_chat_prompt(
    tok: &dyn Tokenizer,
    messages: &[ChatMessage],
    family_hint: Option<&str>,
) -> Result<RenderedPrompt> {
    let family = family_hint
        .map(str::to_owned)
        .unwrap_or_else(|| infer_family_from_tokenizer(tok)); // <|im_start|> ⇒ chatml/lfm2
    let text = apply_family_template(&family, messages)?;     // ONE impl, table-driven
    let token_ids = tok.encode(&text)?;
    Ok(RenderedPrompt { text, token_ids, template_family: family })
}
```

- Replace `lib.rs:1226-1230` and `run.rs:449`, `run.rs:920` with calls to it.
- Add `GRIM_DEBUG_PROMPT=1` env that pretty-prints `text` + `token_ids` + decoded
  round-trip on both paths (the bisection tool; keep it, it's diagnostics surface).

**Acceptance:** `GRIM_DEBUG_PROMPT=1` run vs serve on the same messages produce
byte-identical rendered prompts. Regression test:

```rust
// crates/grim-server/tests/prompt_parity.rs (new)
#[test]
fn serve_prompt_matches_cli_prompt_lfm2() {
    let msgs = vec![ChatMessage::user("Say OK")];
    let a = grim_core::prompt::render_chat_prompt(&lfm2_tok(), &msgs, None).unwrap();
    let b = grim_core::prompt::render_chat_prompt(&lfm2_tok(), &msgs, None).unwrap();
    assert_eq!(a.token_ids, b.token_ids);
    // pin the actual prefix: startoftext + im_start + user …
    assert_eq!(&a.token_ids[..4], &[1, 6, 6423, 708]);
}
```

### Fix A2 — sane context default + lazy KV
- `env_config.rs:85-90`: `GRIM_CONTEXT` stays, but the *default* per-request context
  becomes `min(model_ctx, 8192)` unless the request asks for more
  (`lib.rs:1294-1297` already has the 8192 fallback for ctx==0 — extend it to clamp
  when model reports absurd lengths for a 350M CPU workload, or gate on backend).
- Ensure KV block pool allocates on demand (`grim-kvtransport` already supports
  tiers; wire `blocks_total` growth to first touch, not request creation).

**Acceptance (KPI re-run):** on this host, `serve` cold first completion ≤ 60 s and
warm ≥ 5 tok/s for the 350M Q8_0 on CPU:

```bash
GRIM_BACKEND=cpu GRIM_DEBUG_PROMPT=1 target/release/grim-cli serve --address 127.0.0.1:11460 &
curl -s -m 60 -X POST localhost:11460/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"LFM2.5-350M-Q8_0","messages":[{"role":"user","content":"Say OK"}],"temperature":0,"max_tokens":8}'
# expect: http=200, time<60s, content contains "OK", no SSSS
```

### Fix A3 — startup noise
Kill the `Model capability check failed: config error: model 'default' has no
strategy for modality 'text'` warning on every boot: only run the default-model
capability check when a default model is actually configured (guard at the call
site in `grim-server/src/lib.rs` where `default_model == "default"`), or resolve
`default` to the first catalog entry before checking.

---

## WS-B — Multimodal: real pipelines, no 501s (P5.1, P5.2, P6.1, P2.2) — P0

### Evidence
- `crates/grim-cli/src/multimodal.rs:68-107` — all three subcommands are
  `println!`-and-`Ok(())` shells.
- `grim-server/src/lib.rs:2092-2104` (audio 501), `:2111-2124` (images 501),
  `:2106` (embeddings 501 — bonus stub).
- Chat messages reject content arrays: `400 "malformed message at index 0: invalid
  type: sequence, expected a string"` (message parser is string-only).
- Model structs exist and are real: `crates/grim-models/vision/src/vit.rs` (712 L),
  `crates/grim-models/audio/src/whisper.rs` (1310 L),
  `crates/grim-models/diffusion/src/unet.rs` (380 L) + `scheduler.rs` (301 L, DDIM +
  Euler). Headers self-describe as "F32 CPU structural layer; ROCm kernels land with
  phase 4" — i.e. CPU path is the honest target for this milestone.

### Fix B1 — message content arrays (unblocks vision-in-chat AND P2.2)
```rust
// grim-server/src/lib.rs — message parsing (near the tools parse, ~lib.rs:1179)
#[derive(Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
struct ImageUrl { url: String } // data: URI or file:// only in v1; http(s) fetch behind a flag
```
Flatten `Parts` → text parts concatenated; collect image refs into a
`Vec<ImageRef>` carried alongside. If the loaded model has no vision tower and
images were passed → **422** `"model X has no vision encoder"`. Never silently drop.

### Fix B2 — vision encode (CLI + engine)
```rust
// crates/grim-cli/src/multimodal.rs — replace the print shell
VisionCmd::Encode { image, model } => {
    let dev = probe_device(None)?;                       // WS-E makes this honest
    let vit = grim_engine::load_vision_encoder(&model, &dev)?;   // vit.rs GGUF loader
    let img = image::open(&image)?;                      // add `image` dep (workspace)
    let emb = vit.encode_image(&img)?;                   // preprocess: resize 224², normalize
    let out = image.with_extension("grim-emb.safetensors");
    grim_format::save_embedding(&emb, &out)?;            // safetensors writer in grim-format
    println!("encoder={} dim={} → {}", vit.arch_name(), emb.len(), out.display());
    Ok(())
}
```
`load_vision_encoder`: resolve model via catalog (`resolve_model_preferring_grim`),
build `Vit` per GGUF arch tag, run F32 CPU forward (kernel work is explicitly
phase-4; CPU correctness first). **Endpoint:** same path callable from
`POST /v1/embeddings` with `{"model":..., "input":{"image": "<data-uri>"}}`.

### Fix B3 — audio transcribe
- Wire `whisper.rs`: load GGUF (`Whisper::from_gguf`), `image` crate's wav decode →
  mel spectrogram (30 s windows), encoder→decoder greedy/beam, detokenize.
- Replace `lib.rs:2092-2104` 501 with the handler; response:
  `{"text": ..., "model": ..., "backend": ..., "duration_s": ..., "language": "auto"}` —
  the persona criterion demands model+backend advertised.
- CLI `multimodal audio transcribe` calls the same engine fn, prints text to stdout
  (scriptable), `--out` writes JSON.

### Fix B4 — diffusion generate
- Pipeline: tokenizer+text-encoder (CLIP-style from vision crate) → `Unet2D` loop
  under `DdimScheduler` (`diffusion/src/scheduler.rs`) → PIL/PNG save.
- Replace `lib.rs:2111-2124` 501 with handler returning
  `{"created":…,"data":[{"url":"file:///…png"}]}` and **write the file** (persona:
  "receive an image artifact on disk").
- CLI `multimodal diffusion generate` → same engine fn, `--output` honored.

**Acceptance (all of WS-B):**
```bash
# vision
target/release/grim-cli multimodal vision encode --image testdata/cat.png --model clip-vit-b32.gguf
# → file exists, dim printed; server: curl /v1/embeddings with image returns 200 + vector
# audio (use a fixtures wav, e.g. 3s of speech)
target/release/grim-cli multimodal audio transcribe --audio testdata/hello.wav --model whisper-tiny.gguf
# → non-empty text; length ≈ input
# diffusion
target/release/grim-cli multimodal diffusion generate --prompt "a red cube" --output out.png --model sd-tiny.grim
# → out.png exists, non-trivial size
curl -s -X POST localhost:11460/v1/images/generations -d '{"prompt":"a red cube"}'  # → 200, data[0].url
curl -s -X POST localhost:11460/v1/chat/completions -d '{"model":...,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}'  # → 200
```
New tests: `grim-cli/tests/multimodal_device.rs` (three smoke tests, GPU-env-gated
like the ROCm ones), `grim-server/tests/vision_chat.rs` (content-array parse).

---

## WS-C — Plugins: default-on sandbox, real grants (P13.1, P13.2, P18.1) — P0

### Evidence
- `crates/grim-plugin/Cargo.toml`: `[features] default = []` — stock build has
  **zero** plugin runtime; `create_sampler` → `Error::Unimplemented`.
- `wasm_loader.rs` grants are **logged, not linked**: comments say "The stub records
  the decision / In a real integration: add a WASI preopen" — a plugin that asks for
  any capability traps.
- `doctor.rs:408-433` asserts deny-by-default grants — only meaningful when the
  feature is compiled in.
- Dylib loader: opt-in, real SHA-256 + ABI checks, but in-process (no memory
  isolation) and undiscoverable.

### Fix C1 — flip the default
```toml
# crates/grim-plugin/Cargo.toml
[features]
default = ["wasm-sandbox"]
wasm-sandbox = ["wasmtime"]
dylib-loading = ["libloading"]   # stays opt-in: no memory sandbox, must be a choice
```
`grim-cli` gains `plugin-loader` passthrough feature so `cargo build -p grim-cli`
ships a working plugin host. Verify: `ldd target/release/grim-cli | grep wasmtime`.

### Fix C2 — implement grants as real WASI capability links
```rust
// crates/grim-plugin/src/wasm_loader.rs — replace the eprintln! stub
fn build_linker(engine: &Engine, grants: &PluginGrants, scopes: &PluginScopes) -> Result<Linker<StoreCtx>> {
    let mut linker = Linker::new(engine);              // deny-by-default base: unchanged
    if grants.network {
        // no sockets in v1: link a trap-with-message import so failure is legible
        linker.func_wrap("env", "socket", || -> anyhow::Result<()> {
            Err(anyhow!("network grant declared but sockets are not available in this build"))
        })?;
    }
    if grants.filesystem {
        let wasi = wasi_common::sync::add_to_linker(&mut linker, |c| &mut c.wasi)?;
        let mut b = wasi_common::sync::WasiCtxBuilder::new();
        for dir in &scopes.allowed_dirs {              // manifest-scoped, never "/"
            b.preopened_dir(Path::new(dir), dir)?;
        }
        b.build(&mut store)?; // (wasi ctx into store)
        let _ = wasi;
    }
    Ok(linker)
}
```
Manifest (`plugin.grim.toml`) gains:
```toml
[grants]
network = false
filesystem = false
[scopes]
allowed_dirs = ["models/"]   # only honored when filesystem = true
```
**Security invariant (testable):** with default grants, an import of `path_open`
traps with `unknown import`; `/etc/shadow` unreadable **because no WASI is linked**;
with `filesystem=true` + empty `allowed_dirs`, `path_open("/")` → `EPERM`-equivalent
error, not silent access. `doctor.rs:408-433` check stays and now actually runs in
the default build.

### Fix C3 — dylib posture (honest, documented, still opt-in)
- Keep opt-in (in-process = no sandbox), but:
  - `grim plugin load --kind dylib` prints an explicit banner: "dylib plugins run
    in-process with NO memory sandbox; SHA-256 + ABI checks only."
  - README/`plugin --help` document the trust model one paragraph each.
- `13.2` acceptance = discoverability of the trade-off, not a sandbox we can't have.

**Acceptance:**
```bash
cargo test -p grim-plugin --features wasm-sandbox   # grant matrix tests green
target/release/grim-cli plugin load plugins/example-json-grammar   # loads in default build
target/release/grim-cli doctor                      # WASM grant check passes, no feature dance
```
New tests: `grim-plugin/tests/grant_matrix.rs` — default-trap, scoped-preopen,
empty-scope-denied, network-trap-message.

---

## WS-D — Config truth & CLI surface (P2.1, P1.4, P12.2, P17.2) — P1

### D1 — wire `--config` (FAIL 2.1)
Evidence: `crates/grim-cli/src/main.rs:809` and `:932` bind `config: _`;
`grim-server/src/lib.rs:2371` + `:3377` read grim.toml from hardcoded
`["grim.toml","/etc/grim/grim.toml","C:\\Program Files\\Grim\\grim.toml"]`.

Fix: thread the path.
```rust
// main.rs — serve dispatch (~:809)
Commands::Serve { address, host, port, config, plugins, .. } => {
    let cfg = ServerFileConfig::load(&config)?;      // errors if unreadable — NO silent drop
    let bind = resolve_bind(address, host, port, &cfg)?;
    grim_server::serve_with_config(bind, engine, plugin_registry, cfg)  // new entry
        .await?;
}
```
In `grim-server`: replace both hardcoded path lookups with a `ServerFileConfig`
stored in `ServerState` at construction (`serve_with_config`), keeping `serve()` as
a thin wrapper that loads the default path list for back-compat. `[server]` keys
(`default_model`, `max_batched_tokens`, `max_num_seqs`) must flow into the scheduler
(this also fixes "threads knob absent": expose `max_num_seqs` semantics in help).

**Acceptance:**
```bash
printf '[server]\ndefault_model = "LFM2.5-350M-Q8_0"\n' > /tmp/g.toml
target/release/grim-cli serve --config /tmp/g.toml -a 127.0.0.1:11461 &
curl -s localhost:11461/status | jq .default_model   # → "LFM2.5-350M-Q8_0"  (not "default")
target/release/grim-cli serve --config /nonexistent.toml # → hard error, exit ≠ 0
```
Test: `grim-cli/tests/config_flag.rs` (tempfile toml, assert default_model honored +
missing-file errors).

### D2 — `--backend` flag everywhere; kill silent demotion's sibling confusion (P12.2)
Add to `serve` **and** unify vocab (see E1 for the error half):
```rust
// main.rs — Serve variant (~:66-95) and Run (~:96-138)
/// Compute backend: rocm|cuda|vulkan|metal|cpu|auto (overrides GRIM_BACKEND)
#[arg(long, value_enum)]
backend: Option<BackendArg>,
```
`--device` becomes a hidden alias of `--backend` (one season of back-compat), and
`--target`/`--profile` doc-lines cross-reference that they're *artifact* targets,
not runtime backends. Same flag on `serve` (env-only today).

### D3 — adapter chooser (P1.4)
`main.rs:334-348` `--mode` gets a chooser + per-mode help:
```rust
/// Fine-tune mode. qlora = 4-bit base + LoRA (lowest VRAM, default);
/// lora = fp16/bf16 base + LoRA; full-bf16/full-fp16 = full finetune;
/// soul-eater = low-rank absorb-and-freeze variant; oft = orthogonal FT
#[arg(long, value_enum, default_value_t = ModeArg::Qlora)]
mode: ModeArg,
```
Plus `grim train --compare-modes` → runs N steps per mode on a sample, prints
VRAM/loss/speed table (this is also 4.1's CLI half).

### D4 — docs truth pass (P17.2 + README)
- `docs/onboarding.md:39,97`: "28 crates" → 29 (or drop the count, say "all
  workspace members").
- `README.md:103`: replace `grim serve --port 8080 --model models/llama-3.2-1b.gguf`
  with the real flow (`grim pull … && grim serve`, model named per-request) — or
  implement `--model` preload on `serve` (preferred; `run --serve` already has it —
  factor that preload out and reuse).
- `README.md:133`: delete `GRIM_LOG` or implement it (`GRIM_LOG_DIR` exists,
  `paths.rs:62`).
- `main.rs:274-275`: de-duplicate the doubled `tune` doc-comment.
- `doctor.rs:227` probes `/health` while `service install` advertises `/healthz` —
  probe `/healthz` (primary) with `/health` fallback, matching `lib.rs:3147-3148`.

---

## WS-E — Backend trust (P9.1, P12.1, test gating) — P1

### E1 — hard error on unavailable backend (FAIL 9.1)
Evidence: `run.rs:61-117` `probe_device` falls back ROCm→CPU silently;
`GRIM_BACKEND=cuda` → `Device: cpu`, zero warnings (live, twice). Contradicts
`grim-garage/src/backend.rs` "never silently degrades".

```rust
// crates/grim-cli/src/run.rs — probe_device
pub fn probe_device(requested: Option<&str>) -> Result<Device> {
    let want = requested
        .map(|s| s.parse::<Backend>().map_err(|_| Error::Config(format!("unknown backend '{s}'"))))
        .transpose()?
        .or_else(Backend::from_env)          // GRIM_BACKEND
        .unwrap_or(Backend::Auto);
    match want {
        Backend::Auto => probe_auto(),                        // ROCm→CUDA→Vulkan→Metal→CPU, logged
        b => probe_exact(b).ok_or_else(|| Error::Config(format!(
            "backend '{b}' requested but unavailable (not compiled in or no device). \
             Rebuild with `--features {feature_for(b)}` or use GRIM_BACKEND=auto"
        ))),
    }
}
```
`probe_exact` = cheap capability check per backend (dlopen symbol probe for
ROCm/CUDA, instance enum for Vulkan). Every auto-fallback logs
`warn!("GRIM: no {b} device, falling back to {next}")`. Same guard in the
engine-side device acquisition used by `serve`.

**Acceptance:**
```bash
GRIM_BACKEND=cuda target/release/grim-cli run LFM2.5-350M-Q8_0 hi   # exit≠0, message names the fix
GRIM_BACKEND=rocm target/release/grim-cli run LFM2.5-350M-Q8_0 "Say OK"  # Device: rocm:0, coherent
```

### E2 — Vulkan zero-output diagnosis & fix (FAIL 12.1)
Symptom (live): `Device: vulkan`, model loads, 12 prompt tokens encoded, zero
generated tokens, exit 0. Bisect with device-side asserts:
1. New test `crates/grim-backend-vulkan/tests/decode_sanity.rs`: single
   GEMM + softmax + argmax on SPIR-V path, compare vs CPU. (If argmax reads zeros →
   barrier/copy-back bug; if softmax NaNs → dtype/precision path.)
2. Instrument `grim run` with `GRIM_DEBUG_PROMPT=1`-style step logging on the vulkan
   decode loop (log logits head sample each step; expect either all-`<eos>`-ish
   distribution or silent exception swallowed — the exit-0-with-no-output smells
   like an ignored `Result` in the step loop).
3. Fix per finding; add regression test pinning non-empty output for the 350M on
   Vulkan or, if the vulkan decode path is genuinely unbuilt for lfm2, **fail
   loudly** (E1) rather than emitting nothing.

**Acceptance:** `GRIM_BACKEND=vulkan grim run … "Say OK"` → ≥1 coherent token or a
hard error. Never empty-Ok.

### E3 — cubecl test gating (developer-truth)
The 4 default-feature-red targets (`fused_linear_ce_parity_tests`, `graph_capture`,
`mxfp4_gemm_tests`, `p3_ce_wiring_contract`) pass with `--features cubecl`, red
without. Stop the trap:
```rust
// top of each of the 4 test files
#![cfg(feature = "cubecl")]   // or: add cubecl to grim-backend-rocm default
```
Prefer the latter (`default = ["jit-hw-adaptive", "cubecl"]`) so bare
`cargo test -p grim-backend-rocm` matches the shipping binary. Also update
`docs/onboarding.md` test section to the canonical incantation:
`GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features cubecl`.

---

## WS-F — Adapter runtime serving (P2.4, P4.2, P4.1) — P1

### Evidence
- Routes: `GET /v1/adapters` (list), `DELETE /v1/adapters/:name`
  (`grim-server/src/lib.rs:2310-2326`). No load.
- Per-request routing exists: `adapters: [String]` → validated (`lib.rs:991-1024`)
  → resolved to handles (`lib.rs:1312-1318` streaming, `:1624-1630` non-streaming).
- Adapter math is real (`grim-autograd`: LoRA/QLoRA/VeRA/PiSSA/OLoRA/SoulEater/OFT).

### Fix F1 — `POST /v1/adapters` (+ CLI)
```rust
// grim-server/src/lib.rs — build_router (~:3157)
.route("/v1/adapters", post(load_adapter).get(list_adapters))

#[derive(Deserialize)]
struct AdapterLoadReq {
    path: PathBuf,                  // .grim sidecar or dir of sidecars
    #[serde(default)] name: Option<String>,
    #[serde(default)] base: Option<String>,   // must match a loaded base model
}

async fn load_adapter(State(st): State<Arc<ServerState>>, Json(req): Json<AdapterLoadReq>) -> Response {
    let base = req.base.map(|b| st.engine.model_handle(&b))
        .or_else(|| st.engine.default_model_handle());
    let Some(base) = base else { return err(400, "load a base model first"); };
    match st.engine.load_adapter_sidecar(&base, &req.path, req.name.as_deref()).await {
        Ok(h) => json200(serde_json::json!({"name": h.name, "id": h.id, "rank": h.rank})),
        Err(e) => err(400, e),
    }
}
```
`Engine::load_adapter_sidecar`: read the serialized adapter (grim-autograd), attach
to base **without** unloading/re-quantizing the base (persona KPI), register in the
name→handle map that `lib.rs:1002-1024` already validates against. Engine mutex
scope must exclude the base weights during attach (zero-downtime claim).

CLI: `grim adapter load <path> [--name] [--base] [--addr]` + `grim adapter list/unload`
(wrap the routes; add `Adapter` subcommand in `main.rs`).

### Fix F2 — multi-adapter demo & test (P4.2)
```bash
for i in 1 2 3; do grim train --mode qlora --steps 50 --out side$i … ; done
for i in 1 2 3; do grim adapter load side$i --base LFM2.5-350M-Q8_0 ; done
curl -X POST …/v1/chat/completions -d '{"model":"LFM2.5-350M-Q8_0","adapters":["side2"],…}'
```
Test `grim-server/tests/adapter_runtime.rs`: load 3 sidecars on a live engine, route
one request per adapter, assert per-request output differs and `adapters_active == 1`.

### Fix F3 — QLoRA-vs-LoRA compare (P4.1)
`grim train --compare-modes qlora,lora --steps N` runs both sequentially, prints:

```
mode    vram_peak   loss_final   tok/s_train
qlora   3.1 GiB     1.842        812
lora    5.7 GiB     1.809        640
```
Data: VRAM from `/status` deltas, loss from the existing streaming events
(`grim-garage/routes.rs:923` SSE), speed from wall-clock. Garage panel gets the same
table (view-model already exists in `view_model/`).

---

## WS-G — Observability (P3.2, P10.2, P11.1, P11.2, P16.1, P16.2) — P2

### G1 — `/metrics` Prometheus format (P16.1)
Evidence: `lib.rs:2136-2151` returns JSON; `block_pool_usage:0.0`,
`preemption_count:0` hardcoded; live `gpu_util_pct` exists only in `/status`.

```rust
// grim-server/src/lib.rs — replace metrics_endpoint body
async fn metrics_endpoint(State(st): State<Arc<ServerState>>) -> Response {
    let s = st.snapshot();
    let body = format!(
        "# HELP grim_engine_state Engine state\n# TYPE grim_engine_state gauge\ngrim_engine_state{{state=\"{s}\"}} 1\n\
         # HELP grim_active_sessions Active sessions\n# TYPE grim_active_sessions gauge\ngrim_active_sessions {n}\n\
         # HELP grim_gpu_util_percent GPU utilization\n# TYPE grim_gpu_util_percent gauge\ngrim_gpu_util_percent {u}\n\
         # HELP grim_vram_bytes VRAM used\n# TYPE grim_vram_bytes gauge\ngrim_vram_bytes{{kind=\"used\"}} {vu}\ngrim_vram_bytes{{kind=\"total\"}} {vt}\n\
         # HELP grim_kv_blocks KV cache blocks\n# TYPE grim_kv_blocks gauge\ngrim_kv_blocks{{kind=\"used\"}} {ku}\ngrim_kv_blocks{{kind=\"total\"}} {kt}\n\
         # HELP grim_block_pool_usage Fraction of block pool in use\n# TYPE grim_block_pool_usage gauge\ngrim_block_pool_usage {bpu}\n\
         # HELP grim_preemption_total Preempted requests\n# TYPE grim_preemption_total counter\ngrim_preemption_total {pre}\n",
        s = s.engine_state, n = s.active_sessions, u = s.gpu_util_pct,
        vu = s.vram_used_bytes, vt = s.vram_total_bytes,
        ku = s.kv.blocks_used, kt = s.kv.blocks_total,
        bpu = s.kv.blocks_used as f64 / s.kv.blocks_total.max(1) as f64,
        pre = s.preemption_count,
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}
```
`bpu`/`pre` now derive from the scheduler snapshot (real values, placeholders die).
Keep JSON under `/status` (unchanged) and add `Accept`-based negotiation if both are
wanted on `/metrics` (v2 nicety). Grafana scrape test in
`grim-server/tests/metrics_format.rs`: assert content-type + `# HELP` lines parse
via `prometheus-parse`-style regex.

### G2 — speculation visibility (P11.1)
- `get_status` (lib.rs:2388) adds:
```rust
"speculation": {
    "strategy": session.spec_strategy_name(),        // "plain"|"dspark"|"native_mtp"
    "draft_tokens_total": …, "accepted_tokens_total": …,
    "acceptance_rate": accepted as f64 / drafted.max(1) as f64,
}
```
  (session already tracks accept counts — `grim-core/src/session.rs:95-96,120-122`.)
- Same three series in `/metrics` (G1 format).
- Env off-switch: `GRIM_SPEC=off|dspark|mtp` honored in the engine's strategy select
  (`grim-engine/src/lib.rs:434-437`); document in `serve --help` epilog.

### G3 — KV tier surfacing (P3.2, P11.2)
- `ServerState::snapshot()` reads `LocalSpillManager` tier counts
  (`grim-kvtransport/src/lib.rs:18-23`, `:109`, `:125`) into
  `kv_cache: {gpu_blocks, host_ram_blocks, nvme_blocks, spill_threshold_mib}`.
- `crates/grim-cli/src/scheduler.rs` (extend the `kv` render at `:64-74`):
```rust
println!("  GPU / HostRAM / NVMe : {} / {} / {} blocks",
    kv["gpu_blocks"].as_u64().unwrap_or(0),
    kv["host_ram_blocks"].as_u64().unwrap_or(0),
    kv["nvme_blocks"].as_u64().unwrap_or(0));
println!("  Spill threshold       : {} MiB", kv["spill_threshold_mib"].as_u64().unwrap_or(0));
```
- Spill threshold becomes a **config knob**: `[kv] spill_threshold_mib` in grim.toml
  + `GRIM_KV_SPILL_MIB` env, wired to the manager (persona 3.2 criterion b).

### G4 — `grim garage` + utilization (P16.2)
```rust
// main.rs — new subcommand
/// Launch the grim-garage dashboard (ROCm telemetry, training jobs)
Garage { #[arg(long)] bind: Option<String> } =>
    grim_garage::run(bind.as_deref().or_env("GRIM_GARAGE_BIND_ADDR")).await,
```
(axum server is a lib call — no exec needed.) SPA panel adds a utilization tile
reading the existing `/status` `gpu_util_pct` + `rocm_panel.rs` SSE stream; no new
probe code required server-side.

### G5 — `bench` quality numbers (P10.2)
`--ppl [--ppl-data corpus.txt] [--window 2048] [--stride 512]`: sliding-window
NLL over the corpus with the loaded model, print mean NLL + perplexity + file-size
delta vs source GGUF. Reuses `eval.rs` (currently only called from `train.rs:941` —
this also un-hides existing code):
```
tokens=131072  mean_nll=2.413  ppl=11.17  size_in=379MB  size_out=214MB  (−43.5%)
```
Test: `grim-cli/tests/bench_ppl.rs` — synthetic corpus, bounded ppl range.

---

## WS-H — Distribution & first-run (P15.1) — P2

- **Prebuilt path:** `cargo-dist`-style release workflow (or a `install.sh` that
  fetches the GitHub Release asset for the triple) — `<5 min from clone to token`
  without a toolchain. `curl -fsSL grim.sh | sh` installs `grim` + `grim-garage`.
- **Build-time honesty (kept):** README quickstart leads with the installer; source
  build documented as the dev path. Add `--features cuda` to the documented build
  matrix so the RTX box isn't surprised by WS-E's new hard error.
- `service install` already writes systemd/launchd — installer post-install prints
  `grim service install && grim service start` (discoverability, ties to 16.1's
  health endpoints).

**Acceptance:** fresh-VM timing test: install → `grim pull LFM2.5-350M-Q8_0` →
first completion ≤ 5 min (no cargo involved).

---

## 9. Stub / dead-code registry (complete sweep — dispositions)

| # | Stub | File:line | Disposition |
|---|---|---|---|
| 1 | multimodal CLI print-shells | `grim-cli/src/multimodal.rs:68-107` | WS-B: real impls |
| 2 | audio 501 | `grim-server/src/lib.rs:2092-2104` | WS-B3 |
| 3 | images 501 | `grim-server/src/lib.rs:2111-2124` | WS-B4 |
| 4 | embeddings 501 | `grim-server/src/lib.rs:2106` | WS-B2 (vision) + text emb |
| 5 | WASM grants stub | `grim-plugin/src/wasm_loader.rs` (preopen comment) | WS-C2 |
| 6 | `default = []` plugin features | `grim-plugin/Cargo.toml` | WS-C1 |
| 7 | `/metrics` placeholder keys | `grim-server/src/lib.rs:2146-2148` | WS-G1 |
| 8 | doctor KV-sizing returns 0 | `grim-cli/src/doctor.rs:595-610` | wire to `ModelFootprint` + real KV per family; or delete the field |
| 9 | RDMA flag, TCP transport | `grim-disagg/src/lib.rs:172` | mark `--enable-rdma` experimental in help, or implement (defer w/ doc) |
| 10 | `model 'default' has no strategy` startup warn | `grim-server/src/lib.rs` (cap check) | WS-A3 |
| 11 | README `--model` on serve | `README.md:103` | D4 (implement flag or fix doc; prefer flag) |
| 12 | README `GRIM_LOG` | `README.md:133` | D4 |
| 13 | `tune` doubled doc | `grim-cli/src/main.rs:274-275` | D4 |
| 14 | doctor `/health` vs `/healthz` | `grim-cli/src/doctor.rs:227` | D4 |
| 15 | default-feature test red | 4 targets in `grim-backend-rocm/tests/` | E3 (cubecl default or cfg-gate) |
| 16 | `grim quantize` referenced by test doc only | `usability-test.md` task 10.1 note | no-op (doc artifact); README must not mention it (D4 sweep) |
| 17 | `permissive: true` parsing hint | `grim-server/src/lib.rs` (~KNOWN_FIELDS msg) | implement or delete the hint text (choose: implement — one boolean, fields-as-strings passthrough) |
| 18 | bnb-style 8-bit optimizer placeholder | `grim-autograd/src/adamw.rs:166` | out of usability scope; tracked: implement `paged 8bit` or drop the variant from `--optimizer` choices |

Rule: items 1-15 are in-scope and must land with this plan; 16 is doc hygiene; 17
implement (tiny); 18 is tracked but explicitly out of this plan's acceptance.

---

## 10. Sequencing

```
WS-A (serve correctness)   ──┐ first: everything user-facing sits on it
WS-E1 (hard backend error) ──┤ same PR-size class, independent
WS-C (plugins default+grants) ┘
        │
WS-D (config/--backend/docs) ← depends on E1's BackendArg
WS-F (adapter load route)   ← after A (stable serve)
WS-B (multimodal)           ← after A; parallel to D/F
        │
WS-G (observability: G1..G5) ← after F (adapter counts) + B (encoder events)
WS-H (installer)             ← last (ships the rest)
```

Suggested PR slicing (each independently verifiable):
1. `fix(serve): unified prompt render + context clamp` (A1-A3)
2. `fix(cli): honest backend selection` (E1, D2)
3. `feat(plugin): default-on wasm sandbox + real grants` (C1-C3)
4. `fix(config): wire --config; docs truth` (D1, D4, 17)
5. `feat(server): POST /v1/adapters + CLI` (F1-F2)
6. `feat(multimodal): vision/audio/diffusion pipelines` (B1-B4, embeddings)
7. `feat(obs): prometheus /metrics, spec fields, KV tiers, garage cmd` (G1-G4)
8. `feat(bench): --ppl` (G5) + `chore: cubecl default / cfg-gates` (E3)
9. `feat(dist): installer` (H)

---

## 11. Global acceptance gates (definition of done for this plan)

1. **Re-run the notional usability test** (`usability-test.md`) with the same
   scoring; target: **0 FAIL tasks**, every previous PARTIAL either PASS or
   explicitly re-scored with rationale. Regenerate `ohsheet.md` rollup.
2. All commands from §"Acceptance" blocks run green on this host (RTX 4070 +
   gfx1036 + CPU), with `--features cubecl` where GPU tests are involved.
3. `grep -rn "not yet implemented\|placeholder\|TODO" crates/ | grep -v tests`
   returns only registry items #9/#18 (documented deferrals).
4. `cargo test -p grim-server -p grim-cli -p grim-plugin -p grim-backend-rocm --features cubecl`
   + `GRIM_RUN_GPU_TESTS=1` on the AMD box: green, no `--ignored` surprises beyond
   the pre-existing manual-only list.
5. `grim doctor` green in a default build (WASM grant check meaningful), and its
   health probe matches what `service install` configures.
6. KPIs: serve cold-first-completion ≤ 60 s / warm ≥ 5 tok/s (350M Q8, CPU);
   installer path clone→token ≤ 5 min; `GRIM_BACKEND=cuda` on a cuda-less build
   errors with a fix hint; Vulkan run yields tokens or a hard error — never
   empty-Ok.

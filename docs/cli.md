# CLI Reference — Grim Subcommands and Flags

Generated from clap derive attributes in `crates/grim-cli/src/main.rs`.

## `grim serve`

Starts the inference HTTP server on Ollama-compatible endpoints (default `127.0.0.1:11434`). This is the subcommand used by the systemd/launchd service unit.

`serve` is **model-agnostic**: there is no `--model`, `--backend`, `--temp`,
`--top-p`, `--threads`, `--batch-size`, or `--kv-cache` flag. Those are not
oversights — they are per-request or config concerns:

- **Model**: named per request in the `"model"` field of `/v1/chat/completions`,
  resolved against the local catalog (`grim pull` / `grim list`), or loaded
  explicitly via `POST /v1/models/load`.
- **Backend**: `GRIM_BACKEND` env var or `grim.toml`.
- **Sampling** (`temperature`, `top-p`, `top_k`): per-request JSON body fields,
  or `grim run`'s `--temperature` / `--top-p` / `--top_k` for one-shot CLI use.
- **Threads / batch size / KV cache**: `grim.toml` `[server]` and `[scheduler]`
  keys plus `GRIM_CONTEXT` / `GRIM_MEM_BUDGET_MIB` (see `configuration.md`).

| Flag | Short | Default | Required |
|---|---|---|---|
| `--address` | `-a` | `""` (falls back to `--host`/`--port`, then `GRIM_HOST`/`GRIM_PORT`) | No |
| `--host` | | `127.0.0.1` | No |
| `--port` | `-p` | `11434` | No |
| `--config` | `-c` | `grim.toml` | No |
| `--plugins` | | `plugins` | No |
| `--disagg-role` | | `colocated` | No |
| `--prefill-addr` | | `""` | No (decode role only) |
| `--decode-addr` | | `""` | No (prefill role only) |

Typical flow: `grim pull <model>` → `grim serve` → `POST /v1/chat/completions`
with `{"model": "<name-from-grim-list>", ...}`.

## `grim run`

One-shot inference or HTTP serving for a model.

| Flag | Short | Default | Required |
|---|---|---|---|
| `model` | - | `default` | No (uses `default` model) |
| `[PROMPT]` | - | None | No (interactive mode if absent) |
| `--serve` | - | false | No |
| `--address` | `-a` | `127.0.0.1:11434` | No (only used with `--serve`) |
| `--config` | `-c` | `grim.toml` | No |
| `--plugins` | - | `plugins` | No |
| `--rocml-profile` | - | None | No |
| `--temperature` | - | `0.7` | No |
| `--top-p` | - | `0.9` | No |
| `--top_k` | - | `40` | No |
| `--max-tokens` | - | `256` | No |
| `--seed` | - | `0` (random) | No |
| `--repeat-penalty` | - | `1.1` | No |

## `grim bench`

Run benchmark/smoke test.

| Flag | Short | Default | Required |
|---|---|---|---|
| `--tokens` | - | `128` | No |
| `--concurrency` | - | `1` | No |
| `--model` | `-m` | None (smoke test model) | No |

## `grim dl` / `grim pull`

Download a model from Hugging Face or Ollama registry.

| Flag | Short | Default | Required |
|---|---|---|---|
| `model` | - | (required) | Yes |
| `--output` | `-o` | Local cache | No |
| `--rocml-profile` | - | None | No |

## `grim stop`

Stop a running model (unload from memory).

| Argument | Required |
|---|---|
| `model` | Yes |

## `grim rm`

Delete a model from local cache.

| Argument | Required |
|---|---|
| `model` | Yes |

## `grim check`

Check the local model cache and report completed and partial downloads.

No arguments.

## `grim list` / `grim ps`

Show loaded models, memory usage, and execution backend.

No arguments.

## `grim show`

Show available models organized by format (GRIM, GGUF, others).

| Flag | Short | Default | Required |
|---|---|---|---|
| `--verbose` | `-v` | false | No |

## `grim use`

Set a model (local or cloud-routed) as the default model point for a client context.

| Arguments | Required |
|---|---|
| `context` | Yes |
| `model` | Yes |

## `grim login`

Log in to a registry or cloud provider.

| Argument | Short | Default | Required |
|---|---|---|---|
| `provider` | - | (required) | Yes |
| `--token` | `-t` | (prompt if absent) | No |

## `grim quantize`

Stub command. It performs no quantization; it prints pointers to the commands
that do: `grim convert -i <in.gguf> -o <out.grim> --target-bpw 4.0` for the
one-shot path, or `grim oxidizer convert` for the full calibrate → search →
write pipeline. There is no `--dtype` flag and no `grim oxidize` subcommand.

No arguments.

## `grim convert`

Convert a model file to ROCm-optimized .grim format using Oxidizer.

| Flag | Short | Default | Required |
|---|---|---|---|
| `--input` | `-i` | (required) | Yes |
| `--output` | `-o` | (required) | Yes |
| `--target` | `-t` | `auto` | No |
| `--target_bpw` | - | `4.0` | No |
| `--generations` | - | `50` | No |
| `--dataset` | - | None | No |

## `grim train`

Train / fine-tune LoRA adapters on a dataset (SFT QLoRA).

| Flag | Short | Default | Required |
|---|---|---|---|
| `--model` | `-m` | (required) | Yes |
| `--dataset` | `-d` | (required) | Yes |
| `--output` | `-o` | `adapter.grim.train` | No |
| `--epochs` | - | `3` | No |
| `--lr` | - | `2e-4` | No |
| `--rank` | - | `16` | No |
| `--alpha` | - | `32.0` | No |
| `--batch_size` | - | `2048` | No |
| `--gradient_accumulation_steps` | - | `1` | No |
| `--warmup_steps` | - | `0` | No |
| `--logging_steps` | - | `0` | No |
| `--max_grad_norm` | - | `1.0` | No |
| `--early_stopping_patience` | - | `0` | No |
| `--num_gpus` | - | `1` | No |
| `--device` | - | `cpu` | No |
| `--mode` | - | `qlora` | No |
| `--echo_mode` | - | false | No |
| `--optimizer` | - | `adamw` | No |
| `--scheduler` | - | `cosine-warmup` | No |
| `--use_pissa` | - | false | No |
| `--use_olora` | - | false | No |
| `--olora_lambda` | - | `1.0` | No |

## `grim cp`

Copy a model to a new name in the local cache.

| Arguments | Required |
|---|---|
| `src` | Yes |
| `dst` | Yes |

## `grim start`

Start a client integration.

| Flag | Argument | Required |
|---|---|---|
| `client` | `hermes`, `openclaw`, `claw`, `codex`, `antigravity`, `zcode` | Yes |
| `model` | - | No |
| `args` | - | No |

## `grim reap`

Launch an external app with a grim-tracked model baked in.

| Flag | Argument | Required |
|---|---|---|
| `client` | `hermes`, `openclaw`, `claw`, `codex`, `antigravity`, `zcode` | Yes |
| `--model` | - | No (defaults to `"default"`) |
| `args` | - | No |

## `grim spec`

Speculative decoding commands.

### `grim spec train`

Distill / train a draft model.

| Flag | Short | Default | Required |
|---|---|---|---|
| `--target` | `-t` | (required) | Yes |
| `--output` | `-o` | (required) | Yes |
| `--dataset` | `-d` | (required) | Yes |

## `grim plugin`

Plugin management.

| Subcommand | Arguments/Flags |
|---|---|
| `plugin list` | None |
| `plugin load` | `--path` / `-p` (default: `plugins`) |

## `grim service`

Platform-native background daemon management.

| Subcommand | Flags | Default |
|---|---|---|
| `service install` | `--name` (default: `grim`), `--config` (default: `grim.toml`) |
| `service uninstall` | `--name` (default: `grim`), `--purge` |
| `service start` | `--name` (default: `grim`) |
| `service stop` | `--name` (default: `grim`) |
| `service status` | `--name` (default: `grim`) |
| `service run` | `--config` (default: `grim.toml`) |

## `grim oxidizer`

ROCm-optimized GGUF conversion tool — calibrate, search, and convert.

### `grim oxidizer info`

Display grim metadata from a GGUF/.grim file.

| Argument | Required |
|---|---|
| `path` | Yes |

### `grim oxidizer calibrate`

Run importance-matrix calibration and cache results.

| Flag | Short | Default | Required |
|---|---|---|---|
| `model` | - | (required) | Yes |
| `--output` | `-o` | (required) | Yes |
| `--dataset` | - | None | No |

### `grim oxidizer search`

Run EvoPress evolutionary search on pre-computed importance scores.

| Flag | Argument | Default | Required |
|---|---|---|---|
| `scores_path` | - | (required) | Yes |
| `tensor_sizes` | - | (required) | Yes |
| `--target_bpw` | - | `4.0` | No |
| `--generations` | - | `50` | No |

### `grim oxidizer convert`

Full convert pipeline: calibrate → search → write .grim.

| Flag | Short | Default | Required |
|---|---|---|---|
| `model` | - | (required) | Yes |
| `--output` | `-o` | (required) | Yes |
| `--target_bpw` | - | `4.0` | No |
| `--generations` | - | `50` | No |
| `--profile` | - | None | No |
| `--dataset` | - | None | No |

### `grim oxidizer raven`

Raven FP8/MXFP4 repack pipeline: rewrite model tensors into FP8 format.

| Flag | Short | Default | Required |
|---|---|---|---|
| `model` | - | (required) | Yes |
| `--output` | `-o` | (required) | Yes |
| `--target_bpw` | - | `8.0` | No (optional) |
| `--dataset` | - | None | No |

### `grim oxidizer prepare`

Prepare a training-capable `.grim` artifact from a base checkpoint.

| Flag | Short | Default | Required |
|---|---|---|---|
| `input` | - | (required) | Yes |
| `output` | - | (required) | Yes |
| `--train` | - | `true` | No |
| `--format` | - | `bf16` | No |
| `--profile` | - | None | No |
| `--dataset` | - | None | No |

### `grim oxidizer fuse`

Analyze a checkpoint and bake ROCm fusion hints into the output artifact.

| Flag | Short | Default | Required |
|---|---|---|---|
| `input` | - | (required) | Yes |
| `output` | - | (required) | Yes |
| `--profile` | - | None | No |
| `--rocm` | - | `true` | No |

## `grim doctor`

Re-verify every claim Grim makes about itself (§13.5). Checks: unit on disk, OS service visibility, HTTP health, GPU backend, WASM grant enforcement, and ExecStart consistency.

| Flag | Short | Default | Required |
|---|---|---|---|
| `--addr` | - | `127.0.0.1:11434` | No |
| `--service_name` | - | `grim` | No |
| `--exec_path` | - | `/usr/local/bin/grim` | No |
| `--config_path` | - | `/etc/grim/grim.toml` | No |

## `grim accept`

Validate and install a model architecture plugin into system plugin directory.

| Argument | Required |
|---|---|
| `plugin_path` | Yes |

## `grim compat`

Generate a model architecture compatibility plugin from a HuggingFace config.json.

| Flag | Short | Default | Required |
|---|---|---|---|
| `--config_path` | `-c` | (required) | Yes |
| `--output` | `-o` | stdout | No |

## `grim verify`

Verify a .grim file: structure, compression, payload readability, and QLoRA adapter presence in backup2 slots.

| Flag | Short | Default | Required |
|---|---|---|---|
| `path` | - | (required) | Yes |
| `--verbose` | `-v` | false | No |

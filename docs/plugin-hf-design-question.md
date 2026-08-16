# Design question: what does an HF-generated plugin look like to `scan_plugin_directory`?

## Status
**Open** — the deletion of `compat.rs`/`accept.rs` and the addition of
`ArchCompatSpec::from_hf_model_id` are done. What's missing is the bridge from
a generated spec to something the runtime plugin loader can discover and consume.

## The gap

`scan_plugin_directory` (`grim-plugin/src/lib.rs:370`) only recognises a
specific layout:

```
<plugins-dir>/<plugin-name>/plugin.grim.toml   # TOML, [plugin] table
<plugins-dir>/<plugin-name>/<entry>            # .wasm or .so, referenced by `entry`
```

The `plugin.grim.toml` must contain at minimum:

```toml
[plugin]
name = "..."
abi_version = 1
kind = "wasm" | "dylib"
capabilities = ["model"] | ["sampler"] | ...
entry = "relative/path/to/artifact"
```

`ArchCompatSpec` is a different thing entirely. It is JSON with fields like
`hidden_size`, `num_layers`, `tensor_name_mapping`, `vision_spec`, `base_architecture`.
It has **no** `kind`, **no** `entry`, **no** `abi_version`, and no concept of a
loadable binary artifact. It describes model-loading metadata (tensor remapping,
architecture parameters), not a sandboxed plugin binary.

Today, `grim accept foo.grimplugin` copies a JSON file into the plugins dir and
`scan_plugin_directory` silently ignores it — it isn't named `plugin.grim.toml`
and wouldn't parse as one if it were. `compat.rs` + `accept.rs` are a disconnected
island: they produce a spec that nothing runnable consumes.

## The question to answer before building the bridge

**What should `kind` and `entry` be for a plugin generated from a HuggingFace
model repo?**

There are three plausible answers, and they imply very different work:

### Option A: Arch-compat plugins are a new kind of manifest, not loaded by
`scan_plugin_directory` today

An `ArchCompatSpec` is model-loading metadata, not a sampler/processor plugin.
It belongs to `grim-engine`'s model loading path (`model_loader.rs` reads
`.grimplugin` JSON from the plugins dir via `resolve_arch_compat_spec`), not to
`scan_plugin_directory`'s WASM/dylib sampler registry.

Under this option:
- `scan_plugin_directory` is the wrong consumer. The bridge is in `grim-engine`,
  not `grim-plugin`.
- `kind`/`entry` are the wrong framing. What we need is a `.grimplugin` JSON
  file in the plugins dir that `model_loader.rs` already knows how to read.
- The work is: `grim plugin hf:org/repo` fetches the config, produces the
  `.grimplugin` JSON, installs it into `grim_plugins_dir()`, and `model_loader.rs`
  picks it up on the next model load. No WASM/dylib artifact needed — the spec is
  pure metadata consumed by the model loader.

This is the smallest change and matches what `compat.rs`/`accept.rs` already
produce. The reviewer's criticism that those commands "don't connect to anything
runnable" is only true if you expect them to feed `scan_plugin_directory`. If
they're feeding `model_loader.rs`'s `.grimplugin` resolution path, the connection
already exists — the commands were just never wired to *install* into the right
dir with the right name, and `accept.rs` copied into `grim_plugins_dir()` but
with a `.grimplugin` extension that `model_loader.rs` does read (it scans for
`*.grimplugin` and `*.json` — see `model_loader.rs:156`).

**Verdict**: this is the most likely correct answer. The deletion of `compat.rs`
and `accept.rs` is safe because their functionality (generate + install a
`.grimplugin` JSON) can be reproduced by `grim plugin hf:org/repo` that writes
directly to `grim_plugins_dir()` with the right filename, and `model_loader.rs`
already resolves those. No new manifest kind, no new artifact type.

### Option B: Arch-compat plugins are real WASM/dylib plugins

An `ArchCompatSpec` could be compiled into a WASM component that exports a model
factory, or a dylib that implements the `GrimPluginVTable`'s `model_factory` entry.
Under this option:
- `kind = "wasm"` or `"dylib"`, `entry = "arch-compat.wasm"` / `arch-compat.so`.
- The generated artifact is a compiled binary, not just JSON.
- `scan_plugin_directory` consumes it directly.
- This is substantially more work: a compiler/toolchain step from spec → WASM/dylib,
  plus implementing the `model_factory` vtable entry for arch-compat specs.

**Verdict**: this is a larger architectural change and not obviously motivated.
Arch-compat specs are metadata; compiling them into sandboxed binaries adds a
toolchain and a vtable contract that doesn't exist today.

### Option C: Two parallel plugin systems

Keep `scan_plugin_directory` for WASM/dylib sampler/processor plugins, and add
a separate "model plugin" discovery path in `grim-engine` that reads
`.grimplugin` JSON from the same or a different dir. Under this option:
- The HF generator produces `.grimplugin` JSON for the model-loader path.
- `scan_plugin_directory` continues to handle sampler/processor plugins.
- The two systems coexist without conflating metadata with binaries.

**Verdict**: this is essentially Option A with the discovery paths made explicit
rather than implicit. It's the honest framing of what the codebase already does.

## Recommendation

Proceed with **Option A/C**: `grim plugin hf:org/repo` generates an
`ArchCompatSpec`, serialises it to `.grimplugin` JSON, and installs it into
`grim_plugins_dir()` under a name that `model_loader.rs` can resolve
(`{model_type}.grimplugin`). No new `kind`/`entry` artifact — the spec is pure
metadata consumed by the existing model-loader resolution path.

The remaining work after `from_hf_model_id` is therefore:
1. A `grim plugin generate hf:org/repo` CLI command (new arm on `PluginCommands`
   or a new top-level command — see CLI topology note below).
2. That command writes the `.grimplugin` JSON into `grim_plugins_dir()`.
3. Optionally: validation of required fields (model_type, num_layers, hidden_size)
   before writing, since `from_hf_config_json` silently defaults missing fields.

## CLI topology note (corrects the earlier description)

`Accept` and `Compat` are **top-level** `Commands` variants in `main.rs`, not
`PluginCommands` subcommands. `PluginCommands` is a separate enum with `List` and
`Load`. Folding the HF generator into `PluginCommands` is a design choice, not a
restoration of existing topology — it would add a `Generate` arm (or make `List`/`Load`
share space with generation). The alternative is a new top-level command, which
preserves the current separation. Either is fine; the choice should be made
consciously.

## Validation note

`compat.rs`'s validation (`model_type.is_empty()`, `num_layers == 0`,
`hidden_size == 0`) lived outside `ArchCompatSpec::from_hf_config_json`, which
silently defaults missing fields (hidden_size → 4096, num_layers → 32, etc.).
Any replacement command must reimplement those checks or they disappear, producing
a `.grimplugin` full of wrong defaults with no error. The `from_hf_model_id`
method's doc note flags this; the CLI command that calls it must enforce it.

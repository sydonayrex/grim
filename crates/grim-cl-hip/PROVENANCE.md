# grim-cl-hip provenance

Grim fork of [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)
(`cubecl-hip` 0.10.0), MIT OR Apache-2.0 — see LICENSE-APACHE / LICENSE-MIT.

Source of record: the crates.io published `cubecl-hip-0.10.0`
(`~/.cargo/registry/src/index.crates.io-*/cubecl-hip-0.10.0/`), copied
verbatim under `src/`.

The `0.12.0-pre.2` tree in `old/repo/cubecl-main/crates/cubecl-hip/` is
reference only — **not** vendored: it would require vendoring ~10 workspace
crates (`cubecl`, `-core`, `-cpp`, `-llvm`, `-server`, `-common`, `-environment`,
`-monitoring`, `-runtime`, `-std`) and an MSRV bump from grim's declared `1.86`
to `1.95`. grim already ships on 0.10.0.

## Fork, not a vendor copy

This crate is named `grim-cl-hip` because it is a fork: upstream `cubecl-hip`
is the base, and grim-specific fixes and optimizations land here on top.
Divergence from upstream is recorded below; `src/` changes are expected and
welcome. (Earlier Phase 7 item 6 renamed it back to `cubecl-hip` to satisfy
`[patch.crates-io]`; that was reverted — a patch entry cannot rename a
package, and a fork should carry its own name.)

## Adaptations from upstream

- Package named `grim-cl-hip` (dir `crates/grim-cl-hip`); `version = "0.10.0"`
  tracks the upstream base. `publish = false`.
- `Cargo.toml` de-namespaced: `workspace = true` deps → explicit registry
  versions; `path = "../cubecl-*"` dropped.
- `cubecl-hip-sys` dependency binds to `grim-hip-sys` via a Cargo dep-rename
  (`cubecl-hip-sys = { path = "../grim-hip-sys", package = "grim-hip-sys" }`),
  which carries the ROCm 7.16-generated bindings while keeping the vendored
  `cubecl_hip_sys::…` import paths valid.

## Consumed via

`grim-backend-rocm` depends on it by path
(`grim-cl-hip = { path = "../grim-cl-hip", optional = true }`, feature
`cubecl`) and uses it directly (`use grim_cl_hip::{AmdDevice, HipRuntime}`).
The `cubecl` umbrella is used WITHOUT its `hip` feature, so the registry
`cubecl-hip` — and its `cubecl-hip-sys` 7.2 bindings, which do not compile
against ROCm 7.16 headers — never enter the graph. There is no `[patch]`
entry: a patch requires the package name to match, which a renamed fork
cannot satisfy.

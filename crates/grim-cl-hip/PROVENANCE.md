# grim-cl-hip provenance

Vendored from [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)
(`cubecl-hip` 0.10.0), MIT OR Apache-2.0 — see LICENSE-APACHE / LICENSE-MIT.

Source of record: the crates.io published `cubecl-hip-0.10.0`
(`~/.cargo/registry/src/index.crates.io-*/cubecl-hip-0.10.0/`), copied verbatim
under `src/`.

The `0.12.0-pre.2` tree in `old/repo/cubecl-main/crates/cubecl-hip/` is
reference only — **not** vendored: it would require vendoring ~10 workspace
crates (`cubecl`, `-core`, `-cpp`, `-llvm`, `-server`, `-common`, `-environment`,
`-monitoring`, `-runtime`, `-std`) and an MSRV bump from grim's declared `1.86`
to `1.95`. grim already ships on 0.10.0.

## Adaptations from upstream

- Package renamed `grim-cl-hip` → back to `cubecl-hip` (dir name unchanged) so
  the root `[patch.crates-io]` binds registry consumers directly to this crate;
  no re-export shim exists anymore.
- `Cargo.toml` de-namespaced: `workspace = true` deps → explicit registry
  versions; `path = "../cubecl-*"` dropped.
- **No `src/` changes.**
- `cubecl-hip-sys` dependency binds to `grim-hip-sys` via a Cargo dep-rename
  (`cubecl-hip-sys = { path = "../grim-hip-sys", package = "grim-hip-sys" }`),
  which carries the ROCm 7.16-generated bindings while keeping the vendored
  `cubecl_hip_sys::…` import paths valid.

## Consumed via

Root `[patch.crates-io] cubecl-hip = { path = "crates/grim-cl-hip" }`, so
`grim-backend-rocm`'s `cubecl::hip` (umbrella `pub use cubecl_hip as hip`)
resolves to this vendored source unchanged. Consumers never reference the
crate by a grim-specific name.

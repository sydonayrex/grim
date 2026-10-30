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

- Renamed `cubecl-hip` → `grim-cl-hip` (lib `cubecl_hip` → `grim_cl_hip`).
- `Cargo.toml` de-namespaced: `workspace = true` deps → explicit registry
  versions; `path = "../cubecl-*"` dropped.
- **No `src/` changes.**
- `cubecl-hip-sys` dependency (`7.1.5280200`) still binds to `grim-hip-sys`
  through the root `[patch.crates-io]` shim (`crates/cubecl-hip-sys`), which
  carries the ROCm 7.16-generated bindings.

## Consumed via

`crates/cubecl-hip` re-export shim + root `[patch.crates-io] cubecl-hip`, so
`grim-backend-rocm`'s `cubecl::hip` (umbrella `pub use cubecl_hip as hip`)
resolves to this vendored source unchanged. The consumer crate never references
`grim_cl_hip` directly.

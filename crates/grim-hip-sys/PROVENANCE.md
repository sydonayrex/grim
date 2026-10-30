# grim-hip-sys provenance

Vendored from [tracel-ai/cubecl-hip](https://github.com/tracel-ai/cubecl-hip)
(crate `cubecl-hip-sys`, version 7.14.6085001), dual-licensed MIT OR
Apache-2.0 — both license texts ship in this directory.

## Adaptations from upstream

1. **Static-link default (inverted).** Upstream always rewrites the bindgen
   output into `libloading` wrappers. Here the committed bindgen file ships
   verbatim by default and `build.rs` emits link-time directives discovered by
   the workspace's own `build_rocm_detect.rs` logic (`include!`-ed from
   `crates/grim-backend-rocm/` — single source of truth). The `dynamic`
   feature restores upstream dlopen behavior.
2. **Single bindings include.** Upstream compiles all committed files and
   cfg-selects one; here `build.rs` copies exactly one selected file into
   `OUT_DIR/bindings.rs`.
3. **Bindings source.** `src/bindings/bindings_26385.rs` was generated for
   the installed HIP patch 26385 (ROCm 7.16) with:
   ```bash
   bindgen --no-layout-tests --output src/bindings/bindings_26385.rs wrapper.h -- \
     -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -resource-dir /usr/lib/clang/23
   ```
   To support a new HIP patch, regenerate the same way and add a
   `hip_<patch> = []` feature.
4. `hipGraphExtendFromGlobalStream` is a grim-specific declaration not present
   in the HIP 7.16 headers; it remains hand-declared in
   `grim-backend-rocm/src/device/handles.rs`.

## Failure semantics

* static (default): missing `libamdhip64.so` = loader error before `main`.
* `dynamic`: missing runtime = panic on the first HIP call; use
  `grim_hip_sys::is_available()` for a graceful probe. Off by default.

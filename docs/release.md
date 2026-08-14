# Release Management

## Versioning
Releases follow semantic versioning. The core library (`grim`), backend crates, and the CLI tool are currently synchronized under the same version tag in the workspace `Cargo.toml`.

## Build Profiles

*   **Release**: Maximizes throughput via `opt-level = 3`, LTO, and CPU-specific instructions (`target-cpu=native`).
*   **Debug**: Minimal optimizations for kernel stepping and trace generation.

## Release Artifacts

Pre-compiled binaries are generated for:
*   `x86_64-unknown-linux-gnu` (CPU, CUDA, ROCm)
*   `aarch64-apple-darwin` (Metal)
*   `wasm32-unknown-unknown` (WASM target)

## CI Workflows

GitHub Actions validates pushes to `main`.
1.  **Format & Lint**: Checks `cargo fmt` and `cargo clippy`.
2.  **Test**: Executes tests via `cargo test`. Backend-specific tests (e.g., CUDA) are run on appropriately provisioned runners.
3.  **Build**: Compiles binaries for standard targets.

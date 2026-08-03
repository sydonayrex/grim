# Release & Deployment

This document describes the build, versioning, and CI pipeline derived from the repository's actual configuration files.

## Versioning

The workspace defines a shared version in `Cargo.toml`:

```toml
[workspace.package]
version = "0.1.0"
```

All crates use `version.workspace = true`, so they share the same version string. No CHANGELOG, CHANGELOG.md, or release-please configuration exists in this repository.

TODO: confirm with maintainer whether a versioning policy (e.g., semver) is intended.

## CI Pipeline

The CI configuration is in `.github/workflows/ci.yml` with three jobs:

### Build & Test (`check` job)

Runs on `ubuntu-latest`:

```bash
cargo check --workspace
cargo test --workspace
```

Uses `dtolnay/rust-toolchain@stable` and `Swatinem/rust-cache@v2` for caching.

### Lint (`lint` job)

Runs on `ubuntu-latest`:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Requires the `clippy` and `rustfmt` components.

### Mutants (`mutants` job)

Runs on `ubuntu-latest` (30-minute timeout):

```bash
cargo mutants --workspace -p grim-quant --no-shuffle --timeout 120 --testing-reason CI
```

Uses `cargo-bins/cargo-mutants@v2`. Only runs mutation testing against `grim-quant`.

## Build Profiles

Release profile configuration (`Cargo.toml`):

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

## Packaging

The project does not produce distributable packages (deb, rpm, AppImage) from CI. Users build from source:

```bash
cargo build --release --workspace
```

The primary binary is `grim-cli` (at `target/release/grim-cli`).

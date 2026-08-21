# Maintainer Local CI Guide

This document outlines the commands required to run the full verification and CI suite locally before submitting changes.

## 1. Quick Verification Gate

```bash
# Workspace compilation (zero warnings)
cargo check --workspace --all-targets

# Unit and integration test suite
cargo test --workspace
```

---

## 2. Formatting & Linting

```bash
# Code formatting check
cargo fmt --all -- --check

# Clippy linter
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 3. Mutation Testing

Run mutation testing on core math and autograd crates:

```bash
# Mutation tests via mutants.toml
cargo mutants --in-place -p grim-autograd -p grim-constrain -p grim-quant
```

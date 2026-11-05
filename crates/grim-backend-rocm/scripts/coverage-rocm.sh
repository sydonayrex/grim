#!/usr/bin/env bash
# Local-only coverage for grim-backend-rocm — never a CI gate (consumer ROCm
# project: no GPU CI, see tests/README.md §Coverage).
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). If absent, this
# script prints a clear message and exits instead of requiring it.
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE_DIR"

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
    echo "cargo-llvm-cov not found. Install it with:"
    echo "  cargo install cargo-llvm-cov"
    echo "Coverage run skipped (local-only, not a CI gate)."
    exit 0
fi

# lcov report of the --lib CPU tests + doctests on this crate.
cargo llvm-cov --lib -p grim-backend-rocm --lcov --output-path /tmp/coverage-rocm.lcov

python3 - <<'PY'
import collections, re
path = "/tmp/coverage-rocm.lcov"
hit = collections.defaultdict(int)
found = collections.defaultdict(int)
cur = None
for line in open(path):
    line = line.strip()
    if line.startswith("SF:"):
        cur = line[3:]
    elif line.startswith("DA:") and cur:
        parts = line[3:].split(",")
        found[cur] += 1
        hit[cur] += int(parts[1])
rows = []
for f, n in found.items():
    if n == 0:
        continue
    missed = n - hit[f]
    rows.append((missed / n, missed, n, f))
rows.sort(reverse=True)
print("\nTop-10 files by uncovered-line fraction (local only, not a CI gate):")
for frac, missed, total, f in rows[:10]:
    print(f"  {frac:5.1%}  {missed:5d}/{total:<5d}  {f}")
PY

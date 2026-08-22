#!/usr/bin/env python3
"""WI-X10 eval regression gate: compare a fresh eval JSON against the golden
baseline and fail when the primary metric regresses beyond tolerance.

Usage:
    check_eval_regression.py <baseline.json> <candidate.json> [--tolerance 0.02]

Compares the first numeric value under "metrics" (ppl for ppl tasks,
exact_match for gsm8k). For ppl (lower-is-better) fails when
candidate > baseline * (1 + tolerance). For accuracy-style metrics
(higher-is-better, detected by metric key) fails when
candidate < baseline * (1 - tolerance).
"""

from __future__ import annotations

import argparse
import json
import sys

HIGHER_IS_BETTER_KEYS = ("exact_match", "accuracy", "correct")


def primary_metric(doc: dict) -> tuple[str, float]:
    metrics = doc.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        raise SystemExit("candidate/baseline JSON has no numeric 'metrics' object")
    # Prefer known keys, else first numeric entry.
    for key in HIGHER_IS_BETTER_KEYS + ("ppl", "perplexity"):
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, float(metrics[key])
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    raise SystemExit("no numeric metric found in JSON")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    with open(args.baseline) as f:
        base_doc = json.load(f)
    with open(args.candidate) as f:
        cand_doc = json.load(f)

    name, base = primary_metric(base_doc)
    _, cand = primary_metric(cand_doc)

    higher_better = any(k in name.lower() for k in HIGHER_IS_BETTER_KEYS)

    if higher_better:
        floor = base * (1.0 - args.tolerance)
        ok = cand >= floor
        print(f"{name}: baseline={base} candidate={cand} (must be >= {floor:.6f})")
    else:
        ceiling = base * (1.0 + args.tolerance)
        ok = cand <= ceiling
        print(f"{name}: baseline={base} candidate={cand} (must be <= {ceiling:.6f})")

    if not ok:
        print(f"FAIL: {name} regressed beyond {args.tolerance:.2%} tolerance", file=sys.stderr)
        return 1
    print(f"PASS: {name} within {args.tolerance:.2%} tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())

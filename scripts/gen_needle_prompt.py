#!/usr/bin/env python3
"""Generate a long-context needle/synthesis prompt for grim-cli --prompt-file.

Modes:
  needle  — filler prose with N planted 6-digit secrets at given % depths,
            ending with a user question asking for all secret codes.
  synth   — filler prose with one mid-context fact + a question requiring a
            long (>=600-token) answer that cites that fact.

Usage:
  python3 -I gen_needle_prompt.py needle --tokens 262000 --out /tmp/p.txt
  python3 -I gen_needle_prompt.py synth  --tokens 262000 --out /tmp/p.txt
Secrets are deterministic (seeded) so runs are reproducible; the expected
answers are printed to stderr for the gate script to grep.
"""
import argparse, hashlib, sys

FILLER = ("The merchant ledger records grain shipments across the river "
          "valley, noting tolls, barge counts, and seasonal flood margins. "
          "Clerks copied entries each dusk, sealing wax over the fold. ")

def filler_tokens_approx(words):
    # ~0.75 tokens per word for this prose; caller treats the result as approximate.
    return int(words * 0.75)

def secret_at(index: int, seed: str) -> str:
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    return str(int(h[:8], 16) % 1_000_000).zfill(6)

def build(mode: str, target_tokens: int, seed: str) -> str:
    words_needed = int(target_tokens / 0.75)
    block_words = 40          # one filler paragraph
    n_blocks = words_needed // block_words
    depths = [0.10, 0.50, 0.90]
    secrets = [secret_at(i, seed) for i in range(len(depths))]
    insert_at = [int(n_blocks * d) for d in depths]

    parts = []
    if mode == "needle":
        parts.append("You will read a long merchant ledger. Memorise every "
                     "6-digit verification code that appears in it.\n\n")
    else:
        parts.append("You will read a long merchant ledger. Pay close "
                     "attention to the customs house details.\n\n")
    for i in range(n_blocks):
        parts.append(f"Ledger entry {i}: {FILLER}")
        if i in insert_at:
            k = insert_at.index(i)
            if mode == "needle":
                parts.append(f" The verification code for this shipment is "
                             f"{secrets[k]}. ")
            else:
                parts.append(" The river customs house at Mile 47 keeps its "
                             "sealed ledgers in the north vault, and the "
                             "head clerk signs every barge manifest in red "
                             "ink before dusk each day. ")
    if mode == "needle":
        parts.append("\n\nList every 6-digit verification code that appeared "
                     "in the ledger, in order, separated by commas. Answer "
                     "with the codes only.")
    else:
        parts.append("\n\nDescribe in detail how the customs house at Mile 47 "
                     "operates: where the ledgers are kept, who signs the "
                     "barge manifests, when the signing happens, and what "
                     "material details the ledger records. Write at least "
                     "six full paragraphs.")
    return "".join(parts), secrets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["needle", "synth"])
    ap.add_argument("--tokens", type=int, default=262000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="grim262")
    a = ap.parse_args()
    text, secrets = build(a.mode, a.tokens, a.seed)
    with open(a.out, "w") as f:
        f.write(text)
    approx = len(text) // 4  # ~4 chars/token for this prose
    print(f"prompt-file={a.out} mode={a.mode} chars={len(text)} approx_tokens={approx}",
          file=sys.stderr)
    if a.mode == "needle":
        print("EXPECTED_SECRETS=" + ",".join(secrets), file=sys.stderr)

if __name__ == "__main__":
    main()
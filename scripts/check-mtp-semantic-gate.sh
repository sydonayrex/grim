#!/usr/bin/env bash
# MTP M2/M3 semantic-parity gate (semantic bar, 2026-11-12).
#
# The MTP accept path commits state from a seq_len==2 verify forward; plain
# decode is seq_len==1 per step. The two execution routes use different
# kernel variants of the same math (measured: ~1e-3/layer drift, ~0.27 max
# logit diff at a 2-token window). Byte-identity would require running the
# verify window at M=1, which makes MTP 1.5x SLOWER than plain decode at
# depth 1 — so the gate is SEMANTIC, not bitwise:
#
#   1. Forced-reject (`GRIM_MTP_FORCE_REJECT=1`, greedy, native MTP path)
#      MUST be byte-identical to `GRIM_MTP=off` — the rollback path must have
#      zero net state/output effect. (This one IS bitwise; it passes.)
#   2. Greedy `auto` MUST be deterministic run-to-run (identical text).
#   3. Greedy `auto` output MUST be coherent (no degenerate repetition).
#   4. Acceptance metrics (GRIM_MTP_DEBUG=1): accepts and reject-replays
#      both occur; replay=true rows emit exactly 1 token.
#
# Byte-identity for `auto` is NOT required and NOT achievable without
# nullifying MTP; see plans/PLAN-qwen35-mtp-dualgpu-ttft.md and the probe
# assets (window_decode_parity.rs, mtp_window_kernel_parity.rs).
#
# Usage:
#   scripts/check-mtp-semantic-gate.sh [model_gguf]
#
# Environment:
#   GRIM_MOVE_P2P=0 is forced (the peer's P2P move_to_device path breaks
#   plain decode on this box; tracked separately).

set -euo pipefail

MODEL="${1:-models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRIM_BIN="$REPO_ROOT/target/release/grim-cli"
WORK="$(mktemp -d /tmp/mtp_semantic_gate.XXXXXX)"

export GRIM_MOVE_P2P=0
export GRIM_CONTEXT=4096
export GRIM_KV_CACHE_LEN=4096
export GRIM_DECODE_GRAPH=0

if [ ! -f "$GRIM_BIN" ]; then
    echo "FATAL: $GRIM_BIN not found (cargo build --release -p grim-cli)" >&2
    exit 1
fi
if [ ! -f "$REPO_ROOT/$MODEL" ]; then
    echo "FATAL: model $MODEL not found" >&2
    exit 1
fi

run() { # run <log> <env...> -- <args...>
    local log="$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    env "${envs[@]}" "$GRIM_BIN" run "$REPO_ROOT/$MODEL" "$@" >"$log" 2>&1
}

strip_dbg() { sed -n '/Response:/,/Done\./p' "$1" | grep -v "mtp-debug"; }

echo "== MTP semantic gate =="
echo "model: $MODEL"
echo "work:  $WORK"

# ---------- 1. forced-reject byte-identity (bitwise bar) ----------
run "$WORK/off_count.log" GRIM_MTP=off -- "Count 1 to 10:" --temperature 0 --max-tokens 16 --seed 0
run "$WORK/fr_count.log" GRIM_MTP=auto GRIM_MTP_FORCE_REJECT=1 -- "Count 1 to 10:" --temperature 0 --max-tokens 16 --seed 0
run "$WORK/off_sky.log"  GRIM_MTP=off -- "Explain why the sky is blue." --temperature 0 --max-tokens 280 --seed 0
run "$WORK/fr_sky.log"  GRIM_MTP=auto GRIM_MTP_FORCE_REJECT=1 -- "Explain why the sky is blue." --temperature 0 --max-tokens 280 --seed 0

fail=0
for pair in "off_count.log fr_count.log" "off_sky.log fr_sky.log"; do
    set -- $pair
    if diff -q <(strip_dbg "$WORK/$1") <(strip_dbg "$WORK/$2") >/dev/null; then
        echo "PASS  forced-reject byte-identity: $1 vs $2"
    else
        echo "FAIL  forced-reject byte-identity: $1 vs $2"
        fail=1
    fi
done

# ---------- 2/3. auto determinism + coherence ----------
run "$WORK/auto_count_a.log" GRIM_MTP=auto -- "Count 1 to 10:" --temperature 0 --max-tokens 16 --seed 0
run "$WORK/auto_count_b.log" GRIM_MTP=auto -- "Count 1 to 10:" --temperature 0 --max-tokens 16 --seed 0
if diff -q <(strip_dbg "$WORK/auto_count_a.log") <(strip_dbg "$WORK/auto_count_b.log") >/dev/null; then
    echo "PASS  auto determinism (run-to-run identical)"
else
    echo "FAIL  auto determinism (run-to-run differ)"
    fail=1
fi

# Coherence: no degenerate repetition (same 8+ char token repeated >=8x in a row).
if grep -qE '(\S{8,})\1{7,}' <(strip_dbg "$WORK/auto_count_a.log"); then
    echo "FAIL  auto coherence (degenerate repetition)"
    fail=1
else
    echo "PASS  auto coherence"
fi

# ---------- 4. acceptance metrics ----------
run "$WORK/auto_count_dbg.log" GRIM_MTP=auto GRIM_MTP_DEBUG=1 -- "Count 1 to 10:" --temperature 0 --max-tokens 16 --seed 0
acc=$(grep -c "accepted=1" "$WORK/auto_count_dbg.log" || true)
rej=$(grep -c "accepted=0 .*replay=true" "$WORK/auto_count_dbg.log" || true)
bad=$(grep -c "accepted=0 .*emitted=2" "$WORK/auto_count_dbg.log" || true)
echo "metrics: accepted=$acc reject-replay=$rej bad-emit=$bad"
if [ "$acc" -gt 0 ] && [ "$rej" -gt 0 ] && [ "$bad" -eq 0 ]; then
    echo "PASS  acceptance metrics (mixed accept/reject, replay emits 1)"
else
    echo "FAIL  acceptance metrics"
    fail=1
fi

# ---------- 5. reject path kernel activity sanity ----------
# Forced-reject must actually exercise the speculative loop: kernel launches
# strictly exceed the plain off run (prefill dominates short prompts, so the
# ratio is well under the pure-decode ~3x; the acceptance metrics above are
# the primary activity proof).
kc_off=$(grep -oE 'grim_[a-z0-9_]+ +: [0-9]+' "$WORK/off_count.log" | awk -F': ' '{s+=$2} END {print s+0}')
kc_fr=$(grep -oE 'grim_[a-z0-9_]+ +: [0-9]+' "$WORK/fr_count.log" | awk -F': ' '{s+=$2} END {print s+0}')
echo "kernel launches: off=$kc_off forced-reject=$kc_fr"
if [ "$kc_fr" -gt "$kc_off" ]; then
    echo "PASS  speculative loop active under forced-reject"
else
    echo "FAIL  speculative loop not active under forced-reject"
    fail=1
fi

echo "== gate result: $([ $fail -eq 0 ] && echo PASS || echo FAIL) =="
echo "logs: $WORK"
exit $fail

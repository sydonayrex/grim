#!/usr/bin/env bash
# Task 3 gate: 262k long-context SYNTHESIS sanity (not just needle extraction).
#
# What it proves, end to end on the real Qwen3.8-27B:
#   1. The >=128k auto-profile (Task 2) fires with a clean environment and the
#      run completes with decode-graph active.
#   2. Long-context RETRIEVAL: all three planted secrets at 10%/50%/90% depth
#      are recalled (validates the Task 1a tail/halo protection at 25% budget).
#   3. Long-context SYNTHESIS + Task 1b: the model emits a >600-token answer to
#      a mid-context question — i.e. generation continues well past the 512
#      tokens the decode Quest minmax originally covered (incremental refresh).
#
# Usage: bash scripts/check-262k-synthesis.sh [TOKENS]
#   TOKENS default 131072+4096 headroom (a full 262k run works too — pass 262000).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf}"
TARGET_TOKENS="${1:-131500}"          # ACTUAL prompt tokens (auto-profile fires at >=131072 total budget)
# gen_needle_prompt.py's --tokens is a words-based estimate: actual tokens
# come out ~1.5x the target (measured: 90000 -> 134278, 128000 -> 191502).
GEN_TOKENS=$((TARGET_TOKENS * 2 / 3))
MAX_TOKENS=800                        # >512: exercises the incremental minmax
OUT_DIR="$(mktemp -d /tmp/grim-synth.XXXXXX)"
PROMPT="$OUT_DIR/synth_prompt.txt"
LOG="$OUT_DIR/synth_run.log"

echo "[synth] repo=$REPO_ROOT model=$MODEL target_tokens=$TARGET_TOKENS"
echo "[synth] out_dir=$OUT_DIR"

# 1. Generate the prompt (ChatML path: no --raw, so the model's chat template
#    applies — the earlier "<|im_end|> on token 1" failure was raw-text input).
python3 -I "$REPO_ROOT/scripts/gen_needle_prompt.py" synth \
    --tokens "$GEN_TOKENS" --out "$PROMPT" 2>"$OUT_DIR/expected.txt"

# 2. Run with a CLEAN environment: every profile var must be absent so the
#    auto-profile (not manual env) drives the configuration.
run() { env -u GRIM_KV_ARENA -u GRIM_SPLIT_LAYERS -u GRIM_SPARSE_PREFILL_PCT \
             -u GRIM_SPARSE_DECODE_PCT -u GRIM_MTP -u GRIM_MTP_NEXTN_BACKFILL \
             -u GRIM_MTP_DRAFT_CHAIN -u GRIM_PREFILL_CHUNK_SIZE \
    "$REPO_ROOT/target/release/grim-cli" run "$REPO_ROOT/$MODEL" \
    --prompt-file "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 --seed 0 \
    >"$LOG" 2>&1; }

echo "[synth] running grim-cli (this prefills ~$TARGET_TOKENS tokens — minutes)..."
run
rc=$?
echo "[synth] grim-cli rc=$rc"

fail() { echo "[synth] FAIL: $1"; echo "[synth] log tail:"; tail -5 "$LOG"; exit 1; }

[ "$rc" -eq 0 ] || fail "grim-cli exited rc=$rc"

# 3. Gate 1: auto-profile fired and configuration applied.
grep -q "longctx-profile" "$LOG" || fail "auto-profile banner missing"
grep -q 'GRIM_KV_ARENA=k8v4\|"k8v4"' "$LOG" || grep -q '"dtype":"K8V4"' "$LOG" \
  || fail "k8v4 arena not selected by auto-profile"
grep -q "decode-graph: active" "$LOG" || fail "decode-graph not active"

# 4. Gate 2: retrieval — all three planted secrets recalled in the ANSWER.
#    (The planted secrets also occur in the prompt body; count only
#    post-generation occurrences by stripping the prompt echo.)
ANSWER="$(awk '/Sampling: temp=0/{found=1; next} found' "$LOG")"
MISSING=0
while IFS= read -r secret; do
    [ -z "$secret" ] && continue
    echo "$ANSWER" | grep -q "$secret" || { echo "[synth] missing secret $secret"; MISSING=1; }
done < <(grep -oE "[0-9]{6}" "$PROMPT" | sort -u | head -3)
[ "$MISSING" -eq 0 ] || fail "needle recall incomplete"

# 5. Gate 3: synthesis length — the answer must exceed 600 generated tokens.
#    grim-cli reports decode token counts; accept either its summary or a
#    character-length floor (~4 chars/token x 600).
ANSWER_TOKENS="$(grep -oE "Decode: [0-9.]+ ms/token avg \(([0-9]+) tokens" "$LOG" | grep -oE "\([0-9]+" | tr -d '(' | tail -1)"
if [ -n "${ANSWER_TOKENS:-}" ]; then
    [ "$ANSWER_TOKENS" -ge 600 ] || fail "answer only $ANSWER_TOKENS tokens (<600)"
    echo "[synth] answer tokens=$ANSWER_TOKENS"
else
    ANSWER_CHARS="$(printf '%s' "$ANSWER" | wc -c)"
    [ "$ANSWER_CHARS" -ge 2400 ] || fail "answer only $ANSWER_CHARS chars (<~600 tokens)"
    echo "[synth] answer chars=$ANSWER_CHARS"
fi

echo "[synth] PASS: auto-profile active, decode-graph active, secrets 3/3, answer >= 600 tokens"
echo "[synth] artifacts: $LOG"

#!/usr/bin/env bash
# Mutation gate for the CityCrow repack.
#
# A green test suite is a claim, not evidence. This injects defects into the
# exact lines CityCrow's correctness rests on and asserts the suite catches
# each one. A surviving mutant means the tests cannot see that class of bug.
#
# Usage: scripts/check-citycrow-gate.sh [--keep]
#   --keep  leave a surviving mutation applied, for debugging.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SRC="crates/grim-quant/src/citycrow.rs"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

if [ ! -f "$SRC" ]; then
    echo "FAIL  $SRC not found" >&2
    exit 1
fi

# Pristine copy to restore from. Never use git checkout: a peer agent has
# uncommitted work in this tree and a checkout would destroy it.
PRISTINE="$(mktemp -t citycrow-pristine-XXXXXX.rs)"
cp "$SRC" "$PRISTINE"
restore() { cp "$PRISTINE" "$SRC"; }
trap 'restore; rm -f "$PRISTINE"' EXIT

pass() { printf 'KILL   %s\n' "$1"; }
survive() {
    printf 'SURVIVE %s\n' "$1"
    [ "$KEEP" -eq 0 ] && restore
}

# Run only the CityCrow tests. If the suite does not even build, that counts
# as a kill for a mutation that broke compilation.
run_suite() {
    cargo test -p grim-quant --lib citycrow 2>&1
}

# mutate <label> <python-expression-producing-new-file-body>
# The expression receives the source text and returns the mutated text.
mutate() {
    local label="$1" expr="$2"
    python3 - "$SRC" "$PRISTINE" "$expr" <<'PY'
import sys
src, pristine, expr = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(pristine).read()
new = eval(expr, {"t": text, "re": __import__("re")})
if new == text:
    sys.exit(3)          # mutation was a no-op: the pattern did not apply
open(src, "w").write(new)
PY
    if [ $? -eq 3 ]; then
        survive "$label (NO-OP: pattern did not match, mutation never applied)"
        return 1
    fi
    return 0
}

echo "=== CityCrow mutation gate"
echo "target: $SRC"
echo

TOTAL=0
KILLED=0

check() {
    local label="$1" expr="$2"
    TOTAL=$((TOTAL + 1))
    if ! mutate "$label" "$expr"; then
        return
    fi
    local out
    out="$(run_suite)"
    if printf '%s' "$out" | grep -qE 'FAILED|panicked|error\[|error: could not compile'; then
        pass "$label"
        KILLED=$((KILLED + 1))
    else
        survive "$label"
    fi
    restore
}

# 1. The lane stride. 8 codes per u32 is what V_DOT8_U32_U4 expects; 4 makes
#    every group past the first read out of bounds or pack into the wrong
#    nibbles. This is the bug the first draft of the repack actually had.
check "lane stride 8 codes/word -> 4" \
    't.replace("let word = i / 8;", "let word = i / 4;")'

# 2. The zero point is MEASURED from the decoder, so the mutation that
#    matters is one that breaks the probe: make it pick the wrong code.
check "zero-point probe picks the wrong code" \
    't.replace(".find(|&c| got[c as usize].abs() < 1e-3)",
             ".find(|&c| got[c as usize].abs() < 1.5)")'

# 3. The packer's bias must track the probed bias. If it drifts, a fixture
#    round-trips through the wrong codebook -- the exact silent-wrong-weights
#    failure the module exists to prevent.
check "packer bias detached from the probe" \
    't.replace("let t = (w / d + zero_point_probe() as f32).round().clamp(0.0, 3.0) as u8;",
             "let t = (w / d + 1.0).round().clamp(0.0, 3.0) as u8;")'

# 4. The group-size guard. Removing it lets a K that is not a multiple of 128
#    through, and the dot8 accumulator then reads across a group boundary.
check "K group-size guard removed" \
    't.replace("if k % CITYCROW_GROUP != 0 {", "if false {")'

# 5. The buffer-length guard. A short buffer must be refused, not read past.
check "buffer length guard removed" \
    't.replace("if data.len() < need_bytes {", "if false {")'

# 6. The bf16 scale rounding. Dropping RNE (truncating) shifts every group
#    scale low, so reconstruction drifts by up to one ulp per group.
check "bf16 round-to-nearest -> truncate" \
    't.replace("((bits + rounding) >> 16) as u16", "((bits) >> 16) as u16")'

# 7. (removed) A "code clamp" mutation used to live here. The clamp was
#    genuinely redundant -- t is bounded by construction -- so it was deleted
#    rather than defended. Keeping a mutation for code that does not exist
#    would make the gate look thorough while testing nothing.

# 8. The all-zero group. Treating an absent scale as 1.0 instead of 0 leaves
#    every code in the lane, so a zero group decodes to the zero point rather
#    than to zero.
check "zero-group scale 0 -> 1.0" \
    't.replace("if amax > 0.0 { f32_to_bf16(amax) } else { 0 }",
             "if amax > 0.0 { f32_to_bf16(amax) } else { f32_to_bf16(1.0) }")'

# 9. The packer/decoder mirror. decode_groups must invert the same stride the
#    packer wrote. The round-trip test alone cannot see this: if BOTH used the
#    same wrong stride it would pass, which is why
#    lane_layout_matches_the_isa_definition_bit_for_bit reads the raw words
#    with a longhand stride instead of going through decode_groups.
check "decoder stride diverges from packer" \
    't.replace("let code = ((words[i / 8] >> ((i % 8) * 4)) & 0xF) as f32;",
             "let code = ((words[i / 8] >> ((i % 4) * 4)) & 0xF) as f32;")'

# 10. The K2_0 alias rejection. Accepting Q4K would convert a 4.5 bpw format
#     through a 2-bit path. The test distinguishes the up-front guard from the
#     match's `_` arm, so deleting the guard is visible even though both paths
#     return an error.
check "unsupported-scheme rejection removed" \
    't.replace("if !scheme_supported(scheme) {", "if false {")'

# 11. The source-block count. A 128-wide group is TWO 64-elem blocks. Reading
#     one block per group under-reads the buffer; the reconstruction test
#     cannot see it because the source decoder is handed the same count.
check "source block count halved" \
    't.replace("let need_blocks = num_weights.div_ceil(64);",
             "let need_blocks = num_weights.div_ceil(128);")'

# 12. The group offset. Reading group g at the wrong base word overlaps the
#     previous group. groups_do_not_bleed_into_each_other is built so a
#     positive group followed by a negative one makes the overlap visible as a
#     code from the wrong side of the zero point.
check "group base word misaligned" \
    't.replace("let words = &mut qweight[g * CITYCROW_WORDS_PER_GROUP..][..CITYCROW_WORDS_PER_GROUP];",
             "let words = &mut qweight[(g * CITYCROW_WORDS_PER_GROUP).saturating_sub(1)..][..CITYCROW_WORDS_PER_GROUP];")'

echo
echo "---"
printf 'killed %d/%d\n' "$KILLED" "$TOTAL"
if [ "$KILLED" -ne "$TOTAL" ]; then
    echo "GATE FAILED: a surviving mutation means the suite is blind to that defect."
    exit 1
fi
echo "GATE PASSED: every injected defect was caught."

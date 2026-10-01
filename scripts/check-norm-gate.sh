#!/usr/bin/env bash
# Norm gate for plan items 5 and 5c (commit 9295cb40).
#
# Three norms are selectable now, and each one has a way to be wrong that a
# build will not catch:
#
#   per-layer  LlamaBlock.attn_norm / ffn_norm   kind + optional bias
#   output     Llama.norm                         kind + optional bias
#   post       LlamaBlock.attn_post_norm         always RMS, never biased
#
# This runs the suites that pin them and then two mutations, because both were
# once unverified claims:
#
#   A. attn_post_norm moved AFTER the residual add. This passed the whole
#      281-test library suite, because the only test re-implemented the
#      arithmetic in the test file and asserted two calls to one helper.
#   B. LlamaConfig::default claiming LayerNorm. Scoped to output_norm_spec and
#      not to --lib: integration tests are not part of --lib, so running it
#      against --lib reports a false survivor.
#
# Usage: scripts/check-norm-gate.sh
# Exits non-zero on any failure. Does not mutate the tree; both mutations are
# reverted before the next check runs.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

block="crates/grim-models/transformer/src/block.rs"
model="crates/grim-models/transformer/src/model.rs"
fail=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; fail=1; }

result_of() { grep -m1 '^test result' || true; }

echo "=== builds"
for c in grim-nn grim-models-transformer grim-engine grim-cli \
         grim-server grim-garage grim-speculative; do
    out="$(cargo build -p "$c" 2>&1)"
    diag="$(printf '%s' "$out" | grep -cE '^(error|warning)')"
    [ "$diag" -eq 0 ] && pass "$c builds" || fail "$c: $diag diagnostics"
done

echo
echo "=== library suites"
for c in grim-nn grim-models-transformer grim-engine; do
    r="$(cargo test -p "$c" --lib -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) pass "$c $r" ;;
        *)             fail "$c: ${r:-did not run}" ;;
    esac
done

echo
echo "=== norm suites"
for spec in \
    "grim-nn norm_kind" \
    "grim-models-transformer attn_post_norm" \
    "grim-models-transformer output_norm_spec" \
    "grim-models-transformer block_norm_spec" \
    "grim-models-transformer norm_adoption_seam" \
    "grim-models-transformer llama_shaped_table" \
    "grim-models-transformer model_module_structure"
do
    # shellcheck disable=SC2086
    set -- $spec
    r="$(cargo test -p "$1" --test "$2" -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) pass "$2 $r" ;;
        *)             fail "$2: ${r:-did not run}" ;;
    esac
done

echo
echo "=== MUTATION A: attn_post_norm after the residual add"
# Shape-matched, not a literal: a reformat must not turn a real finding into
# "anchor not found", which reads exactly like a pass.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$block" "$tmp/block.orig"
python3 - "$block" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
m = re.search(r"let added = match &self\.attn_post_norm \{.*?\n        \};", s, re.S)
if not m:
    sys.exit("anchor not found")
mut = ("let added = match &self.attn_post_norm {\n"
       "            None => grim_nn::modules::add_on_device(x_2d, &attn_out)?,\n"
       "            Some(n) => {\n"
       "                let summed = grim_nn::modules::add_on_device(x_2d, &attn_out)?;\n"
       "                n.forward(&summed)?\n"
       "            }\n"
       "        };")
open(p, "w").write(s[:m.start()] + mut + s[m.end():])
PY
if [ $? -ne 0 ]; then
    fail "A: could not locate the attn_post_norm arm"
else
    r="$(cargo test -p grim-models-transformer --lib -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "A SURVIVED -- the post-norm order is unpinned: $r" ;;
        *)             pass "A killed: $r" ;;
    esac
fi
cp "$tmp/block.orig" "$block"
r="$(cargo test -p grim-models-transformer --lib -j 1 2>&1 | result_of)"
case "$r" in *" 0 failed"*) pass "green after restore" ;; *) fail "not green after restore: $r" ;; esac

echo
echo "=== MUTATION B: LlamaConfig::default claims LayerNorm"
cp "$model" "$tmp/model.orig"
python3 - "$model" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
new = s.replace("            norm_kind: NormKind::Rms,",
                "            norm_kind: NormKind::LayerNorm,", 1)
if new == s:
    sys.exit("anchor not found")
open(p, "w").write(new)
PY
if [ $? -ne 0 ]; then
    fail "B: could not locate Default.norm_kind"
else
    r="$(cargo test -p grim-models-transformer --test output_norm_spec -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "B SURVIVED -- the default is unpinned: $r" ;;
        *)             pass "B killed: $r" ;;
    esac
fi
cp "$tmp/model.orig" "$model"
r="$(cargo test -p grim-models-transformer --test output_norm_spec -j 1 2>&1 | result_of)"
case "$r" in *" 0 failed"*) pass "green after restore" ;; *) fail "not green after restore: $r" ;; esac

echo
echo "=== the tree is unchanged by this script"
if git diff --quiet -- "$block" "$model"; then
    pass "block.rs and model.rs match HEAD"
else
    fail "this script left the tree modified"
fi

echo
echo "---"
if [ "$fail" -eq 0 ]; then
    echo "norm gate green"
else
    echo "norm gate FAILED"
fi
exit "$fail"
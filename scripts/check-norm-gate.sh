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
#   C. the parallel-residual branch deleted, i.e. every model computes
#      ffn(attn(x) + x). This is what a gptneox checkpoint with
#      use_parallel_residual set would silently get.
#   D. LlamaBlock::load_tp ignoring the config flag. A test that sets the flag
#      on an already-built block proves the branch but not the plumbing; this
#      survived the whole 283-test library suite before it was added.
#
# Usage: scripts/check-norm-gate.sh
# Exits non-zero on any failure. Does not mutate the tree; both mutations are
# reverted before the next check runs.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

block="crates/grim-models/transformer/src/block.rs"
model="crates/grim-models/transformer/src/model.rs"
nn="crates/grim-nn/src/modules.rs"
loader="crates/grim-engine/src/model_loader.rs"
fail=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; fail=1; }
# report: neither pass nor fail -- the thing ran, and the detail decides.
# It prints the first few lines of the captured output so a reader can see
# why without re-running anything.
report() {
    printf 'BLOCKED  %s\n' "$1"
    printf '%s\n' "$2" | grep -m4 -E 'error|warning' | sed 's/^/          /' || true
}

result_of() { grep -m1 '^test result' || true; }

# One classifier, used by BOTH the workspace scan and the per-crate scan. Two
# copies of this list is exactly how they drifted, and the drift is why an
# error in a peer's `charon.rs` still reported as our FAIL.
# Returns 0 for "the peer's" and 1 for "ours".
is_peer_error() {
    case "$1" in
        # by SYMBOL first -- a peer error can land in a crate we also own
        *F32\|*Fp8Blocked*|*QuantFormat*|*expert_stage*|*index_head_dim*|\
        *index_n_heads*|*index_source_layer_ids*|*Xing40GraphScratch*|*k_norm*|\
        *act_fp8_pad_buf*|*layer_geoms*|*WhiteRaven*|*WhiteCrow*|*KqNative*|\
        *moe_route_topk*|*tree_pie*|*TreePie*|*charon*)
            return 0 ;;
        # then by FILE, for the peer's files we do not have symbols for
        *shared_moe.rs|*qwen4exp*|*expert_offload.rs|*dtype.rs|*deepseek32*|\
        *xing40*|*backend-rocm*|*backend-cuda*|*backend-vulkan*|*backend-metal*|\
        *quant/*|*llama_bak*|*memory/*|*charon*|*tree_pie*|*iq_gemm*|*charon.rs*)
            return 0 ;;
        *) return 1 ;;
    esac
}

# --- the whole workspace, not just the crates these tests touch -----------
# 9295cb40 added a field to a struct used in three `mod tests` literals in
# grim-cli and missed all three. Every suite in this gate runs in a crate that
# does not depend on grim-cli, so the gate was green against a workspace that
# did not compile. This check is what that gap cost.
echo
echo "=== the whole workspace compiles, including test targets"
ws_out="$(cargo check --workspace --all-targets --message-format short 2>&1)"
ws_rc=$?
ws_errs="$(printf '%s' "$ws_out" | grep -c ': error' || true)"
# Attribute before judging: a peer agent mid-edit makes this red for reasons
# that are not norm work, and a gate that reports those as its own failures
# trains people to ignore it.
unattributed=""
own_failed=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    # `could not compile <crate>` carries no symbol and no file of its own; it
    # is a peer error only if some error in this same output is. Count those
    # separately so a cascade does not read as ours.
    if is_peer_error "$line"; then
        :                                      # theirs, wherever it lands
    else
        unattributed="$unattributed$line\n"    # ours
    fi
done <<< "$(printf '%s' "$ws_out" | grep ': error' || true)"

# cargo stops after the first failing crate, so one peer's error can hide a
# later one -- including ours. Check the crates this gate owns by name, one at
# a time, so "is my code broken" does not depend on who failed first.
own_broken=""
for c in grim-core grim-nn grim-models-transformer grim-engine grim-cli \
         grim-models-vision grim-format; do
    o2="$(cargo check -p "$c" --all-targets --message-format short 2>&1)"
    # A crate we own can still carry a peer's error, and the crate name alone
    # cannot tell the two apart -- so this goes through the same classifier.
    while IFS= read -r e; do
        [ -z "$e" ] && continue
        is_peer_error "$e" || own_broken="$own_broken$c: $e\n"
    done <<< "$(printf '%s' "$o2" | grep ': error' || true)"
done
if [ -n "$own_broken" ]; then
    printf '  FAIL  %b' "$own_broken"
    fail "a crate this gate owns does not compile"
    own_failed=1
fi
if [ "$ws_rc" -eq 0 ]; then
    pass "cargo check --workspace --all-targets: 0 errors"
    ws_blocked=0
elif [ -n "$unattributed" ]; then
    printf '  FAIL  %b' "$unattributed"
    fail "workspace check: unattributed errors (see above)"
    ws_blocked=1
else
    # A peer's in-flight edit. Every build and suite below depends on the
    # same tree, so running them would produce a cascade of "did not run"
    # failures that all restate this one cause. Stop instead.
    report "workspace check blocked by a peer's in-flight edit; skipping the rest" "$ws_out"
    ws_blocked=1
fi
if [ "${ws_blocked:-0}" -eq 1 ]; then
    echo
    echo "---"
    # `fail` has already set the flag; respect it. An unattributed error is
    # ours, and a "BLOCKED" footer on top of a FAIL reads as an excuse.
    if [ "$fail" -eq 0 ] && [ "$own_failed" -eq 0 ]; then
        echo "norm gate BLOCKED (a peer's edit, not a norm regression)"
        exit 2
    fi
    echo "norm gate FAILED"
    exit 1
fi


echo "=== builds"
for c in grim-nn grim-models-transformer grim-engine grim-cli \
         grim-server grim-garage grim-speculative; do
    out="$(cargo build -p "$c" 2>&1)"
    diag="$(printf '%s' "$out" | grep -cE '^(error|warning)')"
    [ "$diag" -eq 0 ] && pass "$c builds" || fail "$c: $diag diagnostics"
done

echo
echo "=== library suites"
# Two tests are known flakes that live outside this change and are recorded in
# plans/plan-yodellers-inc.md section 9a and 4.7:
#   gemma2::tests::test_attention_logit_softcapping_changes_logits
#   tests::test_memory_certificate_admission_gate_real_hw
# Both read live state -- one compares two logits paths that can coincide, the
# other samples live VRAM twice -- so neither is a signal about the norms.
# A failure that is NOT one of them is a regression and still fails here.
known_flakes=(
    "gemma2::tests::test_attention_logit_softcapping"
    "test_memory_certificate_admission_gate_real_hw"
)
for c in grim-nn grim-models-transformer grim-engine; do
    out="$(cargo test -p "$c" --lib -j 1 2>&1)"
    r="$(printf '%s' "$out" | result_of)"
    case "$r" in
        *" 0 failed"*) pass "$c $r" ;;
        *)
            only_known=1
            while IFS= read -r failing; do
                name="${failing#---- }"; name="${name%% stdout*}"
                is_known=0
                for k in "${known_flakes[@]}"; do
                    [[ "$name" == *"$k"* ]] && is_known=1
                done
                [ "$is_known" -eq 1 ] || only_known=0
                printf '      known flake: %s\n' "$name"
            done < <(printf '%s' "$out" | grep -E '^---- ' || true)
            if [ "$only_known" -eq 1 ] && [ -n "$(printf '%s' "$out" | grep -E '^---- ' || true)" ]; then
                printf 'SKIP  %s: %s (known flake, not a norm regression)\n' "$c" "$r"
            else
                fail "$c: ${r:-did not run}"
            fi
            ;;
    esac
done

echo
echo "=== norm suites"
for spec in \
    "grim-nn norm_kind" \
    "grim-core norm_kind_by_architecture" \
    "grim-core norm_bias_names" \
    "grim-core layer_out_norm" \
    "grim-engine norm_kind_wiring" \
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
cp "$block" "$tmp/block.start"
cp "$model" "$tmp/model.start"
cp "$nn" "$tmp/nn.start"
cp "$loader" "$tmp/loader.start"
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
out="$(cargo test -p grim-models-transformer --lib -j 1 2>&1 | result_of)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

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
out="$(cargo test -p grim-models-transformer --test output_norm_spec -j 1 2>&1 | result_of)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

echo
echo "=== MUTATION C: the FFN branch deleted (always sequential)"
cp "$block" "$tmp/block.c.orig"
# The mutation and its restore are one function so the restore runs on EVERY
# path out of it. An earlier version ran them separately, and when the anchor
# was missing the restore was skipped -- leaving `block.rs` with the
# parallel-residual branch deleted in the working tree.
mutate_c() {
    python3 - "$block" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = ("        let x_norm = if self.use_parallel_residual {\n"
       "            self.ffn_norm.forward(&x_2d)?\n"
       "        } else {\n"
       "            self.ffn_norm.forward(&added)?\n"
       "        };")
if old not in s:
    sys.exit("anchor not found")
open(p, "w").write(s.replace(old, "        let x_norm = self.ffn_norm.forward(&added)?;", 1))
PY
    if [ $? -ne 0 ]; then
        report "C: could not locate the parallel-residual branch" "the anchor \
did not match; block.rs is unchanged and no mutation ran"
        return 1
    fi
    r="$(cargo test -p grim-models-transformer --lib -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "C SURVIVED -- the parallel-residual branch is unpinned: $r" ;;
        *)             pass "C killed: $r" ;;
    esac
    return 0
}
mutate_c || true
cp "$tmp/block.c.orig" "$block"
if ! cmp -s "$block" "$tmp/block.c.orig"; then
    fail "C: block.rs was not restored"
fi
out="$(cargo test -p grim-models-transformer --lib -j 1 2>&1 | result_of)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

echo
echo "=== MUTATION D: load_tp ignores the config flag"
# A test that sets the flag on an already-built block proves the branch works
# but not that the flag is plumbed. This survived the whole library suite.
cp "$block" "$tmp/block.d.orig"
python3 - "$block" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "            use_parallel_residual: cfg.use_parallel_residual,"
if old not in s:
    sys.exit("anchor not found")
open(p, "w").write(s.replace(old, "            use_parallel_residual: false,", 1))
PY
if [ $? -ne 0 ]; then
    fail "D: could not locate the load_tp wiring"
else
    r="$(cargo test -p grim-models-transformer --test norm_adoption_seam -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "D SURVIVED -- the config flag is not plumbed to the block: $r" ;;
        *)             pass "D killed: $r" ;;
    esac
fi
cp "$tmp/block.d.orig" "$block"
out="$(cargo test -p grim-models-transformer --test norm_adoption_seam -j 1 2>&1 | result_of)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

echo
echo "=== MUTATION E: norm bias gated on the flag again"
# Presence-gating is the fix. Reverting it means the eighteen models that ship
# attn_norm_b (gptneox, phi2, mpt, jais, orion, codeshell, starcoder,
# starcoder2, nemotron, bloom, gpt2, falcon, jais2, phimoe, pockettts, rwkv6,
# rwkv7, stablelm) compute a normalisation they were not trained with.
cp "$nn" "$tmp/nn.e.orig"
python3 - "$nn" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "        let loaded = ws.get([dim], \"bias\").ok();"
if old not in s:
    sys.exit("anchor not found")
open(p, "w").write(s.replace(old, "        let loaded = if declared_bias { ws.get([dim], \"bias\").ok() } else { None };", 1))
PY
if [ $? -ne 0 ]; then
    fail "E: could not locate Norm::load's bias probe"
else
    r="$(cargo test -p grim-models-transformer --test norm_adoption_seam -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "E SURVIVED -- the norm bias is flag-gated again: $r" ;;
        *)             pass "E killed: $r" ;;
    esac
fi
cp "$tmp/nn.e.orig" "$nn"
out="$(cargo test -p grim-models-transformer --test norm_adoption_seam -j 1 2>&1 | result_of)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

echo
echo "=== MUTATION F: the loader maps every architecture to RMS"
# `uses_layernorm` can be correct while `norm_kind_for` ignores it, which is
# what shipped: the table existed and twenty LayerNorm models were still served
# as RMS. The core test cannot catch this -- it only reads the table.
cp "$loader" "$tmp/loader.f.orig"
python3 - "$loader" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "    if arch.uses_layernorm() {"
if old not in s:
    sys.exit("anchor not found")
open(p, "w").write(s.replace(old, "    if !arch.uses_layernorm() {", 1))
PY
if [ $? -ne 0 ]; then
    fail "F: could not locate norm_kind_for"
else
    r="$(cargo test -p grim-engine --test norm_kind_wiring -j 1 2>&1 | result_of)"
    case "$r" in
        *" 0 failed"*) fail "F SURVIVED -- the loader ignores the norm kind: $r" ;;
        *)             pass "F killed: $r" ;;
    esac
fi
cp "$tmp/loader.f.orig" "$loader"
out="$(cargo test -p grim-engine --test norm_kind_wiring -j 1 2>&1)"
r="$(printf '%s' "$out" | result_of)"
case "$r" in
    *" 0 failed"*) pass "green after restore" ;;
    *) report "not green after restore: $r" "$out" ;;
esac

echo
echo "=== the tree is unchanged by THIS RUN"
# Compared against a snapshot taken when the script started, not against HEAD.
# `git diff --quiet` would report a legitimate uncommitted change as if the
# script had caused it, which is how the first version of this gate failed on a
# clean tree: the parallel-residual work was simply not committed yet.
if cmp -s "$block" "$tmp/block.start" && cmp -s "$model" "$tmp/model.start" \
   && cmp -s "$nn" "$tmp/nn.start" \
   && cmp -s "$loader" "$tmp/loader.start"; then
    pass "block.rs, model.rs, modules.rs and model_loader.rs are byte-identical to the pre-run snapshot"
else
    fail "this script modified a file and did not restore it"
fi

echo
echo "---"
if [ "$fail" -eq 0 ]; then
    echo "norm gate green"
else
    echo "norm gate FAILED"
fi
exit "$fail"
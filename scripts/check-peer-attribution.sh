#!/usr/bin/env bash
# Is a compiler error ours, or the peer agent's in-flight work?
#
# ## Why this exists
#
# `scripts/check-norm-gate.sh` runs `cargo check --workspace --all-targets`
# before its builds and suites, because `9295cb40` (mine) left three
# `GgufTokenizer` literals incomplete and the gate stayed green for two days:
# every suite it ran lived in a crate that does not depend on `grim-cli`.
#
# That check must distinguish a regression of OURS from the peer agent's
# half-finished edits, or it is useless in both directions -- it cries wolf on
# their churn, and it learns people to ignore it.
#
# ## Why a rule and not a filename list
#
# The first version classified by filename: `backend-rocm`, `dtype.rs`,
# `shared_moe.rs`, ... and grew a pattern every time the peer touched something
# new. That is the wrong shape twice over. A list cannot be complete, and it
# cannot distinguish a peer's error in `decode_graph.rs` (a file we both edit)
# from ours.
#
# The rule is structural, which is what actually distinguishes them:
#
#   1. If the error names a symbol or path the peer has been adding, it is
#      theirs -- regardless of which file it lands in. That is the list below,
#      and it is genuinely a list: there is no way to infer "they are adding a
#      field called `expert_stage`" from the error text.
#   2. Otherwise, if the file is UNTRACKED or UNCOMMITTED, it is theirs. A
#      committed file that does not compile is ours: nobody else's in-flight
#      work is hiding in the index.
#
# Rule 2 is what makes the list unnecessary in the common case. Most of the
# peer's work is uncommitted, and `git status` says so without anyone having to
# remember to update a pattern.
#
# ## What is asserted here
#
# The classifier is exercised directly, without cargo, against hand-built
# inputs and a real repository. Each case states who owns it and why.
set -uo pipefail

_src="${BASH_SOURCE[0]:-$0}"
REPO="$(cd "$(dirname "$_src")/.." && pwd)"
if [ ! -f "$REPO/scripts/check-norm-gate.sh" ]; then
    printf 'FAIL  cannot locate the repo root (REPO=%s)\n' "$REPO"
    exit 1
fi
GATE="$REPO/scripts/check-norm-gate.sh"

pass_n=0
fail_n=0
pass() { printf 'PASS  %s\n' "$1"; pass_n=$((pass_n + 1)); }
fail() { printf 'FAIL  %s\n' "$1"; fail_n=$((fail_n + 1)); }

# Pull the classifier out of the gate, so this tests the shipped function
# rather than a copy of it. A copy is how the two copies of the old list
# drifted apart in the first place.
classifier() {
    awk '
        /^is_peer_error\(\) \{/ { grab = 1 }
        grab             { print }
        grab && /^}$/     { exit }
    ' "$GATE"
}

# case_is_peer <label> <expect PEER|OURS> <error line>
case_is_peer() {
    local label="$1" expect="$2" line="$3" got
    if is_peer_error "$line"; then got=PEER; else got=OURS; fi
    if [ "$got" = "$expect" ]; then
        pass "$label -> $got"
    else
        fail "$label -> $got, want $expect"
        printf '        %s\n' "$line" | sed 's/^/        /'
    fi
}

# Load the classifier into this shell. `gate_root` must exist first: the
# classifier resolves a relative path from an error line, and the gate defines
# it before calling the classifier. Loading the function without it aborts on
# the first case under `set -u`, which is what the first version did.
gate_root="$REPO"
eval "$(classifier)"
if ! declare -f is_peer_error > /dev/null; then
    fail "could not load is_peer_error from the gate"
    printf -- '---\n%d FAIL\n' "$((fail_n + 1))"
    exit 1
fi

printf '=== 1. the gate ships one classifier, and it is loadable\n'
n="$(grep -c '^is_peer_error() {' "$GATE")"
if [ "$n" -eq 1 ]; then pass "exactly one definition in the gate"; else fail "$n definitions"; fi
if grep -q 'case "\$1" in' <<< "$(classifier)"; then
    pass "it takes the error line as an argument"
else
    fail "it does not look like the classifier"
fi

printf '\n=== 2. peer symbols, wherever they land\n'
# These name something the peer is adding. They are theirs even in a file we
# also edit -- that is the whole reason rule 1 exists.
case_is_peer "Fp8Blocked16 in a file we own"      PEER \
    'crates/grim-format/src/convert.rs:378:65: error[E0004]: non-exhaustive patterns: `&FloatPackScheme::Fp8Blocked16` not covered'
case_is_peer "expert_stage in a file we own"      PEER \
    'crates/grim-models/transformer/src/qwen4exp_flash_next.rs:1961:5: error[E0277]: the trait bound `Spillable` is not satisfied'
case_is_peer "Xing40GraphScratch field missing"   PEER \
    'crates/grim-models/transformer/src/decode_graph.rs:7637:49: error[E0609]: no field `inter` on type `&mut Xing40GraphScratch`'
case_is_peer "DeepSeek32Config index fields"      PEER \
    'crates/grim-engine/src/model_loader.rs:2007:23: error[E0063]: missing fields `index_head_dim`, `index_n_heads`, `index_source_layer_ids`'
case_is_peer "a peer backend crate"               PEER \
    'crates/grim-backend-rocm/tests/whiteraven_journey.rs:94:16: error[E0599]: no associated function named `from_storage` found for struct `Tensor`'

printf '\n=== 3. ours: a norm symbol, in a norm file\n'
case_is_peer "missing added_tokens (mine, 9295cb40)" OURS \
    'crates/grim-cli/src/train.rs:1590:25: error[E0063]: missing field `added_tokens` in initializer of `grim_format::GgufTokenizer`'
case_is_peer "missing has_output_bias (mine)"      OURS \
    'crates/grim-cli/src/train.rs:1945:19: error[E0063]: missing field `has_output_bias` in initializer of `LlamaConfig`'
case_is_peer "missing norm_kind (mine)"            OURS \
    'crates/grim-models/transformer/src/model.rs:88:9: error[E0063]: missing field `norm_kind` in initializer of `LlamaConfig`'
case_is_peer "a borrow error in our own norm code" OURS \
    'crates/grim-nn/src/modules.rs:120:22: error[E0502]: cannot borrow `ws` as mutable more than once'
case_is_peer "a norm suite fails to compile"      OURS \
    'crates/grim-core/tests/norm_kind_by_architecture.rs:44:5: error[E0425]: cannot find value `LAYER_OUT_NORM` in this scope'

printf '\n=== 4. the structural rule: untracked or uncommitted is theirs\n'
# The point of rule 2. These need no pattern at all.
PEER_UNTRACKED='crates/grim-backend-rocm/tests/whiteraven_journey.rs:111:17: error: unused variable: `stage`'
case_is_peer "an untracked file, unrecognised name" PEER "$PEER_UNTRACKED"
case_is_peer "the same file with a nonsense symbol" PEER \
    'crates/grim-backend-rocm/tests/whiteraven_journey.rs:42:3: error[E0308]: mismatched types: expected `u64`, found `usize`'

# A COMMITTED file that does not compile is ours. This is the assertion that
# stops the classifier from becoming an escape hatch: widening the peer list
# can no longer excuse a break in a file we own.
committed_bad='crates/grim-nn/src/modules.rs:201:9: error[E0425]: cannot find value `ZZZ_NOT_A_PEER_SYMBOL` in this scope'
case_is_peer "a nonsense symbol in a committed file" OURS "$committed_bad"

printf '\n=== 5. against the live repository\n'
# The classifier must agree with git about what is uncommitted. This is the part
# that needs the repo; the cases above are pure text.
while IFS= read -r line; do
    [ -z "$line" ] && continue
    file="${line%%:*}"
    if ! git -C "$REPO" ls-files --error-unmatch "$file" > /dev/null 2>&1 \
       || [ -n "$(git -C "$REPO" status --porcelain -- "$file")" ]; then
        want=PEER
    else
        want=OURS
    fi
    case_is_peer "live: $(basename "$file")" "$want" "$line"
done <<< "$(git -C "$REPO" status --porcelain | awk '{print $2}' | grep '\.rs$' | head -20 \
           | sed "s|^|$REPO/|" | while read -r f; do
                   if [ -f "$f" ]; then
                       printf '%s:1:1: error: probe\n' "$f"
                   fi
               done)"

printf '\n=== 6. the classifier is not a filename list\n'
# If it is one, the two copies can drift again. It must consult git.
if grep -q 'ls-files\|status --porcelain' <<< "$(classifier)"; then
    pass "it consults git, so a committed break cannot be excused"
else
    fail "it is a pure pattern list -- a committed break can be misattributed"
fi

printf -- '---\n'
printf '%d FAIL\n' "$fail_n"
[ "$fail_n" -eq 0 ]
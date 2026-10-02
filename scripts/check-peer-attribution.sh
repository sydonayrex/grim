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

printf '\n=== 2b. a multi-word alternative, which `|` cannot carry\n'
# rustc writes "unreachable pattern" with a SPACE. That cannot be one arm of a
# `|` list -- a space ends the pattern list -- so it needs its own arm. The
# first version spelled it `*unreachable_pattern*`, which matched nothing and
# looked like coverage. Caught by section 4, which tests every pattern against
# a line naming its own subject.
case_is_peer "unreachable pattern (two words)"    PEER \
    'crates/grim-format/src/gguf.rs:2097:11: error: unreachable pattern: no value can reach this'
case_is_peer "E0004 unreachable pattern"          PEER \
    'crates/grim-format/src/gguf.rs:2097:11: error[E0004]: unreachable pattern: `GgufDType::Q2_0` not covered'
case_is_peer "unreachable, but not a pattern"     OURS \
    'crates/grim-nn/src/modules.rs:88:9: error: unreachable code after a return'

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
# Do NOT hardcode a fixture. The peer commits and creates files constantly, so
# a hardcoded one silently changes class -- an untracked file that gets committed
# becomes a COMMITTED file, and this rule then reports it as OURS, which is the
# very next assertion. That happened: `whiteraven_journey.rs` and then
# `citycrow.rs`.
#
# Either kind works -- untracked OR uncommitted -- because the rule is "not in
# the index". Prefer a file that is NOT ours, so the assertion stays honest.
PEER_FILE=""
for cand in $(git -C "$REPO" status --porcelain \
              | awk '$1 == "??" && $2 ~ /\.rs$/ { print $2 }'); do
    PEER_FILE="$cand"; break
done
if [ -z "$PEER_FILE" ]; then
    for cand in $(git -C "$REPO" status --porcelain \
                  | awk '$1 != "??" && $2 ~ /\.rs$/ { print $2 }'); do
        # Skip anything this work owns, or the assertion would be circular.
        case "$cand" in
            *model_loader.rs|*modules.rs|*/block.rs|*/model.rs) continue ;;
        esac
        PEER_FILE="$cand"; break
    done
fi
if [ -z "$PEER_FILE" ]; then
    fail "no untracked or uncommitted .rs file exists outside this work, so the \
structural rule has no live fixture"
else
    # "committed" here means COMMITTED AND CLEAN, which is the one state that
    # cannot demonstrate the rule. ls-files alone says tracked, which is true of
    # a modified file too -- the first version reported "committed" for a file
    # with uncommitted changes in it.
    if [ -z "$(git -C "$REPO" status --porcelain -- "$PEER_FILE")" ] \
       && git -C "$REPO" ls-files --error-unmatch "$PEER_FILE" > /dev/null 2>&1; then
        fail "$PEER_FILE is committed and clean -- it cannot demonstrate the rule"
    fi
    case "$(git -C "$REPO" status --porcelain -- "$PEER_FILE" | cut -c1-2)" in
        '??') state=untracked ;;
        *)   state=uncommitted ;;
    esac
    case_is_peer "a $state file ($(basename "$PEER_FILE"))" PEER \
        "$PEER_FILE:1:1: error: probe"
    case_is_peer "the same file with a nonsense symbol" PEER \
        "$PEER_FILE:42:3: error[E0308]: mismatched types: expected `u64`, found `usize`"
fi

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

printf '\n=== 6. every pattern matches a line that names its subject\n'
# A pattern that cannot match the text it exists for is dead weight that looks
# like coverage. Read the WHOLE function, not just the `|` list: a standalone
# `*"..."*)` arm sits outside it, and reading only the list made a dead
# standalone pattern invisible -- the same defect class this section exists to
# catch. `*unreachable_pattern*` sat in the list for a session and matched
# nothing, because rustc writes "unreachable pattern" with a space.
fn="$(classifier)"
in_list="$(printf '%s\n' "$fn" | sed -n '/case "\$1" in/,/^    esac/p')"
subject_of() {
    case "$1" in
        f32)                 echo "value is f32 in a shape mismatch" ;;
        Fp8Blocked)          echo "FloatPackScheme::Fp8Blocked16 not covered" ;;
        QuantFormat)         echo "QuantFormat::Fp8Blocked16 not covered" ;;
        expert_stage)        echo "missing field \`expert_stage\` in initializer" ;;
        index_head_dim)      echo "missing fields \`index_head_dim\`, \`index_n_heads\`" ;;
        index_n_heads)       echo "missing fields \`index_head_dim\`, \`index_n_heads\`" ;;
        index_source_layer_ids)
                            echo "missing fields \`index_head_dim\`, \`index_n_heads\`, \`index_source_layer_ids\`" ;;
        Xing40GraphScratch)  echo "no field \`inter\` on type \`&mut Xing40GraphScratch\`" ;;
        k_norm)              echo 'idx_ws.scoped("k_norm")' ;;
        act_fp8_pad_buf)     echo "no field \`act_fp8_pad_buf\` on struct" ;;
        layer_geoms)         echo "field \`layer_geoms\` is never read" ;;
        WhiteRaven)          echo "WhiteRaven dispatch: no such arm" ;;
        WhiteCrow)           echo "WhiteCrow dispatch: no such arm" ;;
        KqNative)            echo "no field \`KqNativeResident\` on struct" ;;
        moe_route_topk)      echo "no associated function \`moe_route_topk_on_device\`" ;;
        tree_pie)            echo "tree_pie_dispatch: no such arm" ;;
        TreePie)             echo "TreePie: no such arm" ;;
        charon)              echo "charon_kq_native_grouped_parity.rs: error" ;;
        latent_dim)          echo "cannot find value \`latent_dim\` in this scope" ;;
        kv_stride)           echo "cannot find value \`kv_stride\` in this scope" ;;
        GgufDType)           echo "GgufDType::Q2_0 not covered" ;;
        PQ2_0)               echo "GgufDType::PQ2_0 not covered" ;;
        PTQ1_0)              echo "GgufDType::PTQ1_0 not covered" ;;
        TQ1_0)               echo "GgufDType::TQ1_0 not covered" ;;
        TQ2_0)               echo "GgufDType::TQ2_0 not covered" ;;
        Q1_0)                echo "GgufDType::Q1_0 not covered" ;;
        Q2_0)                echo "GgufDType::Q2_0 not covered" ;;
        from_storage)        echo "no associated function named \`from_storage\` found" ;;
        Spillable)           echo "the trait bound \`Spillable\` is not satisfied" ;;
        Greycrow)            echo "Greycrow: no such arm" ;;
        greycrow)            echo "greycrow_geometry: no such arm" ;;
        CityCrow)            echo "CityCrow: no such arm" ;;
        citycrow)            echo "citycrow.rs: no such arm" ;;
        CCProbe)             echo "CCProbe: no such arm" ;;
        cc_probe)            echo "cc_probe.rs: no such arm" ;;
        kq_native)           echo "no field \`kq_native\` on struct" ;;
        from_u8_bytes)       echo "no method named \`from_u8_bytes\` found" ;;
        *)                   echo "" ;;     # no known subject -> tested below
    esac
}
dead=0; unknown=0; n=0
# Both forms: `*name*` tokens in the `|` list, and `*"...")` standalone arms.
list_pats="$(printf '%s\n' "$fn" | grep -oE '\*[A-Za-z0-9_]+\*' | tr -d '*' | sort -u)"
arm_pats="$(printf '%s\n' "$fn" | grep -oE '\*"[^"]+"\*\)' | sed 's/^\*"//; s/"\*)$//')"
for pat in $list_pats; do
    n=$((n + 1))
    subject="$(subject_of "$pat")"
    if [ -z "$subject" ]; then
        unknown=$((unknown + 1))
        continue
    fi
    if ! case "$subject" in *$pat*) true ;; *) false ;; esac; then
        fail "pattern $pat cannot match: $subject"
        dead=$((dead + 1))
    fi
done
if [ "$dead" -eq 0 ]; then
    pass "every known pattern matches a line naming its subject ($n scanned)"
fi
# A pattern with no subject here is not proven dead, but it is also not proven
# live, and silently skipping it is how `unreachable_pattern*` hid. Require the
# list to be explicit.
if [ "$unknown" -eq 0 ]; then
    pass "no list pattern is unscanned"
else
    fail "$unknown list pattern(s) have no subject: add one or remove it"
fi
# A standalone arm must match the text it exists for. `*"unreachable pattern"*`
# is the case: as `*unreachable_pattern*` it sat in the list for a session and
# matched nothing, because rustc writes the two words with a space.
for arm in $arm_pats; do
    if ! case "error: $arm: no value can reach this" in *"$arm"*) true ;; *) false ;; esac; then
        fail "standalone arm cannot match its own text: $arm"
        dead=$((dead + 1))
    fi
done
if [ "$dead" -eq 0 ]; then
    pass "every standalone arm matches the text it exists for ($(printf '%s\n' $arm_pats | wc -l) scanned)"
fi

printf '\n=== 7. the classifier is not a filename list\n'
# If it is one, the two copies can drift again. It must consult git.
if grep -q 'ls-files\|status --porcelain' <<< "$(classifier)"; then
    pass "it consults git, so a committed break cannot be excused"
else
    fail "it is a pure pattern list -- a committed break can be misattributed"
fi

printf -- '---\n'
printf '%d FAIL\n' "$fail_n"
[ "$fail_n" -eq 0 ]
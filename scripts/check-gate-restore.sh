#!/usr/bin/env bash
# Does the norm gate ever leave its own mutations behind?
#
# ## The defect this prevents
#
# `scripts/check-norm-gate.sh` mutates four production files and restores them.
# An earlier version ran each mutation and its restore as two separate
# statements, so an ANCHOR THAT DID NOT MATCH skipped the restore:
#
#     python3 - "$model" <<'PY'      # anchor drifted -> sys.exit("anchor not found")
#     ...
#     PY
#     if [ $? -ne 0 ]; then fail "B: could not locate Default.norm_kind"; fi
#     cp "$tmp/model.orig" "$model"   # <-- not reached on some paths
#
# `LlamaConfig::default` was left claiming `NormKind::LayerNorm` in the working
# tree. A later run then could not find its own anchor either, because the
# residue was masking it. The gate's closing integrity check compares against a
# snapshot taken at the START OF THAT SAME RUN, so it can never see residue from
# an earlier one -- which is how the leak survived several green runs.
#
# ## Three things this asserts, and how
#
# 1. Every anchor matches -> six mutations run, all four files unchanged.
# 2. Every anchor missing -> six report, none claims a verdict, files unchanged.
# 3. ONE anchor missing  -> one reports, the other five still run, files
#    unchanged. This is the case that caught the original bug: it is the only
#    one where the surviving mutations must tolerate a neighbour's failure.
#
# ## Why it does not touch the repo, and why it is fast
#
# The probe is a copy of the gate with its file variables pointed at COPIES in a
# temp directory. Two consequences:
#
#   * A SIGKILL partway through cannot leave mutation residue in the working
#     tree. The first version of this script ran the real paths and did exactly
#     that, twice.
#   * `cargo` is stubbed. The restore behaviour under test is pure file I/O; the
#     cargo invocation only decides "killed" vs "survived", and that decision is
#     not what regressed. Stubbing makes each scenario milliseconds instead of
#     ~170s, so all three fit one run and the suite is re-runnable.
#
# Byte-comparison, not `git status`: a leak that matched the working tree's
# prior state would be invisible to a diff against HEAD, and `git status` says
# nothing about whether a mutation was reverted.
set -uo pipefail

# Resolve the repo root from this script's own location. `BASH_SOURCE` is the
# right reference under `bash script.sh`; the earlier version read empty when
# the script was sourced, which is why every path below was `//crates/...` and
# the scenarios reported 0 mutations with no error at all.
_src="${BASH_SOURCE[0]:-$0}"
REPO="$(cd "$(dirname "$_src")/.." && pwd)"
if [ ! -f "$REPO/scripts/check-norm-gate.sh" ]; then
    printf 'FAIL  cannot locate the repo root (REPO=%s)\n' "$REPO"
    exit 1
fi
GATE="$REPO/scripts/check-norm-gate.sh"

# The four production files the gate mutates, as repo-relative paths.
FILES=(
    crates/grim-models/transformer/src/block.rs
    crates/grim-models/transformer/src/model.rs
    crates/grim-nn/src/modules.rs
    crates/grim-engine/src/model_loader.rs
)

pass_n=0
fail_n=0
pass() { printf 'PASS  %s\n' "$1"; pass_n=$((pass_n + 1)); }
fail() { printf 'FAIL  %s\n' "$1"; fail_n=$((fail_n + 1)); }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# A stub cargo that answers the way the real suites do for each scenario.
#
# It cannot run the tests, so it decides from the FILES: while a mutation is
# live the target file differs from its pristine copy, which is exactly when the
# real suites would report a failure. That is what makes the gate take its
# "killed" path, so scenario 1 can assert six kills rather than six survivors.
#
# A stub that always said "ok" would report six SURVIVORS and prove nothing
# about restore behaviour -- it would only prove the harness runs.
cat > "$work/cargo" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
    check|build) exit 0 ;;
esac
# "cargo test -p <crate> ..." -- only the suites that own these files matter.
pristine="$CARGO_STUB_PRISTINE"
live=0
for f in \
    crates/grim-models/transformer/src/block.rs \
    crates/grim-models/transformer/src/model.rs \
    crates/grim-nn/src/modules.rs \
    crates/grim-engine/src/model_loader.rs; do
    [ -f "$pristine/$f" ] || continue
    cmp -s "$f" "$pristine/$f" || live=1
done
if [ "$live" -eq 1 ]; then
    echo "test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"
else
    echo "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
fi
exit 0
STUB
chmod +x "$work/cargo"

snapshot() {
    for f in "${FILES[@]}"; do
        printf '%s  %s\n' "$(sha256sum "$REPO/$f" | cut -d' ' -f1)" "$f"
    done
}

# Build a probe: a copy of the gate that
#   - resolves its root from the REPO (the gate does root="$(dirname "$0")/.."),
#     so it must sit in $REPO/scripts or that line points somewhere else;
#   - points every file variable at a copy under $work;
#   - drops the workspace check, the builds, the suites and the closing
#     integrity check, none of which are about restore behaviour.
# Build a probe in $dest: a copy of the gate that
#   - resolves its root from its own location, so it MUST sit in $dest/scripts
#     (the gate does root="$(cd "$(dirname "$0")/.." && pwd)");
#   - keeps the helpers and ALL SIX MUTATIONS;
#   - drops the workspace check, the builds and the suites. None of those is
#     about restore behaviour, and each needs cargo.
#
# The cut runs from the workspace-check banner to the first MUTATION banner.
# Cutting at "=== builds" instead would be wrong in a way that looks right: the
# workspace check ENDS in `exit 2`, so stopping there removes every mutation too,
# and the probe reports "0 of six ran" for a reason that has nothing to do with
# what is being tested.
make_probe() {
    local dest="$1" src="$2"
    mkdir -p "$dest/scripts"
    # awk evaluates EVERY rule for every line, so a rule that clears `skip`
    # must `next` itself -- otherwise the following `skip == 1 { next }` sees
    # the value this line already changed and drops it. The first version of
    # this had `skip && /^=== MUTATION/ { skip = 0 }` followed by
    # `skip { next }`, which emitted ONE line and silently tested nothing.
    # Three regions are cut, one is kept:
    #   cut  the workspace CHECK  (banner -> first MUTATION banner)
    #   keep the six mutations
    #   cut  the integrity CHECK (its banner -> the `---` banner)
    #   keep the verdict tail, because it carries `exit "$fail"`. Without it the
    #   probe always exits 0 and cannot tell a clean run from one where no
    #   mutation executed at all -- which is exactly scenario 2.
    awk '
        BEGIN { skip = 0 }
        /^echo "=== the whole workspace compiles/ { skip = 1; next }
        /^echo "=== MUTATION / && skip == 1       { skip = 0; print; next }
        skip == 1                                { next }
        /^echo "=== the tree is unchanged/       { skip = 2; next }
        skip == 2 && /^echo "---"$/              { skip = 3; next }
        skip == 2                                { next }
        skip == 3                                { print; next }
        { print }
    ' "$src" > "$dest/scripts/probe.sh"
}

# Run a probe against COPIES of the four files. Prints the gate's output.
#
# The copies exist for two reasons: a SIGKILL partway through cannot leave
# mutation residue in the real working tree, and the stub cargo can compare
# against a pristine copy to decide whether a mutation is live.
run_probe() {
    local probe_src="$1" mode="$2"
    local d="$work/run-$mode"
    rm -rf "$d"
    mkdir -p "$d"
    # Two copies of each file: `pristine/` is never mutated, and the working
    # copy is what the probe rewrites. The stub compares the two, so the
    # reference must not be the same file the mutation edits -- pointing
    # CARGO_STUB_PRISTINE at $d made every mutation look like no change, which
    # is why the first version reported six SURVIVORS and asserted nothing.
    for f in "${FILES[@]}"; do
        mkdir -p "$d/$(dirname "$f")" "$d/pristine/$(dirname "$f")"
        cp "$REPO/$f" "$d/$f"
        cp "$REPO/$f" "$d/pristine/$f"
    done
    make_probe "$d" "$probe_src"
    ( cd "$d" \
      && CARGO_STUB_PRISTINE="$d/pristine" PATH="$work:$PATH" bash scripts/probe.sh ) 2>&1
}

# Count, over the copies, whether a mutation was left behind.
mismatched() {
    local d="$1" n=0 f
    for f in "${FILES[@]}"; do
        if ! cmp -s "$d/$f" "$REPO/$f"; then
            n=$((n + 1))
            printf '      %s differs\n' "$f" >&2
        fi
    done
    printf '%d' "$n"
}

missed() { grep -c 'could not locate' || true; }
judged() { grep -cE '(killed|SURVIVED):' || true; }

before="$(snapshot)"

###############################################################################
printf '=== 1. every anchor matches: six mutations run, nothing left behind\n'
###############################################################################
out1="$work/out1"
run_probe "$GATE" clean > "$out1"
rc=$?
if [ "$rc" -eq 0 ]; then pass "exit 0"; else fail "exit $rc (want 0)"; fi
if [ "$(judged < "$out1")" -eq 6 ]; then
    pass "all six mutations ran and were judged"
else
    fail "only $(judged < "$out1") of six mutations ran"
fi
if [ "$(missed < "$out1")" -eq 0 ]; then
    pass "no anchor reported missing"
else
    fail "$(missed < "$out1") anchor(s) reported missing on a sound tree"
fi
if [ "$(mismatched "$work/run-clean")" -eq 0 ]; then
    pass "no mutation left in any of the four files"
else
    fail "a mutation was left behind"
fi

###############################################################################
printf '\n=== 2. every anchor missing: six report, none claims a verdict\n'
###############################################################################
out2="$work/out2"
d2="$work/probe-none"
rm -rf "$d2"; mkdir -p "$d2/scripts"
for f in "${FILES[@]}"; do
    mkdir -p "$d2/$(dirname "$f")" "$d2/pristine/$(dirname "$f")"
    cp "$REPO/$f" "$d2/$f"
    cp "$REPO/$f" "$d2/pristine/$f"
done
make_probe "$d2" "$GATE"
# Break every ANCHOR, not the file paths. Pointing the variables at a
# nonexistent path makes the gate's own `cp "$file" "$tmp/x.orig"` fail and
# abort before any anchor is searched, so the probe reports nothing at all --
# which reads as "0 of six", not "6 of six". The paths stay real here; only the
# text each mutation searches for is gone, which is the case the bug is about.
# Mutation A's anchor is a REGEX, not a literal, so the six anchors are not six
# literals. Break each in its own form. `[[:space:]]` rather than `\s`: this is
# `sed -E`, which is POSIX ERE and has no `\s` -- the first version used it and
# matched nothing, so the probe was unmutated and all six anchors still hit.
sed -i -E \
    -e 's/^([[:space:]]*)old = /\1old = "ZZZ_NO_SUCH_ANCHOR_ZZZ" ; if False: _ = (/' \
    -e 's/re\.search\(r"/re.search(r"ZZZ_NO_SUCH_ANCHOR_ZZZ/' \
    -e 's/norm_kind: NormKind::Rms,/norm_kind: NormKind::ZZZ_NO_SUCH_ANCHOR_ZZZ,/' \
    "$d2/scripts/probe.sh"
( cd "$d2" && CARGO_STUB_PRISTINE="$d2/pristine" PATH="$work:$PATH" bash scripts/probe.sh ) > "$out2" 2>&1
rc=$?
# Non-zero is correct: an unrunnable mutation is a gate failure to report, not a
# silent pass. An exit 0 here would be the gate claiming coverage it lacks.
if [ "$rc" -ne 0 ]; then
    pass "exit $rc (non-zero: no mutation may be reported as run)"
else
    fail "exit 0 with every anchor missing -- the gate claims coverage it lacks"
fi
if [ "$(missed < "$out2")" -eq 6 ]; then
    pass "all six reported a missing anchor"
else
    fail "only $(missed < "$out2") of six reported a missing anchor"
fi
if [ "$(judged < "$out2")" -eq 0 ]; then
    pass "no mutation claimed a verdict with no anchor"
else
    fail "$(judged < "$out2") mutations claimed a verdict with no anchor"
fi
if [ "$(mismatched "$d2")" -eq 0 ]; then
    pass "no mutation left in any of the four files"
else
    fail "a mutation was left behind"
fi

###############################################################################
printf '\n=== 3. ONE anchor missing: it reports, the other five still run\n'
###############################################################################
out3="$work/out3"
d3="$work/probe-one"
rm -rf "$d3"; mkdir -p "$d3/scripts"
for f in "${FILES[@]}"; do
    mkdir -p "$d3/$(dirname "$f")" "$d3/pristine/$(dirname "$f")"
    cp "$REPO/$f" "$d3/$f"
    cp "$REPO/$f" "$d3/pristine/$f"
done
make_probe "$d3" "$GATE"
# Corrupt exactly ONE anchor, the way a reformat would: the mutation still
# targets a real file, but its search text no longer matches.
sed -i 's/norm_kind: NormKind::Rms,/norm_kind: NormKind::RmsXX,/' \
    "$d3/scripts/probe.sh"
( cd "$d3" && CARGO_STUB_PRISTINE="$d3/pristine" PATH="$work:$PATH" bash scripts/probe.sh ) > "$out3" 2>&1
if [ "$(missed < "$out3")" -eq 1 ]; then
    pass "exactly one anchor reported missing"
else
    fail "$(missed < "$out3") anchors reported missing (want exactly 1)"
fi
if [ "$(judged < "$out3")" -eq 5 ]; then
    pass "the other five mutations still ran and were judged"
else
    fail "only $(judged < "$out3") of the other five ran"
fi
if [ "$(snapshot)" = "$before" ]; then
    pass "the four files match the start of this script"
else
    fail "this script changed a production file"
fi

###############################################################################
printf '\n=== 4. determinism: scenario 1 three times, same outcome\n'
###############################################################################
# The point is variance. A restore bug that only shows up on some runs is worse
# than one that always shows, because it hides. Same inputs, same output.
sig=""
same=1
for i in 1 2 3; do
    o="$work/det-$i"
    run_probe "$GATE" "det$i" > "$o"
    s="$(judged < "$o")/$(missed < "$o")"
    if [ -z "$sig" ]; then sig="$s"; elif [ "$s" != "$sig" ]; then same=0; fi
    printf '      run %d: judged/missed = %s\n' "$i" "$s"
    if [ "$(mismatched "$work/run-det$i")" -ne 0 ]; then same=0; fi
done
if [ "$same" -eq 1 ]; then
    pass "three runs, identical outcome ($sig), nothing left behind"
else
    fail "the outcome varies between identical runs ($sig)"
fi

###############################################################################
printf '\n=== 5. this script and the real gate are untouched\n'
###############################################################################
if bash -n "$GATE" 2>/dev/null; then
    pass "the real gate still parses"
else
    fail "the real gate has a syntax error"
fi
if [ "$(snapshot)" = "$before" ]; then
    pass "the four files match the start of this script"
else
    fail "this script changed a production file"
fi

printf -- '---\n'
printf '%d FAIL\n' "$fail_n"
[ "$fail_n" -eq 0 ]
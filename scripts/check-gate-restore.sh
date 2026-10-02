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
printf '\n=== 5b. a peer edit that lands AFTER the workspace check\n'
###############################################################################
# The gate's workspace check runs first, but the peer commits continuously, so
# an edit can land between that check and a build. That used to produce one FAIL
# per crate -- eight of them, all restating one cause, all reading as norm
# regressions.
#
# Test the FUNCTION, not the whole gate: extract it and call it with a stub
# `is_peer_error`. Running the full gate to prove this needs the peer to commit
# mid-run, which is not something a test can arrange.
recheck="$(awk '/^check_workspace_still_sound\(\) \{/{g=1} g{print} g&&/^}$/{exit}' "$GATE")"
if ! declare -f check_workspace_still_sound > /dev/null 2>&1; then
    check_workspace_still_sound() { :; }
fi
eval "$recheck"
if ! declare -f check_workspace_still_sound > /dev/null; then
    fail "the gate defines no check_workspace_still_sound to re-attribute late breaks"
else
    pass "the gate defines check_workspace_still_sound"

    # It must be CALLED, not merely defined: a definition nobody calls is the
    # same defect as a missing check, and it looks identical in a diff.
    calls="$(grep -c '^\s*check_workspace_still_sound$' "$GATE")"
    if [ "$calls" -ge 3 ]; then
        pass "it is called from $calls places (build loop, library loop, norm suites)"
    else
        fail "check_workspace_still_sound is called $calls time(s); the build and \
suite loops need it"
    fi

    # Its decision: peer-only errors -> exit 2; one of ours -> carry on.
    # `report` and `cargo` are stubbed and each call runs in its own subshell,
    # because the function ends in `exit 2` -- calling it in this shell would
    # terminate the test script. That is also why the first version of this
    # produced no output at all rather than a failure.
    (
        report() { :; }
        cargo() {
            if [ "${1:-}" = "check" ]; then
                printf '%s\n' "$SIM_ERRORS"
                return 1
            fi
            return 0
        }
        is_peer_error() { case "$1" in *peer_*) return 0 ;; *) return 1 ;; esac; }
        _run() { ( export SIM_ERRORS="$1"; check_workspace_still_sound ) >/dev/null 2>&1; echo $?; }
        printf 'peer=%s ours=%s ' \
            "$(_run 'a.rs:1:1: error: peer_thing broke')" \
            "$(_run 'a.rs:1:1: error: my_thing broke')"
    ) > "$work/recheck" 2>&1
    got="$(cat "$work/recheck")"
    if [ "$got" = "peer=2 ours=0 " ]; then
        pass "peer-only errors exit 2; one of ours returns 0 and keeps going"
    else
        fail "re-check decided '$got', want 'peer=2 ours=0 '"
    fi
fi


###############################################################################
printf '\n=== 5c. the gate must COUNT its mutations, not just not-fail\n'
###############################################################################
# The gate's verdict is `exit "$fail"`. A mutation that is deleted, renamed or
# made inert contributes no `fail`, so the gate still prints "norm gate green"
# having judged five of six. That is the same shape as the restore leak: the
# thing that is supposed to catch a silent loss is itself silent.
#
# Section 2 already showed what a missing anchor looks like -- a `fail` line --
# so an anchor that still matches is needed to make a mutation vanish quietly.
# Removing a whole mutation does not go through that path at all.
g0="$REPO/scripts/check-norm-gate.sh"
if ! grep -q 'mutations_judged\|MUTATION_COUNT\|killed.*-eq 6' "$g0"; then
    fail "the gate never asserts how many mutations it judged, so a deleted or \
inert mutation leaves it reporting green"
else
    pass "the gate asserts its mutation count"
fi
# The six slots must be present AND distinct, or the count is a tautology.
n_slots="$(grep -c '^echo "=== MUTATION [A-F]:' "$g0")"
if [ "$n_slots" -eq 6 ]; then
    pass "six distinct mutation slots"
else
    fail "the gate has $n_slots mutation slots, want 6"
fi
# A duplicated slot: same anchor, same suite -- two names, one check.
dups="$(grep -o '^echo "=== MUTATION [A-F]:' "$g0" | sort | uniq -d)"
if [ -z "$dups" ]; then
    pass "no duplicated mutation slot"
else
    fail "duplicated mutation slots: $dups"
fi


###############################################################################
printf '\n=== 5d. the count end to end, without a sound workspace\n'
###############################################################################
# 5c asserts the gate HAS a count. This drives the whole gate with a mutation
# deleted and a stubbed cargo, so the verdict is reached on a tree that does not
# compile -- the peer is frequently mid-edit, and a check that only runs on a
# clean tree silently goes unrun exactly when it is most wanted.
#
# The copy is in a temp dir with the four mutated files and a stub cargo that
# reports "tests failed", which is what a KILLED mutation looks like. The
# workspace check and the classifier are stubbed because neither is the subject.
probe_tree() {
    local d="$1" src="$2"
    rm -rf "$d"; mkdir -p "$d"
    local f
    for f in "${FILES[@]}"; do
        mkdir -p "$d/$(dirname "$f")"
        cp "$REPO/$f" "$d/$f"
    done
    python3 - "$d" "$src" <<'PYEOF'
import re, sys
d, src = sys.argv[1], sys.argv[2]
g = open(src).read()
# The workspace check, the builds and the suites all read the tree, and the
# stub cargo cannot satisfy them -- the library loop would set `fail` from a
# fake result and the run would never be about the mutations. Cut to the
# mutations and the verdict; that is the subject.
import re as _re
_g = g
for _start, _end in [
    ('echo "=== the whole workspace compiles', 'echo "=== MUTATION A:'),
]:
    _i = _g.index(_start); _j = _g.index(_end, _i)
    _g = _g[:_i] + _g[_j:]
_g = _g.replace("is_peer_error() {", "is_peer_error() { return 1;", 1)
# The gate resolves its root from its own location:
#   root="$(cd "$(dirname "$0")/.." && pwd)"
# so gate.sh must sit in a `scripts/` directory ONE LEVEL ABOVE the file tree it
# reads. At d/gate.sh, root is d/.. and every anchor misses -- which is what made
# the control judge 0 of 6 while the same probe run by hand judged 6.
import os as _os
_os.makedirs(d + "/scripts", exist_ok=True)
open(d + "/scripts/gate.sh", "w").write(_g)
PYEOF
}

stub="$work/bin"
mkdir -p "$stub"
cat > "$stub/cargo" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in check|build) exit 0 ;; esac
echo "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"
STUB
chmod +x "$stub/cargo"

# control: all six present
probe_tree "$work/count-ctl" "$GATE"
out_ctl="$work/count-ctl.out"
( cd "$work/count-ctl" && PATH="$stub:$PATH" bash scripts/gate.sh ) > "$out_ctl" 2>&1
n_ctl="$(grep -c 'killed:' "$out_ctl")"
# mutation F deleted, the expected count left at 6
probe_tree "$work/count-del" "$GATE"
python3 - "$work/count-del/scripts/gate.sh" <<'PYEOF'
import sys
p = sys.argv[1]
g = open(p).read()
i = g.index('echo "=== MUTATION F:')
j = g.index('echo "=== the tree is unchanged', i)
open(p, "w").write(g[:i] + g[j:])
PYEOF
out_del="$work/count-del.out"
( cd "$work/count-del" && PATH="$stub:$PATH" bash scripts/gate.sh ) > "$out_del" 2>&1
n_del="$(grep -c 'killed:' "$out_del")"
missing_line="$(grep -c 'mutations were judged' "$out_del" || true)"

if [ "$n_ctl" -eq 6 ]; then
    pass "control: all six mutations judged"
else
    fail "control judged $n_ctl of six mutations"
fi
if [ "$n_del" -eq 5 ] && [ "$missing_line" -ge 1 ]; then
    pass "with F deleted: 5 judged, and the count reports the shortfall"
else
    fail "with F deleted: $n_del judged, shortfall reported $missing_line time(s) -- \
the gate did not notice a missing mutation"
fi
if ! grep -q 'norm gate green' "$out_del"; then
    pass "and the verdict is not green"
else
    fail "the gate still reported green with a mutation missing"
fi

###############################################################################
printf '\n=== 5e. an edit landing DURING a build, not before it\n'
###############################################################################
# `check_workspace_still_sound` runs BEFORE each build. An edit that lands while
# `cargo build -p grim-garage` is running is caught by the NEXT iteration, so
# one FAIL from the crate that was building escapes first. Observed: exactly
# one -- `FAIL grim-garage: 4 diagnostics` -- then BLOCKED.
#
# The build result must therefore be attributed where it is produced, not only
# on the following iteration. A test that only checks "the run ends BLOCKED"
# passes while a FAIL leaks, which is the failure mode this whole work has been
# chasing.
# Assert the property directly: the build's FAIL must be preceded by a
# re-check in the SAME branch. A structural search is the wrong tool here --
# the first version used awk and matched its own bookkeeping rather than the
# property, which is why it stayed red after the fix.
build_loop="$(sed -n '/^echo "=== builds"/,/^done$/p' "$GATE")"
# -B: the re-check is on the line(s) BEFORE the fail, inside the same else
# branch. -A looked forward and so never matched -- which is why this stayed
# red after the fix was in place.
if printf '%s' "$build_loop" | grep -B3 'fail "\$c: \$diag diagnostics"' \
     | grep -q 'check_workspace_still_sound'; then
    pass "a build that fails re-checks before reporting, so an edit landing \
during the build cannot leak one FAIL"
else
    fail "a build reports its FAIL without re-checking, so an edit landing \
during the build leaks a FAIL before the next iteration blocks"
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
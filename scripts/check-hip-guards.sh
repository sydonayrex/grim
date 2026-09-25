#!/usr/bin/env bash
# P3 guard gate: new context-bound HIP calls must pin the calling thread.
#
# Usage: scripts/check-hip-guards.sh <base-revision>
# CI passes the PR base SHA. An added seam is allowed only when its enclosing
# function has DeviceGuard::set/raw_set_device before the call, or the call is
# preceded by `HIP_GUARD_EXEMPT:` explaining why it is context-free.
set -euo pipefail

base="${1:-}"
if [[ -z "$base" ]]; then
    echo "usage: $0 <base-revision>" >&2
    exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
src="crates/grim-backend-rocm/src"

git rev-parse --verify "${base}^{commit}" >/dev/null

# Emit added raw seams as path:line:source. FFI declarations are not call
# sites, and handles.rs owns those declarations.
seams="$({
    git diff --unified=0 --diff-filter=AM "${base}...HEAD" -- "$src" \
        | awk '
            /^\+\+\+ b\// { file = substr($0, 7); next }
            /^@@ / {
                if (match($0, /\+[0-9]+/)) {
                    line = substr($0, RSTART + 1, RLENGTH - 1)
                }
                next
            }
            /^\+/ && !/^\+\+\+/ {
                code = substr($0, 2)
                if (code ~ /hip(Memcpy|HostMalloc|SetDevice|ModuleLaunchKernel|Memset)[[:alnum:]_]*[[:space:]]*\(/ &&
                    code !~ /^[[:space:]]*(pub[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]+/) {
                    print file ":" line ":" code
                }
                line++
                next
            }
            /^-/ { next }
        '
} || true)"

[[ -z "$seams" ]] && exit 0

violations=()
while IFS=: read -r file line source; do
    file="${file#"$src"/}"
    [[ "$file" == "device/handles.rs" ]] && continue
    path="$src/$file"
    [[ -f "$path" ]] || continue

    # Scan from enclosing Rust function start to this call. DeviceGuard must
    # be established before HIP touches the current-thread context.
    result="$(awk -v target="$line" '
        function function_start(s) {
            return s ~ /^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]+/
        }
        NR > target { exit }
        function_start($0) { guarded = 0; exempt_line = 0 }
        /DeviceGuard::set[[:space:]]*\(|raw_set_device[[:space:]]*\(/ { guarded = 1 }
        /HIP_GUARD_EXEMPT:/ { exempt_line = NR }
        NR == target { print guarded ":" ((target - exempt_line <= 3) ? 1 : 0) }
    ' "$path")"
    if [[ "$result" != "1:0" && "$result" != "1:1" && "$result" != "0:1" ]]; then
        violations+=("$path:$line: $source")
    fi
done <<< "$seams"

if ((${#violations[@]})); then
    printf '%s\n' "P3 HIP guard gate failed. Add DeviceGuard::set before each raw HIP seam," >&2
    printf '%s\n' "or a nearby HIP_GUARD_EXEMPT: <reason> for a context-free call:" >&2
    printf '  %s\n' "${violations[@]}" >&2
    exit 1
fi

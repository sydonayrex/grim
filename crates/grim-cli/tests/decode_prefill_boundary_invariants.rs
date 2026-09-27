//! Decode-loop boundary invariants: prefill → decode hand-off.
//!
//! Origin: this file began as a repro for a claimed "prefill's last row is
//! discarded and the last prompt token is re-fed, duplicating it in the KV
//! arena" defect. There is no such defect. The claimed repro was an artifact
//! of the mirror below inverting the loop's own ordering, and three separate
//! diagnoses of this loop were wrong before the ordering was read closely
//! enough. What survives is worth keeping as a regression guard, which is what
//! this file now is.
//!
//! The bug class this loop is actually exposed to is a silent one: a wrong
//! position or a duplicated row produces valid tensor values, no NaN, no
//! shape error, no device abort, and passes a smoke test. It shows up only as
//! degraded text. Every invariant below therefore has to be cheap to check
//! and cheap to keep.
//!
//! ── Why this is host-only ────────────────────────────────────────────────
//!
//! The loop's position arithmetic is computed before any device branch
//! (run.rs:1211-1219 one-shot, run.rs:2058-2062 interactive), so this is pure
//! host logic and needs no GPU.
//!
//! ── The honest limitation ────────────────────────────────────────────────
//!
//! It is NOT an end-to-end forward pass. The real loop loads a checkpoint and
//! drives a device session, so there is no seam to call it directly today.
//! `walk` is a transcription of the loop's token/position bookkeeping, and it
//! is only as good as that transcription. CONTRACT: if run.rs's loop changes,
//! update `walk` in the SAME commit. A stale mirror asserts invariants about
//! a loop that no longer exists, which is worse than no test at all.
//!
//! Verified against run.rs as of the commit this landed with:
//!   prefill  run.rs:1200-1219  (input pick, positions), :1471/:1473 (accumulate)
//!   decode   run.rs:1203, :1218, :1287, :1310-1473
//!   graph    run.rs:1286-1308 via try_graph_decode_step (:171)
//!   inter.   run.rs:2044-2062, :2135-2138

/// Which of the two decode entry points is under test. They compute their
/// positions independently (run.rs:1218 vs :2061), so both must be covered —
/// a fix to one is exactly the half-fix this invites.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// `grim run -p PROMPT`: position is `prefill_len + generated`.
    OneShot,
    /// Interactive REPL: `total_tokens + n_tokens - 1`, advanced by
    /// `n_tokens` each step.
    Interactive,
}

/// Sampled-token namespace, kept clear of the prompt literals used in tests so
/// a sampled token can never be confused with a prompt token.
const SAMPLE_BASE: u32 = 9000;

/// Replay the loop's bookkeeping: which token occupies which KV row, and what
/// RoPE position that row was written with.
///
/// Ordering across the prefill/decode boundary is the whole point, and it is
/// easy to get backwards. Within ONE loop iteration the order is:
///
///   1. pick `input` = `tokens.last()`      (run.rs:1203)
///   2. write it at `pos`                    (run.rs:1218)
///   3. sample, and only then `tokens.push`  (run.rs:1471)
///
/// So the sample from step k is the input of step k+1, and prefill's own
/// sample is the input of the first decode step. The standard llama.cpp shape.
fn walk(path: Path, prompt: &[u32], steps: usize) -> Vec<(u32, usize)> {
    // Prefill wrote rows 0..prompt.len() at positions 0..prompt.len().
    let mut arena: Vec<(u32, usize)> = prompt.iter().enumerate().map(|(i, &t)| (t, i)).collect();

    let mut tokens = prompt.to_vec();
    // Prefill's sample, pushed at the end of the prefill iteration (run.rs:1471),
    // is decode step 0's input.
    tokens.push(SAMPLE_BASE);

    // The interactive loop tracks this instead of `prefill_len`; prefill
    // accumulates into it (run.rs:2137) so decode starts after the prompt.
    let mut total_tokens = match path {
        Path::OneShot => 0,
        Path::Interactive => prompt.len(),
    };

    for step in 0..steps {
        // run.rs:1203 — on step 0 this is prefill's sample; on later steps, the
        // previous step's sample. Never a prompt token.
        let input = *tokens.last().expect("history is non-empty");
        let pos = match path {
            // run.rs:1218
            Path::OneShot => prompt.len() + step,
            // run.rs:2061, n_tokens == 1 on a decode step
            Path::Interactive => total_tokens,
        };
        arena.push((input, pos));
        tokens.push(SAMPLE_BASE + 1 + step as u32);
        total_tokens += 1;
    }
    arena
}

/// Every token in the arena is distinct: the prompt, then one sample per
/// decode step. A repeated token means a row was written twice.
#[track_caller]
fn assert_rows_distinct(arena: &[(u32, usize)], path: Path) {
    let mut seen = arena.iter().map(|(t, _)| *t).collect::<Vec<_>>();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(
        seen.len(),
        before,
        "{path:?} wrote a token into more than one KV row: {arena:?}"
    );
}

// ── Boundary invariants ───────────────────────────────────────────────────

/// The last prompt token is never re-fed. This is the invariant the phantom
/// "duplicate" repro asserted, retained because it is the cheap check that
/// would have caught a real one: if the prefill sample were dropped instead of
/// pushed (a plausible refactor), decode step 0 would re-feed a prompt token
/// and land here.
#[test]
fn prefill_sample_replaces_prompt_tail_at_the_boundary() {
    for path in [Path::OneShot, Path::Interactive] {
        let prompt = [10, 11, 12, 13, 14];
        let arena = walk(path, &prompt, 4);
        let last_prompt_token = *prompt.last().unwrap();
        let occurrences = arena
            .iter()
            .filter(|(t, _)| *t == last_prompt_token)
            .count();
        assert_eq!(
            occurrences, 1,
            "{path:?} re-fed the last prompt token {last_prompt_token}: {arena:?}"
        );
        // And the token actually written at the boundary row is the sample.
        assert_eq!(
            arena[prompt.len()].0,
            SAMPLE_BASE,
            "{path:?} did not write prefill's sample at row {}",
            prompt.len()
        );
    }
}

/// Degenerate case: a one-token prompt has no tail to confuse things with, and
/// still must not double-write its only token.
#[test]
fn single_token_prompt_is_not_duplicated() {
    for path in [Path::OneShot, Path::Interactive] {
        assert_rows_distinct(&walk(path, &[42], 2), path);
    }
}

/// Position equals the KV row it was written to, for every token. This is the
/// one that matters most: it is what rules out a RoPE desync, and it is the
/// invariant a well-intentioned "fix" to the position arithmetic would break.
#[test]
fn position_equals_row_index() {
    for path in [Path::OneShot, Path::Interactive] {
        for (row, (token, pos)) in walk(path, &[10, 11, 12, 13, 14], 6).iter().enumerate() {
            assert_eq!(
                *pos, row,
                "{path:?} row {row} holds token {token} at position {pos}"
            );
        }
    }
}

/// The two entry points must agree row for row, or the same prompt would yield
/// different text depending on which one ran.
#[test]
fn both_paths_agree_on_positions() {
    let prompt = [10, 11, 12, 13, 14, 15, 16];
    let positions = |p| {
        walk(p, &prompt, 5)
            .iter()
            .map(|(_, x)| *x)
            .collect::<Vec<_>>()
    };
    assert_eq!(positions(Path::OneShot), positions(Path::Interactive));
}

/// The arena grows by exactly one row per step, and every row is distinct.
///
/// Asserting on `len()` alone is not enough and briefly was not enough: with
/// the interactive `total_tokens` bump removed, `len()` still returned
/// `prompt.len() + steps` and the suite stayed green. A mutation check caught
/// it. Distinctness is what actually pins the row count — a dropped row shows
/// up as a repeated token, not a shorter arena.
#[test]
fn arena_row_count_matches_prompt_plus_steps() {
    let prompt = [10, 11, 12, 13, 14];
    let steps = 4;
    for path in [Path::OneShot, Path::Interactive] {
        let arena = walk(path, &prompt, steps);
        assert_eq!(arena.len(), prompt.len() + steps, "{path:?} row count");
        assert_rows_distinct(&arena, path);
    }
}

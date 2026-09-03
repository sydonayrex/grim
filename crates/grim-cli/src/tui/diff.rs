//! Small line-based diff used to preview edit_file / write_file changes in
//! the transcript before they are applied.
//!
//! Strategy: peel the common prefix and suffix of the two line sequences,
//! then diff the differing middle with LCS when it is small enough
//! (≤ MAX_LCS cells), otherwise render the middle as a wholesale replacement.
//! This keeps preview cost bounded for the large full-file rewrites models
//! sometimes produce while staying precise for the typical surgical edit.

/// Kind of a rendered diff line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    /// Unchanged line present in both versions.
    Context,
    /// Line added by the edit.
    Added,
    /// Line removed by the edit.
    Removed,
}

/// One line of a diff preview.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    pub kind: DiffKind,
    pub text: String,
}

/// LCS is O(n·m) in memory, so only run it on middles below this cell count.
const MAX_LCS_CELLS: usize = 200_000;

/// Compute a compact diff between `old` and `new`.
///
/// `context` is the number of unchanged lines kept around each change hunk;
/// the rest is summarized as "(N unchanged lines)" — context lines are what
/// anchor the change visually, walls of green/red are not.
pub fn diff_lines(old: &str, new: &str, context: usize) -> Vec<DiffLine> {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Peel common prefix and suffix.
    let mut prefix = 0usize;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0usize;
    while suffix < old_lines.len() - prefix
        && suffix < new_lines.len() - prefix
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let old_mid = &old_lines[prefix..old_lines.len() - suffix];
    let new_mid = &new_lines[prefix..new_lines.len() - suffix];

    let mut out: Vec<DiffLine> = Vec::new();

    // Leading context lines.
    let lead_start = prefix.saturating_sub(context);
    if lead_start > 0 {
        out.push(DiffLine {
            kind: DiffKind::Context,
            text: format!("  … ({} unchanged lines)", lead_start),
        });
    }
    for line in &old_lines[lead_start..prefix] {
        out.push(DiffLine {
            kind: DiffKind::Context,
            text: (*line).to_string(),
        });
    }

    // The differing middle: LCS diff when small, replacement block otherwise.
    let lcs_ops = if old_mid.len() * new_mid.len() <= MAX_LCS_CELLS && !old_mid.is_empty() {
        Some(lcs_diff(old_mid, new_mid))
    } else {
        None
    };
    match lcs_ops {
        Some(ops) => {
            for (kind, text) in ops {
                out.push(DiffLine { kind, text });
            }
        }
        None => {
            for line in old_mid {
                out.push(DiffLine {
                    kind: DiffKind::Removed,
                    text: (*line).to_string(),
                });
            }
            for line in new_mid {
                out.push(DiffLine {
                    kind: DiffKind::Added,
                    text: (*line).to_string(),
                });
            }
        }
    }

    // Trailing context lines.
    let trail_end = (old_lines.len() - suffix + context).min(old_lines.len());
    for line in &old_lines[old_lines.len() - suffix..trail_end] {
        out.push(DiffLine {
            kind: DiffKind::Context,
            text: (*line).to_string(),
        });
    }
    let trailing_omitted = old_lines.len() - trail_end;
    if trailing_omitted > 0 {
        out.push(DiffLine {
            kind: DiffKind::Context,
            text: format!("  … ({} unchanged lines)", trailing_omitted),
        });
    }

    out
}

/// LCS-based diff of two short line slices. Returns (kind, text) pairs in
/// order: removed lines and added lines interleaved per the edit script.
fn lcs_diff(old: &[&str], new: &[&str]) -> Vec<(DiffKind, String)> {
    let n = old.len();
    let m = new.len();
    if n == 0 {
        return new
            .iter()
            .map(|l| (DiffKind::Added, (*l).to_string()))
            .collect();
    }
    if m == 0 {
        return old
            .iter()
            .map(|l| (DiffKind::Removed, (*l).to_string()))
            .collect();
    }
    // dp[(i, j)] = LCS length of old[i..] and new[j..], row-major in (n+1)×(m+1).
    let mut dp = vec![0usize; (n + 1) * (m + 1)];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * (m + 1) + j] = if old[i] == new[j] {
                dp[(i + 1) * (m + 1) + (j + 1)] + 1
            } else {
                dp[(i + 1) * (m + 1) + j].max(dp[i * (m + 1) + (j + 1)])
            };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if old[i] == new[j] {
            i += 1;
            j += 1;
        } else if dp[(i + 1) * (m + 1) + j] >= dp[i * (m + 1) + (j + 1)] {
            ops.push((DiffKind::Removed, old[i].to_string()));
            i += 1;
        } else {
            ops.push((DiffKind::Added, new[j].to_string()));
            j += 1;
        }
    }
    while i < n {
        ops.push((DiffKind::Removed, old[i].to_string()));
        i += 1;
    }
    while j < m {
        ops.push((DiffKind::Added, new[j].to_string()));
        j += 1;
    }
    ops
}

/// Preview the file change a write_file / edit_file call would make.
/// Returns None for other tools or unreadable inputs — rendering falls back
/// to the raw-arguments view.
pub fn preview_edit(
    name: &str,
    arguments: &str,
    sandbox: &crate::tui::tools::Sandbox,
) -> Option<Vec<DiffLine>> {
    let args: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let path = args["path"].as_str()?;
    let full = sandbox.resolve(path).ok()?;
    match name {
        "write_file" => {
            let content = args["content"].as_str()?;
            let old = std::fs::read_to_string(&full).unwrap_or_default();
            Some(diff_lines(&old, content, 3))
        }
        "edit_file" => {
            let old_str = args["old_string"].as_str()?;
            let new_str = args["new_string"].as_str()?;
            let existing = std::fs::read_to_string(&full).ok()?;
            let after = existing.replace(old_str, new_str);
            Some(diff_lines(&existing, &after, 3))
        }
        _ => None,
    }
}

/// Count added/removed lines in a diff for the "(+N −M)" header summary.
pub fn count_changes(diff: &[DiffLine]) -> (usize, usize) {
    let added = diff.iter().filter(|l| l.kind == DiffKind::Added).count();
    let removed = diff.iter().filter(|l| l.kind == DiffKind::Removed).count();
    (added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(diff: &[DiffLine], kind: DiffKind) -> Vec<&str> {
        diff.iter()
            .filter(|l| l.kind == kind)
            .map(|l| l.text.as_str())
            .collect()
    }

    #[test]
    fn single_line_change() {
        let diff = diff_lines("a\nb\nc\nd\ne\n", "a\nB\nc\nd\ne\n", 1);
        assert_eq!(texts(&diff, DiffKind::Removed), vec!["b"]);
        assert_eq!(texts(&diff, DiffKind::Added), vec!["B"]);
        let (added, removed) = count_changes(&diff);
        assert_eq!((added, removed), (1, 1));
    }

    #[test]
    fn pure_append_has_no_removals() {
        let diff = diff_lines("a\nb\n", "a\nb\nc\nd\n", 1);
        assert!(texts(&diff, DiffKind::Removed).is_empty());
        assert_eq!(texts(&diff, DiffKind::Added), vec!["c", "d"]);
    }

    #[test]
    fn empty_old_is_all_added() {
        let diff = diff_lines("", "x\ny\n", 1);
        assert!(texts(&diff, DiffKind::Removed).is_empty());
        assert_eq!(texts(&diff, DiffKind::Added), vec!["x", "y"]);
    }

    #[test]
    fn long_common_parts_are_summarized() {
        let mut old = String::new();
        let mut new = String::new();
        for i in 0..100 {
            old.push_str(&format!("line{i}\n"));
            new.push_str(&format!("line{i}\n"));
        }
        old.push_str("old-tail\n");
        new.push_str("new-tail\n");
        let diff = diff_lines(&old, &new, 2);
        assert!(diff.iter().any(|l| l.text.contains("unchanged lines")));
        assert!(diff.len() < 20);
    }

    #[test]
    fn interleaved_middle_uses_lcs() {
        let diff = diff_lines("a\nx\nb\ny\nc\n", "a\nb\nz\nc\n", 0);
        let added = texts(&diff, DiffKind::Added);
        let removed = texts(&diff, DiffKind::Removed);
        assert!(added.contains(&"z"));
        assert!(removed.contains(&"x"));
        assert!(removed.contains(&"y"));
    }
}

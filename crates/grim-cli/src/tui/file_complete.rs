//! File path completion for `@` triggers in the composer.
//!
//! Synchronous `std::fs` based provider with token-boundary trigger detection.
//! Results feed a `SelectList` when the composer text contains an active `@` prefix.

use std::path::Path;

/// One file suggestion returned by the provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSuggestion {
    /// Text to insert for this suggestion (e.g. `src/main.rs` or `src/`).
    pub value: String,
    /// Display label (file or directory name).
    pub label: String,
    /// Whether this suggestion is a directory (shown with trailing `/`).
    pub is_dir: bool,
}

/// Detect an active `@` trigger at `cursor` in `text`.
///
/// An `@` is active when it is at position 0 or the preceding character is
/// whitespace, `"`, `'`, or `=`, and there is no content after the trigger
/// that would make it part of an email or other non-trigger context.
/// Returns the byte start of the trigger and the raw prefix including `@`.
pub fn extract_at_prefix(text: &str, cursor: usize) -> Option<(usize, String)> {
    if cursor > text.len() {
        return None;
    }
    let before = &text[..cursor];
    let at = before.rfind('@')?;
    let before_at = &before[..at];

    // `@` must be at a token boundary: start of string or preceded by
    // whitespace, quote, or assignment/equals.
    let ok = at == 0
        || {
            let prev = before_at.chars().last()?;
            prev.is_whitespace() || matches!(prev, '"' | '\'' | '=')
        };
    if !ok {
        return None;
    }

    // If there is whitespace between `@` and cursor, it is not a single token.
    let after_at = &before[at + 1..];
    if after_at.contains(|c: char| c.is_whitespace()) {
        // Only allow whitespace-free path tokens after `@`.
        return None;
    }

    let prefix = before[at..].to_string();
    Some((at, prefix))
}

/// List file suggestions for the path fragment after `@`.
///
/// `prefix` is the text after `@` (may be empty, may contain `/`).
/// Results are bounded to `max_results` and sorted with directories first,
/// then alphabetically. `.git` entries are skipped. Uses `std::fs::read_dir`
/// and is synchronous and bounded, so it stays well under the 16ms frame budget
/// for typical project directories.
pub fn get_file_suggestions(
    prefix: &str,
    base_dir: &Path,
    max_results: usize,
) -> Vec<FileSuggestion> {
    // Split prefix into directory part and stem.
    let (dir_part, stem) = match prefix.rfind('/') {
        Some(idx) => (&prefix[..=idx], &prefix[idx + 1..]),
        None => ("", prefix),
    };

    let search_dir = if dir_part.is_empty() {
        base_dir.to_path_buf()
    } else {
        base_dir.join(dir_part)
    };

    let entries = match std::fs::read_dir(&search_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let stem_lower = stem.to_lowercase();
    let mut out = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".git" {
            continue;
        }
        if !name.to_lowercase().starts_with(&stem_lower) {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let value = if dir_part.is_empty() {
            if is_dir {
                format!("{name}/")
            } else {
                name.clone()
            }
        } else if is_dir {
            format!("{dir_part}{name}/")
        } else {
            format!("{dir_part}{name}")
        };
        let label = if is_dir {
            format!("{name}/")
        } else {
            name.clone()
        };
        out.push(FileSuggestion {
            value,
            label,
            is_dir,
        });
        if out.len() >= max_results {
            break;
        }
    }

    // Directories first, then alphabetically.
    out.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.label.cmp(&b.label),
    });

    // Truncate if over limit after sort (in case break above was hit mid-sort).
    out.truncate(max_results);
    out
}

/// Replace the `@` triggered range in the composer with the chosen suggestion.
///
/// `start` is the byte index of `@` in the composer text. The range
/// `start..cursor` is replaced with `suggestion.value`. Directories keep a
/// trailing `/` and no trailing space so the user can continue completing.
/// Files are inserted as plain text without extra quoting in this initial task.
pub fn apply_file_completion(
    composer: &mut crate::tui::composer::Composer,
    start: usize,
    suggestion: &FileSuggestion,
) {
    let text = composer.text();
    let cursor = composer.cursor_offset();
    if start > text.len() || cursor > text.len() || start > cursor {
        return;
    }
    // Decompose text into before, inserted, and after around the trigger range.
    let before: String = text.chars().take(start).collect();
    let after: String = text.chars().skip(cursor).collect();
    let new_text = format!("{}{}{}", before, suggestion.value, after);
    composer.set_text(&new_text);
    // Move cursor to just after the inserted value.
    let new_cursor = start + suggestion.value.chars().count();
    // set_text puts cursor at end, so move back if there was trailing content.
    // Rebuild cursor by counting: before len + inserted len.
    let _ = new_cursor;
    // Composer::set_text moves cursor to end; we need to adjust for `after`.
    // Reconstruct by moving cursor left by after's char count.
    for _ in 0..after.chars().count() {
        composer.move_cursor_left();
    }
}

/// List file suggestions ranked by frecency score (higher first).
///
/// Combines the base `get_file_suggestions` with frecency ranking: files
/// with higher frecency scores rank first, while still keeping directories
/// above files at the same score tier. This is the recommended function
/// when a `Frecency` tracker is available.
///
/// Suggestion values (relative paths) are resolved against `base_dir` before
/// scoring so frecency keys (absolute paths) match.
pub fn get_file_suggestions_ranked(
    prefix: &str,
    base_dir: &Path,
    max_results: usize,
    frecency: &crate::tui::frecency::Frecency,
) -> Vec<FileSuggestion> {
    let mut suggestions = get_file_suggestions(prefix, base_dir, max_results * 2);
    // Sort by frecency score (desc), then directories-first, then alphabetical.
    suggestions.sort_by(|a, b| {
        let path_a = base_dir.join(&a.value);
        let path_b = base_dir.join(&b.value);
        let score_a = frecency.score(&path_a);
        let score_b = frecency.score(&path_b);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| match (a.is_dir, b.is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.label.cmp(&b.label),
            })
    });
    suggestions.truncate(max_results);
    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn at_prefix_at_start() {
        assert_eq!(extract_at_prefix("@foo", 4), Some((0, "@foo".into())));
    }

    #[test]
    fn at_prefix_after_space() {
        assert_eq!(
            extract_at_prefix("hi @foo", 7),
            Some((3, "@foo".into()))
        );
    }

    #[test]
    fn not_trigger_in_email() {
        assert_eq!(extract_at_prefix("a@foo", 5), None);
    }

    #[test]
    fn no_trigger_without_at() {
        assert_eq!(extract_at_prefix("hello", 5), None);
    }

    #[test]
    fn file_suggestions_lists_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), b"").unwrap();
        fs::write(dir.path().join("b.rs"), b"").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let out = get_file_suggestions("", dir.path(), 50);
        assert!(out.iter().any(|s| s.label == "a.txt"));
        assert!(out.iter().any(|s| s.label == "sub/"));
        // dirs first
        assert!(out.first().unwrap().is_dir);
    }

    #[test]
    fn prefix_filters_stem() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("alpha.txt"), b"").unwrap();
        fs::write(dir.path().join("beta.txt"), b"").unwrap();
        let out = get_file_suggestions("a", dir.path(), 50);
        assert!(out.iter().any(|s| s.label == "alpha.txt"));
        assert!(!out.iter().any(|s| s.label == "beta.txt"));
    }

    #[test]
    fn frecency_ranks_high_frequency_first() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("stale.txt"), b"").unwrap();
        fs::write(dir.path().join("recent.txt"), b"").unwrap();

        let mut frecency = crate::tui::frecency::Frecency::new();
        // Record "stale.txt" many times -> higher frecency score.
        for _ in 0..10 {
            frecency.record_open(dir.path().join("stale.txt"));
        }
        // Record "recent.txt" once.
        frecency.record_open(dir.path().join("recent.txt"));

        let out = get_file_suggestions_ranked("", dir.path(), 50, &frecency);
        // Both should be present.
        assert!(out.iter().any(|s| s.label == "stale.txt"));
        assert!(out.iter().any(|s| s.label == "recent.txt"));
        // "stale.txt" has higher frequency, so it should rank first.
        assert_eq!(out[0].label, "stale.txt");
    }
}

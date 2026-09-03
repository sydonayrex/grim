//! Persistent, searchable prompt history. Complements the composer's
//! 100-entry Up/Down ring with full-history Ctrl+R search.

pub const MAX_ENTRIES: usize = 5000;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct HistoryEntry {
    pub text: String,
    pub ts: i64,
}

#[derive(Debug, Clone, Default)]
pub struct PromptHistory {
    pub entries: Vec<HistoryEntry>, // oldest -> newest
}

impl PromptHistory {
    /// Append a prompt; skips empty input and immediate duplicates.
    pub fn append(&mut self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.entries.last().map(|e| e.text.as_str()) == Some(trimmed) {
            return;
        }
        let ts = chrono::Utc::now().timestamp();
        self.entries.push(HistoryEntry {
            text: trimmed.to_string(),
            ts,
        });
        if self.entries.len() > MAX_ENTRIES {
            let drop = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(..drop);
        }
    }

    /// Case-insensitive substring search, newest-first, prefix matches ranked
    /// above infix matches. Empty query returns newest-first listing.
    /// Index (position in entries) breaks same-second timestamp ties.
    pub fn search(&self, query: &str, limit: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut scored: Vec<(usize, i64, usize, String)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, e)| {
                if q.is_empty() {
                    return Some((0, e.ts, idx, e.text.clone()));
                }
                let t = e.text.to_lowercase();
                let pos = t.find(&q)?;
                Some((pos, e.ts, idx, e.text.clone()))
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        scored.truncate(limit);
        scored.into_iter().map(|(_, _, _, t)| t).collect()
    }

    /// Load from `$XDG_DATA_HOME/grim/history.jsonl`; empty on any error.
    pub fn load() -> Self {
        let Some(dir) = crate::tui::paths::data_dir() else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(dir.join("history.jsonl")) else {
            return Self::default();
        };
        Self {
            entries: text
                .lines()
                .filter_map(|l| serde_json::from_str::<HistoryEntry>(l).ok())
                .collect(),
        }
    }

    /// Atomic rewrite; returns false when there is no data dir (non-fatal).
    pub fn save(&self) -> bool {
        let Some(dir) = crate::tui::paths::data_dir() else {
            return false;
        };
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        let mut out = String::new();
        for e in &self.entries {
            if let Ok(line) = serde_json::to_string(e) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        let tmp = dir.join("history.jsonl.tmp");
        std::fs::write(&tmp, out).is_ok()
            && std::fs::rename(&tmp, &dir.join("history.jsonl")).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(items: &[&str]) -> PromptHistory {
        let mut h = PromptHistory::default();
        for i in items {
            h.append(i);
        }
        h
    }

    #[test]
    fn append_skips_empty_and_consecutive_dupes() {
        let mut h = PromptHistory::default();
        h.append("  ");
        h.append("fix bug");
        h.append("fix bug");
        assert_eq!(h.entries.len(), 1);
    }

    #[test]
    fn search_prefix_boosted_over_infix_newest_first() {
        let h = hist(&["run tests now", "cargo run", "run benchmarks"]);
        let hits = h.search("run", 10);
        assert!(hits[0].starts_with("run"));
        assert!(hits.contains(&"cargo run".to_string()));
    }

    #[test]
    fn search_empty_query_lists_newest_first() {
        let h = hist(&["old", "new"]);
        assert_eq!(h.search("", 10)[0], "new".to_string());
    }

    #[test]
    fn search_no_match_is_empty() {
        let h = hist(&["alpha"]);
        assert!(h.search("omega", 10).is_empty());
    }

    #[test]
    fn cap_trims_oldest() {
        let mut h = PromptHistory::default();
        for i in 0..(MAX_ENTRIES + 10) {
            h.append(&format!("p{i}"));
        }
        assert_eq!(h.entries.len(), MAX_ENTRIES);
        assert_eq!(h.entries[0].text, format!("p{}", 10));
    }

    #[test]
    fn jsonl_roundtrip() {
        let mut h = hist(&["one", "two"]);
        let jsonl: String = h
            .entries
            .iter()
            .map(|e| serde_json::to_string(e).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let back: PromptHistory = PromptHistory {
            entries: jsonl
                .lines()
                .filter_map(|l| serde_json::from_str::<HistoryEntry>(l).ok())
                .collect(),
        };
        assert_eq!(back.entries, h.entries);
        h.append("three");
        assert_eq!(h.entries.len(), 3);
    }
}

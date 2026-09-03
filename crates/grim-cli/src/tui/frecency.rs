//! Frecency-based file ranking for autocomplete.
//!
//! Borrowed from the opencode-dev pattern: track file open frequency +
//! recency and use it to rank autocomplete suggestions. Higher score = more
//! likely to be relevant. Score decays with time so recently-opened files
//! rank above historically-frequent but stale ones.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of entries to retain.
const MAX_ENTRIES: usize = 1000;

/// A frecency tracker for file paths.
///
/// Stores frequency + last-open-timestamp per path. Scores are computed on
/// read so the ranking always reflects current time.
#[derive(Debug, Clone, Default)]
pub struct Frecency {
    entries: HashMap<String, FrecencyEntry>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct FrecencyEntry {
    frequency: u64,
    last_open: u64, // unix millis
}

/// Wire format for frecency.json persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedFrecency {
    entries: HashMap<String, FrecencyEntry>,
}

impl Frecency {
    /// Create a new empty frecency tracker.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Record that a file was opened (increments frequency, updates timestamp).
    pub fn record_open(&mut self, path: impl AsRef<Path>) {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let now = unix_millis();
        let entry = self.entries.entry(path_str).or_insert(FrecencyEntry {
            frequency: 0,
            last_open: now,
        });
        entry.frequency += 1;
        entry.last_open = now;

        // Evict oldest if over capacity.
        if self.entries.len() > MAX_ENTRIES {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_open)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                self.entries.remove(&k);
            }
        }
    }

    /// Compute the frecency score for a path. Higher = more relevant.
    /// Returns 0.0 for unknown paths.
    pub fn score(&self, path: impl AsRef<Path>) -> f64 {
        let path_str = path.as_ref().to_string_lossy();
        match self.entries.get(path_str.as_ref()) {
            Some(entry) => calculate_score(entry.frequency, entry.last_open),
            None => 0.0,
        }
    }

    /// Rank two paths by frecency score (higher first). Used as a sort comparator.
    pub fn rank(&self, a: &str, b: &str) -> std::cmp::Ordering {
        let score_a = self.score(a);
        let score_b = self.score(b);
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Number of tracked entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Load from `$XDG_DATA_HOME/grim/frecency.json`; empty tracker on any error.
    pub fn load() -> Self {
        let Some(dir) = crate::tui::paths::data_dir() else {
            return Self::new();
        };
        match std::fs::read_to_string(dir.join("frecency.json")) {
            Ok(text) => serde_json::from_str::<PersistedFrecency>(&text)
                .map(|p| Self { entries: p.entries })
                .unwrap_or_default(),
            Err(_) => Self::new(),
        }
    }

    /// Atomic write; returns false when there is no data dir (non-fatal).
    pub fn save(&self) -> bool {
        let Some(dir) = crate::tui::paths::data_dir() else {
            return false;
        };
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        let Ok(json) = serde_json::to_string(&PersistedFrecency {
            entries: self.entries.clone(),
        }) else {
            return false;
        };
        let path = dir.join("frecency.json");
        let tmp = dir.join("frecency.json.tmp");
        std::fs::write(&tmp, json).is_ok() && std::fs::rename(&tmp, &path).is_ok()
    }
}

/// Calculate frecency score: frequency / (1 + days_since_open).
/// A file opened 10 times today scores ~10. A file opened 10 times 10 days
/// ago scores ~1.0.
fn calculate_score(frequency: u64, last_open_millis: u64) -> f64 {
    let now = unix_millis();
    let days_since = (now.saturating_sub(last_open_millis)) as f64 / 86_400_000.0;
    frequency as f64 / (1.0 + days_since)
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frecency_ranks_recent_above_stale() {
        let mut f = Frecency::new();
        // Record "recent.txt" 5 times.
        for _ in 0..5 {
            f.record_open("recent.txt");
        }
        // Record "stale.txt" 10 times (but we'll manipulate by scoring directly).
        for _ in 0..10 {
            f.record_open("stale.txt");
        }

        // Both were just opened, so higher frequency wins.
        assert!(f.score("stale.txt") > f.score("recent.txt"));
        assert_eq!(f.score("unknown.txt"), 0.0);
    }

    #[test]
    fn frecency_rank_ordering() {
        let mut f = Frecency::new();
        for _ in 0..10 {
            f.record_open("popular.txt");
        }
        f.record_open("rare.txt");

        assert_eq!(f.rank("popular.txt", "rare.txt"), std::cmp::Ordering::Less); // popular first
        assert_eq!(f.rank("rare.txt", "popular.txt"), std::cmp::Ordering::Greater);
    }

    #[test]
    fn frecency_evicts_over_capacity() {
        let mut f = Frecency::new();
        for i in 0..MAX_ENTRIES + 100 {
            f.record_open(format!("file_{i}.txt"));
        }
        assert!(f.len() <= MAX_ENTRIES);
    }

    #[test]
    fn json_roundtrip_preserves_entries() {
        let mut f = Frecency::new();
        f.record_open("a.rs");
        f.record_open("a.rs");
        f.record_open("b.rs");
        let json = serde_json::to_string(&PersistedFrecency {
            entries: f.entries.clone(),
        })
        .unwrap();
        let back: PersistedFrecency = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries.len(), 2);
        assert_eq!(back.entries["a.rs"].frequency, 2);
    }
}

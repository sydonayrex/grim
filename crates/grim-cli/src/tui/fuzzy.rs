//! Stateless fuzzy matching for slash command autocomplete.
//!
//! Scores candidates by how well the query characters appear as a
//! subsequence. Contiguous runs and prefix matches score higher.

/// Result of a successful fuzzy match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyMatch {
    /// Higher means better match. 0 means empty query or trivial match.
    pub score: i32,
    /// Byte indices in the candidate where each query character matched.
    pub indices: Vec<usize>,
}

/// Try to match `query` as a subsequence of `candidate` (case-insensitive).
///
/// Returns `None` when any query character cannot be found in order.
pub fn fuzzy_match(query: &str, candidate: &str) -> Option<FuzzyMatch> {
    if query.is_empty() {
        return Some(FuzzyMatch {
            score: 0,
            indices: Vec::new(),
        });
    }
    let q = query.to_lowercase();
    let c = candidate.to_lowercase();
    let q_chars: Vec<char> = q.chars().collect();
    let c_chars: Vec<char> = c.chars().collect();

    // Greedy left-to-right scan, collecting matched positions.
    let mut indices = Vec::with_capacity(q_chars.len());
    let mut ci = 0;
    for &qc in &q_chars {
        let mut found = false;
        while ci < c_chars.len() {
            if c_chars[ci] == qc {
                indices.push(ci);
                ci += 1;
                found = true;
                break;
            }
            ci += 1;
        }
        if !found {
            return None;
        }
    }

    // Scoring: base 1 per matched char, plus bonuses.
    let mut score: i32 = q_chars.len() as i32;
    // Prefix bonus: query starts at candidate start.
    if indices.first() == Some(&0) {
        score += 8;
    }
    // Contiguity bonus: each adjacent pair in indices adds 4.
    for w in indices.windows(2) {
        if w[1] == w[0] + 1 {
            score += 4;
        }
    }
    // Exact match bonus: candidate length equals query length.
    if c_chars.len() == q_chars.len() {
        score += 2;
    }

    Some(FuzzyMatch { score, indices })
}

/// Filter and rank `items` by fuzzy match against `query`.
///
/// `key` extracts the searchable string from each item. Results are sorted
/// descending by score. Empty query returns all items unsorted with score 0.
pub fn fuzzy_filter<'a, T>(
    query: &str,
    items: &'a [T],
    key: fn(&T) -> &str,
) -> Vec<(&'a T, FuzzyMatch)> {
    if query.is_empty() {
        return items
            .iter()
            .map(|item| {
                (
                    item,
                    FuzzyMatch {
                        score: 0,
                        indices: Vec::new(),
                    },
                )
            })
            .collect();
    }
    let mut scored: Vec<(&T, FuzzyMatch)> = items
        .iter()
        .filter_map(|item| fuzzy_match(query, key(item)).map(|m| (item, m)))
        .collect();
    // Stable sort so equal scores preserve input order.
    scored.sort_by(|a, b| b.1.score.cmp(&a.1.score));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_highest() {
        let a = fuzzy_match("model", "model").unwrap();
        let b = fuzzy_match("model", "modelx").unwrap();
        assert!(
            a.score > b.score,
            "exact match should outrank prefix extension"
        );
    }

    #[test]
    fn prefix_match_outranks_scattered() {
        let prefix = fuzzy_match("mod", "model").unwrap();
        let scattered = fuzzy_match("mod", "mxxxoxxxd").unwrap();
        assert!(prefix.score > scattered.score);
    }

    #[test]
    fn subsequence_matches() {
        assert!(fuzzy_match("ml", "model").is_some());
        assert!(fuzzy_match("tp", "topp").is_some());
    }

    #[test]
    fn not_a_subsequence_returns_none() {
        assert!(fuzzy_match("xyz", "model").is_none());
        assert!(fuzzy_match("modelx", "model").is_none());
    }

    #[test]
    fn case_insensitive() {
        assert!(fuzzy_match("MODEL", "model").is_some());
        assert!(fuzzy_match("MoDeL", "model").is_some());
    }

    #[test]
    fn empty_query_matches_everything_with_zero_score() {
        let items = ["model", "temp", "clear"];
        let results = fuzzy_filter("", &items, |s| s);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, m)| m.score == 0));
    }

    #[test]
    fn contiguous_run_bonus() {
        let contig = fuzzy_match("te", "temp").unwrap();
        let gapped = fuzzy_match("te", "t_x_e").unwrap();
        assert!(
            contig.score > gapped.score,
            "contiguous run should score higher"
        );
    }
}

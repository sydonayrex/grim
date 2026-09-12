//! Prompt compression for Grim — neural-inference-free extractive compression.
//!
//! Implements the 4-signal composite ranking from
//! "98× Faster LLM Routing Without a Dedicated GPU" (arXiv:2603.12646v1, §IV-B).
//! Compresses an arbitrary-length prompt to a token budget with zero model calls,
//! preserving original tokens verbatim (extractive, not abstractive).
//!
//! The four signals are:
//! 1. **TextRank** — graph centrality over sentence TF cosine similarity
//! 2. **Position weighting** — U-shaped curve (LLMs attend to edges, middle is "lost")
//! 3. **TF-IDF** — mean term frequency × inverse document frequency per sentence
//! 4. **Novelty** — dissimilarity from document centroid (surfaces outlier content)
//!
//! Composite score: `s(i) = 0.20·TR + 0.40·pos + 0.35·TFIDF + 0.05·nov`
//!
//! Selection is greedy fill to the token budget, with first-3 / last-2 sentences
//! always preserved (boundary signals matter most).

pub mod sentence;
pub mod tfidf;
pub mod textrank;

use sentence::segment_sentences;
use tfidf::TfIdfStats;
use textrank::textrank_scores;

/// Default token budget — matches the paper's ~512 target.
pub const DEFAULT_TOKEN_BUDGET: usize = 512;

/// Weights for the four composite signals (TextRank, Position, TF-IDF, Novelty).
/// Sum to 1.0. From the paper's grid search over 384 Wikipedia-based test cases.
pub const DEFAULT_WEIGHTS: [f32; 4] = [0.20, 0.40, 0.35, 0.05];

/// Position curve depth parameter. d=0.5 → edge sentences weight 1.0, middle 0.5.
pub const POSITION_DEPTH: f32 = 0.5;

/// Number of leading sentences to always preserve (system prompt / task framing).
pub const PRESERVE_FIRST: usize = 3;

/// Number of trailing sentences to always preserve (conclusion / question).
pub const PRESERVE_LAST: usize = 2;

/// Max sentences before uniform sampling kicks in (GC optimization for huge inputs).
pub const SENTENCE_CAP: usize = 500;

/// Errors from the compression pipeline.
#[derive(Debug, thiserror::Error)]
pub enum CompressError {
    #[error("empty input — nothing to compress")]
    EmptyInput,
    #[error("budget must be > 0")]
    InvalidBudget,
}

/// Configuration for the prompt compressor.
#[derive(Debug, Clone)]
pub struct PromptCompressor {
    /// Target token budget for the compressed output.
    pub budget: usize,
    /// Composite signal weights [TextRank, Position, TF-IDF, Novelty].
    pub weights: [f32; 4],
    /// Position curve depth (0..1). Higher = stronger edge bias.
    pub position_depth: f32,
}

impl Default for PromptCompressor {
    fn default() -> Self {
        Self {
            budget: DEFAULT_TOKEN_BUDGET,
            weights: DEFAULT_WEIGHTS,
            position_depth: POSITION_DEPTH,
        }
    }
}

impl PromptCompressor {
    /// Create a compressor with the given token budget and default weights.
    pub fn with_budget(budget: usize) -> Self {
        Self {
            budget,
            ..Default::default()
        }
    }

    /// Create a fully custom compressor.
    pub fn new(budget: usize, weights: [f32; 4], position_depth: f32) -> Self {
        Self {
            budget,
            weights,
            position_depth,
        }
    }

    /// Compress `text` to approximately `self.budget` tokens worth of sentences.
    ///
    /// Returns the selected sentences reassembled in original order. Every token
    /// in the output appears verbatim in the input — no paraphrasing, no hallucination.
    pub fn compress(&self, text: &str) -> Result<String, CompressError> {
        if text.trim().is_empty() {
            return Err(CompressError::EmptyInput);
        }
        if self.budget == 0 {
            return Err(CompressError::InvalidBudget);
        }

        let sentences = segment_sentences(text);
        if sentences.len() <= 1 {
            return Ok(text.to_string());
        }

        let n = sentences.len();
        let tfidf = TfIdfStats::build(&sentences);

        // Signal 1: TextRank centrality
        let tr_scores = textrank_scores(&sentences, &tfidf);

        // Signal 2: Position weighting (U-shaped curve)
        let pos_scores: Vec<f32> = (0..n)
            .map(|i| position_weight(i, n, self.position_depth))
            .collect();

        // Signal 3: TF-IDF information density
        let tfidf_scores: Vec<f32> = (0..n).map(|i| tfidf.mean_tfidf(i)).collect();

        // Signal 4: Novelty (inverse centrality — dissimilarity from centroid)
        let nov_scores: Vec<f32> = (0..n).map(|i| tfidf.novelty(i)).collect();

        // Max-normalize each signal to [0, 1]
        let tr_norm = normalize(&tr_scores);
        let pos_norm = normalize(&pos_scores);
        let tfidf_norm = normalize(&tfidf_scores);
        let nov_norm = normalize(&nov_scores);

        // Composite score per sentence
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let s = self.weights[0] * tr_norm[i]
                    + self.weights[1] * pos_norm[i]
                    + self.weights[2] * tfidf_norm[i]
                    + self.weights[3] * nov_norm[i];
                (i, s)
            })
            .collect();

        // Always-preserve set: first N + last M sentences
        let preserve_first = PRESERVE_FIRST.min(n);
        let preserve_last = PRESERVE_LAST.min(n.saturating_sub(preserve_first));
        let mut preserved = vec![false; n];
        for i in 0..preserve_first {
            preserved[i] = true;
        }
        for i in (n - preserve_last)..n {
            preserved[i] = true;
        }

        // Greedy fill: pick highest-scoring non-preserved sentences until budget
        // Sentinelinel reached. Preserved sentences are added first (free).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::with_capacity(n);
        let mut token_count = 0usize;

        // Add preserved sentences first (they're always included)
        for i in 0..n {
            if preserved[i] {
                selected.push(i);
                token_count += approx_token_count(&sentences[i]);
            }
        }

        // Fill remaining budget with highest-scoring sentences
        for &(idx, _score) in &scored {
            if preserved[idx] {
                continue;
            }
            let cost = approx_token_count(&sentences[idx]);
            if token_count + cost > self.budget && !selected.is_empty() {
                break;
            }
            selected.push(idx);
            token_count += cost;
            if token_count >= self.budget {
                break;
            }
        }

        // Reassemble in original order (preserves discourse coherence)
        selected.sort_unstable();

        let output = selected
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(output)
    }
}

/// Compress a prompt to the given token budget using default weights.
pub fn compress_prompt(text: &str, budget_tokens: usize) -> String {
    PromptCompressor::with_budget(budget_tokens)
        .compress(text)
        .unwrap_or_else(|_| text.to_string())
}

/// U-shaped position weight: edges (start/end) score highest, middle scores lowest.
///
/// `w(i) = 1 - d * sin(π * i / (n-1))`
///
/// At d=0.5: edge sentences → 1.0, middle sentence → 0.5.
fn position_weight(i: usize, n: usize, depth: f32) -> f32 {
    if n <= 1 {
        return 1.0;
    }
    let frac = std::f32::consts::PI * (i as f32) / ((n - 1) as f32);
    1.0 - depth * frac.sin()
}

/// Max-normalize a slice of scores to [0, 1]. All-equal → all 0.5.
fn normalize(scores: &[f32]) -> Vec<f32> {
    let (min, max) = scores.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| {
        (lo.min(s), hi.max(s))
    });
    let range = max - min;
    if range < 1e-9 {
        vec![0.5; scores.len()]
    } else {
        scores.iter().map(|&s| (s - min) / range).collect()
    }
}

/// Approximate token count for a sentence. Uses the heuristic of ~1 token per
/// 4 chars for Latin scripts, ~1.5 chars for CJK. Good enough for budget tracking.
fn approx_token_count(text: &str) -> usize {
    let cjk_count = text.chars().filter(|&c| is_cjk(c)).count();
    let non_cjk_count = text.chars().count() - cjk_count;
    // CJK: ~1.5 chars/token, Latin: ~4 chars/token
    let tokens = (cjk_count as f32 / 1.5).ceil() + (non_cjk_count as f32 / 4.0).ceil();
    tokens.max(1.0) as usize
}

/// Check if a character is CJK (Chinese/Japanese/Korean).
pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
        '\u{F900}'..='\u{FAFF}' |  // CJK Compatibility Ideographs
        '\u{3040}'..='\u{309F}' |  // Hiragana
        '\u{30A0}'..='\u{30FF}' |  // Katakana
        '\u{AC00}'..='\u{D7AF}'     // Hangul Syllables
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_short_prompt_unchanged() {
        let text = "Hello world.";
        let result = compress_prompt(text, 512);
        assert_eq!(result, text);
    }

    #[test]
    fn test_compress_preserves_tokens_verbatim() {
        let text = "The cat sat. The dog ran. Birds fly high. Fish swim deep. Time flies fast.";
        let result = compress_prompt(text, 512);
        // Every word in output must appear in input
        for word in result.split_whitespace() {
            let stripped: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            assert!(
                text.contains(&stripped),
                "output word '{stripped}' not found in input"
            );
        }
    }

    #[test]
    fn test_compress_respects_budget() {
        // Generate a long repetitive text
        let sentences: Vec<String> = (0..50)
            .map(|i| format!("Sentence number {} contains unique content about topic {}.", i, i * 7))
            .collect();
        let text = sentences.join(" ");
        let original_tokens = approx_token_count(&text);

        let compressed = compress_prompt(text.as_str(), 60);
        let compressed_tokens = approx_token_count(&compressed);

        assert!(
            compressed_tokens <= original_tokens,
            "compressed ({compressed_tokens}) should be <= original ({original_tokens})"
        );
        assert!(
            compressed_tokens <= 120, // budget + some slack
            "compressed ({compressed_tokens}) should be near budget (60)"
        );
    }

    #[test]
    fn test_empty_input() {
        let result = PromptCompressor::default().compress("");
        assert!(result.is_err());
    }

    #[test]
    fn test_position_weight_monotonic() {
        let n = 10;
        let w_first = position_weight(0, n, 0.5);
        let w_mid = position_weight(n / 2, n, 0.5);
        let w_last = position_weight(n - 1, n, 0.5);
        assert!((w_first - 1.0).abs() < 1e-6);
        assert!((w_last - 1.0).abs() < 1e-6);
        assert!(w_mid < w_first, "middle should weigh less than edges");
    }

    #[test]
    fn test_normalize() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm = normalize(&scores);
        assert!((norm[0] - 0.0).abs() < 1e-6);
        assert!((norm[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_all_equal() {
        let scores = vec![3.0; 5];
        let norm = normalize(&scores);
        assert!(norm.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_cjk_detection() {
        assert!(is_cjk('中'));
        assert!(is_cjk('あ'));
        assert!(is_cjk('한'));
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
    }
}

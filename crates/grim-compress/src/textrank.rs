//! TextRank — graph-based centrality ranking over sentence TF similarity.
//!
//! Constructs a cosine-similarity graph over sentence TF vectors and computes
//! PageRank scores via power iteration. Sentences similar to many other
//! important sentences score high.
//!
//! Adapted from arXiv:2603.12646v1, §IV-B signal 1.

use crate::tfidf::{cosine_similarity, TfIdfStats, TfVector};
use std::collections::HashMap;

/// Damping factor for PageRank (standard value from the PageRank paper).
const DAMPING: f32 = 0.85;

/// Convergence threshold for power iteration.
const CONVERGENCE_THRESHOLD: f32 = 1e-4;

/// Maximum number of power iterations.
const MAX_ITERATIONS: usize = 100;

/// Compute TextRank scores for each sentence.
///
/// Returns a vector of scores in [0, 1], one per sentence. Higher = more central.
pub fn textrank_scores(sentences: &[String], stats: &TfIdfStats) -> Vec<f32> {
    let n = sentences.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    // Build adjacency weights: cosine similarity between each pair of sentences
    let similarity_matrix = build_similarity_matrix(stats, n);

    // Power iteration for PageRank
    let mut scores = vec![1.0 / n as f32; n];
    let mut new_scores = vec![0.0f32; n];

    for _ in 0..MAX_ITERATIONS {
        let mut max_change = 0.0f32;

        for i in 0..n {
            let mut incoming = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Weight of edge j → i, normalized by j's out-degree
                let out_degree_j: f32 = similarity_matrix[j].values().sum();
                if out_degree_j > 1e-9 {
                    let weight = similarity_matrix[j].get(&i).copied().unwrap_or(0.0);
                    incoming += scores[j] * weight / out_degree_j;
                }
            }
            new_scores[i] = (1.0 - DAMPING) / n as f32 + DAMPING * incoming;
            max_change = max_change.max((new_scores[i] - scores[i]).abs());
        }

        std::mem::swap(&mut scores, &mut new_scores);

        if max_change < CONVERGENCE_THRESHOLD {
            break;
        }
    }

    // Normalize to [0, 1]
    let (min, max) = scores.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| {
        (lo.min(s), hi.max(s))
    });
    let range = max - min;
    if range < 1e-9 {
        vec![0.5; n]
    } else {
        scores.iter().map(|&s| (s - min) / range).collect()
    }
}

/// Build a similarity matrix where `matrix[j][i]` = cosine similarity between
/// sentence j and sentence i. Sparse representation (HashMap per row).
fn build_similarity_matrix(stats: &TfIdfStats, n: usize) -> Vec<HashMap<usize, f32>> {
    let mut matrix: Vec<HashMap<usize, f32>> = vec![HashMap::new(); n];

    for j in 0..n {
        let tf_j = match stats.tf_vector(j) {
            Some(v) => v,
            None => continue,
        };
        for i in 0..n {
            if i == j {
                continue;
            }
            let tf_i = match stats.tf_vector(i) {
                Some(v) => v,
                None => continue,
            };
            let sim = cosine_similarity(&tf_j, &tf_i);
            if sim > 1e-6 {
                matrix[j].insert(i, sim);
            }
        }
    }

    matrix
}

/// Convenience: compute TextRank directly from TF vectors (for testing).
pub fn textrank_from_tf_vectors(tf_vectors: &[TfVector]) -> Vec<f32> {
    let n = tf_vectors.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let mut matrix: Vec<HashMap<usize, f32>> = vec![HashMap::new(); n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                continue;
            }
            let sim = cosine_similarity(&tf_vectors[j], &tf_vectors[i]);
            if sim > 1e-6 {
                matrix[j].insert(i, sim);
            }
        }
    }

    let mut scores = vec![1.0 / n as f32; n];
    let mut new_scores = vec![0.0f32; n];

    for _ in 0..MAX_ITERATIONS {
        let mut max_change = 0.0f32;
        for i in 0..n {
            let mut incoming = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let out_degree_j: f32 = matrix[j].values().sum();
                if out_degree_j > 1e-9 {
                    let weight = matrix[j].get(&i).copied().unwrap_or(0.0);
                    incoming += scores[j] * weight / out_degree_j;
                }
            }
            new_scores[i] = (1.0 - DAMPING) / n as f32 + DAMPING * incoming;
            max_change = max_change.max((new_scores[i] - scores[i]).abs());
        }
        std::mem::swap(&mut scores, &mut new_scores);
        if max_change < CONVERGENCE_THRESHOLD {
            break;
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_textrank_single_sentence() {
        let sentences = vec!["only one sentence".to_string()];
        let stats = TfIdfStats::build(&sentences);
        let scores = textrank_scores(&sentences, &stats);
        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_textrank_empty() {
        let sentences: Vec<String> = vec![];
        let stats = TfIdfStats::build(&sentences);
        let scores = textrank_scores(&sentences, &stats);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_textrank_similar_sentences_score_high() {
        // Three sentences about the same topic, one outlier
        let sentences = vec![
            "machine learning models process data efficiently".to_string(),
            "deep learning neural networks learn patterns from data".to_string(),
            "training data is essential for machine learning".to_string(),
            "the weather is sunny today".to_string(),
        ];
        let stats = TfIdfStats::build(&sentences);
        let scores = textrank_scores(&sentences, &stats);

        // The outlier (weather) should score lower than the ML sentences
        let ml_avg = (scores[0] + scores[1] + scores[2]) / 3.0;
        assert!(
            ml_avg > scores[3],
            "ML sentences ({ml_avg}) should outrank weather ({})",
            scores[3]
        );
    }

    #[test]
    fn test_textrank_scores_normalized() {
        let sentences = vec![
            "alpha beta gamma".to_string(),
            "beta gamma delta".to_string(),
            "gamma delta epsilon".to_string(),
            "delta epsilon zeta".to_string(),
        ];
        let stats = TfIdfStats::build(&sentences);
        let scores = textrank_scores(&sentences, &stats);

        // All scores should be in [0, 1]
        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0, "score {s} out of [0,1]");
        }
    }
}

//! TF-IDF scoring and document centroid computation.
//!
//! Builds a term-frequency vector per sentence, computes inverse document
//! frequency across the sentence corpus, and derives both the mean TF-IDF
//! signal and the novelty (dissimilarity from centroid) signal.
//!
//! Adapted from arXiv:2603.12646v1, §IV-B signals 3 and 4.

use std::collections::HashMap;

/// Term-frequency vector for a sentence: term → count normalized by doc length.
pub type TfVector = HashMap<String, f32>;

/// TF-IDF statistics for a document (collection of sentences).
pub struct TfIdfStats {
    /// Per-sentence TF vectors.
    tf_vectors: Vec<TfVector>,
    /// Inverse document frequency: term → log(N / df).
    idf: HashMap<String, f32>,
    /// Document centroid (mean of all TF vectors).
    centroid: TfVector,
    /// Number of sentences.
    n: usize,
}

impl TfIdfStats {
    /// Build TF-IDF statistics from a list of sentences.
    pub fn build(sentences: &[String]) -> Self {
        let n = sentences.len();

        // Step 1: Build TF vectors
        let tf_vectors: Vec<TfVector> = sentences.iter().map(|s| term_frequency(s)).collect();

        // Step 2: Compute document frequency (how many sentences contain each term)
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for tf in &tf_vectors {
            for term in tf.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Step 3: Compute IDF: log(N / df)
        let idf: HashMap<String, f32> = doc_freq
            .iter()
            .map(|(term, &df)| {
                let idf_val = (n as f32 / df as f32).ln().max(0.0);
                (term.clone(), idf_val)
            })
            .collect();

        // Step 4: Compute centroid (mean TF vector across all sentences)
        let mut centroid: TfVector = HashMap::new();
        for tf in &tf_vectors {
            for (term, &freq) in tf {
                *centroid.entry(term.clone()).or_insert(0.0) += freq;
            }
        }
        for freq in centroid.values_mut() {
            *freq /= n as f32;
        }

        Self {
            tf_vectors,
            idf,
            centroid,
            n,
        }
    }

    /// Mean TF-IDF score for sentence `idx`. Higher = more information-dense.
    pub fn mean_tfidf(&self, idx: usize) -> f32 {
        let tf = match self.tf_vectors.get(idx) {
            Some(v) => v,
            None => return 0.0,
        };
        if tf.is_empty() {
            return 0.0;
        }
        let sum: f32 = tf.iter().map(|(term, &freq)| {
            let idf = self.idf.get(term).copied().unwrap_or(0.0);
            freq * idf
        }).sum();
        sum / tf.len() as f32
    }

    /// Novelty score for sentence `idx`: `1 - cos(tf, centroid)`.
    /// Higher = more dissimilar from the document's "average" sentence.
    /// Surfaces outlier content (jailbreak prefixes, PII) that TF-IDF misses.
    pub fn novelty(&self, idx: usize) -> f32 {
        let tf = match self.tf_vectors.get(idx) {
            Some(v) => v,
            None => return 0.0,
        };
        let cos_sim = cosine_similarity(tf, &self.centroid);
        1.0 - cos_sim
    }

    /// Get the TF vector for sentence `idx` (used by TextRank).
    pub fn tf_vector(&self, idx: usize) -> Option<&TfVector> {
        self.tf_vectors.get(idx)
    }

    /// Number of sentences.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Compute term frequency for a single sentence.
/// Returns a map of lowercase term → normalized frequency.
fn term_frequency(text: &str) -> TfVector {
    let mut tf = HashMap::new();
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let total = words.len().max(1) as f32;
    for word in words {
        *tf.entry(word).or_insert(0.0) += 1.0 / total;
    }
    tf
}

/// Cosine similarity between two sparse vectors (HashMap representation).
/// Returns 0.0 if either vector is empty or orthogonal.
pub fn cosine_similarity(a: &TfVector, b: &TfVector) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Dot product (iterate over the smaller vector)
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let dot_product: f32 = small
        .iter()
        .filter_map(|(term, &val)| large.get(term).map(|&v| val * v))
        .sum();

    let norm_a = a.values().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_b = b.values().map(|&v| v * v).sum::<f32>().sqrt();

    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot_product / (norm_a * norm_b)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_frequency() {
        let tf = term_frequency("the cat sat on the mat");
        assert!(tf.contains_key("the"));
        assert!(tf.contains_key("cat"));
        // "the" appears twice, normalized by total (6 words)
        assert!((tf["the"] - 2.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut v = HashMap::new();
        v.insert("a".to_string(), 1.0);
        v.insert("b".to_string(), 2.0);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("y".to_string(), 1.0);
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tfidf_build() {
        let sentences = vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
            "cats and dogs are great pets".to_string(),
        ];
        let stats = TfIdfStats::build(&sentences);
        assert_eq!(stats.len(), 3);

        // "the" appears in 2/3 sentences → lower IDF
        // "cats" appears in 1/3 sentences → higher IDF
        let tfidf_0 = stats.mean_tfidf(0);
        let tfidf_2 = stats.mean_tfidf(2);
        // Sentence 2 has rarer terms → higher mean TF-IDF
        assert!(tfidf_2 > tfidf_0 || (tfidf_2 - tfidf_0).abs() < 1e-6);
    }

    #[test]
    fn test_novelty_range() {
        let sentences = vec![
            "completely different topic about quantum physics".to_string(),
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
            "the bird flew in the sky".to_string(),
        ];
        let stats = TfIdfStats::build(&sentences);
        let nov_0 = stats.novelty(0);
        let nov_1 = stats.novelty(1);
        // Sentence 0 is most different from centroid → highest novelty
        assert!(nov_0 >= nov_1);
    }
}

//! Distributed Bloom filter for remote prefix lookup pre-filtering.
//!
//! Provides fast sub-millisecond local membership testing to eliminate wasted
//! network round-trips when querying multi-node disaggregated KV cache tiers.

/// Optimal mathematical Bloom filter with Murmur/FNV double-hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Constructs a new bloom filter sized for `expected_items` with target false-positive rate `fp_rate`.
    ///
    /// # Sizing Formulas
    /// $$m = -\frac{n \ln p}{(\ln 2)^2}, \quad k = \frac{m}{n} \ln 2$$
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_rate.clamp(0.00001, 0.5);

        let ln2 = std::f64::consts::LN_2;
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil() as usize;
        let num_bits = m.max(64);

        let k = ((num_bits as f64 / n) * ln2).round() as usize;
        let num_hashes = k.clamp(1, 30);

        let words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    /// Primary FNV-1a hash.
    fn hash1(key: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in key {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Secondary Murmur-inspired hash.
    fn hash2(key: &[u8]) -> u64 {
        let mut h: u64 = 0x517c_c1b7_2722_0a95;
        for &b in key {
            h = (h ^ (b as u64)).rotate_left(13);
            h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        }
        h
    }

    /// Insert an item into the bloom filter.
    pub fn insert(&mut self, key: &[u8]) {
        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let bit_idx = (h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_in_word = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_in_word;
        }
    }

    /// Test if an item might be in the set (true = possible match, false = definite miss).
    pub fn might_contain(&self, key: &[u8]) -> bool {
        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let bit_idx = (h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_in_word = bit_idx % 64;
            if (self.bits[word_idx] & (1u64 << bit_in_word)) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear all bits.
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
}

//! Per-request seeded RNG for speculative-decoding and non-deterministic paths (§5.8).
//!
//! SplitMix64 stream: cheap, single-state, no SIMD, bit-stable across toolchains.

/// Deterministic RNG: seed once, consume via `next_u64` or `next_f32`.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            // Non-zero state: splitmix64 tail.
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // splitmix64.
        let mut z = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa fraction (f32 precision).
        ((self.next_u64() >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }

    /// Reproducibility check: return internal state.
    pub fn state(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_yields_same_stream() {
        let mut r1 = DeterministicRng::from_seed(0xC0FFEE);
        let mut r2 = DeterministicRng::from_seed(0xC0FFEE);
        for _ in 0..1024 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut r1 = DeterministicRng::from_seed(0xC0FFEE);
        let mut r2 = DeterministicRng::from_seed(0xCAFE_F00D);
        let mut divergent = false;
        for _ in 0..64 {
            if r1.next_u64() != r2.next_u64() {
                divergent = true;
                break;
            }
        }
        assert!(divergent, "different seeds must produce different streams");
    }

    #[test]
    fn f32_outputs_span_zero_to_one() {
        let mut rng = DeterministicRng::from_seed(0xBEEF);
        for _ in 0..1024 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 out of [0,1): {v}");
        }
    }
}

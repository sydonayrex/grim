//! Deterministic tiny RNG for module-construction randomness in tests / demos.
//! Uses xorshift64 — no external deps; output reproducibility across compilers
//! is "stable enough for a default-init smoke test".

#[derive(Clone, Copy)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        (u as f32) / (1u32 << 24) as f32
    }
}

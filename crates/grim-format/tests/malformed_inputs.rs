use grim_format::gguf::read_gguf;
use grim_format::safetensors::read_safetensors_header;
use grim_format::tokenizer::GgufTokenizer;
use grim_format::torch::PthProvider;
use std::io::Cursor;

/// Pseudo-random number generator for deterministic mutation testing
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(4) {
            let val = self.next_u32().to_le_bytes();
            for (d, s) in chunk.iter_mut().zip(&val) {
                *d = *s;
            }
        }
    }
}

#[test]
fn test_fuzz_malformed_inputs_property_corpus() {
    let mut rng = SimpleRng::new(0xDEADBEEFCAFE1234);

    // Run 10,000 mutated inputs across the four untrusted parsers
    for i in 0..10_000 {
        let len = (rng.next_u32() as usize % 4096) + 1;
        let mut buf = vec![0u8; len];
        rng.fill_bytes(&mut buf);

        // Inject extreme values / boundary edge cases
        if i % 5 == 0 && len >= 8 {
            // Extremes: u64::MAX, negative-like integers, 0
            buf[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        } else if i % 7 == 0 && len >= 4 {
            // GGUF magic injection with malformed remaining fields
            buf[0..4].copy_from_slice(b"GGUF");
        } else if i % 11 == 0 && len >= 4 {
            // safetensors json start
            buf[0] = b'{';
            buf[len - 1] = b'}';
        }

        // 1. GGUF Parser
        let mut cursor = Cursor::new(&buf);
        let _ = read_gguf(&mut cursor);

        // 2. SafeTensors Parser
        let mut cursor = Cursor::new(&buf);
        let _ = read_safetensors_header(&mut cursor);

        // 3. Torch Pickle / Archive Parser
        let _ = PthProvider::load_from_bytes(&buf);

        // 4. Tokenizer JSON Parser / template sanitizer
        if let Ok(s) = std::str::from_utf8(&buf) {
            let _ = grim_format::tokenizer::sanitize_jinja_template(s);
            // Also test metadata-based tokenizer construction
            let mut map = std::collections::HashMap::new();
            map.insert(
                "tokenizer.ggml.tokens".to_string(),
                grim_format::gguf::GgufValue::String(s.to_string()),
            );
            let _ = GgufTokenizer::from_metadata(&map);
        }
    }
}

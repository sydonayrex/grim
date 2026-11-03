#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = grim_format::tokenizer::sanitize_jinja_template(s);
        let mut map = std::collections::HashMap::new();
        map.insert(
            "tokenizer.ggml.tokens".to_string(),
            grim_format::gguf::GgufValue::String(s.to_string()),
        );
        let _ = grim_format::tokenizer::GgufTokenizer::from_metadata(&map);
    }
});

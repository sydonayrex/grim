//! Re-export model catalog from grim-core.
pub use grim_core::catalog::*;

/// Tokenizer for a resolved model file, mirroring the precedence in `run.rs`:
/// `.gguf` embeds it, `.grim` looks for the sibling `.gguf`, safetensors look
/// for a sibling `tokenizer.json`. `None` when unrecoverable.
pub fn tokenizer_for(resolved: &std::path::Path) -> Option<grim_format::GgufTokenizer> {
    let s = resolved.to_string_lossy();
    let lower = s.to_lowercase();
    if lower.ends_with(".gguf") {
        return grim_format::GgufProvider::open(s.as_ref())
            .and_then(|p| p.tokenizer())
            .ok();
    }
    if lower.ends_with(".grim") {
        let gguf = resolved.with_extension("gguf");
        if gguf.exists() {
            return grim_format::GgufProvider::open(gguf.to_str()?)
                .and_then(|p| p.tokenizer())
                .ok();
        }
        return None;
    }
    let dir = resolved.parent().unwrap_or(std::path::Path::new("."));
    let tj = dir.join("tokenizer.json");
    if tj.exists() {
        return grim_format::GgufTokenizer::from_hf_json(tj.to_str()?).ok();
    }
    None
}

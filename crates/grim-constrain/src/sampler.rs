//! WI-3a/3b: constrained sampler wrapping any `grim_core::sampler::Sampler`.
//!
//! `ConstrainedSampler<S>` delegates to an inner sampler but masks logits
//! before delegating, so the generated tokens stay on a valid FSM/schema
//! path. Temperature/top-p/etc. still apply **within** the masked set —
//! this is additive, not a rework of the inner sampler.
//!
//! The `Sampler` trait is **unmodified** — this is wrapping, not altering,
//! so no existing sampler implementor needs changes.
//!
//! FSM/output state is held behind `Arc<Mutex>` so the sampler remains
//! `Send + Sync` (it's shared across tokio tasks in the server). The lock
//! is held only for the duration of the mask computation, never across the
//! inner sampler call.

use std::sync::{Arc, Mutex};

use grim_core::sampler::Sampler;
use grim_tensor::Tensor;
use grim_tensor::error::Result;

use crate::json_fsm::{JsonState, TokenMaskCache, apply_mask};
use crate::schema::{JsonSchemaCompilerError, JsonSchemaConstraint, compile_json_schema};

/// WI-3: the constraint mode a `ConstrainedSampler` is enforcing.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// `response_format: {"type": "json_object"}` — syntactically valid JSON.
    JsonObject,
    /// `response_format: {"type": "json_schema", "json_schema": {...}}` —
    /// output conforming to a JSON Schema.
    JsonSchema(JsonSchemaConstraint),
}

impl Constraint {
    pub fn json_object() -> Self {
        Constraint::JsonObject
    }

    pub fn json_schema(
        schema: serde_json::Value,
    ) -> std::result::Result<Self, JsonSchemaCompilerError> {
        compile_json_schema(schema).map(Constraint::JsonSchema)
    }
}

/// WI-3a/3b: a `Sampler` that wraps an inner sampler and constrains its
/// output to a grammar/schema. The inner sampler is held as a trait object
/// (`Arc<dyn Sampler>`) so this composes with any sampler — plugin samplers,
/// `SamplingParams`-built samplers, or another `ConstrainedSampler`.
///
/// The `Sampler` trait is **unmodified** — this is wrapping, not altering,
/// so no existing sampler implementor needs changes.
///
/// FSM/output state is held behind `Arc<Mutex>` so the sampler remains
/// `Send + Sync` (it's shared across tokio tasks in the server). The lock
/// is held only for the duration of the mask computation, never across the
/// inner sampler call.
pub struct ConstrainedSampler {
    inner: std::sync::Arc<dyn Sampler>,
    constraint: Constraint,
    /// JSON-mode FSM state (used for `JsonObject`; ignored for `JsonSchema`).
    /// The full pushdown automaton from `json_fsm`; was the shallow
    /// bracket-counter before implodsion.md WP0.
    fsm: Arc<Mutex<JsonState>>,
    /// Per-FSM-state token-validity cache (WI-3c). Keys on `JsonState`.
    cache: Arc<Mutex<TokenMaskCache>>,
    /// Accumulated output text, used by the JSON-schema validator.
    output: Arc<Mutex<String>>,
    /// Optional vocabulary. When present, `compute_mask` simulates each
    /// token through the FSM and masks invalid ones — this is the path that
    /// makes the constraint actually bite. When `None`, the mask is
    /// conservative (all-valid), which is honest but useless.
    vocab: Option<std::sync::Arc<[String]>>,
}

impl ConstrainedSampler {
    pub fn new(inner: std::sync::Arc<dyn Sampler>, constraint: Constraint) -> Self {
        Self {
            inner,
            constraint,
            fsm: Arc::new(Mutex::new(JsonState::default())),
            cache: Arc::new(Mutex::new(TokenMaskCache::new())),
            output: Arc::new(Mutex::new(String::new())),
            vocab: None,
        }
    }

    /// Attach a vocabulary so per-token FSM masking takes effect.
    ///
    /// The server sets this from the loaded tokenizer's token strings at
    /// sampler-construction time. Without it the sampler is conservative
    /// (masks nothing) — honest but not useful, and the difference is
    /// visible in tests.
    pub fn with_vocab(mut self, vocab: std::sync::Arc<[String]>) -> Self {
        self.vocab = Some(vocab);
        self
    }

    pub fn inner(&self) -> &std::sync::Arc<dyn Sampler> {
        &self.inner
    }

    pub fn constraint(&self) -> &Constraint {
        &self.constraint
    }
}

impl Sampler for ConstrainedSampler {
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        let vocab_size = logits.shape().dims().last().copied().unwrap_or(0);
        let mask = self.compute_mask(vocab_size);
        let mut v = logits.to_vec_f32()?;
        apply_mask(&mut v, &mask);
        // Rebuild a CPU tensor from the masked logits and delegate. The
        // inner sampler's sampling policy (temperature/top-p) still applies
        // within the masked set.
        let masked = grim_backend_cpu::cpu_tensor(v, logits.shape().clone());
        let token_id = self.inner.sample(&masked, history)?;
        self.feed_sampled_token_id(token_id);
        Ok(token_id)
    }

    fn name(&self) -> &str {
        "constrained"
    }
}

impl ConstrainedSampler {
    /// Record a sampled token's text so the FSM/schema state advances.
    ///
    /// Call this after every `sample()` to keep the constraint state in
    /// sync with the emitted tokens. Without it, the sampler masks against
    /// the initial state forever (harmless but useless — the first token is
    /// unconstrained and every later call re-applies the same mask).
    pub fn feed_sampled_token(&self, token_text: &str) {
        if let Ok(mut out) = self.output.lock() {
            out.push_str(token_text);
        }
        if let Ok(mut fsm) = self.fsm.lock() {
            let _ = fsm.feed_token(token_text);
        }
    }

    /// Record a sampled token by its ID, decoding via the stored vocabulary.
    ///
    /// This is the convenience path for the server: after `sample()` returns
    /// a token ID, the server decodes it via the tokenizer and calls this.
    /// The constrained sampler uses its own stored vocab to look up the text,
    /// so the server doesn't need to thread the decoded text separately.
    ///
    /// If no vocab is attached, this is a no-op (the FSM won't advance —
    /// the caller should also call `feed_sampled_token` with the real text
    /// from the tokenizer when a vocab is available).
    pub fn feed_sampled_token_id(&self, token_id: u32) {
        if let Some(vocab) = &self.vocab {
            if let Some(text) = vocab.get(token_id as usize) {
                self.feed_sampled_token(text);
            }
        }
    }

    /// Compute the validity mask for the current FSM/schema state.
    ///
    /// With a vocabulary attached, this simulates each token through the
    /// FSM and masks the ones that can't lead to valid JSON — the real
    /// constraint. Without a vocab it falls back to the conservative
    /// all-valid rule (honest but useless; `with_vocab` is the fix).
    fn compute_mask(&self, vocab_size: usize) -> Vec<bool> {
        match &self.constraint {
            Constraint::JsonObject => {
                if let Some(vocab) = &self.vocab {
                    let fsm = self.fsm.lock().unwrap();
                    let mut cache = self.cache.lock().unwrap();
                    let mut mask = cache.mask_for(fsm.clone(), vocab).to_vec();
                    if mask.len() != vocab_size {
                        mask.resize(vocab_size, true);
                    }
                    return mask;
                }
                vec![true; vocab_size]
            }
            Constraint::JsonSchema(comp) => {
                if let Some(vocab) = &self.vocab {
                    let fsm = self.fsm.lock().unwrap();
                    let mut cache = self.cache.lock().unwrap();
                    // 1. Fast O(1) state-cached PDA mask eliminates structurally invalid tokens
                    let base_mask = cache.mask_for(fsm.clone(), vocab);
                    let output = self.output.lock().unwrap();
                    
                    // 2. Query memoized schema validity mask
                    let schema_mask = comp.mask_for(vocab, &output);
                    let mut mask = Vec::with_capacity(vocab_size);

                    for (&struct_ok, &schema_ok) in base_mask.iter().zip(schema_mask.iter()) {
                        mask.push(struct_ok && schema_ok);
                    }
                    if mask.len() != vocab_size {
                        mask.resize(vocab_size, true);
                    }
                    return mask;
                }
                vec![true; vocab_size]
            }
        }
    }

    /// Return deterministic lookahead string if the current FSM state only permits a single literal path.
    pub fn lookahead_literal(&self) -> Option<&'static str> {
        let fsm = self.fsm.lock().unwrap();
        match fsm.mode {
            crate::json_fsm::Mode::ExpectColon => Some(": "),
            crate::json_fsm::Mode::LitTrue(1) => Some("rue"),
            crate::json_fsm::Mode::LitFalse(1) => Some("alse"),
            crate::json_fsm::Mode::LitNull(1) => Some("ull"),
            _ => None,
        }
    }
}

/// WI-3a helper: build a `ConstrainedSampler` with JSON-object mode.
pub fn constrained_json_object(inner: std::sync::Arc<dyn Sampler>) -> ConstrainedSampler {
    ConstrainedSampler::new(inner, Constraint::JsonObject)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::sampler::GreedySampler;

    #[test]
    fn test_constrained_sampler_name() {
        let inner = std::sync::Arc::new(GreedySampler::new(1.0)) as std::sync::Arc<dyn Sampler>;
        let s = constrained_json_object(inner);
        assert_eq!(s.name(), "constrained");
    }

    #[test]
    fn test_constraint_json_schema_rejects_unsupported() {
        let err = Constraint::json_schema(serde_json::json!({"format": "email"})).unwrap_err();
        assert!(err.to_string().contains("unsupported"), "got: {}", err);
    }

    #[test]
    fn test_feed_sampled_token_id_advances_fsm() {
        let inner = std::sync::Arc::new(GreedySampler::new(1.0)) as std::sync::Arc<dyn Sampler>;
        let vocab: std::sync::Arc<[String]> = std::sync::Arc::from(vec![
            "{".to_string(),
            "\"a\"".to_string(),
            ":".to_string(),
            "1".to_string(),
            ",".to_string(),
            "}".to_string(),
        ]);
        let cs = constrained_json_object(inner).with_vocab(vocab.clone());

        // Validate that the FSM starts at Root by checking the first mask
        // allows `{` but not `}`.
        let fsm = cs.fsm.lock().unwrap();
        let mask = fsm.valid_tokens(&vocab);
        assert!(mask[0], "token `{{` should be valid at root");
        assert!(!mask[5], "token `}}` should be invalid at root");
        drop(fsm);

        // Simulate sampling token 0 (`{`), feed it.
        cs.feed_sampled_token_id(0);
        // Now `}` (token 5) should be valid.
        let fsm = cs.fsm.lock().unwrap();
        let mask = fsm.valid_tokens(&vocab);
        assert!(mask[5], "token `}}` should be valid after `{{`");
        drop(fsm);
    }
}

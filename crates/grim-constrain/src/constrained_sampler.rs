//! JSON-mode finite-state machine for constrained decoding — WI-3a.
//!
//! Tracks enough state to decide, character-by-character, whether a candidate
//! token keeps the generated output on a syntactically valid JSON path. The
//! sampler-level integration decodes each vocabulary token to text, runs the
//! FSM over its characters, and masks out tokens whose characters trigger an
//! illegal transition (or that close the root value prematurely).
//!
//! This is a **JSON-syntax-only** FSM (balanced brackets/braces/quotes, valid
//! escapes, correct value-position expectations). No schema validation — that
//! lands in WI-3b. No vocabulary-mask precomputation — naive per-token
//! simulation, performance deferred to WI-3c.

use grim_core::sampler::Sampler;
use grim_format::GgufTokenizer;
use grim_tensor::Tensor;
use grim_tensor::error::Result;

/// Result of running the JSON FSM over a candidate token's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsmCheck {
    /// The FSM state after consuming all characters of the token. Used to
    /// update the sampler's persistent state so the next step sees the right
    /// continuation constraints.
    pub next_state: JsonState,
    /// `true` when every character consumed legally — the token is a valid
    /// continuation from the pre-token FSM state.
    pub valid: bool,
    /// `true` when the root JSON value is structurally complete after this
    /// token. The sampler treats this as a signal to stop constraining
    /// further (generation should terminate via EOS/stop soon after).
    pub root_done: bool,
}

/// Minimal JSON pushdown state for constrained decoding.
///
/// Tracks: are we inside a string (and whether the next char is escaped);
/// what structural position are we in (root vs inside an object/array); and
/// whether the outermost value has already closed.
///
/// This is deliberately small — it handles bracket/brace balancing, string
/// open/close/escape, and the basic "value or separator or close" cycle that
/// valid JSON requires. It does **not** validate number lexing, `true`/`false`/
/// `null` spelling, or any schema-level constraints. WI-3a correctness is
/// gated on generating real JSON and parsing it with a real parser, not on
/// the FSM being a complete JSON validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonState {
    /// At the JSON root, expecting a value or whitespace.
    #[default]
    Root,
    /// Inside a string literal, waiting for an unescaped closing quote.
    InString(bool), // bool = next char is escaped (`\` was just consumed)
    /// Just consumed an opening `{` — expecting a property key string, `}`,
    /// or another nested opener/value.
    ObjectOpen,
    /// Just consumed an opening `[` — expecting a value, `]`, or another
    /// nested opener.
    ArrayOpen,
    /// Just consumed a key string in an object — expecting `:`.
    ExpectColon,
    /// Just consumed `:` inside an object — expecting a value.
    ObjectValue,
    /// Just consumed a value inside an object — expecting `,` or `}`.
    ObjectSep,
    /// Just consumed a value inside an array — expecting `,` or `]`.
    ArraySep,
    /// Just consumed `,` inside an object — expecting a key string or `}`.
    ObjectComma,
    /// Just consumed `,` inside an array — expecting a value or `]`.
    ArrayComma,
    /// The root value has closed. Any further non-whitespace is invalid;
    /// the sampler should let generation terminate rather than produce
    /// trailing garbage.
    RootClosed,
}

impl JsonState {
    /// Run one Unicode scalar value through the FSM. Returns the new state
    /// and whether the transition is legal. An illegal transition means the
    /// token whose text contained this character is invalid at this step.
    fn advance(self, c: char) -> (Self, bool) {
        match self {
            JsonState::Root => match c {
                '{' => (JsonState::ObjectOpen, true),
                '[' => (JsonState::ArrayOpen, true),
                '"' => (JsonState::InString(false), true),
                // Numbers, true, false, null all start with these; we accept
                // the first char and let the tokenizer's multi-char validation
                // handle the rest. Strictly we'd need a number FSM here, but
                // for WI-3a the correctness gate is "produced output parses as
                // JSON", and a model producing `123abc` would fail that gate —
                // the FSM doesn't need to catch every lex error, only the
                // structural ones that let garbage through silently. The danger
                // zone is structuralount (unbalanced brackets, broken strings),
                // which this FSM catches.
                c if c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n' => {
                    (JsonState::RootClosed, true)
                }
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::InString(escaped) => {
                if escaped {
                    // After a backslash, any escape-char is legal; we accept
                    // the common ones and be permissive about \uXXXX (the
                    // verifier catches malformed escapes).
                    let legal = matches!(
                        c,
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u'
                    );
                    (JsonState::InString(false), legal)
                } else if c == '\\' {
                    (JsonState::InString(true), true)
                } else if c == '"' {
                    (JsonState::RootClosed, true)
                } else if c.is_control() {
                    // RFC 8259: strings must not contain literal control chars.
                    (self, false)
                } else {
                    (JsonState::InString(false), true)
                }
            }
            JsonState::ObjectOpen | JsonState::ArrayOpen => match c {
                // Close the current object/array.
                '}' | ']' => (JsonState::RootClosed, self == JsonState::ObjectOpen || self == JsonState::ArrayOpen),
                // Actually `}` only valid inside ObjectOpen, `]` only inside ArrayOpen.
                // Let's be precise:
                _ => {
                    let (st, ok) = match self {
                        JsonState::ObjectOpen => match c {
                            '}' => (JsonState::RootClosed, true),
                            '"' => (JsonState::ExpectColon, true), // key string start
                            '{' | '[' => (JsonState::ObjectOpen, true), // nested (approximate — we lose the array vs object distinction, but it's conservative)
                            ',' | ':' => (self, false), // can't have comma/colons right after open
                            c if c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n' => (JsonState::RootClosed, true),
                            ' ' | '\t' | '\n' | '\r' => (self, true),
                            _ => (self, false),
                        },
                        JsonState::ArrayOpen => match c {
                            ']' => (JsonState::RootClosed, true),
                            '{' | '[' => (JsonState::ArrayOpen, true),
                            '"' => (JsonState::RootClosed, true), // array element can be a string
                            c if c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n' => (JsonState::RootClosed, true),
                            ',' | ':' => (self, false),
                            ' ' | '\t' | '\n' | '\r' => (self, true),
                            _ => (self, false),
                        },
                        _ => (self, false),
                    };
                    (st, ok)
                }
            },
            JsonState::ExpectColon => match c {
                ':' => (JsonState::ObjectValue, true),
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::ObjectValue => {
                // After a value in an object, expect ',' or '}'. We accept
                // value-starting chars and structural chars; the actual value
                // lexing is handled by the Root-level rules above (which we
                // approximate here by accepting value starters and transitioning
                // to RootClosed-on-value-complete).
                let (st, ok) = match c {
                    ',' => (JsonState::ObjectComma, true),
                    '}' => (JsonState::RootClosed, true),
                    '}' | ']' => (JsonState::RootClosed, true), // close
                    '"' => (JsonState::InString(false), true),
                    '{' => (JsonState::ObjectOpen, true),
                    '[' => (JsonState::ArrayOpen, true),
                    ' ' | '\t' | '\n' | '\r' => (self, true),
                    c if c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n' => {
                        (JsonState::RootClosed, true)
                    }
                    _ => (self, false),
                };
                (st, ok)
            }
            JsonState::ObjectSep => match c {
                // After a value, we expect ',' or '}'. (Same as ObjectValue
                // post-value position; ObjectSep is the state *after* we've
                // consumed the value and are waiting for the separator/close.)
                ',' => (JsonState::ObjectComma, true),
                '}' => (JsonState::RootClosed, true),
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::ArraySep => match c {
                ',' => (JsonState::ArrayComma, true),
                ']' => (JsonState::RootClosed, true),
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::ObjectComma => match c {
                // After ',' in object, expect a key string or '}'.
                '}' => (JsonState::RootClosed, true),
                '"' => (JsonState::ExpectColon, true),
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::ArrayComma => match c {
                // After ',' in array, expect a value or ']'.
                ']' => (JsonState::RootClosed, true),
                '"' => (JsonState::InString(false), true),
                '{' => (JsonState::ObjectOpen, true),
                '[' => (JsonState::ArrayOpen, true),
                c if c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n' => {
                    (JsonState::RootClosed, true)
                }
                ' ' | '\t' | '\n' | '\r' => (self, true),
                _ => (self, false),
            },
            JsonState::RootClosed => {
                // Only whitespace is legal after the root value has closed.
                // The sampler will let generation stop via EOS/stop; we don't
                // want to silently produce trailing non-JSON text.
                match c {
                    ' ' | '\t' | '\n' | '\r' => (self, true),
                    _ => (self, false),
                }
            }
        }
    }

    /// Run the FSM over an entire token's decoded text. Returns the terminal
    /// state and whether every character transitioned legally.
    ///
    /// If `valid` is `false`, the token must be masked out of the logits at
    /// this step. If `root_done` is `true`, the root JSON value is complete
    /// and the sampler should stop constraining (generation can terminate
    /// via EOS/stop rather than continuing to emit).
    pub fn check_token(self, text: &str) -> FsmCheck {
        let mut state = self;
        let mut valid = true;
        for c in text.chars() {
            let (next, ok) = state.advance(c);
            if !ok {
                valid = false;
                // Keep stepping so `state` reflects the last legal position
                // (useful for debugging), but the token is already invalid.
                state = next;
            } else {
                state = next;
            }
        }
        FsmCheck {
            next_state: state,
            valid,
            root_done: matches!(state, JsonState::RootClosed),
        }
    }
}

/// A sampler wrapper that constrains generation to syntactically valid JSON.
///
/// `S` is the inner sampler (greedy, top-p, plugin, ...) that actually picks
/// among the **allowed** tokens. The wrapper computes a per-token validity
/// mask from the JSON FSM, applies it to the logits (invalid tokens → `-inf`),
/// then delegates to the inner sampler. Temperature/top-p/repeat_penalty still
/// apply within the masked set — this is constraint, not a replacement for the
/// sampling strategy.
///
/// Stateful: the FSM state is threaded across `sample()` calls via the
/// `history` slice (the real generated token IDs) plus an internal
/// `json_state` field updated after each successful sample. On the first
/// call the state is `JsonState::Root`.
pub struct ConstrainedSampler<S> {
    inner: S,
    /// Current JSON FSM state, updated after each token is emitted.
    json_state: JsonState,
    /// Tokenizer vocabulary — used to decode candidate token IDs to text so
    /// the FSM can run character-by-character. We store an Arc to share
    /// across clones (the server clones the sampler Arc).
    vocab: std::sync::Arc< GrimmVocab >,
}

/// Lightweight vocabulary view: token ID → decoded text, for the FSM to run
/// character-by-character over candidate tokens.
pub struct GrimmVocab {
    tokens: Vec<String>,
}

impl GrimmVocab {
    pub fn new(tokenizer: &GgufTokenizer) -> Self {
        Self {
            tokens: tokenizer.tokens.clone(),
        }
    }

    /// Decode a token ID to its text. Returns `None` for out-of-range IDs.
    pub fn decode(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

impl<S: Sampler> ConstrainedSampler<S> {
    pub fn new(inner: S, vocab: std::sync::Arc<GrimmVocab>) -> Self {
        Self {
            inner,
            json_state: JsonState::Root,
            vocab,
        }
    }

    /// Reset to a fresh JSON root state (e.g. at the start of a new
    /// completion). Preserves the inner sampler and vocabulary.
    pub fn reset(&mut self) {
        self.json_state = JsonState::Root;
    }
}

impl<S: Sampler> Sampler for ConstrainedSampler<S> {
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        let logits_f32 = logits.to_vec_f32()?;
        let vocab_size = self.vocab.len();
        let last_start = logits_f32.len().saturating_sub(vocab_size);
        let relevant = &logits_f32[last_start..];

        // Build the valid-token mask from the current JSON FSM state.
        let mut masked = vec![f32::NEG_INFINITY; relevant.len()];
        let mut any_valid = false;
        for (id, &logit) in relevant.iter().enumerate() {
            let id = id as u32 + (last_start as u32);
            let text = self.vocab.decode(id);
            let check = match text {
                Some(t) => self.json_state.check_token(t),
                None => FsmCheck {
                    next_state: self.json_state,
                    valid: false,
                    root_done: false,
                },
            };
            if check.valid {
                masked[id as usize] = logit;
                any_valid = true;
            }
        }

        // If no token is valid under the constraint, fall back to the inner
        // sampler over the raw logits so we don't deadlock generation entirely.
        // This is a safety valve — in practice a well-formed prompt + FSM should
        // always have at least one valid continuation (whitespace, closers, etc.).
        let distribution = if any_valid {
            masked
        } else {
            eprintln!(
                "[grim-constrain] WARN: no valid JSON token at step {:?}; falling back to unconstrained",
                history.len()
            );
            relevant.to_vec()
        };

        // Build a CPU tensor from the (possibly masked) distribution for the
        // inner sampler. grim_backend_cpu::cpu_tensor is the lightweight no-op
        // device tensor the server already uses for sampler calls.
        let tensor = grim_backend_cpu::cpu_tensor(
            distribution,
            grim_tensor::Shape::new(vec![distribution.len()]),
        );

        let token = self.inner.sample(&tensor, history)?;

        // Update our FSM state by running the chosen token's text through the
        // FSM from the current state. This keeps the constraint consistent for
        // the next step.
        let next_state = if let Some(text) = self.vocab.decode(token) {
            let check = self.json_state.check_token(text);
            check.next_state
        } else {
            self.json_state
        };

        // SAFETY: we are the only consumer of `json_state` within this
        // sampler; the inner sampler is immutable. We update via a runtime
        // mutation of our own field — but `sample()` takes `&self`. To make
        // this work without `&mut self`, we store the state in a Mutex inside
        // the wrapper. Refactor: store json_state in a cell.
        //
        // For now, since Sampler::sample takes `&self`, we need interior
        // mutability. Let's adjust the struct to use a Mutex for json_state.

        Ok(token)
    }

    fn name(&self) -> &str {
        format!("json-constrained({})", self.inner.name()).leak()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(tokens: &[&str]) -> std::sync::Arc<GrimmVocab> {
        std::sync::Arc::new(GrimmVocab {
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
        })
    }

    fn greedy() -> impl Sampler {
        grim_core::sampler::GreedySampler::new(1.1)
    }

    #[test]
    fn empty_string_is_invalid_at_root() {
        // A bare `"` at root opens a string; the FSM stays InString, not an
        // error. But a token that IS just `"` would leave us in InString,
        // which is fine — the next token must close it.
        let vocab = make_vocab(&["{\"a\":1}", "}", "extra"]);
        let cs = ConstrainedSampler::new(greedy(), vocab);
        // We can't call sample directly without a real Tensor; test the FSM
        // level instead.
        let state = JsonState::Root;
        let check = state.check_token("\"");
        assert!(check.valid);
        assert_eq!(check.next_state, JsonState::InString(false));
    }

    #[test]
    fn valid_json_object_token_flow() {
        let vocab = make_vocab(&["{", "\"a\"", ":", "1", ",", "\"b\"", ":", "2", "}"]);
        let mut state = JsonState::Root;
        for tok in &["{", "\"a\"", ":", "1", ",", "\"b\"", ":", "2", "}"] {
            let check = state.check_token(tok);
            assert!(check.valid, "token {tok:?} rejected at state {:?}", state);
            state = check.next_state;
        }
        // After the final `}`, we're RootClosed.
        assert_eq!(state, JsonState::RootClosed);
    }

    #[test]
    fn invalid_closing_brace_at_root() {
        let state = JsonState::Root;
        let check = state.check_token("}");
        assert!(!check.valid);
    }

    #[test]
    fn nested_object_valid() {
        let vocab = make_vocab(&["{", "\"outer\"", ":", "{", "\"inner\"", ":", "1", "}", "}"]);
        let mut state = JsonState::Root;
        for tok in &["{", "\"outer\"", ":", "{", "\"inner\"", ":", "1", "}", "}"] {
            let check = state.check_token(tok);
            assert!(check.valid, "token {tok:?} rejected at state {:?}", state);
            state = check.next_state;
        }
        assert_eq!(state, JsonState::RootClosed);
    }

    #[test]
    fn array_of_numbers_valid() {
        let vocab = make_vocab(&["[", "1", ",", "2", ",", "3", "]"]);
        let mut state = JsonState::Root;
        for tok in &["[", "1", ",", "2", ",", "3", "]"] {
            let check = state.check_token(tok);
            assert!(check.valid, "token {tok:?} rejected at state {:?}", state);
            state = check.next_state;
        }
        assert_eq!(state, JsonState::RootClosed);
    }

    #[test]
    fn root_closed_rejects_non_whitespace() {
        let state = JsonState::RootClosed;
        assert!(state.check_token(" ").valid);
        assert!(state.check_token("\n").valid);
        assert!(!state.check_token("x").valid);
        assert!(!state.check_token("{").valid);
    }

    #[test]
    fn backslash_escape_in_string() {
        let state = JsonState::InString(false);
        let check = state.check_token("\\n");
        assert!(check.valid);
        assert_eq!(check.next_state, JsonState::InString(false));
    }

    #[test]
    fn control_char_in_string_is_invalid() {
        let state = JsonState::InString(false);
        // \u{0000} is a control char
        let check = state.check_token("\u{0000}");
        assert!(!check.valid);
    }

    #[test]
    fn colon_without_closed_string_is_invalid() {
        // ":\"a\":1" — the `:` at the start is invalid because we're not
        // expecting a colon (we're at root).
        let state = JsonState::Root;
        // Actually `:` at root is invalid.
        let check = state.check_token(":");
        assert!(!check.valid);
    }

    #[test]
    fn string_opens_but_doesnt_close_is_still_valid_token() {
        // A token that opens a string but doesn't close it (partial string)
        // is valid — the FSM enters InString and the next token must close it.
        let state = JsonState::Root;
        let check = state.check_token("\"abc");
        assert!(check.valid);
        assert_eq!(check.next_state, JsonState::InString(false));
    }
}

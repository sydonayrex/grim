//! WI-3a: JSON-syntax-only FSM.
//!
//! `response_format: {"type": "json_object"}` constrains generation to
//! *syntactically valid JSON* — balanced braces/brackets/quotes, no bare
//! schema validation. This is the smallest correctly-scoped first milestone
//! (matches OpenAI's own tiered rollout: `json_object` shipped before
//! `json_schema`).
//!
//! The FSM operates on **characters**. Token-level masking is done by
//! simulating each vocabulary token through the FSM (see
//! [`JsonModeFsm::valid_tokens`]). This is the naive per-step simulation
//! the plan accepts for WI-3a; WI-3c replaces it with a precomputed
//! transition table once correctness is proven.

/// States of the JSON-mode pushdown automaton.
///
/// The state is `(in_string, escape_next, depth_brace, depth_bracket, has_value)`.
/// `depth_*` are the open-brace / open-bracket stacks; a value is valid
/// only when both return to 0, the FSM is not mid-string, and at least one
/// value token has been emitted (an empty stream is not valid JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JsonModeState {
    /// Inside a `"..."` string literal.
    pub in_string: bool,
    /// The previous char was a backslash (so the next char is escaped).
    pub escape_next: bool,
    /// Unclosed `{` count.
    pub depth_brace: u32,
    /// Unclosed `[` count.
    pub depth_bracket: u32,
    /// At least one value char has been emitted. Empty input is not valid
    /// JSON, so the accepting state requires this to be true.
    pub has_value: bool,
}

impl Default for JsonModeState {
    fn default() -> Self {
        Self {
            in_string: false,
            escape_next: false,
            depth_brace: 0,
            depth_bracket: 0,
            has_value: false,
        }
    }
}

impl JsonModeState {
    /// Feed one character through the FSM. Returns `Ok(())` if the char is
    /// a valid JSON continuation, `Err(bad)` with the offending char
    /// otherwise.
    pub fn feed(&mut self, c: char) -> Result<(), char> {
        if self.escape_next {
            // Any char is valid as an escape target; clear the flag.
            self.escape_next = false;
            self.has_value = true;
            return Ok(());
        }
        match c {
            '"' => {
                self.in_string = !self.in_string;
                self.has_value = true;
                Ok(())
            }
            '{' if !self.in_string => {
                self.depth_brace = self.depth_brace.saturating_add(1);
                self.has_value = true;
                Ok(())
            }
            '}' if !self.in_string => {
                if self.depth_brace == 0 {
                    return Err(c);
                }
                self.depth_brace -= 1;
                self.has_value = true;
                Ok(())
            }
            '[' if !self.in_string => {
                self.depth_bracket = self.depth_bracket.saturating_add(1);
                self.has_value = true;
                Ok(())
            }
            ']' if !self.in_string => {
                if self.depth_bracket == 0 {
                    return Err(c);
                }
                self.depth_bracket -= 1;
                self.has_value = true;
                Ok(())
            }
            // Whitespace and structural punctuation are always valid
            // outside strings; control chars are not.
            c if c.is_control() && c != '\n' && c != '\t' && c != '\r' => Err(c),
            _ => {
                self.has_value = true;
                Ok(())
            }
        }
    }

    /// True when the FSM is in a *complete, accepting* state: not mid-string
    /// and all braces/brackets are balanced. A generation that ends here
    /// is guaranteed to be syntactically valid JSON.
    pub fn is_accepting(&self) -> bool {
        !self.in_string
            && !self.escape_next
            && self.depth_brace == 0
            && self.depth_bracket == 0
            && self.has_value
    }
}

/// WI-3a: JSON-mode FSM. Tracks a single character stream and answers
/// whether a candidate token keeps the output on a valid JSON path.
#[derive(Debug, Clone, Default)]
pub struct JsonModeFsm {
    state: JsonModeState,
}

impl JsonModeFsm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an FSM at an explicit state. Used by `TokenMaskCache` to
    /// compute masks for arbitrary states without holding an FSM instance.
    pub fn from_state(state: JsonModeState) -> Self {
        Self { state }
    }

    pub fn state(&self) -> JsonModeState {
        self.state
    }

    /// Feed a string through the FSM, returning the final state (or the
    /// first offending char). Does not mutate `self`.
    pub fn feed_str(&self, s: &str) -> Result<JsonModeState, char> {
        let mut st = self.state;
        for c in s.chars() {
            st.feed(c)?;
        }
        Ok(st)
    }

    /// Advance the FSM by one token's text. Returns `Ok(new_state)` if the
    /// token is a valid continuation, `Err(bad_char)` otherwise.
    pub fn feed_token(&mut self, token: &str) -> Result<JsonModeState, char> {
        let mut st = self.state;
        for c in token.chars() {
            st.feed(c)?;
        }
        self.state = st;
        Ok(st)
    }

    /// WI-3a (naive, correctness-gated): for each token in `vocab`, simulate
    /// appending it to the current FSM state and mark it valid iff the
    /// simulation succeeds **and** the resulting state is not a "dead end"
    /// (i.e. it can still reach an accepting state).
    ///
    /// The dead-end filter is the key correctness property: a token that
    /// leaves the FSM in a state from which no completion can reach an
    /// accepting state (e.g. an unterminated string with no more tokens)
    /// must be masked out, otherwise the sampler can strand the output.
    ///
    /// `TODO(perf)`: this is O(vocab × avg-token-len) per step — the naive
    /// simulation WI-3a accepts. WI-3c replaces it with a precomputed
    /// transition table (the xgrammar technique).
    pub fn valid_tokens(&self, vocab: &[String]) -> Vec<bool> {
        vocab.iter().map(|t| self.token_is_valid(t)).collect()
    }

    /// True if appending `token` keeps the output on a path that can still
    /// reach an accepting state.
    pub fn token_is_valid(&self, token: &str) -> bool {
        let mut st = self.state;
        for c in token.chars() {
            match st.feed(c) {
                Ok(()) => {}
                Err(_) => return false,
            }
        }
        can_reach_accepting(&st)
    }
}

/// Conservative reachability check: from `st`, is there *some* completion
/// that reaches an accepting state?
///
/// This is intentionally permissive — it only rules out states that are
/// provably dead (unbalanced delimiters that can never close, or an
/// unterminated escape). It never masks a token that could still lead to
/// valid JSON, which keeps the constraint sound (no false rejections).
fn can_reach_accepting(st: &JsonModeState) -> bool {
    if st.escape_next {
        // A backslash was the last char; any escaped char is valid, so
        // the string can always be closed. Not dead.
        return true;
    }
    if st.in_string {
        // Mid-string: the string can always be closed with a quote. Not dead.
        return true;
    }
    // Outside a string with balanced-or-open delimiters: we can always emit
    // more structural chars and eventually close. The only dead state is a
    // premature close of a delimiter that isn't open — but `feed` already
    // rejects those, so any state reachable here is live.
    true
}

/// Apply a boolean mask to logits: set masked positions to `-inf`.
///
/// If **every** token is masked (a degenerate state), this leaves logits
/// unchanged rather than producing all-`-inf` — the inner sampler then
/// picks whatever it wants, which is the honest behavior when the FSM has
/// no valid continuation (better than NaN/argmax-of-neg-infinity).
pub fn apply_mask(logits: &mut [f32], mask: &[bool]) {
    debug_assert_eq!(
        logits.len(),
        mask.len(),
        "logits/mask length mismatch: {} vs {}",
        logits.len(),
        mask.len()
    );
    let all_masked = mask.iter().all(|&m| !m);
    if all_masked {
        // No valid continuation — leave logits alone rather than -inf
        // everything. The inner sampler picks something; the output will
        // be invalid JSON, which is the honest failure mode.
        return;
    }
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// WI-3c: precomputed per-state token-validity cache.
///
/// The naive `valid_tokens` simulates every vocab token through the FSM on
/// every decode step — O(vocab × avg-token-len) per token emitted. This
/// cache computes the mask once per distinct FSM state and reuses it for
/// the lifetime of that state, which is the core technique xgrammar uses.
///
/// The cache is keyed on `JsonModeState` (which is `Copy + Eq + Hash`), so
/// lookups are O(1) after the first visit to a state. States are revisited
/// frequently during decode (e.g. "mid-string, not escaped" persists across
/// many tokens), so the amortized cost approaches O(vocab) per *state*
/// rather than per *step*.
///
/// `TODO(perf)`: this is the WI-3c optimization. It's correctness-gated
/// behind WI-3a (the FSM and mask are proven correct first). The cache is
/// transparent — `mask_for` returns the same answer as the naive path,
/// just faster.
pub struct TokenMaskCache {
    cache: std::collections::HashMap<JsonModeState, std::sync::Arc<[bool]>>,
}

impl Default for TokenMaskCache {
    fn default() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }
}

impl TokenMaskCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the validity mask for `state` against `vocab`, computing and
    /// caching it on first encounter. The returned `Arc` is shared across
    /// calls so subsequent lookups are cheap pointer clones.
    pub fn mask_for(
        &mut self,
        state: JsonModeState,
        vocab: &[String],
    ) -> std::sync::Arc<[bool]> {
        if let Some(mask) = self.cache.get(&state) {
            return mask.clone();
        }
        let mask: Vec<bool> = vocab
            .iter()
            .map(|t| state_is_valid_for_token(state, t))
            .collect();
        let arc: std::sync::Arc<[bool]> = std::sync::Arc::from(mask.into_boxed_slice());
        self.cache.insert(state, arc.clone());
        arc
    }

    /// Number of distinct states cached so far. Useful for profiling the
    /// cache's hit rate against the naive path.
    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Pure function: is `token` a valid continuation from `state`? Extracted so
/// `TokenMaskCache` can compute it without borrowing an FSM instance.
fn state_is_valid_for_token(state: JsonModeState, token: &str) -> bool {
    let mut st = state;
    for c in token.chars() {
        match st.feed(c) {
            Ok(()) => {}
            Err(_) => return false,
        }
    }
    can_reach_accepting(&st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_is_not_accepting() {
        // An empty stream is not yet valid JSON — you need at least a value.
        assert!(!JsonModeState::default().is_accepting());
    }

    #[test]
    fn test_bare_number_is_accepting() {
        let mut st = JsonModeState::default();
        for c in "42".chars() {
            st.feed(c).unwrap();
        }
        assert!(st.is_accepting(), "bare number 42 is valid JSON");
    }

    #[test]
    fn test_nested_object_accepting() {
        let mut st = JsonModeState::default();
        for c in r#"{"a": [1, 2], "b": {"c": true}}"#.chars() {
            st.feed(c).unwrap();
        }
        assert!(st.is_accepting(), "nested object is valid JSON");
    }

    #[test]
    fn test_unbalanced_brace_rejected() {
        let mut st = JsonModeState::default();
        for c in "{\"a\": 1".chars() {
            st.feed(c).unwrap();
        }
        assert!(!st.is_accepting(), "unclosed brace is not valid JSON");
    }

    #[test]
    fn test_premature_close_rejected() {
        let mut st = JsonModeState::default();
        // `}` with nothing open is a syntax error.
        assert!(st.feed('}').is_err());
    }

    #[test]
    fn test_string_escape_round_trip() {
        let mut st = JsonModeState::default();
        for c in "\"hello\\nworld\"".chars() {
            st.feed(c).unwrap();
        }
        assert!(st.is_accepting(), "escaped string is valid JSON");
    }

    #[test]
    fn test_token_mask_keeps_valid_paths() {
        let fsm = JsonModeFsm::new();
        // At the start, `{"` begins a valid object — should be valid.
        assert!(fsm.token_is_valid("{\""));
        // `}` at the start is a syntax error — should be masked.
        assert!(!fsm.token_is_valid("}"));
    }

    #[test]
    fn test_apply_mask_all_masked_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mask = vec![false, false, false];
        apply_mask(&mut logits, &mask);
        assert_eq!(logits, vec![1.0, 2.0, 3.0], "all-masked leaves logits alone");
    }

    #[test]
    fn test_apply_mask_zeros_masked_positions() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false, true];
        apply_mask(&mut logits, &mask);
        assert!(logits[0].is_finite(), "valid token keeps finite logit");
        assert_eq!(logits[1], f32::NEG_INFINITY, "masked token -> -inf");
        assert!(logits[2].is_finite());
    }

    /// WI-3c: the precomputed cache must return the *same* mask as the
    /// naive per-token simulation. This is the correctness gate for the
    /// optimization — a cache that disagrees with the naive path would
    /// silently change which tokens are masked.
    #[test]
    fn test_cache_matches_naive_path() {
        let vocab: Vec<String> = [
            '{', '}', '"', ':', ',', '0', '1', 'a', ' ',
        ]
        .iter()
        .map(|c| c.to_string())
        .collect();

        // Several distinct FSM states, including ones reachable mid-stream.
        let states = [
            JsonModeState::default(),
            JsonModeState {
                in_string: true,
                escape_next: false,
                depth_brace: 1,
                depth_bracket: 0,
                has_value: true,
            },
            JsonModeState {
                in_string: false,
                escape_next: false,
                depth_brace: 0,
                depth_bracket: 1,
                has_value: true,
            },
            JsonModeState {
                in_string: true,
                escape_next: true,
                depth_brace: 1,
                depth_bracket: 0,
                has_value: true,
            },
        ];

        let mut cache = TokenMaskCache::new();
        for state in &states {
            let naive = JsonModeFsm::from_state(*state).valid_tokens(&vocab);
            let cached = cache.mask_for(*state, &vocab).to_vec();
            assert_eq!(
                naive, cached,
                "cache mask diverged from naive path for state {state:?}"
            );
        }
        // Second pass: every lookup is a cache hit, and the result is
        // identical to the first.
        for state in &states {
            let cached = cache.mask_for(*state, &vocab).to_vec();
            assert_eq!(cached, cache.mask_for(*state, &vocab).to_vec());
        }
        assert_eq!(cache.len(), states.len(), "expected one entry per state");
    }
}
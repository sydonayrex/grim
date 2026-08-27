//! Pushdown automaton JSON FSM and TokenMaskCache for structured grammar decoding.

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JsonState {
    pub stack: Vec<Bracket>,
    pub mode: Mode,
    pub escaped: bool,
    pub is_key: bool,
    pub hex_remaining: u8,
}

impl Default for JsonState {
    fn default() -> Self {
        Self {
            stack: Vec::new(),
            mode: Mode::ExpectValue,
            escaped: false,
            is_key: false,
            hex_remaining: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bracket {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    ExpectValue,
    ExpectKey,
    ExpectColon,
    ExpectSep,
    StringKey,
    StringValue,
    NumStart { minus: bool },
    NumInt,
    NumFrac,
    NumExp,
    NumExpSign,
    NumExpDigits,
    LitTrue(u8),
    LitFalse(u8),
    LitNull(u8),
    Done,
}

#[derive(Debug, Clone)]
pub struct FsmCheck {
    pub next_state: JsonState,
    pub valid: bool,
    pub root_done: bool,
}

impl JsonState {
    pub fn advance(&mut self, c: char) -> bool {
        if self.hex_remaining > 0 {
            if c.is_ascii_hexdigit() {
                self.hex_remaining -= 1;
                return true;
            } else {
                return false;
            }
        }

        match self.mode {
            Mode::ExpectValue => match c {
                '{' => {
                    self.stack.push(Bracket::Object);
                    self.mode = Mode::ExpectKey;
                    self.is_key = true;
                    true
                }
                '[' => {
                    self.stack.push(Bracket::Array);
                    self.mode = Mode::ExpectValue;
                    self.is_key = false;
                    true
                }
                '"' => {
                    self.mode = Mode::StringValue;
                    self.escaped = false;
                    self.is_key = false;
                    true
                }
                '}' => {
                    if self.stack.last() == Some(&Bracket::Object) {
                        self.stack.pop();
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                        true
                    } else {
                        false
                    }
                }
                ']' => {
                    if self.stack.last() == Some(&Bracket::Array) {
                        self.stack.pop();
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                        true
                    } else {
                        false
                    }
                }
                '-' => {
                    self.mode = Mode::NumStart { minus: true };
                    true
                }
                c if c.is_ascii_digit() => {
                    self.mode = Mode::ExpectSep;
                    true
                }
                't' => {
                    self.mode = Mode::LitTrue(1);
                    true
                }
                'f' => {
                    self.mode = Mode::LitFalse(1);
                    true
                }
                'n' => {
                    self.mode = Mode::LitNull(1);
                    true
                }
                ' ' | '\t' | '\n' | '\r' => true,
                _ => false,
            },

            Mode::ExpectKey => match c {
                '}' => {
                    if self.stack.last() == Some(&Bracket::Object) {
                        self.stack.pop();
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                        true
                    } else {
                        false
                    }
                }
                '"' => {
                    self.mode = Mode::StringKey;
                    self.escaped = false;
                    self.is_key = true;
                    true
                }
                ' ' | '\t' | '\n' | '\r' => true,
                _ => false,
            },

            Mode::ExpectColon => match c {
                ':' => {
                    self.mode = Mode::ExpectValue;
                    self.is_key = false;
                    true
                }
                ' ' | '\t' | '\n' | '\r' => true,
                _ => false,
            },

            Mode::ExpectSep => match c {
                ',' => match self.stack.last() {
                    Some(Bracket::Object) => {
                        self.mode = Mode::ExpectKey;
                        self.is_key = true;
                        true
                    }
                    Some(Bracket::Array) => {
                        self.mode = Mode::ExpectValue;
                        self.is_key = false;
                        true
                    }
                    None => false,
                },
                '}' => {
                    if self.stack.last() == Some(&Bracket::Object) {
                        self.stack.pop();
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                        true
                    } else if self.stack.is_empty() {
                        self.mode = Mode::Done;
                        true
                    } else {
                        false
                    }
                }
                ']' => {
                    if self.stack.last() == Some(&Bracket::Array) {
                        self.stack.pop();
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                        true
                    } else {
                        false
                    }
                }
                c if c.is_ascii_digit() => true,
                '.' => true,
                'e' | 'E' | '+' | '-' => true,
                ' ' | '\t' | '\n' | '\r' => true,
                _ => false,
            },

            Mode::StringKey | Mode::StringValue => {
                if self.escaped {
                    match c {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                            self.escaped = false;
                            true
                        }
                        'u' => {
                            self.escaped = false;
                            self.hex_remaining = 4;
                            true
                        }
                        _ => false,
                    }
                } else if c == '\\' {
                    self.escaped = true;
                    true
                } else if c == '"' {
                    if self.mode == Mode::StringKey {
                        self.mode = Mode::ExpectColon;
                    } else {
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                    }
                    true
                } else {
                    !c.is_control()
                }
            }

            Mode::NumStart { minus: _ } => match c {
                c if c.is_ascii_digit() => {
                    self.mode = Mode::ExpectSep;
                    true
                }
                _ => false,
            },

            Mode::NumInt | Mode::NumFrac | Mode::NumExp | Mode::NumExpSign | Mode::NumExpDigits => {
                match c {
                    c if c.is_ascii_digit() => true,
                    '.' => true,
                    'e' | 'E' | '+' | '-' => true,
                    '}' | ']' | ',' | ' ' | '\t' | '\n' | '\r' => {
                        self.mode = Mode::ExpectSep;
                        self.advance(c)
                    }
                    _ => false,
                }
            }

            Mode::LitTrue(n) => {
                let expected = b"true";
                if (n as usize) < expected.len() && c == expected[n as usize] as char {
                    let next = n + 1;
                    if next == 4 {
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                    } else {
                        self.mode = Mode::LitTrue(next);
                    }
                    true
                } else {
                    false
                }
            }

            Mode::LitFalse(n) => {
                let expected = b"false";
                if (n as usize) < expected.len() && c == expected[n as usize] as char {
                    let next = n + 1;
                    if next == 5 {
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                    } else {
                        self.mode = Mode::LitFalse(next);
                    }
                    true
                } else {
                    false
                }
            }

            Mode::LitNull(n) => {
                let expected = b"null";
                if (n as usize) < expected.len() && c == expected[n as usize] as char {
                    let next = n + 1;
                    if next == 4 {
                        self.mode = if self.stack.is_empty() {
                            Mode::Done
                        } else {
                            Mode::ExpectSep
                        };
                    } else {
                        self.mode = Mode::LitNull(next);
                    }
                    true
                } else {
                    false
                }
            }

            Mode::Done => matches!(c, ' ' | '\t' | '\n' | '\r'),
        }
    }

    pub fn check_token(&self, text: &str) -> FsmCheck {
        let mut state = self.clone();
        let mut valid = true;
        for c in text.chars() {
            if !state.advance(c) {
                valid = false;
                break;
            }
        }
        let root_done = valid
            && state.stack.is_empty()
            && (matches!(state.mode, Mode::Done | Mode::ExpectSep) || state.is_done());
        FsmCheck {
            next_state: state,
            valid,
            root_done,
        }
    }

    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    pub fn is_done(&self) -> bool {
        matches!(self.mode, Mode::Done)
    }

    pub fn valid_tokens(&self, vocab: &[String]) -> Vec<bool> {
        vocab.iter().map(|t| self.token_is_valid(t)).collect()
    }

    pub fn token_is_valid(&self, token: &str) -> bool {
        let mut state = self.clone();
        for c in token.chars() {
            if !state.advance(c) {
                return false;
            }
        }
        true
    }

    pub fn feed_token(&mut self, token: &str) -> bool {
        for c in token.chars() {
            if !self.advance(c) {
                return false;
            }
        }
        true
    }

    pub fn from_state(stack: Vec<Bracket>, mode: Mode, escaped: bool, is_key: bool) -> Self {
        Self {
            stack,
            mode,
            escaped,
            is_key,
            hex_remaining: 0,
        }
    }
}

pub fn apply_mask(logits: &mut [f32], mask: &[bool]) {
    debug_assert_eq!(
        logits.len(),
        mask.len(),
        "logits/mask length mismatch: {} vs {}",
        logits.len(),
        mask.len()
    );
    if mask.iter().all(|&m| !m) {
        return;
    }
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

#[derive(Default)]
pub struct TokenMaskCache {
    inner: HashMap<JsonState, Arc<[bool]>>,
}

impl TokenMaskCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mask_for(&mut self, state: JsonState, vocab: &[String]) -> Arc<[bool]> {
        if let Some(m) = self.inner.get(&state) {
            return m.clone();
        }
        let mask: Vec<bool> = state.valid_tokens(vocab);
        let arc: Arc<[bool]> = Arc::from(mask.into_boxed_slice());
        self.inner.insert(state, arc.clone());
        arc
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> JsonState {
        JsonState::default()
    }

    #[test]
    fn root_accepts_open_brace() {
        let mut s = state();
        assert!(s.advance('{'));
        assert_eq!(s.stack_depth(), 1);
        assert!(matches!(s.mode, Mode::ExpectKey));
    }

    #[test]
    fn root_accepts_open_bracket() {
        let mut s = state();
        assert!(s.advance('['));
        assert_eq!(s.stack_depth(), 1);
        assert!(matches!(s.mode, Mode::ExpectValue));
    }

    #[test]
    fn root_rejects_close_brace() {
        let mut s = state();
        assert!(!s.advance('}'));
    }

    #[test]
    fn root_rejects_close_bracket() {
        let mut s = state();
        assert!(!s.advance(']'));
    }

    #[test]
    fn object_key_then_colon() {
        let mut s = state();
        assert!(s.advance('{'));
        assert!(matches!(s.mode, Mode::ExpectKey));
        assert!(s.advance('"'));
        assert!(s.is_key);
        assert!(matches!(s.mode, Mode::StringKey));
        assert!(s.advance('a'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectColon));
        assert!(s.advance(':'));
        assert!(matches!(s.mode, Mode::ExpectValue));
        assert!(s.advance('1'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance('}'));
        assert_eq!(s.stack_depth(), 0);
        assert!(matches!(s.mode, Mode::Done));
    }

    #[test]
    fn value_string_closes_to_expect_sep() {
        let mut s = state();
        assert!(s.advance('{'));
        assert!(s.advance('"'));
        assert!(s.advance('k'));
        assert!(s.advance('"'));
        assert!(s.advance(':'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::StringValue));
        assert!(s.advance('v'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance('}'));
        assert_eq!(s.stack_depth(), 0);
        assert!(matches!(s.mode, Mode::Done));
    }

    #[test]
    fn string_with_escape() {
        let mut s = state();
        assert!(s.advance('"'));
        assert!(s.advance('\\'));
        assert!(s.escaped);
        assert!(s.advance('"'));
        assert!(!s.escaped);
        assert!(matches!(s.mode, Mode::StringValue));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::Done));
    }

    #[test]
    fn control_char_in_string_rejected() {
        let mut s = state();
        assert!(s.advance('"'));
        assert!(!s.advance('\n'));
    }

    #[test]
    fn nested_object() {
        let mut s = state();
        assert!(s.advance('{'));
        assert_eq!(s.stack_depth(), 1);
        assert!(s.advance('"'));
        assert!(s.is_key);
        assert!(s.advance('a'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectColon));
        assert!(s.advance(':'));
        assert!(matches!(s.mode, Mode::ExpectValue));
        assert!(s.advance('{'));
        assert_eq!(s.stack_depth(), 2);
        assert!(s.advance('"'));
        assert!(s.advance('b'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectColon));
        assert!(s.advance(':'));
        assert!(s.advance('1'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance('}'));
        assert_eq!(s.stack_depth(), 1);
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance('}'));
        assert_eq!(s.stack_depth(), 0);
        assert!(matches!(s.mode, Mode::Done));
    }

    #[test]
    fn array_of_numbers() {
        let mut s = state();
        assert!(s.advance('['));
        assert!(s.advance('1'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance(','));
        assert!(matches!(s.mode, Mode::ExpectValue));
        assert!(s.advance('2'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance(']'));
        assert_eq!(s.stack_depth(), 0);
        assert!(matches!(s.mode, Mode::Done));
    }

    #[test]
    fn done_state_rejects_non_whitespace() {
        let mut s = state();
        assert!(s.advance('1'));
        assert!(matches!(s.mode, Mode::ExpectSep));
        assert!(s.advance('}'));
        assert_eq!(s.stack_depth(), 0);
        assert!(matches!(s.mode, Mode::Done));
        assert!(s.advance(' '));
        assert!(matches!(s.mode, Mode::Done));
        assert!(!s.advance('x'));
    }

    #[test]
    fn token_check_valid_json_object() {
        let s = state();
        let check = s.check_token("{\"a\":1}");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_valid_json_array() {
        let s = state();
        let check = s.check_token("[1,2,3]");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_invalid_closing_brace() {
        let s = state();
        let check = s.check_token("}");
        assert!(!check.valid);
    }

    #[test]
    fn token_check_true_literal() {
        let s = state();
        let check = s.check_token("true");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_false_literal() {
        let s = state();
        let check = s.check_token("false");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_null_literal() {
        let s = state();
        let check = s.check_token("null");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_number() {
        let s = state();
        let check = s.check_token("42");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_negative_number() {
        let s = state();
        let check = s.check_token("-3.14");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_scientific_notation() {
        let s = state();
        let check = s.check_token("1e10");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_malformed_number_rejected() {
        let s = state();
        let check = s.check_token("123x");
        assert!(!check.valid);
    }

    #[test]
    fn token_check_malformed_literal_rejected() {
        let s = state();
        let check = s.check_token("ture");
        assert!(!check.valid);
    }

    #[test]
    fn token_check_mismatched_brackets_rejected() {
        let s = state();
        let check = s.check_token("{]");
        assert!(!check.valid);
    }

    #[test]
    fn token_check_string_with_unicode_escape() {
        let s = state();
        let check = s.check_token("\"\\u0041\"");
        assert!(check.valid);
        assert!(check.root_done);
    }

    #[test]
    fn token_check_string_with_bad_escape_rejected() {
        let s = state();
        let check = s.check_token("\"\\x\"");
        assert!(!check.valid);
    }

    #[test]
    fn cache_matches_naive_per_state() {
        let vocab: Vec<String> = ['{', '}', '"', ':', ',', '0', '1', 'a', 'e', '-']
            .iter()
            .map(|c| c.to_string())
            .collect();
        let states = [
            JsonState::default(),
            JsonState::from_state(vec![Bracket::Object], Mode::ExpectColon, false, true),
            JsonState::from_state(vec![Bracket::Array], Mode::ExpectSep, false, false),
            JsonState::from_state(vec![], Mode::StringValue, false, false),
        ];
        let mut cache = TokenMaskCache::new();
        for state in &states {
            let naive = JsonState::valid_tokens(state, &vocab);
            let cached = cache.mask_for(state.clone(), &vocab).to_vec();
            assert_eq!(naive, cached, "cache diverged for {state:?}");
        }
        // Second pass: pure cache hits, same answers.
        for state in &states {
            let cached = cache.mask_for(state.clone(), &vocab).to_vec();
            assert_eq!(cached, cache.mask_for(state.clone(), &vocab).to_vec());
        }
        assert_eq!(cache.len(), states.len());
    }
}

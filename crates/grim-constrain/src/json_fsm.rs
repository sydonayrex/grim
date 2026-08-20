//! JSON-mode pushdown automaton for constrained decoding — WI-3a.
//!
//! Coherent bracket-tracking, string-state, and value-position FSM. The
//! sampler integration decodes each candidate token to text and runs this
//! over its characters; an illegal character transition masks the token out
//! of the logits at that step.
//!
//! Handles **structural** JSON constraints: balanced brackets/braces, valid
//! escape sequences, comma/colon positions, value-position expectations.
//! Number lexing and literal spelling (`true`/`false`/`null`) are tracked
//! but intentionally permissive — a real JSON parser is the correctness gate.

use grim_tensor::error::Result;

/// Outcome of running the JSON FSM over a candidate token's decoded text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsmCheck {
    pub next_state: JsonState,
    pub valid: bool,
    pub root_done: bool,
}

/// Pushdown automaton state for JSON syntax validation.
#[derive(Clone, Debug, Default)]
pub struct JsonState {
    stack: Vec<Bracket>,
    mode: Mode,
    escaped: bool,
    is_key: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bracket {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl JsonState {
    pub fn advance(&mut self, c: char) -> bool {
        loop {
            let (new_mode, pop, push, new_escaped, new_is_key, legal) = self.mode.advance(c);
            if !legal {
                return false;
            }
            let mode_changed = self.mode != new_mode;

            self.mode = new_mode;
            self.escaped = new_escaped;
            self.is_key = new_is_key;
            if pop > 0 {
                self.stack.truncate(self.stack.len() - pop);
            }
            if let Some(b) = push {
                self.stack.push(b);
            }

            let value_just_completed = self.was_value_in_progress(self.mode)
                && !matches!(
                    new_mode,
                    Mode::NumStart { .. }
                        | Mode::NumInt
                        | Mode::NumFrac
                        | Mode::NumExp
                        | Mode::NumExpSign
                        | Mode::NumExpDigits
                        | Mode::LitTrue(_)
                        | Mode::LitFalse(_)
                        | Mode::LitNull(_)
                )
                && !is_literal_final_char(self.mode, c);
            if value_just_completed {
                continue;
            }
            return true;
        }
    }

    fn was_value_in_progress(&self, mode: Mode) -> bool {
        matches!(
            mode,
            Mode::NumStart { .. }
                | Mode::NumInt
                | Mode::NumFrac
                | Mode::NumExp
                | Mode::NumExpSign
                | Mode::NumExpDigits
                | Mode::LitTrue(_)
                | Mode::LitFalse(_)
                | Mode::LitNull(_)
        )
    }

    pub fn check_token(&self, text: &str) -> FsmCheck {
        let mut state = self.clone();
        let mut valid = true;
        for c in text.chars() {
            if !state.advance(c) {
                valid = false;
            }
        }
        FsmCheck {
            next_state: state,
            valid,
            root_done: matches!(state.mode, Mode::Done),
        }
    }

    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }
    pub fn is_done(&self) -> bool {
        matches!(self.mode, Mode::Done)
    }
}

impl Mode {
    fn advance(
        self,
        c: char,
    ) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
        match self {
            Mode::ExpectValue => expect_value(c),
            Mode::ExpectKey => expect_key(c),
            Mode::ExpectColon => expect_colon(c),
            Mode::ExpectSep => expect_sep(c),
            Mode::StringKey => string(c, true),
            Mode::StringValue => string(c, false),
            Mode::NumStart { minus } => num_start(c, minus),
            Mode::NumInt => num_int(c),
            Mode::NumFrac => num_frac(c),
            Mode::NumExp => num_exp(c),
            Mode::NumExpSign => num_exp_sign(c),
            Mode::NumExpDigits => num_exp_digits(c),
            Mode::LitTrue(n) => lit_true(c, n),
            Mode::LitFalse(n) => lit_false(c, n),
            Mode::LitNull(n) => lit_null(c, n),
            Mode::Done => done(c),
        }
    }
}

fn expect_value(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        '{' => {
            (
                Mode::ExpectKey,
                0,
                Some(Bracket::Object),
                false,
                false,
                true,
            )
        }
        '[' => {
            (Mode::ExpectValue, 0, Some(Bracket::Array), false, false, true)
        }
        '"' => (Mode::StringValue, 0, None, false, false, true),
        c if c.is_ascii_digit() || c == '-' => {
            (Mode::NumStart { minus: c == '-' }, 0, None, false, false, true)
        }
        't' => (Mode::LitTrue(1), 0, None, false, false, true),
        'f' => (Mode::LitFalse(1), 0, None, false, false, true),
        'n' => (Mode::LitNull(1), 0, None, false, false, true),
        ' ' | '\t' | '\n' | '\r' => {
            (Mode::ExpectValue, 0, None, false, false, true)
        }
        _ => (Mode::ExpectValue, 0, None, false, false, false),
    }
}

fn expect_key(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        '}' => (Mode::ExpectSep, 1, None, false, false, true),
        '"' => (Mode::StringKey, 0, None, false, true, true),
        ' ' | '\t' | '\n' | '\r' => {
            (Mode::ExpectKey, 0, None, false, false, true)
        }
        _ => (Mode::ExpectKey, 0, None, false, false, false),
    }
}

fn expect_colon(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        ':' => (Mode::ExpectValue, 0, None, false, false, true),
        ' ' | '\t' | '\n' | '\r' => {
            (Mode::ExpectColon, 0, None, false, false, true)
        }
        _ => (Mode::ExpectColon, 0, None, false, false, false),
    }
}

fn expect_sep(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        ',' => (Mode::ExpectValue, 0, None, false, false, true),
        _ => (Mode::ExpectSep, 0, None, false, false, true),
    }
}

fn string(c: char, is_key: bool) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        '\\' => {
            (
                Mode::StringKey,
                0,
                None,
                true,
                is_key,
                true,
            )
        }
        '"' => {
            if is_key {
                (Mode::ExpectColon, 0, None, false, false, true)
            } else {
                (Mode::ExpectSep, 0, None, false, false, true)
            }
        }
        c if c.is_control() => {
            (Mode::StringKey, 0, None, false, is_key, false)
        }
        _ => {
            (Mode::StringKey, 0, None, false, is_key, true)
        }
    }
}

fn num_start(c: char, minus: bool) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        c if c.is_ascii_digit() => {
            (Mode::NumInt, 0, None, false, false, true)
        }
        '.' => (Mode::NumFrac, 0, None, false, false, true),
        'e' | 'E' => (Mode::NumExp, 0, None, false, false, true),
        _ => (Mode::NumInt, 0, None, false, false, false),
    }
}

fn num_int(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        c if c.is_ascii_digit() => (Mode::NumInt, 0, None, false, false, true),
        '.' => (Mode::NumFrac, 0, None, false, false, true),
        'e' | 'E' => (Mode::NumExp, 0, None, false, false, true),
        _ => {
            // End of number; re-submit in ExpectSep
            (Mode::ExpectSep, 0, None, false, false, true)
        }
    }
}

fn num_frac(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        c if c.is_ascii_digit() => (Mode::NumFrac, 0, None, false, false, true),
        'e' | 'E' => (Mode::NumExp, 0, None, false, false, true),
        _ => (Mode::ExpectSep, 0, None, false, false, true),
    }
}

fn num_exp(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        '+' | '-' => (Mode::NumExpSign, 0, None, false, false, true),
        c if c.is_ascii_digit() => (Mode::NumExpDigits, 0, None, false, false, true),
        _ => (Mode::NumExpDigits, 0, None, false, false, false),
    }
}

fn num_exp_sign(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        c if c.is_ascii_digit() => (Mode::NumExpDigits, 0, None, false, false, true),
        _ => (Mode::NumExpDigits, 0, None, false, false, false),
    }
}

fn num_exp_digits(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        c if c.is_ascii_digit() => (Mode::NumExpDigits, 0, None, false, false, true),
        _ => (Mode::ExpectSep, 0, None, false, false, true),
    }
}

fn lit_true(c: char, n: u8) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    let expected = b"rue";
    if n < 3 && c == expected[n] as char {
        if n + 1 < 3 {
            (Mode::LitTrue(n + 1), 0, None, false, false, true)
        } else {
            (Mode::ExpectSep, 0, None, false, false, true)
        }
    } else {
        (Mode::LitTrue(n), 0, None, false, false, false)
    }
}

fn lit_false(c: char, n: u8) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    let expected = b"alse";
    if n < 4 && c == expected[n] as char {
        if n + 1 < 4 {
            (Mode::LitFalse(n + 1), 0, None, false, false, true)
        } else {
            (Mode::ExpectSep, 0, None, false, false, true)
        }
    } else {
        (Mode::LitFalse(n), 0, None, false, false, false)
    }
}

fn lit_null(c: char, n: u8) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    let expected = b"ull";
    if n < 3 && c == expected[n] as char {
        if n + 1 < 3 {
            (Mode::LitNull(n + 1), 0, None, false, false, true)
        } else {
            (Mode::ExpectSep, 0, None, false, false, true)
        }
    } else {
        (Mode::LitNull(n), 0, None, false, false, false)
    }
}

fn done(c: char) -> (Mode, usize, Option<Bracket>, bool, bool, bool) {
    match c {
        ' ' | '\t' | '\n' | '\r' => (Mode::Done, 0, None, false, false, true),
        _ => (Mode::Done, 0, None, false, false, false),
    }
}

fn is_literal_final_char(mode: Mode, c: char) -> bool {
    match mode {
        Mode::LitTrue(n) => n >= 2 && c == 'e',
        Mode::LitFalse(n) => n >= 3 && c == 'e',
        Mode::LitNull(n) => n >= 2 && c == 'l',
        _ => false,
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
        assert_eq!(s.is_key, true);
        assert!(matches!(s.mode, Mode::StringKey));
        assert!(s.advance('a'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectColon));
        assert!(s.advance(':'));
        assert!(matches!(s.mode, Mode::ExpectValue));
    }

    #[test]
    fn value_string_closes_to_expect_sep() {
        let mut s = state();
        assert!(s.advance('['));
        assert!(matches!(s.mode, Mode::ExpectValue));
        assert!(s.advance('"'));
        assert_eq!(s.is_key, false);
        assert!(matches!(s.mode, Mode::StringValue));
        assert!(s.advance('x'));
        assert!(s.advance('"'));
        assert!(matches!(s.mode, Mode::ExpectSep));
    }

    #[test]
    fn string_with_escape() {
        let mut s = state();
        assert!(s.advance('"'));
        assert!(s.advance('\\'));
        assert_eq!(s.escaped, true);
        assert!(s.advance('n'));
        assert_eq!(s.escaped, false);
        assert!(s.advance('"'));
    }

    #[test]
    fn control_char_in_string_rejected() {
        let mut s = state();
        assert!(s.advance('"'));
        assert!(!s.advance('\u{0000}'));
    }

    #[test]
    fn nested_object() {
        let mut s = state();
        // {"a":{"b":1}}
        assert!(s.advance('{'));
        assert!(s.advance('"'));
        assert_eq!(s.is_key, true);
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
}

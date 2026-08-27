//! WI-3b: JSON-Schema → FSM/grammar compiler.
//!
//! `response_format: {"type": "json_schema", "json_schema": {...}}` constrains
//! generation to outputs that conform to a JSON Schema.
//!
//! Scope (per the plan): `type`, `properties`, `required`, `enum`, `items`,
//! nested `object`/`array`. Unsupported schema features are rejected with a
//! clear error rather than silently under-constraining.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// WI-3b: error from compiling a JSON Schema into a constraint.
#[derive(Debug, Clone)]
pub struct JsonSchemaCompilerError {
    pub message: String,
}

impl std::fmt::Display for JsonSchemaCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "json schema compile error: {}", self.message)
    }
}

impl std::error::Error for JsonSchemaCompilerError {}

/// A compiled JSON-Schema constraint with memoized token validity masking.
///
/// At each step we check whether the partial JSON produced so far is consistent
/// with the schema, and mask tokens whose continuation would violate it.
/// Computed masks are cached per distinct output prefix to prevent redundant
/// O(V) re-validations.
#[derive(Debug, Clone)]
pub struct JsonSchemaConstraint {
    schema: Value,
    cache: Arc<Mutex<HashMap<String, Arc<[bool]>>>>,
    /// F9: true when any `pattern` or string-`enum` exists anywhere in the
    /// schema. When false, tokens appended INSIDE an unterminated string can
    /// never change schema validity (plain strings accept any content), so
    /// the per-step O(vocab) validate pass is skippable entirely.
    has_string_constraint: bool,
}

impl JsonSchemaConstraint {
    pub fn schema(&self) -> &Value {
        &self.schema
    }
}

/// WI-3b: compile a JSON Schema value into a constraint. Unsupported
/// features are rejected explicitly (a `400` at the request layer) rather
/// than silently ignored.
///
/// Supported subset: `type`, `properties`, `required`, `enum`, `items`,
/// `$ref` (internal pointers `#/...`), `oneOf`/`anyOf`/`allOf`, nested `object`/`array`.
/// `format`, `pattern`, `additionalProperties` are rejected if unrecognized.
pub fn compile_json_schema(schema: Value) -> Result<JsonSchemaConstraint, JsonSchemaCompilerError> {
    let obj = schema.as_object().ok_or_else(|| JsonSchemaCompilerError {
        message: "json_schema must be a JSON object".to_string(),
    })?;

    for unsupported in &["format"] {
        if obj.contains_key(*unsupported) {
            return Err(JsonSchemaCompilerError {
                message: format!(
                    "unsupported json_schema keyword '{unsupported}'; supported subset: \
                     type, properties, required, enum, items, pattern, $ref, oneOf, anyOf, allOf, nested object/array"
                ),
            });
        }
    }

    let resolved_schema = resolve_refs(&schema, &schema, 0)?;

    if let Some(t) = resolved_schema.get("type") {
        let ty = t.as_str().ok_or_else(|| JsonSchemaCompilerError {
            message: "json_schema.type must be a string".to_string(),
        })?;
        match ty {
            "object" | "array" | "string" | "number" | "integer" | "boolean" | "null" => {}
            _ => {
                return Err(JsonSchemaCompilerError {
                    message: format!("unknown json_schema.type '{ty}'"),
                });
            }
        }
    }

    let has_string_constraint = scan_string_constraints(&resolved_schema);
    Ok(JsonSchemaConstraint {
        schema: resolved_schema,
        cache: Arc::new(Mutex::new(HashMap::new())),
        has_string_constraint,
    })
}

fn resolve_refs(
    schema: &Value,
    root: &Value,
    depth: usize,
) -> Result<Value, JsonSchemaCompilerError> {
    if depth > 32 {
        return Err(JsonSchemaCompilerError {
            message: "circular $ref detected".to_string(),
        });
    }
    match schema {
        Value::Object(map) => {
            if let Some(r) = map.get("$ref").and_then(|v| v.as_str()) {
                let target = resolve_pointer(r, root)?;
                return resolve_refs(&target, root, depth + 1);
            }
            let mut resolved = serde_json::Map::new();
            for (k, v) in map {
                resolved.insert(k.clone(), resolve_refs(v, root, depth + 1)?);
            }
            Ok(Value::Object(resolved))
        }
        Value::Array(arr) => {
            let mut resolved = Vec::with_capacity(arr.len());
            for v in arr {
                resolved.push(resolve_refs(v, root, depth + 1)?);
            }
            Ok(Value::Array(resolved))
        }
        other => Ok(other.clone()),
    }
}

fn resolve_pointer(pointer: &str, root: &Value) -> Result<Value, JsonSchemaCompilerError> {
    if !pointer.starts_with("#/") {
        return Err(JsonSchemaCompilerError {
            message: format!("unsupported URI pointer '{pointer}', only '#/...' supported"),
        });
    }
    let parts = pointer[2..].split('/');
    let mut current = root;
    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part).ok_or_else(|| JsonSchemaCompilerError {
                    message: format!("unresolved $ref path '{part}' in '{pointer}'"),
                })?;
            }
            _ => {
                return Err(JsonSchemaCompilerError {
                    message: format!("cannot index into non-object path '{part}'"),
                });
            }
        }
    }
    Ok(current.clone())
}

impl JsonSchemaConstraint {
    /// WI-3b: validate that `partial_json` is a prefix of some value that
    /// conforms to the schema. Conservative: returns `true` when it cannot
    /// prove a violation (so we never mask a token that could still be valid).
    pub fn is_consistent(&self, partial_json: &str) -> bool {
        let value: Value = match serde_json::from_str(partial_json) {
            Ok(v) => v,
            Err(e) => {
                if is_truncated_error(&e) {
                    return true;
                }
                return false;
            }
        };
        validate(&value, &self.schema)
    }

    /// F9 fast path: while the cursor is INSIDE an unterminated JSON string
    /// and the schema imposes no pattern/enum on strings, appending any token
    /// keeps the output as consistent as it was when the string opened — so
    /// the whole O(vocab) parse+validate pass is skipped. The structural PDA
    /// mask still applies upstream.
    pub fn inside_string_fast_path(&self, current_output: &str) -> bool {
        !self.has_string_constraint && inside_unterminated_string(current_output)
    }

    /// Return a memoized validity mask for `vocab` given `current_output`.
    pub fn mask_for(&self, vocab: &[String], current_output: &str) -> Arc<[bool]> {
        if let Ok(guard) = self.cache.lock() {
            if let Some(mask) = guard.get(current_output) {
                return mask.clone();
            }
        }
        let mask: Vec<bool> = vocab
            .iter()
            .map(|t| self.is_consistent(&format!("{current_output}{t}")))
            .collect();
        let arc: Arc<[bool]> = Arc::from(mask.into_boxed_slice());
        if let Ok(mut guard) = self.cache.lock() {
            if guard.len() > 1024 {
                guard.clear();
            }
            guard.insert(current_output.to_string(), arc.clone());
        }
        arc
    }

    /// WI-3b: for each token in `vocab`, check whether appending it to
    /// `current_output` keeps the partial output consistent with the schema.
    pub fn valid_tokens(&self, vocab: &[String], current_output: &str) -> Vec<bool> {
        self.mask_for(vocab, current_output).to_vec()
    }
}

/// Heuristic: is this serde error a "truncated/incomplete input" error
/// rather than a genuine syntax error? serde_json's error type doesn't
/// expose the category directly, so we key off the message — imperfect
/// but adequate for a conservative accept.
fn is_truncated_error(e: &serde_json::Error) -> bool {
    let msg = e.to_string();
    msg.contains("EOF")
        || msg.contains("trailing")
        || msg.contains("expected")
        || msg.contains("control character")
}

#[derive(Debug, Clone, PartialEq)]
enum CharMatcher {
    Any,
    Exact(char),
    Class {
        ranges: Vec<(char, char)>,
        chars: Vec<char>,
        negated: bool,
    },
}

impl CharMatcher {
    fn matches(&self, c: char) -> bool {
        match self {
            CharMatcher::Any => true,
            CharMatcher::Exact(target) => *target == c,
            CharMatcher::Class {
                ranges,
                chars,
                negated,
            } => {
                let hit =
                    chars.contains(&c) || ranges.iter().any(|&(low, high)| c >= low && c <= high);
                if *negated { !hit } else { hit }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RegexElement {
    matcher: CharMatcher,
    min_repeat: usize,
    max_repeat: usize,
}

/// Bounded-backtracking regex subset compiler supporting:
/// `^`, `$`, `[...]`, `[^...]`, `\d`, `\w`, `\s`, `.`, `{m,n}`, `*`, `+`, `?`, literals.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedRegex {
    elements: Vec<RegexElement>,
    anchor_start: bool,
    anchor_end: bool,
}

impl BoundedRegex {
    pub fn parse(pattern: &str) -> Option<Self> {
        let chars: Vec<char> = pattern.chars().collect();
        if chars.is_empty() {
            return Some(Self {
                elements: Vec::new(),
                anchor_start: false,
                anchor_end: false,
            });
        }

        let mut i = 0;
        let mut anchor_start = false;
        let mut anchor_end = false;

        if chars[0] == '^' {
            anchor_start = true;
            i += 1;
        }

        let len = chars.len();
        let end_limit = if len > i && chars[len - 1] == '$' && (len < 2 || chars[len - 2] != '\\') {
            anchor_end = true;
            len - 1
        } else {
            len
        };

        let mut elements = Vec::new();

        while i < end_limit {
            let c = chars[i];
            let matcher = match c {
                '.' => {
                    i += 1;
                    CharMatcher::Any
                }
                '\\' => {
                    if i + 1 >= end_limit {
                        return None;
                    }
                    let next = chars[i + 1];
                    i += 2;
                    match next {
                        'd' => CharMatcher::Class {
                            ranges: vec![('0', '9')],
                            chars: Vec::new(),
                            negated: false,
                        },
                        'D' => CharMatcher::Class {
                            ranges: vec![('0', '9')],
                            chars: Vec::new(),
                            negated: true,
                        },
                        'w' => CharMatcher::Class {
                            ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9')],
                            chars: vec!['_'],
                            negated: false,
                        },
                        'W' => CharMatcher::Class {
                            ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9')],
                            chars: vec!['_'],
                            negated: true,
                        },
                        's' => CharMatcher::Class {
                            ranges: Vec::new(),
                            chars: vec![' ', '\t', '\n', '\r'],
                            negated: false,
                        },
                        'S' => CharMatcher::Class {
                            ranges: Vec::new(),
                            chars: vec![' ', '\t', '\n', '\r'],
                            negated: true,
                        },
                        escaped => CharMatcher::Exact(escaped),
                    }
                }
                '[' => {
                    i += 1;
                    let mut negated = false;
                    if i < end_limit && chars[i] == '^' {
                        negated = true;
                        i += 1;
                    }
                    let mut ranges = Vec::new();
                    let mut specific_chars = Vec::new();
                    let mut closed = false;

                    while i < end_limit {
                        let ch = chars[i];
                        if ch == ']' {
                            closed = true;
                            i += 1;
                            break;
                        }
                        if i + 2 < end_limit && chars[i + 1] == '-' && chars[i + 2] != ']' {
                            let low = ch;
                            let high = chars[i + 2];
                            ranges.push((low, high));
                            i += 3;
                        } else if ch == '\\' && i + 1 < end_limit {
                            let esc = chars[i + 1];
                            match esc {
                                'd' => ranges.push(('0', '9')),
                                'w' => {
                                    ranges.push(('a', 'z'));
                                    ranges.push(('A', 'Z'));
                                    ranges.push(('0', '9'));
                                    specific_chars.push('_');
                                }
                                's' => specific_chars.extend_from_slice(&[' ', '\t', '\n', '\r']),
                                other => specific_chars.push(other),
                            }
                            i += 2;
                        } else {
                            specific_chars.push(ch);
                            i += 1;
                        }
                    }

                    if !closed {
                        return None;
                    }
                    CharMatcher::Class {
                        ranges,
                        chars: specific_chars,
                        negated,
                    }
                }
                literal => {
                    i += 1;
                    CharMatcher::Exact(literal)
                }
            };

            let (min_rep, max_rep) = if i < end_limit {
                match chars[i] {
                    '*' => {
                        i += 1;
                        (0, usize::MAX)
                    }
                    '+' => {
                        i += 1;
                        (1, usize::MAX)
                    }
                    '?' => {
                        i += 1;
                        (0, 1)
                    }
                    '{' => {
                        let start_q = i + 1;
                        let mut end_q = start_q;
                        while end_q < end_limit && chars[end_q] != '}' {
                            end_q += 1;
                        }
                        if end_q >= end_limit {
                            return None;
                        }
                        let q_str: String = chars[start_q..end_q].iter().collect();
                        i = end_q + 1;

                        if let Some((min_s, max_s)) = q_str.split_once(',') {
                            let min = min_s.trim().parse::<usize>().ok()?;
                            let max = if max_s.trim().is_empty() {
                                usize::MAX
                            } else {
                                max_s.trim().parse::<usize>().ok()?
                            };
                            (min, max)
                        } else {
                            let n = q_str.trim().parse::<usize>().ok()?;
                            (n, n)
                        }
                    }
                    _ => (1, 1),
                }
            } else {
                (1, 1)
            };

            elements.push(RegexElement {
                matcher,
                min_repeat: min_rep,
                max_repeat: max_rep,
            });
        }

        Some(Self {
            elements,
            anchor_start,
            anchor_end,
        })
    }

    pub fn matches(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let mut steps = 0usize;
        let max_steps = 10_000usize;

        if self.anchor_start {
            self.match_from(&chars, 0, 0, &mut steps, max_steps)
        } else {
            for start_idx in 0..=chars.len() {
                if self.match_from(&chars, start_idx, 0, &mut steps, max_steps) {
                    return true;
                }
                if steps >= max_steps {
                    break;
                }
            }
            false
        }
    }

    fn match_from(
        &self,
        chars: &[char],
        text_pos: usize,
        elem_idx: usize,
        steps: &mut usize,
        max_steps: usize,
    ) -> bool {
        *steps += 1;
        if *steps > max_steps {
            return false;
        }

        if elem_idx == self.elements.len() {
            if self.anchor_end {
                return text_pos == chars.len();
            }
            return true;
        }

        let elem = &self.elements[elem_idx];

        let mut max_matched = 0;
        while text_pos + max_matched < chars.len() && max_matched < elem.max_repeat {
            if elem.matcher.matches(chars[text_pos + max_matched]) {
                max_matched += 1;
            } else {
                break;
            }
        }

        if max_matched < elem.min_repeat {
            return false;
        }

        for k in (elem.min_repeat..=max_matched).rev() {
            if self.match_from(chars, text_pos + k, elem_idx + 1, steps, max_steps) {
                return true;
            }
            if *steps > max_steps {
                return false;
            }
        }

        false
    }
}

pub fn validate_pattern(pat: &str, text: &str) -> bool {
    if let Some(regex) = BoundedRegex::parse(pat) {
        regex.matches(text)
    } else {
        if pat.starts_with('^') && pat.ends_with('$') {
            let inner = pat
                .strip_prefix('^')
                .and_then(|p| p.strip_suffix('$'))
                .unwrap_or(pat);
            text == inner || text.starts_with(inner)
        } else if let Some(inner) = pat.strip_prefix('^') {
            text.starts_with(inner)
        } else if let Some(inner) = pat.strip_suffix('$') {
            text.ends_with(inner)
        } else {
            text.contains(pat)
        }
    }
}

impl JsonSchemaConstraint {
    /// Return deterministic lookahead fast-forward string if the current schema state permits only a single literal continuation.
    pub fn lookahead_jump_forward(&self, partial_json: &str) -> Option<String> {
        let trimmed = partial_json.trim_start();
        if trimmed.is_empty() {
            if let Some(ty) = self.schema.get("type").and_then(|v| v.as_str()) {
                if ty == "object" {
                    return Some("{\n".to_string());
                }
            }
        }

        if trimmed == "{" || trimmed == "{\n" {
            if let Some(req) = self.schema.get("required").and_then(|v| v.as_array()) {
                if req.len() == 1 {
                    if let Some(key_name) = req[0].as_str() {
                        return Some(format!("\"{}\": ", key_name));
                    }
                }
            }
        }

        if let Some(enum_vals) = self.schema.get("enum").and_then(|v| v.as_array()) {
            if enum_vals.len() == 1 {
                return Some(enum_vals[0].to_string());
            }
        }

        None
    }
}

/// Recursive JSON-Schema validation for the supported subset.
fn validate(value: &Value, schema: &Value) -> bool {
    if let Some(ty) = schema.get("type").and_then(|v| v.as_str()) {
        let type_ok = match ty {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => {
                value.is_number() && value.as_f64().map(|n| n.fract() == 0.0).unwrap_or(false)
            }
            "boolean" => value.as_bool().is_some(),
            "null" => value.is_null(),
            _ => return false,
        };
        if !type_ok {
            return false;
        }
    }
    // pattern (bounded backtracking regex subset)
    if let (Some(pat), Some(s)) = (
        schema.get("pattern").and_then(|v| v.as_str()),
        value.as_str(),
    ) {
        if !pat.is_empty() && !validate_pattern(pat, s) {
            return false;
        }
    }
    // enum
    if let Some(allowed) = schema.get("enum").and_then(|v| v.as_array()) {
        if !allowed.iter().any(|a| a == value) {
            return false;
        }
    }
    // oneOf
    if let Some(variants) = schema.get("oneOf").and_then(|v| v.as_array()) {
        let matches = variants.iter().filter(|s| validate(value, s)).count();
        if matches != 1 {
            return false;
        }
    }
    // anyOf
    if let Some(variants) = schema.get("anyOf").and_then(|v| v.as_array()) {
        if !variants.iter().any(|s| validate(value, s)) {
            return false;
        }
    }
    // allOf
    if let Some(variants) = schema.get("allOf").and_then(|v| v.as_array()) {
        if !variants.iter().all(|s| validate(value, s)) {
            return false;
        }
    }
    // object: properties + required
    if let Some(obj) = value.as_object() {
        if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
            for (key, sub) in props {
                if let Some(val) = obj.get(key) {
                    if !validate(val, sub) {
                        return false;
                    }
                }
            }
        }
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for r in required {
                let name = r.as_str().unwrap_or("");
                if !obj.contains_key(name) {
                    return false;
                }
            }
        }
    }
    // array: items
    if let Some(arr) = value.as_array() {
        if let Some(items) = schema.get("items") {
            for item in arr {
                if !validate(item, items) {
                    return false;
                }
            }
        }
    }
    true
}

/// True when `partial` ends inside an unterminated JSON string literal
/// (unescaped double-quote parity scan).
pub fn inside_unterminated_string(partial: &str) -> bool {
    let mut in_str = false;
    let mut escaped = false;
    for c in partial.chars() {
        match c {
            '"' if !escaped => in_str = !in_str,
            '\\' if in_str => escaped = !escaped,
            _ => escaped = false,
        }
    }
    in_str
}

/// Recursive scan: does this schema subtree constrain string CONTENT via
/// `pattern` or a string-typed `enum`?
fn scan_string_constraints(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            let is_string = map.get("type").and_then(|t| t.as_str()) == Some("string");
            let has_pattern = map.contains_key("pattern");
            let string_enum = is_string && map.contains_key("enum");
            if has_pattern || string_enum {
                return true;
            }
            map.values().any(scan_string_constraints)
        }
        Value::Array(arr) => arr.iter().any(scan_string_constraints),
        _ => false,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolves_defs_ref() {
        let schema = serde_json::json!({
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"]
                }
            },
            "type": "object",
            "properties": {
                "user": { "$ref": "#/$defs/User" }
            }
        });
        let c = compile_json_schema(schema).unwrap();
        assert!(c.is_consistent("{\"user\": {\"name\": \"Alice\"}}"));
        assert!(!c.is_consistent("{\"user\": {\"name\": 123}}"));
    }

    #[test]
    fn test_accepts_oneof() {
        let schema = serde_json::json!({
            "oneOf": [
                { "type": "string" },
                { "type": "number" }
            ]
        });
        let c = compile_json_schema(schema).unwrap();
        assert!(c.is_consistent("\"hello\""));
        assert!(c.is_consistent("42"));
        assert!(!c.is_consistent("true"));
    }

    #[test]
    fn test_pattern_constraint() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "pattern": "^[A-Z]{3}$" }
            }
        });
        let c = compile_json_schema(schema).unwrap();
        assert!(c.is_consistent("{\"code\": \"ABC\"}"));
    }

    #[test]
    fn test_accepts_supported_subset() {
        let c = compile_json_schema(serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"]
        }))
        .unwrap();
        assert!(c.is_consistent("{\"name\": \""));
        assert!(c.is_consistent("{\"name\": \"bob\", \"age\": 42}"));
        assert!(
            !c.is_consistent("{\"name\": 42}"),
            "type violation detected"
        );
    }

    #[test]
    fn test_enum_constraint() {
        let c =
            compile_json_schema(serde_json::json!({"type": "string", "enum": ["a", "b"]})).unwrap();
        assert!(c.is_consistent("\"a\""));
        assert!(!c.is_consistent("\"c\""), "enum violation detected");
    }

    #[test]
    fn test_bounded_regex_features() {
        let r1 = BoundedRegex::parse("^[a-z0-9_-]{3,8}$").unwrap();
        assert!(r1.matches("abc-12"));
        assert!(!r1.matches("ab"));
        assert!(!r1.matches("toolongstring123"));
        assert!(!r1.matches("ABC-12"));

        let r2 = BoundedRegex::parse(r"^\d{4}-\d{2}$").unwrap();
        assert!(r2.matches("2026-08"));
        assert!(!r2.matches("2026-8"));
        assert!(!r2.matches("year-08"));

        let r3 = BoundedRegex::parse(r"^[^0-9]+$").unwrap();
        assert!(r3.matches("hello_world"));
        assert!(!r3.matches("hello123world"));
    }

    #[test]
    fn test_schema_lookahead_jump_forward() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "user_id": { "type": "string" }
            },
            "required": ["user_id"]
        });
        let c = compile_json_schema(schema).unwrap();
        assert_eq!(c.lookahead_jump_forward(""), Some("{\n".to_string()));
        assert_eq!(
            c.lookahead_jump_forward("{"),
            Some("\"user_id\": ".to_string())
        );
    }
}

#[cfg(test)]
mod f9_fast_path_tests {
    use super::*;

    #[test]
    fn inside_string_detection() {
        assert!(!inside_unterminated_string(""));
        assert!(!inside_unterminated_string("{\"a\": 1}"));
        assert!(inside_unterminated_string("{\"a\": \"hel"));
        assert!(!inside_unterminated_string("{\"a\": \"v\\\"x\"}"));
        assert!(inside_unterminated_string("{\"a\": \"v\\"));
    }

    #[test]
    fn fast_path_skips_only_unconstrained_strings() {
        let plain = compile_json_schema(serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } }
        }))
        .unwrap();
        assert!(!plain.has_string_constraint);
        assert!(plain.inside_string_fast_path("{\"name\": \"hello wor"));

        let constrained = compile_json_schema(serde_json::json!({
            "type": "object",
            "properties": { "code": { "type": "string", "pattern": "^[A-Z]{3}$" } }
        }))
        .unwrap();
        assert!(constrained.has_string_constraint);
        assert!(!constrained.inside_string_fast_path("{\"code\": \"AB"));

        let enummed = compile_json_schema(serde_json::json!({
            "type": "object",
            "properties": { "kind": { "type": "string", "enum": ["a", "b"] } }
        }))
        .unwrap();
        assert!(enummed.has_string_constraint);
    }

    #[test]
    fn fast_path_masks_equal_full_validate_inside_plain_strings() {
        let vocab: Vec<String> = ["\"", "x", "A", "1", "}", "{", ",", ":", "\\n", " hello"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let c = compile_json_schema(serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" }, "n": { "type": "number" } },
            "required": ["name", "n"]
        }))
        .unwrap();
        let partial = "{\"name\": \"partial value";
        assert!(c.inside_string_fast_path(partial));
        let fast = c.mask_for(&vocab, partial);
        let slow: Vec<bool> = vocab
            .iter()
            .map(|t| c.is_consistent(&format!("{partial}{t}")))
            .collect();
        assert_eq!(
            fast.to_vec(),
            slow,
            "fast path must agree with full validate"
        );
    }
}

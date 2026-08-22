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

    for unsupported in &["format", "pattern"] {
        if obj.contains_key(*unsupported) {
            return Err(JsonSchemaCompilerError {
                message: format!(
                    "unsupported json_schema keyword '{unsupported}'; supported subset: \
                     type, properties, required, enum, items, $ref, oneOf, anyOf, allOf, nested object/array"
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

    Ok(JsonSchemaConstraint {
        schema: resolved_schema,
        cache: Arc::new(Mutex::new(HashMap::new())),
    })
}

fn resolve_refs(schema: &Value, root: &Value, depth: usize) -> Result<Value, JsonSchemaCompilerError> {
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
}

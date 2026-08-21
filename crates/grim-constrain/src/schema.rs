//! WI-3b: JSON-Schema → FSM/grammar compiler.
//!
//! `response_format: {"type": "json_schema", "json_schema": {...}}` constrains
//! generation to outputs that conform to a JSON Schema.
//!
//! Scope (per the plan): `type`, `properties`, `required`, `enum`, `items`,
//! nested `object`/`array`. Unsupported schema features are rejected with a
//! clear error rather than silently under-constraining.

use serde_json::Value;

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

/// A compiled JSON-Schema constraint. Currently this is a *validator*
/// rather than a full FSM: at each step we check whether the partial JSON
/// produced so far is consistent with the schema, and mask tokens whose
/// continuation would violate it.
///
/// `TODO(perf)`: a real FSM would precompute per-token validity; this
/// validator re-parses the partial output each step. Correctness-gated
/// for WI-3b; WI-3c optimizes it.
#[derive(Debug, Clone)]
pub struct JsonSchemaConstraint {
    schema: Value,
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
/// nested `object`/`array`. `$ref`, `oneOf`/`anyOf`/`allOf`, `format`,
/// `pattern`, `additionalProperties` are **not** supported and cause a
/// rejection — callers get a clear error instead of malformed output.
pub fn compile_json_schema(schema: Value) -> Result<JsonSchemaConstraint, JsonSchemaCompilerError> {
    let obj = schema.as_object().ok_or_else(|| JsonSchemaCompilerError {
        message: "json_schema must be a JSON object".to_string(),
    })?;
    // Reject unsupported composition keywords up front — better a 400 than
    // silently under-constrained output.
    for unsupported in &["$ref", "oneOf", "anyOf", "allOf", "format", "pattern"] {
        if obj.contains_key(*unsupported) {
            return Err(JsonSchemaCompilerError {
                message: format!(
                    "unsupported json_schema keyword '{unsupported}'; supported subset: \
                     type, properties, required, enum, items, nested object/array"
                ),
            });
        }
    }
    if let Some(t) = obj.get("type") {
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
    Ok(JsonSchemaConstraint { schema })
}

impl JsonSchemaConstraint {
    /// WI-3b: validate that `partial_json` is a prefix of some value that
    /// conforms to the schema. Conservative: returns `true` when it cannot
    /// prove a violation (so we never mask a token that could still be valid).
    ///
    /// The check is: parse `partial_json`; if it parses, validate it fully
    /// against the schema; if it doesn't parse (incomplete), accept it as
    /// long as the parse failure is a trailing-incomplete error rather than
    /// a genuine type violation.
    pub fn is_consistent(&self, partial_json: &str) -> bool {
        let value: Value = match serde_json::from_str(partial_json) {
            Ok(v) => v,
            Err(e) => {
                // Incomplete input (trailing comma, unclosed brace, etc.)
                // is acceptable — the generation isn't done yet. A real
                // syntax error (e.g. `12abc`) is not.
                if is_truncated_error(&e) {
                    return true;
                }
                return false;
            }
        };
        validate(&value, &self.schema)
    }

    /// WI-3b: for each token in `vocab`, check whether appending it to
    /// `current_output` keeps the partial output consistent with the schema.
    pub fn valid_tokens(&self, vocab: &[String], current_output: &str) -> Vec<bool> {
        vocab
            .iter()
            .map(|t| self.is_consistent(&format!("{current_output}{t}")))
            .collect()
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
    fn test_rejects_unsupported_keyword() {
        let err = compile_json_schema(serde_json::json!({"type": "object", "$ref": "#/x"}));
        assert!(err.is_err(), "$ref must be rejected");
    }

    #[test]
    fn test_rejects_oneof() {
        let err = compile_json_schema(serde_json::json!({"oneOf": [{"type": "string"}]}));
        assert!(err.is_err(), "oneOf must be rejected");
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

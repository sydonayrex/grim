//! WI-TOOLS-4 — Post-hoc parsing of model output into structured tool calls.
//!
//! Input: the model's raw completion string plus a family hint. Output:
//! [`ParseOutcome`] — `Some(tool_calls)` when parsing clearly succeeds, `None`
//! when no tool call is detected (treat as ordinary content).
//!
//! **Contract:** never guess. If we cannot clearly extract a well-formed call,
//! return `None` so the caller falls back to plain content. A failed parse is
//! not a request failure.

use grim_format::ToolCallMsg;

/// Which family of tool-call convention the model's chat template follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFamily {
    /// Hermes-2-Pro style `<tool_call>...` marker-delimited JSON.
    TagDelimited,
    /// Bare-JSON convention (some Mistral/Qwen variants emit a raw JSON object).
    #[allow(dead_code)]
    BareJson,
    /// LFM2.5 bracket-first convention.
    BracketFirst,
    /// Unknown template — scanner tries conventions in order.
    Auto,
}

/// The result of parsing one completion string.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseOutcome {
    /// Structured calls when a clean parse succeeded; `None` otherwise.
    pub calls: Option<Vec<ToolCallMsg>>,
    /// Debug-only diagnostic when a candidate looked like a call but failed.
    pub diagnostic: Option<String>,
}

/// Resolve the parse convention from a model's embedded chat template text.
///
/// Heuristic, best-effort: a template that mentions `tool_call` (Hermes-2-Pro
/// and its descendants) gets tag-delimited parsing; otherwise we fall back to
/// the Auto scanner. This is deliberately cheap — it only biases which
/// strategy is tried first, and Auto runs all of them anyway.
pub fn resolve_tool_family(template: &str) -> ToolFamily {
    let lower = template.to_ascii_lowercase();
    if lower.contains("tool_call") || lower.contains("<tool_call") {
        ToolFamily::TagDelimited
    } else {
        ToolFamily::Auto
    }
}

/// Map model architecture name to its expected tool-call convention (§WI-E8).
pub fn family_for_arch(arch: &str) -> ToolFamily {
    let lower = arch.to_ascii_lowercase();
    if lower.contains("lfm2") || lower.contains("liquid") {
        ToolFamily::BracketFirst
    } else if lower.contains("llama") || lower.contains("mistral") || lower.contains("qwen") {
        ToolFamily::TagDelimited
    } else if lower.contains("deepseek") {
        ToolFamily::BareJson
    } else {
        ToolFamily::Auto
    }
}

/// Parse a completion string for tool calls under a given family convention.
///
/// Returns `ParseOutcome { calls: Some(..), .. }` on a clean parse, or
/// `{ calls: None, .. }` when no call was detected or the candidate text did
/// not parse as well-formed JSON. Never fails the request — see module docs.
///
/// The `family` hint selects the *primary* convention to try first; on a miss
/// we fall back to the other convention so a mislabeled template still works
/// in the obvious way (rather than giving up). `Auto` tries both in order.
pub fn parse_tool_calls(completion: &str, family: ToolFamily) -> ParseOutcome {
    if family == ToolFamily::BracketFirst {
        let bracket = parse_bracket_call(completion);
        if let Some(calls) = bracket.calls {
            return ParseOutcome {
                calls: Some(calls),
                diagnostic: bracket.diagnostic,
            };
        }
    }

    let (first, second) = match family {
        ToolFamily::BareJson => (parse_bare_json(completion), parse_tag_delimited(completion)),
        ToolFamily::BracketFirst | ToolFamily::TagDelimited | ToolFamily::Auto => {
            (parse_tag_delimited(completion), parse_bare_json(completion))
        }
    };
    if let Some(calls) = first.calls {
        return ParseOutcome {
            calls: Some(calls),
            diagnostic: first.diagnostic,
        };
    }
    if let Some(calls) = second.calls {
        return ParseOutcome {
            calls: Some(calls),
            diagnostic: second.diagnostic,
        };
    }
    // F-4: LFM2.5 convention — `<|tool_call_start|>[name(arg=val, ...)]<|tool_call_end|>`.
    // Tried last so the Hermes/bare-JSON conventions keep priority.
    let bracket = parse_bracket_call(completion);
    if let Some(calls) = bracket.calls {
        return ParseOutcome {
            calls: Some(calls),
            diagnostic: bracket.diagnostic,
        };
    }
    ParseOutcome {
        calls: None,
        diagnostic: first.diagnostic.or(second.diagnostic).or(bracket.diagnostic),
    }
}

/// WI-TOOLS-4b/4c — stable reason strings shared by both runaway-call guards so
/// clients can distinguish a hard 400's cause without parsing free text.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunawayReason {
    /// WI-TOOLS-4b: the (name, args) tuple has already appeared >= 4 times.
    DuplicateToolCall,
    /// WI-TOOLS-4c-i: total tool-call entries across all assistant messages
    /// would exceed the conversation budget.
    TotalToolCallLimit,
    /// WI-TOOLS-4c-ii: the request's `messages` array exceeded the per-request
    /// cap.
    MessageCountLimit,
}

impl RunawayReason {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            RunawayReason::DuplicateToolCall => "duplicate_tool_call_limit",
            RunawayReason::TotalToolCallLimit => "total_tool_call_limit",
            RunawayReason::MessageCountLimit => "message_count_limit",
        }
    }
}

/// WI-TOOLS-4c-ii helper: count every `tool_calls` entry across all assistant
/// messages in `messages`, regardless of name/arguments — the total tool-call
/// budget consumed so far by this conversation.
pub fn count_total_prior_tool_calls(messages: &[grim_format::ChatMessage]) -> usize {
    let mut total = 0;
    for m in messages {
        if m.role == "assistant" {
            if let Some(calls) = &m.tool_calls {
                total += calls.len();
            }
        }
    }
    total
}

/// WI-TOOLS-4b — Repeated tool call guard.
///
/// Count how many times `(name, canonicalized_arguments)` already appears among
/// prior assistant `tool_calls` entries in `messages`. Canonicalizes
/// `arguments` by parsing the JSON string and re-serializing with sorted keys
/// so `{"city":"NYC","units":"F"}` and `{"units":"F","city":"NYC"}` compare
/// equal. Falls back to raw-string comparison if a prior call's `arguments`
/// isn't valid JSON (defensive — a client-replayed history could in principle
/// contain anything; a non-JSON arguments string must never panic the counter).
pub fn count_prior_identical_calls(
    messages: &[grim_format::ChatMessage],
    name: &str,
    arguments: &str,
) -> usize {
    let canonical = canonicalize_args(arguments);
    let mut count = 0;
    for m in messages {
        if m.role != "assistant" {
            continue;
        }
        let Some(calls) = &m.tool_calls else { continue };
        for tc in calls {
            if tc.name == name && canonicalize_args(&tc.arguments) == canonical {
                count += 1;
            }
        }
    }
    count
}

/// Parse + re-serialize a JSON arguments string with sorted keys, so reordered
/// keys compare equal. On a parse failure the raw string is returned unchanged
/// (raw-string fallback per the spec).
fn canonicalize_args(arguments: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v) => {
            serde_json::to_string(&sort_json_keys(v)).unwrap_or_else(|_| arguments.to_string())
        }
        Err(_) => arguments.to_string(),
    }
}

/// Recursively rebuild a `serde_json::Value` with object keys sorted ascending,
/// so two structurally-equal-but-differently-ordered arguments canonicalize to
/// the same string.
fn sort_json_keys(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut sorted: Vec<(String, serde_json::Value)> = map
                .into_iter()
                .map(|(k, v)| (k, sort_json_keys(v)))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(
                sorted
                    .into_iter()
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
            )
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(sort_json_keys).collect())
        }
        other => other,
    }
}

/// Hermes-style `<tool_call>{"name":...,"arguments":{...}}</tool_call>`.
fn parse_tag_delimited(completion: &str) -> ParseOutcome {
    let mut calls = Vec::new();
    let mut diagnostic = None;
    let mut rest = completion;
    loop {
        let open = rest.find("<tool_call>");
        let Some(open) = open else { break };
        let inner_start = open + "<tool_call>".len();
        let after = &rest[inner_start..];
        let close = after.find("</tool_call>");
        let (inner, remainder) = match close {
            Some(c) => (&after[..c], &after[c + "</tool_call>".len()..]),
            None => break,
        };
        match extract_call(inner.trim()) {
            Some(call) => calls.push(call),
            None => {
                diagnostic = Some(format!(
                    "tool_call marker with unparseable interior: {inner}"
                ));
            }
        }
        rest = remainder;
    }
    ParseOutcome {
        calls: if calls.is_empty() { None } else { Some(calls) },
        diagnostic,
    }
}

/// Bare-JSON convention: the entire completion is a JSON object with
/// `{name, arguments}` (OpenAI style) or `{function: {name, arguments}}`.
fn parse_bare_json(completion: &str) -> ParseOutcome {
    let trimmed = completion.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return ParseOutcome {
            calls: None,
            diagnostic: None,
        };
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(v) => match extract_call_value(&v) {
            Some(call) => ParseOutcome {
                calls: Some(vec![call]),
                diagnostic: None,
            },
            None => ParseOutcome {
                calls: None,
                diagnostic: Some("bare JSON object is not a tool call".to_string()),
            },
        },
        Err(e) => ParseOutcome {
            calls: None,
            diagnostic: Some(format!("bare JSON parse failed: {e}")),
        },
    }
}

/// Extract a single `ToolCallMsg` from a JSON object.
fn extract_call(value: &str) -> Option<ToolCallMsg> {
    let v = serde_json::from_str::<serde_json::Value>(value).ok()?;
    extract_call_value(&v)
}

/// F-4: LFM2.5 bracket-call convention —
/// `<|tool_call_start|>[name(arg=val, ...)]<|tool_call_end|>`.
/// Parses Python-literal-style arguments (unquoted strings, True/False/None)
/// into a JSON arguments string. Returns `calls: None` when no marker pair is
/// present so the caller falls back cleanly to plain content.
fn parse_bracket_call(completion: &str) -> ParseOutcome {
    const START: &str = "<|tool_call_start|>";
    const END: &str = "<|tool_call_end|>";
    if !completion.contains(START) {
        return ParseOutcome {
            calls: None,
            diagnostic: None,
        };
    }
    let mut calls = Vec::new();
    let mut diagnostic = None;
    let mut rest = completion;
    while let Some(open) = rest.find(START) {
        let after = &rest[open + START.len()..];
        let Some(close) = after.find(END) else {
            diagnostic = Some("tool_call_start without tool_call_end".to_string());
            break;
        };
        let inner = after[..close].trim();
        // Strip the surrounding [ ] list wrapper.
        let inner = inner.trim_start_matches('[').trim_end_matches(']').trim();
        match parse_fn_literal(inner) {
            Some(call) => calls.push(call),
            None => {
                diagnostic = Some(format!("bracket call unparseable: {inner}"));
            }
        }
        rest = &after[close + END.len()..];
    }
    ParseOutcome {
        calls: if calls.is_empty() { None } else { Some(calls) },
        diagnostic,
    }
}

/// Parse `name(arg=val, ...)` into a `ToolCallMsg`. Arguments use the
/// Python-literal forms LFM2.5 emits: bare identifiers for strings,
/// `True`/`False`/`None`, numbers, and nested lists/objects.
fn parse_fn_literal(s: &str) -> Option<ToolCallMsg> {
    let open = s.find('(')?;
    if !s.ends_with(')') {
        return None;
    }
    let name = s[..open].trim().to_string();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    let args_str = &s[open + 1..s.len() - 1];
    let mut args = serde_json::Map::new();
    for (i, part) in split_top_level(args_str).iter().enumerate() {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, value) = match part.split_once('=') {
            Some((k, v)) => (
                k.trim().trim_matches('"').trim_matches('\'').to_string(),
                v.trim(),
            ),
            None => (format!("arg{i}"), part),
        };
        args.insert(key, py_literal_to_json(value));
    }
    let arguments =
        serde_json::to_string(&serde_json::Value::Object(args)).unwrap_or_else(|_| "{}".into());
    Some(ToolCallMsg {
        id: "call_0".to_string(),
        name,
        arguments,
    })
}

/// Split on top-level commas only (respecting nesting and quotes).
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut in_str: Option<char> = None;
    let mut cur = String::new();
    for c in s.chars() {
        match in_str {
            Some(q) => {
                cur.push(c);
                if c == q {
                    in_str = None;
                }
            }
            None => match c {
                '"' | '\'' => {
                    in_str = Some(c);
                    cur.push(c);
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    cur.push(c);
                }
                ')' | ']' | '}' => {
                    depth = depth.saturating_sub(1);
                    cur.push(c);
                }
                ',' if depth == 0 => parts.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            },
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

/// Convert one Python-literal value token to a JSON value.
fn py_literal_to_json(v: &str) -> serde_json::Value {
    let v = v.trim();
    match v {
        "True" | "true" => serde_json::Value::Bool(true),
        "False" | "false" => serde_json::Value::Bool(false),
        "None" | "null" | "" => serde_json::Value::Null,
        _ => {
            if (v.starts_with('"') && v.ends_with('"') && v.len() >= 2)
                || (v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2)
            {
                serde_json::Value::String(v[1..v.len() - 1].to_string())
            } else if let Ok(i) = v.parse::<i64>() {
                serde_json::json!(i)
            } else if let Ok(f) = v.parse::<f64>() {
                serde_json::json!(f)
            } else if v.starts_with('[') || v.starts_with('{') {
                serde_json::from_str(v).unwrap_or(serde_json::Value::String(v.to_string()))
            } else {
                serde_json::Value::String(v.to_string())
            }
        }
    }
}

fn extract_call_value(v: &serde_json::Value) -> Option<ToolCallMsg> {
    let obj = v.as_object()?;
    // Hermes-style interior: {"name": ..., "arguments": {...}} or
    // {"function": {"name": ..., "arguments": ...}}
    let (name, arguments) = if let Some(func) = obj.get("function") {
        let fo = func.as_object()?;
        (
            fo.get("name")?.as_str()?.to_string(),
            fo.get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
    } else {
        (
            obj.get("name")?.as_str()?.to_string(),
            obj.get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
    };
    // OpenAI wire format: arguments is a JSON-encoded *string*. Normalize an
    // object to its serialized string; keep a plain string as-is.
    let arguments_str = match &arguments {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).ok()?,
    };
    let id = obj
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or("call_0")
        .to_string();
    Some(ToolCallMsg {
        id,
        name,
        arguments: arguments_str,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_format::ChatMessage;

    fn call(name: &str, args: &str) -> ToolCallMsg {
        ToolCallMsg {
            id: "call_0".to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }
    }

    #[test]
    fn bracket_call_parses_lfm25_convention() {
        let out = parse_tool_calls(
            "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>",
            ToolFamily::Auto,
        );
        let calls = out.calls.expect("bracket call must parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].arguments.contains("\"city\""));
        assert!(calls[0].arguments.contains("Paris"));
        // Plain prose still yields no calls.
        assert!(
            parse_tool_calls("It is sunny today", ToolFamily::Auto)
                .calls
                .is_none()
        );
    }

    #[test]
    fn parses_hermes_style_single_call() {
        let completion =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let out = parse_tool_calls(completion, ToolFamily::TagDelimited);
        assert_eq!(
            out.calls,
            Some(vec![call("get_weather", "{\"city\":\"Paris\"}")])
        );
        assert!(out.diagnostic.is_none());
    }

    #[test]
    fn parses_multiple_tag_calls() {
        let completion = concat!(
            "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call>",
            "prefix text",
            "<tool_call>{\"name\":\"b\",\"arguments\":{\"y\":2}}</tool_call>",
        );
        let out = parse_tool_calls(completion, ToolFamily::TagDelimited);
        assert_eq!(
            out.calls,
            Some(vec![call("a", "{\"x\":1}"), call("b", "{\"y\":2}"),])
        );
    }

    #[test]
    fn malformed_interior_falls_back_to_content() {
        let completion = "<tool_call>this is not json</tool_call>";
        let out = parse_tool_calls(completion, ToolFamily::TagDelimited);
        assert_eq!(out.calls, None);
        assert!(out.diagnostic.is_some());
    }

    #[test]
    fn no_tool_call_is_plain_content() {
        let out = parse_tool_calls("The weather in Paris is 72F", ToolFamily::Auto);
        assert_eq!(out.calls, None);
    }

    #[test]
    fn bare_json_convention() {
        let completion = "{\"name\":\"get_time\",\"arguments\":{\"tz\":\"UTC\"}}";
        let out = parse_tool_calls(completion, ToolFamily::BareJson);
        assert_eq!(out.calls, Some(vec![call("get_time", "{\"tz\":\"UTC\"}")]));
    }

    #[test]
    fn bare_json_function_wrapper() {
        let completion = "{\"function\":{\"name\":\"f\",\"arguments\":{}}}";
        let out = parse_tool_calls(completion, ToolFamily::BareJson);
        assert_eq!(out.calls, Some(vec![call("f", "{}")]));
    }

    #[test]
    fn family_resolution_heuristic() {
        assert_eq!(
            resolve_tool_family("{% if tools %}<tool_call>"),
            ToolFamily::TagDelimited
        );
        assert_eq!(
            resolve_tool_family("{{ message.content }}"),
            ToolFamily::Auto
        );
    }

    /// WI-TOOLS-4b: identical call arguments (with reordered keys) must count
    /// as a repeat. Canonicalization normalizes `{"a":1,"b":2}` and
    /// `{"b":2,"a":1}` to the same string.
    #[test]
    fn counts_prior_identical_call_with_reorder() {
        let prior = vec![ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(vec![ToolCallMsg {
                id: "a1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\",\"units\":\"F\"}".into(),
            }]),
            tool_call_id: None,
            name: None,
        }];
        assert_eq!(
            count_prior_identical_calls(
                &prior,
                "get_weather",
                "{\"units\":\"F\",\"city\":\"NYC\"}"
            ),
            1
        );
    }

    /// Zero prior calls must not trigger the guard.
    #[test]
    fn no_prior_calls() {
        let messages: Vec<ChatMessage> = vec![];
        assert_eq!(
            count_prior_identical_calls(&messages, "get_weather", "{}"),
            0
        );
    }

    /// Different arguments must not count as repeats.
    #[test]
    fn different_arguments_not_counted() {
        let prior = vec![ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(vec![ToolCallMsg {
                id: "a1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
            }]),
            tool_call_id: None,
            name: None,
        }];
        assert_eq!(
            count_prior_identical_calls(&prior, "get_weather", "{\"city\":\"LA\"}"),
            0
        );
    }

    /// Different tool names must not count as repeats.
    #[test]
    fn different_tool_name_not_counted() {
        let prior = vec![ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(vec![ToolCallMsg {
                id: "a1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
            }]),
            tool_call_id: None,
            name: None,
        }];
        assert_eq!(
            count_prior_identical_calls(&prior, "get_time", "{\"city\":\"NYC\"}"),
            0
        );
    }

    /// Prior `arguments` that is not valid JSON must not panic the counter —
    /// falls back to raw-string equality.
    #[test]
    fn malformed_prior_arguments_does_not_panic() {
        let prior = vec![ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(vec![ToolCallMsg {
                id: "a1".into(),
                name: "get_weather".into(),
                arguments: "not json".into(),
            }]),
            tool_call_id: None,
            name: None,
        }];
        let n = count_prior_identical_calls(&prior, "get_weather", "not json");
        assert_eq!(n, 1);
    }

    /// WI-TOOLS-4c-i: total prior tool-call count sums every tool_calls entry
    /// across all assistant messages, regardless of name/arguments.
    #[test]
    fn counts_total_prior_tool_calls_across_messages() {
        let messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: Some(vec![
                    ToolCallMsg {
                        id: "a".into(),
                        name: "get_weather".into(),
                        arguments: "{}".into(),
                    },
                    ToolCallMsg {
                        id: "b".into(),
                        name: "get_time".into(),
                        arguments: "{}".into(),
                    },
                ]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: Some(vec![ToolCallMsg {
                    id: "c".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"LA\"}".into(),
                }]),
                tool_call_id: None,
                name: None,
            },
            // Non-assistant / no-calls messages must not contribute.
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        assert_eq!(count_total_prior_tool_calls(&messages), 3);
    }

    #[test]
    fn count_total_zero_when_no_tool_calls() {
        let messages: Vec<ChatMessage> = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        assert_eq!(count_total_prior_tool_calls(&messages), 0);
    }

    #[test]
    fn reason_strings_are_stable() {
        assert_eq!(
            RunawayReason::DuplicateToolCall.as_str(),
            "duplicate_tool_call_limit"
        );
        assert_eq!(
            RunawayReason::TotalToolCallLimit.as_str(),
            "total_tool_call_limit"
        );
        assert_eq!(
            RunawayReason::MessageCountLimit.as_str(),
            "message_count_limit"
        );
    }

    #[test]
    fn test_family_for_arch() {
        assert_eq!(family_for_arch("LFM2.5"), ToolFamily::BracketFirst);
        assert_eq!(family_for_arch("llama-3"), ToolFamily::TagDelimited);
        assert_eq!(family_for_arch("qwen2.5"), ToolFamily::TagDelimited);
        assert_eq!(family_for_arch("deepseek-v3"), ToolFamily::BareJson);
        assert_eq!(family_for_arch("unknown_arch"), ToolFamily::Auto);
    }
}

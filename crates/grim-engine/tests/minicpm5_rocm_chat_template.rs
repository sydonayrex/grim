//! Chat template rendering tests for MiniCPM5 on ROCm/CPU.
//!
//! MiniCPM5 uses an advanced Jinja chat template that exercises many minijinja
//! features (namespace(), |reverse, .items(), is string, | split, etc.). If the
//! template fails to render, raw Jinja syntax leaks into the model output and
//! floods the TUI. These tests verify the template renders cleanly on both
//! ROCm and CPU backends.
//!
//! GPU execution is gated behind `GRIM_RUN_GPU_TESTS=1`.

use grim_format::{ChatMessage, FunctionDef, ToolDef, render_chat_template};

fn minicpm5_model_path() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let workspace_root = std::path::Path::new(&manifest_dir).parent()?.parent()?;
    let p = workspace_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
    if !p.exists() {
        eprintln!(
            "[test-skip] models/MiniCPM5-1B-Q4_K_M.gguf not found at {}",
            p.display()
        );
        return None;
    }
    p.to_str().map(|s| s.to_string())
}

fn get_chat_template(path: &str) -> Option<String> {
    let provider = grim_format::GgufProvider::open(path).ok()?;
    let tmpl = provider.metadata("tokenizer.chat_template")?;
    tmpl.as_str().map(String::from)
}

/// Assert that rendered output contains no raw Jinja/template syntax.
fn assert_no_jinja_leakage(rendered: &str) {
    // Check for Jinja control markers. Note: we do NOT check for standalone
    // `}}` because JSON content like `{"key":"value"}}` legitimately contains `}}`.
    for marker in ["{%-", "{%", "-%}", "%}", "{{", "raise_exception", "namespace(", "set ns"] {
        assert!(
            !rendered.contains(marker),
            "rendered output contains raw Jinja marker '{marker}': {rendered:.200}"
        );
    }
}

/// Assert rendered output is TUI-safe (no excessively long lines).
fn assert_tui_safe(rendered: &str) {
    let max_line_len = rendered.lines().map(|l| l.len()).max().unwrap_or(0);
    assert!(
        max_line_len <= 500,
        "rendered output has a very long line ({} chars) that could break TUI: {}",
        max_line_len,
        rendered.lines().find(|l| l.len() > 500).unwrap_or("")
    );
}

// ===========================================================================
// 1. Simple user message rendering.
// ===========================================================================
#[test]
fn minicpm5_simple_user_message_renders_cleanly() {
    let Some(path) = minicpm5_model_path() else { return };
    let Some(tmpl) = get_chat_template(&path) else {
        eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
        return;
    };

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "What is the capital of France?".into(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];

    let rendered = render_chat_template(
        &tmpl,
        &messages,
        true,   // add_generation_prompt
        "",     // bos_token
        "",     // eos_token
        None,   // tools
        None,   // tool_choice
    )
    .expect("render_chat_template failed for simple user message");

    assert_no_jinja_leakage(&rendered);
    assert_tui_safe(&rendered);
    assert!(
        rendered.contains("<|im_start|>user"),
        "expected user role marker in rendered output: {rendered:.200}"
    );
    assert!(
        rendered.contains("What is the capital of France?"),
        "expected user content in rendered output: {rendered:.200}"
    );
    eprintln!(
        "[minicpm5] Simple user message rendered ({} chars, max line {} chars)",
        rendered.len(),
        rendered.lines().map(|l| l.len()).max().unwrap_or(0)
    );
}

// ===========================================================================
// 2. System + user message rendering.
// ===========================================================================
#[test]
fn minicpm5_system_message_renders_cleanly() {
    let Some(path) = minicpm5_model_path() else { return };
    let Some(tmpl) = get_chat_template(&path) else {
        eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
        return;
    };

    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a helpful assistant.".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".into(),
            content: "Hello!".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    let rendered = render_chat_template(
        &tmpl,
        &messages,
        true,
        "",
        "",
        None,
        None,
    )
    .expect("render_chat_template failed for system + user message");

    assert_no_jinja_leakage(&rendered);
    assert_tui_safe(&rendered);
    assert!(
        rendered.contains("<|im_start|>system"),
        "expected system role marker in rendered output: {rendered:.200}"
    );
    assert!(
        rendered.contains("You are a helpful assistant."),
        "expected system content in rendered output: {rendered:.200}"
    );
    eprintln!(
        "[minicpm5] System + user message rendered ({} chars)",
        rendered.len()
    );
}

// ===========================================================================
// 3. Tool calls rendering.
// ===========================================================================
#[test]
fn minicpm5_tool_calls_renders_cleanly() {
    let Some(path) = minicpm5_model_path() else { return };
    let Some(tmpl) = get_chat_template(&path) else {
        eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
        return;
    };

    let tools = vec![ToolDef {
        r#type: "function".into(),
        function: FunctionDef {
            name: "get_weather".into(),
            description: Some("Get the current weather".into()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            })),
        },
    }];

    let messages = vec![
        ChatMessage {
            role: "user".into(),
            content: "What's the weather in Paris?".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "assistant".into(),
            content: "<tool_sep>".into(),
            tool_calls: Some(vec![grim_format::ToolCallMsg {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}).to_string(),
            }]),
            tool_call_id: None,
            name: None,
        },
    ];

    let rendered = render_chat_template(
        &tmpl,
        &messages,
        false,  // no generation prompt (tool response follows)
        "",
        "",
        Some(&tools),
        None,
    )
    .expect("render_chat_template failed for tool calls");

    assert_no_jinja_leakage(&rendered);
    assert_tui_safe(&rendered);
    assert!(
        rendered.contains("get_weather"),
        "expected tool name in rendered output: {rendered:.200}"
    );
    eprintln!(
        "[minicpm5] Tool calls rendered ({} chars)",
        rendered.len()
    );
}

// ===========================================================================
// 4. Reasoning/thinking content rendering.
// ===========================================================================
#[test]
fn minicpm5_reasoning_content_renders_cleanly() {
    let Some(path) = minicpm5_model_path() else { return };
    let Some(tmpl) = get_chat_template(&path) else {
        eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
        return;
    };

    let messages = vec![
        ChatMessage {
            role: "user".into(),
            content: "Think step by step: 2+2?".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "assistant".into(),
            content: "<think>Let me think. 2+2 = 4.</think>The answer is 4.".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    let rendered = render_chat_template(
        &tmpl,
        &messages,
        false,
        "",
        "",
        None,
        None,
    )
    .expect("render_chat_template failed for reasoning content");

    assert_no_jinja_leakage(&rendered);
    assert_tui_safe(&rendered);
    assert!(
        rendered.contains("<think>"),
        "expected think tag in rendered output: {rendered:.200}"
    );
    eprintln!(
        "[minicpm5] Reasoning content rendered ({} chars)",
        rendered.len()
    );
}

// ===========================================================================
// 5. Long conversation rendering (TUI safety check).
// ===========================================================================
#[test]
fn minicpm5_long_conversation_fits_tui() {
    let Some(path) = minicpm5_model_path() else { return };
    let Some(tmpl) = get_chat_template(&path) else {
        eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
        return;
    };

    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a helpful assistant that provides detailed answers.".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".into(),
            content: "Tell me a very long story about the history of computing, \
                      from the abacus to modern quantum computers, including \
                      all the key milestones and inventors.".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    let rendered = render_chat_template(
        &tmpl,
        &messages,
        true,
        "",
        "",
        None,
        None,
    )
    .expect("render_chat_template failed for long conversation");

    assert_no_jinja_leakage(&rendered);
    assert_tui_safe(&rendered);
    eprintln!(
        "[minicpm5] Long conversation rendered ({} chars, max line {} chars)",
        rendered.len(),
        rendered.lines().map(|l| l.len()).max().unwrap_or(0)
    );
}

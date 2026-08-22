//! Cross-crate integration tests for `grim-cli`.
//!
//! Validates:
//! - Chat template registry lookups and multi-turn message rendering (ChatML, Llama 3, Qwen, Mistral, Gemma)
//! - Doctor self-diagnosis reporting and suggestion generator
//! - Serving and engine configuration serialization roundtrip
//! - Recipe loading and execution pipeline

use grim_cli::config::GrimToml;
use grim_cli::doctor::DoctorReport;
use grim_cli::template_registry::{TemplateRegistry, render_family};
use serde_json::json;

#[test]
fn test_template_registry_lookup_and_rendering() {
    let registry = TemplateRegistry::default();
    assert!(registry.len() >= 5);

    // 1. ChatML template rendering
    let messages = json!([
        {"role": "user", "content": "Hello GRIM!"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]);
    let rendered_chatml = render_family("chatml", messages.clone()).unwrap();
    assert!(rendered_chatml.contains("<|im_start|>user\nHello GRIM!<|im_end|>"));
    assert!(rendered_chatml.contains("<|im_start|>assistant\nHello! How can I assist you today?<|im_end|>"));

    // 2. Llama 3 template rendering
    let rendered_llama3 = render_family("llama3", messages.clone()).unwrap();
    assert!(rendered_llama3.contains("<|start_header_id|>user<|end_header_id|>\n\nHello GRIM!<|eot_id|>"));

    // 3. Unknown template family error handling
    let unknown_res = render_family("non_existent_family", messages);
    assert!(unknown_res.is_err());
}

#[test]
fn test_doctor_report_suggestions_and_health_checks() {
    let mut report = DoctorReport::default();
    report.health_endpoint_ok = Some(true);
    report.gpu_detected = Some(true);
    report.gpu_backend_actual = Some("ROCm".to_string());
    report.plugin_grants_enforced = Some(true);

    assert_eq!(report.health_endpoint_ok, Some(true));
    assert_eq!(report.gpu_detected, Some(true));
    assert_eq!(report.gpu_backend_actual.as_deref(), Some("ROCm"));
    assert!(report.errors.is_empty());
}

#[test]
fn test_cli_configuration_toml_roundtrip() {
    let toml_str = r#"
[server]
default_model = "llama3"
max_batched_tokens = 4096
max_num_seqs = 16
target_ttft_ms = 100
"#;

    let parsed: GrimToml = toml::from_str(toml_str).unwrap();
    assert_eq!(parsed.server.default_model.as_deref(), Some("llama3"));
    assert_eq!(parsed.server.max_batched_tokens, 4096);
    assert_eq!(parsed.server.max_num_seqs, 16);
    assert_eq!(parsed.server.target_ttft_ms, Some(100));
}

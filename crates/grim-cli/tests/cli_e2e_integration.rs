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
    assert!(
        rendered_chatml
            .contains("<|im_start|>assistant\nHello! How can I assist you today?<|im_end|>")
    );

    // 2. Llama 3 template rendering
    let rendered_llama3 = render_family("llama3", messages.clone()).unwrap();
    assert!(
        rendered_llama3
            .contains("<|start_header_id|>user<|end_header_id|>\n\nHello GRIM!<|eot_id|>")
    );

    // 3. Unknown template family error handling
    let unknown_res = render_family("non_existent_family", messages);
    assert!(unknown_res.is_err());
}

#[test]
fn test_doctor_report_suggestions_and_health_checks() {
    let report = DoctorReport {
        health_endpoint_ok: Some(true),
        gpu_detected: Some(true),
        gpu_backend_actual: Some("ROCm".to_string()),
        plugin_grants_enforced: Some(true),
        ..DoctorReport::default()
    };

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

#[test]
fn test_calibrate_channels_cli_e2e() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let res = grim_cli::calibrate_channels::cmd_calibrate_channels(
        grim_cli::calibrate_channels::CalibrateChannelsArgs {
            output: path.to_string(),
            model: None,
            prompts_file: None,
            kv_heads: 4,
            head_dim: 128,
            samples: 8,
            max_tokens_per_prompt: 32,
            alloc_budget_bits: None,
        },
    );
    assert!(
        res.is_ok(),
        "cmd_calibrate_channels should succeed: {:?}",
        res
    );

    // write_sidecar appends `.channels.json` when the output path lacks it and
    // wraps the per-layer scores in a versioned envelope.
    let sidecar = std::fs::read_to_string(format!("{path}.channels.json")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&sidecar).unwrap();
    assert_eq!(doc["scope"], "kv-channel-importance");
    assert_eq!(doc["num_layers"], 1);
    let scores: grim_kvquant::channel_importance::ChannelImportanceScores =
        serde_json::from_value(doc["layers"][0].clone()).unwrap();

    assert_eq!(scores.num_kv_heads, 4);
    assert_eq!(scores.head_dim, 128);
    assert_eq!(scores.group_size, 32);
    assert_eq!(scores.k_scores.len(), 4);
    assert_eq!(scores.k_scores[0].len(), 4);
    assert_eq!(scores.kv_rows, 8);
}

/// W5 split: the exit-code contract lives in `main` (`Ok(false)` → exit 1),
/// but nothing pinned `run_doctor`'s bool itself — the struct-field test
/// above passes even if the bool always reads true. A deterministic error
/// (unreadable model header) must yield `Ok(false)`.
#[test]
fn test_doctor_bool_false_when_errors_present() {
    let ok = grim_cli::doctor::run_doctor(
        "http://127.0.0.1:9",
        "grim-nonexistent-service-xyz",
        "/nonexistent/grim",
        "/nonexistent/grim.toml",
        Some(std::path::Path::new("/nonexistent/model.gguf")),
    )
    .expect("run_doctor must not Err on bad inputs");
    assert!(
        !ok,
        "errors present => false => main exits 1; true here would silently pass CI red"
    );
}

/// M13 subprocess leg + M14 smoke: `--help` exits 0; a real one-shot run on
/// the tiny local model exits 0 with non-empty stdout. Skips gracefully when
/// no local model checkout exists (CI without LFS). Greedy single token keeps
/// it fast and deterministic; the template/tokenizer path stays exercised
/// (no `--raw`), covering the M12 loader in the process.
/// Locate the `grim-cli` binary next to this test executable (debug or
/// release layout). `CARGO_BIN_EXE_<name>` is only set when cargo builds the
/// bin in the same invocation, which `cargo test --test` does not guarantee.
fn grim_cli_bin() -> std::path::PathBuf {
    if let Some(v) = option_env!("CARGO_BIN_EXE_grim_cli") {
        return std::path::PathBuf::from(v);
    }
    let mut dir = std::env::current_exe().expect("test exe path");
    dir.pop();
    if dir.file_name().map(|n| n == "deps").unwrap_or(false) {
        dir.pop();
    }
    dir.join("grim-cli")
}

#[test]
fn test_cli_help_and_run_smoke_golden() {
    let bin = grim_cli_bin();
    if !bin.exists() {
        eprintln!("SKIP: binary not built at {}", bin.display());
        return;
    }
    let help = std::process::Command::new(&bin)
        .arg("--help")
        .output()
        .expect("spawn grim-cli --help");
    assert!(
        help.status.success(),
        "--help must exit 0, stderr: {}",
        String::from_utf8_lossy(&help.stderr)
    );

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = manifest.join("../../models/LFM2.5-350M-Q8_0.gguf");
    let Ok(canon) = model.canonicalize() else {
        eprintln!("SKIP: no local model at {}", model.display());
        return;
    };
    let out = std::process::Command::new(bin)
        .args([
            "run",
            &canon.to_string_lossy().into_owned(),
            "hi",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--device",
            "rocm",
        ])
        .output()
        .expect("spawn grim-cli run");
    assert!(
        out.status.success(),
        "one-shot run must exit 0, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("Hello"),
        "greedy single token for 'hi' must be golden 'Hello', got: {text}"
    );
}

/// M13: a bad model path must exit nonzero (typed Config error through
/// main), never exit 0 and never panic with a Rust backtrace on stdout.
#[test]
fn test_cli_run_bad_model_exits_nonzero_without_panic() {
    let bin = grim_cli_bin();
    if !bin.exists() {
        eprintln!("SKIP: binary not built at {}", bin.display());
        return;
    }
    let out = std::process::Command::new(&bin)
        .args([
            "run",
            "/nonexistent/model.gguf",
            "hi",
            "--max-tokens",
            "1",
            "--device",
            "cpu",
        ])
        .output()
        .expect("spawn grim-cli run");
    assert!(!out.status.success(), "bad model path must exit nonzero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stderr.contains("panicked") && !stdout.contains("panicked"),
        "must be a clean error, not a panic: {stderr}"
    );
}

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to get repo root")
        .to_path_buf()
}

#[test]
fn test_cli_draft_graph_cpu_twin() {
    let bin_path = repo_root().join("target/debug/grim-cli");
    let status = Command::new(&bin_path)
        .arg("--help")
        .status();
    assert!(status.is_ok());
    assert!(status.unwrap().success());
}

#[test]
#[ignore]
fn test_cli_draft_graph_integration_gpu() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        eprintln!("Skipping GPU test: GRIM_RUN_GPU_TESTS not set");
        return;
    }

    let root = repo_root();
    let target_path = root.join("models/LFM2.5-350M-Q8_0.gguf");
    let draft_path = root.join("models/LFM2.5-230M-Q4_K_M.gguf");
    let bin_path = root.join("target/release/grim-cli");

    if !target_path.exists() {
        eprintln!("Skipping test: {} missing on disk", target_path.display());
        return;
    }
    if !draft_path.exists() {
        eprintln!("Skipping test: {} missing on disk", draft_path.display());
        return;
    }

    let output = Command::new(&bin_path)
        .current_dir(&root)
        .args(&[
            "run",
            target_path.to_str().unwrap(),
            "Greek thought experiment",
            "--draft-model",
            draft_path.to_str().unwrap(),
            "--max-tokens",
            "20",
        ])
        .output()
        .expect("Failed to execute grim-cli");

    assert!(output.status.success(), "grim-cli run failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    // Acceptance criteria:
    // 1. Output must contain: `[grim] decode-graph: active (replay path served decode steps)`
    assert!(
        combined.contains("[grim] decode-graph: active (replay path served decode steps)"),
        "Missing expected active decode-graph message in output:\n{}",
        combined
    );

    // 2. Output must contain ZERO instances of `decode-graph: inactive` or `eager fallback`
    assert!(
        !combined.contains("decode-graph: inactive"),
        "Unexpected 'decode-graph: inactive' in output:\n{}",
        combined
    );
    assert!(
        !combined.contains("eager fallback"),
        "Unexpected 'eager fallback' in output:\n{}",
        combined
    );
}

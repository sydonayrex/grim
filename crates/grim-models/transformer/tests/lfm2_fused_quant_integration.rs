use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("Failed to get repo root")
        .to_path_buf()
}

#[test]
fn test_lfm2_fused_quant_cpu_twin() {
    let bin_path = repo_root().join("target/debug/grim-cli");
    let status = Command::new(&bin_path).arg("--help").status();
    assert!(status.is_ok());
    assert!(status.unwrap().success());
}

#[test]
#[ignore]
fn test_lfm2_fused_quant_integration_gpu() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        eprintln!("Skipping GPU test: GRIM_RUN_GPU_TESTS not set");
        return;
    }

    let root = repo_root();
    let target_path = root.join("models/LFM2.5-350M-Q8_0.gguf");
    let bin_path = root.join("target/release/grim-cli");

    if !target_path.exists() {
        eprintln!("Skipping test: {} missing on disk", target_path.display());
        return;
    }

    let output = Command::new(&bin_path)
        .current_dir(&root)
        .args(&[
            "run",
            target_path.to_str().unwrap(),
            "Greek thought experiment",
            "--max-tokens",
            "5",
        ])
        .output()
        .expect("Failed to execute grim-cli");

    assert!(output.status.success(), "grim-cli run failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("[grim] decode-graph: active"),
        "Missing expected active decode-graph in output:\n{}",
        combined
    );
}

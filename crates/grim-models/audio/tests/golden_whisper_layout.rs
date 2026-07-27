use std::process::Command;

/// Verify that `cargo build --lib` for the audio crate produces zero warnings
/// matching the dead-field pattern "field `_` is never read".
///
/// If a `_`-prefixed field is re-introduced to WhisperDecoderBlock or
/// WhisperEncoderBlock, the compiler will warn. This test catches that.
#[test]
fn golden_whisper_no_unused_field_warnings() {
    let output = Command::new("cargo")
        .args(["build", "--lib", "-p", "grim-models-audio"])
        .output()
        .expect("cargo build");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let dead_field_warnings: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("warning: field `") && l.contains("is never read"))
        .collect();

    assert!(
        dead_field_warnings.is_empty(),
        "Found {} dead-field warning(s):\n{}",
        dead_field_warnings.len(),
        dead_field_warnings.join("\n"),
    );
}

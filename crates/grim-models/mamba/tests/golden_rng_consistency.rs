use std::process::Command;

/// SimpleRng from grim_core is deterministic: same seed → same first 8 values.
#[test]
fn golden_rng_deterministic_seed_identity() {
    use grim_core::rng::SimpleRng;

    let mut rng = SimpleRng::new(0xDEAD_BEEF);
    let values: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();

    // Pre-computed by running the canonical xorshift64(13,7,17) once (see grim-core/src/rng.rs).
    // Seed = 0x0000_0000_DEAD_BEEF.
    // Changes here should ONLY happen if the algorithm changes deliberately.
    let expected = [
        0.21785907,
        0.08779941,
        0.66758688,
        0.898_030_3,
        0.24182903,
        0.551_624,
        0.93937888,
        0.38869425,
    ];

    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        let abs = (got - want).abs();
        assert!(
            abs < 1e-7,
            "SimpleRng[0xDEAD_BEEF][{i}]: got {got:.8} want {want:.8} (diff={abs})",
        );
    }
}

/// Verify exactly one SimpleRng definition exists in the crates/grim-models/ tree.
/// The canonical one lives in grim-core/src/rng.rs.
/// Any model crate with its own local copy is a regression.
#[test]
fn golden_rng_remove_duplicate_impls() {
    let output = Command::new("rg")
        .args(["-c", "struct SimpleRng", "crates/grim-models/"])
        .current_dir(workspace_root())
        .output()
        .expect("ripgrep check");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let total: usize = stdout
        .lines()
        .filter_map(|l| {
            let (_file, count) = l.split_once(':')?;
            count.trim().parse::<usize>().ok()
        })
        .sum();

    assert_eq!(
        total, 0,
        "Expected 0 SimpleRng definitions in crates/grim-models/. Found {total}:\n{stdout}",
    );
}

fn workspace_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // mamba/tests → mamba → grim-models → grim → /
    p.pop();
    p.pop();
    p.pop();
    p.pop();
    p
}

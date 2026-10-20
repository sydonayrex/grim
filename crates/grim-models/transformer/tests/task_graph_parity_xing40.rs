//! Plan task M2 — the compatibility oracle, Xing4.0 half.
//!
//! Same assertion as the Llama half (`task_graph_parity.rs`): interpreting the
//! emitted task graph through the model's existing launchers, in the IR's
//! dependency order, must reproduce graph replay. Xing4.0 is the MoE + MLA
//! vehicle: its IR carries MOE_DISPATCH nodes for the routed layers and dense
//! GEMM nodes for the leading dense blocks, so the walk and the QuantFormat
//! axis are exercised by a shape the dense model cannot produce.
//!
//! Separate BINARY from the Llama half on purpose: the 29B checkpoint and the
//! 1B one cannot be resident together on a 17 GB card, and tearing the first
//! down mid-process faulted even under a device guard (recorded). Both files
//! `include!` the same body; only the checkpoint and the tokens differ.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 AMD_SERIALIZE_KERNEL=3 GRIM_CONTEXT=512 \
//!  cargo test -p grim-models-transformer --test task_graph_parity_xing40 -- --nocapture --test-threads=1`

include!("common/task_graph_parity_body.rs");

#[test]
fn task_graph_interpretation_matches_graph_replay() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the M2 oracle");
        return;
    }
    set_var("GRIM_CONTEXT", "512");
    // Both arms run the same launchers; interpret is eager-ordered and replay
    // is the recorded graph, so the bar is fp32-tight. If this ever needs
    // loosening, the IR has stopped describing the walk.
    const INTERPRET_BAR: f32 = 1e-5;
    run_oracle(
        "models/Xing4.0/Xing4.0-29B-A4B-IQ3_M-MTP.gguf",
        "xing40-moe",
        &[128000, 6727],
        INTERPRET_BAR,
    );
}

//! Plan task M2 — the compatibility oracle: interpreting the emitted task
//! graph (per-node dispatch through the model's EXISTING launchers, in the
//! IR's dependency order) must reproduce what replaying the captured HIP
//! graph produces. If this holds for a model, its task graph is faithful and
//! M4's persistent ring has a working reference to be diffed against — before
//! a single new kernel exists.
//!
//! This file is the Llama half. The Xing4.0 half is the sibling
//! `task_graph_parity_xing40.rs` — a separate BINARY, because the two
//! checkpoints cannot be resident together (17 GB card) and tearing the first
//! one down mid-process faulted even under a device guard (recorded). Both
//! `include!` the same body (`common/task_graph_parity_body.rs`); the only
//! thing either file says is which checkpoint and which tokens.
//!
//! The Llama Q4_1 checkpoint is the dense vehicle: its linears are dequantized
//! to f32 at load (the GEMM coverage axis records `0`/native for them) and its
//! head is the checkpoint's real Q6_K (axis id 7).
//!
//! Also asserted here: the emitters FEED the coverage table's QuantFormat axis
//! — every GEMM node's `params[0]` carries the weight's `QuantFormat` ordinal
//! (`0` = native), `params[1]` its output width, and `caps.quant_modes` is
//! derived from them.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 AMD_SERIALIZE_KERNEL=3 GRIM_CONTEXT=512 \
//!  cargo test -p grim-models-transformer --test task_graph_parity -- --nocapture --test-threads=1`
//!
//! `AMD_SERIALIZE_KERNEL=3`: real-checkpoint replay still faults without it
//! (open, recorded in the plan) — serialization supplies the ordering that
//! lets the oracle run; it does not affect what is compared.
//! `GRIM_IR_TRACE=1` prints each dispatched node AND drains the queue at every
//! node boundary, so a device fault names the node whose kernels faulted.

include!("common/task_graph_parity_body.rs");

#[test]
fn task_graph_interpretation_matches_graph_replay() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the M2 oracle");
        return;
    }
    set_var("GRIM_CONTEXT", "512");
    // Both arms run the same launchers; interpret is eager-ordered and replay
    // is the recorded graph, so the bar is fp32-tight. If this ever needs
    // loosening, the IR has stopped describing the walk.
    const INTERPRET_BAR: f32 = 1e-5;
    let _ = run_oracle(
        "models/Llama-3.2-1B-Instruct-Q4_1.gguf",
        "llama-q41",
        &[128000, 11, 322],
        INTERPRET_BAR,
    );
}

//! Plan task M2 — the compatibility oracle, Xing4.0's IQ4_NL sibling half.
//!
//! The coverage table's rows are PER QuantFormat, and the IQ4_NL row was the
//! one no gate could exercise: the MoE proof vehicle on disk
//! (`Xing4.0-29B-A4B-IQ3_M-MTP.gguf`) is Iq3S in its expert banks, so its
//! axis never showed Iq4Nl. The sibling checkpoint
//! (`xing4_0-29b-IQ4_NL.gguf`) IS on disk — the row was unexercised because no
//! gate pointed at it, not because the format was unreachable.
//!
//! Same assertion as the other two halves (`task_graph_parity.rs` for the
//! dense Llama, `task_graph_parity_xing40.rs` for the Iq3S MoE): interpreting
//! the emitted task graph through the model's own launchers, in the IR's
//! dependency order, must reproduce graph replay. This file `include!`s the
//! same body; only the checkpoint, the tokens and the expected format differ.
//!
//! Separate BINARY from the other halves on purpose: the 29B checkpoints and
//! the 1B one cannot be resident together on a 17 GB card, and tearing the
//! first down mid-process faulted even under a device guard (recorded).
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 AMD_SERIALIZE_KERNEL=3 GRIM_CONTEXT=512 \
//!  cargo test -p grim-models-transformer --test task_graph_parity_xing40_iq4nl -- --nocapture --test-threads=1`

include!("common/task_graph_parity_body.rs");

#[test]
fn task_graph_interpretation_matches_graph_replay_iq4nl() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the M2 oracle");
        return;
    }
    set_var("GRIM_CONTEXT", "512");
    const INTERPRET_BAR: f32 = 1e-5;
    let rel = "models/Xing4.0/xing4_0-29b-IQ4_NL.gguf";
    // Pre-flight: a model that does not FIT is not a parity failure — it is
    // a machine that cannot run the row. The IQ4_NL sibling is 20.1 GB of
    // 4.5 bpw weights against a 17.1 GB card, and the failure mode is
    // misleading: the load "succeeds" through the managed-memory fallback
    // and the FIRST KERNEL LAUNCH faults with `vram_free = 0.00` (measured
    // 2026-10-20) — a device fault that looks nothing like "this card is
    // too small". The gate says so instead.
    let path = workspace_path(rel);
    let (free, total) = grim_backend_rocm::device::capability_profiler::vram_info(0);
    let weights = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // The weights plus a gigabyte of headroom for the arena, the activations
    // and the scratch — the model cannot be the last thing on the card.
    let need = weights + (1 << 30);
    if need > free {
        eprintln!(
            "[SKIP] the IQ4_NL sibling needs {need} bytes (weights {weights} + 1 GiB \
             headroom) and the card has {free} free of {total} — this row needs a \
             bigger card, and a model that does not fit is not a parity failure"
        );
        return;
    }
    let axis = run_oracle(rel, "xing40-moe-iq4nl", &[128000, 6727], INTERPRET_BAR);
    // The point of this half: the row. A checkpoint whose weights are in
    // IQ4_NL must put Iq4Nl on the axis, or the coverage table claims a
    // format no model exercises (and a format no emitter has been proven
    // against).
    let iq4nl = grim_tensor::QuantFormat::Iq4Nl as u32;
    assert!(
        axis.contains(&iq4nl),
        "the IQ4_NL checkpoint's axis {axis:?} does not contain Iq4Nl ({iq4nl}) — \
         the coverage row is still unexercised"
    );
    eprintln!("[xing40-moe-iq4nl] the axis is {axis:?} (Iq4Nl = {iq4nl})");
}

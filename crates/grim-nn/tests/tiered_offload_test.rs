//! TDD tests for 3-tier VRAM/DRAM/SSD offload:
//! 1. `OffloadedPleTable`: direct `SplitGgufProvider::get_range` row reads from 28GB PLE table.
//! 2. `ExpertOffloadBanks`: per-bank mixed quant (`Q3K` gate/up + `Q2_0` down) staging.

use grim_nn::{ExpertOffloadBanks, OffloadedPleTable};
use grim_tensor::Device;

const REAL_GGUF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf"
);

#[test]
fn offloaded_ple_table_reads_rows_from_real_28gb_gguf_without_oom() {
    if !std::path::Path::new(REAL_GGUF).exists() {
        return;
    }
    let ple = OffloadedPleTable::open(REAL_GGUF, "per_layer_token_embd.weight", 160)
        .expect("OffloadedPleTable must open 28GB PLE metadata without loading table");
    assert!(ple.num_rows() >= 320_001_446);
    for row in [0usize, 1, 42, 100_000] {
        let v = ple.get_row_f32(row).expect("get_row_f32 must succeed");
        assert_eq!(v.len(), 160);
        assert!(v.iter().all(|x| x.is_finite()), "row {row} has non-finite values");
    }
    // Second read hits DRAM cache
    let v0_again = ple.get_row_f32(0).expect("cached read");
    assert_eq!(v0_again.len(), 160);
}

#[test]
fn expert_offload_banks_supports_mixed_quant_q3k_and_q20() {
    if !std::path::Path::new(REAL_GGUF).exists() {
        return;
    }
    let banks = ExpertOffloadBanks::open(
        REAL_GGUF,
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        512,
    )
    .expect("ExpertOffloadBanks::open must succeed on mixed-quant real GGUF");
    let (gate, up, down) = banks
        .get(0, &Device::Cpu)
        .expect("staging expert 0 on CPU must succeed for mixed Q3K/Q2_0");
    assert_ne!(
        gate.weight.dtype(),
        down.weight.dtype(),
        "gate (Q3K) and down (Q2_0) must preserve their distinct per-bank dtypes"
    );
    assert_eq!(gate.weight.dtype(), up.weight.dtype());
}

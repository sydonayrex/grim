//! M=1 step-by-step decode perplexity harness (skeleton).
//!
//! The 18-line stub this file replaced imported APIs that moved or never
//! existed (`grim_core::session::SessionInner`, `grim_nn::VarBuilder`,
//! `grim_models::...::qwen35::Qwen35`), which broke every plain
//! `cargo test -p grim-backend-rocm` build. The real per-step PPL harness
//! lives in the CLI: `grim-cli` drives the same load path
//! (`grim_engine::model_loader::load_model_from_gguf`) with graph capture,
//! so the driver-side numbers are authoritative. The production 27B
//! measurement this skeleton anticipated is implemented as a gate:
//! `tests/qwen35_graph_capture.rs::qwen35_production_27b_route_wall_and_utilization`
//! (route dump, cold-flush wall, utilization band).

use grim_backend_rocm::RocmDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== M=1 STEP-BY-STEP DECODE PERPLEXITY HARNESS ===");
    // The harness is a skeleton: two-card residency probe only. The per-step
    // PPL measurement itself is the CLI's job (see module doc above).
    let dev0 = RocmDevice::try_new(0)?;
    println!("device 0: {}", dev0.gcn_arch());
    match RocmDevice::try_new(1) {
        Ok(dev1) => println!("device 1: {}", dev1.gcn_arch()),
        Err(e) => println!("device 1 unavailable: {e}"),
    }
    println!("(measurement moved to qwen35_production_27b_route_wall_and_utilization)");
    Ok(())
}

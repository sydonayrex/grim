use grim_backend_rocm::RocmDevice;
use grim_core::session::SessionInner;
use grim_models::transformer::qwen35::Qwen35;
use grim_nn::VarBuilder;
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== M=1 STEP-BY-STEP DECODE PERPLEXITY HARNESS ===");
    // Pinned 32:32 split
    let dev0 = RocmDevice::try_new(0)?;
    let dev1 = RocmDevice::try_new(1)?;

    println!("Loading model on Device 0 and Device 1...");
    // We will run this through grim-cli harness directly for full graph capture integration
    Ok(())
}

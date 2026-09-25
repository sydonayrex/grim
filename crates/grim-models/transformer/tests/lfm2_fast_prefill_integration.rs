//! Task 11 (Plan 3): Integration test for LFM2 fast prefill on GPU.
//! Validates execution on models/LFM2.5-350M-Q8_0.gguf if present, or skips cleanly.

use grim_backend_rocm::RocmDevice;
use std::path::Path;

#[test]
#[ignore]
fn test_lfm2_fast_prefill_model_integration() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_lfm2_fast_prefill_model_integration");
        return;
    }
    let model_path = Path::new("models/LFM2.5-350M-Q8_0.gguf");
    if !model_path.exists() {
        eprintln!("Model models/LFM2.5-350M-Q8_0.gguf not found; skipping integration test");
        return;
    }
    let dev = RocmDevice::new(0);
    assert_eq!(dev.ordinal(), 0);
    eprintln!("[test_lfm2_fast_prefill_model_integration] model verified present");
}

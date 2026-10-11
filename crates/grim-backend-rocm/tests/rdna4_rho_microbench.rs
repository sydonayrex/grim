//! rho = Tensor-Core : CUDA-core throughput on the running GPU (memory-free
//! loops) — decides whether dequant-on-the-fly (grim's fused fills) is
//! structurally right for this silicon, per the APEX4 framework (old/res4).
//!
//! RUN: GRIM_RUN_GPU_TESTS=1 HIP_VISIBLE_DEVICES=1 cargo test -p grim-backend-rocm \
//!   --test rdna4_rho_microbench -- --ignored --nocapture

use grim_backend_rocm::RocmDevice;

#[test]
#[ignore]
fn rdna4_rho() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() { return; }
    let dev = RocmDevice::try_new(0).unwrap();
    let (cc, tc) = grim_backend_rocm::rho_microbench_for_test(&dev, 20000).unwrap();
    let rho = tc / cc;
    eprintln!("[rho] fp32 VALU: {cc:.1} TFLOP/s | wmma f16 TC: {tc:.1} TFLOP/s | rho = {rho:.2}");
    eprintln!("[rho] rho >> 1 favours pure-int tiles (W4A8/W4A4); rho ~ 1 favours grim's fused-dequant fills.");
    assert!(cc > 1.0 && tc > 1.0, "microbench failed (cc={cc}, tc={tc})");
}

//! The WhiteRaven dispatch arm must actually launch the blocked kernel.
//!
//! Alone in its own test binary on purpose: the route counter is process-wide,
//! so any other test in this binary dispatching the same kernel would make an
//! exact `+1` assertion read as a failure (or, worse, mask a genuine zero).
//! `cargo test` runs test binaries sequentially, so a dedicated binary is what
//! makes the delta meaningful.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

#[test]
fn dispatch_routes_to_the_blocked_kernel_and_not_the_row_major_one() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let (m, n, k) = (1usize, 256usize, 128usize);

    let mut s = 0xD15EA7Cu64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, n, k).expect("block");
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blocked,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
        },
    )
    .map_err(|e| format!("b blocked h2d: {e}"))?;

    grim_backend_rocm::reset_kernel_route_counters();
    let blocked_before =
        grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3_blocked");
    let rowmajor_before = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3");
    // The act prologue is a separate entry; graph decode depends on it, so its
    // count is part of what this route claims.
    let prologue_before = grim_backend_rocm::rocm_kernel_route_counter("grim_quant_fp8_pad16");

    let (_out, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Fp8Blocked16,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();

    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3_blocked"),
        blocked_before + 1,
        "dispatch must launch the blocked kernel exactly once"
    );
    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3"),
        rowmajor_before,
        "blocked dispatch must not also launch the row-major kernel -- B would \
         be read in the wrong layout"
    );
    // The eager dispatch converts A on the host, so the prologue must NOT run
    // here. It runs only in graph capture (linear_decode_blocked_into).
    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_quant_fp8_pad16"),
        prologue_before,
        "the eager path must not use the capture-only act prologue"
    );
    Ok(())
}

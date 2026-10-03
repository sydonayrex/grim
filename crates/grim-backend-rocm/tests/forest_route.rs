//! The ForestRaven dispatch arm must actually launch the int8 dot4 kernel.
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
fn dispatch_routes_to_the_forest_kernel() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    //ARCH: the kernel is gated on RDNA3/4 (sudot4). A runner on older iron
    // takes the host fallback, which is correct but launches no kernel --
    // asserting +1 there would fail for the wrong reason.
    let arch = dev.gcn_arch().to_string();
    if !(arch.starts_with("gfx11") || arch.starts_with("gfx12")) {
        return Ok(());
    }
    let (m, n, k) = (1usize, 256usize, 128usize);

    let mut s = 0xF02E57u64;
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

    // Framed blob, built the way convert builds it.
    let (codes, scales) =
        grim_quant::quant_forest_per_channel(&w, n, k).map_err(|e| format!("quant: {e}"))?;
    let mut blob = Vec::with_capacity(16 + codes.len() + scales.len());
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    blob.extend_from_slice(&scales);
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(grim_tensor::BlockDtype::Int8PerChannel),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    grim_backend_rocm::reset_kernel_route_counters();
    let forest_before = grim_backend_rocm::rocm_kernel_route_counter("grim_dot4_forest_gemv");

    let (_out, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Int8PerChannel,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();

    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_dot4_forest_gemv"),
        forest_before + 1,
        "dispatch must launch the ForestRaven kernel exactly once (if it took \
         the host fallback instead, the blob gate or arch gate misclassified)"
    );
    Ok(())
}

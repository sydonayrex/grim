//! The GreyRaven-HW dispatch arm must actually launch the SWMMAC kernel.
//!
//! Alone in its own test binary on purpose: the route counter is process-wide.
//! Gated on gfx12 (RDNA4/UDNA carry V_SWMMAC; nothing else does): elsewhere
//! the arm takes the host fallback, which is correct but launches no kernel.

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
fn dispatch_routes_to_the_grey_kernel() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    if !dev.gcn_arch().starts_with("gfx12") {
        return Ok(());
    }
    let (m, n, k) = (16usize, 16usize, 32usize);

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.007) - 0.3).collect();

    // HW blob the way convert builds it: coupled patterns + HW pack.
    let pats = grim_quant::grey_raven::coupled_patterns_2_4(&w, n, k)
        .map_err(|e| format!("couple: {e}"))?;
    let blob = grim_quant::grey_raven::pack_grey_raven_hw(&w, n, k, &pats)
        .map_err(|e| format!("pack: {e}"))?;

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
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(grim_tensor::BlockDtype::Fp8Sparse24Hw),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    grim_backend_rocm::reset_kernel_route_counters();
    let before = grim_backend_rocm::rocm_kernel_route_counter("grim_grey_raven_gemm");

    let (_out, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Fp8Sparse24Hw,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();

    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_grey_raven_gemm"),
        before + 1,
        "dispatch must launch the GreyRaven kernel exactly once"
    );
    Ok(())
}

//! Route-through-`quantized_matmul` parity for the shimmed GPTQ kernel.
//!
//! The GPTQ fused dequant-GEMM (`grim_gptq_dequant_gemm`) is exposed to
//! integration tests through `device::gptq_test_shim` (gated behind the
//! `gpu-test-shims` feature) and exercised directly by
//! `tests/gptq_gemm_gpu_parity.rs`. This file adds the production-dispatch
//! route: the same packed blob goes through
//! `QuantOps::quantized_matmul` with `DTypeStorage::GroupInt` and must land
//! in the same `grim_gptq_dequant_gemm` kernel (route counter + numeric
//! parity), imitating the `grey_raven_route` pattern.
//!
//! The route is reachable: `src/device/quant/mod.rs`
//! (`DTypeStorage::GroupInt` arm) calls `gptq_segment_offsets` +
//! `launch_gptq_dequant_gemm`, so no structural-note fallback is needed.
//! (AWQ is symmetric — `DTypeStorage::Awq` arm + `awq_segment_offsets` —
//! and already has `tests/awq_gemm_gpu_parity.rs` for the direct path;
//! GPTQ is the one picked to avoid a third copy of the blob builder.)
//!
//! Alone in its own test binary on purpose: the route counter is process-wide.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendStorage, CoreTensorOps, DType, MemoryOps, QuantOps, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// Same blob builder as `tests/gptq_gemm_gpu_parity.rs` (`build_blob`),
/// copied because integration tests cannot share code across binaries.
fn build_blob(k: usize, n: usize, bits: u8, group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let bits = bits as usize;
    let vpw: usize = match bits {
        2 => 16,
        4 => 8,
        _ => 1,
    };
    let mask = (1u32 << bits) - 1;
    let groups = k.div_ceil(group_size);

    let mut qweight = vec![0u32; k.div_ceil(vpw) * n];
    let mut qzeros = vec![0u32; groups * n.div_ceil(vpw)];
    let mut scales = vec![0f32; groups * n];
    let mut oracle_w = vec![0f32; k * n];

    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for g in 0..groups {
        for ni in 0..n {
            let true_zero = 1.0 + (rnd() % 8) as f32;
            let scale = 0.25 + (rnd() % 16) as f32 / 64.0;
            scales[g * n + ni] = scale;
            let zw = g * n.div_ceil(vpw) + ni / vpw;
            let zoff = ni % vpw;
            qzeros[zw] |= ((true_zero as u32 - 1) & mask) << (zoff * bits);
        }
    }
    for ki in 0..k {
        let g = ki / group_size;
        for ni in 0..n {
            let code = (rnd() % (mask as u64)) as u32;
            let w = (ki / vpw) * n + ni;
            qweight[w] |= code << ((ki % vpw) * bits);
            let zero = (((qzeros[g * n.div_ceil(vpw) + ni / vpw] >> ((ni % vpw) * bits)) & mask) + 1) as f32;
            oracle_w[ki * n + ni] = (code as f32 - zero) * scales[g * n + ni];
        }
    }

    let qw_bytes: Vec<u8> = qweight.iter().flat_map(|w| w.to_le_bytes()).collect();
    let qz_bytes: Vec<u8> = qzeros.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sc_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut blob = Vec::with_capacity(32 + qw_bytes.len() + qz_bytes.len() + sc_bytes.len());
    blob.extend_from_slice(&(qw_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qw_bytes);
    blob.extend_from_slice(&(qz_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qz_bytes);
    blob.extend_from_slice(&(sc_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&sc_bytes);
    blob.extend_from_slice(&0u64.to_le_bytes());
    (blob, oracle_w)
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn gptq_route_through_quantized_matmul_matches_oracle_and_kernel() {
    let Some(dev) = gpu_device() else { return };
    const M: usize = 3;
    const K: usize = 64;
    const N: usize = 48;
    const BITS: u8 = 4;
    const GS: usize = 32;

    let (blob, oracle_w) = build_blob(K, N, BITS, GS);
    let a_data: Vec<f32> = (0..M * K)
        .map(|i| ((i % 11) as f32) * 0.125 - 0.625)
        .collect();

    let a_t = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::new(vec![M, K]), DType::F32)
        .expect("a h2d");
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![N, K]),
        DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::dtype::Storage::GroupInt(grim_tensor::dtype::GpuIntConfig {
                bits: BITS,
                group_size: GS,
                scheme: grim_tensor::dtype::GroupQuantScheme::Asymmetric,
                desc_act: false,
            }),
        },
    )
    .expect("b h2d");

    grim_backend_rocm::reset_kernel_route_counters();
    let before = grim_backend_rocm::rocm_kernel_route_counter("grim_gptq_dequant_gemm");

    let (out, handle) = dev
        .quantized_matmul(
            a_t.as_ref(),
            b_t.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K, // ignored; dispatch keys on storage
            &Shape::new(vec![M, N]),
        )
        .expect("quantized_matmul dispatch");
    handle.synchronize().expect("sync");

    assert_eq!(
        grim_backend_rocm::rocm_kernel_route_counter("grim_gptq_dequant_gemm"),
        before + 1,
        "quantized_matmul must route GroupInt through grim_gptq_dequant_gemm exactly once"
    );

    let got = out.to_cpu_vec_f32().expect("download C");
    // BackendStorage is in scope for the method above; the underscore keeps
    // the trait's import from tripping `-D unused-imports` when the method
    // resolves through the inherent RocmStorage path instead.
    let _ = core::any::type_name::<dyn BackendStorage>();
    let mut worst = 0.0f32;
    for m in 0..M {
        for n in 0..N {
            let want: f32 = (0..K)
                .map(|kk| a_data[m * K + kk] * oracle_w[kk * N + n])
                .sum();
            let rel = (got[m * N + n] - want).abs() / want.abs().max(1.0);
            if rel > worst {
                worst = rel;
            }
        }
    }
    println!("gptq route parity: worst rel error {worst:.3e}");
    assert!(worst <= 2e-3, "GPTQ route parity worse than house dot4 bar: {worst:.3e}");
}

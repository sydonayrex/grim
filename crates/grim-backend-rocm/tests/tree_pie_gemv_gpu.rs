//! A6 — TreePie 5.0 bpw decode + GEMV on the GPU, checked against a CPU oracle.
//!
//! What this establishes: the in-register TreePie decode reproduces the *host*
//! `e2m2_to_fp16_bits` bit-for-bit on real silicon, and the `V_DOT2_F32_F16`
//! accumulation over those decoded halves matches an f32 reference.
//!
//! Two things are deliberately *not* claimed here:
//!   - Accuracy vs E4M3. That is A4's number and it is a CPU measurement; this
//!     test compares GPU against a CPU oracle fed the *same* TreePie codes, so
//!     it isolates the kernel, not the format's error.
//!   - End-to-end viability. ~5% weight RMSE may be survivable or fatal
//!     depending on the model; only A7's real forward pass settles that.
//!
//! RUN: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!      cargo test -p grim-backend-rocm --test tree_pie_gemv_gpu -- --ignored

use grim_backend_rocm::RocmDevice;
use grim_quant::tree_pie::{pack_tree_pie_32, TREE_PIE_WORDS_PER_32};
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Deterministic LCG: a time- or address-dependent fixture would make a failure
/// unreproducible.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-1, 1) then scale into the range TreePie represents well.
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        (unit - 0.5) * 2.0 * 6.0
    }
}

/// Round through f16, which is what the GPU actually consumes on both operands:
/// activations arrive as packed f16 and the decoded TreePie halves *are* f16.
fn round_to_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

#[test]
fn tree_pie_dot2_gpu_matches_cpu() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };

    // K must be a multiple of 32 (the 5.0 bpw packing granularity); N is the
    // decode output width. Both are decode-shaped: M == 1.
    const K: usize = 1024;
    let mut rng = Lcg(0x7EE71E5EED as u64);

    for &n in &[64usize, 257, 1024] {
        // --- fixture -------------------------------------------------------
        let a: Vec<f32> = (0..K).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * K).map(|_| rng.next_f32()).collect();

        // Pack B as TreePie: 5 i32 per 32 values, N columns of ceil(K/32) groups.
        let groups_per_col = K / 32;
        let mut packed = vec![0i32; n * groups_per_col * TREE_PIE_WORDS_PER_32];
        for col in 0..n {
            for g in 0..groups_per_col {
                let mut blk = [0f32; 32];
                for (i, slot) in blk.iter_mut().enumerate() {
                    *slot = b[col * K + g * 32 + i];
                }
                let words = pack_tree_pie_32(&blk);
                let base = (col * groups_per_col + g) * TREE_PIE_WORDS_PER_32;
                packed[base..base + TREE_PIE_WORDS_PER_32].copy_from_slice(&words);
            }
        }

        // --- upload -------------------------------------------------------
        let act_f16: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let act_t = MemoryOps::from_cpu_bytes(
            &dev,
            &act_bytes(&act_f16),
            &Shape::new(vec![K]),
            DType { arith: ArithType::F16, storage: Storage::Native },
        )
        .map_err(|e| format!("act h2d: {e}"))?;
        let b_t = MemoryOps::from_cpu_bytes(
            &dev,
            &as_i32_bytes(&packed),
            &Shape::new(vec![packed.len()]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )
        .map_err(|e| format!("b h2d: {e}"))?;
        let out_t = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![n]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )
        .map_err(|e| format!("out alloc: {e}"))?;

        // --- launch -------------------------------------------------------
        dev.launch_tree_pie_gemv(
            grim_backend_rocm::as_rocm(act_t.as_ref()).unwrap(),
            grim_backend_rocm::as_rocm(b_t.as_ref()).unwrap(),
            grim_backend_rocm::as_rocm(out_t.as_ref()).unwrap(),
            n,
            K,
        )
        .map_err(|e| format!("launch: {e}"))?;
        dev.synchronize();

        let got = download_f32(&out_t)?;

        // --- oracle -------------------------------------------------------
        // The oracle must read the *decoded* weights, not the f32 source, or it
        // would be scoring the quantizer instead of the kernel.
        let mut worst_rel = 0.0f32;
        let mut worst_abs = 0.0f32;
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..K {
                // Decode this column's TreePie codes back to the exact f16 the
                // kernel reconstructed, so the only difference left is the dot.
                let g = kk / 32;
                let j = kk % 32;
                let base = (col * groups_per_col + g) * TREE_PIE_WORDS_PER_32;
                let payload = [packed[base], packed[base + 1], packed[base + 2], packed[base + 3]];
                let signs = packed[base + 4] as u32;
                let nib = ((payload[j >> 3] >> ((j & 7) * 4)) & 0xF) as u8;
                let code = ((((signs >> j) & 1) as u8) << 4) | nib;
                let w = round_to_f16(grim_quant::tree_pie::e2m2_to_f32(code));
                acc += round_to_f16(a[kk]) * w;
            }
            let expect = acc;
            let d = (got[col] - expect).abs();
            worst_abs = worst_abs.max(d);
            let denom = expect.abs().max(1e-3);
            worst_rel = worst_rel.max(d / denom);
        }

        // 2e-2 matches the f16 rationale in wmma_gemm_cpu_gpu_parity.rs: the
        // kernel accumulates in f32 but both operands are f16, so a long K
        // accumulates real rounding.
        assert!(
            worst_rel <= 2e-2,
            "n={n}: worst relative error {worst_rel:.3e} (abs {worst_abs:.3e}) exceeds 2e-2"
        );
        eprintln!("n={n:<5} worst_rel={worst_rel:.3e} worst_abs={worst_abs:.3e}");
    }
    Ok(())
}

fn act_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_i32_bytes(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn download_f32(t: &Box<dyn BackendStorage>) -> TestResult<Vec<f32>> {
    let bytes = grim_backend_rocm::as_rocm(t.as_ref()).map_err(|e| e.to_string())?.copy_to_host()?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

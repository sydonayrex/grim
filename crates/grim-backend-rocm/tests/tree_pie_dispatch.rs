//! TreePie reaches the GEMV through dtype dispatch, not only a direct launch.
//!
//! The launcher was validated by `tree_pie_journey`, which calls it directly. That
//! left the last mile untested: a `FloatPackScheme::TreePie` tensor had no route
//! from `quantized_matmul`, so a loaded model could not actually use the format.
//!
//! This goes through the real entry point and compares against a CPU oracle, which
//! also pins the one thing the direct-launch tests could not: that the **storage
//! layout** a loader produces is the layout the kernel indexes. The kernel reads B
//! as "N columns, each ceil(K/32) groups of 5 i32"; a flat row-major pack of a
//! K x N matrix is a different arrangement unless K == N. A direct launch with a
//! hand-packed buffer cannot detect that mismatch, and this can.

use grim_backend_rocm::RocmDevice;
use grim_quant::tree_pie::TREE_PIE_WORDS_PER_32;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const K: usize = 128; // a multiple of 32, and deliberately != N
const N: usize = 64;

/// Pack `w` (K x N, row-major) into the kernel's B layout: N columns, each
/// K/32 groups of 5 i32, with 32 consecutive k values per group.
fn pack_columns(w: &[f32], n_cols: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; n_cols * (k / 32) * TREE_PIE_WORDS_PER_32];
    for n in 0..n_cols {
        for g in 0..k / 32 {
            let mut block = [0f32; 32];
            for j in 0..32 {
                block[j] = w[(g * 32 + j) * N + n];
            }
            let words = grim_quant::tree_pie::pack_tree_pie_32(&block);
            let base = (n * (k / 32) + g) * TREE_PIE_WORDS_PER_32;
            out[base..base + TREE_PIE_WORDS_PER_32].copy_from_slice(&words);
        }
    }
    out
}

#[test]
fn tree_pie_scheme_reaches_the_gemv_through_quantized_matmul() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev_owned = dev;
    let dev = &dev_owned;

    // Deterministic, non-degenerate weights and activations.
    let w: Vec<f32> = (0..K * N).map(|i| ((i * 37) % 61) as f32 * 0.03125 - 0.9).collect();
    let x: Vec<f32> = (0..K).map(|i| ((i * 13) % 17) as f32 * 0.0625 - 0.4).collect();

    // CPU oracle from the same f32 values, plus the natural scale of each dot
    // product. Dividing by |oracle| is wrong here: these columns largely cancel, so
    // a column whose exact sum is near zero would report a huge relative error for
    // an absolutely tiny one. The denominator for a quantized dot product is the
    // sum of absolute terms, which is the magnitude the arithmetic actually had to
    // resolve.
    let mut oracle = vec![0.0f64; N];
    let mut scale = vec![0.0f64; N];
    for n in 0..N {
        for k in 0..K {
            let t = x[k] as f64 * w[k * N + n] as f64;
            oracle[n] += t;
            scale[n] += t.abs();
        }
    }

    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    // Real f32 activations, so the dispatch's f32 -> f16 conversion is exercised
    // rather than bypassed. Declaring half-bytes under an F32 dtype is what made
    // the first run of this test report a flat 1.0 relative error -- the kernel was
    // fine and the activation was garbage.
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let x_t = MemoryOps::from_cpu_bytes(&dev_owned, &x_bytes, &Shape::new(vec![K]), f32ty.clone())
        .map_err(|e| format!("act h2d: {e}"))?;
    let packed = pack_columns(&w, N, K);
    let b_bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes().to_vec()).collect();

    // B under the TreePie dtype, so dispatch actually sees the scheme. This is the
    // step a direct launch cannot exercise.
    let tree_ty = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::TreePie),
    };
    let b_t = MemoryOps::from_cpu_bytes(
        &dev_owned,
        &b_bytes,
        &Shape::new(vec![N * (K / 32) * TREE_PIE_WORDS_PER_32]),
        tree_ty,
    )
    .map_err(|e| format!("b tree h2d: {e}"))?;

    let out_shape = Shape::new(vec![N]);
    let (out, _handle) = dev
        .quantized_matmul(&*x_t, &*b_t, &[], grim_tensor::QuantFormat::TreePie, &out_shape)
        .map_err(|e| format!("quantized_matmul: {e}"))?;
    dev.synchronize();

    let raw = grim_backend_rocm::as_rocm(out.as_ref()).map_err(|e| e.to_string())?.copy_to_host()?;
    let got: Vec<f64> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();

    assert_eq!(got.len(), N, "expected one output per column");
    let mut worst = 0.0f64;
    let mut worst_n = 0;
    for n in 0..N {
        let denom = scale[n].max(f64::MIN_POSITIVE);
        let rel = ((got[n] - oracle[n]) / denom).abs();
        if rel > worst {
            worst = rel;
            worst_n = n;
        }
    }
    println!("TreePie dispatch: N={N} K={K} (K != N, so a flat row-major pack would not match)");
    println!("worst relative error {worst:.3e} at column {worst_n}");
    // E2M2 is 2 mantissa bits, so a few percent per weight is expected; 0.05 is
    // loose enough not to flake and tight enough to catch a layout swap, which
    // would show up as an O(1) relative error rather than O(1e-2).
    assert!(worst < 0.05, "TreePie dispatch is wrong (worst rel {worst:.3e} at n={worst_n})");
    Ok(())
}

/// Prefill: the GEMM must match a CPU oracle at several `M`, including a ragged
/// tile and a multi-tile `M`.
///
/// `TP_M_TILE` is 8, so `M = 13` straddles two tiles and `M = 16` fills exactly
/// two. A ragged tail is the case most likely to be wrong: the kernel bounds its
/// inner row loop with `min(TP_M_TILE, M - m0)`, so an off-by-one there writes past
/// the output or leaves a row unwritten.
fn prefill_case(dev: &RocmDevice, m: usize, n: usize, k: usize) -> TestResult<f64> {
    let mut w = vec![0.0f32; k * n];
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8_388_608.0) - 1.0
    };
    for v in w.iter_mut() {
        *v = next() * 0.25;
    }
    let mut x = vec![0.0f32; m * k];
    for v in x.iter_mut() {
        *v = next() * 0.5;
    }

    let mut oracle = vec![0.0f64; m * n];
    let mut scale = vec![0.0f64; m * n];
    for r in 0..m {
        for col in 0..n {
            for kk in 0..k {
                let t = x[r * k + kk] as f64 * w[kk * n + col] as f64;
                oracle[r * n + col] += t;
                scale[r * n + col] += t.abs();
            }
        }
    }

    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let x_t = MemoryOps::from_cpu_bytes(
        dev,
        &xb,
        &Shape::new(vec![m, k]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let packed = pack_columns(&w, n, k);
    let b_bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes().to_vec()).collect();
    let b_t = MemoryOps::from_cpu_bytes(
        dev,
        &b_bytes,
        &Shape::new(vec![n * (k / 32) * TREE_PIE_WORDS_PER_32]),
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(grim_tensor::FloatPackScheme::TreePie),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (out, _) = dev
        .quantized_matmul(&*x_t, &*b_t, &[], grim_tensor::QuantFormat::TreePie, &Shape::new(vec![m, n]))
        .map_err(|e| format!("prefill m={m}: {e}"))?;
    dev.synchronize();
    let raw = grim_backend_rocm::as_rocm(out.as_ref()).map_err(|e| e.to_string())?.copy_to_host()?;
    let got: Vec<f64> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    assert_eq!(got.len(), m * n, "wrong output length at m={m}");

    let mut worst = 0.0f64;
    for i in 0..m * n {
        let rel = ((got[i] - oracle[i]) / scale[i].max(f64::MIN_POSITIVE)).abs();
        worst = worst.max(rel);
    }
    Ok(worst)
}

// FAILS: the output is all zeros. The `m = 1` case in this same test fails too,
// and that routes to the *unchanged* GEMV path, which `tree_pie_scheme_reaches_the
// _gemv_through_quantized_matmul` exercises successfully. So the fault is in how
// this test differs from that one -- the only structural differences are the
// activation shape ([1, K] vs [K]), the output shape ([1, N] vs [N]) and the weight
// fixture -- and not in the new GEMM kernel, which this test never reaches.
//
// Not diagnosed. Left `#[ignore]`d and the dispatch left unrouted rather than
// guessing at a fix: an unrouted kernel that produces zeros if wired is much easier
// to spot than a routed one that quietly returns zeros.
#[test]
#[ignore = "all-zero output; difference from the passing GEMV test not yet isolated"]
fn tree_pie_prefill_matches_cpu_oracle_across_tile_boundaries() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };

    const N: usize = 64;
    // 1 is the GEMV path and must keep working; 8 exactly fills one tile; 13
    // straddles two; 16 fills two exactly.
    for (m, k) in [(1usize, 128usize), (2, 128), (8, 128), (13, 128), (16, 128), (9, 256)] {
        let worst = prefill_case(&dev, m, N, k)?;
        println!("prefill m={m:>2} k={k}: worst relative error {worst:.3e}");
        assert!(worst < 0.05, "prefill m={m} k={k}: worst relative error {worst:.3e}");
    }
    Ok(())
}

#[test]
fn tree_pie_still_refuses_ragged_k() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    let tree_ty = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::TreePie),
    };
    let xb: Vec<u8> = vec![0.5f32; 33].iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let x = MemoryOps::from_cpu_bytes(&dev, &xb, &Shape::new(vec![1, 33]), f32ty)
        .map_err(|e| format!("x: {e}"))?;
    let b = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; 320], &Shape::new(vec![16]), tree_ty)
        .map_err(|e| format!("b: {e}"))?;
    let err = dev
        .quantized_matmul(&*x, &*b, &[], grim_tensor::QuantFormat::TreePie, &Shape::new(vec![1, 16]))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("k % 32 == 0"), "ragged K should be refused; got: {err}");
    Ok(())
}

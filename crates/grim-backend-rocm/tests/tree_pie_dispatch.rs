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
    for (i, v) in w.iter_mut().enumerate() {
        // Deterministic O(1) weights for every m (same construction as the
        // passing GEMV test): E2M2's smallest nonzero magnitude is 0.25, so a
        // small-range fixture packs to all-zero payloads, and uniform ±1 data
        // carries ~7.6e-2 of pure quant noise (plan App. A4: 0.0517 weight
        // RMSE), which a 0.05 bar misreads as a kernel defect. One
        // distribution, one bar, both paths.
        *v = ((i as f32 * 37.0) % 61.0) * 0.03125 - 0.9;
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
    let a_shape = Shape::new(vec![m, k]);
    // GRIM_TP_F16_ACT uploads activations already in f16, which takes the dispatch's
    // "already f16, pass through untouched" branch and bypasses the host conversion
    // entirely. If the GEMM works there and faults with f32 activations, the
    // conversion path is confirmed as the cause rather than the kernel.
    let x_t = if std::env::var_os("GRIM_TP_F16_ACT").is_some() {
        let bits: Vec<u8> = x.iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes().to_vec())
            .collect();
        MemoryOps::from_cpu_bytes(
            dev,
            &bits,
            &a_shape,
            DType { arith: ArithType::F16, storage: Storage::Native },
        )
        .map_err(|e| format!("act f16 h2d: {e}"))?
    } else {
        MemoryOps::from_cpu_bytes(
            dev,
            &xb,
            &a_shape,
            DType { arith: ArithType::F32, storage: Storage::Native },
        )
        .map_err(|e| format!("act h2d: {e}"))?
    };
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

    let o_shape = Shape::new(vec![m, n]);
    let (out, _) = dev
        .quantized_matmul(&*x_t, &*b_t, &[], grim_tensor::QuantFormat::TreePie, &o_shape)
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

// Diagnosis history, kept because three of the four steps were wrong and the
// record is more useful than a conclusion would be.
//
//   1.-3. (unchanged -- see git history for df4b1d22, 4fdf2ce8, 27f882fc.)
//
//   4. With prefill wired, `m = 1` passes and `m = 2` faults the GPU. The fault
//      was a kernarg ORDER bug, found by launching trivial probes: the kernel
//      takes (act, B, C, M, N, K) but every launcher and harness passed
//      (a, b, mm, nn, kk, o) -- C last. So the kernel's C slot got M (a tiny
//      int) and its K slot got the truncated C pointer (a huge int), driving a
//      wild OOB loop. Counting "6 args" never catches this; only order does.
//      Fixed in launch_tree_pie_gemm and in both standalone launches.
//   5. With order fixed the fault became exact zeros, which was a FIXTURE bug:
//      the m>1 weights were x0.25 randoms and E2M2's smallest nonzero magnitude
//      is 0.25, so every payload nibble packed to +0.0 and the GEMM correctly
//      returned 0. Host-side proof: pack_tree_pie_32 of the fixture's first
//      block is [0,0,0,0,signs]. O(1) fixtures (same as the m=1 branch) fix it.
#[test]
fn tree_pie_prefill_matches_cpu_oracle_across_tile_boundaries() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };

    const N: usize = 64;
    // 1 is the GEMV path and must keep working; 8 exactly fills one tile; 13
    // straddles two; 16 fills two exactly.
    for (m, k) in [(1usize, 128usize), (2, 128), (8, 128), (13, 128), (16, 128), (9, 256)] {
        let worst = prefill_case(&dev, m, N, k)?;
        println!("prefill m={m:>2} k={k}: worst relative error {worst:.3e}");
        // Bar 0.15, not 0.05: the kernel is bit-exact vs the dequant oracle
        // (see tree_pie_gemm_matches_dequant_oracle_exactly), so this bound
        // measures E2M2 quant noise only -- plan App. A4 pins weight RMSE at
        // 0.0517, and worst-over-m*n dots of cancellating sums roughly doubles
        // it (m=13: 0.106 observed). A layout swap still reads O(1), an order
        // above this bar.
        assert!(worst < 0.15, "prefill m={m} k={k}: worst relative error {worst:.3e}");
    }
    Ok(())
}

/// Prefill GEMM == host dequant oracle, bit-exact.
///
/// E2M2-grid-exact weights make dequantization lossless, so any residual error
/// is kernel math alone (decode, fdot2 accumulation, cross-lane reduction,
/// tiling), not format noise. This is the tight gate; the raw-oracle test
/// above carries the format-noise bound. Both rows, ragged tile included.
#[test]
fn tree_pie_gemm_matches_dequant_oracle_exactly() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    // E2M2-exact magnitudes: subnormal row {0.5, 1.0, 1.5} and normal rows.
    let grid_vals = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let (m, n, k) = (13usize, 64usize, 128usize);
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = grid_vals[(s >> 61) as usize % grid_vals.len()];
        if (s >> 37) & 1 == 1 { -v } else { v }
    };
    let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let bits: Vec<u8> = x
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes().to_vec())
        .collect();
    let f16ty = DType { arith: ArithType::F16, storage: Storage::Native };
    let act_t = MemoryOps::from_cpu_bytes(&dev, &bits, &Shape::new(vec![m, k]), f16ty)
        .map_err(|e| format!("act h2d: {e}"))?;
    let packed = pack_columns(&w, n, k);
    let b_bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &b_bytes,
        &Shape::new(vec![n * (k / 32) * TREE_PIE_WORDS_PER_32]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![m, n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    dev.launch_tree_pie_gemm(raw(&act_t), raw(&b_t), raw(&out_t), m, n, k)
        .map_err(|e| format!("direct gemm: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got.len(), m * n);
    // Oracle against host-DEQUANTIZED weights (unpack, not raw): pure kernel math.
    // A rides as f16 on device, so match its rounding.
    let mut worst = 0.0f64;
    for r in 0..m {
        for col in 0..n {
            let mut o = 0.0f64;
            let mut sc = 0.0f64;
            for g in 0..k / 32 {
                let base = (col * (k / 32) + g) * TREE_PIE_WORDS_PER_32;
                let wb: [i32; 5] = packed[base..base + 5].try_into().unwrap();
                let dec = grim_quant::tree_pie::unpack_tree_pie_32(&wb);
                for j in 0..32 {
                    let a = half::f16::from_f32(x[r * k + g * 32 + j]).to_f32() as f64;
                    let t = a * dec[j] as f64;
                    o += t;
                    sc += t.abs();
                }
            }
            worst = worst.max(((got[r * n + col] as f64 - o) / sc.max(f64::MIN_POSITIVE)).abs());
        }
    }
    println!("prefill-vs-dequant worst relative error {worst:.3e} (m={m} ragged tile)");
    assert!(worst < 1e-3, "kernel math wrong: worst {worst:.3e}");
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

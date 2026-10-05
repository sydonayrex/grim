//! GPU backward gradient numerics for quantized matmuls (A.5 coverage gap).
//!
//! Verifies the ROCm fused backward kernel `dX = dY @ B^T` for Q8_0 weights
//! against an FP32 CPU reference.  This test was originally in
//! `grim-quant/tests/quant_backward_audit.rs` but was gated off with
//! `#[cfg(any())]` after `grim-quant` dropped the `rocm` feature to break a
//! dependency cycle.  Moved here so it can actually compile and run.
//!
//! Run with:
//!   GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test quant_backward_gpu -- --ignored --nocapture

use grim_quant::quant_q80;
use grim_tensor::{
    CoreTensorOps, MemoryOps, QuantOps, QuantizedMatmulBackwardResiduals, Shape,
    dtype::{ArithType, DType, KQuantScheme, Storage},
};

/// Maximum allowed RMS relative error for Q8_0 (8-bit).
const MAX_RMS_REL_ERROR_Q8: f32 = 0.05;

/// RMS relative error: sqrt(mean((orig-recon)^2 / orig^2)).
fn rms_rel_err(orig: &[f32], recon: &[f32]) -> f32 {
    assert_eq!(orig.len(), recon.len());
    let sum_sq: f32 = orig
        .iter()
        .zip(recon.iter())
        .map(|(o, r)| {
            let denom = o.abs().max(1e-3);
            ((o - r) / denom).powi(2)
        })
        .sum();
    (sum_sq / orig.len() as f32).sqrt()
}

/// Compute matrix gradient `dX[M, K] = dY[M, N] @ B[K, N]^T` on CPU.
fn compute_dx(dy: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for l in 0..n {
                sum += dy[i * n + l] * b[j * n + l];
            }
            dx[i * k + j] = sum;
        }
    }
    dx
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn quant_backward_rocm_q8_0_gemm_dx_numerics() {
    let rocm_devices = match grim_backend_rocm::RocmDevice::probe() {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let dev = grim_backend_rocm::RocmDevice::try_new(rocm_devices[0].ordinal())
        .expect("RocmDevice::try_new should succeed for probed device");

    let (m, k, n) = (8, 32, 32);
    let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

    let mut b_trans = vec![0.0f32; k * n];
    for j in 0..k {
        for l in 0..n {
            b_trans[l * k + j] = b_orig[j * n + l];
        }
    }
    let b_packed = quant_q80(&b_trans).unwrap();
    let b_rocm_shape = Shape::from_slice(&[k * n]);

    // Dequantize b to FP32 for exact CPU reference gradient comparison
    let b_dequant = grim_quant::dequant_q80(&b_packed, k * n).unwrap();
    let mut b_dequant_untrans = vec![0.0f32; k * n];
    for l in 0..n {
        for j in 0..k {
            b_dequant_untrans[j * n + l] = b_dequant[l * k + j];
        }
    }

    // Reference gradient on CPU using dequantized weights
    let dx_ref = compute_dx(&dy_host, &b_dequant_untrans, m, n, k);

    let b_rocm = dev
        .from_cpu_bytes(
            &b_packed,
            &b_rocm_shape,
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q80),
            },
        )
        .unwrap();

    // Call ROCm fused backward kernel for dX
    let out_shape = Shape::from_slice(&[m, k]);
    let residuals = QuantizedMatmulBackwardResiduals::default();
    let (dx_rocm, _handle) = dev
        .quantized_matmul_backward_dx(
            dy_rocm.as_ref(),
            b_rocm.as_ref(),
            &[],
            8, // bpw for Q8_0
            m,
            n,
            k,
            &out_shape,
            Some(&residuals),
        )
        .expect("ROCm quantized_matmul_backward_dx must succeed on a real ROCm device");

    // Copy result back to CPU
    let dx_rocm_vec = dx_rocm
        .to_cpu_vec_f32()
        .expect("ROCm result must be readable");

    let rms = rms_rel_err(&dx_ref, &dx_rocm_vec);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "ROCm Q8_0 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}

// ============================================================================
// Q2_0 (GGUF tag 42)
// ============================================================================

/// Q2_0 packed geometry: 64 weights per 18-byte block, `[fp16 d][16 B of 2-bit
/// codes]`, codebook `{-1, 0, +1, +2} * d` i.e. `(q - 1) * d`.
const Q2_0_BLOCK: usize = 64;
const Q2_0_BYTES: usize = 18;

/// Relative tolerance for the Q2_0 backward comparisons.
///
/// Sized from the failure it must catch, not picked and then justified: a
/// single 2-bit code differing by one level shifts that weight by `d >= amax`,
/// so one bad code moves `dX` by roughly `d * |dY|` -- on the order of half
/// the signal magnitude here. `1e-5` sits ~4 orders of magnitude below that, so
/// the assertions cannot pass on a mis-decoded weight, while still clearing
/// the ~1e-7 f32 summation error of a short `N` dot by two orders.
const Q2_0_REL_TOL: f64 = 1e-5;

/// Probe a real ROCm device, or skip loudly.
///
/// An early `return` that prints nothing is how a GPU test turns green on a
/// machine with no GPU at all, so this says it skipped.
fn q2_0_rocm_device(test: &str) -> Option<grim_backend_rocm::RocmDevice> {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping {test}: set GRIM_RUN_GPU_TESTS=1");
        return None;
    }
    let probed = match grim_backend_rocm::RocmDevice::probe() {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("Skipping {test}: no ROCm device probed");
            return None;
        }
    };
    Some(
        grim_backend_rocm::RocmDevice::try_new(probed[0].ordinal())
            .expect("RocmDevice::try_new must succeed for a probed device"),
    )
}

/// Assert `got` matches `want` to `Q2_0_REL_TOL` relative to `scale`, the largest
/// magnitude in the comparison set.
///
/// Scaling by the set's own peak rather than per-element: a `dX` component can
/// be legitimately near zero (block 2 of the hand-built test is exactly zero),
/// and dividing by `|want|` there would demand infinite relative accuracy of an
/// absolute value that is already at f32 noise.
#[track_caller]
fn assert_q2_0_close(got: f64, want: f64, scale: f64, what: &str) {
    let err = (got - want).abs();
    let bound = Q2_0_REL_TOL * scale.max(1.0);
    assert!(
        err <= bound,
        "{what}: got {got:.9}, want {want:.9}, |err| {err:.3e} > tol {bound:.3e} \
         (rel tol {Q2_0_REL_TOL}, scale {scale:.6})"
    );
}

/// Run `quantized_matmul_backward_dx` for a Q2_0 weight and return `dX` on host.
///
/// `w_host` is the `[n, k]` weight in row-major, **k contiguous** -- the layout
/// the kernel's `(K/64)*18` row stride assumes. (The Q8_0 test above builds the
/// same layout but calls it `b_trans`, which reads as the transpose of what it
/// is; indexing it directly avoids that confusion.)
fn q2_0_backward_dx_on_device(
    dev: &grim_backend_rocm::RocmDevice,
    dy_host: &[f32],
    packed: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let dy_rocm = dev
        .from_cpu(dy_host, &Shape::from_slice(&[m, n]), DType::F32)
        .expect("dY upload must succeed");

    // Shape is the logical element count; `copy_from_host_raw_bytes` sizes the
    // allocation from the byte slice length, so the packed length is what must
    // be right here.
    let b_rocm = dev
        .from_cpu_bytes(
            packed,
            &Shape::from_slice(&[n * k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q2_0),
            },
        )
        .expect("packed Q2_0 weight upload must succeed");

    let (dx_rocm, _handle) = dev
        .quantized_matmul_backward_dx(
            dy_rocm.as_ref(),
            b_rocm.as_ref(),
            &[],
            2, // bits per weight; unused on the KQuant(Q2_0) arm
            m,
            n,
            k,
            &Shape::from_slice(&[m, k]),
            Some(&QuantizedMatmulBackwardResiduals::default()),
        )
        .expect("Q2_0 backward_dx must succeed on a real ROCm device");

    let out = dx_rocm
        .to_cpu_vec_f32()
        .expect("dX must be readable back to host");
    assert_eq!(out.len(), m * k, "dX must have exactly m*k elements");
    out
}

/// Parity against the crate's own host Q2_0 reader.
///
/// The oracle is `grim_quant::dequant_q2_0`, a host Rust reader with no shared
/// code with the HIP kernel, and the dot is accumulated in f64 so the reference
/// carries none of the f32 rounding the tolerance has to absorb.
#[test]
#[ignore = "requires real ROCm device; GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn quant_backward_rocm_q2_0_gemm_dx_matches_host_oracle() {
    let Some(dev) = q2_0_rocm_device("q2_0 backward oracle parity") else {
        return;
    };

    // k = 192 = three whole Q2_0 blocks, so the kernel's `k_idx / 64` and
    // `k_idx % 64` split and its `blk * 18` block stride are exercised *across*
    // block boundaries. A k <= 64 test would pass with the stride logic absent.
    // m = 3 and n = 5 are deliberately ragged against the 256-thread block, so
    // the `idx >= total` tail guard is covered too.
    let (m, n, k) = (3usize, 5usize, 192usize);
    assert_eq!(k % Q2_0_BLOCK, 0, "k must be whole Q2_0 blocks");

    let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.7).sin() + 0.25).collect();
    // The `+ 0.7` keeps block 0 clear of all-zeros: `quantize_q2_0_block`
    // rejects a block whose amax is 0 as having no representable scale, and
    // `sin(0.0) == 0.0` would otherwise make exactly that block.
    let w_host: Vec<f32> = (0..n * k)
        .map(|i| (i as f32 * 0.11).cos() * 3.0 + 0.7)
        .collect();

    let mut packed = vec![0u8; (n * k / Q2_0_BLOCK) * Q2_0_BYTES];
    grim_quant::quantize_q2_0_block(&w_host, &mut packed).expect("Q2_0 packing must succeed");
    assert_eq!(packed.len(), (n * k / Q2_0_BLOCK) * Q2_0_BYTES);

    let dx_gpu = q2_0_backward_dx_on_device(&dev, &dy_host, &packed, m, n, k);

    let w_deq = grim_quant::dequant_q2_0(&packed, n * k).expect("host Q2_0 dequant must succeed");
    assert_eq!(w_deq.len(), n * k);

    let dx_ref: Vec<f64> = (0..m * k)
        .map(|idx| {
            let (i, j) = (idx / k, idx % k);
            (0..n)
                .map(|l| dy_host[i * n + l] as f64 * w_deq[l * k + j] as f64)
                .sum()
        })
        .collect();

    // Non-vacuity: a degenerate reference (all zeros) would make every relative
    // check below trivially true, so require real signal before comparing.
    let scale = dx_ref.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(
        scale > 0.5,
        "reference dX must carry real signal for this test to mean anything, \
         got peak {scale:.6}"
    );

    for (idx, (got, want)) in dx_gpu.iter().zip(dx_ref.iter()).enumerate() {
        let (i, j) = (idx / k, idx % k);
        assert_q2_0_close(*got as f64, *want, scale, &format!("dX[{i},{j}]"));
    }
}

/// Pins the kernel's codebook and block stride from **hand-built bytes**, with
/// no packer and no crate reader in the loop.
///
/// The oracle test above cannot do this on its own: if `quantize_q2_0_block`
/// and the HIP kernel ever agreed on the wrong offset, it would still pass.
/// Q2_0 and GSQ-RCO share this exact 18-byte geometry and differ *only* in the
/// codebook -- `(q - 1)` versus `(q - 2)` -- so a kernel reading GSQ-RCO bytes
/// is byte-compatible, finite, plausible, and wrong. Here each block's codes are
/// written directly, one level per block, so both the level mapping and the
/// block stride are pinned simultaneously:
///
/// * block 0: all codes 3 -> weight `+2 * d`
/// * block 1: all codes 0 -> weight `-1 * d`
/// * block 2: all codes 1 -> weight ` 0 * d`
///
/// With `d = 1.0` (exactly `0x3C00` in fp16), every expected weight is an
/// integer, so the whole comparison is against values a reader cannot fudge.
#[test]
#[ignore = "requires real ROCm device; GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn quant_backward_rocm_q2_0_gemm_dx_reads_q_minus_one_codebook() {
    let Some(dev) = q2_0_rocm_device("q2_0 codebook pin") else {
        return;
    };

    let (m, n, k) = (2usize, 4usize, 192usize);
    assert_eq!(k % Q2_0_BLOCK, 0, "k must be whole Q2_0 blocks");

    // fp16 1.0 == 0x3C00, stored little-endian.
    const D_FP16_ONE: [u8; 2] = [0x00, 0x3C];
    // The weight is `[n, k]`, so the three-block pattern repeats once per row --
    // `n * 3` blocks total, not three.
    let blocks_per_row = k / Q2_0_BLOCK;
    let per_block_code: [u8; 3] = [3, 0, 1];
    let mut packed = Vec::with_capacity(n * blocks_per_row * Q2_0_BYTES);
    for _row in 0..n {
        for &code in &per_block_code {
            packed.extend_from_slice(&D_FP16_ONE);
            // 4 codes per byte, low bits first: every byte is `code` repeated.
            packed
                .extend(std::iter::repeat(code | (code << 2) | (code << 4) | (code << 6)).take(16));
        }
    }
    assert_eq!(blocks_per_row, per_block_code.len());
    assert_eq!(packed.len(), n * blocks_per_row * Q2_0_BYTES);

    // y = (q - 1) * d with d = 1, indexed by flat `[n, k]` weight position: the
    // block index is within the row, so it is `(idx % k) / Q2_0_BLOCK` and not
    // `idx / Q2_0_BLOCK` -- the pattern restarts on every row.
    let expected_w =
        |idx: usize| -> f64 { f64::from(per_block_code[(idx % k) / Q2_0_BLOCK]) - 1.0 };

    // Cross-check the hand-built bytes against the crate reader too, so a
    // disagreement localises to the bytes rather than to either side alone.
    let deq =
        grim_quant::dequant_q2_0(&packed, n * k).expect("host Q2_0 dequant of hand-built bytes");
    for (j, w) in deq.iter().enumerate() {
        assert_eq!(
            *w as f64,
            expected_w(j),
            "hand-built block {} disagrees with the crate reader at weight {j}: \
             reader says {w}, hand math says {}",
            j / Q2_0_BLOCK,
            expected_w(j)
        );
    }

    let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.9).cos() + 0.5).collect();
    let dx_gpu = q2_0_backward_dx_on_device(&dev, &dy_host, &packed, m, n, k);

    // W is constant down each 64-column band, so dX[i, j] = w(j) * rowsum(dY[i]).
    let row_sums: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|l| dy_host[i * n + l] as f64).sum())
        .collect();
    let scale = row_sums.iter().fold(0.0f64, |acc, v| acc.max(v.abs())) * 2.0;
    assert!(scale > 0.0, "dY must have non-zero row sums");

    for i in 0..m {
        for j in 0..k {
            assert_q2_0_close(
                dx_gpu[i * k + j] as f64,
                expected_w(j) * row_sums[i],
                scale,
                &format!("dX[{i},{j}] (block {})", j / Q2_0_BLOCK),
            );
        }
    }
}

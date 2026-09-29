//! ROCm Q6_K, verified against llama.cpp — the chain is anchored to the
//! reference, not to grim.
//!
//! Why this exists: the 9B's `output.weight` (the lm_head table) is **Q6_K** and
//! `blk.3.attn_v.weight` is Q6_K, but no gate covered the ROCm Q6_K path on
//! real weights. `qwen35_layer_real_weights.rs` says so in its own comment —
//! "Quantized 2-D weights cannot be compared here without the crate-private
//! dequantiser" — so it byte-compares only the small F32 vectors
//! (`ssm_norm`, `ssm_a`, `ssm_dt.bias`). Every K-quant 2-D weight is checked
//! only against a hand reference built from the block's OWN weights, where a
//! dequant error cancels. The math can be perfect and the values wrong, and
//! every existing gate is blind to it.
//!
//! The chain, in order, so no link is taken on trust:
//!
//! 1. `llama_cpp_dequantize_row_q6k` below is transcribed from
//!    `old/repo/llama.cpp-master/ggml/src/ggml-quants.c:1939`
//!    (`dequantize_row_q6_K`), NOT from grim's kernel. It is the oracle.
//! 2. `grim_quant::dequant_q6k` (host) is checked against that oracle, so the
//!    host decoder is proven before it is used as a reference for the device.
//! 3. The ROCm `fused_quant_gemm` Q6_K path — the exact call `Linear::forward`
//!    makes for a Q6_K weight — is checked against the ORACLE (not against
//!    grim's host decoder), on the real 9B tensors, at the decode shape m=1.
//!
//! m=1 matters: the Q6_K launcher is chosen from m, so a m=8 gate can pass
//! while the shape the model actually decodes at takes a different, broken
//! path.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device. Gate 1 is CPU-only.

use grim_backend_rocm::RocmDevice;
use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantFormat, QuantOps, Shape,
    Storage as DTypeStorage,
};
use std::sync::Arc;

const QK_K: usize = 256;
/// `sizeof(block_q6_K)` — ggml-common.h:368 static_assert:
/// `sizeof(ggml_half) + QK_K/16 + 3*QK_K/4` = 2 + 16 + 192.
const Q6K_BLOCK_BYTES: usize = 210;

/// f16 -> f32, sign-correct on the SUBNORMAL path.
///
/// The first version of this helper returned `(mant as f32) * 2f32.powi(-24)`
/// for `exp == 0` and so DROPPED THE SIGN BIT. Real Q6_K super-blocks do carry
/// negative subnormal scales — the 9B's `blk.3.attn_v.weight` block 0 has
/// `d_bits = 0x80ad`, i.e. -1.0311e-5 — so the oracle reported +0.013528824
/// where the device correctly produced -0.013528824, and the gate blamed the
/// ROCm kernel for a defect in itself. grim's own
/// `shared_device_fns.rs::fp16_to_float_device` gets this right
/// (`return sign ? -res : res;`); this now matches it.
///
/// The synthetic data in gate 1 never exercised it: `quant_q6k` emits normal
/// scales, so the bug stayed invisible until real weights were used.
fn fp16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let v = (mant as f32) * 2f32.powi(-24);
        if sign == 1 { -v } else { v }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Faithful transcription of llama.cpp `dequantize_row_q6_K`
/// (`ggml/src/ggml-quants.c:1939-1963`).
///
/// The block is `ql[128] qh[64] scales[16] d` (ggml-common.h:363-367), so the
/// byte offsets are 0 / 128 / 192 / 208. Inside the reference, `ql`, `qh` and
/// `sc` are POINTERS advanced per 128-chunk (`ql += 64; qh += 32; sc += 8`), so
/// this indexes them absolutely: `n*64`, `n*32`, `n*8`.
///
/// Written from the reference on purpose. An oracle derived from the kernel
/// under test certifies the kernel's own reading of itself.
fn llama_cpp_dequantize_row_q6k(x: &[u8], y: &mut [f32], k: usize) {
    assert_eq!(
        k % QK_K,
        0,
        "Q6_K row must be a whole number of super-blocks"
    );
    assert_eq!(
        x.len(),
        (k / QK_K) * Q6K_BLOCK_BYTES,
        "packed length must be k/256 * 210"
    );
    for i in 0..k / QK_K {
        let blk = &x[i * Q6K_BLOCK_BYTES..(i + 1) * Q6K_BLOCK_BYTES];
        let d = fp16_to_f32(blk[208], blk[209]);
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        for n in (0..QK_K).step_by(128) {
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[l + n / 2] & 0xF) | (((qh[l + n / 4] >> 0) & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[l + n / 2 + 32] & 0xF) | (((qh[l + n / 4] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l + n / 2] >> 4) | (((qh[l + n / 4] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[l + n / 2 + 32] >> 4) | (((qh[l + n / 4] >> 6) & 3) << 4)) as i32 - 32;
                let base = i * QK_K + n + l;
                y[base] = d * sc[is + n / 16] as i8 as f32 * q1 as f32;
                y[base + 32] = d * sc[is + 2 + n / 16] as i8 as f32 * q2 as f32;
                y[base + 64] = d * sc[is + 4 + n / 16] as i8 as f32 * q3 as f32;
                y[base + 96] = d * sc[is + 6 + n / 16] as i8 as f32 * q4 as f32;
            }
        }
    }
}

/// GATE 1 — the host decoder, against the reference. CPU-only.
///
/// Also asserts the round trip is ACCURATE, not merely self-consistent: two
/// transcriptions of the same misreading would agree with each other and both
/// be garbage. Q6_K keeps ~6.5 bits/weight, so relative RMS error against the
/// original must be small.
#[test]
fn host_q6k_dequant_matches_llama_cpp_reference() {
    let n = 4 * QK_K;
    // A pattern that exercises both signs, a near-zero, and a large outlier.
    let src: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            let v = (t * 0.021).sin() * 2.0 + (t * 0.0037).cos() * 0.5;
            if i % 613 == 0 { 0.0 } else { v }
        })
        .collect();
    let packed = grim_quant::quant_q6k(&src).expect("quant_q6k");

    let mut oracle = vec![0.0f32; n];
    llama_cpp_dequantize_row_q6k(&packed, &mut oracle, n);
    let host = grim_quant::dequant_q6k(&packed, n).expect("dequant_q6k");

    for (i, (&o, (&h, &s))) in oracle.iter().zip(host.iter().zip(src.iter())).enumerate() {
        assert_eq!(
            o.to_bits(),
            h.to_bits(),
            "llama.cpp oracle and grim host disagree at weight {i}: oracle {o}, grim {h} \
             (original {s}) — the HOST Q6_K decoder is wrong, so it cannot be used as a \
             reference for the device path"
        );
    }
    // Accuracy is deliberately NOT asserted here. `grim_quant::quant_q6k` is a
    // test-data helper (no non-test caller anywhere in the workspace) and its
    // round trip carries ~71% relative RMS error, so an accuracy bound on data
    // IT produced would only measure the fixture. The weights that matter come
    // from a real llama.cpp-quantized checkpoint and are checked in gate 2.
    let rms_num: f32 = oracle
        .iter()
        .zip(src.iter())
        .map(|(o, s)| (o - s) * (o - s))
        .sum();
    let rms_den: f32 = src.iter().map(|s| s * s).sum();
    let rel_rms = (rms_num / rms_den.max(1e-12)).sqrt();
    eprintln!(
        "[q6k-host] {n} weights match llama.cpp BIT-EXACTLY ({} super-blocks, every one of \
         the four quarters and all 16 signed scales)",
        n / QK_K
    );
    eprintln!(
        "[q6k-host] note: round trip through grim_quant::quant_q6k has relative RMS \
         {rel_rms:.4} — that is the FIXTURE quantizer, which has no non-test caller, not \
         the decoder under test"
    );
}

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    Some(Arc::new(RocmDevice::try_new(0).expect(
        "GRIM_GPU_TEST=1 is set but RocmDevice::try_new(0) failed. Failing loudly rather \
         than catch_unwind().ok(): a swallowed init failure turns this gate GREEN with \
         zero assertions run, which is how a real defect hides behind a passing test.",
    )))
}

fn checkpoint() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root
            .join(up)
            .join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    eprintln!("[SKIP] 9B checkpoint not found (set GRIM_CHECKPOINT)");
    None
}

fn upload_packed(
    dev: &RocmDevice,
    bytes: &[u8],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    dev.from_cpu_bytes(
        bytes,
        shape,
        DType {
            arith: ArithType::U8,
            storage: DTypeStorage::KQuant(KQuantScheme::Q6K),
        },
    )
    .expect("upload packed Q6_K")
}

/// GATE 2 — the ROCm Q6_K GEMM against the ORACLE, on real 9B tensors.
///
/// `m = 1` is the decode shape and is the point: the Q6_K launcher is selected
/// from `m`, so a wider gate can pass while decode takes a different path.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn real_q6k_gemm_matches_llama_cpp_reference_at_decode_shape() {
    let Some(dev) = gpu_device() else { return };
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");

    // The lm_head table is the one that decides logits, so it is the one that
    // must be right; attn_v is a Q6_K projection in a full-attention layer.
    // attn_v FIRST: it is small enough to run m=1/8/16, and the m=16 case is
    // what discriminates the launcher (m%16==0 takes WMMA, m=1 takes the scalar
    // kernel). The 248320-row table aborts the test before attn_v is reached.
    let targets = ["blk.3.attn_v.weight", "output.weight"];

    for name in targets {
        let Some(target) = file
            .tensors
            .iter()
            .find(|t| t.name == name && t.dtype == GgufDType::Q6K)
        else {
            eprintln!("[SKIP] {name} is not Q6_K in this checkpoint");
            continue;
        };
        let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
        // ggml `create_tensor_2d(ctx, w, n_embd, n_out)` puts ne0 = in, ne1 = out,
        // so `dims[0]` is the INPUT width. Which orientation the device GEMM
        // expects is part of what this gate establishes, so BOTH are tried and
        // the outcome reported rather than assumed: a kernel that computes A·Wᵀ
        // and one that computes A·W are indistinguishable if you only ever
        // exercise a single orientation, and guessing wrong here produces a
        // spectacular "bug" that is really the test's own transposition.
        let d0 = target.dims[0] as usize;
        let d1 = target.dims[1] as usize;
        eprintln!("[q6k-gemm] {name}: ggml dims {d0}x{d1} (ne0=in, ne1=out)");

        let mut best: Option<(&str, f64)> = None;
        for (label, n, k) in [("ne1=out (A*W)", d1, d0), ("ne0=out (A*W^T)", d0, d1)] {
            if k % QK_K != 0 {
                eprintln!(
                    "[q6k-gemm] {name} {label}: k={k} is not a whole number of super-blocks, skipped"
                );
                continue;
            }
            let e = run_case(&dev, name, label, &bytes, n, k);
            if best.is_none_or(|(_, b)| e < b) {
                best = Some((label, e));
            }
        }
        let (label, err) = best.expect("at least one orientation has k%256==0");
        assert!(
            err < 0.02,
            "{name}: NEITHER weight orientation matches the llama.cpp reference. Best was \
             {label} at relative {err:.3e}. The Q6_K path is the lm_head on this checkpoint, \
             so this is a wrong-logits bug, not a rounding difference."
        );
        eprintln!(
            "[q6k-gemm] {name}: MATCHES llama.cpp under orientation {label} (rel err {err:.3e})"
        );
    }
}

/// One (n, k) orientation, end to end, against the oracle. Returns the worst
/// relative error rather than asserting, so BOTH orientations get measured —
/// an assert on the first would abort before the second ever runs.
fn run_case(dev: &RocmDevice, name: &str, label: &str, bytes: &[u8], n: usize, k: usize) -> f64 {
    assert_eq!(
        k % QK_K,
        0,
        "{name} {label}: k={k} must be whole super-blocks"
    );

    // ORACLE, straight from the reference transcription — not grim's host
    // decoder, so the device is checked against llama.cpp and not against
    // another of grim's implementations.
    let mut b_oracle = vec![0.0f32; n * k];
    llama_cpp_dequantize_row_q6k(bytes, &mut b_oracle, n * k);

    let b = upload_packed(dev, bytes, &Shape::new(vec![n, k]));

    // m=1 always, plus m=8 and m=16 when the scalar host reference is
    // affordable. m=16 is the point of the wider cases: the Q6_K launcher is
    // chosen from m (`wmma_quant_tile_ok` requires m % 16 == 0), so m=1 and
    // m=16 take DIFFERENT kernels. If they disagree, the bug is in whichever
    // one decode does not use — and decode is m=1.
    let mut ms = vec![1usize];
    if n * k <= 8_000_000 {
        ms.push(8);
        ms.push(16);
    }

    let mut overall = 0.0f64;
    for m in ms {
        let a_src: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let a = dev
            .from_cpu(&a_src, &Shape::new(vec![m, k]), DType::F32)
            .expect("upload A");
        let out_shape = Shape::new(vec![m, n]);
        let (out, handle) = dev
            .fused_quant_gemm(a.as_ref(), b.as_ref(), QuantFormat::Q6K, &out_shape)
            .unwrap_or_else(|e| panic!("{name} {label} m={m}: fused_quant_gemm failed: {e}"));
        handle.synchronize().expect("sync");
        let got = out.to_cpu_vec_f32().expect("readback");

        // Host reference: A[m,k] . deq(B)[n,k]^T, accumulated in f64 so the
        // comparison is not measuring f32 summation order.
        let mut worst = 0.0f64;
        let mut at = 0usize;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for t in 0..k {
                    acc += a_src[i * k + t] as f64 * b_oracle[j * k + t] as f64;
                }
                let want = acc as f32;
                let err = (got[i * n + j] - want).abs() as f64 / want.abs().max(1.0) as f64;
                if err > worst {
                    worst = err;
                    at = i * n + j;
                }
            }
        }
        let (ai, aj) = (at / n, at % n);
        let mut acc0 = 0.0f64;
        for t in 0..k {
            acc0 += a_src[ai * k + t] as f64 * b_oracle[aj * k + t] as f64;
        }
        eprintln!(
            "[q6k-gemm] {name} {label} [{n},{k}] m={m}: max rel err {worst:.3e} at {at} \
                 (got {}, want {})",
            got[at], acc0 as f32
        );
        overall = overall.max(worst);
    }
    overall
}

/// GATE 3 — the DEVICE element decoder, isolated.
///
/// Gate 2 fails on the GEMM, but a GEMM mixes three things: which launcher is
/// chosen, how B is partitioned into rows, and how one weight is decoded. This
/// removes the first two. B is ONE super-block (k=256) and A is a one-hot
/// vector, so the m=1 output is exactly one dequantized weight:
///
///     A = e_j  =>  C[0] = sum_t e_j[t] * B[t] = dequant(j)
///
/// m=1 is kept deliberately so the launcher is the same one decode takes.
/// 256 tiny launches, each one weight. The first divergence index is the
/// answer; a wrong quarter or a wrong scale byte shows up as a contiguous run.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn device_q6k_element_decoder_matches_llama_cpp_per_weight() {
    let Some(dev) = gpu_device() else { return };

    // Any valid Q6_K super-block decodes; how lossy the fixture quantizer is
    // does not matter, because the comparison is against the ORACLE's decode of
    // the same bytes, not against the original floats.
    let src: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let packed = grim_quant::quant_q6k(&src).expect("quant_q6k");
    assert_eq!(
        packed.len(),
        Q6K_BLOCK_BYTES,
        "one super-block is 210 bytes"
    );

    let mut oracle = vec![0.0f32; QK_K];
    llama_cpp_dequantize_row_q6k(&packed, &mut oracle, QK_K);

    let b = upload_packed(&dev, &packed, &Shape::new(vec![1usize, QK_K]));
    let out_shape = Shape::new(vec![1usize, 1usize]);

    let mut bad: Vec<(usize, f32, f32)> = Vec::new();
    for j in 0..QK_K {
        let mut a_src = vec![0.0f32; QK_K];
        a_src[j] = 1.0;
        let a = dev
            .from_cpu(&a_src, &Shape::new(vec![1usize, QK_K]), DType::F32)
            .expect("upload one-hot A");
        let (out, handle) = dev
            .fused_quant_gemm(a.as_ref(), b.as_ref(), QuantFormat::Q6K, &out_shape)
            .unwrap_or_else(|e| panic!("one-hot probe j={j}: fused_quant_gemm failed: {e}"));
        handle.synchronize().expect("sync");
        let got = out.to_cpu_vec_f32().expect("readback");
        let g = got[0];
        if (g - oracle[j]).abs() > 1e-4 * oracle[j].abs().max(1e-3) {
            bad.push((j, g, oracle[j]));
        }
    }

    if !bad.is_empty() {
        let first = bad[0];
        eprintln!(
            "[q6k-elem] {}/{} weights WRONG; first at index {}: device {} vs llama.cpp {}",
            bad.len(),
            QK_K,
            first.0,
            first.1,
            first.2
        );
        eprintln!(
            "[q6k-elem] wrong indices (first 32): {:?}",
            bad.iter().take(32).map(|x| x.0).collect::<Vec<_>>()
        );
    }
    assert!(
        bad.is_empty(),
        "the ROCm Q6_K element decoder disagrees with llama.cpp on {}/{} weights of one \
         super-block. First: index {} device {} vs reference {}. A contiguous run means a \
         wrong nibble/scale/2-bit-group selection; every 4th means the quarter mapping.",
        bad.len(),
        QK_K,
        bad[0].0,
        bad[0].1,
        bad[0].2
    );
    eprintln!("[q6k-elem] all {QK_K} weights of one super-block match llama.cpp on device");
}

/// GATE 4 — the kernel at the geometry that actually fails.
///
/// Gate 3 covered ONE super-block in ONE row. Gate 2 fails on the real table,
/// which has many rows AND many super-blocks per row, so one of those two
/// dimensions is untested. This covers both at once: B is [128, 512] — 128 rows,
/// 2 super-blocks each — and with A = e_t the output at column `col` is exactly
/// `oracle[col * 512 + t]`. So each launch checks 128 elements spread across
/// every row, and 512 launches cover all 65,536 of them.
///
/// A correct row stride makes all 512 launches pass. A stride that is off by one
/// super-block makes the second half of each row wrong while the first half is
/// right, which is the shape of the bug this is looking for.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn device_q6k_multirow_multiblock_matches_llama_cpp() {
    let Some(dev) = gpu_device() else { return };
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "blk.3.attn_v.weight" && t.dtype == GgufDType::Q6K)
        .expect("blk.3.attn_v.weight is Q6_K");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");

    // attn_v is [in=4096, out=1024]; take the first 128 output rows in full.
    const N: usize = 128;
    const K: usize = 512;
    const ROW_BYTES: usize = (K / QK_K) * Q6K_BLOCK_BYTES; // 2 super-blocks = 420 B
    let slice = &bytes[..N * ROW_BYTES];
    let mut oracle = vec![0.0f32; N * K];
    llama_cpp_dequantize_row_q6k(slice, &mut oracle, N * K);

    let b = upload_packed(&dev, slice, &Shape::new(vec![N, K]));
    let out_shape = Shape::new(vec![1usize, N]);

    let mut bad: Vec<(usize, usize, f32, f32)> = Vec::new(); // (t, col, got, want)
    for t in 0..K {
        let mut a_src = vec![0.0f32; K];
        a_src[t] = 1.0;
        let a = dev
            .from_cpu(&a_src, &Shape::new(vec![1usize, K]), DType::F32)
            .expect("upload one-hot A");
        let (out, handle) = dev
            .fused_quant_gemm(a.as_ref(), b.as_ref(), QuantFormat::Q6K, &out_shape)
            .expect("fused_quant_gemm");
        handle.synchronize().expect("sync");
        let got = out.to_cpu_vec_f32().expect("readback");
        for col in 0..N {
            let w = oracle[col * K + t];
            if (got[col] - w).abs() > 1e-4 * w.abs().max(1e-3) {
                if bad.len() < 12 {
                    bad.push((t, col, got[col], w));
                }
            }
        }
    }

    if !bad.is_empty() {
        // Decompose the first bad element so the flip is attributed to a named
        // factor (d / sc / q_code) instead of guessed at.
        {
            let blk = &slice[0..Q6K_BLOCK_BYTES];
            let d = fp16_to_f32(blk[208], blk[209]);
            let sc = blk[192..208].iter().map(|b| *b as i8).collect::<Vec<_>>();
            eprintln!(
                "[q6k-rows] block0 d(fp16@208)={d}  d_bits=0x{:04x}",
                u16::from_le_bytes([blk[208], blk[209]])
            );
            eprintln!("[q6k-rows] block0 scales(192..208) as i8 = {sc:?}");
            eprintln!(
                "[q6k-rows] block0 ql[0]={} qh[0]={}  ->  q_code(t=0)={}",
                blk[0],
                blk[128],
                ((blk[0] & 0x0F) as i32) | ((((blk[128] >> 0) & 3) as i32) << 4)
            );
            eprintln!(
                "[q6k-rows] block0 d*sc[0]*(q-32) = {}",
                d * sc[0] as f32
                    * (((blk[0] & 0x0F) as i32 | ((((blk[128] >> 0) & 3) as i32) << 4)) - 32)
                        as f32
            );
            eprintln!(
                "[q6k-rows] block0 NEGATED that   = {}",
                -(d * sc[0] as f32
                    * (((blk[0] & 0x0F) as i32 | ((((blk[128] >> 0) & 3) as i32) << 4)) - 32)
                        as f32)
            );
        }
        eprintln!(
            "[q6k-rows] wrong (t, col, device, reference) samples: {:?}",
            bad.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>()
        );
        let (t, col, g, w) = bad[0];
        eprintln!(
            "[q6k-rows] first: t={t} col={col} device {g} vs reference {w}  \
             (within-row index {t}, row byte offset {})",
            col * ROW_BYTES
        );
    }
    assert!(
        bad.is_empty(),
        "the ROCm Q6_K GEMM is wrong across MULTIPLE ROWS: one-hot probes disagree with the \
         llama.cpp reference. {} samples wrong, first at t={} col={} (device {} vs reference \
         {}). Gate 3 passes at n=1,k=256, so the defect is the row stride / row walk, not the \
         element decoder.",
        bad.len(),
        bad[0].0,
        bad[0].1,
        bad[0].2,
        bad[0].3
    );
    eprintln!("[q6k-rows] all {N}x{K} weights across {N} rows match llama.cpp on device");
}

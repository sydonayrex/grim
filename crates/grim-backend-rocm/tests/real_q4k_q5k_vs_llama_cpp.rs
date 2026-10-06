//! ROCm Q4_K and Q5_K, verified against llama.cpp — the same chain as
//! `real_q6k_vs_llama_cpp.rs`, for the two formats that carry most of the 9B.
//!
//! Why: `qwen35_layer_real_weights.rs` states in its own comment that
//! "Quantized 2-D weights cannot be compared here without the crate-private
//! dequantiser", so it byte-compares only `ssm_norm`, `ssm_a`, `ssm_dt.bias`.
//! That blocker is false — `grim-quant` is a public dependency. The consequence
//! is that every K-quant 2-D weight is verified only against a hand reference
//! built from the block's OWN weights, where a dequant error CANCELS: the
//! forward math can be perfect and the values wrong, and no existing gate sees
//! it. On this checkpoint that blind spot covers `attn_qkv` and `ssm_out`
//! (Q5_K — the recurrent branch's projections) and `attn_q/k/v/output`,
//! `attn_gate` and the FFNs (Q4_K).
//!
//! The oracles below are transcribed from the reference, NOT from grim:
//!   * `dequantize_row_q4_K` — `ggml/src/ggml-quants.c:1529`
//!   * `dequantize_row_q5_K` — `ggml/src/ggml-quants.c:1731`
//!   * `get_scale_min_k4`   — `ggml/src/ggml-quants.c:880`
//!   * block layouts        — `ggml/src/ggml-common.h:329` (Q4_K), `:347` (Q5_K)
//!
//! Chain, so no link is taken on trust:
//!   1. host `grim_quant::dequant_q4k` / `dequant_q5k` vs the oracle, bit-exact
//!   2. the device element decoder, one super-block, one-hot probe
//!   3. 128 rows x 512 weights — multi-row AND multi-super-block
//!   4. the real 9B tensors through `fused_quant_gemm` at m=1 (the decode shape)
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
const K_SCALE_SIZE: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KQ {
    Q4K,
    Q5K,
}

impl KQ {
    /// `sizeof(block_q4_K)` = 2 + 2 + K_SCALE_SIZE + QK_K/2 = 144.
    /// `sizeof(block_q5_K)` = 2 + 2 + K_SCALE_SIZE + QK_K/8 + QK_K/2 = 176.
    fn block_bytes(self) -> usize {
        match self {
            KQ::Q4K => 4 + K_SCALE_SIZE + QK_K / 2,
            KQ::Q5K => 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
        }
    }
    fn name(self) -> &'static str {
        match self {
            KQ::Q4K => "Q4_K",
            KQ::Q5K => "Q5_K",
        }
    }
    fn gguf_dtype(self) -> GgufDType {
        match self {
            KQ::Q4K => GgufDType::Q4K,
            KQ::Q5K => GgufDType::Q5K,
        }
    }
    fn scheme(self) -> KQuantScheme {
        match self {
            KQ::Q4K => KQuantScheme::Q4K,
            KQ::Q5K => KQuantScheme::Q5K,
        }
    }
    fn qformat(self) -> QuantFormat {
        match self {
            KQ::Q4K => QuantFormat::Q4K,
            KQ::Q5K => QuantFormat::Q5K,
        }
    }
    fn host_dequant(self, bytes: &[u8], n: usize) -> Vec<f32> {
        match self {
            KQ::Q4K => grim_quant::dequant_q4k(bytes, n).expect("dequant_q4k"),
            KQ::Q5K => grim_quant::dequant_q5k(bytes, n).expect("dequant_q5k"),
        }
    }
    fn quantize(self, src: &[f32]) -> Vec<u8> {
        match self {
            KQ::Q4K => grim_quant::quant_q4k(src).expect("quant_q4k"),
            KQ::Q5K => grim_quant::quant_q5k(src).expect("quant_q5k"),
        }
    }
    /// Real 9B tensors of this format, chosen to cover both layer types.
    fn real_targets(self) -> &'static [&'static str] {
        match self {
            // Q4_K: a full-attention projection and the FFN of a recurrent layer.
            KQ::Q4K => &["blk.3.attn_q.weight", "blk.0.attn_gate.weight"],
            // Q5_K: the fused recurrent QKV and the recurrent out projection.
            KQ::Q5K => &["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"],
        }
    }
}

/// f16 -> f32. The `exp == 0` subnormal path MUST apply the sign bit.
///
/// This is not theoretical: real K-quant super-blocks carry negative subnormal
/// scales (the 9B's `blk.3.attn_v.weight` block 0 has `d_bits = 0x80ad`, i.e.
/// -1.0311e-5). A first version of the Q6_K oracle in this repo returned the
/// bare magnitude here and reported the ROCm kernel as producing wrong logits
/// on the lm_head — the DEVICE was right. grim's own device helper
/// `shared_device_fns.rs::fp16_to_float_device` does `return sign ? -res : res;`
/// and is the thing to match. Synthetic fixtures never catch this because
/// `quant_q4k`/`quant_q5k` emit normal scales.
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

/// Transcription of `get_scale_min_k4` — `ggml/src/ggml-quants.c:880-887`.
///
/// The `j >= 4` branch is the one that is easy to get wrong: the scale `d` takes
/// its low nibble from `q[j+4]` and its high 2 bits from `q[j-4]`, while the min
/// `m` takes its high 2 bits from `q[j]` ITSELF. Reading `m`'s high bits from
/// `q[j-4]` as well is a real historical bug in this codebase and silently
/// corrupts every min above sub-block 3.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Transcriptions of `dequantize_row_q4_K` (`ggml-quants.c:1529-1550`) and
/// `dequantize_row_q5_K` (`:1731-1755`).
///
/// Both walk 64-weight groups: for group `g` they take `is = 2g` and `is+1`,
/// advance the code array by 32 bytes, and advance `is` by 2. Q5_K additionally
/// carries `u1`/`u2` masks starting at 1 and 2 and shifting left by 2 per group,
/// which selects the qh bit. Written per-weight here; the reference's pointer
/// arithmetic is inlined.
fn llama_cpp_dequantize_row(x: &[u8], y: &mut [f32], k: usize, f: KQ) {
    assert_eq!(k % QK_K, 0);
    assert_eq!(x.len(), (k / QK_K) * f.block_bytes());
    for i in 0..k / QK_K {
        let blk = &x[i * f.block_bytes()..(i + 1) * f.block_bytes()];
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let d = fp16_to_f32(blk[0], blk[1]);
        let dmin = fp16_to_f32(blk[2], blk[3]);
        match f {
            KQ::Q4K => {
                // block_q4_K: qs[QK_K/2] at byte 4 + K_SCALE_SIZE.
                //
                // The reference advances `q += 32` once per 64-weight group and
                // emits 32 LOW nibbles of q[0..31] then 32 HIGH nibbles of the
                // SAME q[0..31]. So the code byte is NOT w/2: 32 consecutive
                // weights share 16 bytes, and the low/high split is at w%64==32,
                // not at every second weight. (Getting that backwards was the
                // first version's bug, and it disagreed at weight 1.)
                let qs = &blk[4 + K_SCALE_SIZE..];
                for w in 0..QK_K {
                    let g = w / 64;
                    let is = 2 * g + (w % 64) / 32;
                    let within = w % 32;
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let byte = qs[g * 32 + within];
                    let q = if w % 64 < 32 { byte & 0x0F } else { byte >> 4 };
                    y[i * QK_K + w] = d * sc as f32 * q as f32 - dmin * m as f32;
                }
            }
            KQ::Q5K => {
                // block_q5_K: qh[QK_K/8] then qs[QK_K/2], both after scales.
                //
                // `qh` is NEVER ADVANCED in the reference — only `ql += 32` per
                // group. The group is selected by the `u1`/`u2` masks, which
                // shift left by 2 each iteration (1,2 then 4,8 then 16,32 then
                // 64,128), i.e. they pick bit 2g+sub of the SAME qh byte. So the
                // qh index is `w % 32` for all four groups; advancing it by 32
                // per group (the first version's bug) reads past the 32-byte
                // array on the second group.
                let qh = &blk[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
                let ql = &blk[4 + K_SCALE_SIZE + QK_K / 8..];
                for w in 0..QK_K {
                    let g = w / 64;
                    let sub = (w % 64) / 32;
                    let within = w % 32;
                    let is = 2 * g + sub;
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let byte = ql[g * 32 + within];
                    let low = if sub == 0 { byte & 0x0F } else { byte >> 4 };
                    let bit = 2 * g + sub;
                    let q = low as i32 + if (qh[within] >> bit) & 1 == 1 { 16 } else { 0 };
                    y[i * QK_K + w] = d * sc as f32 * q as f32 - dmin * m as f32;
                }
            }
        }
    }
}

/// GATE 1 — the host decoders against the reference. CPU-only.
#[test]
fn host_q4k_q5k_dequant_match_llama_cpp_reference() {
    let n = 4 * QK_K;
    let src: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            let v = (t * 0.021).sin() * 2.0 + (t * 0.0037).cos() * 0.5;
            if i % 613 == 0 { 0.0 } else { v }
        })
        .collect();
    for f in [KQ::Q4K, KQ::Q5K] {
        let packed = f.quantize(&src);
        assert_eq!(
            packed.len(),
            (n / QK_K) * f.block_bytes(),
            "{}: packed length must be n/256 * {}",
            f.name(),
            f.block_bytes()
        );
        let mut oracle = vec![0.0f32; n];
        llama_cpp_dequantize_row(&packed, &mut oracle, n, f);
        let host = f.host_dequant(&packed, n);
        for (i, (&o, &h)) in oracle.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                h.to_bits(),
                "{}: llama.cpp oracle and grim host disagree at weight {i}: oracle {o}, grim {h}",
                f.name()
            );
        }
        eprintln!(
            "[{}-host] {n} weights match llama.cpp BIT-EXACTLY ({} super-blocks, all 8 sub-block \
             scale/min pairs)",
            f.name(),
            n / QK_K
        );
    }
}

/// GATE 1b — the host decoders against REAL llama.cpp-quantized bytes.
///
/// This is the half of gate 1 that can actually convict the host decoder. The
/// synthetic half above compares grim against llama.cpp on blocks that
/// `grim_quant::quant_q4k` / `quant_q5k` produced, and if those fixture
/// quantizers lay a block out differently from llama.cpp then BOTH sides are
/// "right" about their own bytes and the disagreement says nothing about real
/// weights. (That is not hypothetical: `quant_q6k`'s round trip carries ~71%
/// relative RMS error.) Real checkpoint bytes have only one correct layout, so
/// a disagreement here is unambiguous.
#[test]
fn host_q4k_q5k_dequant_match_llama_cpp_on_real_weights() {
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    for f in [KQ::Q4K, KQ::Q5K] {
        for name in f.real_targets() {
            let Some(target) = file
                .tensors
                .iter()
                .find(|t| t.name == *name && t.dtype == f.gguf_dtype())
            else {
                eprintln!("[SKIP] {name} is not {} here", f.name());
                continue;
            };
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            // One super-block is enough and keeps the host decode instant.
            let one = f.block_bytes();
            let slice = &bytes[..one];
            let mut oracle = vec![0.0f32; QK_K];
            llama_cpp_dequantize_row(slice, &mut oracle, QK_K, f);
            let host = f.host_dequant(slice, QK_K);
            for (i, (&o, &h)) in oracle.iter().zip(host.iter()).enumerate() {
                assert_eq!(
                    o.to_bits(),
                    h.to_bits(),
                    "{}: on REAL {name} block 0, llama.cpp and grim's host decoder disagree at \
                     weight {i}: oracle {o}, grim {h}. These bytes were produced by \
                     llama.cpp's own quantizer, so there is exactly one correct layout and \
                     grim's {} host decoder is wrong.",
                    f.name(),
                    f.name()
                );
            }
            eprintln!(
                "[{}-host] {name} block 0 ({} B) matches llama.cpp BIT-EXACTLY on real weights",
                f.name(),
                one
            );
        }
    }
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
    f: KQ,
) -> Box<dyn grim_tensor::BackendStorage> {
    dev.from_cpu_bytes(
        bytes,
        shape,
        DType {
            arith: ArithType::U8,
            storage: DTypeStorage::KQuant(f.scheme()),
        },
    )
    .expect("upload packed")
}

/// GATES 2 and 3 — the device element decoder, isolated.
///
/// Gate 2 is one super-block, one row. Gate 3 is 128 rows x 2 super-blocks.
/// Both use a one-hot A so the m=1 output is exactly one dequantized weight:
/// `A = e_t` gives `C[0][col] = oracle[col * K + t]`, so each launch checks one
/// weight in EVERY row at once.
///
/// m=1 is kept deliberately — it is the decode shape and the launcher is chosen
/// from m, so this exercises the kernel production actually uses.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn device_q4k_q5k_element_decoders_match_llama_cpp() {
    let Some(dev) = gpu_device() else { return };

    for f in [KQ::Q4K, KQ::Q5K] {
        // ---- gate 2: one super-block, all 256 weights --------------------
        let src: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let one = f.quantize(&src);
        assert_eq!(one.len(), f.block_bytes());
        let mut oracle1 = vec![0.0f32; QK_K];
        llama_cpp_dequantize_row(&one, &mut oracle1, QK_K, f);

        // Tolerance for the one-hot probe below, DERIVED from the route the
        // device actually takes -- not chosen to make the gate pass.
        //
        // At m=1 on RDNA3/4 `fused_quant_gemm` routes to the dot4 GEMV, which
        // quantizes activations to Q8_1: an int8 code per weight plus an **fp16**
        // scale `d_a = amax/127`. For a one-hot activation the int8 grid is
        // exact (amax = 1.0 -> the single nonzero maps to code 127 exactly), so
        // the only error is fp16 representation of `d_a`: relative 2^-11.
        //
        // That error acts on `d * sc * q`, which is `w + dmin * m` -- the
        // magnitude BEFORE cancellation. So a tolerance relative to `|w|` is the
        // wrong denominator: where `d*sc*q` and `dmin*m` nearly cancel, `|w|` is
        // small while the error is not, and the ratio inflates without the
        // decode being any less correct. Q5_K weight 0 of the standard fixture
        // is exactly that case: |d*sc*q| = 2.89 against w = -0.0928, so a
        // 6.1e-5 input error surfaces as 1.9e-3 on the result -- which is why
        // the previous `1e-4 * |w|` bound was unmeetable for ANY correct
        // decoder on this route.
        //
        // Bounding against the block's peak decoded magnitude instead keeps the
        // gate's power: one wrong 5-bit code moves a weight by ~`d*sc*16`,
        // about half the block peak, so a real decode bug still misses the
        // bound by ~2 orders of magnitude (and on the exact scalar route the
        // observed error is ~1e-7, far inside it).
        const Q8_1_FP16_SCALE_EPS: f64 = 1.0 / 2048.0; // 2^-11
        let block_absmax = oracle1.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
        let elem_atol = (block_absmax as f64 * Q8_1_FP16_SCALE_EPS).max(1e-7);

        let b1 = upload_packed(&dev, &one, &Shape::new(vec![1usize, QK_K]), f);
        for t in 0..QK_K {
            let mut a = vec![0.0f32; QK_K];
            a[t] = 1.0;
            let a = dev
                .from_cpu(&a, &Shape::new(vec![1usize, QK_K]), DType::F32)
                .expect("A");
            let (out, h) = dev
                .fused_quant_gemm(
                    a.as_ref(),
                    b1.as_ref(),
                    f.qformat(),
                    &Shape::new(vec![1usize, 1usize]),
                )
                .expect("gemm");
            h.synchronize().expect("sync");
            let got = out.to_cpu_vec_f32().expect("readback")[0];
            let w = oracle1[t];
            assert!(
                (got - w).abs() as f64 <= elem_atol,
                "{}: one super-block, weight {t}: device {got} vs llama.cpp {w} \
                 (|err| {:.3e} > tol {:.3e}, block |max| {:.3e})",
                f.name(),
                (got - w).abs(),
                elem_atol,
                block_absmax,
            );
        }
        eprintln!(
            "[{}elem] all 256 weights of one super-block match on device \
             (atol {:.3e} = block|max| {:.3e} x 2^-11)",
            f.name(),
            elem_atol,
            block_absmax,
        );

        // ---- gate 3: 128 rows x 512 weights ------------------------------
        let Some(path) = checkpoint() else { return };
        let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
        let file = read_gguf(&mut reader).expect("read_gguf");
        let name = f.real_targets()[0];
        let Some(target) = file
            .tensors
            .iter()
            .find(|t| t.name == name && t.dtype == f.gguf_dtype())
        else {
            eprintln!("[SKIP] {name} is not {} here", f.name());
            continue;
        };
        let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
        const N: usize = 128;
        const K: usize = 512;
        let row_bytes = (K / QK_K) * f.block_bytes();
        let slice = &bytes[..N * row_bytes];
        let mut oracle = vec![0.0f32; N * K];
        llama_cpp_dequantize_row(slice, &mut oracle, N * K, f);
        let b = upload_packed(&dev, slice, &Shape::new(vec![N, K]), f);
        // Same derivation as gate 2, per row: the Q8_1 fp16 scale error acts on
        // each weight's pre-cancellation magnitude `d*sc*q = w + dmin*m`, so the
        // bound scales with the row's peak decoded weight, not with `|w|`. These
        // are real 9B projections where K-quant outputs reach ~1e6, so a
        // relative-to-|w| denominator is meaningless wherever a weight lands
        // near zero through cancellation.
        let row_atol: Vec<f64> = (0..N)
            .map(|col| {
                let peak = oracle[col * K..(col + 1) * K]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs())) as f64;
                (peak * Q8_1_FP16_SCALE_EPS).max(1e-7)
            })
            .collect();
        for t in 0..K {
            let mut a = vec![0.0f32; K];
            a[t] = 1.0;
            let a = dev
                .from_cpu(&a, &Shape::new(vec![1usize, K]), DType::F32)
                .expect("A");
            let (out, h) = dev
                .fused_quant_gemm(
                    a.as_ref(),
                    b.as_ref(),
                    f.qformat(),
                    &Shape::new(vec![1usize, N]),
                )
                .unwrap_or_else(|e| panic!("{}: gemm failed: {e}", f.name()));
            h.synchronize().expect("sync");
            let got = out.to_cpu_vec_f32().expect("readback");
            for col in 0..N {
                let w = oracle[col * K + t];
                assert!(
                    (got[col] - w).abs() as f64 <= row_atol[col],
                    "{}: {name} row {col} (byte offset {}) weight {t}: device {} vs llama.cpp {w} \
                     (|err| {:.3e} > tol {:.3e}) — gate 2 passes at one row, so this is the \
                     row stride, not the decoder",
                    f.name(),
                    col * row_bytes,
                    got[col],
                    (got[col] - w).abs(),
                    row_atol[col],
                );
            }
        }
        eprintln!(
            "[{}rows] all {N}x{K} weights across {N} rows match on device",
            f.name()
        );
    }
}

/// GATE 4 — the real 9B tensors through `fused_quant_gemm` at the decode shape.
///
/// ggml's `create_tensor_2d(ctx, w, n_embd, n_out)` puts ne0 = in, ne1 = out, so
/// `dims[0]` is the INPUT width. Which orientation the device GEMM consumes is
/// part of what this establishes, so BOTH are measured rather than assumed — a
/// kernel computing A·W and one computing A·Wᵀ are indistinguishable if only
/// one orientation is ever exercised.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn real_q4k_q5k_gemm_match_llama_cpp_at_decode_shape() {
    let Some(dev) = gpu_device() else { return };
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");

    for f in [KQ::Q4K, KQ::Q5K] {
        for name in f.real_targets() {
            let Some(target) = file
                .tensors
                .iter()
                .find(|t| t.name == *name && t.dtype == f.gguf_dtype())
            else {
                eprintln!("[SKIP] {name} is not {} here", f.name());
                continue;
            };
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            let d0 = target.dims[0] as usize;
            let d1 = target.dims[1] as usize;
            let mut best: Option<(&str, f64)> = None;
            for (label, n, k) in [("ne1=out (A*W)", d1, d0), ("ne0=out (A*W^T)", d0, d1)] {
                if k % QK_K != 0 {
                    continue;
                }
                // Only the smaller orientation is host-referenced: output.weight
                // sized shapes would be 1e9 f64 MACs.
                if n * k > 40_000_000 {
                    continue;
                }
                let mut oracle = vec![0.0f32; n * k];
                llama_cpp_dequantize_row(&bytes, &mut oracle, n * k, f);
                let b = upload_packed(&dev, &bytes, &Shape::new(vec![n, k]), f);
                for m in [1usize, 8, 16] {
                    let a_src: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
                    let a = dev
                        .from_cpu(&a_src, &Shape::new(vec![m, k]), DType::F32)
                        .expect("A");
                    let (out, h) = dev
                        .fused_quant_gemm(
                            a.as_ref(),
                            b.as_ref(),
                            f.qformat(),
                            &Shape::new(vec![m, n]),
                        )
                        .unwrap_or_else(|e| panic!("{name} {label} m={m}: {e}"));
                    h.synchronize().expect("sync");
                    let got = out.to_cpu_vec_f32().expect("readback");
                    // Error bound DERIVED from the route, replacing a
                    // max-relative-error metric that was ill-conditioned here.
                    //
                    // `fused_quant_gemm` at m=1 quantizes activations to Q8_1
                    // (int8 code + fp16 scale `d_a = amax/127`). Every activation
                    // is therefore off by up to `d_a/2 <= amax/254`, and those
                    // per-element errors add up over the K-long dot. So the
                    // correct bound on the output error is
                    //
                    //     |err| <= (amax_row / 254) * sum_t |w_t|
                    //
                    // i.e. proportional to the L1 magnitude of the weight row --
                    // NOT to |result|. Dividing by |result| (floored at 1.0, as
                    // this test did) is unbounded whenever a dot cancels: a
                    // result near zero through cancellation has a tiny
                    // denominator and a completely ordinary absolute error, so
                    // the ratio reports a "wrong-value bug" for arithmetic noise.
                    //
                    // Normalising by the bound itself makes the assertion a
                    // ratio against theory: ~1.0 means exactly the error the
                    // activation quantisation permits, and a genuine decode bug
                    // (one wrong 5-bit code moves a weight by ~half the row peak)
                    // overshoots it by orders of magnitude.
                    const Q8_1_STEP_NUM: f64 = 254.0; // 2 * 127
                    let mut worst = 0.0f64;
                    let mut worst_at = (0usize, 0usize);
                    for i in 0..m {
                        let amax_row = a_src[i * k..(i + 1) * k]
                            .iter()
                            .fold(0.0f32, |acc, v| acc.max(v.abs()))
                            as f64;
                        for j in 0..n {
                            let mut acc = 0.0f64;
                            let mut l1 = 0.0f64;
                            for t in 0..k {
                                let at = a_src[i * k + t] as f64;
                                let wt = oracle[j * k + t] as f64;
                                acc += at * wt;
                                l1 += wt.abs();
                            }
                            let w = acc as f32;
                            let bound = ((amax_row / Q8_1_STEP_NUM) * l1).max(1e-9);
                            let e = (got[i * n + j] - w).abs() as f64 / bound;
                            if e > worst {
                                worst = e;
                                worst_at = (i, j);
                            }
                        }
                    }
                    eprintln!(
                        "[{}-gemm] {name} {label} [{n},{k}] m={m}: worst |err| / \
                         Q8_1_bound = {worst:.3} at (m={},n={})",
                        f.name(),
                        worst_at.0,
                        worst_at.1,
                    );
                    let cur = best.map(|(_, _)| worst).unwrap_or(f64::INFINITY);
                    if worst < cur {
                        best = Some((label, worst));
                    }
                }
            }
            if let Some((label, err)) = best {
                assert!(
                    err <= 1.5,
                    "{name} ({}): NEITHER weight orientation matches the llama.cpp reference. \
                     Best {label} at {err:.3}x the Q8_1 activation-quantisation bound. These \
                     are the real {} projections, so past ~1.5x this is a wrong-value bug in \
                     the decode path, not the rounding the bound already allows.",
                    f.name(),
                    f.name()
                );
                eprintln!(
                    "[{}-gemm] {name}: MATCHES llama.cpp under {label} (rel err {err:.3e})",
                    f.name()
                );
            }
        }
    }
}

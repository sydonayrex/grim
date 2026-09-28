//! The 9B's real `token_embd` gathered on device, against the verified Q4_K
//! oracle.
//!
//! Why this is the right next gate: with Q4_K, Q5_K and Q6_K all verified
//! against llama.cpp on real weights, and per-layer parity already passing on
//! real 9B weights (recurrent 9.18e-6, attention 5.84e-6), the one
//! composition-stage tensor left unchecked is the EMBEDDING. Every layer test
//! supplies its own input tensor, so a wrong embedding row passes all of them
//! and then poisons the whole stack — which is exactly the observed symptom:
//! step 0, predicted from the prefill where positions are already correct, is
//! confidently wrong (`'\n'` at p=0.882 where Ollama says " Paris").
//!
//! `token_embd.weight` is `[248320, 4096]` Q4_K, 572,129,280 B. At 4096 = 16
//! super-blocks per row the row stride is 16 * 144 = 2304 B and every row starts
//! block-aligned, so the gather's address arithmetic is exact *if* it is right.
//! That "if" is the thing under test: the element decoder
//! (`dequant_q4k_element`) is already verified bit-exact elsewhere, so what is
//! left is the ROW INDEXING — `packed + row * row_bytes` — which no existing
//! gate covers on a real table.
//!
//! The oracle is the Q4_K transcription from `real_q4k_q5k_vs_llama_cpp.rs`,
//! itself transcribed from llama.cpp `dequantize_row_q4_K`
//! (`ggml/src/ggml-quants.c:1529`) and `get_scale_min_k4` (`:880`). It is not
//! derived from grim's kernel.
//!
//! The existing `q4k_embedding_gather.rs` gate does not reach this: it builds
//! its table with `grim_quant::quant_q4k` (a fixture quantizer with ~71% relative
//! RMS round-trip error for Q6_K) and gathers a 64-row synthetic table at
//! dim 5120, so the real 248320-row stride is never exercised and a fixture-layout
//! mistake would cancel against the host reference.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_tensor::{ArithType, CoreTensorOps, DType, MemoryOps, Shape, Storage as DTypeStorage};
use std::sync::Arc;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
const Q4K_BLOCK_BYTES: usize = 4 + K_SCALE_SIZE + QK_K / 2; // 144

fn checkpoint() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root.join(up).join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    eprintln!("[SKIP] 9B checkpoint not found (set GRIM_CHECKPOINT)");
    None
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
         zero assertions run, which is how a real defect hides behind a passing test."
    )))
}

/// f16 -> f32, sign applied on the subnormal path.
///
/// Real K-quant super-blocks carry negative subnormal scales — the 9B's
/// `blk.3.attn_v.weight` block 0 has `d_bits = 0x80ad` = -1.0311e-5 — so a
/// helper that returns the bare magnitude here disagrees with the device on real
/// bytes while passing every synthetic fixture, because `quant_q4k` emits normal
/// scales. Matches `shared_device_fns.rs::fp16_to_float_device`.
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

/// `get_scale_min_k4` — `ggml/src/ggml-quants.c:880-887`. The `j >= 4` branch
/// takes the min's high 2 bits from `q[j]` ITSELF, not from `q[j-4]`.
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

/// `dequantize_row_q4_K` — `ggml/src/ggml-quants.c:1529-1550`.
///
/// Per 64-weight group the reference emits 32 LOW nibbles of `q[0..31]` then 32
/// HIGH nibbles of the SAME 16 bytes, advancing `q += 32` once per group. So the
/// code byte is `qs[(w/64)*32 + (w%32)]` and the low/high split is at
/// `w % 64 == 32`.
fn llama_cpp_dequantize_row_q4k(x: &[u8], y: &mut [f32], k: usize) {
    assert_eq!(k % QK_K, 0);
    assert_eq!(x.len(), (k / QK_K) * Q4K_BLOCK_BYTES);
    for i in 0..k / QK_K {
        let blk = &x[i * Q4K_BLOCK_BYTES..(i + 1) * Q4K_BLOCK_BYTES];
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let qs = &blk[4 + K_SCALE_SIZE..];
        let d = fp16_to_f32(blk[0], blk[1]);
        let dmin = fp16_to_f32(blk[2], blk[3]);
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
}

/// Oracle for one vocabulary row: slice the row's bytes out of the packed table
/// and decode them. The row offset is computed HERE, independently of the
/// kernel, from the dim — which is the arithmetic under test.
fn oracle_row(table: &[u8], row: usize, dim: usize) -> Vec<f32> {
    let row_bytes = (dim / QK_K) * Q4K_BLOCK_BYTES;
    let start = row * row_bytes;
    let end = start + row_bytes;
    assert!(end <= table.len(), "row {row} past the end of the table");
    let mut out = vec![0.0f32; dim];
    llama_cpp_dequantize_row_q4k(&table[start..end], &mut out, dim);
    out
}

fn upload_table(dev: &RocmDevice, bytes: &[u8], rows: usize, dim: usize) -> Box<dyn grim_tensor::BackendStorage> {
    dev.from_cpu_bytes(
        bytes,
        &Shape::new(vec![rows, dim]),
        DType {
            arith: ArithType::U8,
            storage: DTypeStorage::KQuant(grim_tensor::KQuantScheme::Q4K),
        },
    )
    .expect("upload packed token_embd")
}

/// PREFLIGHT — validate THIS FILE'S ORACLE before any GPU result is read.
///
/// The oracle here is a second, independent transcription of llama.cpp's
/// `dequantize_row_q4_K`, living in a different file from the one already
/// proven. A transcription can be wrong in ways a green GPU run cannot reveal:
/// if the oracle and the kernel share a misreading, they agree and the gate
/// reports a false pass.
///
/// So this asserts the local oracle against `grim_quant::dequant_q4k`, which
/// `real_q4k_q5k_vs_llama_cpp.rs` has already established is BIT-EXACT against
/// llama.cpp on real checkpoint bytes (gates 1 and 1b). If the two disagree,
/// the oracle is wrong and no GPU number below means anything.
///
/// CPU-only and instant. Run it first.
#[test]
fn oracle_agrees_with_the_already_verified_host_decoder_on_real_rows() {
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight" && t.dtype == GgufDType::Q4K)
        .expect("token_embd.weight is Q4_K");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let (dim, rows) = (target.dims[0] as usize, target.dims[1] as usize);
    let row_bytes = (dim / QK_K) * Q4K_BLOCK_BYTES;
    assert_eq!(bytes.len(), rows * row_bytes, "table size must be rows * row_bytes");

    let probes: Vec<usize> = vec![0, 1, 561, 314, rows / 2, rows - 2, rows - 1];
    for row in &probes {
        let start = row * row_bytes;
        let slice = &bytes[start..start + row_bytes];
        let mut oracle = vec![0.0f32; dim];
        llama_cpp_dequantize_row_q4k(slice, &mut oracle, dim);
        let host = grim_quant::dequant_q4k(slice, dim).expect("dequant_q4k");
        for (j, (&o, &h)) in oracle.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                h.to_bits(),
                "PREFLIGHT FAILED: this file's oracle disagrees with grim's verified host \
                 Q4_K decoder on token_embd row {row} (byte offset {start}), dim index {j}: \
                 oracle {o}, host {h}. The oracle is wrong, so the GPU gates below would be \
                 measuring this file's misreading rather than the kernel."
            );
        }
    }
    eprintln!(
        "[embd-preflight] oracle == verified host decoder on {} real token_embd rows \
         ({} dims each); the GPU result below is meaningful",
        probes.len(),
        dim
    );
}

/// The real 248320-row table, gathered at the ids the failing run used, plus the
/// first and last row so an off-by-one at either end is caught.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn real_token_embd_rows_match_llama_cpp_oracle() {
    let Some(dev) = gpu_device() else { return };
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight" && t.dtype == GgufDType::Q4K)
        .expect("token_embd.weight is Q4_K");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");

    let (dim, rows) = (target.dims[0] as usize, target.dims[1] as usize);
    let row_bytes = (dim / QK_K) * Q4K_BLOCK_BYTES;
    eprintln!("[embd] token_embd ggml dims {dim}x{rows} (ne0=in=hidden, ne1=out=vocab)");
    assert_eq!(dim % QK_K, 0, "dim must be a whole number of super-blocks");
    assert_eq!(
        bytes.len(),
        rows * row_bytes,
        "table must be exactly rows * {row_bytes} B"
    );
    eprintln!("[embd] {rows} rows x {dim} dims, row stride {row_bytes} B, {} MB packed", bytes.len() / 1_048_576);

    // The 9B run's own prompt tokens, plus both ends of the table.
    let ids: Vec<u32> = vec![561, 6511, 314, 9338, 369, 0, 1, (rows - 1) as u32];

    let table = upload_table(&dev, &bytes, rows, dim);
    let out_shape = Shape::new(vec![ids.len(), dim]);
    let (got, handle) = dev
        .embedding_q4k(table.as_ref(), &ids, &out_shape, dim)
        .unwrap_or_else(|e| panic!("embedding_q4k on the real table failed: {e}"));
    handle.synchronize().expect("sync");
    let got = got.to_cpu_vec_f32().expect("readback");
    assert_eq!(got.len(), ids.len() * dim, "one row per id");

    for (n, &id) in ids.iter().enumerate() {
        let want = oracle_row(&bytes, id as usize, dim);
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..dim {
            let g = got[n * dim + j];
            let d = (g - want[j]).abs() / want[j].abs().max(1e-3);
            if d > worst {
                worst = d;
                at = j;
            }
        }
        eprintln!(
            "[embd] token {id} (row byte offset {}): max rel err {worst:.3e} at dim {at} \
             (device {} vs oracle {})",
            id as usize * row_bytes,
            got[n * dim + at],
            want[at]
        );
        assert!(
            worst < 1e-3,
            "token_embd row {id} (byte offset {}) differs from the llama.cpp Q4_K oracle by \
             relative {worst:.3e} at dim index {at}: device {} vs oracle {}. The element \
             decoder is verified elsewhere, so this is the gather's ROW addressing \
             (`packed + row * row_bytes`) or the dim/stride it is handed.",
            id as usize * row_bytes,
            got[n * dim + at],
            want[at]
        );
    }
    eprintln!("[embd] all {} probed rows of the real 248320-row table match the oracle", ids.len());
}

/// The same rows, but every id in a contiguous window plus one at each end, so a
/// stride error that happens to be zero for row 0 cannot pass. A row-index bug
/// that is a multiple of the id set would be invisible with 8 scattered ids.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn real_token_embd_contiguous_rows_expose_a_stride_error() {
    let Some(dev) = gpu_device() else { return };
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight" && t.dtype == GgufDType::Q4K)
        .expect("token_embd.weight is Q4_K");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let (dim, rows) = (target.dims[0] as usize, target.dims[1] as usize);
    let row_bytes = (dim / QK_K) * Q4K_BLOCK_BYTES;

    // A window in the middle of the vocabulary, where a bad stride lands on real
    // data rather than past the end of the table.
    let lo = rows / 2;
    let ids: Vec<u32> = (lo..lo + 24).map(|r| r as u32).collect();

    let table = upload_table(&dev, &bytes, rows, dim);
    let (got, handle) = dev
        .embedding_q4k(table.as_ref(), &ids, &Shape::new(vec![ids.len(), dim]), dim)
        .expect("embedding_q4k");
    handle.synchronize().expect("sync");
    let got = got.to_cpu_vec_f32().expect("readback");

    for (n, &id) in ids.iter().enumerate() {
        let want = oracle_row(&bytes, id as usize, dim);
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..dim {
            let d = (got[n * dim + j] - want[j]).abs() / want[j].abs().max(1e-3);
            if d > worst {
                worst = d;
                at = j;
            }
        }
        assert!(
            worst < 1e-3,
            "contiguous probe: token_embd row {id} (byte offset {}) differs from the oracle by \
             relative {worst:.3e} at dim index {at}. Rows {lo}..{} are contiguous, so a stride \
             error shows up here even if it vanished for the scattered ids.",
            id as usize * row_bytes,
            lo + ids.len()
        );
    }
    eprintln!(
        "[embd] contiguous rows {lo}..{} all match the oracle",
        lo + ids.len()
    );
}

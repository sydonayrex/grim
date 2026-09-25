//! Packed Q4_K embedding gather parity.
//!
//! The Qwen3.8-27B Q4_K checkpoint stores `token_embd.weight` as Q4_K,
//! `[248320, 5120]`, 715 MB packed. `Embedding::load` currently sees a
//! quantized tensor and calls `ws.get_f32(...)`, which dequantizes the WHOLE
//! table into a host `Vec<f32>` (5.09 GB) and uploads it as f32 (5.09 GB) — a
//! 7.1x blowup that is the dominant cause of VRAM oversubscription on this
//! model, plus a 5 GB transient host allocation per load.
//!
//! The fix is to keep the table packed and gather rows on device. The leaf
//! decode already exists as a random-access primitive,
//! `dequant_q4k_element(block_ptr, in_sb)`, so a gather only needs address
//! arithmetic: `dim = 5120` is exactly 20 Q4_K super-blocks, so row `r` starts
//! at byte `r * 20 * 144 = r * 2880` and is perfectly aligned.
//!
//! **The reference is deliberately the HOST implementation**
//! (`grim_quant::dequant_q4k`), not the device weight path. The device gather
//! will call the same `dequant_q4k_element` that the weight path uses, so
//! comparing against another device kernel could let a shared error cancel out
//! and report a false pass. Comparing to the host encoder/decoder round trip
//! is an independent check of the layout arithmetic, which is the part actually
//! at risk.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, MemoryOps};
use grim_tensor::{DType, Shape, Storage};
use std::sync::Arc;

const Q4K_BLOCK: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

/// Deterministic values spanning the range Q4_K must represent, including
/// near-zero (which the format handles via dmin) and both signs.
fn synth_table(vocab: usize, dim: usize) -> Vec<f32> {
    (0..vocab * dim)
        .map(|i| {
            let t = i as f32;
            let v = (t * 0.017).sin() * 2.5 + (t * 0.0031).cos() * 0.5;
            if i % 977 == 0 { 0.0 } else { v }
        })
        .collect()
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q4k_embedding_gather_matches_host_reference() {
    let Some(dev) = gpu_device() else { return };

    // dim 5120 mirrors the real Qwen geometry: exactly 20 Q4_K super-blocks
    // per row, which is the alignment the gather relies on.
    let vocab = 64usize;
    let dim = 5120usize;
    assert_eq!(dim % Q4K_BLOCK, 0, "row must be a whole number of super-blocks");
    assert_eq!(dim / Q4K_BLOCK, 20);

    let host = synth_table(vocab, dim);
    let packed = grim_quant::quant_q4k(&host).expect("quant_q4k");
    let expected_bytes = (vocab * dim / Q4K_BLOCK) * Q4K_BLOCK_BYTES;
    assert_eq!(packed.len(), expected_bytes, "packed size must match the Q4_K layout");

    // The tokens a decode step actually asks for: first row, a mid row, and the
    // last row, so an off-by-one in the row-to-block math is caught at both ends.
    let indices: Vec<u32> = vec![0, 1, (vocab / 2) as u32, (vocab - 1) as u32];

    let packed_dev = dev
        .from_cpu_bytes(
            &packed,
            &Shape::new(vec![packed.len()]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload packed Q4_K table");

    // The gather under test.
    let out_shape = Shape::new(vec![indices.len(), dim]);
    let (storage, _handle) = dev
        .embedding_q4k(packed_dev.as_ref(), &indices, &out_shape, dim)
        .expect("embedding_q4k gather");
    let got = storage.to_cpu_vec_f32().expect("read gather output");
    assert_eq!(got.len(), indices.len() * dim);

    // Host reference: dequantize the same rows from the same packed bytes.
    let bytes_per_row = (dim / Q4K_BLOCK) * Q4K_BLOCK_BYTES;
    let full = grim_quant::dequant_q4k(&packed, vocab * dim).expect("dequant_q4k");
    for (n, &tok) in indices.iter().enumerate() {
        let row = tok as usize;
        for j in 0..dim {
            let want = full[row * dim + j];
            let have = got[n * dim + j];
            // Q4_K reconstruction error; generous but far below O(1).
            let tol = 0.25f32.max(want.abs() * 0.15);
            assert!(
                (want - have).abs() <= tol,
                "token {tok} elem {j}: want {want}, got {have} \
                 (row starts at byte {} of {})",
                row * bytes_per_row,
                packed.len()
            );
        }
    }
    eprintln!(
        "[q4k-gather] {} tokens x {dim} dims matched host reference (row stride {bytes_per_row} B)",
        indices.len()
    );
}

/// A single-token gather is the decode hot path; it must not be a special case
/// that silently diverges from the multi-token path.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q4k_embedding_gather_single_token_matches_batch() {
    let Some(dev) = gpu_device() else { return };

    let vocab = 32usize;
    let dim = 5120usize;
    let host = synth_table(vocab, dim);
    let packed = grim_quant::quant_q4k(&host).expect("quant_q4k");
    let packed_dev = dev
        .from_cpu_bytes(
            &packed,
            &Shape::new(vec![packed.len()]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload packed");

    let full = grim_quant::dequant_q4k(&packed, vocab * dim).expect("dequant");

    for tok in [0usize, 7, vocab - 1] {
        let (storage, _h) = dev
            .embedding_q4k(
                packed_dev.as_ref(),
                &[tok as u32],
                &Shape::new(vec![1, dim]),
                dim,
            )
            .expect("single-token gather");
        let got = storage.to_cpu_vec_f32().expect("read");
        for j in 0..dim {
            let want = full[tok * dim + j];
            let tol = 0.25f32.max(want.abs() * 0.15);
            assert!(
                (want - got[j]).abs() <= tol,
                "single-token {tok} elem {j}: want {want}, got {}",
                got[j]
            );
        }
    }
}

/// Out-of-range token ids must be rejected, not read out of bounds. A gather
/// that trusts the index would fault or silently read another token's row.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q4k_embedding_gather_rejects_out_of_range_token() {
    let Some(dev) = gpu_device() else { return };

    let vocab = 8usize;
    let dim = 5120usize;
    let host = synth_table(vocab, dim);
    let packed = grim_quant::quant_q4k(&host).expect("quant_q4k");
    let packed_dev = dev
        .from_cpu_bytes(
            &packed,
            &Shape::new(vec![packed.len()]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload");

    let res = dev.embedding_q4k(
        packed_dev.as_ref(),
        &[vocab as u32], // one past the end
        &Shape::new(vec![1, dim]),
        dim,
    );
    assert!(
        res.is_err(),
        "token id {} is out of range for a {vocab}-row table and must be rejected",
        vocab
    );
}

//! Arbitration probe: which Q6_K head GEMM is right?
//!
//! The graph decode path and the eager decode path pick DIFFERENT Q6_K
//! kernels at m=1 on this card:
//!
//!   eager  `quantized_matmul`        -> `grim_wmma_fused_dequant_q6k`  (grid n/64)
//!   graph  `linear_decode_into`      -> `grim_fused_dequant_gemm_q6k`  (grid m*n/4/256)
//!
//! Both take a raw F32 activation (`wmma_quantized_gemm.rs` stamps
//! `const float* A`; `q6k_gemm.rs` likewise), so the disagreement is not a
//! type mismatch — one of the two decodes the weight wrong. Reading both
//! kernels says they implement the same canonical block_q6_K layout, which is
//! exactly when a third implementation is needed: this probe builds the
//! reference from `grim_quant::dequant_q6k` (the host decoder, itself gated
//! against llama.cpp) on the REAL `token_embd.weight` bytes of a REAL
//! checkpoint at the REAL shape, and measures each leg against it.
//!
//! Only the first `REF_COLS` output columns are referenced: that is enough to
//! catch an index/stride defect and keeps the host reference cheap
//! (`REF_COLS * k` MACs instead of 128256 * 2048).
//!
//! Run:
//! `HIP_VISIBLE_DEVICES=0 GRIM_GPU_TEST=1 cargo run --release \
//!  -p grim-backend-rocm --example q6k_head_probe`

use grim_backend_rocm::memory::storage::RocmStorage;
use grim_backend_rocm::RocmDevice;
use grim_tensor::provider::TensorProvider;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, MemoryOps, QuantOps, Shape, Storage,
};
use std::io::BufReader;

/// Columns of the head to reference on the host.
const REF_COLS: usize = 64;
const BLOCK_BYTES: usize = 210;
const BLOCK_ELEMS: usize = 256;

fn main() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let path = std::env::var("GRIM_Q6K_PROBE_MODEL")
        .unwrap_or_else(|_| "models/Llama-3.2-1B-Instruct-Q4_1.gguf".to_string());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let path = root.join(&path);
    let path = path.to_string_lossy().into_owned();

    let dev = RocmDevice::shared(0);

    // ── The real Q6_K head bytes.
    let f = std::fs::File::open(&path).expect("open checkpoint");
    let gguf = grim_format::gguf::read_gguf(BufReader::new(f)).expect("parse gguf");
    let provider = grim_format::tprov::GgufProvider::open(&path).expect("open provider");
    let t = gguf
        .tensors
        .iter()
        .filter(|t| t.dtype == grim_format::gguf::GgufDType::Q6K)
        .max_by_key(|t| t.size_bytes)
        .expect("a Q6_K tensor (the tied token_embd/output head)");
    // GGUF dims are [ne0, ne1] = [hidden, vocab].
    let k = t.dims[0] as usize;
    let n = t.dims[1] as usize;
    eprintln!(
        "[q6k-probe] tensor={} dims={:?} -> n={n} k={k} bytes={}",
        t.name, t.dims, t.size_bytes
    );
    assert_eq!(k % BLOCK_ELEMS, 0, "k must be a whole number of super-blocks");
    let raw = provider.get_packed(&t.name).expect("packed bytes");
    let blocks_per_row = k / BLOCK_ELEMS;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let need = REF_COLS * row_bytes;
    assert!(raw.bytes.len() >= need, "checkpoint shorter than REF_COLS rows");
    let packed = raw.bytes[..need].to_vec();

    // ── Host reference for the first REF_COLS rows.
    let mut w_f32: Vec<f32> = Vec::with_capacity(REF_COLS * k);
    for b in 0..REF_COLS * blocks_per_row {
        let blk = &packed[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        w_f32.extend(grim_quant::dequant_q6k(blk, BLOCK_ELEMS).expect("host dequant"));
    }
    assert!(
        w_f32.iter().all(|v| v.is_finite()),
        "host reference non-finite: the fixture is bad, not the kernel"
    );

    // ── A: a full-length activation, the shape the head actually gets.
    let a_host: Vec<f32> = (0..k).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();
    let want: Vec<f32> = (0..REF_COLS)
        .map(|col| {
            (0..k)
                .map(|kk| a_host[kk] * w_f32[col * k + kk])
                .sum::<f32>()
        })
        .collect();
    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    eprintln!("[q6k-probe] reference built: {REF_COLS} cols, scale {scale:.3e}");

    // ── Device side: the FULL weight (both legs need the whole tensor).
    let full = provider.get_packed(&t.name).expect("full packed bytes");
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q6K),
    };
    // The tensor's LOGICAL element count is n*k; the buffer holds the packed
    // bytes. `linear_decode_into` derives `w_n = elem_count / k` from the
    // shape, so passing the byte length here makes it read 105210 rows.
    let w = MemoryOps::from_cpu_bytes(&dev, &full.bytes, &Shape::new(vec![n * k]), q_dtype)
        .expect("upload packed weight");
    let w_r = w
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("weight is RocmStorage");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![1, k]), DType::F32)
        .expect("upload A");

    // ── Leg 1: the graph's leg (`linear_decode_into`).
    let out_shape = Shape::new(vec![1, n]);
    let out = CoreTensorOps::from_cpu(&dev, &vec![0.0f32; n], &out_shape, DType::F32)
        .expect("out");
    let out_r = out
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("out is RocmStorage");
    let act = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; (k / 32) * 36],
        &Shape::new(vec![(k / 32) * 36]),
        DType::U8,
    )
    .expect("act_q81 scratch");
    let act_r = act
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("act is RocmStorage");
    let h = dev
        .linear_decode_into(a_dev.as_ref(), w_r, out_r, act_r)
        .expect("graph leg");
    h.synchronize().expect("sync");
    report("graph  linear_decode_into", &out.to_cpu_vec_f32().expect("readback"), &want, scale);

    // ── Leg 2: the eager leg (`quantized_matmul`).
    let (out2_s, h2) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            w.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q6K,
            &out_shape,
        )
        .expect("eager leg");
    h2.synchronize().expect("sync");
    report(
        "eager  quantized_matmul",
        &out2_s.to_cpu_vec_f32().expect("readback"),
        &want,
        scale,
    );
}

fn report(tag: &str, got: &[f32], want: &[f32], scale: f32) {
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len().min(got.len()) {
        let d = (got[i] - want[i]).abs() / scale;
        let d = if d.is_finite() { d } else { f32::INFINITY };
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let verdict = if worst <= 1e-3 { "ok" } else { "FAIL" };
    eprintln!(
        "[q6k-probe] {tag}: worst rel {worst:.3e} at col {at} (got {:.4e} want {:.4e}) {verdict}",
        got.get(at).copied().unwrap_or(f32::NAN),
        want.get(at).copied().unwrap_or(f32::NAN)
    );
}

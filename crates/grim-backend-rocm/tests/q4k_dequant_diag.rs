//! Diagnostic: verify Q4_K dequant kernel output on ROCm.
//!
//! Run manually on a ROCm host:
//!   cargo test -p grim-backend-rocm --test q4k_dequant_diag -- \
//!       --nocapture --ignored
//!
//! Requires `GRIM_RUN_GPU_TESTS=1`.

use grim_backend_rocm::RocmDevice;
use grim_quant::dequant_q4k;
use grim_tensor::{
    BackendStorage, MemoryOps,
    dtype::{ArithType, DType, KQuantScheme, Storage},
    shape::Shape,
};

const QK4_K: usize = 256;
const BLOCK_BYTES: usize = 144;

/// Build a known Q4_K super-block exercising both nibbles and the cross-byte
/// scale-packing branch (mirrors golden_dequant::q4k_golden_...).
fn build_q4k_block(buf: &mut Vec<u8>, d: f32, min: f32, q_lo: u8, q_hi: u8) {
    let start = buf.len();
    buf.resize(start + BLOCK_BYTES, 0u8);
    let block = &mut buf[start..start + BLOCK_BYTES];

    let d_bits = half::f16::from_f32(d).to_bits().to_le_bytes();
    let min_bits = half::f16::from_f32(min).to_bits().to_le_bytes();
    block[0..2].copy_from_slice(&d_bits);
    block[2..4].copy_from_slice(&min_bits);

    let mut scales = [0u8; 12];
    scales[0] = 0x01; // sc0 = 1
    scales[4] = 0x00; // m0  = 0
    scales[8] = 0x35; // sc4 = 5 (lo nibble), m4 = 3 (hi nibble) — cross-byte branch
    block[4..16].copy_from_slice(&scales);

    // qs bytes: pair k=1 (is=2,3 low branch) and k=2 (is=4,5 cross-byte).
    // out index 128 = pair k=2 lo → qs byte qs[32*2 + 0] = qs[64] = block[80].
    block[80] = q_lo | (q_hi << 4);
    // out index 0   = pair k=0 lo → qs byte qs[0]        = block[16].
    block[16] = q_lo | (q_hi << 4);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_GPU_TEST=1 and -- --ignored"]
fn q4k_kernel_matches_cpu_dequant() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run on a real ROCm device");
        return;
    }
    let dev = RocmDevice::try_new(0)
        .expect("RocmDevice::try_new(0) should succeed on a system with ROCm");

    let mut packed = Vec::new();
    build_q4k_block(&mut packed, 1.0, 0.25, 10, 7);
    build_q4k_block(&mut packed, 2.0, 1.0, 8, 3);
    build_q4k_block(&mut packed, 0.5, 0.0, 6, 12);

    let n_blocks = packed.len() / BLOCK_BYTES;
    let n_weights = n_blocks * QK4_K;

    let expected = dequant_q4k(&packed, n_weights).expect("cpu dequant oracle");

    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let packed_storage = dev
        .from_cpu_bytes(&packed, &Shape::new(vec![packed.len()]), q4k_dtype)
        .expect("from_cpu_bytes");
    let packed_ref = packed_storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("downcast RocmStorage");

    let gpu_out = dev.dequantize_q4k(packed_ref).expect("dequantize_q4k");
    let got = gpu_out.to_cpu_vec_f32().expect("to_cpu_vec_f32");

    eprintln!("[q4k_diag] n_blocks={n_blocks} n_weights={n_weights}");
    eprintln!("[q4k_diag] expected[0]   = {}", expected[0]);
    eprintln!("[q4k_diag] got[0]       = {}", got[0]);
    eprintln!("[q4k_diag] expected[128] = {}", expected[128]);
    eprintln!("[q4k_diag] got[128]     = {}", got[128]);
    eprintln!(
        "[q4k_diag] expected[384] = {} (2nd superblock)",
        expected[384]
    );
    eprintln!(
        "[q4k_diag] got[384]       = {}",
        got.get(384).copied().unwrap_or(f32::NAN)
    );

    assert_eq!(got.len(), expected.len());
    let max_err = max_abs_diff(&got, &expected);
    eprintln!("[q4k_diag] max_err={max_err:e}");
    assert!(
        max_err == 0.0,
        "Q4_K kernel deviates from CPU oracle (max_err={max_err:e})"
    );
}

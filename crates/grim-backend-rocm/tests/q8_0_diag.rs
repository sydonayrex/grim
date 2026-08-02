//! Diagnostic: verify Q8_0 dequant kernel output on ROCm.
//!
//! Run with:
//!   cargo test -p grim-backend-rocm --test q8_0_diag -- --nocapture

use grim_tensor::{
    BackendDevice, BackendStorage,
    dtype::{ArithType, DType, KQuantScheme, Storage},
    shape::Shape,
};

const QK8_0: usize = 32;

fn build_q8_0_bytes() -> Vec<u8> {
    let mut raw = Vec::new();
    let f16_bits = half::f16::from_f32(2.0).to_bits();
    // Block 1: scale=2.0, codes=[1,2,...,32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for i in 1..=QK8_0 {
        raw.push(i as u8);
    }
    // Block 2: scale=2.0, codes=[-1,-2,...,-32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for i in 1..=QK8_0 {
        raw.push((-(i as i8)) as u8);
    }
    // Block 3: scale=2.0, codes=[0;32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for _ in 0..QK8_0 {
        raw.push(0);
    }
    // Block 4: scale=2.0, codes=[7;32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for _ in 0..QK8_0 {
        raw.push(7);
    }
    raw
}

fn cpu_dequant(bytes: &[u8], n_weights: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_weights);
    for blk in bytes.chunks_exact(QK8_0 + 2) {
        let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let qs = &blk[2..2 + QK8_0];
        for &q in qs {
            out.push(d * (q as i8 as f32));
        }
    }
    out
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn q8_0_kernel_matches_cpu_dequant() {
    let bytes = build_q8_0_bytes();
    let n_blocks = bytes.len() / (QK8_0 + 2);
    let n_weights = n_blocks * QK8_0;
    let expected = cpu_dequant(&bytes, n_weights);

    // GPU dequant via the kernel.
    let dev = grim_backend_rocm::RocmDevice::try_new(0)
        .expect("RocmDevice::try_new(0) should succeed on a system with ROCm");
    let q8_0_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let packed = dev
        .from_cpu_bytes(&bytes, &Shape::new(vec![bytes.len()]), q8_0_dtype)
        .unwrap();
    let packed_ref = packed
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let result = dev.dequantize_q8_0(packed_ref).unwrap();
    let got = result.to_cpu_vec_f32().unwrap();

    eprintln!("[q8_0_diag] expected[0..8]  = {:?}", &expected[0..8]);
    eprintln!("[q8_0_diag] got[0..8]      = {:?}", &got[0..8]);
    eprintln!("[q8_0_diag] expected[32..40]= {:?}", &expected[32..40]);
    eprintln!("[q8_0_diag] got[32..40]    = {:?}", &got[32..40]);

    assert_eq!(got.len(), expected.len());
    let mut max_err = 0.0f32;
    for i in 0..n_weights {
        let err = (got[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if err > 1e-3 {
            eprintln!(
                "[q8_0_diag] MISMATCH i={}: got={} expected={} err={}",
                i, got[i], expected[i], err
            );
        }
    }
    eprintln!("[q8_0_diag] max_err={}", max_err);
    assert!(
        max_err < 1e-3,
        "Q8_0 kernel deviates from CPU (max_err={})",
        max_err
    );
}

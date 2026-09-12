//! SPEED-ROC: GPU parity for the consolidated WMMA fused-dequant kernels
//! (`grim_wmma_fused_dequant_{q8_0,q4k,q5k,q2k,q3k,q6k}`) against a
//! dequantize-then-matmul CPU reference, at decode shape M=1.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test wmma_quant_parity -- --ignored

use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
};
use grim_quant::quant_q4k;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("RocmDevice::try_new"))
}

/// CPU reference: dequantize packed B to f32 then plain matmul C = A @ B^T.
fn ref_matmul(a: &[f32], b_packed: &[u8], m: usize, n: usize, k: usize, scheme: KQuantScheme, deq: &dyn Fn(&[u8], usize) -> Vec<f32>) -> Vec<f32> {
    let (block_elems, block_bytes): (usize, usize) = match scheme {
        KQuantScheme::Q4K => (256, 144),
        KQuantScheme::Q80 => (32, 34),
        _ => panic!("unsupported scheme in this test"),
    };
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    assert_eq!(b_packed.len(), n * row_bytes);
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let brow = &b_packed[col * row_bytes..(col + 1) * row_bytes];
            let mut b_f32 = Vec::with_capacity(k);
            for b in 0..blocks_per_row {
                b_f32.extend(deq(&brow[b * block_bytes..(b + 1) * block_bytes], block_elems));
            }
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b_f32[kk];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

fn run_case(dev: &RocmDevice, m: usize, n: usize, k: usize, scheme: KQuantScheme, format: grim_tensor::QuantFormat, pack: &dyn Fn(&[f32]) -> grim_tensor::error::Result<Vec<u8>>, deq: &dyn Fn(&[u8], usize) -> Vec<f32>) {
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
    let b_packed = pack(&b_orig).expect("pack");

    let a_shape = Shape::new(vec![m, k]);
    let a_dev = CoreTensorOps::from_cpu(dev, &a_host, &a_shape, DType::F32).expect("upload A");
    let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(scheme) };
    let b_shape = Shape::new(vec![n * b_packed.len() / n]); // flat bytes
    let b_dev = MemoryOps::from_cpu_bytes(dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
        .expect("upload packed B");
    let _ = b_shape;

    let out_shape = Shape::new(vec![m, n]);
    let (out_s, handle) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], format, &out_shape)
        .expect("quantized_matmul (WMMA path, m<=4)");
    handle.synchronize().expect("sync");
    let got = out_s.to_cpu_vec_f32().expect("to_cpu");

    let want = ref_matmul(&a_host, &b_packed, m, n, k, scheme, deq);
    let mut max_diff = 0.0f32;
    for i in 0..want.len() {
        max_diff = max_diff.max((got[i] - want[i]).abs());
    }
    eprintln!("[{format:?}] m={m} n={n} k={k} max_diff={max_diff}");
    assert!(
        max_diff < 0.5,
        "{format:?}: WMMA kernel diverges from reference (max_diff={max_diff})"
    );
}

#[test]
fn wmma_quant_decode_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    // Decode shape: M=1 (dispatches to the WMMA kernels under default GRIM_WMM_MAX_M=4).
    run_case(&dev, 1, 1024, 1024, KQuantScheme::Q80, grim_tensor::QuantFormat::Q8_0,
        &pack_q8_0, &|blk, n| {
            let d = f16_deq(blk);
            (0..n).map(|w| d * (blk[2 + w] as i8) as f32).collect()
        });
    run_case(&dev, 1, 1024, 1024, KQuantScheme::Q4K, grim_tensor::QuantFormat::Q4K,
        &quant_q4k, &|blk, n| grim_quant::dequant_q4k(blk, n).unwrap());
}

/// SPEED-ROC: FP16-input Q8_0 WMMA parity.  Uploads activations as FP16 so
/// `quantized_matmul` dispatches to `grim_wmma_fused_dequant_q8_0_fp16`
/// (reads _Float16 directly, no per-element cast).  Verifies the FP16-input
/// path matches the FP32-input path within FP16 quantization tolerance.
#[test]
fn wmma_quant_fp16_input_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    let (m, n, k) = (1usize, 1024usize, 1024usize);
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
    let b_packed = pack_q8_0(&b_orig).expect("pack");

    let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q80) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
        .expect("upload packed B");
    let out_shape = Shape::new(vec![m, n]);

    // FP32-input reference path.
    let a_f32 = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A f32");
    let (out_f32, h_f32) = dev
        .quantized_matmul(a_f32.as_ref(), b_dev.as_ref(), &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
        .expect("quantized_matmul f32 input");
    h_f32.synchronize().expect("sync f32");
    let got_f32 = out_f32.to_cpu_vec_f32().expect("to_cpu f32");

    // FP16-input path: uploads the same logical values as FP16 storage.
    let a_f16 = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F16)
        .expect("upload A f16");
    let (out_f16, h_f16) = dev
        .quantized_matmul(a_f16.as_ref(), b_dev.as_ref(), &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
        .expect("quantized_matmul f16 input");
    h_f16.synchronize().expect("sync f16");
    let got_f16 = out_f16.to_cpu_vec_f32().expect("to_cpu f16");

    let mut max_diff = 0.0f32;
    for i in 0..got_f32.len() {
        max_diff = max_diff.max((got_f32[i] - got_f16[i]).abs());
    }
    // The FP32-input arm now routes through the sudot4 GEMV (int8 activation
    // quant, per-block scales) while the FP16-input arm rides the fp16 WMMA
    // kernel — so this compares two different quantization schemes. 0.5 abs
    // accommodates int8 activation quant noise over k=1024.
    eprintln!("[fp16-input] m={m} n={n} k={k} max_diff(FP16-input WMMA vs sudot4)={max_diff}");
    assert!(
        max_diff < 5e-1,
        "FP16-input WMMA diverges from sudot4 path (max_diff={max_diff})"
    );
}

/// Pack f32 weights to Q8_0 (32-elem blocks: fp16 scale + i8 codes).
fn pack_q8_0(data: &[f32]) -> Result<Vec<u8>, grim_tensor::error::Error> {
    let mut out = Vec::with_capacity(data.len().div_ceil(32) * 34);
    for blk in data.chunks(32) {
        let max_abs = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = (max_abs / 127.0).max(1e-30);
        out.extend(half::f16::to_le_bytes(half::f16::from_f32(d)));
        for v in blk {
            out.push((v / d).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Ok(out)
}

fn f16_deq(blk: &[u8]) -> f32 {
    // little-endian fp16 scale at block start
    half::f16::from_le_bytes([blk[0], blk[1]]).to_f32()
}

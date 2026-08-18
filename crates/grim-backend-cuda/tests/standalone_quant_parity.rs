//! GPU-vs-CPU validation for the device-side CUDA quantize kernels.
//!
//! Validates that `CudaDevice::quantize_on_device` produces packed bytes that
//! are bit-identical (Q8_0) or numerically equivalent (FP8) to the CPU
//! reference in `grim_quant::quant_*`. Also validates the fused quantize+GEMM
//! kernels against a manual quantize-then-matmul reference.
//!
//! Run with:
//!   cargo test -p grim-backend-cuda --test standalone_quant_parity -- --nocapture

use grim_backend_cuda::{CudaDevice, CudaStorage};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, QuantFormat, Storage};
use grim_tensor::{BackendDevice, Shape};

/// Skip the test gracefully if no CUDA device is available.
fn device_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

#[test]
fn test_quantize_q8_0_matches_cpu() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    // 64 weights = 2 Q8_0 blocks.
    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 32.0) / 8.0).sin() * 3.0)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    // Device quantize.
    let q = dev.quantize(x.as_ref(), QuantFormat::Q8_0).unwrap();
    assert_eq!(q.dtype().storage, Storage::KQuant(KQuantScheme::Q80));

    let q_cuda = q.as_any().downcast_ref::<CudaStorage>().unwrap();
    let device_bytes = q_cuda.copy_to_host_raw_bytes().unwrap();

    // CPU reference.
    let cpu_bytes = grim_quant::quant_q80(&input).unwrap();

    assert_eq!(
        device_bytes.len(),
        cpu_bytes.len(),
        "byte length mismatch: device={} cpu={}",
        device_bytes.len(),
        cpu_bytes.len()
    );
    assert_eq!(
        device_bytes, cpu_bytes,
        "Q8_0 device quant bytes must be bit-identical to CPU reference"
    );
}

#[test]
fn test_quantize_fp8_matches_cpu() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 32.0) / 10.0).cos() * 2.0)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    // Device quantize.
    let q = dev.quantize(x.as_ref(), QuantFormat::Fp8).unwrap();
    assert!(matches!(
        q.dtype().storage,
        Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8)
    ));

    let q_cuda = q.as_any().downcast_ref::<CudaStorage>().unwrap();
    let device_bytes = q_cuda.copy_to_host_raw_bytes().unwrap();

    // CPU reference.
    let cpu_bytes = grim_quant::quant_fp8(&input).unwrap();

    assert_eq!(device_bytes.len(), cpu_bytes.len());

    // FP8 codes must be bit-identical (both use the same f32_to_fp8_e4m3 logic).
    assert_eq!(
        device_bytes, cpu_bytes,
        "FP8 device quant bytes must be bit-identical to CPU reference"
    );
}

#[test]
fn test_quantize_q8_0_roundtrip() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let n = 128;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.1;
            if i % 2 == 0 { x } else { -x }
        })
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let q = dev.quantize(x.as_ref(), QuantFormat::Q8_0).unwrap();
    let q_cuda = q.as_any().downcast_ref::<CudaStorage>().unwrap();
    let device_bytes = q_cuda.copy_to_host_raw_bytes().unwrap();

    // Dequantize via CPU reference and compare.
    let deq = grim_quant::dequant_q80(&device_bytes, n).unwrap();

    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Q8_0 max quantization error is scale/2 ≈ max_abs / (2 * 127).
    let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tolerance = (max_abs / 127.0).max(1e-6);
    assert!(
        max_err <= tolerance,
        "Q8_0 roundtrip max error {max_err} exceeds tolerance {tolerance}"
    );
}

#[test]
fn test_fused_quant_gemm_q8_0() {
    // Wait 3 seconds between Q8_0 CUDA tests to avoid GPU resource
    // contention false negatives (cuBLAS context thrashing under concurrent loads).
    std::thread::sleep(std::time::Duration::from_secs(3));
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let m = 4;
    let k = 32; // one Q8_0 block
    let n = 8;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1 - 2.0).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.05).cos()).collect();

    // Pack B as real Q8_0 bytes. The fused kernel expects B packed column-major:
    // for each output column col (0..N-1), a 34-byte block (f16 scale + 32 i8 codes)
    // containing the 32 elements B[0..K][col]. We transpose b_data (row-major [K,N])
    // into column-major order before quantizing so each 32-element chunk = one column.
    let mut b_col_major = Vec::with_capacity(k * n);
    for col in 0..n {
        for row in 0..k {
            b_col_major.push(b_data[row * n + col]);
        }
    }
    let b_packed = grim_quant::quant_q80(&b_col_major).expect("quant_q80");
    assert_eq!(b_packed.len(), n * (k / 32) * 34, "Q8_0 packed size mismatch");

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]); // logical shape is still [k, n]
    let out_shape = Shape::new(vec![m, n]);

    let a = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    // Upload B as KQuant(Q80) storage so the dispatch sees real packed Q8_0 bytes.
    let b = dev.from_cpu_bytes(
        &b_packed,
        &b_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        },
    ).unwrap();

    let (out, handle) = dev
        .fused_quant_gemm(a.as_ref(), b.as_ref(), QuantFormat::Q8_0, &out_shape)
        .unwrap();
    handle.synchronize().unwrap();

    let result = out.to_cpu_vec_f32().unwrap();

    // Manual reference: quantize A per-row-block, then matmul against dequantized B.
    // Dequantize the column-major packed B back and transpose to row-major for the
    // reference matmul (which uses b_data[row * n + col] indexing).
    let k_val = k;
    let b_col_deq = grim_quant::dequant_q80(&b_packed, k * n).expect("dequant_q80");
    let b_dequant: Vec<f32> = (0..k_val)
        .flat_map(|row| {
            let b = b_col_deq.clone();
            (0..n).map(move |col| b[col * k_val + row])
        })
        .collect();
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a_data[row * k..(row + 1) * k];
        // Inline Q8_0 quantization of the 32-element row.
        let amax = a_row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        let quantized: Vec<f32> = a_row
            .iter()
            .map(|v| {
                let q = (v / scale).round().clamp(-128.0, 127.0);
                q * scale
            })
            .collect();
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += quantized[i] * b_dequant[i * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "fused_quant_gemm_q8_0 max error {max_err} exceeds 1e-3\nresult: {result:?}\nexpected: {expected:?}"
    );
}

#[test]
fn test_fused_quant_gemm_fp8() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let m = 2;
    let k = 16;
    let n = 4;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.3 - 1.0).tan()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2).sin()).collect();

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]);
    let out_shape = Shape::new(vec![m, n]);

    let a = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

    let (out, handle) = dev
        .fused_quant_gemm(a.as_ref(), b.as_ref(), QuantFormat::Fp8, &out_shape)
        .unwrap();
    handle.synchronize().unwrap();

    let result = out.to_cpu_vec_f32().unwrap();

    // Manual reference: FP8 round-trip on A, then matmul.
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let fp8_code = grim_quant::f32_to_fp8_e4m3(a_data[row * k + i]);
                let a_deq = grim_quant::fp8_e4m3_to_f32(fp8_code);
                sum += a_deq * b_data[i * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.5,
        "fused_quant_gemm_fp8 max error {max_err} exceeds 0.5"
    );
}

#[test]
fn test_fused_quant_gemm_q4_k() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let m = 2;
    let k = 256; // 1 Q4_K superblock
    let n = 4;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();

    // Hand construct packed Q4_K weights for B (n_blocks * 144 = 4 * 144 = 576 bytes).
    let mut b_packed = vec![0u8; n * 144];
    for col in 0..n {
        let blk = &mut b_packed[col * 144..(col + 1) * 144];
        let d_bits = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        let min_bits = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        blk[0..2].copy_from_slice(&d_bits);
        blk[2..4].copy_from_slice(&min_bits);
        blk[4] = 2; // sc0 = 2
        blk[8] = 1; // m0 = 1
        blk[16] = 5 | (3 << 4); // lo nibble 5, hi nibble 3
    }

    let a_shape = Shape::new(vec![m, k]);
    let b_packed_shape = Shape::new(vec![n, k]);
    let out_shape = Shape::new(vec![m, n]);

    let a = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b = CudaStorage::copy_from_host_raw_bytes(
        &b_packed,
        &b_packed_shape,
        DType {
            arith: grim_tensor::dtype::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        },
        0,
    )
    .unwrap();

    let (out, handle) = dev
        .fused_quant_gemm(a.as_ref(), &b, QuantFormat::Q4K, &out_shape)
        .unwrap();
    handle.synchronize().unwrap();

    let result = out.to_cpu_vec_f32().unwrap();

    // Independent CPU oracle: dequant B per col, then matmul A @ B_deq.
    let mut b_deq = vec![0.0f32; k * n];
    for col in 0..n {
        let col_bytes = &b_packed[col * 144..(col + 1) * 144];
        let col_weights = grim_quant::dequant_q4k(col_bytes, 256).unwrap();
        for r in 0..k {
            b_deq[r * n + col] = col_weights[r];
        }
    }

    let mut expected = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_data[r * k + p] * b_deq[p * n + c];
            }
            expected[r * n + c] = sum;
        }
    }

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "fused_quant_gemm_q4_k max error {max_err} exceeds 1e-3"
    );
}

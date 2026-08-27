//! Quant workstream gates: fused dequant-GEMMs for CompressedTensorsW8A8
//! Int8/Fp8 and WNA16 vs golden CPU references (exact-dequant weights, so
//! parity is quantization-exact).
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendDevice;
use grim_tensor::{ArithType, DType, Device, Shape, Storage, Tensor};
use std::sync::Arc;

fn wrap(st: Box<dyn grim_tensor::BackendStorage>, shape: Shape, device: Device) -> Tensor {
    let dtype = st.dtype();
    Tensor::new(
        Arc::from(st),
        shape,
        dtype,
        grim_tensor::QuantProvenance::default(),
        device,
    )
}

fn gpu(_dev: &RocmDevice) -> bool {
    grim_backend_rocm::gpu_test_enabled()
}

fn run_quantized_matmul(
    dev: &RocmDevice,
    a: &[f32],
    blob: &[u8],
    blob_dtype: DType,
    (m, n, k): (usize, usize, usize),
) -> Vec<f32> {
    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![n, k]);
    let out_shape = Shape::new(vec![m, n]);
    let a_gpu = dev.from_cpu(a, &a_shape, DType::F32).unwrap();
    let blob_gpu = dev.from_cpu_bytes(blob, &b_shape, blob_dtype).unwrap();
    let (out, handle) = dev
        .quantized_matmul(
            a_gpu.as_ref(),
            blob_gpu.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K, // ignored; dispatch keys on storage
            &out_shape,
        )
        .expect("quantized_matmul dispatch");
    handle.synchronize().unwrap();
    wrap(out, out_shape, Device::Rocm(0)).to_vec_f32().unwrap()
}

fn cpu_matmul(a: &[f32], w: &[f32], (m, n, k): (usize, usize, usize)) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            c[r * n + j] = (0..k).map(|p| a[r * k + p] * w[j * k + p]).sum();
        }
    }
    c
}

fn check_parity(label: &str, got: &[f32], want: &[f32], tol: f32) {
    let md = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < tol,
        "{label}: diverged from golden reference (max diff {md:.5})\ngot={got:?}\nwant={want:?}"
    );
}

#[test]
fn w8a8_int8_gemm_matches_golden_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !gpu(&dev) {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    let dims = (2usize, 6usize, 8usize); // m, n, k
    let a: Vec<f32> = (0..dims.0 * dims.2)
        .map(|i| ((i % 9) as f32 * 0.3) - 1.2)
        .collect();

    // int8 codes [n, k] row-major, per-output-channel f32 scales; weights
    // exactly representable → golden reference is exact.
    let mut codes: Vec<i8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut w = vec![0.0f32; dims.1 * dims.2];
    let mut seed = 0x51D3u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as i32 % 121 - 60) as f32 / 10.0
    };
    for j in 0..dims.1 {
        let scale = 0.05 + (j as f32) * 0.01;
        scales.push(scale);
        for p in 0..dims.2 {
            let code = rand().clamp(-8.0, 7.0) as i8;
            codes.push(code);
            w[j * dims.2 + p] = code as f32 * scale;
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&((dims.1 * 4) as u64).to_le_bytes()); // scales_len (bytes)
    for c in &codes {
        blob.push(*c as u8);
    }
    for s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }

    let got = run_quantized_matmul(
        &dev,
        &a,
        &blob,
        DType {
            arith: ArithType::F32,
            storage: Storage::CompressedTensorsW8A8Int8,
        },
        dims,
    );
    let want = cpu_matmul(&a, &w, dims);
    check_parity("W8A8-Int8", &got, &want, 1e-4);
}

/// Host e4m3 decode mirroring grim-quant::fp8_e4m3_to_f32 (OCP E4M3).
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;
    match exp {
        0 => sign * (mant / 8.0) * 2.0f32.powi(-6), // subnormal: (m/8)·2^-6
        0x0F => {
            if mant != 0.0 {
                f32::NAN
            } else {
                sign * f32::INFINITY
            }
        }
        e => sign * (1.0 + mant / 8.0) * 2.0f32.powi(e - 7),
    }
}

#[test]
fn w8a8_fp8_gemm_matches_golden_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !gpu(&dev) {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    let dims = (2usize, 6usize, 8usize);
    let a: Vec<f32> = (0..dims.0 * dims.2)
        .map(|i| ((i % 7) as f32 * 0.4) - 1.0)
        .collect();

    // Finite E4M3 codes (exclude 0x7F/0xFF NaN and infinities) + per-tensor
    // f32 scale; golden weights exact by construction.
    let tensor_scale = 0.125f32;
    let mut codes: Vec<u8> = Vec::new();
    let mut w = vec![0.0f32; dims.1 * dims.2];
    let mut seed = 0xF00Du64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as u32) as u8
    };
    for j in 0..dims.1 {
        for p in 0..dims.2 {
            // Mask to a finite code: clear the mantissa-all-ones + exp-15 NaN
            // pattern by keeping exp < 15.
            let mut b = rand() & 0x7F;
            if (b >> 3) == 0x0F {
                b &= 0x77;
            }
            codes.push(b);
            w[j * dims.2 + p] = e4m3_to_f32(b) * tensor_scale;
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&4u64.to_le_bytes()); // scale_len (bytes)
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&tensor_scale.to_le_bytes());

    let got = run_quantized_matmul(
        &dev,
        &a,
        &blob,
        DType {
            arith: ArithType::F32,
            storage: Storage::CompressedTensorsW8A8Fp8,
        },
        dims,
    );
    let want = cpu_matmul(&a, &w, dims);
    check_parity("W8A8-Fp8", &got, &want, 1e-4);
}

fn pack_msb_nbit(codes: &[u32], n_bit: u8) -> Vec<u8> {
    let total_bits = codes.len() * n_bit as usize;
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    for (lane, &code) in codes.iter().enumerate() {
        for bit in 0..n_bit as usize {
            if code & (1 << (n_bit as usize - 1 - bit)) != 0 {
                let pos = lane * n_bit as usize + bit;
                bytes[pos / 8] |= 1 << (7 - (pos % 8));
            }
        }
    }
    bytes
}

fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | mant as u16
}

#[test]
fn wna16_fused_gemm_matches_golden_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !gpu(&dev) {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    let dims = (2usize, 4usize, 8usize); // m, n, k — 32 weights total, 1 block if 256-blocks... need blocks covering n*k=32 → ceil(32/256)=1
    let n_bit = 4u8;
    let a: Vec<f32> = (0..dims.0 * dims.2)
        .map(|i| ((i % 5) as f32 * 0.5) - 1.0)
        .collect();

    let total = dims.1 * dims.2; // 32
    let blocks = total.div_ceil(256); // 1
    let ts = 1.25f32;
    let block_scale_f = 0.75f32;
    let block_scale_h = f32_to_f16_bits(block_scale_f);

    let mut seed = 0xAAA5u64;
    let codes: Vec<u32> = (0..total)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as u32) % 16
        })
        .collect();
    let mut w = vec![0.0f32; total];
    for i in 0..total {
        w[i] = codes[i] as f32 * block_scale_f * ts;
    }

    // Codes occupy FIXED 256-weight block strides (ceil(256*n_bit/8) bytes
    // per block) — a partial final block is zero-padded to full stride.
    let code_bytes_per_block = (256 * n_bit as usize).div_ceil(8);
    let mut code_seg = vec![0u8; blocks * code_bytes_per_block];
    let packed = pack_msb_nbit(&codes, n_bit);
    code_seg[..packed.len()].copy_from_slice(&packed);

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&(n_bit as u32).to_le_bytes());
    blob.extend_from_slice(&(blocks as u32).to_le_bytes());
    blob.extend_from_slice(&code_seg);
    blob.extend_from_slice(&block_scale_h.to_le_bytes());
    blob.extend_from_slice(&ts.to_le_bytes());

    let got = run_quantized_matmul(
        &dev,
        &a,
        &blob,
        DType {
            arith: ArithType::F32,
            storage: Storage::WNA16,
        },
        dims,
    );
    let want = cpu_matmul(&a, &w, dims);
    check_parity("WNA16-fused", &got, &want, 1e-4);
}

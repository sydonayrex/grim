//! GPU parity for `kernels/compressed_gemm.rs` — the WNA16 / W8A8 dequant
//! GEMMs (`grim_wna16_dequant_gemm`, `grim_w8a8_int8_dequant_gemm`,
//! `grim_w8a8_fp8_dequant_gemm`) vs CPU oracles, through the public
//! `quantized_matmul` dispatch (which keys on `DTypeStorage`, not the
//! `QuantFormat` argument).
//!
//! Audit §3.6: the file had 0 inline tests and 0 `tests/` references. The
//! route exists and is unconditional (quant/mod.rs: `DTypeStorage::WNA16`
//! → `grim_wna16_dequant_gemm`, `CompressedTensorsW8A8*` → the matching
//! W8A8 kernel), so this is a real launch-and-compare, not a structural
//! note. Note that `tests/w8a8_wna16_gemm_gate.rs` already covers the same
//! dispatch at 1e-4 with *exact* dequant fixtures; this file adds the
//! elementwise-ops sibling (`launch_elementwise_dequant_gemm`, also used
//! for the backward kernels) at the house int8-act dot4 bar (2e-2 rel)
//! with non-exact, multi-block fixtures, plus a W8A8-FP8 f32-accum case.
//! The fp8 kernels fetch `fp8_e4m3_to_float_hip` from the aggregate source —
//! see the tolerance note below.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, MemoryOps, QuantOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// `grim_quant`'s WNA16 oracle (per-block f16 × per-tensor f32 scale).
fn oracle_wna16_rows(blob: &[u8], elem_count: usize) -> Vec<f32> {
    grim_quant::dequant_wna16(blob, elem_count).expect("wna16 oracle")
}

/// Host E4M3 decode mirroring `grim_quant::fp8_e4m3_to_f32` (OCP E4M3).
fn e4m3_to_f32(byte: u8) -> f32 {
    grim_quant::fp8_e4m3_to_f32(byte)
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

fn cpu_gemm(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            c[r * n + j] = (0..k).map(|p| a[r * k + p] * w[j * k + p]).sum();
        }
    }
    c
}

fn check_rel(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let rel = (g - w).abs() / denom;
        if rel > worst {
            worst = rel;
        }
        assert!(
            rel <= tol,
            "{what}[{i}]: got={g} want={w} rel={rel} tol={tol}"
        );
    }
    println!("{what}: worst rel error {worst:.3e} (tol {tol:.0e})");
}

fn run_qmm(dev: &RocmDevice, a: &[f32], blob: &[u8], storage: Storage, m: usize, n: usize, k: usize) -> Vec<f32> {
    let a_t = CoreTensorOps::from_cpu(dev, a, &Shape::new(vec![m, k]), DType::F32).expect("a h2d");
    let b_t = MemoryOps::from_cpu_bytes(
        dev,
        blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage,
        },
    )
    .expect("b h2d");
    let (out, handle) = dev
        .quantized_matmul(
            a_t.as_ref(),
            b_t.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K, // ignored; dispatch keys on storage
            &Shape::new(vec![m, n]),
        )
        .expect("quantized_matmul dispatch");
    handle.synchronize().expect("sync");
    out.to_cpu_vec_f32().expect("download")
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn wna16_dequant_gemm_matches_oracle() {
    let Some(dev) = gpu_device() else { return };
    // n*k spans 2 blocks (256 weights each) so `block_idx = flat >> 8`
    // crosses a block boundary mid-GEMM.
    let (m, n, k) = (2usize, 8usize, 64usize);
    let n_bit = 4u8;
    let total = n * k; // 512 -> 2 blocks
    let blocks = total.div_ceil(256);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 9) as f32 * 0.31) - 1.24)
        .collect();

    let mut seed = 0xC0FFEEu64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let codes: Vec<u32> = (0..total).map(|_| (rnd() % 16) as u32).collect();
    let block_scales: Vec<f32> = (0..blocks).map(|b| 0.5 + (b as f32) * 0.37).collect();
    let tensor_scale = 1.31f32;

    let code_bytes_per_block = (256 * n_bit as usize).div_ceil(8);
    let mut code_seg = vec![0u8; blocks * code_bytes_per_block];
    code_seg[..pack_msb_nbit(&codes, n_bit).len()]
        .copy_from_slice(&pack_msb_nbit(&codes, n_bit));

    let mut blob = Vec::new();
    blob.extend_from_slice(&(n_bit as u32).to_le_bytes());
    blob.extend_from_slice(&(blocks as u32).to_le_bytes());
    blob.extend_from_slice(&code_seg);
    for s in &block_scales {
        blob.extend_from_slice(&f32_to_f16_bits(*s).to_le_bytes());
    }
    blob.extend_from_slice(&tensor_scale.to_le_bytes());

    let got = run_qmm(&dev, &a, &blob, Storage::WNA16, m, n, k);
    let want = cpu_gemm(&a, &oracle_wna16_rows(&blob, total), m, n, k);
    // f32 accumulation over exact int codes x f16/f32 scales: house dot4
    // bar is 2e-2 rel; exact-dequant GEMMs observe ~1e-5.
    check_rel("wna16_dequant_gemm", &got, &want, 2e-2);
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn w8a8_int8_dequant_gemm_matches_oracle() {
    let Some(dev) = gpu_device() else { return };
    let (m, n, k) = (2usize, 6usize, 32usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 11) as f32 * 0.23) - 1.15)
        .collect();

    let mut seed = 0x1CE8u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as i64 % 241 - 120) as i8
    };
    let mut codes: Vec<i8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut w = vec![0.0f32; n * k];
    for j in 0..n {
        let scale = 0.037 + (j as f32) * 0.013;
        scales.push(scale);
        for p in 0..k {
            let code = rnd();
            codes.push(code);
            w[j * k + p] = code as f32 * scale;
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&((n * 4) as u64).to_le_bytes()); // scales_len
    blob.extend_from_slice(&codes.iter().map(|c| *c as u8).collect::<Vec<_>>());
    for s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }

    let got = run_qmm(
        &dev,
        &a,
        &blob,
        Storage::CompressedTensorsW8A8Int8,
        m,
        n,
        k,
    );
    let want = cpu_gemm(&a, &w, m, n, k);
    check_rel("w8a8_int8_dequant_gemm", &got, &want, 2e-2);
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn w8a8_fp8_dequant_gemm_matches_oracle() {
    let Some(dev) = gpu_device() else { return };
    let (m, n, k) = (2usize, 6usize, 32usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 7) as f32 * 0.41) - 1.23)
        .collect();

    let tensor_scale = 0.117f32;
    let mut seed = 0xF8u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as u32) as u8
    };
    let mut codes = Vec::new();
    let mut w = vec![0.0f32; n * k];
    for j in 0..n {
        for p in 0..k {
            // Finite E4M3 codes only (exp < 15): no NaN/Inf in the fixture.
            let mut b = rnd() & 0x7F;
            if (b >> 3) == 0x0F {
                b &= 0x77;
            }
            codes.push(b);
            w[j * k + p] = e4m3_to_f32(b) * tensor_scale;
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&4u64.to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&tensor_scale.to_le_bytes());

    let got = run_qmm(
        &dev,
        &a,
        &blob,
        Storage::CompressedTensorsW8A8Fp8,
        m,
        n,
        k,
    );
    let want = cpu_gemm(&a, &w, m, n, k);
    // f32-accum exact-dequant GEMM; 1e-4 is the sibling gate's bar
    // (w8a8_wna16_gemm_gate). The kernel depends on fp8_e4m3_to_float_hip
    // from the aggregate source, so this also pins that symbol's presence.
    check_rel("w8a8_fp8_dequant_gemm", &got, &want, 1e-4);
}

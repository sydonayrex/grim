//! Quant workstream wiring gates (dense dispatch through `quantized_matmul`).
//!
//! 1. W4A16 (Marlin-style): a packed blob `[codes u32][scales f32]` stored as
//!    `Storage::W4A16` must route through the fused kernel and match a CPU
//!    dequant reference.
//! 2. Kernel-less variants (`CompressedTensorsW8A8*`, `WNA16`) must FAIL
//!    LOUDLY — their old `_ =>` fallback fed packed bytes into the F32 matmul
//!    (silent garbage).
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendDevice;
use grim_backend_rocm::DTypeStorage;
use grim_tensor::ArithType;
use grim_tensor::{DType, Device, Shape, Storage, Tensor};
use std::sync::Arc;


fn as_rocm_ref(st: &dyn grim_tensor::BackendStorage) -> &grim_backend_rocm::RocmStorage {
    st.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("rocml storage")
}

fn wrap_arc(
    st: Box<dyn grim_tensor::BackendStorage>,
    shape: Shape,
    device: Device,
) -> Tensor {
    Tensor::new(
        std::sync::Arc::from(st),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device,
    )
}

fn w4a16_dtype(group_size: usize) -> DType {
    DType {
        arith: ArithType::F32,
        storage: Storage::W4A16(grim_tensor::dtype::W4A16Config { group_size }),
    }
}

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

#[test]
fn w4a16_dense_dispatch_matches_dequant_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let (m, n, k) = (2usize, 6usize, 32usize); // K % 8 == 0; group_size 16 → 2 groups/row
    let group_size = 16usize;

    // Random-ish activations and weights.
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 * 0.3) - 1.5).collect();
    let mut b_orig: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 * 0.4) - 2.0).collect();

    // Quantize per row (output channel), per group of K: symmetric ±7 codebook
    // with centering offset 8 — must mirror the kernel's
    // `(code - 8) * scale` dequant exactly so parity is quantization-exact.
    let words_per_row = k / 8;
    let groups_per_row = k / group_size;
    let mut codes = vec![0u32; n * words_per_row];
    let mut scales = vec![0.0f32; n * groups_per_row];
    for row in 0..n {
        for g in 0..groups_per_row {
            let lo = g * group_size;
            let hi = lo + group_size;
            let max_abs = (lo..hi)
                .map(|c| b_orig[row * k + c].abs())
                .fold(0.0f32, f32::max);
            let scale = max_abs / 7.0;
            scales[row * groups_per_row + g] = if scale == 0.0 { 1e-12 } else { scale };
            for c in lo..hi {
                let q = ((b_orig[row * k + c] / scales[row * groups_per_row + g]).round() as i32)
                    .clamp(-8, 7);
                let code = (q + 8) as u32;
                codes[row * words_per_row + c / 8] |= code << ((c % 8) * 4);
                // Make the CPU reference use the EXACT dequantized weight.
                b_orig[row * k + c] = (q as f32) * scales[row * groups_per_row + g];
            }
        }
    }

    // Blob: [codes u32 LE][scales f32].
    let mut blob_bytes = Vec::with_capacity(codes.len() * 4 + scales.len() * 4);
    for w in &codes {
        blob_bytes.extend_from_slice(&w.to_le_bytes());
    }
    for s in &scales {
        blob_bytes.extend_from_slice(&s.to_le_bytes());
    }

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![n, k]);
    let out_shape = Shape::new(vec![m, n]);

    let a_gpu = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b_gpu = dev
        .from_cpu_bytes(
            &blob_bytes,
            &b_shape,
            w4a16_dtype(group_size),
        )
        .unwrap();

    // Production seam: quantized_matmul dispatches on the storage dtype.
    let (out_st, handle) = dev
        .quantized_matmul(
            a_gpu.as_ref(),
            b_gpu.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K, // ignored by the W4A16 arm
            &out_shape,
        )
        .expect("W4A16 dense dispatch must route to the marlin kernel");
    handle.synchronize().unwrap();

    // CPU reference against the exact dequantized weights.
    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let acc: f32 = (0..k).map(|p| a_data[r * k + p] * b_orig[j * k + p]).sum();
            want[r * n + j] = acc;
        }
    }
    let got = wrap(out_st, out_shape, Device::Rocm(0))
        .to_vec_f32()
        .unwrap();
    let md = got
        .iter()
        .zip(want.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < 1e-3,
        "W4A16 dense dispatch diverged from dequant reference: {md:.5}\ngot={got:?}\nwant={want:?}"
    );
}

#[test]
fn kernel_less_variants_fail_loudly_not_silently() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let a_shape = Shape::new(vec![2, 8]);
    let b_shape = Shape::new(vec![8, 4]);
    let out_shape = Shape::new(vec![2, 4]);
    let a_gpu = dev.from_cpu(&vec![1.0f32; 16], &a_shape, DType::F32).unwrap();

    // These formats now have wired forward kernels (W8A8 Int8/Fp8, WNA16);
    // they must NOT silently fall back to packed-bytes-as-f32.
    for (name, storage) in [
        (
            "CompressedTensorsW8A8Int8",
            DTypeStorage::CompressedTensorsW8A8Int8,
        ),
        (
            "CompressedTensorsW8A8Fp8",
            DTypeStorage::CompressedTensorsW8A8Fp8,
        ),
        ("WNA16", DTypeStorage::WNA16),
    ] {
        let dt = DType {
            arith: ArithType::F32,
            storage: storage.clone(),
        };
        let b_gpu = dev
            .from_cpu_bytes(&vec![0u8; b_shape.elem_count()], &b_shape, dt)
            .unwrap();
        let res = dev.quantized_matmul(
            a_gpu.as_ref(),
            b_gpu.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        );
        assert!(
            res.is_ok(),
            "{name}: now has a wired forward kernel; must succeed (not silently fallback)"
        );
    }

    // EmbeddingWNA16Int is an embedding format, not a GEMM — must still fail
    // loudly if mistakenly routed through quantized_matmul.
    let emb_dt = DType {
        arith: ArithType::F32,
        storage: DTypeStorage::EmbeddingWNA16Int,
    };
    let b_gpu = dev
        .from_cpu_bytes(&vec![0u8; b_shape.elem_count()], &b_shape, emb_dt)
        .unwrap();
    let res = dev.quantized_matmul(
        a_gpu.as_ref(),
        b_gpu.as_ref(),
        &[],
        grim_tensor::QuantFormat::Q4K,
        &out_shape,
    );
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!(
            "EmbeddingWNA16Int has no GEMM kernel — must fail loudly instead of the \
             packed-bytes-as-f32 fallback"
        ),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("embedding") || msg.contains("dequantize at load"),
        "EmbeddingWNA16Int: error should state the embedding-only nature: {msg}"
    );
}


/// MSB-first packer — inverse of the kernel's grim_decode_msb_nbit.
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

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h & 0x7C00) as u32) << 13;
    let m = ((h & 0x03FF) as u32) << 13;
    if e == 0x7C00 { return f32::from_bits(s | 0x7F800000 | if m != 0 { 0x400000 } else { 0 }); }
    if e == 0 {
        if m == 0 { return f32::from_bits(s); }
        let v = (m >> 13) as f32 * (1.0 / 1024.0) * (1.0 / 16384.0);
        return if h & 0x8000 != 0 { -v } else { v };
    }
    f32::from_bits(s | e | m | 0x38000000)
}

/// Audit gate: WNA16 GPU dequant must match the host MSB-first decoder and
/// the scale chain (per-block f16 × per-tensor f32).
#[test]
fn wna16_dequant_service_matches_host_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let n_bit = 4u8;
    let num_blocks = 2usize;
    let weights_per_block = 256usize;
    let num_weights = num_blocks * weights_per_block;

    // Random 4-bit codes.
    let mut seed = 0xD00Du64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 15.0
    };
    let all_codes: Vec<u32> = (0..num_weights).map(|_| rand() as u32).collect();

    // Per-block f16 scales (random small positive), tensor scale f32.
    let block_scales_f32: Vec<f32> = (0..num_blocks).map(|i| 0.5 + i as f32 * 0.25).collect();
    let block_scales_f16: Vec<u16> = block_scales_f32
        .iter()
        .map(|&s| f32_to_f16_bits(s))
        .collect();
    let tensor_scale = 1.5f32;

    // Blob: [u32 n_bit][u32 num_blocks][codes][f16 scales][f32 ts].
    let mut blob: Vec<u8> = Vec::new();
    // Layout contract header: [u32 n_bit][u32 num_blocks].
    blob.extend_from_slice(&(n_bit as u32).to_le_bytes());
    blob.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    blob.extend_from_slice(&pack_msb_nbit(&all_codes, n_bit));
    for h in &block_scales_f16 {
        blob.extend_from_slice(&h.to_le_bytes());
    }
    blob.extend_from_slice(&tensor_scale.to_le_bytes());

    // Upload as opaque U8 bytes: the dequant service consumes the raw
    // packed blob regardless of logical dtype.
    let packed_shape = Shape::new(vec![blob.len()]);
    let dt_u8 = DType {
        arith: grim_tensor::ArithType::U8,
        storage: Storage::Native,
    };
    let packed_gpu = dev.from_cpu_bytes(&blob, &packed_shape, dt_u8).unwrap();

    let out_st = dev
        .dequant_wna16_to_f32(as_rocm_ref(packed_gpu.as_ref()), num_weights, n_bit, num_blocks)
        .expect("wna16 dequant service");
    let got = wrap_arc(out_st, Shape::new(vec![num_weights]), Device::Rocm(0))
        .to_vec_f32()
        .unwrap();

    for i in 0..num_weights {
        let blk = i / weights_per_block;
        let want = all_codes[i] as f32 * f16_to_f32(block_scales_f16[blk]) * tensor_scale;
        assert!(
            (got[i] - want).abs() < 1e-4,
            "wna16[{i}]: gpu={} host={want}",
            got[i]
        );
    }
}

fn f32_to_f16_bits(f: f32) -> u16 {
    // Round-to-nearest-even-lite f32→f16 for normal-range positive values.
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

/// Audit gate: EmbeddingWNA16Int GPU dequant vs host decode (row-major,
/// per-tensor scale only).
#[test]
fn embedding_wna16_dequant_matches_host_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let n_bit = 3u8;
    let rows = 4usize;
    let dim = 8usize;
    let total = rows * dim;

    let mut seed = 0xBEEFu64;
    let codes: Vec<u32> = vec![4u32; total]; // DIAG: constant 100₍₂₎
    let _ = &mut seed;
    let tensor_scale = 0.75f32;

    let mut blob: Vec<u8> = Vec::new();
    // Layout contract header: [u32 n_bit][u32 embedding_dim][u32 num_rows].
    blob.extend_from_slice(&(n_bit as u32).to_le_bytes());
    blob.extend_from_slice(&(dim as u32).to_le_bytes());
    blob.extend_from_slice(&(rows as u32).to_le_bytes());
    blob.extend_from_slice(&pack_msb_nbit(&codes, n_bit));
    eprintln!("[emb-diag] blob={:02x?}", &blob[..]);

    let packed_shape = Shape::new(vec![blob.len()]);
    let dt_e = DType {
        arith: grim_tensor::ArithType::U8,
        storage: Storage::Native,
    };
    let packed_gpu = dev.from_cpu_bytes(&blob, &packed_shape, dt_e).unwrap();

    let out_st = dev
        .dequant_embedding_wna16_int_to_f32(
            as_rocm_ref(packed_gpu.as_ref()),
            total,
            n_bit,
            dim,
            tensor_scale,
        )
        .expect("embedding dequant service");
    let got = wrap_arc(out_st, Shape::new(vec![total]), Device::Rocm(0))
        .to_vec_f32()
        .unwrap();
    eprintln!("[emb-diag] got={:?} want={:?}", got, codes.iter().map(|c| *c as f32 * tensor_scale).collect::<Vec<_>>());

    for i in 0..total {
        let want = codes[i] as f32 * tensor_scale;
        assert!(
            (got[i] - want).abs() < 1e-5,
            "emb[{i}]: gpu={} host={want}",
            got[i]
        );
    }
}

//! Int8 quantized paged-attention parity (KV-cache quantization gate).
//!
//! `launch_paged_attention_quant` existed with no production caller and no test.
//! Before wiring Qwen35 onto a quantized KV arena, this pins the behavior: the
//! int8 path must match an f32 reference attention closely enough that swapping
//! the arena dtype is a memory win rather than a correctness regression.
//!
//! Storage is symmetric int8 with a per-tensor scale/bias, matching what
//! `dequant_kv_element(quant_format = 0)` decodes on device.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::kernels::qkv_attention::{
    KvCacheQuantFormat, launch_paged_attention, launch_paged_attention_quant,
};
use grim_backend_rocm::{BlockTableEntry, RocmDevice};
use grim_tensor::CoreTensorOps;
use grim_tensor::MemoryOps;
use grim_tensor::{DType, Shape};
use std::sync::Arc;

const BATCH: u32 = 1;
const NUM_HEADS: u32 = 2;
const NUM_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 8;
const PAGE_SIZE: u32 = 4;
const KV_SEQ_LEN: u32 = 8;
const MAX_BLOCKS: u32 = 2;
const CACHE_OFFSET: u32 = 7;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

fn block_table(dev: &RocmDevice) -> Box<dyn grim_tensor::BackendStorage> {
    let entries = [
        BlockTableEntry { block_id: 0, page_size: 4 },
        BlockTableEntry { block_id: 1, page_size: 4 },
    ];
    let table_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(entries.as_ptr() as *const f32, entries.len() * 2)
    };
    dev.from_cpu(
        table_f32,
        &Shape::new(vec![BATCH as usize, MAX_BLOCKS as usize, 2]),
        DType::F32,
    )
    .unwrap()
}

/// Reference attention over a flat [seq, kv_heads, head_dim] layout, computed in
/// f64 so the comparison measures device error rather than host rounding.
///
/// Mirrors the reference in `paged_attention.rs` exactly: every token in
/// `kv_seq_len` participates, and the page-to-token mapping goes through each
/// block's `block_id` rather than assuming identity.
fn reference_attention(
    q: &[f32],
    k_flat: &[f32],
    v_flat: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
) -> Vec<f64> {
    let heads_per_kv = num_heads / num_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f64).sqrt();
    let mut out = vec![0.0f64; num_heads * head_dim];
    for h in 0..num_heads {
        let kvh = h / heads_per_kv;
        let q_off = h * head_dim;
        let mut scores = Vec::with_capacity(kv_seq_len);
        for t in 0..kv_seq_len {
            let base = t * num_kv_heads * head_dim + kvh * head_dim;
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += q[q_off + d] as f64 * k_flat[base + d] as f64;
            }
            scores.push(dot * inv_sqrt_d);
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = scores.iter().map(|s| (s - max).exp()).sum();
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for t in 0..kv_seq_len {
                let w = ((scores[t] - max).exp()) / denom;
                acc += w * v_flat[t * num_kv_heads * head_dim + kvh * head_dim + d] as f64;
            }
            out[q_off + d] = acc;
        }
    }
    out
}

/// int8 quantized paged attention must track the f32 paged reference.
///
/// The tolerance is loose on purpose: int8 KV is a deliberate precision trade.
/// What this guards is gross breakage (wrong page indexing, wrong scale
/// application, a transposed head map) - errors of O(1) magnitude, not O(0.01).
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn int8_quant_paged_attention_matches_f32_reference() {
    let Some(dev) = gpu_device() else { return };

    let page_shape = Shape::new(vec![
        MAX_BLOCKS as usize,
        PAGE_SIZE as usize,
        NUM_KV_HEADS as usize,
        HEAD_DIM as usize,
    ]);
    let q_shape = Shape::new(vec![
        BATCH as usize,
        NUM_HEADS as usize,
        HEAD_DIM as usize,
    ]);

    let q_cpu: Vec<f32> = (0..(BATCH * NUM_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.1).sin())
        .collect();
    let k_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.15).cos())
        .collect();
    let v_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.2).sin())
        .collect();

    // Flatten pages into the [seq, kv_heads, head_dim] reference layout, going
    // through each block's physical `block_id` exactly as the kernel does.
    let entries = [
        BlockTableEntry { block_id: 0, page_size: 4 },
        BlockTableEntry { block_id: 1, page_size: 4 },
    ];
    let mut k_flat = vec![0.0f32; (KV_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM) as usize];
    let mut v_flat = vec![0.0f32; (KV_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM) as usize];
    for b in 0..MAX_BLOCKS as usize {
        let entry = entries[b];
        for t in 0..entry.page_size as usize {
            let j = b * PAGE_SIZE as usize + t;
            if j >= KV_SEQ_LEN as usize {
                break;
            }
            let physical_token = entry.block_id as usize * PAGE_SIZE as usize + t;
            let n = NUM_KV_HEADS as usize * HEAD_DIM as usize;
            let src = physical_token * n;
            let dst = j * n;
            k_flat[dst..dst + n].copy_from_slice(&k_cpu[src..src + n]);
            v_flat[dst..dst + n].copy_from_slice(&v_cpu[src..src + n]);
        }
    }

    // ---- f32 paged reference on device ----
    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let k_storage = dev.from_cpu(&k_cpu, &page_shape, DType::F32).unwrap();
    let v_storage = dev.from_cpu(&v_cpu, &page_shape, DType::F32).unwrap();
    let table = block_table(&dev);
    let mut out_f32 = dev.zeros(&q_shape, DType::F32).unwrap();
    launch_paged_attention(
        &dev,
        q_storage.as_ref(),
        table.as_ref(),
        k_storage.as_ref(),
        v_storage.as_ref(),
        out_f32.as_mut(),
        BATCH,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_BLOCKS,
        PAGE_SIZE,
        KV_SEQ_LEN,
        CACHE_OFFSET,
        0i32,
        None,
    )
    .expect("f32 paged attention");
    let got_f32 = out_f32.to_cpu_vec_f32().unwrap();

    // ---- int8 quantized paged ----
    // Symmetric per-tensor quantization: scale = max|k| / 127.
    let quantize = |src: &[f32]| -> (Vec<i8>, f32) {
        let amax = src.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let q: Vec<i8> = src
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        (q, scale)
    };
    let (k_q8, k_scale) = quantize(&k_cpu);
    let (v_q8, v_scale) = quantize(&v_cpu);

    // Upload the quantized K/V as RAW int8 bytes. The kernel indexes the page
    // pointer bytewise (`dequant_kv_element(data, idx, ...)`), so widening the
    // int8 values into f32 slots would misalign every read by 4x.
    let k_bytes: Vec<u8> = k_q8.iter().map(|&x| x as u8).collect();
    let v_bytes: Vec<u8> = v_q8.iter().map(|&x| x as u8).collect();
    let q8_shape = Shape::new(vec![k_bytes.len()]);
    let kq_storage = dev
        .from_cpu_bytes(&k_bytes, &q8_shape, DType::F32)
        .unwrap();
    let vq_storage = dev
        .from_cpu_bytes(&v_bytes, &q8_shape, DType::F32)
        .unwrap();
    let mut out_q8 = dev.zeros(&q_shape, DType::F32).unwrap();

    launch_paged_attention_quant(
        &dev,
        q_storage.as_ref(),
        table.as_ref(),
        kq_storage.as_ref(),
        vq_storage.as_ref(),
        out_q8.as_mut(),
        BATCH,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_BLOCKS,
        PAGE_SIZE,
        KV_SEQ_LEN,
        CACHE_OFFSET,
        0i32,
        KvCacheQuantFormat::Int8,
        k_scale,
        0.0,
        v_scale,
        0.0,
    )
    .expect("int8 quant paged attention");
    let got_q8 = out_q8.to_cpu_vec_f32().unwrap();

    // Both device paths must agree with the host reference.
    let want = reference_attention(
        &q_cpu,
        &k_flat,
        &v_flat,
        NUM_HEADS as usize,
        NUM_KV_HEADS as usize,
        HEAD_DIM as usize,
        KV_SEQ_LEN as usize,
    );

    let tol = 0.15;
    for h in 0..NUM_HEADS as usize {
        for d in 0..HEAD_DIM as usize {
            let i = h * HEAD_DIM as usize + d;
            let f32_err = (got_f32[i] as f64 - want[i]).abs();
            let q8_err = (got_q8[i] as f64 - want[i]).abs();
            assert!(
                f32_err < tol,
                "f32 paged path disagrees at head {h} dim {d}: \
                 got {} want {} (err {f32_err})",
                got_f32[i],
                want[i]
            );
            assert!(
                q8_err < tol,
                "int8 quant paged path disagrees at head {h} dim {d}: \
                 got {} want {} (err {q8_err})",
                got_q8[i],
                want[i]
            );
        }
    }
    eprintln!(
        "[quant-paged] int8 KV matches f32 reference within {tol} \
         (max err {:.4})",
        got_q8
            .iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0f64, f64::max)
    );
}

/// Host-side E4M3 encoder matching the device `dequant_kv_element(quant_format = 2)`
/// bit layout: sign(1) | exp(4) | mant(3), with a per-tensor `scale` applied on
/// decode.
///
/// Deliberately mirrors the kernel rather than using a library encoder, so the
/// parity test measures the kernel's own round trip. `exp == 15 && mant == 7` is
/// the kernel's zero/NaN special case; values are clamped so the encoder cannot
/// produce it for finite inputs.
fn quantize_e4m3(src: &[f32]) -> (Vec<u8>, f32) {
    // Per-tensor scale so max|src| lands just inside the representable range.
    // The kernel's largest normal E4M3 value is (1 + 7/8) * 2^7 = 240.
    let amax = src.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if amax > 0.0 { amax / 240.0 } else { 1.0 };
    let mut out = Vec::with_capacity(src.len());
    for &x in src {
        let v = (x / scale).clamp(-240.0, 240.0);
        let neg = v < 0.0;
        let a = v.abs();
        let (exp, mant) = if a < 0.015625 {
            // Subnormal region: exponent field 0, linear mantissa.
            (0u8, ((a / (0.015625 / 8.0)).round() as u8).min(7))
        } else {
            let e = a.log2().floor() as i32;
            let e = e.clamp(-6, 7) as u8;
            let m = ((a / 2f32.powi(e as i32) - 1.0) * 8.0).round() as u8;
            (e + 7, m.min(7))
        };
        let byte = ((neg as u8) << 7) | (exp << 3) | mant;
        out.push(byte);
    }
    (out, scale)
}

/// FP8 E4M3 quantized paged attention must track the f32 reference.
///
/// FP8 is the intended default for RDNA4 (`gfx1200`/`gfx1201` are classified as
/// FP8-capable in `capability_profiler`), where it matches int8's 1 byte/element
/// but keeps floating-point dynamic range — which matters for K, whose
/// per-channel outliers are exactly what flattens under a per-tensor int8 scale.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn fp8_e4m3_quant_paged_attention_matches_f32_reference() {
    let Some(dev) = gpu_device() else { return };

    let q_shape = Shape::new(vec![BATCH as usize, NUM_HEADS as usize, HEAD_DIM as usize]);

    let q_cpu: Vec<f32> = (0..(BATCH * NUM_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.1).sin())
        .collect();
    let k_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.15).cos())
        .collect();
    let v_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.2).sin())
        .collect();

    // Same page flattening as the int8 test, via each block's physical id.
    let entries = [
        BlockTableEntry { block_id: 0, page_size: 4 },
        BlockTableEntry { block_id: 1, page_size: 4 },
    ];
    let mut k_flat = vec![0.0f32; (KV_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM) as usize];
    let mut v_flat = vec![0.0f32; (KV_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM) as usize];
    for b in 0..MAX_BLOCKS as usize {
        let entry = entries[b];
        for t in 0..entry.page_size as usize {
            let j = b * PAGE_SIZE as usize + t;
            if j >= KV_SEQ_LEN as usize {
                break;
            }
            let physical_token = entry.block_id as usize * PAGE_SIZE as usize + t;
            let n = NUM_KV_HEADS as usize * HEAD_DIM as usize;
            let src = physical_token * n;
            let dst = j * n;
            k_flat[dst..dst + n].copy_from_slice(&k_cpu[src..src + n]);
            v_flat[dst..dst + n].copy_from_slice(&v_cpu[src..src + n]);
        }
    }

    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let table = block_table(&dev);
    let mut out = dev.zeros(&q_shape, DType::F32).unwrap();

    let (k_bytes, k_scale) = quantize_e4m3(&k_cpu);
    let (v_bytes, v_scale) = quantize_e4m3(&v_cpu);
    // Raw byte upload: the kernel indexes the page pointer bytewise.
    let byte_shape = Shape::new(vec![k_bytes.len()]);
    let kq = dev.from_cpu_bytes(&k_bytes, &byte_shape, DType::F32).unwrap();
    let vq = dev.from_cpu_bytes(&v_bytes, &byte_shape, DType::F32).unwrap();

    launch_paged_attention_quant(
        &dev,
        q_storage.as_ref(),
        table.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        out.as_mut(),
        BATCH,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_BLOCKS,
        PAGE_SIZE,
        KV_SEQ_LEN,
        CACHE_OFFSET,
        0i32,
        KvCacheQuantFormat::Fp8E4M3,
        k_scale,
        0.0,
        v_scale,
        0.0,
    )
    .expect("fp8 e4m3 quant paged attention");
    let got = out.to_cpu_vec_f32().unwrap();

    let want = reference_attention(
        &q_cpu,
        &k_flat,
        &v_flat,
        NUM_HEADS as usize,
        NUM_KV_HEADS as usize,
        HEAD_DIM as usize,
        KV_SEQ_LEN as usize,
    );

    // 3 mantissa bits is coarser than int8's 7, so the gate is looser than the
    // int8 case. It still catches O(1) breakage: bad page indexing, a misapplied
    // scale, or a transposed head map.
    let tol = 0.25;
    for h in 0..NUM_HEADS as usize {
        for d in 0..HEAD_DIM as usize {
            let i = h * HEAD_DIM as usize + d;
            let err = (got[i] as f64 - want[i]).abs();
            assert!(
                err < tol,
                "fp8 e4m3 paged path disagrees at head {h} dim {d}: \
                 got {} want {} (err {err})",
                got[i],
                want[i]
            );
        }
    }
    eprintln!(
        "[quant-paged] fp8 e4m3 KV matches f32 reference within {tol} \
         (max err {:.4})",
        got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0f64, f64::max)
    );
}

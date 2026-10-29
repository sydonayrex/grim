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
        BlockTableEntry {
            block_id: 0,
            page_size: 4,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: 4,
        },
    ];
    let table_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(entries.as_ptr() as *const f32, entries.len() * 2) };
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

    // Flatten pages into the [seq, kv_heads, head_dim] reference layout, going
    // through each block's physical `block_id` exactly as the kernel does.
    let entries = [
        BlockTableEntry {
            block_id: 0,
            page_size: 4,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: 4,
        },
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
    let kq_storage = dev.from_cpu_bytes(&k_bytes, &q8_shape, DType::F32).unwrap();
    let vq_storage = dev.from_cpu_bytes(&v_bytes, &q8_shape, DType::F32).unwrap();
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
        BlockTableEntry {
            block_id: 0,
            page_size: 4,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: 4,
        },
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
    let kq = dev
        .from_cpu_bytes(&k_bytes, &byte_shape, DType::F32)
        .unwrap();
    let vq = dev
        .from_cpu_bytes(&v_bytes, &byte_shape, DType::F32)
        .unwrap();

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
        got.iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0f64, f64::max)
    );
}

/// Nutcracker quantized paged attention must track the f32 reference.
///
/// Nutcracker is E2M1 codes plus a per-16 block scale carried *inline* in the
/// buffer, whose low 2 bits are stolen as a special-value selector. It is the
/// only variant that ignores the caller's per-tensor `k_scale`/`v_scale` — the
/// scale travels in the data — so the launch passes 1.0 to make that explicit.
///
/// At 4.5 bits/element this is the format that would actually let a long
/// context fit, which is why it needs the same gate as int8 and FP8.
///
/// The host side uses the real `grim_quant::quant_nutcracker`, so this measures
/// the device decoder against the production encoder rather than a mirror of it.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn nutcracker_quant_paged_attention_matches_f32_reference() {
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

    let entries = [
        BlockTableEntry {
            block_id: 0,
            page_size: 4,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: 4,
        },
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

    // Production encoder: 9 bytes per 16 values.
    let k_packed = grim_quant::quant_nutcracker(&k_cpu).expect("quantize K");
    let v_packed = grim_quant::quant_nutcracker(&v_cpu).expect("quantize V");
    assert_eq!(
        k_packed.len(),
        k_cpu.len() / 16 * 9,
        "9 bytes per 16 values"
    );

    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let table = block_table(&dev);
    let mut out = dev.zeros(&q_shape, DType::F32).unwrap();

    // Raw byte upload: the decoder indexes the page pointer bytewise.
    let kq = dev
        .from_cpu_bytes(&k_packed, &Shape::new(vec![k_packed.len()]), DType::F32)
        .unwrap();
    let vq = dev
        .from_cpu_bytes(&v_packed, &Shape::new(vec![v_packed.len()]), DType::F32)
        .unwrap();

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
        KvCacheQuantFormat::NutFp4,
        1.0, // ignored: Nutcracker carries its scale inline
        0.0,
        1.0, // ignored
        0.0,
    )
    .expect("nutcracker quant paged attention");
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

    // 4-bit codes on a coarse E2M1 grid, so the gate is far looser than int8 or
    // FP8. It must still catch O(1) breakage: wrong block indexing, ignoring the
    // inline scale, or misreading the stolen selector.
    //
    // Calibrated against measured max error on the reference data: the correct
    // path lands at ~0.08. Removing the special-value branch degrades it to
    // ~0.25, so 0.15 discriminates a correct selector decode from a broken one.
    // The previous 1.0 passed both.
    let tol = 0.15;
    for h in 0..NUM_HEADS as usize {
        for d in 0..HEAD_DIM as usize {
            let i = h * HEAD_DIM as usize + d;
            let err = (got[i] as f64 - want[i]).abs();
            assert!(
                err < tol,
                "nutcracker paged path disagrees at head {h} dim {d}: \
                 got {} want {} (err {err})",
                got[i],
                want[i]
            );
        }
    }
    eprintln!(
        "[quant-paged] nutcracker KV matches f32 reference within {tol} \
         (max err {:.4})",
        got.iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0f64, f64::max)
    );
}

/// WhiteCrow (u4 group-128 OSTQuant) quantized paged attention must track the
/// f32 reference.
///
/// WhiteCrow packs 128 values into 67 bytes (64 B u4 codes + 2 B bf16 scale +
/// 1 B u8 zero), with an affine decode `scale * (nibble - zero)` per group. The
/// scale travels inline in the block, so the launch passes `k_scale = v_scale =
/// 1.0` to make that explicit (like Nutcracker).
///
/// The host side uses the production `grim_quant::quant_whitecrow`, so this
/// measures the device `dequant_kv_element(format = 9)` leg against the real
/// encoder. The flat physical element index the kernel passes (`elem_offset +
/// dim` = `[physical_token][kv_head][dim]`) is exactly the sequential grouping
/// the host packer uses, so the two agree by construction — which is the
/// invariant this pins.
///
/// Gated: `GRIM_GPU_TEST=1` + a real ROCm device.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn whitecrow_quant_paged_attention_matches_f32_reference() {
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

    // Same page flattening as the other tests, via each block's physical id.
    let entries = [
        BlockTableEntry {
            block_id: 0,
            page_size: 4,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: 4,
        },
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

    // Production encoder: 67 bytes per 128 values. Here the tensors are exactly
    // 128 elements, so each packs to a single 67-byte group.
    let k_packed = grim_quant::quant_whitecrow(&k_cpu).expect("quantize K");
    let v_packed = grim_quant::quant_whitecrow(&v_cpu).expect("quantize V");
    assert_eq!(
        k_packed.len(),
        k_cpu.len().div_ceil(128) * 67,
        "67 bytes per 128 values"
    );

    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let table = block_table(&dev);
    let mut out = dev.zeros(&q_shape, DType::F32).unwrap();

    // Raw byte upload: the decoder indexes the page pointer bytewise.
    let kq = dev
        .from_cpu_bytes(&k_packed, &Shape::new(vec![k_packed.len()]), DType::F32)
        .unwrap();
    let vq = dev
        .from_cpu_bytes(&v_packed, &Shape::new(vec![v_packed.len()]), DType::F32)
        .unwrap();

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
        KvCacheQuantFormat::WhiteCrow,
        1.0, // ignored: WhiteCrow carries its bf16 scale inline per group
        0.0,
        1.0, // ignored
        0.0,
    )
    .expect("whitecrow quant paged attention");
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

    // Two gates:
    //
    // 1. TIGHT — device output vs a host reference that decodes the packed K/V
    //    with the *production* `dequant_whitecrow` and runs the same attention.
    //    head_dim here is 8 (shared consts), so this leg exercises the LEGACY
    //    per-element format-9 dequant with UNQUANTIZED Q; the sudot8 leg
    //    (head_dim % 128 == 0, quantized Q) is gated separately in
    //    `whitecrow_paged_sudot8_head128`.
    let k_host =
        grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&k_flat).unwrap(), k_flat.len())
            .unwrap();
    let v_host =
        grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&v_flat).unwrap(), v_flat.len())
            .unwrap();
    let host_ref = reference_attention(
        &q_cpu,
        &k_host,
        &v_host,
        NUM_HEADS as usize,
        NUM_KV_HEADS as usize,
        HEAD_DIM as usize,
        KV_SEQ_LEN as usize,
    );
    let decode_err = got
        .iter()
        .zip(&host_ref)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        decode_err < 0.01,
        "whitecrow device decode disagrees with production host decoder: \
         device/host max err {decode_err} (nibble/scale/zero/index bug)"
    );

    // 2. LOOSE — parity vs the pure f32 reference. The residual is inherent
    //    per-128 4-bit quantization coarseness, NOT a device error: the host
    //    calibration (`whitecrow_host_quant_fidelity_baseline`) measures the
    //    encoder's own fidelity on this data at ~0.0496, so the device lands
    //    exactly at the encoder's floor. Calibrated with headroom above that
    //    floor; a broken decode degrades to O(0.1+).
    let tol = 0.06;
    for h in 0..NUM_HEADS as usize {
        for d in 0..HEAD_DIM as usize {
            let i = h * HEAD_DIM as usize + d;
            let err = (got[i] as f64 - want[i]).abs();
            assert!(
                err < tol,
                "whitecrow paged path disagrees at head {h} dim {d}: \
                 got {} want {} (err {err})",
                got[i],
                want[i]
            );
        }
    }
    eprintln!(
        "[quant-paged] whitecrow KV matches f32 reference within {tol} \
         (max err {:.4}; device-vs-host-decode {decode_err:.6})",
        got.iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0f64, f64::max)
    );
}

/// sudot8 Q·K leg of the paged WhiteCrow path, at head_dim = 128 (the
/// geometry where the flat-128 K groups align with per-head Q groups and the
/// kernel takes the hardware u4 dot). Q is quantized to the symmetric u4
/// group-128 grid on device (`grim_whitecrow_qk_dot_scalar`); the tight gate
/// compares against a host reference using the canonical
/// `quant_q_whitecrow_symmetric` mirror, so it isolates the device dot's
/// packing/algebra, not the grid's inherent coarseness.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn whitecrow_paged_sudot8_head128() {
    let Some(dev) = gpu_device() else { return };

    // head_dim = 128: one WhiteCrow group per (token, kv_head) exactly.
    let (num_heads, num_kv_heads, head_dim) = (2usize, 2usize, 128usize);
    let (page_size, kv_seq_len, max_blocks, cache_offset) = (4usize, 8usize, 2usize, 7usize);
    let q_shape = Shape::new(vec![1, num_heads, head_dim]);

    let q_cpu: Vec<f32> = (0..num_heads * head_dim)
        .map(|x| ((x as f32) * 0.1).sin())
        .collect();
    let total = max_blocks * page_size * num_kv_heads * head_dim;
    let k_cpu: Vec<f32> = (0..total).map(|x| ((x as f32) * 0.15).cos()).collect();
    let v_cpu: Vec<f32> = (0..total).map(|x| ((x as f32) * 0.2).sin()).collect();

    // Identity page mapping (block b owns tokens [b*page_size, ..)).
    let entries = [
        BlockTableEntry {
            block_id: 0,
            page_size: page_size as u32,
        },
        BlockTableEntry {
            block_id: 1,
            page_size: page_size as u32,
        },
    ];
    let n = num_kv_heads * head_dim;
    let mut k_flat = vec![0.0f32; kv_seq_len * n];
    let mut v_flat = vec![0.0f32; kv_seq_len * n];
    for b in 0..max_blocks {
        for t in 0..page_size {
            let j = b * page_size + t;
            if j >= kv_seq_len {
                break;
            }
            let src = entries[b].block_id as usize * page_size * n + t * n;
            k_flat[j * n..(j + 1) * n].copy_from_slice(&k_cpu[src..src + n]);
            v_flat[j * n..(j + 1) * n].copy_from_slice(&v_cpu[src..src + n]);
        }
    }

    let k_packed = grim_quant::quant_whitecrow(&k_cpu).expect("quantize K");
    let v_packed = grim_quant::quant_whitecrow(&v_cpu).expect("quantize V");

    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let table = block_table(&dev);
    let mut out = dev.zeros(&q_shape, DType::F32).unwrap();
    let kq = dev
        .from_cpu_bytes(&k_packed, &Shape::new(vec![k_packed.len()]), DType::F32)
        .unwrap();
    let vq = dev
        .from_cpu_bytes(&v_packed, &Shape::new(vec![v_packed.len()]), DType::F32)
        .unwrap();

    launch_paged_attention_quant(
        &dev,
        q_storage.as_ref(),
        table.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        out.as_mut(),
        1,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        max_blocks as u32,
        page_size as u32,
        kv_seq_len as u32,
        cache_offset as u32,
        0i32,
        KvCacheQuantFormat::WhiteCrow,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    .expect("whitecrow sudot8 paged attention");
    let got = out.to_cpu_vec_f32().unwrap();

    // Tight: quantized-Q host reference over production-decoded K/V.
    let k_host =
        grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&k_flat).unwrap(), k_flat.len())
            .unwrap();
    let v_host =
        grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&v_flat).unwrap(), v_flat.len())
            .unwrap();
    let q_host = grim_quant::quant_q_whitecrow_symmetric(&q_cpu);
    let host_ref = reference_attention(
        &q_host,
        &k_host,
        &v_host,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
    );
    let decode_err = got
        .iter()
        .zip(&host_ref)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0f64, f64::max);
    eprintln!("[quant-paged-sudot8] device vs quantized-Q host ref: max err {decode_err:.6}");
    assert!(
        decode_err < 0.01,
        "whitecrow sudot8 paged dot disagrees with quantized-Q host reference: \
         max err {decode_err} (packing/zero-correction bug)"
    );

    // Loose: f32 end-to-end fidelity (K/V affine grid + Q symmetric grid).
    let want = reference_attention(
        &q_cpu,
        &k_flat,
        &v_flat,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
    );
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0f64, f64::max);
    eprintln!("[quant-paged-sudot8] vs f32 reference: max err {worst:.4}");
    assert!(worst < 0.15, "whitecrow sudot8 f32 fidelity too low: {worst}");
}

/// Host-only calibration: how much error does the WhiteCrow encoder itself
/// introduce on this exact data, independent of the GPU? Quantize K/V with the
/// production encoder, dequantize with the production decoder, run the same
/// `reference_attention`, and diff against the f32 reference.
///
/// This is NOT device-gated (pure host math). It separates "the 4-bit per-128
/// grid is inherently ~X accurate" from "the device decode leg is buggy": if
/// the device result lands within the same ballpark as this host-dequant
/// fidelity, the decode is correct and any residual gap is encoder coarseness.
#[test]
fn whitecrow_host_quant_fidelity_baseline() {
    let k_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.15).cos())
        .collect();
    let v_cpu: Vec<f32> = (0..(MAX_BLOCKS * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.2).sin())
        .collect();
    let q_cpu: Vec<f32> = (0..(BATCH * NUM_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.1).sin())
        .collect();

    let kd = grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&k_cpu).unwrap(), k_cpu.len()).unwrap();
    let vd = grim_quant::dequant_whitecrow(&grim_quant::quant_whitecrow(&v_cpu).unwrap(), v_cpu.len()).unwrap();
    let kerr = k_cpu.iter().zip(&kd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let verr = v_cpu.iter().zip(&vd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    // The test's flat layout already matches the physical-token order (block_id
    // == b), so no page reordering is needed here.
    let want = reference_attention(&q_cpu, &k_cpu, &v_cpu, NUM_HEADS as usize, NUM_KV_HEADS as usize, HEAD_DIM as usize, KV_SEQ_LEN as usize);
    let got = reference_attention(&q_cpu, &kd, &vd, NUM_HEADS as usize, NUM_KV_HEADS as usize, HEAD_DIM as usize, KV_SEQ_LEN as usize);
    let attn_err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f64, f64::max);
    eprintln!(
        "[whitecrow-host] dequant elem max err K={kerr:.4} V={verr:.4}; \
         host-dequant attention vs f32 ref max err {attn_err:.4}"
    );
}

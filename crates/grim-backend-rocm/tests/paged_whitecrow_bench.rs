//! Paged WhiteCrow Q·K roofline characterization at M=1 cold decode.
//!
//! The GEMV v3 lesson: M=1 decode is often DRAM-bound, so sudot8 (which speeds
//! compute, not K-byte traffic) may yield no wall-clock win. Before building a
//! sudot8 Q·K inner loop for the paged WhiteCrow path, measure whether the
//! current scalar dequant+f32-mul-add Q·K is compute-bound or memory-bound.
//!
//! Method: time `launch_paged_attention_quant` (WhiteCrow) across context
//! lengths at M=1. Compare against (a) the DRAM roofline prediction (bytes on
//! the wire ÷ effective bandwidth) and (b) the f32 paged baseline
//! (`launch_paged_attention`) at the same grid. If WhiteCrow ≈ f32, the
//! slowness is generic (1-workgroup serial loop), not dequant → sudot8 is
//! irrelevant. If WhiteCrow >> f32, dequant serial reads dominate → sudot8 has
//! a shot.
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

const NUM_HEADS: u32 = 1;
const NUM_KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 128;
const PAGE_SIZE: u32 = 16;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

struct Setup {
    q: Box<dyn grim_tensor::BackendStorage>,
    table: Box<dyn grim_tensor::BackendStorage>,
    k: Box<dyn grim_tensor::BackendStorage>,
    v: Box<dyn grim_tensor::BackendStorage>,
    out: Box<dyn grim_tensor::BackendStorage>,
    max_blocks: u32,
}

fn setup(dev: &RocmDevice, kv_seq_len: u32, whitecrow: bool) -> Setup {
    let max_blocks = kv_seq_len.div_ceil(PAGE_SIZE);
    let q_shape = Shape::new(vec![1, NUM_HEADS as usize, HEAD_DIM as usize]);
    let n_kv = (kv_seq_len * NUM_KV_HEADS * HEAD_DIM) as usize;

    let q_cpu: Vec<f32> = (0..(NUM_HEADS * HEAD_DIM) as usize)
        .map(|x| ((x as f32) * 0.1).sin())
        .collect();
    let k_cpu: Vec<f32> = (0..n_kv).map(|x| ((x as f32) * 0.15).cos()).collect();
    let v_cpu: Vec<f32> = (0..n_kv).map(|x| ((x as f32) * 0.2).sin()).collect();

    let q = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();

    let entries: Vec<BlockTableEntry> = (0..max_blocks)
        .map(|i| BlockTableEntry { block_id: i, page_size: PAGE_SIZE })
        .collect();
    let table_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(entries.as_ptr() as *const f32, entries.len() * 2) };
    let table = dev
        .from_cpu(table_f32, &Shape::new(vec![1, max_blocks as usize, 2]), DType::F32)
        .unwrap();

    let (k, v) = if whitecrow {
        let k_packed = grim_quant::quant_whitecrow(&k_cpu).expect("quantize K");
        let v_packed = grim_quant::quant_whitecrow(&v_cpu).expect("quantize V");
        (
            dev.from_cpu_bytes(&k_packed, &Shape::new(vec![k_packed.len()]), DType::F32).unwrap(),
            dev.from_cpu_bytes(&v_packed, &Shape::new(vec![v_packed.len()]), DType::F32).unwrap(),
        )
    } else {
        // f32 baseline: [num_pages, page_size, num_kv_heads, head_dim]
        let kv_shape =
            Shape::new(vec![max_blocks as usize, PAGE_SIZE as usize, NUM_KV_HEADS as usize, HEAD_DIM as usize]);
        (
            dev.from_cpu(&k_cpu, &kv_shape, DType::F32).unwrap(),
            dev.from_cpu(&v_cpu, &kv_shape, DType::F32).unwrap(),
        )
    };
    let out = dev.zeros(&q_shape, DType::F32).unwrap();
    Setup { q, table, k, v, out, max_blocks }
}

fn run_whitecrow(dev: &RocmDevice, s: &mut Setup, kv_seq_len: u32) {
    launch_paged_attention_quant(
        dev,
        s.q.as_ref(),
        s.table.as_ref(),
        s.k.as_ref(),
        s.v.as_ref(),
        s.out.as_mut(),
        1,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        s.max_blocks,
        PAGE_SIZE,
        kv_seq_len,
        // The kernel's attention range is [0, min(cache_offset+1, kv_seq_len)).
        // Point cache_offset at the last token so the Q·K loop covers the FULL context.
        kv_seq_len - 1,
        0i32,
        KvCacheQuantFormat::WhiteCrow,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    .expect("launch whitecrow");
}

fn run_f32(dev: &RocmDevice, s: &mut Setup, kv_seq_len: u32) {
    launch_paged_attention(
        dev,
        s.q.as_ref(),
        s.table.as_ref(),
        s.k.as_ref(),
        s.v.as_ref(),
        s.out.as_mut(),
        1,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        s.max_blocks,
        PAGE_SIZE,
        kv_seq_len,
        kv_seq_len - 1,
        0i32,
        None,
    )
    .expect("launch f32");
}

/// Returns (synced_us, k_bytes) for one config. iters syncs kept low (kernel is slow).
fn bench(dev: &RocmDevice, kv_seq_len: u32, whitecrow: bool) -> f64 {
    let mut s = setup(dev, kv_seq_len, whitecrow);
    let mut run = || {
        if whitecrow { run_whitecrow(dev, &mut s, kv_seq_len); } else { run_f32(dev, &mut s, kv_seq_len); }
    };
    // Warmup + sync.
    run();
    dev.synchronize();

    let iters = 3u32;
    let t = std::time::Instant::now();
    for _ in 0..iters { run(); }
    dev.synchronize();
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[test]
#[ignore = "roofline characterization — run manually with --ignored"]
fn paged_whitecrow_qk_roofline_m1() {
    let Some(dev) = gpu_device() else { return };

    // Cap context so the full sweep stays under test timeout; the f32 run at
    // 262144 is ~5s x iters alone.
    let contexts = [1024u32, 4096, 16384, 65536];

    eprintln!("\n=== paged Q·K roofline @ M=1 (head_dim={}, 1 workgroup) ===", HEAD_DIM);
    eprintln!(
        "{:>8} {:>13} {:>13} {:>9} {:>11}",
        "ctx", "whitecrow-us", "f32-us", "wc/f32", "WC-us/tok"
    );

    for &ctx in &contexts {
        let wc_us = bench(&dev, ctx, true);
        let f32_us = bench(&dev, ctx, false);
        let ratio = wc_us / f32_us;
        eprintln!(
            "{:>8} {:>13.1} {:>13.1} {:>9.2} {:>11.2}",
            ctx, wc_us, f32_us, ratio, wc_us / ctx as f64
        );
    }

    // Verdict.
    eprintln!("\n--- verdict ---");
    eprintln!("WhiteCrow dequant-vs-f32 ratio per ctx (see table above).");
    eprintln!("If ratio ≈ 1.0 → slowness is GENERIC (1-workgroup serial loop, not dequant).");
    eprintln!("  → sudot8 targets the inner-loop compute, irrelevant to the real bottleneck. SKIP sudot8.");
    eprintln!("If ratio >> 1.0 (e.g. >2x) → WhiteCrow dequant serial reads dominate. sudot8 WORTH building.");
}

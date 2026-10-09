//! Shape sweep for the f16-WMMA fused-dequant Q4_K GEMM, to localise a stall
//! that cannot be found from the outside.
//!
//! Why this exists: at M=251 on the 9B this kernel is 38x off its ALU bound,
//! 369x off the DRAM floor and ~20x off f16-WMMA peak, with healthy occupancy
//! (vgpr=88, LDS=6656 B -> ~5 blocks/CU). `rocprofv3 --pmc` cannot bracket it
//! (counter replay plus interdependent kernels gives a queue-sync timeout), so
//! the cost has to be split by timing controlled shapes:
//!
//!   * K sweep at fixed M and N -> if time is linear in K, the cost lives in
//!     the per-k-iteration body (LDS fill + 2x __syncthreads + 4 fragment
//!     loads + 4 MMAs, repeated K/16 times).
//!   * M sweep at fixed K and N -> if time is linear in ceil(M/16), the cost is
//!     per M-tile (i.e. the weight re-reads / grid waves), not in the body.
//!
//! Both are reported per iteration so the dominant term is readable directly.
//!
//! Kernel geometry (see `launch_wmma_fused_dequant_q4k`): tile 16 rows x 64
//! columns, `grid = (n/64, ceil(m/16))`, `block = 128` (4 waves). Reference for
//! the atom: FlyDSL `kernels/gemm/rdna_f16_gemm.py`, written for gfx120x, which
//! documents the 16x16x16 WMMA shape and the 2x2 wave layout this kernel does
//! NOT use (it gives all four waves the same tile).
//!
//! ---------------------------------------------------------------------------
//! WHAT WAS REFUTED HERE, so the next attempt does not re-run it (2026-10-08):
//!
//! With per-k-step at ~2.06 us and 38x off the ALU bound, three candidate
//! mechanisms were tested and all three are refuted:
//!
//!   1. LDS BANK CONFLICTS. Padded every LDS row stride 16 -> 24 halves (the
//!      `a_k_pad=8` convention of FlyDSL `kernels/gemm/rdna_f16_gemm.py`), which
//!      takes the B-fill store (`lds_b[t] + r*16 + half*8`, a 16-half stride =
//!      8 banks apart on a 32-bank LDS, ~4-way) down to ~2-way, and passed the
//!      padded `ld` to `load_matrix_sync`. Result: 1782 -> 1796 ns per k-step,
//!      i.e. NO CHANGE within noise. Not bank conflicts.
//!
//!   2. PIPELINE DEPTH. Went from a 2-stage to a 4-stage LDS ring so the
//!      prefetch sits DEPTH-1 slices ahead instead of one. Result: 1818 ns,
//!      slightly WORSE (more LDS = lower occupancy). Not depth-limited.
//!
//!   3. WMMA EMULATION. Disassembled the JIT hsaco for this kernel
//!      (`llvm-objdump -d --arch-name=amdgcn` on
//!      `~/.cache/grim/.../grim_grim_wmma_fused_dequant_q4k_gfx1200_*.hsaco`):
//!      `v_wmma_f32_16x16x16_f16` is emitted 60 times, so rocWMMA is using the
//!      native RDNA4 matrix path, not a VALU fallback. Not emulation.
//!
//! THE MECHANISM THAT SURVIVES is in the dequant itself, on both axes at once:
//!
//!   * BYTE GRANULARITY. `grim_deq_q4k` reads the block one byte at a time —
//!     `scales[s]`, `scales[s+4]`, `qs[qsb]` — about 10 scalar 1-byte loads per
//!     8-weight call, 128 calls per k-step, with each lane on a DIFFERENT
//!     column's block, so a warp's load becomes ~32 separate transactions.
//!   * 16x REDUNDANCY. `kk / blk` is constant for k0 in [0,240], so the same
//!     144-byte block is re-read on each of the 16 k-steps inside its
//!     256-weight span.
//!
//! ALSO TESTED AND REVERTED (M-DEPENDENT, 2026-10-08): vectorizing the Q4K
//! dequant read — the 8 weights of one call are eight CONTIGUOUS qs bytes
//! sharing one 6-bit scale pair (qsb = 32*(w/64) + (w%64), a multiple of 8), so
//! two 4-byte loads replace ~10 scalar 1-byte loads. CORRECT: parity max_diff
//! dropped to exactly 0 for Q4K at m=1/3/5/17 (the Q8_0 and fp16 gates, which do
//! not use this dequant, were unchanged at 0.19168091 / 0.27461243). FASTER at
//! M<=251: M=256/n=1024 1.392 -> 0.909 ms (1.53x, 2.36 TFLOP/s); model prefill
//! M=251 4767 -> 3354 ms. **SLOWER at M=4056: model prefill 39451 -> 62577 ms
//! (reproduced twice)**. Reverted: the user's real workload is the 4K prompt.
//!
//! The M-dependence is REAL and now MEASURED on the real model, in one healthy
//! host state (decode steady-state 20.13-20.28 ms/token in the same window):
//!
//!   prompt tokens   byte-wise   vectorized
//!         5            480 ms     351 ms    vectorized 1.37x faster
//!       251           4767        3354      1.42x faster
//!      4056          39451       61496      1.56x SLOWER
//!
//! vgpr fell 88 -> 72 (better occupancy) and the staged-in-LDS variant
//! reproduced the same 62 s, so neither occupancy nor staging explains the
//! inversion. The byte-wise dequant ships: it wins at 4K prompts, which are the
//! workload this work was measured on, and the small-M gap is 129 ms where the
//! large-M gap is 22 s. A per-M dispatch needs the crossover localised (somewhere
//! in 251..4056) and its mechanism understood first — a threshold without that
//! would be an unverified heuristic.
//!
//! ---------------------------------------------------------------------------
//! FIX DESIGN (not implemented): stage the PACKED block bytes into LDS once per
//! 256-weight span with a coalesced copy, then extract per k-step from LDS. For
//! a 64-column tile that is 64 cols x (128 qs + 12 scales) B = 8.9 KB per stage;
//! two stages 17.9 KB + c_out 4 KB = 22 KB, i.e. ~2 blocks/CU against ~5 today —
//! so occupancy has to be re-checked. This is the Marlin/Petit offline-shuffle
//! idea applied in-kernel, and it subsumes the re-scoped Step 2: the shuffle
//! exists to make that staging copy contiguous and 16-byte vectorized.
//!
//! ---------------------------------------------------------------------------
//! RUN: GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test wmma_q4k_shape_bench -- --ignored --nocapture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage,
};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("RocmDevice::try_new"))
}

/// Time one shape. Returns (median ms, weight GB moved, GFLOP).
fn time_shape(dev: &RocmDevice, m: usize, n: usize, k: usize) -> (f64, f64, f64) {
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();
    let b_host: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0)
        .collect();
    let b_packed = grim_quant::quant_q4k(&b_host).expect("pack q4k");

    let a_dev = CoreTensorOps::from_cpu(dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A");
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        dev,
        &b_packed,
        &Shape::new(vec![b_packed.len()]),
        q_dtype,
    )
    .expect("upload packed B");
    let out = CoreTensorOps::zeros(dev, &Shape::new(vec![m, n]), DType::F32).expect("alloc out");

    let a = grim_backend_rocm::as_rocm(a_dev.as_ref()).expect("a rocm");
    let b = grim_backend_rocm::as_rocm(b_dev.as_ref()).expect("b rocm");
    let o = grim_backend_rocm::as_rocm(out.as_ref()).expect("out rocm");

    // Warmup: JIT + first-touch of the LDS/scratch paths.
    for _ in 0..3 {
        dev.launch_wmma_fused_dequant_q4k_for_ab(a, b, o, m, n, k)
            .expect("wmma launch");
    }
    dev.synchronize();

    let mut samples = Vec::new();
    for _ in 0..7 {
        let t = std::time::Instant::now();
        dev.launch_wmma_fused_dequant_q4k_for_ab(a, b, o, m, n, k)
            .expect("wmma launch");
        dev.synchronize();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|x, y| x.total_cmp(y));
    let ms = samples[samples.len() / 2];

    // Weight traffic: every M-tile pass re-reads the whole weight.
    let passes = m.div_ceil(16) as f64;
    let weight_gb = (n * k) as f64 * (4.5 / 8.0) * passes / 1e9;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    (ms, weight_gb, gflop)
}

#[test]
#[ignore]
fn wmma_q4k_shape_sweep() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1 + GPU");
        return;
    };
    let n = 1024usize;
    let m_fixed = 64usize;
    let k_fixed = 4096usize;

    eprintln!("\n[K sweep] n={n} m={m_fixed} (4 M-tiles)");
    eprintln!("{:>6} {:>9} {:>8} {:>9} {:>14}", "K", "ms", "GFLOP/s", "GB/s", "ns/k-iter");
    let mut k_pts = Vec::new();
    for k in [512usize, 1024, 2048, 4096] {
        let (ms, gb, gflop) = time_shape(&dev, m_fixed, n, k);
        let iters = (k / 16) as f64;
        eprintln!(
            "{k:>6} {ms:>9.3} {:>8.1} {:>9.1} {:>14.1}",
            gflop / (ms / 1e3) * 1e3,
            gb / (ms / 1e3),
            ms * 1e6 / iters
        );
        k_pts.push((k as f64, ms));
    }

    eprintln!("\n[M sweep] n={n} k={k_fixed}");
    eprintln!("{:>6} {:>7} {:>9} {:>8} {:>9} {:>14}", "M", "tiles", "ms", "GFLOP/s", "GB/s", "ms/M-tile");
    let mut m_pts = Vec::new();
    for m in [16usize, 32, 64, 128, 256] {
        let (ms, gb, gflop) = time_shape(&dev, m, n, k_fixed);
        let tiles = m.div_ceil(16) as f64;
        eprintln!(
            "{m:>6} {tiles:>7.0} {ms:>9.3} {:>8.1} {:>9.1} {:>14.3}",
            gflop / (ms / 1e3) * 1e3,
            gb / (ms / 1e3),
            ms / tiles
        );
        m_pts.push((tiles, ms));
    }

    // Fit both axes through the origin and report which term carries the cost.
    let slope = |pts: &[(f64, f64)]| -> f64 {
        let num: f64 = pts.iter().map(|(x, y)| x * y).sum();
        let den: f64 = pts.iter().map(|(x, _)| x * x).sum();
        num / den
    };
    let per_k = slope(&k_pts.iter().map(|(k, ms)| (k / 16.0, *ms)).collect::<Vec<_>>());
    let per_m = slope(&m_pts);
    eprintln!(
        "\n[fit] {:.4} ms per k-iteration  |  {:.4} ms per M-tile",
        per_k, per_m
    );
    eprintln!(
        "[fit] at n={n} k={k_fixed} m=256: k-term {:.2} ms + M-term {:.2} ms = {:.2} ms",
        per_k * (k_fixed as f64 / 16.0),
        per_m * 16.0,
        per_k * (k_fixed as f64 / 16.0) + per_m * 16.0
    );

    // ---- Findings pinned as assertions, so a change in the scaling regime is
    // ---- caught rather than silently absorbed.
    //
    // (1) The M re-reads are L2-served: 16 M-tiles must NOT cost 16x one M-tile.
    //     Measured 3.4x. A regression here means the weight slice stopped being
    //     reused, which is the difference between a 0.09 and a 1.4 ms/M-tile
    //     kernel (the tiled fallback's failure mode).
    let one_tile = m_pts[0].1;
    let sixteen = m_pts[m_pts.len() - 1].1;
    assert!(
        sixteen < 6.0 * one_tile,
        "M-tile scaling regressed: 16 tiles cost {sixteen:.3} ms vs 1 tile {one_tile:.3} ms \
         (ratio {:.2}); the weight re-reads are no longer being reused",
        sixteen / one_tile
    );
    // (2) Efficiency must RISE with batch (the AMD low-latency-GEMM regime: small
    //     M under-utilises). If this inverts, the kernel became compute-bound at
    //     large M and the small-M specialisation is no longer the win.
    let gflops = |ms: f64, m: usize| 2.0 * m as f64 * n as f64 * k_fixed as f64 / (ms / 1e3) / 1e9;
    let eff_small = gflops(m_pts[0].1, 16);
    let eff_large = gflops(sixteen, 256);
    assert!(
        eff_large > eff_small,
        "efficiency no longer rises with batch: M=16 {eff_small:.0} vs M=256 {eff_large:.0} GFLOP/s"
    );
    // (3) The per-k-iteration cost is the dominant term and is ~2 us — 38x off
    //     the ALU bound. Pin the order of magnitude so that a Step-3 pipelining
    //     change has a number to move, and a regression is visible.
    eprintln!(
        "[guard] per-k-iteration {per_k:.4} ms, 16-tile/1-tile ratio {:.2}, \
         eff {eff_small:.0} -> {eff_large:.0} GFLOP/s",
        sixteen / one_tile
    );
    assert!(
        per_k > 0.0005,
        "per-k-iteration cost fell to {per_k:.6} ms — if Step 3 landed, retune this guard"
    );
}

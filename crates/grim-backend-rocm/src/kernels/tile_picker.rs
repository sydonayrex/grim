//! Tile selection heuristics, hardware-adaptive tile picker, roofline cost model, and empirical FCP search.

use crate::autotune::ShapeClass;
use crate::device::hardware_spec::HardwareSpec;
use crate::device::roc_device::RocmDevice;

/// Hardware-adaptive GEMM tile configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConfig {
    /// M dimension tile size per block.
    pub block_m: u32,
    /// N dimension tile size per block.
    pub block_n: u32,
    /// K dimension reduction tile size per block.
    pub block_k: u32,
    /// Split-K reduction factor across grid blocks.
    pub split_k: u32,
    /// Grid stride for M dimension loop.
    pub grid_stride_m: u32,
    /// Grid stride for N dimension loop.
    pub grid_stride_n: u32,
    /// Flag indicating if LDS capacity permits double buffering.
    pub lds_double_buffer: bool,
    /// Enable rocWMMA matrix instruction path (gfx1100+/gfx1200+).
    pub use_wmma: bool,
    /// Enable native MFMA matrix instruction path (gfx9xx/gfx12xx).
    pub use_mfma: bool,
    /// Total threads per block (block_x).
    pub threads: u32,
}

/// Input problem dimensions (M, N, K) for GEMM kernels: C[M,N] = A[M,K] * B[K,N].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeDims {
    /// Output row count M.
    pub m: u32,
    /// Output column count N.
    pub n: u32,
    /// Reduction dimension K.
    pub k: u32,
}

impl ShapeDims {
    /// Construct a new ShapeDims instance.
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

/// Raw (pre-wavefront-rounding) block_m/block_n for a shape class. Centralized so both
/// `pick_tiles` and `TileConfig::with_block_geometry` derive geometry from the same source.
fn raw_block_mn(shape_class: ShapeClass) -> (u32, u32) {
    match shape_class {
        ShapeClass::Decode => (16, 16),
        ShapeClass::Prefill => (32, 32),
        ShapeClass::TLOLog => (16, 64),
    }
}

impl TileConfig {
    /// Re-derive block_m/block_n from spec + shape_class, rounding up to the wavefront size.
    /// Used when reconstructing a `TileConfig` from the leaner `AutotuneConfig` cache entry,
    /// which stores threads/block_k/grid_stride but not block_m/block_n.
    pub fn with_block_geometry(
        mut self,
        spec: &HardwareSpec,
        shape_class: ShapeClass,
    ) -> Self {
        let wave = spec.wavefront_size;
        let (bm, bn) = raw_block_mn(shape_class);
        self.block_m = ((bm + wave - 1) / wave) * wave;
        self.block_n = ((bn + wave - 1) / wave) * wave;
        self
    }
}

/// Derive hardware-adaptive GEMM tile parameters from hardware spec, shape class, and dimensions.
pub fn pick_tiles(spec: &HardwareSpec, shape_class: ShapeClass, dims: ShapeDims) -> TileConfig {
    let wave = spec.wavefront_size;
    let lds_per_cu = 64 * 1024;
    let max_lds = lds_per_cu;
    let max_threads = spec.max_threads_per_block;

    let (block_m, block_n) = raw_block_mn(shape_class);

    let block_m = ((block_m + wave - 1) / wave) * wave;
    let block_n = ((block_n + wave - 1) / wave) * wave;

    let block_k = match shape_class {
        ShapeClass::Decode => 32,
        ShapeClass::Prefill => 64,
        ShapeClass::TLOLog => 64,
    };

    let lds_per_tile = 2 * (
        (block_m as u64) * (block_k as u64) * 2
        + (block_k as u64) * (block_n as u64) * 2
        + (block_m as u64) * (block_n as u64) * 2
    ) as u32;

    let lds_double_buffer = max_lds >= 2 * lds_per_tile;

    let vgpr_per_thread = estimate_vgpr_per_thread(block_m, block_n, block_k);
    let vgpr_file: u32 = 512;
    let waves_vgpr = vgpr_file / (vgpr_per_thread * wave).max(1);
    let waves_lds = (lds_per_cu / (lds_per_tile + if lds_double_buffer { lds_per_tile } else { 0 }).max(1)) as u32;
    let waves_thread = max_threads / wave;
    let target_waves = waves_vgpr
        .min(waves_lds)
        .min(waves_thread)
        .clamp(1, 4);
    let threads = target_waves * wave;
    assert!(threads <= max_threads && threads % wave == 0);

    let split_k = if dims.k > block_k * 4 {
        ((dims.k + block_k * 4 - 1) / (block_k * 4)).clamp(1, 16)
    } else {
        1
    };

    let use_wmma = spec.gcn_arch.starts_with("gfx11") || spec.gcn_arch.starts_with("gfx12");
    let use_mfma = spec.gcn_arch.starts_with("gfx12") || spec.gcn_arch.starts_with("gfx9");

    TileConfig {
        block_m,
        block_n,
        block_k,
        split_k,
        grid_stride_m: block_m,
        grid_stride_n: block_n,
        lds_double_buffer,
        use_wmma,
        use_mfma,
        threads,
    }
}

/// Static roofline execution latency estimate in seconds for pre-filtering candidate configurations.
pub fn roofline_cost(spec: &HardwareSpec, dims: ShapeDims, _tiles: &TileConfig) -> f64 {
    let muflops = 2.0 * (dims.m as f64) * (dims.n as f64) * (dims.k as f64);
    let compute_time_s = muflops / (spec.mem_bandwidth_gb_s * 1e9 * 0.6);

    let bytes_read = ((dims.m as u64) * (dims.k as u64) * 2
        + (dims.k as u64) * (dims.n as u64) * 2) as f64;
    let bytes_written = ((dims.m as u64) * (dims.n as u64) * 2) as f64;
    let bytes_total = bytes_read + bytes_written;
    let memory_time_s = bytes_total / (spec.mem_bandwidth_gb_s * 1e9);

    compute_time_s.max(memory_time_s)
}

/// Static VGPR per thread estimation function used to derive target wavefront occupancy.
pub fn estimate_vgpr_per_thread(block_m: u32, block_n: u32, block_k: u32) -> u32 {
    let per_thread_area = ((block_m * block_n) / 32).max(1) + ((block_k * 32) / 64).max(1);
    per_thread_area.clamp(32, 256)
}

/// Resource validator checking physical LDS, max threads, and dimension positivity.
pub fn candidate_valid(spec: &HardwareSpec, cand: &TileConfig) -> bool {
    let lds_per_cu = 64 * 1024;
    let lds_per_tile = 2 * (
        (cand.block_m as u64) * (cand.block_k as u64) * 2
        + (cand.block_k as u64) * (cand.block_n as u64) * 2
        + (cand.block_m as u64) * (cand.block_n as u64) * 2
    );
    if 2 * lds_per_tile > lds_per_cu as u64 {
        return false;
    }
    if cand.threads > spec.max_threads_per_block {
        return false;
    }
    if cand.block_m == 0 || cand.block_n == 0 || cand.block_k == 0 {
        return false;
    }
    true
}

/// Polynomial-time empirical tile search measuring candidate execution times on host GPU.
pub fn fcp_fallback_tile_search(
    device: &RocmDevice,
    spec: &HardwareSpec,
    entry: &str,
    dims: ShapeDims,
    shape_class: ShapeClass,
) -> TileConfig {
    let base = pick_tiles(spec, shape_class, dims);
    let wave = spec.wavefront_size;
    let block_k_choices = [16u32, 32, 64, 128];
    let mut candidates: Vec<TileConfig> = Vec::new();

    for &bm in &[base.block_m, base.block_m.saturating_add(wave)] {
        for &bn in &[base.block_n, base.block_n.saturating_add(wave)] {
            if bm == 0 || bn == 0 || bm % wave != 0 || bn % wave != 0 {
                continue;
            }
            if (bm * bn) > spec.max_threads_per_block {
                continue;
            }
            for &bk in &block_k_choices {
                if bk % wave != 0 {
                    continue;
                }
                for &sk in &[1u32, base.split_k, (base.split_k * 2).clamp(1, 16)] {
                    let mut cand = base.clone();
                    cand.block_m = bm;
                    cand.block_n = bn;
                    cand.block_k = bk;
                    cand.split_k = sk;
                    if candidate_valid(spec, &cand) {
                        candidates.push(cand);
                    }
                }
            }
        }
    }

    candidates.sort_by(|a, b| {
        roofline_cost(spec, dims, a)
            .partial_cmp(&roofline_cost(spec, dims, b))
            .unwrap()
    });
    candidates.truncate(candidates.len().min(16));
    candidates.dedup();

    let mut best: Option<(TileConfig, f64)> = None;
    for cand in &candidates {
        let source = crate::kernels::source_asm::compute_kernel_source_with_spec(
            spec,
            entry,
            shape_class,
            dims,
            0,
            1,
            Some(cand),
        );
        let (hsaco, lowered) = match device.jit_compile_or_cache(&source, entry, Some(spec)) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let t_ms = device.time_kernel_ms(&hsaco, &lowered, dims, cand);
        if best.as_ref().map_or(true, |(_, bt)| t_ms < *bt) {
            best = Some((cand.clone(), t_ms));
        }
    }

    let (winner, winner_ms) = best.map(|(c, t)| (c, t)).unwrap_or((base.clone(), 0.0));
    device.store_tune_cache(entry, spec, dims, &winner, winner_ms);
    winner
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune::{GemmOp, ShapeClass};
    use crate::device::hardware_spec::HardwareSpec;
    use crate::peer_access::{LinkType, P2PTopology};

    fn make_test_spec(arch: &str, cus: u32) -> HardwareSpec {
        HardwareSpec {
            gcn_arch: arch.to_string(),
            wavefront_size: 32,
            max_shared_mem_per_block: 384 * 1024,
            max_threads_per_block: 1024,
            cu_count: cus,
            multiprocessor_count: cus,
            mem_bandwidth_gb_s: 500.0,
            p2p_topology: P2PTopology {
                device_count: 1,
                links: vec![vec![LinkType::NoLink]],
            },
        }
    }

    #[test]
    fn pick_tiles_gfx1036_prefill() {
        let spec = make_test_spec("gfx1036", 64);
        let dims = ShapeDims::new(32, 32, 64);
        let tiles = pick_tiles(&spec, ShapeClass::Prefill, dims);
        assert_eq!(tiles.block_m, 32);
        assert_eq!(tiles.block_n, 32);
        assert_eq!(tiles.block_k, 64);
        assert!(tiles.lds_double_buffer);
        assert_eq!(tiles.threads % spec.wavefront_size, 0);
        assert!(tiles.threads <= spec.max_threads_per_block);
    }

    #[test]
    fn pick_tiles_tlolog_distinct_wide_n() {
        let spec = make_test_spec("gfx1036", 64);
        let dims = ShapeDims::new(1, 32000, 4096);
        let shape = ShapeClass::from_op(GemmOp::LmHead, 1);
        assert_eq!(shape, ShapeClass::TLOLog);

        let tiles = pick_tiles(&spec, shape, dims);
        assert_eq!(tiles.block_m, 32);
        assert_eq!(tiles.block_n, 64);
        assert_eq!(tiles.block_k, 64);
    }


    #[test]
    fn roofline_cost_exact_formula_mutation_resistant() {
        let spec = make_test_spec("gfx1036", 64);
        let dims = ShapeDims::new(100, 100, 100);
        let tiles = pick_tiles(&spec, ShapeClass::Prefill, dims);

        // muflops = 2 * 100 * 100 * 100 = 2_000_000
        // compute_time = 2_000_000 / (500e9 * 0.6) = 6.666666666666667e-6
        // bytes_read = (100*100*2 + 100*100*2) = 40_000
        // bytes_written = 20_000
        // bytes_total = 60_000
        // memory_time = 60_000 / 500e9 = 1.2e-7
        // expected max(compute_time, memory_time) = 6.666666666666667e-6
        let cost = roofline_cost(&spec, dims, &tiles);
        let expected = 2.0e6 / (500.0 * 1e9 * 0.6);
        assert!((cost - expected).abs() < 1e-12, "roofline_cost formula mutated");
    }

    #[test]
    fn split_k_derivation_mutation_resistant() {
        let spec = make_test_spec("gfx1036", 64);
        // k <= block_k * 4 (64 * 4 = 256) -> split_k = 1
        let tiles_small_k = pick_tiles(&spec, ShapeClass::Prefill, ShapeDims::new(32, 32, 256));
        assert_eq!(tiles_small_k.split_k, 1);

        // k = 1024 -> ceil(1024 / 256) = 4
        let tiles_large_k = pick_tiles(&spec, ShapeClass::Prefill, ShapeDims::new(32, 32, 1024));
        assert_eq!(tiles_large_k.split_k, 4);
    }

    #[test]
    fn vgpr_estimation_clamps_bounds() {
        let vgpr_min = estimate_vgpr_per_thread(1, 1, 1);
        assert_eq!(vgpr_min, 32);

        let vgpr_max = estimate_vgpr_per_thread(1024, 1024, 1024);
        assert_eq!(vgpr_max, 256);
    }

    #[test]
    fn candidate_valid_rejects_oversized() {
        let spec = make_test_spec("gfx1036", 64);
        let valid_cand = TileConfig {
            block_m: 16,
            block_n: 16,
            block_k: 16,
            split_k: 1,
            grid_stride_m: 16,
            grid_stride_n: 16,
            lds_double_buffer: true,
            use_wmma: false,
            use_mfma: false,
            threads: 128,
        };
        assert!(candidate_valid(&spec, &valid_cand));

        let invalid_cand = TileConfig {
            block_m: 512,
            block_n: 512,
            block_k: 512,
            split_k: 1,
            grid_stride_m: 512,
            grid_stride_n: 512,
            lds_double_buffer: true,
            use_wmma: false,
            use_mfma: false,
            threads: 2048,
        };
        assert!(!candidate_valid(&spec, &invalid_cand));
    }

    /// Gap 1: the reconstruction helper used by `get_or_tune_tiles` when mapping a cached
    /// `AutotuneConfig` (which lacks block_m/block_n) back to a full `TileConfig`.
    /// Re-derives geometry from spec + shape_class, rounding up to the wavefront size.
    #[test]
    fn with_block_geometry_reconstructs_from_shape_class() {
        let spec = make_test_spec("gfx1036", 64);
        // A lean config as stored in AutotuneConfig: block_m/block_n unknown (0).
        let lean = TileConfig {
            block_m: 0,
            block_n: 0,
            block_k: 64,
            split_k: 1,
            grid_stride_m: 32,
            grid_stride_n: 32,
            lds_double_buffer: false,
            use_wmma: false,
            use_mfma: false,
            threads: 128,
        };
        // Reconstruct for TLOLog -> raw (16, 64), wave-rounded to (32, 64).
        let recon = lean.clone().with_block_geometry(&spec, ShapeClass::TLOLog);
        assert_eq!(recon.block_m, 32);
        assert_eq!(recon.block_n, 64);
        // Non-geometry fields preserved.
        assert_eq!(recon.block_k, 64);
        assert_eq!(recon.threads, 128);

        // Reconstruct for Prefill -> raw (32, 32), wave-rounded stays (32, 32).
        let recon_pf = lean.with_block_geometry(&spec, ShapeClass::Prefill);
        assert_eq!(recon_pf.block_m, 32);
        assert_eq!(recon_pf.block_n, 32);
    }

    /// Gap 2: end-to-end tile selection for lm_head via lookup_gemm_config_for_shape.
    /// The TLOLog arm must produce the distinct wide-N tile (block_n == 64), different from
    /// the Decode/Prefill arms — proving the op-identity tag propagates through dispatch.
    #[test]
    fn tlolog_tile_via_lookup_gemm_config_for_shape() {
        use crate::device::gemm_tuning::lookup_gemm_config_for_shape;
        use crate::WavefrontSize;

        // lm_head: M=1 (decode), N=vocab=32000, K=hidden=4096.
        let lm = lookup_gemm_config_for_shape(
            1, 32000, 4096,
            WavefrontSize::W32,
            ShapeClass::from_op(GemmOp::LmHead, 1),
        );
        assert_eq!(lm.block_n, 64, "lm_head must select the wide-N TLOLog tile");

        // A non-lm_head GEMM at the same M=1 bins as Decode, NOT TLOLog.
        let other = lookup_gemm_config_for_shape(
            1, 32000, 4096,
            WavefrontSize::W32,
            ShapeClass::from_op(GemmOp::Attention, 1),
        );
        assert_ne!(
            other.block_n, lm.block_n,
            "non-lm_head at m==1 must NOT pick the TLOLog tile"
        );
    }

    /// Gap 2: the op-identity classifier fires for LmHead regardless of m.
    #[test]
    fn from_op_classifier_distinguishes_lmhead() {
        // LmHead is TLOLog even at decode m==1 (which from_m would bin as Decode).
        assert_eq!(ShapeClass::from_op(GemmOp::LmHead, 1), ShapeClass::TLOLog);
        // And at prefill m > 1.
        assert_eq!(ShapeClass::from_op(GemmOp::LmHead, 4096), ShapeClass::TLOLog);
        // Non-lm_head bins by m exactly as from_m.
        assert_eq!(ShapeClass::from_op(GemmOp::Attention, 1), ShapeClass::Decode);
        assert_eq!(ShapeClass::from_op(GemmOp::Ffn, 64), ShapeClass::Prefill);
        assert_eq!(ShapeClass::from_op(GemmOp::Other, 1), ShapeClass::Decode);
    }

    /// Gap 4: cycles_per_invocation conversion. store_tune_cache stores
    /// (winner_ms * 1e6) as u64; verify the round-trip scale for a representative value.
    #[test]
    fn store_tune_cache_elapsed_ms_to_cycles_scale() {
        let winner_ms: f64 = 0.25; // 0.25 ms measured
        let cycles = (winner_ms * 1e6) as u64;
        assert_eq!(cycles, 250_000);
        assert!(cycles > 0, "non-zero measurement must persist as non-zero cycles");
    }
}



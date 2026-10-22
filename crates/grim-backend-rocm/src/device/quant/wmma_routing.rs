//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! WMMA capability gating, route decision helpers, shared tiled launchers,
//! and the `GpuDequant` impl. Owns `wmma_route_tests`.

use std::ffi::c_void;

use grim_tensor::BackendStorage;
use grim_tensor::dtype::{ArithType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, arg};

impl RocmDevice {
    /// Returns `true` when the activation storage holds native FP16 data
    /// (`ArithType::F16` + `DTypeStorage::Native`).  The FP16-input WMMA
    /// kernels read such activations directly, halving A-read bandwidth.
    pub(crate) fn is_fp16_activation(storage: &RocmStorage) -> bool {
        storage.dtype().arith == ArithType::F16
            && matches!(storage.dtype().storage, DTypeStorage::Native)
    }

    /// SPEED-ROC: dispatch helper for IQ-family formats.
    /// Uses WMMA kernel on RDNA3/4 for small M (decode), falls back to scalar otherwise.
    pub(crate) fn launch_iq_wmma_fallback(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        wmma_method: fn(
            &Self,
            &RocmStorage,
            &RocmStorage,
            &RocmStorage,
            usize,
            usize,
            usize,
        ) -> Result<*mut c_void>,
        // Set false for formats whose fused WMMA kernel has no bit-exact
        // parity gate against the CPU reference (IQ3_S measured rel 0.76 on
        // real Xing4.0 weights vs 2.6e-6 for the scalar kernel).
        allow_wmma: bool,
        scalar_method: fn(
            &Self,
            &RocmStorage,
            &RocmStorage,
            &RocmStorage,
            usize,
            usize,
            usize,
        ) -> Result<*mut c_void>,
    ) -> Result<()> {
        let is_rdna34 = self.is_rdna34;
        // GRIM_WMMA=0 forces the scalar dequant-GEMM for every format. The
        // WMMA path is an optimization: when it disagrees with the CPU
        // reference on real weights, correctness wins over the fused kernel.
        let wmma_off = matches!(
            std::env::var("GRIM_WMMA").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        );
        // Small-M threshold, env-tunable, default 0 = never prefer WMMA.
        //
        // It used to be hardcoded `m <= 4`, which routed exactly the SHORT
        // PROMPT through WMMA — the case it is worst at. Measured on the 27B
        // (4-token prompt, single device): the WMMA fused-dequant kernels cost
        // 19.6 s of the run's 21.6 s of kernel time at ~40 ms PER TENSOR
        // (~2.5 GFLOP/s), while the group-hoisted scalar kernels run the same
        // shapes at ~0.5 ms. The scalar path is also the one with the bit-exact
        // parity gate (the WMMA IQ3_S kernel measured rel 0.76 on real
        // Xing4.0 weights), so preferring it at small m is both faster and
        // safer. `GRIM_WMMA_M_MAX=<n>` restores a WMMA preference for m <= n.
        let wmma_m_max: usize = std::env::var("GRIM_WMMA_M_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if allow_wmma && is_rdna34 && m <= wmma_m_max && !wmma_off {
            wmma_method(self, a, b, out, m, n, k)?;
        } else {
            scalar_method(self, a, b, out, m, n, k)?;
        }
        Ok(())
    }

    /// M7.2a: the SHAPE gate for the IQ recipe (the LDS-tiled prompt GEMM)
    /// on a given format.
    ///
    /// The tiled kernel stages a 64x64 weight tile in LDS and reuses it
    /// across TILE_M=4 prompt rows, so it pays off when the prompt has
    /// enough rows to amortise the staging and the tile is expressible:
    /// `n` must be a multiple of the 64-column tile and `k` a multiple of the
    /// format's 256-element super-block.
    ///
    /// `m >= 2` is the floor: at m == 1 there is no row reuse to amortise,
    /// and the decode shape has its own (faster) route.
    ///
    /// The per-format policy switch is `tiled_quant_enabled`, so every
    /// format the recipe reaches can be turned off individually for an A/B.
    pub(crate) fn tiled_quant_prompt_ok(tag: &str, m: usize, n: usize, k: usize) -> bool {
        Self::tiled_quant_enabled(tag) && m >= 2 && n >= 64 && n % 64 == 0 && k % 256 == 0
    }

    /// SPEED-ROC-8: per-format opt-in flag (GRIM_Q5K_TILED, GRIM_IQ4XS_TILED,
    /// GRIM_Q8_0_TILED, ...). Cached per format tag; default off — the scalar
    /// kernels remain the shipping default until per-model benchmarks flip it.
    pub(crate) fn tiled_quant_enabled(tag: &str) -> bool {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = match cache.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        *guard.entry(tag.to_string()).or_insert_with(|| {
            let var = format!("GRIM_{}_TILED", tag.to_uppercase());
            if let Ok(v) = std::env::var(&var) {
                return matches!(v.as_str(), "1" | "true" | "on");
            }
            // 9B perf plan Step 3: default-ON for the measured k-quant
            // formats. The scalar kernel re-reads every weight block M times;
            // at the 9B prefill shape it ran at 0.2 GB/s (skinny_m gate).
            //
            // M7.2a/f: q2k/q3k are NOT default-ON, and the reason is
            // measured, not a missing measurement.
            //
            // The plain LDS-tiled recipe stages a 64(k) x 64(COLUMN) tile,
            // so the 4096 weights it stages come from 64 DIFFERENT weight
            // rows and each needs its own super-block header decoded: there
            // is no header reuse to amortise, and Q3_K's header is a 12-byte
            // aux shuffle into 16 scale bytes. Measured at the 27B's
            // dominant shape (n=17408, k=5120): the plain tile ran 99.4 ms
            // where the incumbent ran 7.8 ms — a 12.7x REGRESSION.
            //
            // M7.2f applied the IQ recipe's group hoist to that tile
            // (GRIM_TILED_QUANT_FWD_BWD_GROUPED: the header is decoded once
            // per 16-weight group, and the four codes of a group come out
            // of one wide read). It is bit-identical to the scalar kernel,
            // forward and backward, and it makes the TILE 6.5x faster
            // (Q3_K 99.4 -> 15.3 ms, Q2_K 99.7 -> 11.6 ms) — so the recipe
            // DOES transfer to the tiled route.
            //
            // But the tile is still 1.6-2.0x SLOWER than the incumbent at
            // every M (Q3_K: 15.3 vs 7.6 ms at m=64, 59.7 vs 27.6 at m=256,
            // 233 vs 111 at m=1024), so the route stays off. What the tile
            // spends its time on is still not identified: with the header
            // stubbed out it is 10.7 ms, against the plain macro's ~0.6 ms
            // of staging at the same shape, and neither the thread mapping
            // nor force-inlining the quad moves it.
            //
            // The incumbent for these two formats is the SCALAR dequant
            // GEMM, and it is already at the L2 limit (38 MB of weights read
            // M=64 times over is 2.4 GB in 7.6 ms = 320 GB/s). The headroom
            // is the tile's 4x-lower traffic; the blocker is the tile's
            // staging cost, which is the next thing to measure.
            matches!(tag, "q4k" | "q5k" | "q6k")
        })
    }

    /// K divisibility required by the tiled loops: the 64-wide tile stride must
    /// land on whole super-blocks for every row (256-element formats trivially
    /// satisfy it; Q8_0's 32-element blocks require K % 64 == 0).
    pub(crate) fn tiled_k_multiple(blk_elems: u32) -> usize {
        if blk_elems == 32 {
            64
        } else {
            blk_elems as usize
        }
    }

    /// SPEED-ROC-8: tiled-GEMM dispatch table for the macro-stamped kernels in
    /// kernels::quant_tiled_gemm — (scalar forward entry, format tag,
    /// super-block elements, bytes per block). The tiled entry names are
    /// derived from the tag: `grim_fused_dequant_gemm_{tag}_tiled` and
    /// `grim_fused_dequant_gemm_{tag}_backward_tiled`.
    /// Sizes are the llama.cpp block layouts in
    /// `old/repo/llama.cpp-master/ggml/src/ggml-common.h` with `QK_K = 256`,
    /// `QK4_NL = 32` and `IQ3S_N_SCALE = QK_K/64 = 4`. Pinned by
    /// `tiled_quant_block_sizes_match_llama_cpp`.
    const TILED_QUANT_TABLE: &[(&str, &str, u32, u32)] = &[
        ("grim_fused_dequant_gemm_q5k", "q5k", 256, 176),
        ("grim_fused_dequant_gemm_q6k", "q6k", 256, 210),
        ("grim_fused_dequant_gemm_iq2xxs", "iq2xxs", 256, 66),
        ("grim_fused_dequant_gemm_iq2xs", "iq2xs", 256, 74),
        ("grim_fused_dequant_gemm_iq2s", "iq2s", 256, 82),
        ("grim_fused_dequant_gemm_iq3xxs", "iq3xxs", 256, 98),
        ("grim_fused_dequant_gemm_iq3s", "iq3s", 256, 110),
        // iq4_nl is DELIBERATELY ABSENT. Its llama.cpp block is QK4_NL = 32
        // elements in 18 bytes (ggml-common.h `#define QK4_NL 32`, and
        // `crates/grim-tensor/src/dtype.rs:324` agrees), which cannot fill a
        // 256-element WMMA tile. Listing it at (256, 170) imposed
        // `k % tiled_k_multiple(256) == 0` on a 32-element-block format and
        // named a block size matching no llama.cpp layout — Q4_K is 256/144.
        // Absence routes it to the non-tiled `grim_fused_dequant_gemm_iq4nl`,
        // which is the correct general path.
        // 136 = d(2) + scales_h(2) + scales_l[4] + qs[128]. The 134 here
        // omitted the `sizeof(uint16_t) scales_h` term and matched no
        // llama.cpp layout.
        ("grim_fused_dequant_gemm_iq4xs", "iq4xs", 256, 136),
        ("grim_fused_dequant_gemm_q8_0", "q8_0", 32, 34),
    ];

    pub(crate) fn tiled_quant_lookup(entry_or_tag: &str) -> Option<(&'static str, u32)> {
        Self::TILED_QUANT_TABLE
            .iter()
            .find(|(entry, tag, _, _)| *entry == entry_or_tag || *tag == entry_or_tag)
            .map(|(_, tag, blk, _)| (*tag, *blk))
    }

    /// Generic tiled forward launcher — grid (ceil(N/64), ceil(M/4)), block (64, 4, 1).
    pub(crate) fn launch_fused_deq_gemm_tiled(
        &self,
        kernel: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // SMALL-M (decode AND prefill) goes to the u4-lane GEMV first: the
        // tiled kernel measures 7 ms/launch at the 9B's shapes while the lane
        // GEMV runs at 70 us (485 GB/s vs ~10 GB/s), and the tiled kernel's
        // weight reuse factor is only TILE_M = 4, so prefill re-reads and
        // re-dequantizes the whole weight set once per 4 rows of A. Only
        // K-quant formats with a lane conversion are eligible. The M bound is
        // shared with the q4k arm — one switch, one bound, so the schemes
        // cannot drift apart.
        if m <= crate::device::quant::dequant_k_quants::lane_gemm_max_m() {
            let scheme = if kernel.contains("q4k") {
                Some(grim_tensor::dtype::KQuantScheme::Q4K)
            } else if kernel.contains("q5k") {
                Some(grim_tensor::dtype::KQuantScheme::Q5K)
            } else if kernel.contains("q6k") {
                Some(grim_tensor::dtype::KQuantScheme::Q6K)
            } else {
                None
            };
            if let Some(s) = scheme {
                if let Some(stream) =
                    self.try_u4_lane_gemm(a_storage, b_storage, out_storage, m, n, k, s)?
                {
                    return Ok(stream);
                }
            }
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: a has no device ptr", kernel)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", kernel)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", kernel)))?;
        let grid_dim = HipDim3::new(n.div_ceil(64) as u32, m.div_ceil(4) as u32, 1);
        let block_dim = HipDim3::new(64, 4, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            kernel,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Generic tiled backward launcher — grid (ceil(K/64), ceil(M/4)), block (64, 4, 1).
    pub(crate) fn launch_fused_deq_gemm_tiled_backward(
        &self,
        kernel: &str,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dY has no device ptr", kernel)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", kernel)))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dX has no device ptr", kernel)))?;
        let grid_dim = HipDim3::new(k.div_ceil(64) as u32, m.div_ceil(4) as u32, 1);
        let block_dim = HipDim3::new(64, 4, 1);
        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            kernel,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    pub(crate) fn launch_fused_deq_gemm_simple(
        &self,
        name: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        cols_per_thread: usize,
    ) -> Result<*mut c_void> {
        // SPEED-ROC-8: opt-in LDS-tiled prefill path (per-format GRIM_*_TILED).
        if let Some((tag, blk)) = Self::tiled_quant_lookup(name) {
            // m >= 2 (not 16): the tiled kernel is TILE_M=4 and covers the
            // 9B prefill shape M=5 with one partial tile; the scalar fallback
            // re-reads every weight block M times.
            if Self::tiled_quant_enabled(tag)
                && m >= 2
                && n >= 64
                && k % Self::tiled_k_multiple(blk) == 0
            {
                let tiled_kernel = format!("grim_fused_dequant_gemm_{}_tiled", tag);
                return self.launch_fused_deq_gemm_tiled(
                    &tiled_kernel,
                    a_storage,
                    b_storage,
                    out_storage,
                    m,
                    n,
                    k,
                );
            }
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: a has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        // Kernels that handle multiple columns per thread (Q4/Q5/Q6_K scalar
        // forwards do 4) cover m*ceil(n/cols) threads, not m*n.
        let cols = cols_per_thread.max(1);
        let total_elems: u64 = (m as u64)
            .checked_mul(n.div_ceil(cols) as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*n overflow", name)))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Generic backward fused dequant GEMM launcher.
    pub(crate) fn launch_fused_deq_backward_gemm_simple(
        &self,
        name: &str,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // SPEED-ROC-8: opt-in LDS-tiled backward path (same GRIM_*_TILED flags).
        // Backward entries are "grim_fused_dequant_backward_gemm_{tag}".
        if let Some(tag) = name
            .strip_prefix("grim_fused_dequant_backward_gemm_")
            .and_then(|fmt| Self::tiled_quant_lookup(fmt))
        {
            let (tag, blk) = tag;
            if Self::tiled_quant_enabled(tag) && n >= 64 && k % Self::tiled_k_multiple(blk) == 0 {
                let tiled_kernel = format!("grim_fused_dequant_gemm_{}_backward_tiled", tag);
                return self.launch_fused_deq_gemm_tiled_backward(
                    &tiled_kernel,
                    dy_storage,
                    b_storage,
                    dx_storage,
                    m,
                    n,
                    k,
                );
            }
        }
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dY has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: B has no device ptr", name)))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dX has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*k overflow", name)))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }
}

/// P1-WI-1: pure routing decision for the WMMA GEMM path. Extracted from [see: `RocmDevice::should_use_wmma_path`]
pub(crate) fn wmma_route_decision(
    ext: Option<&grim_format::spec::GrimTensorExt>,
    out_arith: ArithType,
    cfg_enabled: bool,
) -> bool {
    // No extension ⇒ no per-tensor hint ⇒ stick with the existing dispatcher
    let Some(ext) = ext else {
        return false;
    };
    if !cfg_enabled {
        return false;
    }
    match ext.layout_hint {
        grim_format::spec::LayoutHintTag::PackedQuantWmma { bits, .. } => {
            matches!(bits, 2 | 3 | 4 | 8) && out_arith == ArithType::F16
        }
        _ => false,
    }
}

/// Serializes GpuDequant dispatches per device ordinal.
///
/// The conversion pack loop runs under rayon, so many tensors dequantize
/// concurrently on one device. Driven SERIALLY the standalone dequant path is
/// bit-exact at the largest checkpoint sizes (xing_dequant_real_parity gates
/// token_embd 469M / ffn_*_exps 235M elements against the CPU oracle), but the
/// concurrent conversion run faulted with hipErrorLaunchFailure (719). A
/// device-side mutex makes the conversion take the proven-serial path; these
/// kernels are bandwidth-bound, so cross-tensor parallelism bought little.
fn dequant_serial_lock(ordinal: usize) -> &'static std::sync::Mutex<()> {
    static LOCKS: std::sync::OnceLock<Vec<std::sync::Mutex<()>>> = std::sync::OnceLock::new();
    let locks = LOCKS.get_or_init(|| (0..16).map(|_| std::sync::Mutex::new(())).collect());
    &locks[ordinal.min(15)]
}

impl grim_format::convert::GpuDequant for RocmDevice {
    /// `Ok(None)` => storage has no WMMA host-dequant arm; caller must use the generic dequant path.
    fn dequantize(
        &self,
        storage: &grim_tensor::dtype::Storage,
        bytes: &[u8],
        elem_count: usize,
    ) -> grim_tensor::error::Result<Option<Vec<f32>>> {
        let _serial = dequant_serial_lock(self.ordinal)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme, Storage};
        match storage {
            Storage::KQuant(KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XXS) => {
                Ok(Some(self.dequantize_iq2xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XS) => {
                Ok(Some(self.dequantize_iq2xs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2S) => {
                Ok(Some(self.dequantize_iq2s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3XXS) => {
                Ok(Some(self.dequantize_iq3xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3S) => {
                Ok(Some(self.dequantize_iq3s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4NL) => {
                Ok(Some(self.dequantize_iq4nl_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4XS) => {
                Ok(Some(self.dequantize_iq4xs_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::Fp8) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp4) => {
                Ok(Some(self.dequantize_mxfp4_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp8) => {
                Ok(Some(self.dequantize_mxfp8_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::NvFp4) => {
                Ok(Some(self.dequantize_nvfp4_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::NutFp4) => {
                Ok(Some(self.dequantize_nutcracker_host(bytes, elem_count)?))
            }
            Storage::Block(BlockDtype::Fp8 | BlockDtype::Fp8Block16) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod wmma_route_tests {
    use super::*;
    use grim_format::spec::{GrimTensorExt, LayoutHintTag};

    fn ext_packed(bits: u8) -> GrimTensorExt {
        GrimTensorExt {
            tensor_name: "test.weight".into(),
            layout_hint: LayoutHintTag::PackedQuantWmma {
                bits,
                frag_m: 16,
                frag_n: 16,
            },
            ..Default::default()
        }
    }

    #[test]
    fn no_extension_routes_to_default() {
        assert!(!wmma_route_decision(None, ArithType::F16, true));
    }

    #[test]
    fn disabled_config_skips_wmma() {
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            false,
        ));
    }

    #[test]
    fn packed_4bit_f16_enabled() {
        assert!(wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_2bit_supported_too() {
        assert!(wmma_route_decision(
            Some(&ext_packed(2)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_unsupported_bpw_falls_back() {
        // 6-bit is not in {2,3,4,8} → must not dispatch to WMMA.
        assert!(!wmma_route_decision(
            Some(&ext_packed(6)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn non_f16_output_skips_wmma() {
        // WMMA path only registered for F16; F32 arch falls to rocBLAS.
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F32,
            true,
        ));
    }

    #[test]
    fn default_hint_skips_wmma() {
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::Default,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }

    #[test]
    fn wavefront_tiled_does_not_route_wmma() {
        // WavefrontTiled goes through a different (existing) tiled path; do
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::WavefrontTiled,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }
}

#[cfg(test)]
mod block_size_tests {
    use super::RocmDevice as T;
    use std::path::PathBuf;

    /// The table's block sizes must equal llama.cpp's block layouts in
    /// `old/repo/llama.cpp-master/ggml/src/ggml-common.h`.
    ///
    /// This gate previously missed three real defects because it computed its
    /// own expectations with hand-written arithmetic instead of parsing them:
    /// it omitted the `sizeof(uint16_t) scales_h` term, so it expected iq4xs
    /// to be 134 (the table's wrong value) instead of the reference's 136 and
    /// passed on a wrong table. The expectations are now derived from the
    /// header's own member lists.
    ///
    /// If the reference is absent the test panics rather than passing
    /// vacuously — a gate that cannot fail is worse than no gate.
    ///
    /// Scope: this pins STRIDE only. `dequant_iq2s` has the correct 82-byte
    /// size and is still wrong in three ways against `dequantize_row_iq2_s`.
    fn header() -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crate -> repo root")
            .join("old/repo/llama.cpp-master/ggml/src/ggml-common.h");
        std::fs::read_to_string(&p).unwrap_or_else(|e| {
            panic!(
                "cannot read the llama.cpp reference at {}: {e}\n\
                 This gate compares grim against the REFERENCE; without it there \
                 is nothing to compare to and the check must not pass.",
                p.display()
            )
        })
    }

    fn qk_k(src: &str) -> u32 {
        src.lines()
            .find_map(|l| l.trim().strip_prefix("#define QK_K "))
            .and_then(|r| r.trim().trim_end_matches('u').parse().ok())
            .expect("ggml-common.h defines QK_K")
    }

    #[test]
    fn tiled_quant_block_sizes_match_llama_cpp() {
        let src = header();
        let qk = qk_k(&src);
        let iq3s_n_scale = src
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define IQ3S_N_SCALE QK_K/"))
            .and_then(|r| r.trim().parse::<u32>().ok())
            .map(|d| qk / d)
            .expect("ggml-common.h defines IQ3S_N_SCALE");

        // (tag, elements per block, bytes per block), transcribed from each
        // struct's own `static_assert` in ggml-common.h. The `2` in the
        // iq4xs row is `sizeof(uint16_t) scales_h`; omitting it is what let
        // the table's wrong 134 pass this gate.
        let expected: Vec<(&str, u32, u32)> = vec![
            ("q8_0", 32, 34),
            ("q5k", qk, 176),
            ("q6k", qk, 210),
            ("iq2xxs", qk, 2 + qk / 8 * 2),
            ("iq2xs", qk, 2 + qk / 8 * 2 + qk / 32),
            ("iq2s", qk, 2 + qk / 4 + qk / 16),
            ("iq3xxs", qk, 2 + 3 * (qk / 8)),
            ("iq3s", qk, 2 + 13 * (qk / 32) + iq3s_n_scale),
            // block_iq4_xs: d, scales_h(u16), scales_l[QK_K/64], qs[QK_K/2]
            ("iq4xs", qk, 2 + 2 + qk / 64 + qk / 2),
        ];

        // Read the TABLE, not `tiled_quant_lookup`: that accessor returns
        // only (tag, elements) and would make the bytes column vacuous. It
        // demonstrably was — restoring iq3xxs 96 / iq4xs 136 left this test
        // green, because nothing looked at those numbers.
        let mut problems: Vec<String> = Vec::new();
        let row = |tag: &str| {
            T::TILED_QUANT_TABLE
                .iter()
                .find(|(_, t, _, _)| *t == tag)
                .copied()
        };
        for (tag, elems, bytes) in expected {
            match row(tag) {
                None => problems.push(format!(
                    "{tag}: absent from TILED_QUANT_TABLE, expected {elems} elements / {bytes} B"
                )),
                Some((_, got_tag, got_blk, got_bytes)) => {
                    if got_tag != tag {
                        problems.push(format!("{tag}: row is tagged {got_tag}"));
                    }
                    if got_blk != elems {
                        problems.push(format!(
                            "{tag}: {got_blk} elements/block, reference says {elems}"
                        ));
                    }
                    if got_bytes != bytes {
                        problems.push(format!(
                            "{tag}: {got_bytes} B/block, reference says {bytes} B"
                        ));
                    }
                }
            }
        }
        // iq4_nl must NOT be tiled: QK4_NL = 32 cannot fill a 256-element tile.
        if row("iq4nl").is_some() {
            problems.push(
                "iq4nl: present in TILED_QUANT_TABLE, but its block is QK4_NL = 32 \
                 elements and cannot fill a 256-element WMMA tile"
                    .into(),
            );
        }
        assert!(
            problems.is_empty(),
            "tiled-quant block geometry disagrees with ggml-common.h:\n{}\n\
             A wrong stride is a wrong memory access, not just a wrong number.",
            problems.join("\n")
        );
    }
}

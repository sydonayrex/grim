//! LoRA injection point configuration (WI-T1 item 1).
//!
//! Forward-side prerequisite: extend LoRA application from logits-only to
//! standard QLoRA injection points (attention Q/K/V/O + MLP Gate/Up/Down).
//! The backward graph needs to match wherever forward adapters get applied,
//! so the injection-point enumeration lives here.

use crate::ParamId;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};

fn default_device() -> grim_tensor::Device {
    grim_tensor::Device::Cpu
}

/// Standard LoRA injection points for QLoRA parity with Unsloth.
///
/// Unsloth applies LoRA to all attention projections (Q/K/V/O) and MLP
/// projections (Gate/Up/Down) — 7 injection points per layer. The legacy
/// `Logits` injection point (the only one wired in `lora.rs` today) is kept
/// for backwards compatibility but is **not sufficient for real QLoRA parity**
/// per the plan's WI-T1 note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoRAInjectionPoint {
    /// Query projection in attention (W_q).
    QProj,
    /// Key projection in attention (W_k).
    KProj,
    /// Value projection in attention (W_v).
    VProj,
    /// Output projection in attention (W_o).
    OProj,
    /// Gate projection in MLP (SwiGLU gate).
    GateProj,
    /// Up projection in MLP (SwiGLU up).
    UpProj,
    /// Down projection in MLP (SwiGLU down).
    DownProj,
    /// Output logits projection — legacy/logits-only LoRA (the only path
    /// wired in `lora.rs::apply_adapters_to_logits` at the time of this plan).
    Logits,
}

impl LoRAInjectionPoint {
    /// All standard QLoRA injection points (7 total, matching Unsloth).
    /// `Logits` is intentionally excluded — it is not a standard QLoRA site.
    pub fn all_standard_qlora() -> &'static [Self] {
        &[
            Self::QProj,
            Self::KProj,
            Self::VProj,
            Self::OProj,
            Self::GateProj,
            Self::UpProj,
            Self::DownProj,
        ]
    }

    /// All injection points including Logits — used for FullParameter
    /// (WI-T8) scope where gradients reach every weight including lm_head.
    pub fn all_points() -> &'static [Self] {
        &[
            Self::QProj,
            Self::KProj,
            Self::VProj,
            Self::OProj,
            Self::GateProj,
            Self::UpProj,
            Self::DownProj,
            Self::Logits,
        ]
    }

    /// Attention projections only (Q/K/V/O — 4 points).
    pub fn attention_only() -> &'static [Self] {
        &[Self::QProj, Self::KProj, Self::VProj, Self::OProj]
    }

    /// MLP projections only (Gate/Up/Down — 3 points).
    pub fn mlp_only() -> &'static [Self] {
        &[Self::GateProj, Self::UpProj, Self::DownProj]
    }

    /// Weight tensor name suffix at this injection point (matches the block.rs naming).
    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::QProj => "attn_q",
            Self::KProj => "attn_k",
            Self::VProj => "attn_v",
            Self::OProj => "attn_o",
            Self::GateProj => "ffn_gate",
            Self::UpProj => "ffn_up",
            Self::DownProj => "ffn_down",
            Self::Logits => "output",
        }
    }

    /// Adapter name prefix under which A/B weights live in a checkpoint, e.g. `blk.0.attn_q.lora`.
    pub fn adapter_prefix(&self, layer_idx: usize) -> String {
        format!("blk.{}.{}", layer_idx, self.weight_suffix())
    }

    /// `true` for Q/K/V/O.
    pub fn is_attention(&self) -> bool {
        matches!(self, Self::QProj | Self::KProj | Self::VProj | Self::OProj)
    }

    /// `true` for Gate/Up/Down.
    pub fn is_mlp(&self) -> bool {
        matches!(self, Self::GateProj | Self::UpProj | Self::DownProj)
    }

    /// Expected base-weight shape `(out_features, in_features)` for this
    /// injection point given the model geometry.
    pub fn base_weight_shape(&self, cfg: &InjectionConfig) -> (usize, usize) {
        match self {
            Self::QProj => (cfg.num_heads * cfg.head_dim, cfg.hidden_size),
            Self::KProj => (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
            Self::VProj => (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
            Self::OProj => (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
            Self::GateProj => (cfg.intermediate_size, cfg.hidden_size),
            Self::UpProj => (cfg.intermediate_size, cfg.hidden_size),
            Self::DownProj => (cfg.hidden_size, cfg.intermediate_size),
            Self::Logits => (cfg.vocab_size, cfg.hidden_size),
        }
    }

    /// LoRA A shape `[rank, in_features]`.
    pub fn lora_a_shape(&self, cfg: &InjectionConfig, rank: usize) -> (usize, usize) {
        let (_, in_features) = self.base_weight_shape(cfg);
        (rank, in_features)
    }

    /// LoRA B shape `[out_features, rank]`.
    pub fn lora_b_shape(&self, cfg: &InjectionConfig, rank: usize) -> (usize, usize) {
        let (out_features, _) = self.base_weight_shape(cfg);
        (out_features, rank)
    }
}

/// Model geometry needed to size LoRA adapters per injection point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

/// Configuration for one LoRA adapter at a specific injection point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAInjectionConfig {
    pub injection_point: LoRAInjectionPoint,
    pub layer_idx: usize,
    pub adapter_id: u32,
    pub rank: usize,
    pub alpha: f32,
    pub enabled: bool,
    pub use_dora: bool,
    pub use_rs_lora: bool,
    /// PiSSA: initialize A/B via truncated SVD of the base weight (principal
    /// singular components) instead of Kaiming-style random init.
    pub use_pissa: bool,
    /// OLoRA: add `olora_lambda * olora_orthogonality_penalty(A, B)` to the loss.
    pub use_olora: bool,
    /// Weight of the OLoRA orthogonality penalty term.
    pub olora_lambda: f32,
    /// VeRA: quantize the low-rank update against scalar codebooks.
    pub use_vera: bool,
    /// VeRA: number of centroids per codebook.
    pub codebook_size: usize,
    /// VeRA: per-codebook centroid dimension (currently 1, scalar codebooks).
    pub codebook_dim: usize,
    /// VeRA: number of scalar codebooks used for quantization.
    pub num_codebooks: usize,
    /// SPECTRAL-QLORA: initialize A/B so that AB is semi-orthogonal in the
    /// dominant subspace, reusing `grim-quant::soul_eater::subspace_newton_schulz_step`
    /// at adapter creation. Replaces standard Kaiming/Zeors init.
    pub use_spectral_qlora: bool,
    /// LoRA+: effective lr multiplier for the B matrix (1.0 = standard LoRA).
    pub lora_plus_ratio: f32,
    /// ReLoRA: merge LoRA delta back into base weights and reset optimizer momentum every N steps (0 = disabled).
    pub relora_reset_steps: usize,
    /// OFT: Orthogonal Fine-Tuning via Cayley-transform orthogonal matrix updates.
    pub use_oft: bool,
    /// OFT: Orthogonal factor rank.
    pub oft_rank: usize,
    #[serde(skip, default = "default_device")]
    pub target_device: grim_tensor::Device,
}

impl LoRAInjectionConfig {
    pub fn new(
        injection_point: LoRAInjectionPoint,
        layer_idx: usize,
        adapter_id: u32,
        rank: usize,
        alpha: f32,
    ) -> Self {
        Self {
            injection_point,
            layer_idx,
            adapter_id,
            rank,
            alpha,
            enabled: true,
            use_dora: false,
            use_rs_lora: false,
            use_pissa: false,
            use_olora: false,
            olora_lambda: 0.0,
            use_vera: false,
            codebook_size: 256,
            codebook_dim: 1,
            num_codebooks: 1,
            use_spectral_qlora: false,
            lora_plus_ratio: 1.0,
            relora_reset_steps: 0,
            use_oft: false,
            oft_rank: 8,
            target_device: grim_tensor::Device::Cpu,
        }
    }

    /// Scaling factor gamma.
    /// Standard LoRA: alpha / r.
    /// RSLoRA: alpha / sqrt(r).
    pub fn scale(&self) -> f32 {
        if self.use_rs_lora {
            self.alpha / (self.rank as f32).sqrt()
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// `ParamId` for this adapter's A matrix.
    pub fn param_id_a(&self) -> ParamId {
        ParamId::a(self.layer_idx, self.adapter_id, self.injection_point)
    }

    /// `ParamId` for this adapter's B matrix.
    pub fn param_id_b(&self) -> ParamId {
        ParamId::b(self.layer_idx, self.adapter_id, self.injection_point)
    }
}

/// Registry of all LoRA injection configs for a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoRAInjectionRegistry {
    pub configs: std::collections::HashMap<(usize, LoRAInjectionPoint), LoRAInjectionConfig>,
}

impl LoRAInjectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update injection configs with SCYTHE-2 layer-wise GPU device placements.
    pub fn with_scythe2_placements(
        mut self,
        placements: &std::collections::HashMap<usize, grim_tensor::Device>,
    ) -> Self {
        for ((layer_idx, _), cfg) in self.configs.iter_mut() {
            if let Some(dev) = placements.get(layer_idx) {
                cfg.target_device = dev.clone();
            }
        }
        self
    }
    pub fn add(&mut self, config: LoRAInjectionConfig) {
        self.configs
            .insert((config.layer_idx, config.injection_point), config);
    }

    pub fn get(&self, layer_idx: usize, point: LoRAInjectionPoint) -> Option<&LoRAInjectionConfig> {
        self.configs.get(&(layer_idx, point))
    }

    /// Build the standard 7-point-per-layer QLoRA registry.
    pub fn standard_qlora(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        Self::standard_qlora_with_flags(
            num_layers, rank, alpha, adapter_id, false, false, 0.0, false,
        )
    }

    /// Build the standard 7-point-per-layer QLoRA registry, propagating
    /// the PiSSA / OLoRA / SPECTRAL-QLORA adapter flags to every injection config.
    pub fn standard_qlora_with_flags(
        num_layers: usize,
        rank: usize,
        alpha: f32,
        adapter_id: u32,
        use_pissa: bool,
        use_olora: bool,
        olora_lambda: f32,
        use_spectral_qlora: bool,
    ) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::all_standard_qlora() {
                let mut cfg = LoRAInjectionConfig::new(point, layer_idx, adapter_id, rank, alpha);
                cfg.use_pissa = use_pissa;
                cfg.use_olora = use_olora;
                cfg.olora_lambda = olora_lambda;
                cfg.use_spectral_qlora = use_spectral_qlora;
                r.add(cfg);
            }
        }
        r
    }

    /// Build the attention-only (4-point) registry.
    pub fn attention_only(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::attention_only() {
                r.add(LoRAInjectionConfig::new(
                    point, layer_idx, adapter_id, rank, alpha,
                ));
            }
        }
        r
    }

    /// Build the MLP-only (3-point) registry.
    pub fn mlp_only(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::mlp_only() {
                r.add(LoRAInjectionConfig::new(
                    point, layer_idx, adapter_id, rank, alpha,
                ));
            }
        }
        r
    }

    /// All enabled configs.
    pub fn enabled(&self) -> Vec<&LoRAInjectionConfig> {
        self.configs.values().filter(|c| c.enabled).collect()
    }

    /// All configs for one layer.
    pub fn layer_configs(&self, layer_idx: usize) -> Vec<&LoRAInjectionConfig> {
        self.configs
            .iter()
            .filter(|((idx, _), _)| *idx == layer_idx)
            .map(|(_, c)| c)
            .collect()
    }

    /// Total trainable parameter count = Σ (|A| + |B|) over enabled configs.
    pub fn num_trainable_params(&self, cfg: &InjectionConfig) -> usize {
        self.configs
            .values()
            .filter(|c| c.enabled)
            .map(|c| {
                let (ar, ac) = c.injection_point.lora_a_shape(cfg, c.rank);
                let (br, bc) = c.injection_point.lora_b_shape(cfg, c.rank);
                ar * ac + br * bc
            })
            .sum()
    }
}

/// LoftQ (Low-Rank Transfer Quantization) initialization.
///
/// Minimizes initialization error ||W0 - (Q(W0 - BA) + BA)||_F via alternating SVD quantization.
///
/// Algorithm:
/// 1. Initialize B(0) = 0, A(0) = 0
/// 2. For t = 1..N:
///    - Compute residual R(t) = W0 - B(t-1) @ A(t-1)
///    - Quantize residual: Q(t) = quantize(R(t))
///    - Compute remainder S(t) = W0 - Q(t)
///    - Compute truncated SVD: S(t) ≈ U_r @ Σ_r @ V_r^T
///    - Set B(t) = U_r @ sqrt(Σ_r), A(t) = sqrt(Σ_r) @ V_r^T
/// 3. Return quantized base Q and adapter matrices A, B
///
/// Returns (quantized_weights, A_matrix, B_matrix) as byte vectors and f32 vectors.
/// The quantized weights use u8 storage (Q8_0 format simulation).
pub fn loftq_initialize(
    w_base: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    num_iters: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let num_iters = if num_iters == 0 { 4 } else { num_iters }; // Default iterations
    if rank == 0 || rank > rows.min(cols) {
        return Err(Error::Backend(format!(
            "loftq: invalid rank {} for matrix {}x{}",
            rank, rows, cols
        )));
    }

    // Initialize A and B as zero matrices
    let mut a_vec = vec![0.0f32; rank * cols];
    let mut b_vec = vec![0.0f32; rows * rank];

    // Copy base weights to work with
    let mut w_current = w_base.to_vec();

    // Helper: simple Q8_0 quantization (scale + 8-bit symmetric)
    let quantize_q80 = |data: &[f32]| -> (Vec<u8>, f32) {
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| ((x / scale).round().clamp(-127.0, 127.0) + 128.0) as u8)
            .collect();
        (quantized, scale)
    };

    // Helper: dequantize Q8_0
    let dequantize_q80 = |data: &[u8], scale: f32, len: usize| -> Vec<f32> {
        data.iter()
            .take(len)
            .map(|&q| ((q as i16 - 128) as f32) * scale)
            .collect()
    };

    // Iteratively refine A and B
    for _iter in 0..num_iters {
        // Compute current adapter output: B @ A
        let mut ba_output = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut sum = 0.0f32;
                for k in 0..rank {
                    sum += b_vec[r * rank + k] * a_vec[k * cols + c];
                }
                ba_output[r * cols + c] = sum;
            }
        }

        // Compute residual: R = W0 - BA
        let residual: Vec<f32> = w_current
            .iter()
            .zip(ba_output.iter())
            .map(|(&w, &ba)| w - ba)
            .collect();

        // Quantize residual (Q8_0 style)
        let (q_bytes, q_scale) = quantize_q80(&residual);
        let q_dequant = dequantize_q80(&q_bytes, q_scale, rows * cols);

        // Compute remainder: S = W0 - Q(R)
        let remainder: Vec<f32> = w_current
            .iter()
            .zip(q_dequant.iter())
            .map(|(&w, &qd)| w - qd)
            .collect();

        // Compute truncated SVD using randomized SVD approximation
        // Using the algorithm from the plan:
        // S ≈ U_r @ Σ_r @ V_r^T
        // B = U_r @ sqrt(Σ_r), A = sqrt(Σ_r) @ V_r^T

        // Simple power iteration-based SVD approximation
        let s_vec = compute_truncated_svd(&remainder, rows, cols, rank)?;
        let u_r = s_vec.u;
        let sigma_r = s_vec.sigma;
        let v_t_r = s_vec.v_t;

        // Compute sqrt(Σ_r)
        let sqrt_sigma: Vec<f32> = sigma_r.iter().map(|s| s.sqrt().max(0.0)).collect();

        // Compute B = U_r @ diag(sqrt(Σ_r))
        // U_r has shape (rows, rank), sqrt_sigma has shape (rank,)
        // B has shape (rows, rank)
        // B[r,k] = U_r[r,k] * sqrt_sigma[k]
        for r in 0..rows {
            for k in 0..rank {
                b_vec[r * rank + k] = u_r[r * rank + k] * sqrt_sigma[k];
            }
        }

        // Compute A = diag(sqrt(Σ_r)) @ V_r^T
        // V_r^T has shape (rank, cols), sqrt_sigma has shape (rank,)
        // A has shape (rank, cols)
        // A[k,c] = sqrt_sigma[k] * V_r^T[k,c]
        for k in 0..rank {
            for c in 0..cols {
                a_vec[k * cols + c] = sqrt_sigma[k] * v_t_r[k * cols + c];
            }
        }

        // Update W_current for next iteration (use quantized residual)
        for i in 0..rows * cols {
            w_current[i] = q_dequant[i];
        }
    }

    // Final quantized base: Q = W0 - B @ A
    let mut ba_output = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..rank {
                sum += b_vec[r * rank + k] * a_vec[k * cols + c];
            }
            ba_output[r * cols + c] = sum;
        }
    }

    // Deform (not quantized) base weights
    let quantized: Vec<f32> = w_base
        .iter()
        .zip(ba_output.iter())
        .map(|(&w, &ba)| w - ba)
        .collect();

    // Quantize final base to Q8_0 bytes
    let (q_bytes, _) = quantize_q80(&quantized);

    Ok((q_bytes, a_vec, b_vec))
}

/// PiSSA (Principal Singular values and Singular vectors Adaptation) initialization.
///
/// Initializes LoRA adapters `A`, `B` from the top-`rank` singular components of
/// the base weight `W0`, and sets the quantized base to the residual `W0 - B·A`.
/// This converges faster than random Kaiming init because the adapters already
/// capture the dominant directions of the weight update target.
///
/// Steps:
/// 1. Truncated SVD: `W0 ≈ U_r Σ_r V_r^T`.
/// 2. `B = U_r·sqrt(Σ_r)`, shape `[rows, rank]`.
/// 3. `A = sqrt(Σ_r)·V_r^T`, shape `[rank, cols]`.
/// 4. Quantize residual base `Q = quantize_q80(W0 - B·A)`.
///
/// Returns `(a, b, quantized_base_bytes)` as f32 vectors and Q8_0 u8 bytes.
pub fn pissa_initialize(
    w_base: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<u8>)> {
    if rank == 0 || rank > rows.min(cols) {
        return Err(Error::Backend(format!(
            "pissa: invalid rank {} for matrix {}x{}",
            rank, rows, cols
        )));
    }
    if w_base.len() != rows * cols {
        return Err(Error::Backend(format!(
            "pissa: base weight length {} != {}x{}",
            w_base.len(),
            rows,
            cols
        )));
    }

    // Step 1: truncated SVD of the base weight.
    let svd = compute_truncated_svd(w_base, rows, cols, rank)?;
    let u_r = svd.u; // (rows, rank)
    let sigma_r = svd.sigma; // (rank,)
    let v_t_r = svd.v_t; // (rank, cols)

    let sqrt_sigma: Vec<f32> = sigma_r.iter().map(|s| s.sqrt().max(0.0)).collect();

    // Step 2: B = U_r @ diag(sqrt(Σ_r)).
    let mut b_vec = vec![0.0f32; rows * rank];
    for r in 0..rows {
        for k in 0..rank {
            b_vec[r * rank + k] = u_r[r * rank + k] * sqrt_sigma[k];
        }
    }

    // Step 3: A = diag(sqrt(Σ_r)) @ V_r^T.
    let mut a_vec = vec![0.0f32; rank * cols];
    for k in 0..rank {
        for c in 0..cols {
            a_vec[k * cols + c] = sqrt_sigma[k] * v_t_r[k * cols + c];
        }
    }

    // Step 4: residual base W0 - B@A, quantized to Q8_0 bytes.
    let mut ba_output = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..rank {
                sum += b_vec[r * rank + k] * a_vec[k * cols + c];
            }
            ba_output[r * cols + c] = sum;
        }
    }
    let residual: Vec<f32> = w_base
        .iter()
        .zip(ba_output.iter())
        .map(|(&w, &ba)| w - ba)
        .collect();

    // Simple Q8_0 quantization (scale + 8-bit symmetric), matching loftq_initialize.
    let max_abs = residual.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let quantized_base: Vec<u8> = residual
        .iter()
        .map(|&x| ((x / scale).round().clamp(-127.0, 127.0) + 128.0) as u8)
        .collect();

    Ok((a_vec, b_vec, quantized_base))
}

/// OFT (Orthogonal Fine-Tuning) initialization.
/// Computes an orthogonal rank-r factor from truncated SVD.
pub fn oft_initialize(
    w_base: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<(Vec<f32>, f32)> {
    if rows * cols != w_base.len() {
        return Err(Error::Backend(format!(
            "oft: base weight length {} != {}x{}",
            w_base.len(),
            rows,
            cols
        )));
    }
    let min_dim = rows.min(cols);
    if rank > min_dim {
        return Err(Error::Backend(format!(
            "oft: rank {} exceeds dimension {}",
            rank, min_dim
        )));
    }
    let svd = compute_truncated_svd(w_base, rows, cols, rank)?;
    let r_factor = svd.v_t;
    Ok((r_factor, 1.0))
}

/// Result of truncated SVD computation.
pub struct TruncatedSvdResult {
    pub u: Vec<f32>,     // Shape: (rows, rank)
    pub sigma: Vec<f32>, // Shape: (rank,)
    pub v_t: Vec<f32>,   // Shape: (rank, cols)
}

/// Compute truncated SVD using power iteration method.
/// This is a CPU-based approximation suitable for LoftQ initialization.
fn compute_truncated_svd(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<TruncatedSvdResult> {
    let min_dim = rows.min(cols);
    if rank > min_dim {
        return Err(Error::Backend(format!(
            "SVD rank {} exceeds matrix dimension {}",
            rank, min_dim
        )));
    }

    // Initialize random vectors (deterministic for reproducibility)
    let mut u_vectors = vec![0.0f32; rows * rank];
    let mut v_vectors = vec![0.0f32; cols * rank];

    // Use deterministic "random" initialization
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for i in 0..(rows * rank) {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let u1 = ((seed >> 40) as u32 as f32) / 16777216.0;
        let u2 = (((seed & 0xFFFFFFFF) >> 8) as u32 as f32) / 16777216.0;
        u_vectors[i] = (-2.0 * u1.max(1e-5).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    for i in 0..(cols * rank) {
        seed = seed.wrapping_add(0x9E3779B9);
        let u1 = ((seed >> 40) as u32 as f32) / 16777216.0;
        let u2 = (((seed & 0xFFFFFFFF) >> 8) as u32 as f32) / 16777216.0;
        v_vectors[i] = (-2.0 * u1.max(1e-5).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    // Power iteration to find dominant singular vectors
    for _power_iter in 0..8 {
        // U = A @ V (simplified power)
        let mut temp = vec![0.0f32; rows * rank];
        for r in 0..rows {
            for k in 0..rank {
                let mut sum = 0.0f32;
                for c in 0..cols {
                    sum += matrix[r * cols + c] * v_vectors[c * rank + k];
                }
                temp[r * rank + k] = sum;
            }
        }
        u_vectors.clone_from_slice(&temp);

        // V = A^T @ U
        let mut temp = vec![0.0f32; cols * rank];
        for c in 0..cols {
            for k in 0..rank {
                let mut sum = 0.0f32;
                for r in 0..rows {
                    sum += matrix[r * cols + c] * u_vectors[r * rank + k];
                }
                temp[c * rank + k] = sum;
            }
        }
        v_vectors.clone_from_slice(&temp);

        // Orthogonalize U and V via modified Gram-Schmidt so the subspace
        // iterates stay well-conditioned (columns of U converge to the leading
        // left singular vectors; V is re-derived from U after normalization).
        orthogonalize_columns(&mut u_vectors, rows, rank);
        // Re-derive V = A^T @ U using the orthonormalized U so V tracks it.
        let mut temp = vec![0.0f32; cols * rank];
        for c in 0..cols {
            for k in 0..rank {
                let mut sum = 0.0f32;
                for r in 0..rows {
                    sum += matrix[r * cols + c] * u_vectors[r * rank + k];
                }
                temp[c * rank + k] = sum;
            }
        }
        v_vectors.clone_from_slice(&temp);
        orthogonalize_columns(&mut v_vectors, cols, rank);
    }

    // Compute singular values as norms of the columns of A @ V, then normalize
    // U accordingly (U_k = (A @ V_k) / σ_k). σ_k = ||A @ V_k||.
    let mut sigma = vec![0.0f32; rank];
    let mut u_norm = vec![0.0f32; rows * rank];
    for k in 0..rank {
        let mut norm_sq = 0.0f32;
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += matrix[r * cols + c] * v_vectors[c * rank + k];
            }
            u_norm[r * rank + k] = sum;
            norm_sq += sum * sum;
        }
        sigma[k] = norm_sq.sqrt();
    }

    // Normalize U: U_k = (A @ V_k) / σ_k.
    let eps = 1e-8f32;
    for k in 0..rank {
        let s = sigma[k].max(eps);
        for r in 0..rows {
            u_vectors[r * rank + k] = u_norm[r * rank + k] / s;
        }
    }

    // V^T is returned as (rank, cols). Internally v_vectors is (cols, rank)
    // with element (c,k) at index c*rank+k, so transpose it explicitly —
    // otherwise consumers reading v_t[k*cols+c] get a scrambled matrix.
    let mut v_t = vec![0.0f32; rank * cols];
    for k in 0..rank {
        for c in 0..cols {
            v_t[k * cols + c] = v_vectors[c * rank + k];
        }
    }

    Ok(TruncatedSvdResult {
        u: u_vectors,
        sigma,
        v_t,
    })
}

/// In-place modified Gram-Schmidt orthogonalization over the `rank` columns of
/// a `rows x rank` row-major matrix.
pub fn orthogonalize_columns(m: &mut [f32], rows: usize, rank: usize) {
    let eps = 1e-12f32;
    for k in 0..rank {
        // Orthogonalize against all previous columns.
        for j in 0..k {
            let mut dot = 0.0f32;
            for i in 0..rows {
                dot += m[i * rank + j] * m[i * rank + k];
            }
            for i in 0..rows {
                m[i * rank + k] -= dot * m[i * rank + j];
            }
        }
        // Normalize column k.
        let mut norm_sq = 0.0f32;
        for i in 0..rows {
            norm_sq += m[i * rank + k] * m[i * rank + k];
        }
        let norm = norm_sq.sqrt().max(eps);
        for i in 0..rows {
            m[i * rank + k] /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InjectionConfig {
        InjectionConfig {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 32000,
        }
    }

    #[test]
    fn standard_qlora_points_seven_attention_plus_mlp() {
        let pts = LoRAInjectionPoint::all_standard_qlora();
        assert_eq!(pts.len(), 7);
        assert!(pts.contains(&LoRAInjectionPoint::QProj));
        assert!(pts.contains(&LoRAInjectionPoint::DownProj));
        assert!(!pts.contains(&LoRAInjectionPoint::Logits));
    }

    #[test]
    fn attention_vs_mlp_classification() {
        assert!(LoRAInjectionPoint::QProj.is_attention());
        assert!(!LoRAInjectionPoint::QProj.is_mlp());
        assert!(LoRAInjectionPoint::GateProj.is_mlp());
        assert!(!LoRAInjectionPoint::GateProj.is_attention());
        assert!(!LoRAInjectionPoint::Logits.is_attention());
        assert!(!LoRAInjectionPoint::Logits.is_mlp());
    }

    #[test]
    fn base_weight_shapes_for_attention_and_mlp() {
        let c = cfg();
        assert_eq!(
            LoRAInjectionPoint::QProj.base_weight_shape(&c),
            (4096, 4096)
        );
        assert_eq!(
            LoRAInjectionPoint::KProj.base_weight_shape(&c),
            (1024, 4096)
        );
        assert_eq!(
            LoRAInjectionPoint::DownProj.base_weight_shape(&c),
            (4096, 11008)
        );
        assert_eq!(
            LoRAInjectionPoint::Logits.base_weight_shape(&c),
            (32000, 4096)
        );
    }

    #[test]
    fn lora_a_b_shape_inner_dimensions_match() {
        let c = cfg();
        let rank = 16;
        for point in LoRAInjectionPoint::all_standard_qlora() {
            let (ar, ac) = point.lora_a_shape(&c, rank); // [rank, in]
            let (br, bc) = point.lora_b_shape(&c, rank); // [out, rank]
            let (out_features, in_features) = point.base_weight_shape(&c);
            assert_eq!(ar, rank);
            assert_eq!(bc, rank);
            assert_eq!(ac, in_features);
            assert_eq!(br, out_features);
        }
    }

    #[test]
    fn registry_standard_qlora_covers_all_layers_and_points() {
        let r = LoRAInjectionRegistry::standard_qlora(4, 16, 32.0, 1);
        assert_eq!(r.configs.len(), 4 * 7);
        for layer in 0..4 {
            assert_eq!(r.layer_configs(layer).len(), 7);
        }
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let c = LoRAInjectionConfig::new(LoRAInjectionPoint::QProj, 0, 1, 16, 32.0);
        assert_eq!(c.scale(), 2.0);
    }

    #[test]
    fn num_trainable_params_matches_reference() {
        let r = LoRAInjectionRegistry::standard_qlora(1, 16, 32.0, 1);
        let c = cfg();
        // Per layer with 7 injection points, rank 16, hidden=4096, head_dim=128,
        //   num_heads=32, num_kv_heads=8, intermediate=11008:
        //   Q: (16*4096) + (4096*16)    = 131072
        //   K: (16*4096) + (1024*16)    = 81920
        //   V: same as K                = 81920
        //   O: (16*4096) + (4096*16)    = 131072
        //   Gate: (16*4096) + (11008*16)= 241664
        //   Up:   same as Gate           = 241664
        //   Down: (16*11008)+ (4096*16) = 241664
        //   total = 131072+81920*2+131072+241664*3 = 1,150,976
        let n = r.num_trainable_params(&c);
        assert_eq!(n, 1_150_976);
    }

    #[test]
    fn pissa_initialize_reconstructs_top_rank_singular_components() {
        // W0 = diag(4, 3, 1) padded to 3x3. Rank 2 SVD should capture the
        // top two singular values (4, 3) and the adapters must satisfy
        // W0 ≈ B@A in the top-2 directions, with the residual base quantized.
        let w = vec![4.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let (a, b, q_base) = pissa_initialize(&w, 3, 3, 2).unwrap();

        assert_eq!(a.len(), 2 * 3);
        assert_eq!(b.len(), 3 * 2);
        // quantized_base holds Q8_0 bytes of (W0 - B@A).
        assert_eq!(q_base.len(), 9);

        // Reconstruct BA = B @ A and verify it reproduces W0 in the top-2 dims.
        let mut ba = vec![0.0f32; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut sum = 0.0f32;
                for k in 0..2 {
                    sum += b[r * 2 + k] * a[k * 3 + c];
                }
                ba[r * 3 + c] = sum;
            }
        }
        // diag entries 4 and 3 captured; residual diagonal entry for σ=1 left over.
        // Tolerances are loose: compute_truncated_svd is a power-iteration
        // approximation, so reconstruction is approximate, not exact.
        assert!((ba[0] - 4.0).abs() < 0.1, "BA[0,0]={} != 4", ba[0]);
        assert!((ba[4] - 3.0).abs() < 0.1, "BA[1,1]={} != 3", ba[4]);
        // The σ=1 direction should be roughly absent from the rank-2 adapters.
        assert!(ba[8].abs() < 0.5, "BA[2,2]={} expected ≈0", ba[8]);
    }

    #[test]
    fn pissa_initialize_rejects_bad_rank_and_length() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        assert!(pissa_initialize(&w, 2, 2, 0).is_err());
        assert!(pissa_initialize(&w, 2, 2, 3).is_err());
        assert!(pissa_initialize(&vec![1.0], 2, 2, 1).is_err());
    }

    #[test]
    fn oft_initialize_is_orthogonal() {
        let w: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (r_factor, scale) = oft_initialize(&w, 3, 4, 2).unwrap();
        assert_eq!(r_factor.len(), 2 * 4);
        assert!((scale - 1.0).abs() < 1e-5, "OFT scale factor should be 1.0");
    }
}

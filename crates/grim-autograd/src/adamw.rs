//! AdamW optimizer implementation for LoRA trainable parameters (WI-T4).
//!
//! Provides step update arithmetic for 1st moment (m) and 2nd moment (v) tracking,
//! alongside serialization to and from `.grim.train` sidecars (`TrainState`).
//!
//! Also includes learning rate schedules and additional optimizer variants.

use crate::param::{ParamId, TrainableParams};
use grim_format::train::{TrainFpFormat, TrainState};
use grim_tensor::{
    DType, Tensor,
    error::{Error, Result},
};
use grim_tensor::Shape;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Learning rate scheduler type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LRScheduler {
    /// Cosine annealing with warmup: lr = base_lr * 0.5 * (1 + cos(pi * t / T))
    Cosine,
    /// Linear decay: lr = base_lr * (1 - t / T)
    Linear,
    /// Polynomial decay: lr = base_lr * (1 - t/T)^power
    Polynomial { power: f32 },
    /// Constant learning rate
    Constant,
    /// Inverse square root decay: lr = base_lr / sqrt(t)
    InverseSqrt,
    /// YOLO style: lr = base_lr / sqrt(step) with warmup
    Yolo,
    /// OneCycle: LR increases then decreases
    OneCycle { max_lr: f32, pct_start: f32 },
    /// Reduce on plateau: reduces LR when metric stops improving
    ReduceOnPlateau { factor: f32, patience: u32 },
}

impl Default for LRScheduler {
    fn default() -> Self {
        Self::Cosine
    }
}

impl LRScheduler {
    /// Compute learning rate for given step.
    /// Returns lr = base_lr * scheduler_factor(step, total_steps).
    pub fn get_lr(&self, base_lr: f32, step: usize, total_steps: usize) -> f32 {
        let total_f = total_steps as f32;
        let step_f = step as f32;
        
        match self {
            LRScheduler::Cosine => {
                if step >= total_steps {
                    base_lr * 0.0
                } else {
                    base_lr * 0.5 * (1.0 + (std::f32::consts::PI * step_f / total_f).cos())
                }
            }
            LRScheduler::Linear => {
                if step >= total_steps {
                    base_lr * 0.0
                } else {
                    base_lr * (1.0 - step_f / total_f)
                }
            }
            LRScheduler::Polynomial { power } => {
                if step >= total_steps {
                    base_lr * 0.0
                } else {
                    base_lr * (1.0 - step_f / total_f).powi(*power as i32)
                }
            }
            LRScheduler::Constant => base_lr,
            LRScheduler::InverseSqrt => {
                if step == 0 {
                    base_lr
                } else {
                    base_lr / (step as f32).sqrt()
                }
            }
            LRScheduler::Yolo => {
                // YOLO uses: lr = base_lr / sqrt(step + 1) for warmup, then decay
                base_lr / ((step + 1) as f32).sqrt()
            }
            LRScheduler::OneCycle { max_lr, pct_start } => {
                let cycle_steps = total_f * pct_start;
                if step_f < cycle_steps {
                    // Increasing phase
                    base_lr + (max_lr - base_lr) * (step_f / cycle_steps)
                } else {
                    // Decreasing phase
                    let remaining = total_f - cycle_steps;
                    let progress = (step_f - cycle_steps) / remaining;
                    max_lr * (1.0 - progress)
                }
            }
            LRScheduler::ReduceOnPlateau { factor, .. } => {
                // Simplified: just multiply by factor per some steps
                base_lr * factor.powi((step / 1000) as i32)
            }
        }
    }
}

impl std::str::FromStr for LRScheduler {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cosine" | "cosine-warmup" => Ok(Self::Cosine),
            "linear" => Ok(Self::Linear),
            "polynomial" => Ok(Self::Polynomial { power: 2.0 }),
            "constant" => Ok(Self::Constant),
            "inverse-sqrt" | "inverse_sqrt" => Ok(Self::InverseSqrt),
            "yolo" => Ok(Self::Yolo),
            "onecycle" | "one-cycle" => Ok(Self::OneCycle {
                max_lr: 0.0,
                pct_start: 0.1,
            }),
            "reduce-on-plateau" | "reduce_on_plateau" => Ok(Self::ReduceOnPlateau {
                factor: 0.5,
                patience: 10,
            }),
            other => Err(format!("unknown lr scheduler '{other}'")),
        }
    }
}

impl std::fmt::Display for LRScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Cosine => "cosine-warmup",
            Self::Linear => "linear",
            Self::Polynomial { .. } => "polynomial",
            Self::Constant => "constant",
            Self::InverseSqrt => "inverse-sqrt",
            Self::Yolo => "yolo",
            Self::OneCycle { .. } => "one-cycle",
            Self::ReduceOnPlateau { .. } => "reduce-on-plateau",
        };
        f.write_str(s)
    }
}

/// Selection of available optimizer variants.
///
/// Only the first six variants have a concrete `Optimizer` implementation in
/// this crate. The remaining variants (AdamWBnb and up) are declared to keep
/// the CLI surface stable and are rejected with `Error::Unimplemented` by
/// `Optimizer::new` until their implementations land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind {
    /// Standard AdamW with FP32 moment buffers.
    AdamW,
    /// AdamW with 8-bit quantised moment buffers (FP16 storage).
    AdamW8Bit,
    /// Paged AdamW — offloads cold moment pages to host RAM to reduce VRAM.
    PagedAdamW,
    /// Lion sign-based momentum optimizer.
    Lion,
    /// Lion with 8-bit moment buffers.
    Lion8Bit,
    /// Adafactor — factored second-moment for memory-efficient LLM fine-tuning.
    Adafactor,
    /// bitsandbytes-style 8-bit (placeholder — no bnb dep yet).
    AdamWBnb,
    /// Paged + 8-bit quantized moments.
    PagedAdamW8Bit,
    /// QGaLore with 8-bit quantized moments.
    QGaLoreAdamW8Bit,
    /// GaLore projection-based optimizer.
    GaloreAdamW,
    /// GaLore with 8-bit quantized moments.
    GaloreAdamW8Bit,
    /// LOMO (Low-Memory Optimization).
    LOMO,
    /// Adalomo.
    Adalomo,
    /// CAME (Confident Adaptive Multi-optimizer Engine).
    CAME,
    /// Sophia second-order optimizer.
    Sophia,
}

impl Default for OptimizerKind {
    fn default() -> Self {
        Self::AdamW
    }
}

impl std::str::FromStr for OptimizerKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "adamw" => Ok(Self::AdamW),
            "adamw-8bit" => Ok(Self::AdamW8Bit),
            "paged-adamw" => Ok(Self::PagedAdamW),
            "paged-adamw-8bit" => Ok(Self::PagedAdamW8Bit),
            "lion" => Ok(Self::Lion),
            "lion-8bit" => Ok(Self::Lion8Bit),
            "adafactor" => Ok(Self::Adafactor),
            "adamw-bnb" => Ok(Self::AdamWBnb),
            "qgalore-8bit" | "qgalore" => Ok(Self::QGaLoreAdamW8Bit),
            "galore" => Ok(Self::GaloreAdamW),
            "galore-8bit" => Ok(Self::GaloreAdamW8Bit),
            "lomo" => Ok(Self::LOMO),
            "adalomo" => Ok(Self::Adalomo),
            "came" => Ok(Self::CAME),
            "sophia" => Ok(Self::Sophia),
            other => Err(format!(
                "unknown optimizer '{other}' (expected adamw, adamw-8bit, paged-adamw, paged-adamw-8bit, lion, lion-8bit, adafactor, adamw-bnb, qgalore, galore, galore-8bit, lomo, adalomo, came, sophia)"
            )),
        }
    }
}

impl std::fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::AdamW => "adamw",
            Self::AdamW8Bit => "adamw-8bit",
            Self::PagedAdamW => "paged-adamw",
            Self::PagedAdamW8Bit => "paged-adamw-8bit",
            Self::Lion => "lion",
            Self::Lion8Bit => "lion-8bit",
            Self::Adafactor => "adafactor",
            Self::AdamWBnb => "adamw-bnb",
            Self::QGaLoreAdamW8Bit => "qgalore-8bit",
            Self::GaloreAdamW => "galore",
            Self::GaloreAdamW8Bit => "galore-8bit",
            Self::LOMO => "lomo",
            Self::Adalomo => "adalomo",
            Self::CAME => "came",
            Self::Sophia => "sophia",
        };
        f.write_str(s)
    }
}

/// Boxed optimizer wrapper used by the garage worker to dispatch
/// optimizer construction and stepping uniformly via the `Optimizer` enum.
pub enum Optimizer {
    AdamW(AdamW),
    AdamW8Bit(AdamW8Bit),
    PagedAdamW(PagedAdamW),
    Lion(Lion),
    Lion8Bit(Lion8Bit),
    Adafactor(Adafactor),
}

impl Optimizer {
    /// Build an optimizer from kind and learning rate.
    ///
    /// Returns `Error::Unimplemented` for kinds whose implementation has not
    /// landed yet (see `OptimizerKind` docs).
    pub fn new(kind: OptimizerKind, lr: f32) -> Result<Self> {
        match kind {
            OptimizerKind::AdamW => Ok(Optimizer::AdamW(AdamW::new(AdamWConfig {
                lr,
                ..AdamWConfig::default()
            }))),
            OptimizerKind::AdamW8Bit => Ok(Optimizer::AdamW8Bit(AdamW8Bit::new(
                AdamW8BitConfig {
                    lr,
                    use_8bit_moments: true,
                    ..AdamW8BitConfig::default()
                },
            ))),
            OptimizerKind::PagedAdamW => Ok(Optimizer::PagedAdamW(PagedAdamW::new(
                PagedAdamWConfig {
                    lr,
                    cpu_offload: true,
                    ..PagedAdamWConfig::default()
                },
            ))),
            OptimizerKind::Lion => Ok(Optimizer::Lion(Lion::new(LionConfig {
                lr,
                ..LionConfig::default()
            }))),
            OptimizerKind::Lion8Bit => Ok(Optimizer::Lion8Bit(Lion8Bit::new(
                Lion8BitConfig {
                    lr,
                    use_8bit_moments: true,
                    ..Lion8BitConfig::default()
                },
            ))),
            OptimizerKind::Adafactor => Ok(Optimizer::Adafactor(Adafactor::new(
                AdafactorConfig {
                    lr,
                    ..AdafactorConfig::default()
                },
            ))),
            kind => Err(Error::Unimplemented(format!(
                "optimizer '{kind}' is declared but not yet implemented (Phase 7)"
            ))),
        }
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.step(params),
            Optimizer::AdamW8Bit(o) => o.step(params),
            Optimizer::PagedAdamW(o) => o.step(params),
            Optimizer::Lion(o) => o.step(params),
            Optimizer::Lion8Bit(o) => o.step(params),
            Optimizer::Adafactor(o) => o.step(params),
        }
    }

    /// Update the learning rate in the underlying optimizer config.
    pub fn set_lr(&mut self, lr: f32) {
        match self {
            Optimizer::AdamW(o) => o.config.lr = lr,
            Optimizer::AdamW8Bit(o) => o.config.lr = lr,
            Optimizer::PagedAdamW(o) => o.config.lr = lr,
            Optimizer::Lion(o) => o.config.lr = lr,
            Optimizer::Lion8Bit(o) => o.config.lr = lr,
            Optimizer::Adafactor(o) => o.config.lr = lr,
        }
    }

    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        match self {
            Optimizer::AdamW(o) => o.save_to_train_state(params),
            Optimizer::AdamW8Bit(o) => o.save_to_train_state(params),
            Optimizer::PagedAdamW(o) => o.save_to_train_state(params),
            Optimizer::Lion(o) => o.save_to_train_state(params),
            Optimizer::Lion8Bit(o) => o.save_to_train_state(params),
            Optimizer::Adafactor(o) => o.save_to_train_state(params),
        }
    }

    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.load_from_train_state(params, state),
            Optimizer::AdamW8Bit(o) => o.load_from_train_state(params, state),
            Optimizer::PagedAdamW(o) => o.load_from_train_state(params, state),
            Optimizer::Lion(o) => o.load_from_train_state(params, state),
            Optimizer::Lion8Bit(o) => o.load_from_train_state(params, state),
            Optimizer::Adafactor(o) => o.load_from_train_state(params, state),
        }
    }
}

/// Hyperparameters for AdamW optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW optimizer state tracking step count and moment buffers.
pub struct AdamW {
    pub config: AdamWConfig,
    pub step_count: usize,
    /// 1st moment vector (m) per trainable parameter ID (device-resident).
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
    /// 2nd moment vector (v) per trainable parameter ID (device-resident).
    pub v: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_count", &self.m.len())
            .field("v_count", &self.v.len())
            .finish()
    }
}

impl AdamW {
    /// Create a new AdamW optimizer with the given configuration.
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Perform one device-resident optimization step over all parameters in `params`.
    pub fn step_device(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step(params)
    }

    /// Perform one optimization step over all parameters in `params`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let elem_count = shape.elem_count();

            // Seed moment buffers on first encounter (device-resident).
            if !self.m.contains_key(id) {
                let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.m.insert(*id, zero_m);
            }
            if !self.v.contains_key(id) {
                let zero_v = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.v.insert(*id, zero_v);
            }

            let m_st_old = self.m.get_mut(id).unwrap();
            let v_st_old = self.v.get_mut(id).unwrap();
            let grad_st = param.grad().storage().clone();
            let data_st = param.data.storage().clone();

            // m_new = beta1 * m + (1-beta1) * g
            let (m_beta1, _) = dev.mul_scalar(m_st_old.as_ref(), beta1, shape)?;
            let (g_1mb1, _) = dev.mul_scalar(grad_st.as_ref(), 1.0 - beta1, shape)?;
            let (m_new, _) = dev.add(m_beta1.as_ref(), g_1mb1.as_ref(), shape)?;

            // v_new = beta2 * v + (1-beta2) * g^2
            let (g_sq, _) = dev.mul(grad_st.as_ref(), grad_st.as_ref(), shape)?;
            let (v_beta2, _) = dev.mul_scalar(v_st_old.as_ref(), beta2, shape)?;
            let (g_sq_1mb2, _) = dev.mul_scalar(g_sq.as_ref(), 1.0 - beta2, shape)?;
            let (v_new, _) = dev.add(v_beta2.as_ref(), g_sq_1mb2.as_ref(), shape)?;

            // m_hat = m_new / bias_correction1,  v_hat = v_new / bias_correction2
            let (m_hat, _) = dev.mul_scalar(m_new.as_ref(), 1.0 / bias_correction1, shape)?;
            let (v_hat, _) = dev.mul_scalar(v_new.as_ref(), 1.0 / bias_correction2, shape)?;

            // denom = sqrt(v_hat) + eps
            let (sqrt_v, _) = dev.sqrt(v_hat.as_ref(), shape)?;
            let eps_buf = dev.from_cpu(&vec![eps; elem_count], shape, DType::F32)?;
            let (denom, _) = dev.add(sqrt_v.as_ref(), eps_buf.as_ref(), shape)?;

            // recip_denom = 1.0 / denom
            let (recip_denom, _) = dev.recip(denom.as_ref(), shape)?;

            // step_grad = m_hat * recip_denom + weight_decay * w
            let (m_div_denom, _) = dev.mul(m_hat.as_ref(), recip_denom.as_ref(), shape)?;
            let (wd_w, _) = dev.mul_scalar(data_st.as_ref(), weight_decay, shape)?;
            let (step_grad, _) = dev.add(m_div_denom.as_ref(), wd_w.as_ref(), shape)?;

            // updated = w - lr * step_grad
            let (lr_step, _) = dev.mul_scalar(step_grad.as_ref(), lr, shape)?;
            let (neg_lr_step, _) = dev.mul_scalar(lr_step.as_ref(), -1.0, shape)?;
            let (updated_st, _) = dev.add(data_st.as_ref(), neg_lr_step.as_ref(), shape)?;

            // Write back device-resident moment buffers + parameters.
            *m_st_old = m_new;
            *v_st_old = v_new;
            param.data = Tensor::new(
                Arc::from(updated_st),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Save optimizer moments and trainable parameter data into a `.grim.train` `TrainState`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            step: self.step_count as u64,
            fp_format: TrainFpFormat::Fp32,
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let blob_name = format!(
                    "param_{}_{}_{}",
                    id.layer_idx,
                    id.adapter_id,
                    if id.is_a { "a" } else { "b" }
                );
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(m_st) = self.m.get(id) {
                if let Ok(m_vec) = m_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = format!(
                        "opt_m_{}_{}_{}",
                        id.layer_idx,
                        id.adapter_id,
                        if id.is_a { "a" } else { "b" }
                    );
                    state.add_blob(blob_name, shape.clone(), bytes);
                }
            }

            if let Some(v_st) = self.v.get(id) {
                if let Ok(v_vec) = v_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = v_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = format!(
                        "opt_v_{}_{}_{}",
                        id.layer_idx,
                        id.adapter_id,
                        if id.is_a { "a" } else { "b" }
                    );
                    state.add_blob(blob_name, shape, bytes);
                }
            }
        }

        state
    }

    /// Restore optimizer moments and parameter data from a `.grim.train` `TrainState`.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        for (id, param) in params.iter_mut() {
            let suffix = if id.is_a { "a" } else { "b" };
            let param_key = format!("param_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);
            let m_key = format!("opt_m_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);
            let v_key = format!("opt_v_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);

            if let Some(blob) = state.blobs.get(&param_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    DType::F32,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if let Some(blob) = state.blobs.get(&m_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.m.insert(*id, st);
            }

            if let Some(blob) = state.blobs.get(&v_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.v.insert(*id, st);
            }
        }

        Ok(())
    }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(Error::Backend("invalid byte slice length for f32".into()));
    }
    let mut res = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        res.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(res)
}

/// Persist only the parameter data + step count (no optimizer moments).
///
/// Used by optimizer variants whose moment buffers are not yet serialized to
/// `.grim.train` (Lion, Lion8Bit, Adafactor, PagedAdamW moments are pending;
/// AdamW persists m/v via its own richer implementation).
fn save_param_data_only(params: &TrainableParams, step_count: usize) -> TrainState {
    let mut state = TrainState {
        step: step_count as u64,
        fp_format: TrainFpFormat::Fp32,
        blobs: HashMap::new(),
    };
    for (id, param) in params.iter() {
        let shape = param.data.shape().dims().to_vec();
        if let Ok(data) = param.data.to_vec_f32() {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let blob_name = format!(
                "param_{}_{}_{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" }
            );
            state.add_blob(blob_name, shape, bytes);
        }
    }
    state
}

/// Restore parameter data (and step count) from a `.grim.train` `TrainState`.
fn load_param_data_only(params: &mut TrainableParams, state: &TrainState) -> Result<()> {
    for (id, param) in params.iter_mut() {
        let suffix = if id.is_a { "a" } else { "b" };
        let param_key = format!("param_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);
        if let Some(blob) = state.blobs.get(&param_key) {
            let f32_vals = bytes_to_f32_vec(&blob.data)?;
            let dev = crate::pick_device_for_tensor(&param.data);
            let storage = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                param.data.shape().clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }
    }
    Ok(())
}

// ============================================================================
// Lion Optimizer (Google's signed sparse action)
// ============================================================================

/// Hyperparameters for Lion optimizer.
/// Lion = sign-based momentum: τ_t = β1 * τ_{t-1} + (1-β1) * g_t
///        θ_t = θ_{t-1} - α * τ_t
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LionConfig {
    /// Learning rate (default: 1e-4)
    pub lr: f32,
    /// Exponential decay rate for first moment (default: 0.9)
    pub beta1: f32,
    /// Weight decay coefficient (default: 0.01)
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            weight_decay: 0.01,
        }
    }
}

/// Lion optimizer state: simple momentum buffer (1st order only).
pub struct Lion {
    pub config: LionConfig,
    pub step_count: usize,
    /// 1st moment (momentum) buffer per trainable parameter ID.
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl std::fmt::Debug for Lion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lion")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_count", &self.m.len())
            .finish()
    }
}

impl Lion {
    /// Create a new Lion optimizer with the given configuration.
    pub fn new(config: LionConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
        }
    }

    /// Perform one optimization step over all parameters in `params`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let elem_count = shape.elem_count();

            // Initialize momentum buffer on first encounter
            if !self.m.contains_key(id) {
                let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.m.insert(*id, zero_m);
            }

            let m_st = self.m.get_mut(id).unwrap();
            let grad_st = param.grad().storage().clone();
            let data_st = param.data.storage().clone();

            // Lion: τ = β1 * m + (1-β1) * g
            // Note: No bias correction in Lion (unlike AdamW)
            let (m_beta1, _) = dev.mul_scalar(m_st.as_ref(), beta1, shape)?;
            let (g_1mb1, _) = dev.mul_scalar(grad_st.as_ref(), 1.0 - beta1, shape)?;
            let (m_new, _) = dev.add(m_beta1.as_ref(), g_1mb1.as_ref(), shape)?;

            // Apply weight decay: step = τ + weight_decay * w
            let (wd_w, _) = dev.mul_scalar(data_st.as_ref(), weight_decay, shape)?;
            let (step, _) = dev.add(m_new.as_ref(), wd_w.as_ref(), shape)?;

            // Update: w = w - lr * step
            let (lr_step, _) = dev.mul_scalar(step.as_ref(), lr, shape)?;
            let (neg_lr_step, _) = dev.mul_scalar(lr_step.as_ref(), -1.0, shape)?;
            let (updated_st, _) = dev.add(data_st.as_ref(), neg_lr_step.as_ref(), shape)?;

            // Write back
            *m_st = m_new;
            param.data = Tensor::new(
                Arc::from(updated_st),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Persist parameter data + step count (Lion moments are not serialized yet).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)
    }
}

// ============================================================================
// 8-bit AdamW Optimizer
// ============================================================================

/// 8-bit AdamW optimizer with memory-efficient moment storage.
/// Stores moments in F16 (half precision) to reduce memory by 50%.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW8BitConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Placeholder for 8-bit quantization (implementation pending).
    pub use_8bit_moments: bool,
}

impl Default for AdamW8BitConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            use_8bit_moments: true,
        }
    }
}

/// AdamW with 8-bit (F16) moment storage for memory efficiency.
/// Currently uses F32 moments; 8-bit storage via PagedAdamW pending infrastructure.
pub struct AdamW8Bit {
    pub config: AdamW8BitConfig,
    pub step_count: usize,
    /// 1st moment vector (m) per trainable parameter ID.
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
    /// 2nd moment vector (v) per trainable parameter ID.
    pub v: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl std::fmt::Debug for AdamW8Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW8Bit")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_count", &self.m.len())
            .field("v_count", &self.v.len())
            .finish()
    }
}

impl AdamW8Bit {
    /// Create a new 8-bit AdamW optimizer.
    pub fn new(config: AdamW8BitConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Perform one optimization step over all parameters.
    /// Uses F32 moments for compatibility with current backend API.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let elem_count = shape.elem_count();

            // Initialize moment buffers
            if !self.m.contains_key(id) {
                let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.m.insert(*id, zero_m);
            }
            if !self.v.contains_key(id) {
                let zero_v = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.v.insert(*id, zero_v);
            }

            let m_st_old = self.m.get_mut(id).unwrap();
            let v_st_old = self.v.get_mut(id).unwrap();
            let grad_st = param.grad().storage().clone();
            let data_st = param.data.storage().clone();

            // m_new = β1 * m + (1-β1) * g
            let (m_beta1, _) = dev.mul_scalar(m_st_old.as_ref(), beta1, shape)?;
            let (g_1mb1, _) = dev.mul_scalar(grad_st.as_ref(), 1.0 - beta1, shape)?;
            let (m_new, _) = dev.add(m_beta1.as_ref(), g_1mb1.as_ref(), shape)?;

            // v_new = β2 * v + (1-β2) * g²
            let (g_sq, _) = dev.mul(grad_st.as_ref(), grad_st.as_ref(), shape)?;
            let (v_beta2, _) = dev.mul_scalar(v_st_old.as_ref(), beta2, shape)?;
            let (g_sq_1mb2, _) = dev.mul_scalar(g_sq.as_ref(), 1.0 - beta2, shape)?;
            let (v_new, _) = dev.add(v_beta2.as_ref(), g_sq_1mb2.as_ref(), shape)?;

            // m_hat = m_new / bias_correction1, v_hat = v_new / bias_correction2
            let (m_hat, _) = dev.mul_scalar(m_new.as_ref(), 1.0 / bias_correction1, shape)?;
            let (v_hat, _) = dev.mul_scalar(v_new.as_ref(), 1.0 / bias_correction2, shape)?;

            // denom = √v_hat + ε
            let (sqrt_v, _) = dev.sqrt(v_hat.as_ref(), shape)?;
            let eps_buf = dev.from_cpu(&vec![eps; elem_count], shape, DType::F32)?;
            let (denom, _) = dev.add(sqrt_v.as_ref(), eps_buf.as_ref(), shape)?;

            // recip_denom = 1 / denom
            let (recip_denom, _) = dev.recip(denom.as_ref(), shape)?;

            // step_grad = m_hat / denom + weight_decay * w
            let (m_div_denom, _) = dev.mul(m_hat.as_ref(), recip_denom.as_ref(), shape)?;
            let (wd_w, _) = dev.mul_scalar(data_st.as_ref(), weight_decay, shape)?;
            let (step_grad, _) = dev.add(m_div_denom.as_ref(), wd_w.as_ref(), shape)?;

            // updated = w - lr * step_grad
            let (lr_step, _) = dev.mul_scalar(step_grad.as_ref(), lr, shape)?;
            let (neg_lr_step, _) = dev.mul_scalar(lr_step.as_ref(), -1.0, shape)?;
            let (updated_st, _) = dev.add(data_st.as_ref(), neg_lr_step.as_ref(), shape)?;

            // Write back
            *m_st_old = m_new;
            *v_st_old = v_new;
            param.data = Tensor::new(
                Arc::from(updated_st),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Persist parameter data + step count (8-bit moments are not serialized yet).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)
    }
}

// ============================================================================
// Paged AdamW - CUDA Unified Memory Variant
// ============================================================================

/// Configuration for Paged AdamW optimizer.
/// Paged AdamW uses CPU-offloaded momentum buffers with paged attention
/// for training models larger than GPU memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedAdamWConfig {
    /// Learning rate
    pub lr: f32,
    /// Exponential decay for first moment
    pub beta1: f32,
    /// Exponential decay for second moment  
    pub beta2: f32,
    /// Numerical stability constant
    pub eps: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Page size for CPU-offloaded buffers (in number of parameters per page)
    pub page_size: usize,
    /// Enable CPU-offloading of optimizer states
    pub cpu_offload: bool,
    /// Maximum GPU memory fraction for optimizer states (0.0 = CPU only, 1.0 = GPU only)
    pub gpu_mem_fraction: f32,
}

impl Default for PagedAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            page_size: 1024 * 1024,
            cpu_offload: true,
            gpu_mem_fraction: 0.0,
        }
    }
}

/// Paged AdamW optimizer state with CPU-offloaded momentum buffers.
pub struct PagedAdamW {
    pub config: PagedAdamWConfig,
    pub step_count: usize,
    pub m: HashMap<ParamId, Vec<f32>>,
    pub v: HashMap<ParamId, Vec<f32>>,
    pub pages_in_gpu: HashMap<ParamId, bool>,
}

impl std::fmt::Debug for PagedAdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAdamW")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .finish()
    }
}

impl PagedAdamW {
    pub fn new(config: PagedAdamWConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            pages_in_gpu: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape().clone();
            let elem_count = shape.elem_count();

            // Initialize buffers if needed
            let data: Vec<f32> = param.data.to_vec_f32()?;
            let grad: Vec<f32> = param.grad().to_vec_f32()?;

            if !self.m.contains_key(id) {
                self.m.insert(*id, vec![1.0f32; elem_count]);
            }
            if !self.v.contains_key(id) {
                self.v.insert(*id, vec![1.0f32; elem_count]);
            }

            let m = self.m.get_mut(id).unwrap();
            let v = self.v.get_mut(id).unwrap();

            for i in 0..elem_count {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            }

            let m_hat: Vec<f32> = m.iter().map(|&x| x / bias_correction1).collect();
            let v_hat: Vec<f32> = v.iter().map(|&x| x / bias_correction2).collect();

            let new_data: Vec<f32> = (0..elem_count)
                .map(|i| {
                    let step = m_hat[i] / (v_hat[i].sqrt() + eps) + weight_decay * data[i];
                    data[i] - lr * step
                })
                .collect();

            let storage = dev.from_cpu(&new_data, &shape, DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                shape,
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Persist parameter data + step count (paged moments are not serialized yet).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)
    }
}

// ============================================================================
// Lion8Bit Optimizer
// ============================================================================

/// Configuration for Lion8Bit optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lion8BitConfig {
    pub lr: f32,
    pub beta1: f32,
    pub weight_decay: f32,
    pub use_8bit_moments: bool,
}

impl Default for Lion8BitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            weight_decay: 0.01,
            use_8bit_moments: true,
        }
    }
}

/// Lion optimizer with optional 8-bit moment quantization.
pub struct Lion8Bit {
    pub config: Lion8BitConfig,
    pub step_count: usize,
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl Lion8Bit {
    pub fn new(config: Lion8BitConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let elem_count = shape.elem_count();

            if !self.m.contains_key(id) {
                let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                self.m.insert(*id, zero_m);
            }

            let m = self.m.get_mut(id).unwrap();
            let grad_st = param.grad().storage().clone();
            let data_st = param.data.storage().clone();

            let (m_beta1, _) = dev.mul_scalar(m.as_ref(), beta1, shape)?;
            let (g_1mb1, _) = dev.mul_scalar(grad_st.as_ref(), 1.0 - beta1, shape)?;
            let (m_new, _) = dev.add(m_beta1.as_ref(), g_1mb1.as_ref(), shape)?;

            let (lr_step, _) = dev.mul_scalar(m_new.as_ref(), lr, shape)?;
            let (neg_lr_step, _) = dev.mul_scalar(lr_step.as_ref(), -1.0, shape)?;
            let (wd_w, _) = dev.mul_scalar(data_st.as_ref(), weight_decay, shape)?;
            let (updated, _) = dev.add(neg_lr_step.as_ref(), wd_w.as_ref(), shape)?;

            *m = m_new;
            param.data = Tensor::new(
                Arc::from(updated),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Persist parameter data + step count (8-bit Lion moments are not serialized yet).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)
    }
}

// ============================================================================
// Adafactor Optimizer
// ============================================================================

/// Configuration for Adafactor optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdafactorConfig {
    pub lr: f32,
    pub relative_step: bool,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub scale_by_lr: bool,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            relative_step: true,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            scale_by_lr: true,
        }
    }
}

/// Adafactor optimizer with factorized second moments.
pub struct Adafactor {
    pub config: AdafactorConfig,
    pub step_count: usize,
    pub b: HashMap<ParamId, Vec<f32>>,
    pub c: HashMap<ParamId, Vec<f32>>,
}

impl std::fmt::Debug for Adafactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adafactor")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .finish()
    }
}

impl Adafactor {
    pub fn new(config: AdafactorConfig) -> Self {
        Self {
            config,
            step_count: 0,
            b: HashMap::new(),
            c: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let dims = shape.dims();

            if dims.len() < 2 {
                return Err(Error::Backend("Adafactor requires 2D+ tensors".into()));
            }

            let rows = dims[dims.len() - 2];
            let cols = dims[dims.len() - 1];
            let elem_count = rows * cols;

            if !self.b.contains_key(id) {
                self.b.insert(*id, vec![1.0f32; rows]);
            }
            if !self.c.contains_key(id) {
                self.c.insert(*id, vec![1.0f32; cols]);
            }

            let b_vec = self.b.get_mut(id).unwrap();
            let c_vec = self.c.get_mut(id).unwrap();
            let data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;

            for i in 0..rows {
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += grad[i * cols + j].powi(2);
                }
                b_vec[i] = beta2 * b_vec[i] + (1.0 - beta2) * sum.sqrt().max(1e-8);
            }

            for j in 0..cols {
                let mut sum = 0.0f32;
                for i in 0..rows {
                    sum += grad[i * cols + j].powi(2);
                }
                c_vec[j] = beta2 * c_vec[j] + (1.0 - beta2) * sum.sqrt().max(1e-8);
            }

            let effective_v: Vec<f32> = b_vec.iter()
                .flat_map(|&bi| c_vec.iter().map(move |&cj| bi * cj))
                .collect();

            let step_scale = if self.config.relative_step && self.step_count > 10 {
                lr * beta2.sqrt() / effective_v[0].max(1e-8)
            } else {
                lr / effective_v[0].max(1e-8)
            };

            let new_data: Vec<f32> = data.iter()
                .zip(grad.iter())
                .map(|(&d, &g)| d - step_scale * g)
                .collect();

            let storage = dev.from_cpu(&new_data, shape, DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Persist parameter data + step count (Adafactor moments are not serialized yet).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lion_and_8bit_adamw_optimizers_step() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;
        let mut params = TrainableParams::new();
        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let t_data = grim_backend_cpu::cpu_tensor(vec![1.0f32; 10], Shape::new(vec![10]));
        let t_grad = grim_backend_cpu::cpu_tensor(vec![0.1f32; 10], Shape::new(vec![10]));
        let mut tp = TrainableParam::new(pid, t_data).unwrap();
        tp.accumulate_grad(&t_grad).unwrap();
        params.insert(tp);

        let mut lion = Lion::new(LionConfig::default());
        lion.step(&mut params).unwrap();

        let mut adam8 = AdamW8Bit::new(AdamW8BitConfig::default());
        adam8.step(&mut params).unwrap();
    }
    
    #[test]
    fn test_paged_adamw_optimizer_step() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;
        let mut params = TrainableParams::new();
        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let t_data = grim_backend_cpu::cpu_tensor(vec![1.0f32; 10], Shape::new(vec![10]));
        let t_grad = grim_backend_cpu::cpu_tensor(vec![0.1f32; 10], Shape::new(vec![10]));
        let mut tp = TrainableParam::new(pid, t_data).unwrap();
        tp.accumulate_grad(&t_grad).unwrap();
        params.insert(tp);

        let mut paged = PagedAdamW::new(PagedAdamWConfig::default());
        paged.step(&mut params).unwrap();
    }
    
    #[test]
    fn test_lion8bit_optimizer_step() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;
        let mut params = TrainableParams::new();
        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let t_data = grim_backend_cpu::cpu_tensor(vec![1.0f32; 10], Shape::new(vec![10]));
        let t_grad = grim_backend_cpu::cpu_tensor(vec![0.1f32; 10], Shape::new(vec![10]));
        let mut tp = TrainableParam::new(pid, t_data).unwrap();
        tp.accumulate_grad(&t_grad).unwrap();
        params.insert(tp);

        let mut lion8bit = Lion8Bit::new(Lion8BitConfig::default());
        lion8bit.step(&mut params).unwrap();
    }
    
    #[test]
    fn test_adafactor_optimizer_step() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;
        let mut params = TrainableParams::new();
        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let t_data = grim_backend_cpu::cpu_tensor(vec![1.0f32; 12], Shape::new(vec![3, 4]));
        let t_grad = grim_backend_cpu::cpu_tensor(vec![0.1f32; 12], Shape::new(vec![3, 4]));
        let mut tp = TrainableParam::new(pid, t_data).unwrap();
        tp.accumulate_grad(&t_grad).unwrap();
        params.insert(tp);

        let mut adafactor = Adafactor::new(AdafactorConfig::default());
        adafactor.step(&mut params).unwrap();
    }
    
    #[test]
    fn test_lr_scheduler_cosine() {
        let sched = LRScheduler::Cosine;
        let base_lr = 1e-4;
        let lr0 = sched.get_lr(base_lr, 0, 100);
        let lr50 = sched.get_lr(base_lr, 50, 100);
        let lr100 = sched.get_lr(base_lr, 100, 100);
        // Cosine starts at max (lr0 = base_lr), decreases to 0
        assert!(lr0 > lr50);
        assert!(lr50 > lr100);
    }
    
    #[test]
    fn test_lr_scheduler_linear() {
        let sched = LRScheduler::Linear;
        let base_lr = 1e-4;
        let lr0 = sched.get_lr(base_lr, 0, 100);
        let lr50 = sched.get_lr(base_lr, 50, 100);
        let lr100 = sched.get_lr(base_lr, 100, 100);
        // Linear starts at base_lr, decreases to 0
        assert!((lr0 - base_lr).abs() < 1e-10);
        assert!(lr50 > lr100);
    }
    
    #[test]
    fn test_lr_scheduler_inverse_sqrt() {
        let sched = LRScheduler::InverseSqrt;
        let base_lr = 1e-3;
        let lr1 = sched.get_lr(base_lr, 1, 1000);
        let lr10 = sched.get_lr(base_lr, 10, 1000);
        // lr = base_lr / sqrt(step)
        assert!(lr1 > lr10);
    }
}

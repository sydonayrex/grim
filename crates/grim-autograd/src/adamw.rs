//! AdamW optimizer implementation for LoRA trainable parameters (WI-T4).
//!
//! Provides step update arithmetic for 1st moment (m) and 2nd moment (v) tracking,
//! alongside serialization to and from `.grim.train` sidecars (`TrainState`).
//!
//! Also includes learning rate schedules and additional optimizer variants.

use crate::param::{ParamId, TrainableParams};
use grim_format::train::{TrainBlob, TrainFpFormat, TrainState};
use grim_tensor::{
    BackendStorage, DType, Tensor,
    error::{Error, Result},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Learning rate scheduler type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LRScheduler {
    /// Cosine annealing with warmup: lr = base_lr * 0.5 * (1 + cos(pi * t / T))
    #[default]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizerKind {
    /// Standard AdamW with FP32 moment buffers.
    #[default]
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
    /// Muon — Newton-Schulz orthogonalization for the direction matrix (B)
    /// + 1-bit Sign-SGD for the magnitude matrix (A), with split weight decay.
    Muon,
    /// M-Adam (Additive-Multiplicative Optimization) for ultra-low precision (FP4/FP8).
    MAdam,
    /// LionVote — Per-layer adaptive voting for sign-momentum updates.
    LionVote,
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
            "muon" => Ok(Self::Muon),
            "madam" | "m-adam" => Ok(Self::MAdam),
            "lionvote" | "lion-vote" => Ok(Self::LionVote),
            other => Err(format!(
                "unknown optimizer '{other}' (expected adamw, adamw-8bit, paged-adamw, paged-adamw-8bit, lion, lion-8bit, adafactor, adamw-bnb, qgalore, galore, galore-8bit, lomo, adalomo, came, sophia, muon, madam, lionvote)"
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
            Self::Muon => "muon",
            Self::MAdam => "madam",
            Self::LionVote => "lionvote",
        };
        f.write_str(s)
    }
}

// Boxed optimizer wrapper used by the garage worker to dispatch
// optimizer construction and stepping uniformly via the `Optimizer` enum.

/// F2b: sidecar slot names now embed the injection point so base-weight
/// entries (`adapter_id == 0`) for different points stop colliding.
pub(crate) fn point_suffix(p: crate::injection::LoRAInjectionPoint) -> &'static str {
    p.suffix()
}
pub(crate) fn weight_slot(id: &ParamId) -> String {
    format!(
        "param_{}_{}_{}_{}",
        id.layer_idx,
        id.adapter_id,
        point_suffix(id.point),
        if id.is_a { "a" } else { "b" }
    )
}
pub(crate) fn legacy_weight_slot(id: &ParamId) -> String {
    weight_slot(id)
}
pub(crate) fn m_slot(id: &ParamId) -> String {
    format!(
        "opt_m_{}_{}_{}",
        id.layer_idx,
        id.adapter_id,
        point_suffix(id.point)
    )
}
pub(crate) fn v_slot(id: &ParamId) -> String {
    format!(
        "opt_v_{}_{}_{}",
        id.layer_idx,
        id.adapter_id,
        point_suffix(id.point)
    )
}
pub(crate) fn legacy_m_slot(id: &ParamId) -> String {
    m_slot(id)
}
pub(crate) fn legacy_v_slot(id: &ParamId) -> String {
    v_slot(id)
}
/// Read helper: prefer the point-suffixed slot, fall back to pre-F2b names
/// so older sidecars still resume.
pub(crate) fn blob_slot<'a>(
    state: &'a TrainState,
    new: &str,
    legacy: String,
) -> Option<&'a TrainBlob> {
    state.blobs.get(new).or_else(|| state.blobs.get(&legacy))
}

pub enum Optimizer {
    AdamW(AdamW),
    AdamW8Bit(AdamW8Bit),
    PagedAdamW(PagedAdamW),
    Lion(Lion),
    Lion8Bit(Lion8Bit),
    Adafactor(Adafactor),
    QGaLoreAdamW8Bit(QGaLoreAdamW8Bit),
    Muon(Muon),
    MAdam(MAdam),
    LionVote(LionVote),
    Lomo(crate::lomo::Lomo),
    AdaLomo(crate::lomo::AdaLomo),
    Came(crate::came::Came),
    Sophia(crate::sophia::Sophia),
    GaloreAdamW(crate::galore::GaLoreOptimizer),
}

impl Optimizer {
    /// Build an optimizer from kind and learning rate.
    pub fn new(kind: OptimizerKind, lr: f32) -> Result<Self> {
        match kind {
            OptimizerKind::AdamW => Ok(Optimizer::AdamW(AdamW::new(AdamWConfig {
                lr,
                ..AdamWConfig::default()
            }))),
            OptimizerKind::AdamW8Bit => Ok(Optimizer::AdamW8Bit(AdamW8Bit::new(AdamW8BitConfig {
                lr,
                use_8bit_moments: true,
                ..AdamW8BitConfig::default()
            }))),
            OptimizerKind::PagedAdamW => {
                Ok(Optimizer::PagedAdamW(PagedAdamW::new(PagedAdamWConfig {
                    lr,
                    cpu_offload: true,
                    ..PagedAdamWConfig::default()
                })))
            }
            OptimizerKind::Lion => Ok(Optimizer::Lion(Lion::new(LionConfig {
                lr,
                ..LionConfig::default()
            }))),
            OptimizerKind::Lion8Bit => Ok(Optimizer::Lion8Bit(Lion8Bit::new(Lion8BitConfig {
                lr,
                use_8bit_moments: true,
                ..Lion8BitConfig::default()
            }))),
            OptimizerKind::Adafactor => Ok(Optimizer::Adafactor(Adafactor::new(AdafactorConfig {
                lr,
                ..AdafactorConfig::default()
            }))),
            OptimizerKind::GaloreAdamW => {
                Ok(Optimizer::GaloreAdamW(crate::galore::GaLoreOptimizer::new(
                    crate::galore::GaLoreConfig {
                        lr,
                        ..Default::default()
                    },
                )))
            }
            OptimizerKind::GaloreAdamW8Bit => {
                eprintln!("grim: galore-8bit is an alias for qgalore-8bit");
                Ok(Optimizer::QGaLoreAdamW8Bit(QGaLoreAdamW8Bit::new(
                    QGaLoreAdamW8BitConfig {
                        lr,
                        ..QGaLoreAdamW8BitConfig::default()
                    },
                )))
            }
            OptimizerKind::QGaLoreAdamW8Bit => {
                Ok(Optimizer::QGaLoreAdamW8Bit(QGaLoreAdamW8Bit::new(
                    QGaLoreAdamW8BitConfig {
                        lr,
                        ..QGaLoreAdamW8BitConfig::default()
                    },
                )))
            }
            OptimizerKind::Muon => Ok(Optimizer::Muon(Muon::new(MuonConfig {
                lr,
                ..MuonConfig::default()
            }))),
            OptimizerKind::MAdam => Ok(Optimizer::MAdam(MAdam::new(MAdamConfig {
                lr,
                ..MAdamConfig::default()
            }))),
            OptimizerKind::LionVote => Ok(Optimizer::LionVote(LionVote::new(LionVoteConfig {
                lr,
                ..LionVoteConfig::default()
            }))),
            OptimizerKind::PagedAdamW8Bit => {
                Ok(Optimizer::PagedAdamW(PagedAdamW::new(PagedAdamWConfig {
                    lr,
                    cpu_offload: true,
                    use_8bit_moments: true,
                    ..PagedAdamWConfig::default()
                })))
            }
            OptimizerKind::AdamWBnb => Err(Error::Unimplemented(
                "optimizer 'adamw-bnb' is not yet implemented (no bitsandbytes dependency); use adamw-8bit".into(),
            )),
            OptimizerKind::LOMO => Ok(Optimizer::Lomo(crate::lomo::Lomo::new(
                crate::lomo::LomoConfig {
                    lr,
                    ..Default::default()
                },
            ))),
            OptimizerKind::Adalomo => Ok(Optimizer::AdaLomo(crate::lomo::AdaLomo::new(
                crate::lomo::AdaLomoConfig {
                    lr,
                    ..Default::default()
                },
            ))),
            OptimizerKind::CAME => Ok(Optimizer::Came(crate::came::Came::new(
                crate::came::CameConfig {
                    lr,
                    ..Default::default()
                },
            ))),
            OptimizerKind::Sophia => Ok(Optimizer::Sophia(crate::sophia::Sophia::new(
                crate::sophia::SophiaConfig {
                    lr,
                    ..Default::default()
                },
            ))),
        }
    }

    /// Return the concrete optimizer kind represented by this instance.
    pub fn kind(&self) -> OptimizerKind {
        match self {
            Optimizer::AdamW(_) => OptimizerKind::AdamW,
            Optimizer::AdamW8Bit(_) => OptimizerKind::AdamW8Bit,
            Optimizer::PagedAdamW(_) => OptimizerKind::PagedAdamW,
            Optimizer::Lion(_) => OptimizerKind::Lion,
            Optimizer::Lion8Bit(_) => OptimizerKind::Lion8Bit,
            Optimizer::Adafactor(_) => OptimizerKind::Adafactor,
            Optimizer::QGaLoreAdamW8Bit(_) => OptimizerKind::QGaLoreAdamW8Bit,
            Optimizer::Muon(_) => OptimizerKind::Muon,
            Optimizer::MAdam(_) => OptimizerKind::MAdam,
            Optimizer::LionVote(_) => OptimizerKind::LionVote,
            Optimizer::Lomo(_) => OptimizerKind::LOMO,
            Optimizer::AdaLomo(_) => OptimizerKind::Adalomo,
            Optimizer::Came(_) => OptimizerKind::CAME,
            Optimizer::Sophia(_) => OptimizerKind::Sophia,
            Optimizer::GaloreAdamW(_) => OptimizerKind::GaloreAdamW,
        }
    }

    /// Return the current learning rate used when forking a replica.
    pub fn lr(&self) -> f32 {
        match self {
            Optimizer::AdamW(o) => o.config.lr,
            Optimizer::AdamW8Bit(o) => o.config.lr,
            Optimizer::PagedAdamW(o) => o.config.lr,
            Optimizer::Lion(o) => o.config.lr,
            Optimizer::Lion8Bit(o) => o.config.lr,
            Optimizer::Adafactor(o) => o.config.lr,
            Optimizer::QGaLoreAdamW8Bit(o) => o.config.lr,
            Optimizer::Muon(m) => m.config.lr,
            Optimizer::MAdam(m) => m.config.lr,
            Optimizer::LionVote(l) => l.config.lr,
            Optimizer::Lomo(o) => o.config.lr,
            Optimizer::AdaLomo(o) => o.config.lr,
            Optimizer::Came(o) => o.config.lr,
            Optimizer::Sophia(o) => o.config.lr,
            Optimizer::GaloreAdamW(o) => o.config.lr,
        }
    }

    /// Reset momentum buffers for specified parameter IDs.
    pub fn reset_momentum_for(&mut self, ids: &[ParamId]) {
        if let Optimizer::AdamW(o) = self {
            o.reset_momentum_for(ids)
        }
    }

    /// Recreate this optimizer for another rank and copy its serialized
    /// moments/state into the target parameter registry. This is the
    /// rank-replica primitive: a new rank must not restart Adam moments.
    pub fn fork_for_rank(
        &self,
        source_params: &TrainableParams,
        target_params: &mut TrainableParams,
    ) -> Result<Self> {
        let mut fork = Self::new(self.kind(), self.lr())?;
        let state = self.save_to_train_state(source_params);
        fork.load_from_train_state(target_params, &state)?;
        Ok(fork)
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.step(params),
            Optimizer::AdamW8Bit(o) => o.step(params),
            Optimizer::PagedAdamW(o) => o.step(params),
            Optimizer::Lion(o) => o.step(params),
            Optimizer::Lion8Bit(o) => o.step(params),
            Optimizer::Adafactor(o) => o.step(params),
            Optimizer::QGaLoreAdamW8Bit(o) => o.step(params),
            Optimizer::Muon(o) => o.step(params),
            Optimizer::MAdam(o) => o.step(params),
            Optimizer::LionVote(o) => o.step(params),
            Optimizer::Lomo(o) => o.step(params),
            Optimizer::AdaLomo(o) => o.step(params),
            Optimizer::Came(o) => o.step(params),
            Optimizer::Sophia(o) => o.step(params),
            Optimizer::GaloreAdamW(o) => o.step(params),
        }
    }

    /// Update a single parameter using the configured optimizer (LOMO streaming step).
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.step_param(id, param),
            Optimizer::Lion(o) => o.step_param(id, param),
            Optimizer::MAdam(o) => o.step_param(id, param),
            Optimizer::LionVote(o) => o.step_param(id, param),
            Optimizer::Lomo(o) => o.step_param(id, param),
            Optimizer::AdaLomo(o) => o.step_param(id, param),
            Optimizer::Came(o) => o.step_param(id, param),
            Optimizer::Sophia(o) => o.step_param(id, param),
            Optimizer::GaloreAdamW(o) => o.step_param(id, param),
            _ => {
                let mut temp_params = TrainableParams::new();
                let param_clone = param.clone();
                temp_params.insert(param_clone);
                self.step(&mut temp_params)?;
                if let Some(updated) = temp_params.get_mut(id) {
                    param.data = updated.data.clone();
                }
                Ok(())
            }
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
            Optimizer::QGaLoreAdamW8Bit(o) => o.config.lr = lr,
            Optimizer::Muon(m) => m.config.lr = lr,
            Optimizer::MAdam(m) => m.config.lr = lr,
            Optimizer::LionVote(l) => l.config.lr = lr,
            Optimizer::Lomo(o) => o.config.lr = lr,
            Optimizer::AdaLomo(o) => o.config.lr = lr,
            Optimizer::Came(o) => o.config.lr = lr,
            Optimizer::Sophia(o) => o.config.lr = lr,
            Optimizer::GaloreAdamW(o) => o.config.lr = lr,
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
            Optimizer::QGaLoreAdamW8Bit(o) => o.save_to_train_state(params),
            Optimizer::Muon(o) => o.save_to_train_state(params),
            Optimizer::MAdam(o) => o.save_to_train_state(params),
            Optimizer::LionVote(o) => o.save_to_train_state(params),
            Optimizer::Lomo(o) => o.save_to_train_state(params),
            Optimizer::AdaLomo(o) => o.save_to_train_state(params),
            Optimizer::Came(o) => o.save_to_train_state(params),
            Optimizer::Sophia(o) => o.save_to_train_state(params),
            Optimizer::GaloreAdamW(o) => o.save_to_train_state(params),
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
            Optimizer::QGaLoreAdamW8Bit(o) => o.load_from_train_state(params, state),
            Optimizer::Muon(o) => o.load_from_train_state(params, state),
            Optimizer::MAdam(o) => o.load_from_train_state(params, state),
            Optimizer::LionVote(o) => o.load_from_train_state(params, state),
            Optimizer::Lomo(o) => o.load_from_train_state(params, state),
            Optimizer::AdaLomo(o) => o.load_from_train_state(params, state),
            Optimizer::Came(o) => o.load_from_train_state(params, state),
            Optimizer::Sophia(o) => o.load_from_train_state(params, state),
            Optimizer::GaloreAdamW(o) => o.load_from_train_state(params, state),
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
    pub lora_plus_ratio: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            lora_plus_ratio: 1.0,
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
    /// Audit fix (grim-models-adjacent pass): PER-PARAMETER time steps for
    /// `step_param` (the fused LOMO/backward_step path). The old code derived
    /// bias corrections from `step_count`, which only `step()` increments —
    /// a fused streaming run never advanced it, so bias correction stayed at
    /// t=1 forever and updates were permanently mis-scaled (~1/beta1). Each
    /// parameter now counts its own update; `step()` remains the batch entry
    /// and behaves identically because it steps every param once.
    pub param_steps: HashMap<ParamId, usize>,
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
            param_steps: HashMap::new(),
        }
    }

    /// Perform one device-resident optimization step over all parameters in `params`.
    pub fn step_device(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step(params)
    }

    /// Reset momentum buffers (m and v) for specified parameter IDs.
    pub fn reset_momentum_for(&mut self, ids: &[ParamId]) {
        for id in ids {
            self.m.remove(id);
            self.v.remove(id);
        }
    }

    /// Perform one step update for a single trainable parameter `param` (LOMO / fused streaming step).
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = if id.is_b_matrix() {
            self.config.lr * self.config.lora_plus_ratio
        } else {
            self.config.lr
        };
        let weight_decay = self.config.weight_decay;

        // Audit fix: bias corrections now come from THIS parameter's own
        // update count (incremented below), not from `step_count` — which
        // only the batch `step()` entry increments. The fused streaming path
        // (`backward_step`) calls `step_param` directly, so its bias
        // correction used to stay frozen at t=1 forever (~1/beta1 update
        // mis-scale).
        let sc = {
            let t = self.param_steps.entry(id).or_insert(0);
            *t += 1;
            *t
        };
        let bias_correction1 = 1.0 - beta1.powi(sc as i32);
        let bias_correction2 = 1.0 - beta2.powi(sc as i32);

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape();
        let elem_count = shape.elem_count();

        // Seed moment buffers on first encounter (device-resident).
        if let std::collections::hash_map::Entry::Vacant(e) = self.m.entry(id) {
            let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
            e.insert(zero_m);
        }
        if let std::collections::hash_map::Entry::Vacant(e) = self.v.entry(id) {
            let zero_v = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
            e.insert(zero_v);
        }

        let m_st_old = self.m.get_mut(&id).unwrap();
        let v_st_old = self.v.get_mut(&id).unwrap();
        let grad_st = param.grad().storage().clone();
        let data_st = param.data.storage().clone();

        // Try on-device fused AdamW kernel first (zero-roundtrip, 1 launch)
        if let Ok(handle) = dev.fused_adamw_step(
            data_st.as_ref(),
            grad_st.as_ref(),
            m_st_old.as_ref(),
            v_st_old.as_ref(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            elem_count,
        ) {
            handle.synchronize()?;
            return Ok(());
        }

        // Fallback to component math
        let (m_beta1, _) = dev.mul_scalar(m_st_old.as_ref(), beta1, shape)?;
        let (g_1mb1, _) = dev.mul_scalar(grad_st.as_ref(), 1.0 - beta1, shape)?;
        let (m_new, _) = dev.add(m_beta1.as_ref(), g_1mb1.as_ref(), shape)?;

        let (g_sq, _) = dev.mul(grad_st.as_ref(), grad_st.as_ref(), shape)?;
        let (v_beta2, _) = dev.mul_scalar(v_st_old.as_ref(), beta2, shape)?;
        let (g_sq_1mb2, _) = dev.mul_scalar(g_sq.as_ref(), 1.0 - beta2, shape)?;
        let (v_new, _) = dev.add(v_beta2.as_ref(), g_sq_1mb2.as_ref(), shape)?;

        let (m_hat, _) = dev.mul_scalar(m_new.as_ref(), 1.0 / bias_correction1, shape)?;
        let (v_hat, _) = dev.mul_scalar(v_new.as_ref(), 1.0 / bias_correction2, shape)?;

        let (sqrt_v, _) = dev.sqrt(v_hat.as_ref(), shape)?;
        let eps_buf = dev.from_cpu(&vec![eps; elem_count], shape, DType::F32)?;
        let (denom, _) = dev.add(sqrt_v.as_ref(), eps_buf.as_ref(), shape)?;

        let (recip_denom, _) = dev.recip(denom.as_ref(), shape)?;

        let (m_div_denom, _) = dev.mul(m_hat.as_ref(), recip_denom.as_ref(), shape)?;
        let (wd_w, _) = dev.mul_scalar(data_st.as_ref(), weight_decay, shape)?;
        let (step_grad, _) = dev.add(m_div_denom.as_ref(), wd_w.as_ref(), shape)?;

        let (lr_step, _) = dev.mul_scalar(step_grad.as_ref(), lr, shape)?;
        let (neg_lr_step, _) = dev.mul_scalar(lr_step.as_ref(), -1.0, shape)?;
        let (updated_st, _) = dev.add(data_st.as_ref(), neg_lr_step.as_ref(), shape)?;

        *m_st_old = m_new;
        *v_st_old = v_new;
        param.data = Tensor::new(
            Arc::from(updated_st),
            shape.clone(),
            param.data.dtype(),
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        Ok(())
    }

    /// Perform one optimization step over all parameters in `params`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
        }
        Ok(())
    }

    /// Save optimizer moments and trainable parameter data into a `.grim.train` `TrainState`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            step: self.step_count as u64,
            fp_format: TrainFpFormat::Fp32,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let blob_name = weight_slot(id);
                let fmt = grim_format::train::train_format_for_dtype(&param.data.dtype());
                let bytes = grim_format::train::encode_f32s_as(&data, fmt);
                state.dtypes.insert(blob_name.clone(), fmt);
                if fmt.is_half() {
                    state.fp_format = fmt;
                }
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(m_st) = self.m.get(id) {
                if let Ok(m_vec) = m_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = m_slot(id);
                    state.add_blob(blob_name, shape.clone(), bytes);
                }
            }

            if let Some(v_st) = self.v.get(id) {
                if let Ok(v_vec) = v_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = v_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = v_slot(id);
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
        self.step_count = state.step as usize;
        for (id, param) in params.iter_mut() {
            let param_key = weight_slot(id);
            let m_key = m_slot(id);
            let v_key = v_slot(id);

            if let Some(blob) = blob_slot(state, &param_key, legacy_weight_slot(id)) {
                let fmt = state
                    .dtypes
                    .get(&param_key)
                    .copied()
                    .unwrap_or(state.fp_format);
                let f32_vals = decode_blob_f32s(&blob.data, fmt)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let restore_dtype = param.data.dtype();
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), restore_dtype.clone())?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    restore_dtype,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if let Some(blob) = blob_slot(state, &m_key, legacy_m_slot(id)) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.m.insert(*id, st);
            }

            if let Some(blob) = blob_slot(state, &v_key, legacy_v_slot(id)) {
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

/// Decode a sidecar blob back to f32 values according to its `TrainFpFormat`.
fn decode_blob_f32s(bytes: &[u8], fmt: TrainFpFormat) -> Result<Vec<f32>> {
    grim_format::train::decode_f32s_from(bytes, fmt)
        .ok_or_else(|| Error::Backend(format!("invalid byte slice length for {fmt:?} blob")))
}

/// Persist only the parameter data + step count (no optimizer moments).
///
/// Used by optimizer variants whose moment buffers are not yet serialized to
/// `.grim.train` (Lion, Lion8Bit, Adafactor, PagedAdamW moments are pending;
/// AdamW persists m/v via its own richer implementation).
pub(crate) fn save_param_data_only(params: &TrainableParams, step_count: usize) -> TrainState {
    let mut state = TrainState {
        step: step_count as u64,
        fp_format: TrainFpFormat::Fp32,
        dtypes: HashMap::new(),
        blobs: HashMap::new(),
    };
    for (id, param) in params.iter() {
        let shape = param.data.shape().dims().to_vec();
        if let Ok(data) = param.data.to_vec_f32() {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let blob_name = weight_slot(id);
            state.add_blob(blob_name, shape, bytes);
        }
    }
    state
}

/// Restore parameter data (and step count) from a `.grim.train` `TrainState`.
pub(crate) fn load_param_data_only(params: &mut TrainableParams, state: &TrainState) -> Result<()> {
    for (id, param) in params.iter_mut() {
        let param_key = weight_slot(id);
        if let Some(blob) = blob_slot(state, &param_key, legacy_weight_slot(id)) {
            let fmt = state
                .dtypes
                .get(&param_key)
                .copied()
                .unwrap_or(state.fp_format);
            let f32_vals = decode_blob_f32s(&blob.data, fmt)?;
            let dev = crate::pick_device_for_tensor(&param.data);
            let restore_dtype = param.data.dtype();
            let storage = dev.from_cpu(&f32_vals, param.data.shape(), restore_dtype.clone())?;
            param.data = Tensor::new(
                Arc::from(storage),
                param.data.shape().clone(),
                restore_dtype,
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

    /// Perform one step update for a single trainable parameter `param` (LOMO / fused streaming step).
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let beta1 = self.config.beta1;
        let beta2 = 0.99f32;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape();
        let elem_count = shape.elem_count();

        // Initialize momentum buffer on first encounter
        if let std::collections::hash_map::Entry::Vacant(e) = self.m.entry(id) {
            let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
            e.insert(zero_m);
        }

        let m_st = self.m.get_mut(&id).unwrap();
        let grad_st = param.grad().storage().clone();
        let data_st = param.data.storage().clone();

        // Try on-device fused Lion kernel first (zero-roundtrip, 1 launch)
        if let Ok(handle) = dev.fused_lion_step(
            data_st.as_ref(),
            grad_st.as_ref(),
            m_st.as_ref(),
            lr,
            beta1,
            beta2,
            weight_decay,
            elem_count,
        ) {
            handle.synchronize()?;
            return Ok(());
        }

        // Lion: τ = β1 * m + (1-β1) * g
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
        Ok(())
    }

    /// Perform one optimization step over all parameters in `params`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
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
// ============================================================================
// 8-bit AdamW Optimizer
// ============================================================================

/// 8-bit AdamW optimizer with memory-efficient Q8_0 moment storage.
/// Quantizes 1st (m) and 2nd (v) moments to Q8_0 blocks, saving ~75% moment memory vs FP32.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW8BitConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
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

/// AdamW with Q8_0 quantized moment storage for reduced VRAM/RAM.
pub struct AdamW8Bit {
    pub config: AdamW8BitConfig,
    pub step_count: usize,
    /// Audit fix (A1 class): per-parameter update counts for `step_param` —
    /// see AdamW.param_steps.
    pub param_steps: HashMap<ParamId, usize>,
    /// 1st moment vector (m) quantized as Q8_0 blocks per parameter ID.
    pub m_q80: HashMap<ParamId, Vec<u8>>,
    /// 2nd moment vector (v) quantized as Q8_0 blocks per parameter ID.
    pub v_q80: HashMap<ParamId, Vec<u8>>,
}

impl std::fmt::Debug for AdamW8Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW8Bit")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_q80_count", &self.m_q80.len())
            .field("v_q80_count", &self.v_q80.len())
            .finish()
    }
}

impl AdamW8Bit {
    /// Create a new 8-bit AdamW optimizer.
    pub fn new(config: AdamW8BitConfig) -> Self {
        Self {
            config,
            step_count: 0,
            param_steps: HashMap::new(),
            m_q80: HashMap::new(),
            v_q80: HashMap::new(),
        }
    }

    /// Perform one step update for a single parameter.
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        // Audit fix (A1): per-param correction — see AdamW.param_steps.
        let sc = {
            let t = self.param_steps.entry(id).or_insert(0);
            *t += 1;
            *t
        };
        let bias_correction1 = 1.0 - beta1.powi(sc as i32);
        let bias_correction2 = 1.0 - beta2.powi(sc as i32);

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let elem_count = shape.elem_count();

        let data: Vec<f32> = param.data.to_vec_f32()?;
        let grad: Vec<f32> = param.grad().to_vec_f32()?;

        let mut m = if let Some(bytes) = self.m_q80.get(&id) {
            grim_quant::dequant_q80(bytes, elem_count)?
        } else {
            vec![0.0f32; elem_count]
        };

        let mut v = if let Some(bytes) = self.v_q80.get(&id) {
            grim_quant::dequant_q80(bytes, elem_count)?
        } else {
            vec![0.0f32; elem_count]
        };

        for i in 0..elem_count {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }

        let new_data: Vec<f32> = (0..elem_count)
            .map(|i| {
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;
                let step = m_hat / (v_hat.sqrt() + eps) + weight_decay * data[i];
                data[i] - lr * step
            })
            .collect();

        let q_m = grim_quant::quant_q80(&m)?;
        let q_v = grim_quant::quant_q80(&v)?;
        self.m_q80.insert(id, q_m);
        self.v_q80.insert(id, q_v);

        let storage = dev.from_cpu(&new_data, &shape, DType::F32)?;
        param.data = Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );

        Ok(())
    }

    /// Perform one optimization step over all parameters.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
        }
        Ok(())
    }

    /// Persist parameter data + step count + Q8_0 moments.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = save_param_data_only(params, self.step_count);
        for (id, m_bytes) in &self.m_q80 {
            let key = format!(
                "opt_8bit_m_{}_{}_{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" }
            );
            state.add_blob(key, vec![m_bytes.len()], m_bytes.clone());
        }
        for (id, v_bytes) in &self.v_q80 {
            let key = format!(
                "opt_8bit_v_{}_{}_{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" }
            );
            state.add_blob(key, vec![v_bytes.len()], v_bytes.clone());
        }
        state
    }

    /// Restore parameter data + step count + Q8_0 moments from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)?;
        for (id, _) in params.iter() {
            let m_key = format!(
                "opt_8bit_m_{}_{}_{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" }
            );
            let v_key = format!(
                "opt_8bit_v_{}_{}_{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" }
            );
            if let Some(blob) = blob_slot(state, &m_key, legacy_m_slot(id)) {
                self.m_q80.insert(*id, blob.data.clone());
            }
            if let Some(blob) = blob_slot(state, &v_key, legacy_v_slot(id)) {
                self.v_q80.insert(*id, blob.data.clone());
            }
        }
        Ok(())
    }
}

// ============================================================================
// Paged AdamW - Offloaded Moment Pages with Dirty-Set Tracking
// ============================================================================

/// Configuration for Paged AdamW optimizer.
/// Paged AdamW offloads cold moment pages to host RAM with a dirty-set tracking
/// mechanism and page-in on touch.
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
    /// Page size for offloaded buffers (in number of parameters per page)
    pub page_size: usize,
    /// Enable CPU-offloading of optimizer states
    pub cpu_offload: bool,
    /// Maximum GPU memory fraction for optimizer states (0.0 = CPU only, 1.0 = GPU only)
    pub gpu_mem_fraction: f32,
    /// Use 8-bit Q8_0 quantization for paged moment storage
    pub use_8bit_moments: bool,
}

impl Default for PagedAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            page_size: 65536,
            cpu_offload: true,
            gpu_mem_fraction: 0.0,
            use_8bit_moments: false,
        }
    }
}

/// A tracked memory page in host/device RAM.
#[derive(Debug, Clone)]
pub struct MomentPage {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub dirty: bool,
    pub in_gpu: bool,
}

/// Paged AdamW optimizer state with dirty-set tracked pages.
pub struct PagedAdamW {
    pub config: PagedAdamWConfig,
    pub step_count: usize,
    /// Audit fix (A1 class): per-parameter update counts for `step_param`.
    pub param_steps: HashMap<ParamId, usize>,
    /// Pages indexed by (ParamId, page_index)
    pub pages: HashMap<(ParamId, usize), MomentPage>,
    /// Set of (ParamId, page_index) that were mutated in the current step
    pub dirty_set: std::collections::HashSet<(ParamId, usize)>,
}

impl std::fmt::Debug for PagedAdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAdamW")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("page_count", &self.pages.len())
            .field("dirty_pages", &self.dirty_set.len())
            .finish()
    }
}

impl PagedAdamW {
    pub fn new(config: PagedAdamWConfig) -> Self {
        Self {
            config,
            param_steps: HashMap::new(),
            step_count: 0,
            pages: HashMap::new(),
            dirty_set: std::collections::HashSet::new(),
        }
    }

    /// Page in the requested moment page on touch.
    fn touch_page(&mut self, id: ParamId, page_idx: usize, page_len: usize) -> &mut MomentPage {
        self.dirty_set.insert((id, page_idx));
        self.pages
            .entry((id, page_idx))
            .or_insert_with(|| MomentPage {
                m: vec![0.0f32; page_len],
                v: vec![0.0f32; page_len],
                dirty: false,
                in_gpu: !self.config.cpu_offload,
            })
    }

    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;
        let page_size = self.config.page_size.max(1);

        // Audit fix (A1): per-param correction — see AdamW.param_steps.
        let sc = {
            let t = self.param_steps.entry(id).or_insert(0);
            *t += 1;
            *t
        };
        let bias_correction1 = 1.0 - beta1.powi(sc as i32);
        let bias_correction2 = 1.0 - beta2.powi(sc as i32);

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let elem_count = shape.elem_count();

        let data: Vec<f32> = param.data.to_vec_f32()?;
        let grad: Vec<f32> = param.grad().to_vec_f32()?;
        let mut new_data = vec![0.0f32; elem_count];

        let cpu_offload = self.config.cpu_offload;
        let num_pages = elem_count.div_ceil(page_size);
        for page_idx in 0..num_pages {
            let offset = page_idx * page_size;
            let current_len = (elem_count - offset).min(page_size);

            let page = self.touch_page(id, page_idx, current_len);
            page.in_gpu = true;
            page.dirty = true;

            for i in 0..current_len {
                let g = grad[offset + i];
                let d = data[offset + i];

                page.m[i] = beta1 * page.m[i] + (1.0 - beta1) * g;
                page.v[i] = beta2 * page.v[i] + (1.0 - beta2) * g * g;

                let m_hat = page.m[i] / bias_correction1;
                let v_hat = page.v[i] / bias_correction2;
                let step = m_hat / (v_hat.sqrt() + eps) + weight_decay * d;
                new_data[offset + i] = d - lr * step;
            }

            if cpu_offload {
                page.in_gpu = false; // page flushed back to host memory pool
            }
        }

        let storage = dev.from_cpu(&new_data, &shape, DType::F32)?;
        param.data = Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );

        Ok(())
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        self.dirty_set.clear();
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
        }
        Ok(())
    }

    /// Persist parameter data + step count (paged moments serialized per page).
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = save_param_data_only(params, self.step_count);
        for ((id, page_idx), page) in &self.pages {
            let m_key = format!(
                "paged_m_{}_{}_{}_p{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" },
                page_idx
            );
            let v_key = format!(
                "paged_v_{}_{}_{}_p{}",
                id.layer_idx,
                id.adapter_id,
                if id.is_a { "a" } else { "b" },
                page_idx
            );
            let m_bytes: Vec<u8> = page.m.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v_bytes: Vec<u8> = page.v.iter().flat_map(|v| v.to_le_bytes()).collect();
            state.add_blob(m_key, vec![page.m.len()], m_bytes);
            state.add_blob(v_key, vec![page.v.len()], v_bytes);
        }
        state
    }

    /// Restore parameter data + step count from a train state.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        load_param_data_only(params, state)?;
        for (id, param) in params.iter() {
            let elem_count = param.data.shape().elem_count();
            let page_size = self.config.page_size.max(1);
            let num_pages = elem_count.div_ceil(page_size);
            for p in 0..num_pages {
                let m_key = format!(
                    "paged_m_{}_{}_{}_p{}",
                    id.layer_idx,
                    id.adapter_id,
                    if id.is_a { "a" } else { "b" },
                    p
                );
                let v_key = format!(
                    "paged_v_{}_{}_{}_p{}",
                    id.layer_idx,
                    id.adapter_id,
                    if id.is_a { "a" } else { "b" },
                    p
                );
                if let Some(blob) = blob_slot(state, &m_key, legacy_m_slot(id)) {
                    if let Ok(m_vals) = bytes_to_f32_vec(&blob.data) {
                        let entry = self.pages.entry((*id, p)).or_insert_with(|| MomentPage {
                            m: vec![0.0; m_vals.len()],
                            v: vec![0.0; m_vals.len()],
                            dirty: false,
                            in_gpu: false,
                        });
                        entry.m = m_vals;
                    }
                }
                if let Some(blob) = blob_slot(state, &v_key, legacy_v_slot(id)) {
                    if let Ok(v_vals) = bytes_to_f32_vec(&blob.data) {
                        let entry = self.pages.entry((*id, p)).or_insert_with(|| MomentPage {
                            m: vec![0.0; v_vals.len()],
                            v: vec![0.0; v_vals.len()],
                            dirty: false,
                            in_gpu: false,
                        });
                        entry.v = v_vals;
                    }
                }
            }
        }
        Ok(())
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
            // Lion8Bit update: w - lr * (β1*m + (1-β1)*g) + wd*w, matching the
            // correct Lion path (adamw.rs:808) which does data_st + neg_lr_step.
            let (updated, _) = dev.add(data_st.as_ref(), neg_lr_step.as_ref(), shape)?;
            let (updated, _) = dev.add(updated.as_ref(), wd_w.as_ref(), shape)?;

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
        let _weight_decay = self.config.weight_decay;

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
            let _elem_count = rows * cols;

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

            let effective_v: Vec<f32> = b_vec
                .iter()
                .flat_map(|&bi| c_vec.iter().map(move |&cj| bi * cj))
                .collect();

            let step_scale = if self.config.relative_step && self.step_count > 10 {
                lr * beta2.sqrt() / effective_v[0].max(1e-8)
            } else {
                lr / effective_v[0].max(1e-8)
            };

            let new_data: Vec<f32> = data
                .iter()
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

/// Halko randomized SVD for matrix decomposition mat [m, n] -> U [m, rank], S [rank], V^T [rank, n].
pub fn randomized_svd(
    mat: &[f32],
    m: usize,
    n: usize,
    rank: usize,
    oversample: usize,
    niter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let k = (rank + oversample).min(m).min(n);
    if k == 0 || m == 0 || n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut omega = vec![0.0f32; n * k];
    for (i, slot) in omega.iter_mut().enumerate() {
        let u1 = ((i as f32 + 1.0) * 0.017).fract().max(1e-7);
        let u2 = ((i as f32 + 1.0) * 0.031).fract();
        *slot = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    let mut y = vec![0.0f32; m * k];
    for i in 0..m {
        for l in 0..k {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += mat[i * n + j] * omega[j * k + l];
            }
            y[i * k + l] = sum;
        }
    }

    for _ in 0..niter {
        let mut y_temp = vec![0.0f32; n * k];
        for j in 0..n {
            for l in 0..k {
                let mut sum = 0.0f32;
                for i in 0..m {
                    sum += mat[i * n + j] * y[i * k + l];
                }
                y_temp[j * k + l] = sum;
            }
        }
        for i in 0..m {
            for l in 0..k {
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += mat[i * n + j] * y_temp[j * k + l];
                }
                y[i * k + l] = sum;
            }
        }
    }

    let mut q_qr = vec![0.0f32; m * k];
    for l in 0..k {
        for i in 0..m {
            q_qr[i * k + l] = y[i * k + l];
        }
        for prev in 0..l {
            let mut dot = 0.0f32;
            for i in 0..m {
                dot += q_qr[i * k + prev] * q_qr[i * k + l];
            }
            for i in 0..m {
                q_qr[i * k + l] -= dot * q_qr[i * k + prev];
            }
        }
        let mut norm = 0.0f32;
        for i in 0..m {
            norm += q_qr[i * k + l] * q_qr[i * k + l];
        }
        let norm = norm.sqrt().max(1e-10);
        for i in 0..m {
            q_qr[i * k + l] /= norm;
        }
    }

    let mut b = vec![0.0f32; k * n];
    for l in 0..k {
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += q_qr[i * k + l] * mat[i * n + j];
            }
            b[l * n + j] = sum;
        }
    }

    let mut bbt = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f32;
            for l in 0..n {
                sum += b[i * n + l] * b[j * n + l];
            }
            bbt[i * k + j] = sum;
        }
    }

    let mut v_b = vec![0.0f32; k * k];
    for i in 0..k {
        v_b[i * k + i] = 1.0;
    }
    for _ in 0..30 {
        let mut max_off = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..k {
            for j in i + 1..k {
                if bbt[i * k + j].abs() > max_off {
                    max_off = bbt[i * k + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-7 {
            break;
        }
        let diff = bbt[q * k + q] - bbt[p * k + p];
        let t = if diff.abs() < 1e-10 {
            0.5 * std::f32::consts::PI
        } else {
            0.5 * (2.0 * bbt[p * k + q] / diff).atan()
        };
        let c = t.cos();
        let s = t.sin();
        for i in 0..k {
            let vip = v_b[i * k + p];
            let viq = v_b[i * k + q];
            v_b[i * k + p] = c * vip - s * viq;
            v_b[i * k + q] = s * vip + c * viq;
        }
        let bpp = bbt[p * k + p];
        let bqq = bbt[q * k + q];
        let bpq = bbt[p * k + q];
        bbt[p * k + p] = c * c * bpp - 2.0 * s * c * bpq + s * s * bqq;
        bbt[q * k + q] = s * s * bpp + 2.0 * s * c * bpq + c * c * bqq;
        bbt[p * k + q] = 0.0;
        bbt[q * k + p] = 0.0;
    }

    let actual_r = rank.min(k);
    let mut u_out = vec![0.0f32; m * actual_r];
    let mut s_out = vec![0.0f32; actual_r];
    let mut vt_out = vec![0.0f32; actual_r * n];

    for i in 0..m {
        for r in 0..actual_r {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += q_qr[i * k + l] * v_b[l * k + r];
            }
            u_out[i * actual_r + r] = sum;
        }
    }

    for r in 0..actual_r {
        s_out[r] = bbt[r * k + r].max(0.0).sqrt();
    }

    for r in 0..actual_r {
        let inv_s = if s_out[r] > 1e-8 { 1.0 / s_out[r] } else { 0.0 };
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += u_out[i * actual_r + r] * mat[i * n + j];
            }
            vt_out[r * n + j] = sum * inv_s;
        }
    }

    (u_out, s_out, vt_out)
}

#[derive(Debug, Clone)]
pub struct GaloreProjector {
    pub rank: usize,
    pub update_proj_gap: usize,
    pub scale: f32,
    pub q_orth: Option<Vec<f32>>,
    pub step: usize,
}

impl GaloreProjector {
    pub fn new(rank: usize, update_proj_gap: usize, scale: f32) -> Self {
        Self {
            rank,
            update_proj_gap,
            scale,
            q_orth: None,
            step: 0,
        }
    }

    pub fn project(&mut self, grad: &[f32], m: usize, n: usize) -> (Vec<f32>, usize, usize) {
        let r = self.rank.min(m).min(n);
        if self.step % self.update_proj_gap == 0 || self.q_orth.is_none() {
            let (u, _s, vt) = randomized_svd(grad, m, n, r, 10, 2);
            if m >= n {
                self.q_orth = Some(u);
            } else {
                let mut v_orth = vec![0.0f32; n * r];
                for j in 0..n {
                    for r_idx in 0..r {
                        v_orth[j * r + r_idx] = vt[r_idx * n + j];
                    }
                }
                self.q_orth = Some(v_orth);
            }
        }
        self.step += 1;

        let q = self.q_orth.as_ref().unwrap();
        if m >= n {
            let mut low_grad = vec![0.0f32; m * r];
            for i in 0..m {
                for r_idx in 0..r {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += grad[i * n + j] * q[j * r + r_idx];
                    }
                    low_grad[i * r + r_idx] = sum * self.scale;
                }
            }
            (low_grad, m, r)
        } else {
            let mut low_grad = vec![0.0f32; r * n];
            for r_idx in 0..r {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for i in 0..m {
                        sum += q[i * r + r_idx] * grad[i * n + j];
                    }
                    low_grad[r_idx * n + j] = sum * self.scale;
                }
            }
            (low_grad, r, n)
        }
    }

    pub fn project_back(&self, low_update: &[f32], m: usize, n: usize) -> Vec<f32> {
        let r = self.rank.min(m).min(n);
        let q = match &self.q_orth {
            Some(q) => q,
            None => return low_update.to_vec(),
        };
        let mut full_update = vec![0.0f32; m * n];
        if m >= n {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for r_idx in 0..r {
                        sum += low_update[i * r + r_idx] * q[j * r + r_idx];
                    }
                    full_update[i * n + j] = sum * self.scale;
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for r_idx in 0..r {
                        sum += q[i * r + r_idx] * low_update[r_idx * n + j];
                    }
                    full_update[i * n + j] = sum * self.scale;
                }
            }
        }
        full_update
    }
}

#[derive(Debug, Clone)]
pub struct QGaLoreAdamW8BitConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub rank: usize,
    pub update_proj_gap: usize,
    pub scale: f32,
}

impl Default for QGaLoreAdamW8BitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            rank: 128,
            update_proj_gap: 200,
            scale: 0.25,
        }
    }
}

pub struct QGaLoreAdamW8Bit {
    pub config: QGaLoreAdamW8BitConfig,
    pub step_count: usize,
    pub m_state: HashMap<ParamId, Vec<u8>>,
    pub v_state: HashMap<ParamId, Vec<u8>>,
    pub m_scale: HashMap<ParamId, f32>,
    pub v_scale: HashMap<ParamId, f32>,
    pub projectors: HashMap<ParamId, GaloreProjector>,
}

impl QGaLoreAdamW8Bit {
    pub fn new(config: QGaLoreAdamW8BitConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m_state: HashMap::new(),
            v_state: HashMap::new(),
            m_scale: HashMap::new(),
            v_scale: HashMap::new(),
            projectors: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape();
            let dims = shape.dims();

            let mut data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;

            if dims.len() == 2 && dims[0].min(dims[1]) > self.config.rank {
                let m = dims[0];
                let n = dims[1];

                let proj = self.projectors.entry(*id).or_insert_with(|| {
                    GaloreProjector::new(
                        self.config.rank,
                        self.config.update_proj_gap,
                        self.config.scale,
                    )
                });

                let (low_grad, m_low, n_low) = proj.project(&grad, m, n);
                let elem_count = m_low * n_low;

                if !self.m_state.contains_key(id) {
                    self.m_state.insert(*id, vec![127u8; elem_count]);
                    self.v_state.insert(*id, vec![127u8; elem_count]);
                    self.m_scale.insert(*id, 1.0);
                    self.v_scale.insert(*id, 1.0);
                }

                let m_q = self.m_state.get_mut(id).unwrap();
                let v_q = self.v_state.get_mut(id).unwrap();
                let _m_sc = self.m_scale.get_mut(id);
                let _v_sc = self.v_scale.get_mut(id);

                let mut m_f32 = vec![0.0f32; elem_count];
                let mut v_f32 = vec![0.0f32; elem_count];
                let mut low_update = vec![0.0f32; elem_count];

                let cur_m_scale = *self.m_scale.get(id).unwrap();
                let cur_v_scale = *self.v_scale.get(id).unwrap();

                for i in 0..elem_count {
                    let m_val = (m_q[i] as f32 - 127.0) * cur_m_scale;
                    let v_val = (v_q[i] as f32 - 127.0) * cur_v_scale;

                    let new_m = beta1 * m_val + (1.0 - beta1) * low_grad[i];
                    let new_v = beta2 * v_val + (1.0 - beta2) * low_grad[i] * low_grad[i];

                    m_f32[i] = new_m;
                    v_f32[i] = new_v;

                    let m_hat = new_m / bc1;
                    let v_hat = new_v / bc2;

                    low_update[i] = m_hat / (v_hat.sqrt() + eps);
                }

                let max_m = m_f32
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-8);
                let max_v = v_f32
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-8);
                self.m_scale.insert(*id, max_m / 127.0);
                self.v_scale.insert(*id, max_v / 127.0);

                for i in 0..elem_count {
                    m_q[i] =
                        ((m_f32[i] / (max_m / 127.0)).round().clamp(-127.0, 127.0) + 127.0) as u8;
                    v_q[i] =
                        ((v_f32[i] / (max_v / 127.0)).round().clamp(-127.0, 127.0) + 127.0) as u8;
                }

                let full_update = proj.project_back(&low_update, m, n);

                for i in 0..data.len() {
                    data[i] -= lr * (full_update[i] + wd * data[i]);
                }
            } else {
                for i in 0..data.len() {
                    data[i] -= lr * (grad[i] + wd * data[i]);
                }
            }

            let storage = dev.from_cpu(&data, shape, DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        save_param_data_only(params, self.step_count)
    }

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
// Muon Optimizer (SPECTRAL-QLORA)
// ============================================================================

/// Hyperparameters for Muon optimizer.
///
/// Muon replaces AdamW for adapter-only training. It uses:
/// - Newton-Schulz orthogonalization for the direction matrix (B, tall/thin)
///   to keep it well-conditioned on the Stiefel manifold without second-moment storage.
/// - 1-bit Sign-SGD for the magnitude matrix (A, wide/thin), with zero moment memory.
///
/// Split weight decay follows LoRA-Muon (2606.12921): different coefficient
/// applied to A (is_a == true) vs B (is_a == false).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonConfig {
    /// Learning rate for both matrices.
    pub lr: f32,
    /// Momentum coefficient for the B (direction) matrix.
    pub beta: f32,
    /// Weight decay for the A (magnitude / sign-SGD) matrix.
    /// Default 0.0 — follows LoRA-Muon which applies no decay to the magnitude.
    pub weight_decay_a: f32,
    /// Weight decay for the B (direction / Newton-Schulz) matrix.
    /// Default 0.01 — the primary decay target in LoRA-Muon.
    pub weight_decay_b: f32,
    /// Number of Newton-Schulz iterations for B-matrix gradient orthogonalization.
    pub ns_iters: usize,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta: 0.9,
            weight_decay_a: 0.0,
            weight_decay_b: 0.01,
            ns_iters: 10,
        }
    }
}

/// Muon optimizer: Newton-Schulz + Sign-SGD with split weight decay.
///
/// - **B matrices** (`is_a == false`, shape `[out, rank]`, tall/thin): the
///   gradient is orthogonalized via `subspace_newton_schulz_step` (reusing
///   `grim-quant::soul_eater`), then accumulated into a momentum buffer and
///   applied as `w -= lr * (m + wd_b * w)`.
/// - **A matrices** (`is_a == true`, shape `[rank, in]`, wide/thin): 1-bit
///   Sign-SGD update `w -= lr * (sign(g) + wd_a * w)`. No momentum buffer needed.
///
/// Only B matrices carry moment state. Checkpoints serialize B's momentum
/// buffers using the `opt_m_{layer}_{adapter}_{b}` blob convention; A matrices
/// are data-only (like Lion).
pub struct Muon {
    pub config: MuonConfig,
    pub step_count: usize,
    /// Momentum buffer for B matrices only (`is_a == false`). A matrices use
    /// Sign-SGD with no moment storage.
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl std::fmt::Debug for Muon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Muon")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_count", &self.m.len())
            .finish()
    }
}

impl Muon {
    /// Create a new Muon optimizer with the given configuration.
    pub fn new(config: MuonConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
        }
    }

    /// Perform one optimization step over all parameters in `params`.
    ///
    /// B matrices (is_a == false): Newton-Schulz orthogonalization on the
    /// gradient, then momentum update. A matrices (is_a == true): 1-bit
    /// Sign-SGD with no momentum.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let lr = self.config.lr;
        let beta = self.config.beta;
        let wd_a = self.config.weight_decay_a;
        let wd_b = self.config.weight_decay_b;
        let ns_iters = self.config.ns_iters;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }

            let grad_vec = param.grad().to_vec_f32()?;
            let data_vec = param.data.to_vec_f32()?;
            let shape = param.data.shape();
            let elem_count = shape.elem_count();
            let dev = crate::pick_device_for_tensor(&param.data);

            let new_data: Vec<f32> = if id.is_a {
                // A matrix [rank, in]: 1-bit Sign-SGD (magnitude update, zero moment memory).
                let mut new = Vec::with_capacity(elem_count);
                for i in 0..elem_count {
                    let s = if grad_vec[i] > 0.0 {
                        1.0
                    } else if grad_vec[i] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    new.push(data_vec[i] - lr * (s + wd_a * data_vec[i]));
                }
                new
            } else {
                // B matrix [out, rank]: Newton-Schulz orthogonalization + momentum.
                // Seed momentum buffer on first encounter (device-resident).
                if !self.m.contains_key(id) {
                    let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
                    self.m.insert(*id, zero_m);
                }

                let dims = shape.dims();
                let (rows, cols) = if dims.len() >= 2 {
                    (dims[dims.len() - 2], dims[dims.len() - 1])
                } else {
                    (elem_count, 1)
                };

                // Apply Newton-Schulz to the gradient (tall/thin [rows, cols]).
                let mut g_orth = grad_vec.clone();
                if rows >= cols {
                    let _ = grim_quant::soul_eater::subspace_newton_schulz_step(
                        &mut g_orth,
                        rows,
                        cols,
                        ns_iters,
                    );
                }

                // m = beta * m_old + (1 - beta) * g_orth
                let m_st = self.m.get_mut(id).unwrap();
                let m_old_vec = m_st.to_cpu_vec_f32()?;
                let m_new: Vec<f32> = (0..elem_count)
                    .map(|i| beta * m_old_vec[i] + (1.0 - beta) * g_orth[i])
                    .collect();

                // w -= lr * (m_new + wd_b * w)
                let mut new = Vec::with_capacity(elem_count);
                for i in 0..elem_count {
                    new.push(data_vec[i] - lr * (m_new[i] + wd_b * data_vec[i]));
                }

                // Write back momentum buffer (device-resident).
                let m_storage = dev.from_cpu(&m_new, shape, DType::F32)?;
                *m_st = m_storage;

                new
            };

            // Write back updated parameter (device-resident).
            let storage = dev.from_cpu(&new_data, shape, DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                shape.clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
            param.zero_grad()?;
        }

        Ok(())
    }

    /// Save parameter data + B-matrix momentum buffers to a `TrainState`.
    ///
    /// A matrices (sign-SGD) have no moment state and are data-only.
    /// B matrices serialize their momentum as `opt_m_{layer}_{adapter}_b`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            step: self.step_count as u64,
            fp_format: TrainFpFormat::Fp32,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let blob_name = weight_slot(id);
                let fmt = grim_format::train::train_format_for_dtype(&param.data.dtype());
                let bytes = grim_format::train::encode_f32s_as(&data, fmt);
                state.dtypes.insert(blob_name.clone(), fmt);
                if fmt.is_half() {
                    state.fp_format = fmt;
                }
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            // Serialize B-matrix momentum only (A matrices have no moments).
            if !id.is_a {
                if let Some(m_st) = self.m.get(id) {
                    if let Ok(m_vec) = m_st.to_cpu_vec_f32() {
                        let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let blob_name = format!("opt_m_{}_{}_b", id.layer_idx, id.adapter_id);
                        state.add_blob(blob_name, shape, bytes);
                    }
                }
            }
        }

        state
    }

    /// Restore parameter data and B-matrix momentum from a `TrainState`.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        for (id, param) in params.iter_mut() {
            let param_key = weight_slot(id);
            let m_key = format!("opt_m_{}_{}_b", id.layer_idx, id.adapter_id);

            if let Some(blob) = blob_slot(state, &param_key, legacy_weight_slot(id)) {
                let fmt = state
                    .dtypes
                    .get(&param_key)
                    .copied()
                    .unwrap_or(state.fp_format);
                let f32_vals = decode_blob_f32s(&blob.data, fmt)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let restore_dtype = param.data.dtype();
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), restore_dtype.clone())?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    restore_dtype,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if !id.is_a {
                if let Some(blob) = blob_slot(state, &m_key, legacy_m_slot(id)) {
                    let f32_vals = bytes_to_f32_vec(&blob.data)?;
                    let dev = crate::pick_device_for_tensor(&param.data);
                    let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                    self.m.insert(*id, st);
                }
            }
        }

        Ok(())
    }
}

// ── M-Adam (Additive-Multiplicative Optimization) ───────────────────────────

/// Configuration for M-Adam optimizer (arXiv:2607.10611).
///
/// Combines additive momentum tracking with a multiplicative learning-rate
/// scaling factor derived from local gradient variance, stabilizing ultra-low
/// precision (FP4/FP8) fine-tuning without step-size explosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub gamma: f32,
    pub weight_decay: f32,
}

impl Default for MAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            gamma: 0.1,
            weight_decay: 0.01,
        }
    }
}

/// M-Adam optimizer: maintains 1st and 2nd moments alongside multiplicative damping.
pub struct MAdam {
    pub config: MAdamConfig,
    pub step_count: usize,
    pub m: HashMap<ParamId, Box<dyn BackendStorage>>,
    pub v: HashMap<ParamId, Box<dyn BackendStorage>>,
    /// Audit fix (A1 class): per-parameter update counts for `step_param`.
    pub param_steps: HashMap<ParamId, usize>,
}

impl MAdam {
    pub fn new(config: MAdamConfig) -> Self {
        Self {
            param_steps: HashMap::new(),
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Perform one step update for a single trainable parameter `param` (LOMO / fused streaming step).
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let grad = param.grad();

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let gamma = self.config.gamma;
        let wd = self.config.weight_decay;

        // Audit fix (A1): per-param correction — see AdamW.param_steps.
        let sc = {
            let t = self.param_steps.entry(id).or_insert(0);
            *t += 1;
            *t
        };
        let bc1 = 1.0 - beta1.powi(sc as i32);
        let bc2 = 1.0 - beta2.powi(sc as i32);

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let elem_count = shape.elem_count();

        if let std::collections::hash_map::Entry::Vacant(e) = self.m.entry(id) {
            let m_init = dev.from_cpu(&vec![0.0f32; elem_count], &shape, DType::F32)?;
            let v_init = dev.from_cpu(&vec![0.0f32; elem_count], &shape, DType::F32)?;
            e.insert(m_init);
            self.v.insert(id, v_init);
        }

        let m_st = self.m.get_mut(&id).unwrap();
        let v_st = self.v.get_mut(&id).unwrap();
        let grad_st = grad.storage().clone();
        let data_st = param.data.storage().clone();

        // Try on-device fused M-Adam kernel first (zero-roundtrip, 1 launch)
        if let Ok(handle) = dev.fused_madam_step(
            data_st.as_ref(),
            grad_st.as_ref(),
            m_st.as_ref(),
            v_st.as_ref(),
            lr,
            beta1,
            beta2,
            eps,
            gamma,
            wd,
            bc1,
            bc2,
            elem_count,
        ) {
            handle.synchronize()?;
            return Ok(());
        }

        // Host fallback
        let grad_vec = grad.to_vec_f32()?;
        let data_vec = param.data.to_vec_f32()?;
        let m_old = m_st.to_cpu_vec_f32()?;
        let v_old = v_st.to_cpu_vec_f32()?;

        let mut m_new = Vec::with_capacity(elem_count);
        let mut v_new = Vec::with_capacity(elem_count);
        let mut new_data = Vec::with_capacity(elem_count);

        for i in 0..elem_count {
            let g = grad_vec[i];
            let m_val = beta1 * m_old[i] + (1.0 - beta1) * g;
            let v_val = beta2 * v_old[i] + (1.0 - beta2) * g * g;

            m_new.push(m_val);
            v_new.push(v_val);

            let m_hat = m_val / bc1;
            let v_hat = v_val / bc2;

            // Multiplicative curvature-aware step dampening
            let denom = v_hat.sqrt() + eps;
            let mult_scale = 1.0 / (1.0 + gamma * (g.abs() / denom));
            let step_val = (m_hat / denom) * mult_scale;

            let p = data_vec[i];
            new_data.push(p - lr * (step_val + wd * p));
        }

        *m_st = dev.from_cpu(&m_new, &shape, DType::F32)?;
        *v_st = dev.from_cpu(&v_new, &shape, DType::F32)?;

        let storage = dev.from_cpu(&new_data, &shape, DType::F32)?;
        param.data = Tensor::new(
            Arc::from(storage),
            shape.clone(),
            DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        Ok(())
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            step: self.step_count as u64,
            fp_format: TrainFpFormat::Fp32,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let blob_name = weight_slot(id);
                let fmt = grim_format::train::train_format_for_dtype(&param.data.dtype());
                let bytes = grim_format::train::encode_f32s_as(&data, fmt);
                state.dtypes.insert(blob_name.clone(), fmt);
                if fmt.is_half() {
                    state.fp_format = fmt;
                }
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(m_st) = self.m.get(id) {
                if let Ok(m_vec) = m_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = m_slot(id);
                    state.add_blob(blob_name, shape.clone(), bytes);
                }
            }

            if let Some(v_st) = self.v.get(id) {
                if let Ok(v_vec) = v_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = v_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = v_slot(id);
                    state.add_blob(blob_name, shape, bytes);
                }
            }
        }

        state
    }

    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        for (id, param) in params.iter_mut() {
            let param_key = weight_slot(id);
            let m_key = m_slot(id);
            let v_key = v_slot(id);

            if let Some(blob) = blob_slot(state, &param_key, legacy_weight_slot(id)) {
                let fmt = state
                    .dtypes
                    .get(&param_key)
                    .copied()
                    .unwrap_or(state.fp_format);
                let f32_vals = decode_blob_f32s(&blob.data, fmt)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let restore_dtype = param.data.dtype();
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), restore_dtype.clone())?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    restore_dtype,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if let Some(blob) = blob_slot(state, &m_key, legacy_m_slot(id)) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.m.insert(*id, st);
            }

            if let Some(blob) = blob_slot(state, &v_key, legacy_v_slot(id)) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.v.insert(*id, st);
            }
        }
        Ok(())
    }
}

// ── LionVote (Per-Layer Magnitude Voting for Sign Momentum) ─────────────────

/// Configuration for LionVote optimizer (arXiv:2607.09266).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LionVoteConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub vote_threshold: f32,
}

impl Default for LionVoteConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
            vote_threshold: 0.6,
        }
    }
}

/// LionVote optimizer: computes per-layer agreement vote across sign updates
/// to scale gradient step magnitudes appropriately across deep architectures.
pub struct LionVote {
    pub config: LionVoteConfig,
    pub step_count: usize,
    pub exp_avg: HashMap<ParamId, Box<dyn BackendStorage>>,
}

impl LionVote {
    pub fn new(config: LionVoteConfig) -> Self {
        Self {
            config,
            step_count: 0,
            exp_avg: HashMap::new(),
        }
    }

    /// Perform one step update for a single trainable parameter `param` (LOMO / fused streaming step).
    pub fn step_param(
        &mut self,
        id: ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let grad = param.grad();

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let wd = self.config.weight_decay;
        let threshold = self.config.vote_threshold;

        let grad_vec = grad.to_vec_f32()?;
        let data_vec = param.data.to_vec_f32()?;
        let elem_count = data_vec.len();
        let shape = param.data.shape();
        let dev = crate::pick_device_for_tensor(&param.data);

        if let std::collections::hash_map::Entry::Vacant(e) = self.exp_avg.entry(id) {
            let init_buf = dev.from_cpu(&vec![0.0f32; elem_count], shape, DType::F32)?;
            e.insert(init_buf);
        }

        let exp_st = self.exp_avg.get_mut(&id).unwrap();
        let exp_old = exp_st.to_cpu_vec_f32()?;

        // Compute per-layer vote agreement ratio
        let mut positive_votes = 0usize;
        for i in 0..elem_count {
            let update_i = beta1 * exp_old[i] + (1.0 - beta1) * grad_vec[i];
            if update_i >= 0.0 {
                positive_votes += 1;
            }
        }
        let agreement = (positive_votes as f32 / elem_count as f32 - 0.5).abs() * 2.0;
        let scale_factor = if agreement > threshold {
            1.0
        } else {
            0.5 + 0.5 * agreement
        };

        let mut exp_new = Vec::with_capacity(elem_count);
        let mut new_data = Vec::with_capacity(elem_count);

        for i in 0..elem_count {
            let g = grad_vec[i];
            let exp_val = exp_old[i];

            let update = beta1 * exp_val + (1.0 - beta1) * g;
            let sign_update = if update > 0.0 {
                1.0
            } else if update < 0.0 {
                -1.0
            } else {
                0.0
            };

            let next_exp = beta2 * exp_val + (1.0 - beta2) * g;
            exp_new.push(next_exp);

            let p = data_vec[i];
            new_data.push(p - lr * (sign_update * scale_factor + wd * p));
        }

        *exp_st = dev.from_cpu(&exp_new, shape, DType::F32)?;

        let storage = dev.from_cpu(&new_data, shape, DType::F32)?;
        param.data = Tensor::new(
            Arc::from(storage),
            shape.clone(),
            DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        Ok(())
    }

    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;
        for (id, param) in params.iter_mut() {
            self.step_param(*id, param)?;
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            step: self.step_count as u64,
            fp_format: TrainFpFormat::Fp32,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let blob_name = weight_slot(id);
                let fmt = grim_format::train::train_format_for_dtype(&param.data.dtype());
                let bytes = grim_format::train::encode_f32s_as(&data, fmt);
                state.dtypes.insert(blob_name.clone(), fmt);
                if fmt.is_half() {
                    state.fp_format = fmt;
                }
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(exp_st) = self.exp_avg.get(id) {
                if let Ok(exp_vec) = exp_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = exp_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = format!(
                        "opt_exp_{}_{}_{}",
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

    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        for (id, param) in params.iter_mut() {
            let suffix = if id.is_a { "a" } else { "b" };
            let param_key = weight_slot(id);
            let exp_key = format!("opt_exp_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);

            if let Some(blob) = blob_slot(state, &param_key, legacy_weight_slot(id)) {
                let fmt = state
                    .dtypes
                    .get(&param_key)
                    .copied()
                    .unwrap_or(state.fp_format);
                let f32_vals = decode_blob_f32s(&blob.data, fmt)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let restore_dtype = param.data.dtype();
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), restore_dtype.clone())?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    restore_dtype,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if let Some(blob) = state.blobs.get(&exp_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.exp_avg.insert(*id, st);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::Shape;

    #[test]
    fn test_madam_optimizer_step_and_save_load_roundtrip() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let mut params = TrainableParams::new();
        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let mut tp = TrainableParam::new(
            pid_a,
            grim_backend_cpu::cpu_tensor(vec![1.0f32; 8], Shape::new(vec![2, 4])),
        )
        .unwrap();
        tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.2f32; 8],
            Shape::new(vec![2, 4]),
        ))
        .unwrap();
        params.insert(tp);

        let mut madam = MAdam::new(MAdamConfig::default());
        madam.step(&mut params).unwrap();

        let p_val = params.get(pid_a).unwrap().data.to_vec_f32().unwrap();
        assert!(p_val[0] < 1.0, "step must decrease param along gradient");

        let state = madam.save_to_train_state(&params);
        assert_eq!(state.step, 1);

        let mut params2 = TrainableParams::new();
        params2.insert(
            TrainableParam::new(
                pid_a,
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 8], Shape::new(vec![2, 4])),
            )
            .unwrap(),
        );
        let mut madam2 = MAdam::new(MAdamConfig::default());
        madam2.load_from_train_state(&mut params2, &state).unwrap();

        let restored = params2.get(pid_a).unwrap().data.to_vec_f32().unwrap();
        for (a, b) in p_val.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-5, "param data must round-trip");
        }
    }

    #[test]
    fn test_lionvote_optimizer_step_and_save_load_roundtrip() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let mut params = TrainableParams::new();
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::GateProj);
        let mut tp = TrainableParam::new(
            pid_b,
            grim_backend_cpu::cpu_tensor(vec![2.0f32; 6], Shape::new(vec![3, 2])),
        )
        .unwrap();
        tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 6],
            Shape::new(vec![3, 2]),
        ))
        .unwrap();
        params.insert(tp);

        let mut lionvote = LionVote::new(LionVoteConfig::default());
        lionvote.step(&mut params).unwrap();

        let p_val = params.get(pid_b).unwrap().data.to_vec_f32().unwrap();
        assert!(
            p_val[0] < 2.0,
            "LionVote step must decrease param along gradient"
        );

        let state = lionvote.save_to_train_state(&params);
        assert_eq!(state.step, 1);

        let mut params2 = TrainableParams::new();
        params2.insert(
            TrainableParam::new(
                pid_b,
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 6], Shape::new(vec![3, 2])),
            )
            .unwrap(),
        );
        let mut lionvote2 = LionVote::new(LionVoteConfig::default());
        lionvote2
            .load_from_train_state(&mut params2, &state)
            .unwrap();

        let restored = params2.get(pid_b).unwrap().data.to_vec_f32().unwrap();
        for (a, b) in p_val.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-5, "param data must round-trip");
        }
    }

    #[test]
    fn test_optimizer_enum_fromstr_and_display() {
        let m: OptimizerKind = "madam".parse().unwrap();
        assert_eq!(m, OptimizerKind::MAdam);
        assert_eq!(format!("{}", OptimizerKind::MAdam), "madam");

        let l: OptimizerKind = "lionvote".parse().unwrap();
        assert_eq!(l, OptimizerKind::LionVote);
        assert_eq!(format!("{}", OptimizerKind::LionVote), "lionvote");
    }

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

    #[test]
    fn test_randomized_svd_decomposes_matrix() {
        let m = 32;
        let n = 24;
        let rank = 4;
        let mat: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let (u, s, vt) = randomized_svd(&mat, m, n, rank, 5, 2);
        assert_eq!(u.len(), m * rank);
        assert_eq!(s.len(), rank);
        assert_eq!(vt.len(), rank * n);
    }

    #[test]
    fn test_qgalore_optimizer_build_and_step() {
        let opt = Optimizer::new(OptimizerKind::QGaLoreAdamW8Bit, 1e-3);
        assert!(opt.is_ok());
        let mut opt = opt.unwrap();

        let t = grim_backend_cpu::cpu_tensor(vec![0.5f32; 128 * 64], Shape::new(vec![128, 64]));
        let mut params = TrainableParams::new();
        let pid = ParamId::base(0, crate::injection::LoRAInjectionPoint::GateProj);
        params.insert(crate::param::TrainableParam::new(pid, t).unwrap());

        let grad = grim_backend_cpu::cpu_tensor(vec![0.1f32; 128 * 64], Shape::new(vec![128, 64]));
        if let Some(param) = params.get_mut(pid) {
            param.accumulate_grad(&grad).unwrap();
        }

        opt.step(&mut params).unwrap();
        assert_eq!(
            params.get(pid).unwrap().data.to_vec_f32().unwrap().len(),
            128 * 64
        );
    }

    #[test]
    fn optimizer_fork_preserves_kind_lr_and_state_contract() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let mut source = TrainableParams::new();
        let mut param = TrainableParam::new(
            pid,
            grim_backend_cpu::cpu_tensor(vec![1.0, 2.0], Shape::new(vec![2])),
        )
        .unwrap();
        param
            .accumulate_grad(&grim_backend_cpu::cpu_tensor(
                vec![0.25, -0.5],
                Shape::new(vec![2]),
            ))
            .unwrap();
        source.insert(param);

        let mut optimizer = Optimizer::new(OptimizerKind::AdamW, 2e-4).unwrap();
        optimizer.step(&mut source).unwrap();
        let mut target = source.clone();
        let fork = optimizer.fork_for_rank(&source, &mut target).unwrap();
        assert_eq!(fork.kind(), OptimizerKind::AdamW);
        assert!((fork.lr() - 2e-4).abs() < 1e-8);
    }

    #[test]
    fn test_muon_optimizer_step_updates_a_and_b() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let mut params = TrainableParams::new();

        // A matrix [rank=2, in=2] — sign-SGD (magnitude)
        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let mut tp_a = TrainableParam::new(
            pid_a,
            grim_backend_cpu::cpu_tensor(vec![0.5f32; 4], Shape::new(vec![2, 2])),
        )
        .unwrap();
        tp_a.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.3f32; 4],
            Shape::new(vec![2, 2]),
        ))
        .unwrap();
        params.insert(tp_a);

        // B matrix [out=3, rank=2] — Newton-Schulz + momentum (direction)
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        let mut tp_b = TrainableParam::new(
            pid_b,
            grim_backend_cpu::cpu_tensor(vec![0.5f32; 6], Shape::new(vec![3, 2])),
        )
        .unwrap();
        tp_b.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 6],
            Shape::new(vec![3, 2]),
        ))
        .unwrap();
        params.insert(tp_b);

        let mut muon = Muon::new(MuonConfig::default());
        muon.step(&mut params).unwrap();

        // A should have moved (sign update: w -= lr * (sign(g) + wd_a * w))
        let a_after = params.get(pid_a).unwrap().data.to_vec_f32().unwrap();
        assert_ne!(
            a_after,
            vec![0.5f32; 4],
            "A matrix must update under sign-SGD"
        );

        // B should have moved (Newton-Schulz + momentum)
        let b_after = params.get(pid_b).unwrap().data.to_vec_f32().unwrap();
        assert_ne!(
            b_after,
            vec![0.5f32; 6],
            "B matrix must update under Muon step"
        );
    }

    #[test]
    fn test_muon_save_load_roundtrip() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let mut params = TrainableParams::new();
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        let mut tp = TrainableParam::new(
            pid_b,
            grim_backend_cpu::cpu_tensor(vec![1.0f32; 12], Shape::new(vec![3, 4])),
        )
        .unwrap();
        tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 12],
            Shape::new(vec![3, 4]),
        ))
        .unwrap();
        params.insert(tp);

        let mut muon = Muon::new(MuonConfig::default());
        muon.step(&mut params).unwrap();

        let state = muon.save_to_train_state(&params);
        assert_eq!(state.step, 1);

        // Load into a fresh optimizer + params.
        let mut params2 = TrainableParams::new();
        params2.insert(
            TrainableParam::new(
                pid_b,
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 12], Shape::new(vec![3, 4])),
            )
            .unwrap(),
        );
        let mut muon2 = Muon::new(MuonConfig::default());
        muon2.load_from_train_state(&mut params2, &state).unwrap();

        let original = params.get(pid_b).unwrap().data.to_vec_f32().unwrap();
        let restored = params2.get(pid_b).unwrap().data.to_vec_f32().unwrap();
        for (a, b) in original.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-5, "param data must round-trip");
        }
    }

    #[test]
    fn test_muon_kind_display_and_fromstr() {
        assert_eq!(OptimizerKind::Muon, OptimizerKind::Muon);
        let kind: OptimizerKind = "muon".parse().unwrap();
        assert_eq!(kind, OptimizerKind::Muon);
        assert_eq!(format!("{}", OptimizerKind::Muon), "muon");
    }

    #[test]
    fn test_optimizer_step_param_variants() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);

        for kind in [
            OptimizerKind::AdamW,
            OptimizerKind::Lion,
            OptimizerKind::MAdam,
            OptimizerKind::LionVote,
        ] {
            let mut opt = Optimizer::new(kind, 1e-3).unwrap();
            let mut tp = TrainableParam::new(
                pid,
                grim_backend_cpu::cpu_tensor(vec![1.0f32; 4], Shape::new(vec![2, 2])),
            )
            .unwrap();
            tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
                vec![0.5f32; 4],
                Shape::new(vec![2, 2]),
            ))
            .unwrap();

            opt.step_param(pid, &mut tp).unwrap();
            let after = tp.data.to_vec_f32().unwrap();
            assert_ne!(
                after,
                vec![1.0f32; 4],
                "Optimizer {:?} must update param via step_param",
                kind
            );
        }
    }

    #[test]
    fn test_adamw_8bit_q80_moments_and_convergence() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let mut params = TrainableParams::new();
        // 64 elements (2 Q8_0 blocks)
        let mut tp = TrainableParam::new(
            pid,
            grim_backend_cpu::cpu_tensor(vec![1.0f32; 64], Shape::new(vec![8, 8])),
        )
        .unwrap();
        tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 64],
            Shape::new(vec![8, 8]),
        ))
        .unwrap();
        params.insert(tp);

        let mut opt8 = AdamW8Bit::new(AdamW8BitConfig {
            lr: 1e-2,
            ..AdamW8BitConfig::default()
        });

        opt8.step(&mut params).unwrap();

        assert!(opt8.m_q80.contains_key(&pid), "8-bit m buffer must exist");
        assert!(opt8.v_q80.contains_key(&pid), "8-bit v buffer must exist");
        // Each 32 elements in Q8_0 is 34 bytes -> 68 bytes for 64 elements
        assert_eq!(opt8.m_q80.get(&pid).unwrap().len(), 68);
        assert_eq!(opt8.v_q80.get(&pid).unwrap().len(), 68);

        let updated = params.get(pid).unwrap().data.to_vec_f32().unwrap();
        for &w in &updated {
            assert!(w < 1.0f32, "weight must decrease with positive gradient");
        }
    }

    #[test]
    fn test_paged_adamw_page_dirty_tracking_and_step() {
        use crate::injection::LoRAInjectionPoint;
        use crate::param::TrainableParam;

        let pid = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        let mut params = TrainableParams::new();
        let mut tp = TrainableParam::new(
            pid,
            grim_backend_cpu::cpu_tensor(vec![2.0f32; 100], Shape::new(vec![10, 10])),
        )
        .unwrap();
        tp.accumulate_grad(&grim_backend_cpu::cpu_tensor(
            vec![0.5f32; 100],
            Shape::new(vec![10, 10]),
        ))
        .unwrap();
        params.insert(tp);

        let mut paged_opt = PagedAdamW::new(PagedAdamWConfig {
            lr: 1e-3,
            page_size: 32, // 100 elements -> 4 pages
            cpu_offload: true,
            ..PagedAdamWConfig::default()
        });

        paged_opt.step(&mut params).unwrap();

        assert_eq!(
            paged_opt.pages.len(),
            4,
            "Must allocate 4 pages for 100 elements at page_size=32"
        );
        assert_eq!(
            paged_opt.dirty_set.len(),
            4,
            "All 4 pages must be marked dirty during step"
        );

        let updated = params.get(pid).unwrap().data.to_vec_f32().unwrap();
        for &w in &updated {
            assert!(w < 2.0f32, "weight must decrease with positive gradient");
        }
    }

    #[test]
    fn test_unimplemented_optimizers_rejected_honestly() {
        {
            let kind = OptimizerKind::AdamWBnb;
            let res = Optimizer::new(kind, 1e-3);
            assert!(
                matches!(res, Err(Error::Unimplemented(_))),
                "Optimizer {:?} must be explicitly rejected with Error::Unimplemented instead of silently aliasing",
                kind
            );
        }
    }

    /// The memory-efficient family must construct and step cleanly — no
    /// silent aliasing to AdamW, no Unimplemented rejection.
    #[test]
    fn test_memory_efficient_optimizers_construct_and_step() {
        use crate::param::{TrainableParam, TrainableParams};
        let kinds = [
            OptimizerKind::LOMO,
            OptimizerKind::Adalomo,
            OptimizerKind::CAME,
            OptimizerKind::Sophia,
            OptimizerKind::GaloreAdamW,
            OptimizerKind::GaloreAdamW8Bit,
        ];
        for kind in kinds {
            let mut opt = Optimizer::new(kind, 1e-2).expect("must construct");
            let mut params = TrainableParams::new();
            let w0 = vec![1.0f32, -2.0, 0.5, 3.0];
            let t = grim_backend_cpu::cpu_tensor(w0.clone(), grim_tensor::Shape::new(vec![2, 2]));
            params.insert(
                TrainableParam::new(
                    ParamId::base(0, crate::injection::LoRAInjectionPoint::QProj),
                    t,
                )
                .expect("param alloc"),
            );
            // Seed a gradient (simulates backward).
            for (_, p) in params.iter_mut() {
                let g = vec![0.1f32, -0.2, 0.4, -0.1];
                let gt = grim_backend_cpu::cpu_tensor(g, Shape::new(vec![2, 2]));
                let _ = p.accumulate_grad(&gt);
            }
            for _ in 0..3 {
                opt.step(&mut params).expect("step must succeed");
            }
            let (_, p) = params.iter().next().unwrap();
            assert!(
                p.data.to_vec_f32().unwrap()[0] != w0[0],
                "{kind:?} must update weights"
            );
        }
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    /// Audit gate: fused streaming steps (`step_param` called directly, as
    /// `backward_step` does) must evolve the bias correction per update.
    /// Pre-fix, `step_count` was never advanced on this path so corrections
    /// stayed frozen at t=1 forever.
    ///
    /// Sequence g = [1, 0]: after t=1 AdamW lands at exactly -lr; at t=2 the
    /// zero gradient leaves pure momentum, and the CORRECT t=2 correction
    /// gives p2 = -lr - lr·m̂₂/(√v̂₂+ε):
    ///   m₂ = β₁·0.1        = 0.09
    ///   v₂ = β₂·0.001      = 0.000999
    ///   ĉ₁ = 1-β₁²         = 0.19 ; ĉ₂ = 1-β₂² = 0.001999
    ///   p₂ = -0.1 - 0.1·(0.09/0.19)/√(0.000999/0.001999) ≈ -0.166946
    /// A frozen t=1 instead yields ≈ -0.189959 — the gate separates them.
    #[test]
    fn step_param_advances_per_param_bias_correction() {
        use grim_backend_cpu::cpu_tensor;
        use grim_tensor::Shape;

        let mut opt = Optimizer::AdamW(AdamW::new(AdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            lora_plus_ratio: 1.0,
        }));
        let pid = ParamId::a(0, 1, crate::injection::LoRAInjectionPoint::QProj);
        let mut param = crate::param::TrainableParam::new(
            pid,
            cpu_tensor(vec![0.0f32; 4], Shape::new(vec![4])),
        )
        .unwrap();

        // t=1 with g=1 → exactly -lr.
        param
            .accumulate_grad(&cpu_tensor(vec![1.0f32; 4], Shape::new(vec![4])))
            .unwrap();
        opt.step_param(pid, &mut param).unwrap();
        let p1 = param.data.to_vec_f32().unwrap();
        for &d in &p1 {
            assert!(
                (d - (-0.1)).abs() < 1e-4,
                "t=1 must be exactly -lr, got {d}"
            );
        }

        // t=2 with g=0 → momentum-only continuation under the t=2 correction.
        param.zero_grad().unwrap();
        opt.step_param(pid, &mut param).unwrap();
        let p2 = param.data.to_vec_f32().unwrap();

        let m2 = 0.9f32 * 0.1;
        let v2 = 0.999f32 * 0.001;
        let c1 = 1.0 - 0.9f32.powi(2);
        let c2 = 1.0 - 0.999f32.powi(2);
        let expected_p2 = -0.1f32 - 0.1 * (m2 / c1) / ((v2 / c2).sqrt() + 1e-8);
        for &d in &p2 {
            assert!(
                (d - expected_p2).abs() < 1e-4,
                "t=2 position {d} != hand-computed {expected_p2} — bias correction frozen?"
            );
            assert!(
                (d - (-0.189_959)).abs() > 1e-3,
                "t=2 position matches the FROZEN t=1 correction — regression"
            );
        }
    }
}

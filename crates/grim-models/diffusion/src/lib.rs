//! Diffusion model for Grim — UNet/DiT + noise schedulers. Implements the
//! `grim_core::model::DiffusionModel` trait per §4.4.
//!
//! `DiffusionModel::denoise_step` returns the predicted noise (epsilon-
//! prediction) or velocity (v-prediction, configurable). The companion
//! `NoiseScheduler` (`DDIM`, `Euler`) wraps the iterative loop a real
//! sampler runs.

pub mod scheduler;
pub mod unet;

pub use scheduler::{DdimScheduler, EulerScheduler};
pub use unet::{Unet2D, UnetConfig};

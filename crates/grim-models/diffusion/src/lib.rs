//! Diffusion models (Flux 2 MM-DiT, UNet 2D) and noise schedulers (FlowMatch Euler, DDIM).
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

pub mod flow_match;
pub mod flux2;
pub mod scheduler;
pub mod unet;
pub mod vae;

pub use flow_match::{FlowMatchEulerConfig, FlowMatchEulerScheduler};
pub use flux2::{Flux2Config, Flux2Transformer2D};
pub use scheduler::{DdimScheduler, EulerScheduler};
pub use unet::{Unet2D, UnetConfig};
pub use vae::{Flux2VAE, Flux2VaeConfig};

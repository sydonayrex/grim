//! `Model` trait family + capability traits.
//!
//! See architecture §4.4. The hard design problem — transformers, Mamba,
//! vision, audio, diffusion have genuinely different call shapes — is
//! solved by a small `Model` base trait plus capability traits (`CausalLm`,
//! `Encoder`, `EncoderDecoderLm`, `StatefulSequence`, `DiffusionModel`)
//! that models implement as applicable. A hybrid Mamba+attention model
//! just implements `CausalLm` and mixes SSM state internally inside
//! `forward`; the trait boundary is at the request level, not forced
//! down into every layer.

use grim_tensor::{ArithType, Device, Tensor};

use crate::error::Result;

/// Concrete dynamic model config — what every `load` constructor expects.
pub trait ModelConfig: Send + Sync {
    fn name(&self) -> &str;
    /// Return a coarse `Modality` tag for routing in the serving layer.
    /// Capability traits stay the source of truth — this is just a hint.
    fn modality(&self) -> ModalityHint;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Coarse modality hint for serving-side heuristics. Capability traits
/// (`CausalLm`, `Encoder`, etc.) remain authoritative — this enum only
/// powers request-routing shortcuts, not legality checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityHint {
    TextInTextOut,
    VisionEncoder,
    AudioEncoderDecoder,
    Diffusion,
}

/// Every model implements this. It says nothing about modality.
pub trait Model: Send + Sync {
    fn config(&self) -> &dyn ModelConfig;
    fn device(&self) -> &Device;
    /// Arithmetic type used for inner-product / softmax computation. Most
    /// backends compute in F32 or F16 regardless of how the weights are
    /// stored — this is the compute-time type, not the storage type.
    fn param_arith(&self) -> ArithType;
}

/// Autoregressive, token-level generation — dense transformers, Mamba, hybrids.
pub trait CausalLm: Model {
    fn new_session(&self) -> Box<dyn crate::session::SessionT>;
    fn forward(
        &self,
        session: &mut dyn crate::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor>;
}

/// Sequence-state models — Mamba/SSM/hybrid. These need an explicit state
/// cache instead of KV blocks. `init_state` allocates a fresh per-sequence
/// state; `step` advances it by one token (or a small chunk in chunked-step
/// variants).
pub trait StatefulSequence: Model {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState>;
    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor>;
}

/// Per-sequence SSM state. Cheap to init/drop because Mamba-style state is
/// O(model dimension) per sequence, not O(sequence-length) like KV.
pub trait SsmState: Send {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>>;
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()>;
}

/// Non-autoregressive encoders — vision towers, CLIP, audio encoders.
pub trait Encoder: Model {
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
}

/// Encoder-decoder, cross-attention-conditioned generation — Whisper-style
/// ASR. The encoder runs once; the decoder consumes encoder output via
/// cross-attention.
pub trait EncoderDecoderLm: Model {
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
    fn decode_step(
        &self,
        session: &mut dyn crate::session::SessionT,
        encoder_out: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor>;
}

/// Iterative denoising models — UNet/DiT diffusion.
pub trait DiffusionModel: Model {
    /// One denoising step. Returns the predicted noise (epsilon-prediction),
    /// velocity (v-prediction), or sample, depending on scheduler.
    fn denoise_step(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor>;
    fn scheduler(&self) -> &dyn crate::model::NoiseScheduler;
}

/// Noise scheduler for diffusion / flow models. Concrete impls: DDPM,
/// DDIM, Euler, DPM++, Karras — registry-driven so models bring their own.
pub trait NoiseScheduler: Send + Sync {
    fn step(&self, model_output: &Tensor, latents: &Tensor, timestep: u32) -> Result<Tensor>;
}

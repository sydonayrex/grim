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

/// Handle to a loaded adapter (LoRA weights + A/B rank + scaling factor).
/// Zero or more adapters may be active per request; the engine fuses their
/// low-rank updates into the base forward pass (Punica-style batched LoRA).
#[derive(Clone)]
pub struct AdapterHandle {
    pub id: u32,
    pub a: Tensor,
    pub b: Tensor,
    pub alpha: f32,
}

/// Concrete dynamic model config — what every `load` constructor expects.
pub trait ModelConfig: Send + Sync {
    fn name(&self) -> &str;
    /// Return a coarse `Modality` tag for routing in the serving layer.
    /// Capability traits stay the source of truth — this is just a hint.
    fn modality(&self) -> ModalityHint;
    /// Return the model's context window in tokens, or `0` if unknown.
    /// The server uses this to reject requests whose total token count
    /// (prompt + max_tokens) exceeds the model's context window.
    /// Default returns `0` (no enforcement for models that don't report it).
    fn context_length(&self) -> u64 {
        0
    }
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
    MultimodalInTextOut,
    TextToSpeech,
    VoiceConversion,
    AudioVocoder,
}

/// Inputs for multimodal causal models (text + vision patches + audio mel frames).
#[derive(Debug, Clone)]
pub struct MultimodalInputs {
    pub input_ids: Tensor,
    pub image_patches: Option<Tensor>,
    pub mel_frames: Option<Tensor>,
    pub image_placeholder_mask: Option<Vec<usize>>,
    pub audio_placeholder_mask: Option<Vec<usize>>,
}

/// Every model implements this. It says nothing about modality.
pub trait Model: Send + Sync {
    fn config(&self) -> &dyn ModelConfig;
    fn device(&self) -> &Device;
    /// Arithmetic type used for inner-product / softmax computation. Most
    /// backends compute in F32 or F16 regardless of how the weights are
    /// stored — this is the compute-time type, not the storage type.
    fn param_arith(&self) -> ArithType;
    /// Downcast to concrete type for debugging/inspection.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Autoregressive, token-level generation — dense transformers, Mamba, hybrids.
pub trait CausalLm: Model {
    fn new_session(&self) -> Box<dyn crate::session::SessionT>;
    fn forward(
        &self,
        session: &mut dyn crate::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor>;
}

/// Multimodal autoregressive generation — accepts text tokens + vision/audio inputs.
pub trait MultimodalCausalLm: CausalLm {
    fn forward_multimodal(
        &self,
        session: &mut dyn crate::session::SessionT,
        inputs: &MultimodalInputs,
        positions: &Tensor,
        adapters: &[AdapterHandle],
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
///
/// `as_any` is essential for downcasting to concrete state types — the
/// `StatefulSequence::step` impl needs to mutate state fields, which
/// requires a concrete reference.
pub trait SsmState: Send {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>>;
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
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
    fn denoise_step(&self, latents: &Tensor, timestep: &Tensor, cond: &Tensor) -> Result<Tensor>;
    fn scheduler(&self) -> &dyn crate::model::NoiseScheduler;
}

/// Noise scheduler for diffusion / flow models. Concrete impls: DDPM,
/// DDIM, Euler, DPM++, Karras — registry-driven so models bring their own.
pub trait NoiseScheduler: Send + Sync {
    fn step(&self, model_output: &Tensor, latents: &Tensor, timestep: u32) -> Result<Tensor>;
}

/// Text-to-speech synthesis models (e.g. Kokoro, StyleTTS2).
pub trait TextToSpeechModel: Model {
    /// Synthesizes raw audio waveform samples from input phoneme/token IDs and style conditioning.
    fn synthesize(&self, phoneme_ids: &[u32], style: &Tensor, speed: f32) -> Result<Tensor>;
}

/// Voice conversion and speech-to-speech models (e.g. MeanVC2, FastU2++).
pub trait VoiceConversionModel: Model {
    /// Converts source mel spectrogram / audio features into target voice audio features.
    fn convert_voice(&self, source_mel: &Tensor, target_style: &Tensor) -> Result<Tensor>;
}

/// Neural audio vocoders (e.g. Vocos, iSTFTNet, HiFi-GAN).
pub trait AudioVocoder: Model {
    /// Decodes intermediate mel-spectrogram or latent features into raw time-domain audio samples.
    fn mel_to_audio(&self, mel: &Tensor) -> Result<Tensor>;
}

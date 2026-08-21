//! Audio models architecture implementations (Kokoro-82M TTS, MeanVC2 Voice Conversion, Vocos Vocoder, Whisper ASR).
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

pub mod kokoro;
pub mod meanvc2;
pub mod vocos;
pub mod whisper;

pub use kokoro::{Kokoro, KokoroConfig};
pub use meanvc2::{MeanVC2, MeanVC2Config};
pub use vocos::{Vocos, VocosConfig};
pub use whisper::{Whisper, WhisperConfig};

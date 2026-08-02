//! Audio encoder-decoder architecture implementations (Whisper-style speech recognition).

pub mod whisper;

pub use whisper::{Whisper, WhisperConfig};

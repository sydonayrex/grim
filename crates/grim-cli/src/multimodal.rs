//! `grim multimodal` — Top-level routing shell for Vision, Audio, and Diffusion workflows.

use clap::Subcommand;
use grim_core::error::{Error, Result};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub enum MultimodalCmd {
    /// Vision model operations (image encoding and VLM embeddings).
    Vision {
        #[command(subcommand)]
        cmd: VisionCmd,
    },
    /// Audio model operations (speech-to-text transcription and audio tokens).
    Audio {
        #[command(subcommand)]
        cmd: AudioCmd,
    },
    /// Diffusion operations (text-to-image and latent diffusion generation).
    Diffusion {
        #[command(subcommand)]
        cmd: DiffusionCmd,
    },
}

#[derive(Subcommand, Debug)]
pub enum VisionCmd {
    /// Encode an image into embedding tensors using a Vision Transformer model.
    Encode {
        /// Path to input image file (PNG, JPEG, WebP).
        #[arg(short, long)]
        image: PathBuf,
        /// Model name or path (e.g. Qwen2-VL, CogVLM, Gemma-3N).
        #[arg(short, long)]
        model: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum AudioCmd {
    /// Transcribe speech audio into text using an ASR model.
    Transcribe {
        /// Path to input audio file (WAV, MP3, FLAC).
        #[arg(short, long)]
        audio: PathBuf,
        /// Model name or path (e.g. Whisper, WavTokenizer).
        #[arg(short, long)]
        model: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum DiffusionCmd {
    /// Generate an image from a text prompt using a diffusion model.
    Generate {
        /// Text prompt describing the desired image.
        #[arg(short, long)]
        prompt: String,
        /// Output path for the generated image (PNG).
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,
        /// Model name or path (e.g. Diffusion-Gemma, StableDiffusion).
        #[arg(short, long)]
        model: String,
    },
}

pub fn cmd_multimodal(cmd: MultimodalCmd) -> Result<()> {
    // Every arm validates inputs against the real catalog and fails with a
    // precise, actionable error. No arm prints a success banner for an
    // unwired pipeline (registry item 1 in fit-it-damn-you.md §9).
    match cmd {
        MultimodalCmd::Vision { cmd } => match cmd {
            VisionCmd::Encode { image, model } => {
                println!("=== Grim Multimodal: Vision ===");
                println!("Image Input : {}", image.display());
                println!("Model       : {model}");
                if !image.exists() {
                    return Err(Error::Config(format!(
                        "image file not found: {}",
                        image.display()
                    )));
                }
                let resolved = grim_core::catalog::resolve_model_preferring_grim(&model)
                    .ok_or_else(|| {
                        Error::Config(format!(
                            "vision model '{model}' not found in the catalog. The ViT/glimmer \
                             encoders load from GGUF checkpoints via WeightSource::root; no \
                             vision checkpoint with that name is installed. Pull one (e.g. a \
                             CLIP/ViT GGUF) and retry."
                        ))
                    })?;
                Err(Error::Config(format!(
                    "vision encoder pipeline is not wired in this build: model '{model}' \
                     resolved to '{}' but no ViT/glimmer forward is reachable from the CLI. \
                     The encoder structs (grim-models-vision Vit::load + encode_image) are \
                     real; wiring them to catalog checkpoints lands with the first vision \
                     model release.",
                    resolved.display()
                )))
            }
        },
        MultimodalCmd::Audio { cmd } => match cmd {
            AudioCmd::Transcribe { audio, model } => {
                println!("=== Grim Multimodal: Audio ===");
                println!("Audio Input : {}", audio.display());
                println!("Model       : {model}");
                if !audio.exists() {
                    return Err(Error::Config(format!(
                        "audio file not found: {}",
                        audio.display()
                    )));
                }
                let resolved = grim_core::catalog::resolve_model_preferring_grim(&model)
                    .ok_or_else(|| {
                        Error::Config(format!(
                            "ASR model '{model}' not found in the catalog. Transcription \
                             requires a Whisper-family GGUF; pull one and retry."
                        ))
                    })?;
                Err(Error::Config(format!(
                    "ASR pipeline is not wired in this build: '{}' resolved but the Whisper \
                     mel-encoder → decoder → detokenize path has no runtime caller. The \
                     whisper structs are real; wiring lands with the first ASR model release.",
                    resolved.display()
                )))
            }
        },
        MultimodalCmd::Diffusion { cmd } => match cmd {
            DiffusionCmd::Generate {
                prompt,
                output,
                model,
            } => {
                println!("=== Grim Multimodal: Diffusion ===");
                println!("Prompt      : \"{prompt}\"");
                println!("Output File : {}", output.display());
                println!("Model       : {model}");
                let resolved = grim_core::catalog::resolve_model_preferring_grim(&model)
                    .ok_or_else(|| {
                        Error::Config(format!(
                            "diffusion model '{model}' not found in the catalog. Generation \
                             requires a UNet + DDIM checkpoint; pull one and retry."
                        ))
                    })?;
                Err(Error::Config(format!(
                    "diffusion pipeline is not wired in this build: '{}' resolved but the \
                     Unet2D + DdimScheduler sampling loop has no runtime caller. The structs \
                     are real; wiring lands with the first diffusion model release.",
                    resolved.display()
                )))
            }
        },
    }
}

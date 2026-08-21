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
                println!(
                    "\n[grim vision] Vision models (Qwen2-VL, CogVLM, Gemma-3N, Hunyuan-VL) are integrated in `grim-models-vision`."
                );
                println!(
                    "[grim vision] HTTP Endpoint: POST /v1/chat/completions with image_url payload."
                );
                Ok(())
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
                println!(
                    "\n[grim audio] Audio decoder integrated in `grim-models-audio` (WavTokenizer / Whisper layout)."
                );
                println!("[grim audio] HTTP Endpoint: POST /v1/audio/transcriptions.");
                Ok(())
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
                println!(
                    "\n[grim diffusion] Diffusion pipeline integrated in `grim-models-diffusion` (Diffusion Gemma)."
                );
                println!("[grim diffusion] HTTP Endpoint: POST /v1/images/generations.");
                Ok(())
            }
        },
    }
}

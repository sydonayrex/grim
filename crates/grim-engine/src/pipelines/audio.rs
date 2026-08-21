use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AudioVocoder, TextToSpeechModel};
use grim_models_audio::{Kokoro, KokoroConfig, Vocos, VocosConfig};
use grim_tensor::{Device, Shape};
use grim_tensor::tensor::Tensor;

/// Configuration for the end-to-end audio pipeline.
#[derive(Debug, Clone)]
pub struct AudioPipelineConfig {
    pub sample_rate: usize,
    pub num_mel_bins: usize,
    pub hop_length: usize,
}

impl Default for AudioPipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_mel_bins: 100,
            hop_length: 256,
        }
    }
}

/// End-to-end Audio Pipeline orchestrating text encoding, mel synthesis, and vocoding.
pub struct AudioPipeline {
    pub kokoro: Kokoro,
    pub vocos: Vocos,
    pub config: AudioPipelineConfig,
    pub device: Device,
}

impl AudioPipeline {
    /// Create a new AudioPipeline instance.
    pub fn new(
        kokoro_config: &KokoroConfig,
        vocos_config: &VocosConfig,
        pipeline_config: AudioPipelineConfig,
        device: Device,
    ) -> Result<Self> {
        let kokoro = Kokoro::random(device.clone(), kokoro_config.clone());
        let vocos = Vocos::random(device.clone(), vocos_config.clone());

        Ok(Self {
            kokoro,
            vocos,
            config: pipeline_config,
            device,
        })
    }

    /// Synthesize raw audio waveform samples from input text token sequence and optional voice style.
    pub fn generate(&self, token_ids: &[u32], style_embed: Option<&Tensor>) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        let default_style = cpu_tensor(vec![0.0f32; self.kokoro.config.style_dim], Shape::new(vec![self.kokoro.config.style_dim]));
        let style = style_embed.unwrap_or(&default_style);

        // 1. Synthesize audio waveform via Kokoro
        let waveform_tensor = self.kokoro.synthesize(token_ids, style, 1.0)?;

        // 2. Extract float audio samples
        Ok(waveform_tensor.to_vec_f32()?)
    }

    /// Decode an arbitrary mel-spectrogram to raw audio waveform via Vocos.
    pub fn decode_mel(&self, mel_spec: &Tensor) -> Result<Vec<f32>> {
        let waveform_tensor = self.vocos.mel_to_audio(mel_spec)?;
        Ok(waveform_tensor.to_vec_f32()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_pipeline_instantiation() {
        let kokoro_cfg = KokoroConfig::default();
        let vocos_cfg = VocosConfig::default();
        let pipe_cfg = AudioPipelineConfig::default();

        let pipe = AudioPipeline::new(&kokoro_cfg, &vocos_cfg, pipe_cfg, Device::Cpu).unwrap();
        assert_eq!(pipe.config.sample_rate, 24000);
    }
}

//! End-to-end integration tests for the AudioPipeline (Kokoro + Vocos).

use grim_engine::pipelines::audio::{AudioPipeline, AudioPipelineConfig};
use grim_models_audio::{KokoroConfig, VocosConfig};
use grim_tensor::Device;

#[test]
fn test_audio_pipeline_full_synthesis() {
    let kokoro_cfg = KokoroConfig::default();
    let vocos_cfg = VocosConfig::default();
    let pipe_cfg = AudioPipelineConfig::default();

    let pipe = AudioPipeline::new(&kokoro_cfg, &vocos_cfg, pipe_cfg, Device::Cpu).unwrap();
    let tokens = vec![1, 15, 23, 42, 88];
    let audio_samples = pipe.generate(&tokens, None).unwrap();

    assert!(
        !audio_samples.is_empty(),
        "synthesized audio waveform must contain samples"
    );
}

use grim_backend_cpu::cpu_tensor;
use grim_core::model::{AudioVocoder, TextToSpeechModel, VoiceConversionModel};
use grim_models_audio::{Kokoro, KokoroConfig, MeanVC2, MeanVC2Config, Vocos, VocosConfig};
use grim_tensor::{Device, Shape};

#[test]
fn test_kokoro_tts_synthesis() {
    let cfg = KokoroConfig {
        vocab_size: 64,
        hidden_dim: 64,
        style_dim: 32,
        n_mels: 40,
        n_layers: 2,
        plbert_hidden: 64,
        plbert_layers: 2,
        plbert_heads: 4,
        plbert_ffn: 128,
        upsample_rates: vec![4, 2],
        upsample_kernel_sizes: vec![8, 4],
        hop_size: 4,
        n_fft: 16,
    };
    let kokoro = Kokoro::random(Device::Cpu, cfg);

    let phonemes = vec![1, 15, 30, 42, 5, 20];
    let style = cpu_tensor(vec![0.5f32; 32], Shape::new(vec![32]));
    let speed = 1.0f32;

    let waveform = kokoro
        .synthesize(&phonemes, &style, speed)
        .expect("TTS synthesis should succeed");
    assert_eq!(waveform.shape().dims().len(), 1);
    let total_samples = phonemes.len() * (4 * 2);
    assert_eq!(waveform.shape().dims()[0], total_samples);
    assert_eq!(waveform.to_vec_f32().unwrap().len(), total_samples);
}

#[test]
fn test_meanvc2_voice_conversion() {
    let cfg = MeanVC2Config {
        dim: 64,
        depth: 2,
        heads: 2,
        ff_mult: 2,
        bn_dim: 32,
        conv_layers: 2,
        chunk_size: 4,
        block_size: 2,
        n_mels: 40,
        style_dim: 32,
    };
    let meanvc = MeanVC2::random(Device::Cpu, cfg);

    let seq_len = 8;
    let source_mel = cpu_tensor(vec![0.1f32; seq_len * 40], Shape::new(vec![seq_len, 40]));
    let target_style = cpu_tensor(vec![0.2f32; 32], Shape::new(vec![32]));

    let converted_mel = meanvc
        .convert_voice(&source_mel, &target_style)
        .expect("Voice conversion should succeed");
    assert_eq!(converted_mel.shape().dims(), &[seq_len, 40]);
}

#[test]
fn test_vocos_neural_vocoder() {
    let cfg = VocosConfig {
        input_dim: 40,
        dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        n_fft: 64,
        hop_length: 16,
    };
    let vocos = Vocos::random(Device::Cpu, cfg);

    let num_frames = 10;
    let mel = cpu_tensor(
        vec![0.05f32; num_frames * 40],
        Shape::new(vec![num_frames, 40]),
    );

    let audio = vocos
        .mel_to_audio(&mel)
        .expect("Vocoder synthesis should succeed");
    assert_eq!(audio.shape().dims(), &[num_frames * 16]);
}

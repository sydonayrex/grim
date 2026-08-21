//! Weight-true loading of a real Vocos checkpoint through the engine loader.
//!
//! Skips silently when the checkpoint isn't present (CI without model files).

use grim_core::model::Model;
use grim_engine::model_loader::load_audio_model_from_path;
use grim_tensor::Device;

#[test]
fn vocos_pth_loads_real_weights() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/audio/MeanVC2/vocos.pt"
    );
    if !std::path::Path::new(path).exists() {
        return;
    }

    let model = load_audio_model_from_path(path, Device::Cpu)
        .expect("vocos.pt should load through the engine audio loader");

    let vocos = model
        .as_any()
        .downcast_ref::<grim_models_audio::Vocos>()
        .expect("loaded model should be a Vocos instance");

    // Config must be inferred from the checkpoint, not the 512-dim default.
    let cfg = vocos
        .config()
        .as_any()
        .downcast_ref::<grim_models_audio::VocosConfig>()
        .unwrap();
    assert_eq!(cfg.dim, 320);
    assert_eq!(cfg.input_dim, 80);
    assert_eq!(cfg.intermediate_dim, 1536);
    assert_eq!(cfg.num_layers, 8);
    assert_eq!(cfg.n_fft, 640);
    assert_eq!(cfg.hop_length, 320);

    // The learned iSTFT window is weight-true (a random-init fallback window
    // is the periodic Hann this checkpoint doesn't use).
    let any_nonzero = vocos.istft_window().iter().any(|v| v.abs() > 0.0);
    assert!(any_nonzero, "window should be materialized");
}

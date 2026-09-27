use grim_format::tprov::SplitGgufProvider;
use grim_models_transformer::qwen4exp_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};
use grim_nn::WeightSource;
use grim_tensor::Device;

#[test]
fn test_qwen4exp_loads_from_real_gguf_tensors() {
    let main_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf"
    );
    println!(
        "Checking if exists at {}: {}",
        main_path,
        std::path::Path::new(main_path).exists()
    );
    assert!(std::path::Path::new(main_path).exists());
    let provider = SplitGgufProvider::open(main_path).unwrap();
    let ws = WeightSource::root(&provider, Device::Cpu);
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.num_layers = 1;
    let model = Qwen38FlashNext::load(Device::Cpu, &ws, cfg);
    println!("Model load result: {:?}", model.is_ok());
    if let Err(e) = &model {
        println!("Error: {e}");
    }
    assert!(
        model.is_ok(),
        "Model must load without missing tensor errors: {:?}",
        model.err()
    );
}

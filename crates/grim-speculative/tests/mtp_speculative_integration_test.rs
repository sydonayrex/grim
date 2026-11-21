//! End-to-end integration test across Grim's crate stack:
//! `grim-nn::MtpHcBlock` -> `grim-models-transformer::Qwen38FlashNextMtp` ->
//! `grim-speculative::MtpProviderAdapter` -> `grim-speculative::SpeculativeCausalLm` + `MayVerifyGate`.

use std::sync::{Arc, Mutex};

use grim_backend_cpu::cpu_tensor;
use grim_core::CausalLm;
use grim_models_transformer::native_mtp::Qwen38FlashNextMtp;
use grim_models_transformer::qwen4exp_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};
use grim_speculative::{MtpProviderAdapter, NativeMtp, SpeculativeCausalLm};
use grim_tensor::{Device, Shape};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn generic_mtp_provider_adapter_drives_flash_next_hc_draft1_draft2_and_may_verify() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev_hc = std::env::var("GRIM_MTP_HC").ok();
    let prev_mv = std::env::var("GRIM_SPEC_MAY_VERIFY").ok();
    unsafe {
        std::env::set_var("GRIM_MTP_HC", "1");
        std::env::set_var("GRIM_SPEC_MAY_VERIFY", "1");
    }

    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.hc_count = 2;
    cfg.num_layers = 0;
    cfg.layer_types = vec![];
    cfg.attention_compress_ratios = vec![];
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;

    let base = Qwen38FlashNext::random(Device::Cpu, cfg.clone());
    let mtp_model = Arc::new(Qwen38FlashNextMtp::new_random_hc(base, 2));
    let adapter = Arc::new(MtpProviderAdapter::new(mtp_model.clone()));
    assert_eq!(adapter.mtp_depth(), 2, "must expose depth=2 (draft1 + draft2)");

    let wrapped = SpeculativeCausalLm::with_native_mtp(
        Box::new(MtpProviderAdapter::new(mtp_model)),
        adapter,
    );
    let mut sess = wrapped.new_session();
    sess.set_sampling_greedy(true);

    // Prefill seeds pending draft1+draft2 via MtpHcBlock
    let prefill_ids = cpu_tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let prefill_pos = cpu_tensor(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));
    let prefill_logits = wrapped
        .forward(sess.as_mut(), &prefill_ids, &prefill_pos, &[])
        .expect("prefill through SpeculativeCausalLm");
    assert_eq!(prefill_logits.shape().dims(), &[3, cfg.vocab_size]);
    assert!(sess.get_last_wide_hidden_state().is_some());

    // Decode step exercises batched verify + MayVerifyGate observation
    let dec_ids = cpu_tensor(vec![4.0], Shape::new(vec![1]));
    let dec_pos = cpu_tensor(vec![3.0], Shape::new(vec![1]));
    let dec_logits = wrapped
        .forward(sess.as_mut(), &dec_ids, &dec_pos, &[])
        .expect("decode through SpeculativeCausalLm");
    assert_eq!(dec_logits.shape().dims().last().copied(), Some(cfg.vocab_size));

    unsafe {
        match prev_hc {
            Some(v) => std::env::set_var("GRIM_MTP_HC", v),
            None => std::env::remove_var("GRIM_MTP_HC"),
        }
        match prev_mv {
            Some(v) => std::env::set_var("GRIM_SPEC_MAY_VERIFY", v),
            None => std::env::remove_var("GRIM_SPEC_MAY_VERIFY"),
        }
    }
}

//! Q9 tests for `Qwen38FlashNextMtp::verify_draft_tokens` and `GRIM_FLASH_MTP`.

use std::sync::Mutex;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_core::CausalLm;
use grim_tensor::{Device, Shape};

use grim_models_transformer::native_mtp::{flash_mtp_enabled, Qwen38FlashNextMtp};
use grim_models_transformer::qwen4exp_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn flash_mtp_env_flag_defaults_off() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var("GRIM_FLASH_MTP").ok();
    unsafe { std::env::remove_var("GRIM_FLASH_MTP") };
    assert!(!flash_mtp_enabled());
    unsafe { std::env::set_var("GRIM_FLASH_MTP", "1") };
    assert!(flash_mtp_enabled());
    unsafe {
        match prev {
            Some(v) => std::env::set_var("GRIM_FLASH_MTP", v),
            None => std::env::remove_var("GRIM_FLASH_MTP"),
        }
    }
}

#[test]
fn verify_draft_tokens_returns_full_logits_under_last_row_lmhead_and_restores_flag() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var("GRIM_FLASH_LAST_ROW_LMHEAD").ok();
    unsafe { std::env::set_var("GRIM_FLASH_LAST_ROW_LMHEAD", "1") };

    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 0;
    cfg.layer_types = vec![];
    cfg.attention_compress_ratios = vec![];
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;

    let base = Qwen38FlashNext::random(Device::Cpu, cfg.clone());
    let mtp = Qwen38FlashNextMtp::new_random(base, 3);
    let mut sess = mtp.new_session();
    assert!(!sess.full_output_logits());

    let ids = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    let pos = cpu_tensor(vec![0.0, 1.0, 2.0, 3.0], Shape::new(vec![4]));
    let verify_logits = mtp.verify_draft_tokens(sess.as_mut(), &ids, &pos, &[])?;
    assert_eq!(
        verify_logits.shape().dims(),
        &[4, cfg.vocab_size],
        "verify_draft_tokens must return all K+1 logit rows even when GRIM_FLASH_LAST_ROW_LMHEAD=1"
    );
    assert!(
        !sess.full_output_logits(),
        "verify_draft_tokens must restore previous full_output_logits=false"
    );

    unsafe {
        match prev {
            Some(v) => std::env::set_var("GRIM_FLASH_LAST_ROW_LMHEAD", v),
            None => std::env::remove_var("GRIM_FLASH_LAST_ROW_LMHEAD"),
        }
    }
    Ok(())
}

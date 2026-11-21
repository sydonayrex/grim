//! TDD tests for Grim-native wide-stream Hyper-Connection MTP (`grim-nn::mtp_hc`).

use std::sync::Mutex;
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_nn::{mtp_hc_enabled, Linear, MtpHcBlock, MtpHcCarry};
use grim_tensor::{Device, Shape};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lcg_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
        out.push(u * scale);
    }
    out
}

#[test]
fn mtp_hc_env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var("GRIM_MTP_HC").ok();
    unsafe { std::env::remove_var("GRIM_MTP_HC") };
    assert!(!mtp_hc_enabled());
    unsafe { std::env::set_var("GRIM_MTP_HC", "1") };
    assert!(mtp_hc_enabled());
    unsafe {
        match prev {
            Some(v) => std::env::set_var("GRIM_MTP_HC", v),
            None => std::env::remove_var("GRIM_MTP_HC"),
        }
    }
}

#[test]
fn mtp_hc_draft1_draft2_draft3_carry_isolation_on_cpu() {
    let hc = 4usize;
    let hidden = 16usize;
    let vocab = 64usize;
    let block = MtpHcBlock::seeded(hc, hidden, 1e-6, 101);
    let embed_w = cpu_tensor(lcg_vec(202, vocab * hidden, 0.2), Shape::new(vec![vocab, hidden]));
    let lm_head = Linear::from_tensor(
        cpu_tensor(lcg_vec(303, vocab * hidden, 0.2), Shape::new(vec![vocab, hidden])),
        None,
    );
    let trunk_wide = cpu_tensor(lcg_vec(404, hc * hidden, 0.2), Shape::new(vec![1, hc * hidden]));

    let mut carry = MtpHcCarry::new(&Device::Cpu, hc, hidden).expect("carry alloc");
    let (toks, logits) = block
        .draft_k_tokens(
            &trunk_wide,
            7,
            10,
            3,
            &embed_w,
            vocab,
            &lm_head,
            &mut carry,
            |wide, _pos| Ok(wide.clone()),
        )
        .expect("draft_k_tokens depth=3 must succeed");
    assert_eq!(toks.len(), 3, "must produce draft1, draft2, draft3");
    assert_eq!(logits.len(), 3);
    assert!(toks.iter().all(|&t| (t as usize) < vocab));

    // Deterministic repeat on fresh carry produces identical draft1..draft3
    let mut carry2 = MtpHcCarry::new(&Device::Cpu, hc, hidden).expect("carry2 alloc");
    let (toks2, _) = block
        .draft_k_tokens(
            &trunk_wide,
            7,
            10,
            3,
            &embed_w,
            vocab,
            &lm_head,
            &mut carry2,
            |wide, _pos| Ok(wide.clone()),
        )
        .expect("repeat draft_k_tokens");
    assert_eq!(toks, toks2);
}

#[ignore = "ROCm Card 1 (gfx1200) MtpHcBlock draft1..draft3 parity vs CPU & 3-run A/B"]
#[test]
fn mtp_hc_draft_rdna4_parity_and_bench() {
    let dev = Device::Rocm(0);
    let hc = 4usize;
    let hidden = 256usize;
    let vocab = 1024usize;

    let block_cpu = MtpHcBlock::seeded(hc, hidden, 1e-6, 555);
    let block_gpu = block_cpu.to_device(&dev).expect("move MtpHcBlock to ROCm");

    let embed_cpu = cpu_tensor(lcg_vec(666, vocab * hidden, 0.1), Shape::new(vec![vocab, hidden]));
    let embed_gpu = grim_nn::modules::move_to_device(&embed_cpu, &dev).expect("embed to ROCm");

    let lm_w_cpu = cpu_tensor(lcg_vec(777, vocab * hidden, 0.1), Shape::new(vec![vocab, hidden]));
    let lm_cpu = Linear::from_tensor(lm_w_cpu.clone(), None);
    let lm_gpu = Linear::from_tensor(
        grim_nn::modules::move_to_device(&lm_w_cpu, &dev).expect("lm_head to ROCm"),
        None,
    );

    let wide_cpu = cpu_tensor(lcg_vec(888, hc * hidden, 0.2), Shape::new(vec![1, hc * hidden]));
    let wide_gpu = grim_nn::modules::move_to_device(&wide_cpu, &dev).expect("wide to ROCm");

    let mut carry_cpu = MtpHcCarry::new(&Device::Cpu, hc, hidden).expect("cpu carry");
    let (toks_cpu, logits_cpu) = block_cpu
        .draft_k_tokens(
            &wide_cpu,
            5,
            0,
            3,
            &embed_cpu,
            vocab,
            &lm_cpu,
            &mut carry_cpu,
            |w, _| Ok(w.clone()),
        )
        .expect("cpu draft");

    let mut carry_gpu = MtpHcCarry::new(&dev, hc, hidden).expect("gpu carry");
    let (toks_gpu, logits_gpu) = block_gpu
        .draft_k_tokens(
            &wide_gpu,
            5,
            0,
            3,
            &embed_gpu,
            vocab,
            &lm_gpu,
            &mut carry_gpu,
            |w, _| Ok(w.clone()),
        )
        .expect("gpu draft");
    grim_backend_rocm::RocmDevice::shared(0).synchronize();

    assert_eq!(toks_gpu, toks_cpu, "RDNA4 draft1..draft3 tokens must match CPU");
    let mut max_diff = 0.0f32;
    for (lc, lg) in logits_cpu.iter().zip(logits_gpu.iter()) {
        let vc = lc.to_vec_f32().expect("vc");
        let vg = lg.to_vec_f32().expect("vg");
        for (a, b) in vc.iter().zip(vg.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    eprintln!("[MTP-HC-PARITY] RDNA4 draft1..draft3 tokens={toks_gpu:?}, max_logit_diff={max_diff:.6e}");
    assert!(max_diff < 1e-3, "max_logit_diff={max_diff} >= 1e-3");

    let mut runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        let _ = block_gpu
            .draft_k_tokens(
                &wide_gpu,
                5,
                0,
                3,
                &embed_gpu,
                vocab,
                &lm_gpu,
                &mut carry_gpu,
                |w, _| Ok(w.clone()),
            )
            .expect("bench draft");
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    runs.sort_by(|a, b| a.total_cmp(b));
    eprintln!("[MTP-HC-BENCH] RDNA4 3-draft (hc=4, H=256) runs={runs:?} ms, median={:.3} ms", runs[1]);
}

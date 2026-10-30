//! Packed-Q8_0 GEMM parity at production shapes (27B regression gate).
//!
//! Root cause of the 27B "finite-but-wrong logits" issue: the Q8_0 dot4
//! kernels (`grim_dot4_q80_*_gemv`) fold B through `grim_sdot4`, which on
//! gfx1201 lowers to the UNSIGNED `V_DOT4_I32_IU8` dot (the I8 dot the
//! builtin names is absent on RDNA3/4 — see the SPEED-ROC comment at
//! `device/quant/mod.rs:846`). Q4K/Q5K/Q6K survive because their codes are
//! all < 128 (sign-agnostic, as their kernel comments note). Q8_0 codes are
//! signed (-128..127): every negative code reads back +256 too large, so
//! every output of every packed-Q8_0 GEMM is garbage — while finite, and
//! while the N=32 ssm_alpha gates stay green (they now load f32 post
//! `GRIM_SSM_AB_F32`, and never covered wide N anyway).
//!
//! The 27B Unsloth quant is the first checkpoint in the corpus that keeps
//! packed Q8_0 weights in the forward path (`blk.{11,27,31,51,55,59,63}`
//! `attn_{k,v}`); the 9B has none, which is why it was unaffected. Layer 11
//! is the first divergence (CPU-vs-GPU per-layer fingerprints agree to 1e-3
//! through layer 10, then 0.73 at layer 11).
//!
//! Gate: `blk.11.attn_v` (Q8_0, [1024, 5120]) device GEMM at prefill M=5
//! against a host dequant+matvec reference. Must stay <= 8e-2 (activation
//! quantization noise is ~1e-3; the dot4 defect measured 6.5).
//!
//! Gated: `GRIM_GPU_TEST=1` + the 27B checkpoint on disk. Skips cleanly.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config};
use grim_tensor::{Device, Shape};

fn model_path() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root
            .join(up)
            .join("models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = 248320;
    c.hidden_size = 5120;
    c.num_heads = 24;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 64;
    c.intermediate_size = 17408;
    c.rope_theta = 10_000_000.0;
    c.full_attention_interval = 4;
    c.ssm_d_conv = 4;
    c.ssm_d_state = 128;
    c.ssm_dt_rank = 48;
    c.ssm_n_group = 16;
    c.ssm_d_inner = 48 * 128;
    c.rotary_dim = Some(64);
    c.devices = Vec::new();
    c
}

/// y = W x with W stored [out, in], f64 accumulation.
fn matvec(x: &[f32], wt: &[f32], in_n: usize, out_n: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_n);
    let mut y = vec![0.0f32; out_n];
    for (o, yv) in y.iter_mut().enumerate() {
        let mut s = 0.0f64;
        for (i, xv) in x.iter().enumerate() {
            s += (*xv as f64) * (wt[o * in_n + i] as f64);
        }
        *yv = s as f32;
    }
    y
}

#[test]
fn qwen38_packed_q80_gemm_parity_at_production_width() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 27B checkpoint not found");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8")).expect("open 27B");
    let c = cfg();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);

    let ws_gpu = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.11");
    let bg = Qwen35Block::load_tp(&ws_gpu, &c, 11, grim_nn::TensorParallelConfig::default())
        .expect("load blk.11 on GPU");
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.11");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 11, grim_nn::TensorParallelConfig::default())
        .expect("load blk.11 on CPU");
    assert!(bg.is_full_attention, "blk.11 must be full-attention");
    let (Some(g), Some(cl)) = (&bg.wv, &bc.wv) else {
        panic!("blk.11 attn_v absent");
    };
    assert!(
        matches!(
            cl.weight().dtype().storage,
            grim_tensor::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80)
        ),
        "gate requires blk.11.attn_v to be packed Q8_0"
    );

    const S: usize = 5;
    const HIDDEN: usize = 5120;
    const NV_OUT: usize = 4 * 256;
    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(&x, &Shape::new(vec![S, HIDDEN]), grim_tensor::DType::F32)
                .expect("upload"),
        ),
        Shape::new(vec![S, HIDDEN]),
        grim_tensor::DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let got = g
        .forward(&x_t)
        .expect("attn_v forward")
        .to_vec_f32()
        .expect("read");

    let t = cl.weight();
    let d = t.shape().dims().to_vec();
    assert_eq!(d, vec![NV_OUT, HIDDEN]);
    let wd = t.to_vec_f32().expect("host dequant");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for s in 0..S {
        let want = matvec(&x[s * HIDDEN..(s + 1) * HIDDEN], &wd, HIDDEN, NV_OUT);
        for (i, (&gg, &ww)) in got[s * NV_OUT..(s + 1) * NV_OUT]
            .iter()
            .zip(want.iter())
            .enumerate()
        {
            let diff = (gg - ww).abs();
            let rel = diff / ww.abs().max(1e-3);
            if diff.min(rel) > worst {
                worst = diff.min(rel);
                at = s * NV_OUT + i;
            }
        }
    }
    eprintln!("[q80-parity] blk.11.attn_v Q8_0 [1024x5120] M=5: worst {worst:.3e} at {at}");
    assert!(
        worst <= 8e-2,
        "packed-Q8_0 GEMM disagrees with host reference: worst {worst:.3e} at {at} \
         (the Q8_0 dot4 path misreads signed codes; see module docs)"
    );
}

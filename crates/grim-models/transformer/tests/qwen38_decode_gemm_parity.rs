//! Decode-shape GEMM parity gates (27B follow-up to the step-0 Q8_0 fix).
//!
//! Two independent M=1 defects corrupted every decode step while prefill
//! stayed green:
//!
//! 1. IQ WMMA grid coverage (`gemm_launchers.rs`): the shared
//!    `launch_wmma_fused_dequant_quant` launcher grids N/64, but the
//!    explicit IQ kernels (`wmma_iq_gemm.rs`) tile 16-wide, so cols
//!    >= 16*ceil(N/64) were never written (measured: IQ4_NL ffn_down
//!    [5120,17408] computed only cols 0..1280). Fixed by passing the
//!    kernel tile width to the launcher (16 for IQ, 64 for the stamped
//!    K-quant kernels).
//!
//! 2. u4-lane GEMV for Q5K (`dequant_k_quants.rs::try_u4_lane_decode`):
//!    steady-state ~0.2 off vs host at M=1 on Qwen3.8-27B and Qwen3.5-9B
//!    Q5K weights alike (warmed, sleep-invariant), while Q4K/Q6K sit at
//!    ~3e-4 through the same kernel. Q5K decode is routed to the exact
//!    dot4/WMMA legs instead; the M>1 u4-GEMM route is unaffected
//!    (verified ~2e-3) and stays.
//!
//! Gate: `blk.3.ffn_down` (IQ4_NL, [5120,17408]) full-column coverage at
//! M=1, and `blk.3.attn_q` (Q5K, [12288,5120]) at M=1, both against host
//! dequant+matvec references. Tolerances admit activation quantization
//! noise, not wrong kernels.
//!
//! Gated: `GRIM_GPU_TEST=1` + the 27B checkpoint on disk. Skips cleanly.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config};
use grim_nn::modules::Linear;
use grim_tensor::{Device, Shape};

const HIDDEN: usize = 5120;
const INTER: usize = 17408;

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
    c.hidden_size = HIDDEN;
    c.num_heads = 24;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 64;
    c.intermediate_size = INTER;
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

fn lin_w(l: &Linear) -> (Vec<f32>, usize, usize) {
    let t = l.weight();
    let d = t.shape().dims().to_vec();
    assert_eq!(d.len(), 2);
    (t.to_vec_f32().expect("weight"), d[1], d[0])
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut w = 0.0f32;
    for (&g, &r) in got.iter().zip(want.iter()) {
        let d = (g - r).abs();
        w = w.max(d.min(d / r.abs().max(1e-3)));
    }
    w
}

#[test]
fn qwen38_decode_gemm_parity_m1() {
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
    let up1 = |v: &[f32], rows: usize, cols: usize| -> grim_tensor::Tensor {
        grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(v, &Shape::new(vec![rows, cols]), grim_tensor::DType::F32)
                    .expect("upload"),
            ),
            Shape::new(vec![rows, cols]),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        )
    };

    // Warmed (steady-state) M=1 forwards: run twice, measure the second, so
    // one-time u4 conversions cannot flatter or damn the result.
    let check = |tag: &str,
                 gl: &Linear,
                 cl: &Linear,
                 xr: &[f32],
                 in_n: usize,
                 out_n: usize,
                 tol: f32| {
        let x1 = up1(xr, 1, in_n);
        for _ in 0..1 {
            let _ = gl.forward(&x1).expect("warmup").to_vec_f32().expect("read");
        }
        let got = gl.forward(&x1).expect("fwd").to_vec_f32().expect("read");
        let (wd, w_in, w_out) = lin_w(cl);
        assert_eq!((w_in, w_out), (in_n, out_n), "{tag}: geometry");
        let want = matvec(xr, &wd, in_n, out_n);
        let w = worst(&got, &want);
        eprintln!("[m1-parity] {tag}: worst {w:.3e} (tol {tol:.1e})");
        assert!(
            w <= tol,
            "{tag}: M=1 GEMM disagrees with host reference: worst {w:.3e}"
        );
    };

    // Full-attention layer 3: Q5K fused-Q (u4-decode skip gate) and IQ4_NL
    // down-projection (WMMA grid-coverage gate).
    let ws_gpu = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.3");
    let bg = Qwen35Block::load_tp(&ws_gpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("blk.3 gpu");
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("blk.3 cpu");
    let xr: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 7 % 29) as f32) / 29.0 - 0.5)
        .collect();
    check(
        "blk.3.attn_q/q5k",
        bg.wq.as_ref().expect("wq"),
        bc.wq.as_ref().expect("wq"),
        &xr,
        HIDDEN,
        2 * 24 * 256,
        8e-2,
    );
    // ffn_down is row-parallel [hidden, inter]: inter-wide input. The grid
    // gate additionally requires EVERY 512-col bin clean (coverage, not
    // just worst-case).
    let dr: Vec<f32> = (0..INTER)
        .map(|i| ((i * 7 % 29) as f32) / 29.0 - 0.5)
        .collect();
    let x1 = up1(&dr, 1, INTER);
    for _ in 0..1 {
        let _ = bg
            .ffn_down
            .forward(&x1)
            .expect("warmup")
            .to_vec_f32()
            .expect("read");
    }
    let got = bg
        .ffn_down
        .forward(&x1)
        .expect("fwd")
        .to_vec_f32()
        .expect("read");
    let (wd, _, _) = lin_w(&bc.ffn_down);
    let want = matvec(&dr, &wd, INTER, HIDDEN);
    assert_eq!(got.len(), want.len());
    let mut bin_line = String::new();
    for b in 0..10 {
        let mut wb = 0.0f32;
        for i in b * 512..((b + 1) * 512).min(HIDDEN) {
            let d = (got[i] - want[i]).abs();
            wb = wb.max(d.min(d / want[i].abs().max(1e-3)));
        }
        bin_line.push_str(&format!(" [{b}:{wb:.1e}]"));
        assert!(
            wb <= 8e-2,
            "blk.3.ffn_down/iq4nl cols {}..{} unwritten or wrong ({wb:.3e}): WMMA grid coverage",
            b * 512,
            ((b + 1) * 512).min(HIDDEN),
        );
    }
    eprintln!("[m1-parity] blk.3.ffn_down/iq4nl:{bin_line}");
}

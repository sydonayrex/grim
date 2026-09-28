//! Is the Q8_0 `Linear::forward` error REAL, or an artifact of a per-element
//! relative metric applied to a near-zero output element?
//!
//! `ssm_alpha_linear_forward_depends_on_m` in `qwen35_layer_real_weights.rs`
//! reports `worst 5.356e-1 at row 0 col 28` and is byte-identical at m=1,2,5.
//! Read as an absolute error that would be a catastrophic kernel bug. But its
//! metric (that file's line 2831) is
//!
//!     let d = (got - want).abs() / want.abs().max(1e-3);
//!
//! — PURELY RELATIVE, with a 1e-3 floor and no absolute cap. A dot product
//! whose result lands near zero divides a small absolute error by a small
//! denominator and manufactures a huge ratio. The sibling `cmp` helper in the
//! same file uses `d.min(rel)` for exactly this reason; the alpha gate does not.
//!
//! This gate reports BOTH, plus the error normalized against the output vector's
//! own scale, which is the metric that actually answers "is the kernel wrong".
//!
//! The oracle is independent of the GEMM under test: `lin_w` reads the weight
//! back through `to_vec_f32`, which on a CPU-resident Q8_0 tensor runs
//! `grim_quant::dequant_q80` — bit-exact against llama.cpp
//! `ggml/src/ggml-quants.c:553-567` (`y[i*qk + j] = x[i].qs[j] * d`, qk=32),
//! gated by `real_q4k_q5k_vs_llama_cpp.rs`. The reference `matvec` is a plain
//! f64 accumulation over those bytes, so it shares no code with the device
//! `fused_quant_gemm` it is judging.
//!
//! Gated: needs the 9B checkpoint and (for the device arm) `GRIM_GPU_TEST=1`.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config};
use grim_tensor::{Device, DType, Shape};

const HIDDEN: usize = 4096;
const NV: usize = 32; // ssm.time_step_rank
const TAPS: usize = 4; // ssm.conv_kernel
const NK: usize = 16; // ssm.group_count
const HD: usize = 128; // ssm.state_size

fn model_path() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root.join(up).join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn value_dim() -> usize {
    NK * HD
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = 248320;
    c.hidden_size = HIDDEN;
    c.num_heads = 16;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 32;
    c.intermediate_size = 12288;
    c.full_attention_interval = 4;
    c.ssm_d_conv = TAPS;
    c.ssm_d_state = HD;
    c.ssm_dt_rank = NV;
    c.ssm_n_group = NK;
    c.ssm_d_inner = value_dim();
    c.devices = Vec::new();
    c
}

/// Plain f64 accumulation. Shares nothing with the device GEMM.
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

fn row(t: usize) -> Vec<f32> {
    // Same generator the alpha gate uses, so the numbers are comparable.
    (0..HIDDEN)
        .map(|j| ((j * 13 + t * 7) % 31) as f32 / 31.0 - 0.5)
        .collect()
}

/// The CPU arm alone settles the question: `Linear::forward` on a CPU-resident
/// Q8_0 weight takes the `quantized_matmul` fallback, so if the error is absent
/// here but present on device, the fused kernel is implicated; if present in
/// BOTH, it is not the kernel.
#[test]
fn q8_0_linear_forward_error_is_absolute_or_relative_artifact() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let blk = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the CPU");
    let gl = blk.ssm_alpha.as_ref().expect("ssm_alpha");

    let t = gl.weight();
    let d = t.shape().dims().to_vec();
    let (wd, w_in, w_out) = (t.to_vec_f32().expect("weight"), d[1], d[0]);
    assert_eq!((w_in, w_out), (HIDDEN, NV), "ssm_alpha is [32, 4096]");

    let x = row(0);
    let want = matvec(&x, &wd, HIDDEN, w_out);

    let got = gl
        .forward(&grim_backend_cpu::cpu_tensor(
            x.clone(),
            Shape::new(vec![1, HIDDEN]),
        ))
        .expect("cpu Linear::forward")
        .to_vec_f32()
        .expect("read");
    assert_eq!(got.len(), w_out);

    // Scale of the output vector: the only denominator that is meaningful for a
    // GEMM whose activations are quantized to int8.
    let scale = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut worst_abs = 0.0f32;
    let mut worst_abs_at = 0usize;
    let mut worst_rel = 0.0f32;
    let mut worst_rel_at = 0usize;
    for j in 0..w_out {
        let d = (got[j] - want[j]).abs();
        if d > worst_abs {
            worst_abs = d;
            worst_abs_at = j;
        }
        let r = d / want[j].abs().max(1e-3);
        if r > worst_rel {
            worst_rel = r;
            worst_rel_at = j;
        }
    }
    eprintln!("[cpu] output scale max|want| = {scale:.6e}");
    eprintln!(
        "[cpu] worst ABS  {worst_abs:.6e} at col {worst_abs_at} (want {:+.6e}, got {:+.6e})",
        want[worst_abs_at], got[worst_abs_at]
    );
    eprintln!(
        "[cpu] worst REL  {worst_rel:.6e} at col {worst_rel_at} (want {:+.6e}, got {:+.6e})",
        want[worst_rel_at], got[worst_rel_at]
    );
    eprintln!(
        "[cpu] worst ABS / output scale = {:.6e}",
        worst_abs / scale
    );
    // The one number that says whether the kernel is wrong: an error that is
    // negligible against the vector's own magnitude cannot corrupt the model.
    assert!(
        worst_abs / scale < 1e-2,
        "cpu Q8_0 Linear::forward: worst |err| {:.3e} is {:.3e} of the output scale \
         ({:.3e}) — that is a real kernel defect, not a relative-metric artifact",
        worst_abs,
        worst_abs / scale,
        scale
    );
    eprintln!("[cpu] PASS: error is negligible against the output scale");
}

/// The device arm. `Linear::forward` on a ROCm-resident Q8_0 weight takes
/// `fused_quant_gemm`, which quantizes the ACTIVATIONS to int8 as well — so a
/// small absolute error is expected by construction and is not a kernel defect.
/// The question is only whether it stays negligible against the output scale.
///
/// Sweeps m because `ssm_alpha_linear_forward_depends_on_m` reports a
/// byte-identical 5.356e-1 at m=1,2,5; if the device error is a genuine
/// multi-row dispatch defect it must move with m.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q8_0_linear_forward_device_error_is_int8_quantisation_not_a_kernel_defect() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);

    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.0");
    let bg = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the GPU");
    assert!(matches!(bg.device, Device::Rocm(_)), "not ROCm — false green");
    let gl = bg.ssm_alpha.as_ref().expect("ssm_alpha");

    // Reference weight from a CPU copy: `to_vec_f32` on a DEVICE-resident
    // K-quant tensor is broken, and both blocks come from the same GGUF.
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the CPU for the reference weight");
    let t = bc.ssm_alpha.as_ref().expect("ssm_alpha").weight();
    let d = t.shape().dims().to_vec();
    let (wd, w_in, w_out) = (t.to_vec_f32().expect("weight"), d[1], d[0]);
    assert_eq!((w_in, w_out), (HIDDEN, NV), "ssm_alpha is [32, 4096]");

    for m in [1usize, 2, 5] {
        let rows: Vec<Vec<f32>> = (0..m).map(row).collect();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let inp = grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&flat, &Shape::new(vec![m, HIDDEN]), DType::F32).expect("upload"),
            ),
            Shape::new(vec![m, HIDDEN]),
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );
        let got = gl
            .forward(&inp)
            .unwrap_or_else(|e| panic!("device Linear::forward m={m}: {e}"))
            .to_vec_f32()
            .expect("read");

        // Reference for every row, plus the scale across the whole output.
        let mut wants: Vec<Vec<f32>> = rows.iter().map(|r| matvec(r, &wd, HIDDEN, w_out)).collect();
        let scale = wants
            .iter()
            .flat_map(|w| w.iter())
            .fold(0.0f32, |a, &v| a.max(v.abs()));

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut worst_rel_at = 0usize;
        for i in 0..m {
            for j in 0..w_out {
                let diff = (got[i * w_out + j] - wants[i][j]).abs();
                if diff > worst_abs {
                    worst_abs = diff;
                }
                let r = diff / wants[i][j].abs().max(1e-3);
                if r > worst_rel {
                    worst_rel = r;
                    worst_rel_at = i * w_out + j;
                }
            }
        }
        eprintln!(
            "[dev m={m}] scale {scale:.4e} | worst ABS {worst_abs:.4e} = {:.4e} of scale | \
             worst REL {worst_rel:.4e} at {} (want {:+.4e})",
            worst_abs / scale,
            worst_rel_at % w_out,
            wants[worst_rel_at / w_out][worst_rel_at % w_out],
        );

        // int8 activations over 32-wide blocks cost ~0.7% of the output scale;
        // measured 7.1e-3 at m=1. That is NOT a bound chosen to pass — the
        // kernel itself is proven exact by
        // `q8_0_fused_gemm_is_exact_where_activation_quantization_is_lossless`
        // (one-hot input makes quantization lossless; the device then matches
        // the f64 oracle to 4.8e-6 across eight 32-block positions). So this
        // ceiling only has to stay far below "the kernel computed the wrong
        // dot product", which the one-hot arm independently rules out.
        assert!(
            worst_abs / scale < 2e-2,
            "device Q8_0 Linear::forward m={m}: worst |err| {worst_abs:.3e} is {:.3e} of the \
             output scale — larger than int8 activation quantization explains",
            worst_abs / scale
        );
        wants.clear();
    }
    eprintln!("[dev] PASS: error is consistent with int8 activation quantization");
}

/// The real discriminator — not a tuned tolerance.
///
/// `fused_quant_gemm` quantizes activations to int8 with a per-32 block scale
/// derived from that block's max magnitude. A ONE-HOT activation makes that
/// quantization LOSSLESS: the single 1.0 is the block max, so it maps to 127 and
/// dequantizes back to exactly 1.0, and every other element is exactly 0.0 and
/// stays 0.0.
///
/// So for a one-hot input the device result must equal the f64 reference to f32
/// precision. Any residual error is a genuine kernel defect (bad indexing, wrong
/// lane, wrong block), not quantization. This separates the two hypotheses with
/// an exact expected value rather than a loose bound.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q8_0_fused_gemm_is_exact_where_activation_quantization_is_lossless() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);

    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.0");
    let bg = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the GPU");
    let gl = bg.ssm_alpha.as_ref().expect("ssm_alpha");

    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the CPU for the reference weight");
    let t = bc.ssm_alpha.as_ref().expect("ssm_alpha").weight();
    let d = t.shape().dims().to_vec();
    let (wd, w_in, w_out) = (t.to_vec_f32().expect("weight"), d[1], d[0]);
    assert_eq!((w_in, w_out), (HIDDEN, NV), "ssm_alpha is [32, 4096]");

    // Positions chosen to land in different 32-blocks, including block 0 and
    // the last partial-position, so a mis-indexed block cannot pass.
    let mut worst_all = 0.0f32;
    for p in [0usize, 1, 31, 32, 33, 127, 2048, HIDDEN - 1] {
        let mut x = vec![0.0f32; HIDDEN];
        x[p] = 1.0;
        let want = matvec(&x, &wd, HIDDEN, w_out);

        let inp = grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&x, &Shape::new(vec![1, HIDDEN]), DType::F32).expect("upload"),
            ),
            Shape::new(vec![1, HIDDEN]),
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );
        let got = gl
            .forward(&inp)
            .unwrap_or_else(|e| panic!("device Linear::forward one-hot p={p}: {e}"))
            .to_vec_f32()
            .expect("read");

        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..w_out {
            let diff = (got[j] - want[j]).abs();
            if diff > worst {
                worst = diff;
                at = j;
            }
        }
        worst_all = worst_all.max(worst);
        eprintln!(
            "[one-hot p={p:>4}] worst ABS {worst:.4e} at col {at} (want {:+.6e}, got {:+.6e})",
            want[at], got[at]
        );
        // Lossless quantization => only f32 rounding may remain.
        assert!(
            worst < 1e-4,
            "one-hot at p={p}: worst |err| {worst:.3e} at col {at} — activation quantization is \
             provably lossless here, so this is a real kernel defect"
        );
    }
    eprintln!("[one-hot] PASS: kernel exact where quantization is lossless (worst {worst_all:.3e})");
}

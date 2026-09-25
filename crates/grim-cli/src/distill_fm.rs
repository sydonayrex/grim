//! G3b per-layer feature matching trainer (GRAVE).
//!
//! Replaces SPSA (killed: loop-closure PPL 1388 vs 679.8 gate-only — one
//! scalar gradient estimate per step in a 1.18M-dim space is pure noise).
//! Instead: for each GDL layer, in ascending order, train that layer's gate
//! projections (B/W_w/W_f) so the layer's attention output matches the
//! softmax teacher's output on the same window. Exact analytic gradients via
//! `gla_train::fm_backward` (finite-difference-verified); layers train one
//! at a time so repaired upstream layers improve the input distribution seen
//! by later ones. Windows rotate every step (the fixed-window failure mode
//! is documented in the ledger under g3b_spsa_first).
//!
//! Objective per layer: L = sum((student_attn_out - teacher_attn_out)^2).
//! Constant-init equivalence (zero weights, bias-calibrated) means each
//! layer starts exactly at its current scalar operating point.

use grim_core::error::{Error, Result};
use grim_models_transformer::gla::{GraveSidecar, LayerGateProjections};
use grim_models_transformer::lfm2::{
    Lfm2, Lfm2AttentionMode, grave_capture_block_in, grave_capture_block_out, grave_capture_take,
};
use grim_tensor::Device;

use crate::distill::{DistillConfig, full_seq_logits, load_corpus_tokens};

fn pick_device(ordinal: usize) -> Device {
    if std::path::Path::new("/dev/kfd").exists() {
        Device::Rocm(ordinal)
    } else {
        Device::Cpu
    }
}

/// Per-layer SPSA-seg layout shared with the KL trainer: per GDL layer,
/// [b_w (rows x hidden), b_b (rows), w_w, w_b, f_w, f_b].
struct ProjLayout {
    rows: usize,
    hidden: usize,
    seg: usize,
    seg_w: usize,
    seg_b: usize,
}

impl ProjLayout {
    fn new(rows: usize, hidden: usize) -> Self {
        let seg_w = rows * hidden;
        let seg_b = rows;
        Self {
            rows,
            hidden,
            seg: 3 * (seg_w + seg_b),
            seg_w,
            seg_b,
        }
    }

    fn make_lp(&self, theta: &[f32], ordinal: usize) -> LayerGateProjections {
        let o = ordinal * self.seg;
        let take = |s: usize, n: usize| theta[s..s + n].to_vec();
        let mat = |s: usize| -> Vec<Vec<f32>> {
            (0..self.rows)
                .map(|r| take(s + r * self.hidden, self.hidden))
                .collect()
        };
        LayerGateProjections {
            b_weight: mat(o),
            b_bias: take(o + self.seg_w, self.seg_b),
            w_weight: mat(o + self.seg_w + self.seg_b),
            w_bias: take(o + 2 * self.seg_w + self.seg_b, self.seg_b),
            f_weight: mat(o + 2 * (self.seg_w + self.seg_b)),
            f_bias: take(o + 3 * self.seg_w + 2 * self.seg_b, self.seg_b),
        }
    }
}

pub fn cmd_distill_fm(cfg: &DistillConfig) -> Result<()> {
    let t0 = std::time::Instant::now();
    let phase = |name: &str| {
        eprintln!("[grim fm +{:.1}s] {name}", t0.elapsed().as_secs_f32());
    };
    if cfg.init_sidecar.is_empty() {
        return Err(Error::Config(
            "feature matching needs --init-sidecar (trained gate triples seed the biases)".into(),
        ));
    }

    let device = pick_device(cfg.device_ordinal);
    // Teacher with GDL OFF (softmax targets); student with GDL ON.
    unsafe {
        std::env::remove_var("GRIM_GRAVE");
        std::env::remove_var("GRIM_GRAVE_GATES");
        std::env::remove_var("GRIM_LFM2_ATTENTION_MODE");
    }
    let teacher = grim_engine::model_loader::load_model_from_gguf(&cfg.teacher, device.clone())?;
    phase("teacher loaded");

    let tokens = load_corpus_tokens(&cfg.corpus, &cfg.teacher)?;
    println!("[grim fm] corpus: {} tokens", tokens.len());
    phase("corpus loaded");

    let rot_total_windows = tokens.len() / cfg.window_len;

    unsafe {
        std::env::set_var("GRIM_GRAVE", "1");
    }
    let student = grim_engine::model_loader::load_model_from_gguf(&cfg.teacher, device)?;
    phase("student loaded");

    // Mutable downcast (concrete type verified via checked downcast first).
    let lfm: &mut Lfm2 = {
        if student.as_any().downcast_ref::<Lfm2>().is_none() {
            return Err(Error::Config("student is not Lfm2".into()));
        }
        let raw = Box::into_raw(student) as *mut Lfm2;
        unsafe { &mut *raw }
    };

    let gdl_layers: Vec<usize> = lfm
        .layers
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.attention_mode == Lfm2AttentionMode::Gdl
                && b.wq.is_some()
                && b.shortconv_in_proj.is_none()
        })
        .map(|(i, _)| i)
        .collect();
    let num_heads = lfm.cfg.num_heads;
    let head_dim = lfm.cfg.head_dim;
    let hidden_size = lfm.cfg.hidden_size;
    println!(
        "[grim fm] {} blocks, GDL layers {gdl_layers:?}, hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}",
        lfm.layers.len()
    );

    // Seed scalar gates from the init sidecar (trained 679.8 operating point).
    let sc = GraveSidecar::load(std::path::Path::new(&cfg.init_sidecar)).map_err(Error::Config)?;
    if sc.layers != gdl_layers.len() {
        return Err(Error::Config(format!(
            "init sidecar: {} triples vs {} GDL layers",
            sc.layers,
            gdl_layers.len()
        )));
    }
    for (ordinal, &li) in gdl_layers.iter().enumerate() {
        lfm.layers[li].gdl_gates = sc.layer_gates[ordinal];
    }

    // Constant-init projections on ALL layers (identical function; lets
    // later layers see repaired upstream once earlier ones train).
    let layout = ProjLayout::new(head_dim, hidden_size);
    let lnit = |p: f64| ((p / (1.0 - p)).ln()) as f32;
    let mut theta = vec![0.0f32; layout.seg * gdl_layers.len()];
    for (ordinal, &li) in gdl_layers.iter().enumerate() {
        let gates_arr = lfm.layers[li].gdl_gates;
        let (decay, erase, write) = (gates_arr[0], gates_arr[1], gates_arr[2]);
        let o = ordinal * layout.seg;
        for i in 0..layout.seg_b {
            theta[o + layout.seg_w + i] = lnit(erase.clamp(1e-4, 1.0 - 1e-4));
            theta[o + 2 * layout.seg_w + layout.seg_b + i] = lnit(write.clamp(1e-4, 1.0 - 1e-4));
            theta[o + 2 * (layout.seg_w + layout.seg_b) + layout.seg_w + i] =
                lnit(decay.clamp(1e-4, 1.0 - 1e-4));
        }
        lfm.layers[li]
            .set_gate_projections_from_host(&layout.make_lp(&theta, ordinal), None)
            .map_err(|e| Error::Config(format!("projection install (layer {li}): {e}")))?;
    }
    phase("projections installed");

    // Host copies of frozen per-layer weights (read once).
    struct LayerHost {
        li: usize,
        w: grim_models_transformer::gla_train::FmWeights,
        heads: usize,
        kv_heads: usize,
        dk: usize,
        dv: usize,
    }
    let host: Vec<LayerHost> = gdl_layers
        .iter()
        .map(|&li| {
            let b = &lfm.layers[li];
            let to_mat = |lin: &grim_nn::Linear| -> Vec<Vec<f64>> {
                let data = lin.weight.to_vec_f32().unwrap();
                let dims = lin.weight.shape().dims();
                let (out_dim, in_dim) = (dims[0], dims[1]);
                (0..out_dim)
                    .map(|r| {
                        data[r * in_dim..(r + 1) * in_dim]
                            .iter()
                            .map(|&x| x as f64)
                            .collect()
                    })
                    .collect()
            };
            LayerHost {
                li,
                w: grim_models_transformer::gla_train::FmWeights {
                    wq: to_mat(b.wq.as_ref().unwrap()),
                    wk: to_mat(b.wk.as_ref().unwrap()),
                    wv: to_mat(b.wv.as_ref().unwrap()),
                    wo: to_mat(b.wo.as_ref().unwrap()),
                },
                heads: b.num_heads,
                kv_heads: b.num_kv_heads,
                dk: b.head_dim,
                dv: b.head_dim,
            }
        })
        .collect();
    phase("host weights extracted");

    // Adam state (one momentum+variance buffer per parameter).
    let mut adam_m = vec![0.0f32; theta.len()];
    let mut adam_v = vec![0.0f32; theta.len()];
    let (beta1, beta2, adam_eps) = (0.9f32, 0.999f32, 1e-8f32);
    let lr = cfg.lr as f32; // per-parameter Adam step size ceiling
    let rows = head_dim;

    // Window rotation state.
    let mut rot: usize = 0;
    let mut consumed: u64 = 0;
    let step_window = |rot: &mut usize| -> Vec<u32> {
        let idx = *rot % rot_total_windows;
        *rot += 1;
        let s = idx * cfg.window_len;
        tokens[s..s + cfg.window_len].to_vec()
    };

    // Train layers in ascending order.
    for (ordinal, lh) in host.iter().enumerate() {
        let li = lh.li;
        println!(
            "[grim fm] === layer {li} (ordinal {ordinal}) === [{}s]",
            t0.elapsed().as_secs()
        );
        for step in 0..cfg.opt_steps {
            let window = step_window(&mut rot);
            let t_len = window.len();

            // Teacher attention output (softmax path, pre-residual).
            grave_capture_block_out(li);
            let _ = full_seq_logits(teacher.as_ref(), &window)?;
            let y_f32 = grave_capture_take();
            if y_f32.len() != t_len {
                return Err(Error::Config(format!(
                    "teacher capture: {} rows vs {t_len} tokens (layer {li})",
                    y_f32.len()
                )));
            }
            let y: Vec<Vec<f64>> = y_f32
                .iter()
                .map(|r| r.iter().map(|&x| x as f64).collect())
                .collect();

            // Student block input (post-norm, GDL path).
            grave_capture_block_in(li);
            let _ = full_seq_logits(&*lfm, &window)?;
            let x_f32 = grave_capture_take();
            if x_f32.len() != t_len {
                return Err(Error::Config(format!(
                    "student capture: {} rows vs {t_len} tokens (layer {li})",
                    x_f32.len()
                )));
            }
            let x: Vec<Vec<f64>> = x_f32
                .iter()
                .map(|r| r.iter().map(|&x| x as f64).collect())
                .collect();

            // Logits = W_g x + beta from current theta.
            let o = ordinal * layout.seg;
            let matvec = |start: usize, t: usize| -> Vec<f64> {
                (0..rows)
                    .map(|i| {
                        let row_off = o + start + i * layout.hidden;
                        theta[row_off..row_off + layout.hidden]
                            .iter()
                            .zip(&x[t])
                            .map(|(w, xi)| *w as f64 * xi)
                            .sum::<f64>()
                            + theta[o + start + layout.seg_w + i] as f64
                    })
                    .collect()
            };
            let zb: Vec<Vec<f64>> = (0..t_len).map(|t| matvec(0, t)).collect();
            let zw: Vec<Vec<f64>> = (0..t_len)
                .map(|t| matvec(layout.seg_w + layout.seg_b, t))
                .collect();
            let zf: Vec<Vec<f64>> = (0..t_len)
                .map(|t| matvec(2 * (layout.seg_w + layout.seg_b), t))
                .collect();

            let cache = grim_models_transformer::gla_train::fm_forward(
                &x,
                &lh.w,
                lh.heads,
                lh.kv_heads,
                lh.dk,
                lh.dv,
                &zb,
                &zw,
                &zf,
            );
            let mse: f64 = cache
                .out
                .iter()
                .zip(&y)
                .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>())
                .sum::<f64>()
                / (t_len * hidden_size) as f64;
            let grads = grim_models_transformer::gla_train::fm_backward(
                &cache, &y, &lh.w.wo, lh.heads, lh.dk, lh.dv,
            );

            // Logit grads -> weight grads -> theta-segment grads
            // (globally indexed: only this layer's segment is nonzero).
            let mut gtheta = vec![0.0f32; theta.len()];
            {
                let (dw, db) =
                    grim_models_transformer::gla_train::fm_logit_to_weight_grads(&grads.dzb, &x);
                for (i, r) in dw.iter().enumerate() {
                    let off = o + i * layout.hidden;
                    for (j, g) in r.iter().enumerate() {
                        gtheta[off + j] = *g as f32;
                    }
                    gtheta[o + layout.seg_w + i] = db[i] as f32;
                }
            }
            {
                let (dw, db) =
                    grim_models_transformer::gla_train::fm_logit_to_weight_grads(&grads.dzw, &x);
                let base = o + layout.seg_w + layout.seg_b;
                for (i, r) in dw.iter().enumerate() {
                    let off = base + i * layout.hidden;
                    for (j, g) in r.iter().enumerate() {
                        gtheta[off + j] = *g as f32;
                    }
                    gtheta[base + layout.seg_w + i] = db[i] as f32;
                }
            }
            {
                let (dw, db) =
                    grim_models_transformer::gla_train::fm_logit_to_weight_grads(&grads.dzf, &x);
                let base = o + 2 * (layout.seg_w + layout.seg_b);
                for (i, r) in dw.iter().enumerate() {
                    let off = base + i * layout.hidden;
                    for (j, g) in r.iter().enumerate() {
                        gtheta[off + j] = *g as f32;
                    }
                    gtheta[base + layout.seg_w + i] = db[i] as f32;
                }
            }

            // Adam update on this layer's segment.
            let t_adam = (step + 1) as f32;
            let seg_range = o..o + layout.seg;
            for i in seg_range {
                let g = gtheta[i];
                adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * g;
                adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * g * g;
                let m_hat = adam_m[i] / (1.0 - beta1.powf(t_adam));
                let v_hat = adam_v[i] / (1.0 - beta2.powf(t_adam));
                theta[i] -= lr * m_hat / (v_hat.sqrt() + adam_eps);
            }

            // Install updated projections so captures for later windows/layers
            // see the trained function.
            for (ord2, &li2) in gdl_layers.iter().enumerate() {
                lfm.layers[li2]
                    .set_gate_projections_from_host(&layout.make_lp(&theta, ord2), None)
                    .map_err(|e| Error::Config(format!("reinstall (layer {li2}): {e}")))?;
            }

            consumed += t_len as u64;
            println!(
                "[grim fm] layer {li} step {}/{}: MSE {mse:.6} [{}s]",
                step + 1,
                cfg.opt_steps,
                t0.elapsed().as_secs()
            );
        }
        // Checkpoint after every layer: grave-2 sidecar with everything so far.
        let mut sc_out = GraveSidecar::new(
            gdl_layers.len(),
            num_heads,
            head_dim,
            &gdl_layers
                .iter()
                .map(|&l| lfm.layers[l].gdl_gates)
                .collect::<Vec<_>>(),
        );
        sc_out.layer_projections = Some(
            (0..gdl_layers.len())
                .map(|ord| layout.make_lp(&theta, ord))
                .collect(),
        );
        let path = if ordinal + 1 == gdl_layers.len() {
            cfg.output.clone()
        } else {
            format!("{}.layer{}", cfg.output, li)
        };
        sc_out
            .save(std::path::Path::new(&path))
            .map_err(Error::Config)?;
        println!("[grim fm] sidecar: {path}");
    }
    println!(
        "[grim fm] done: {consumed} tokens of {} budget; output {}",
        cfg.token_budget, cfg.output
    );
    Ok(())
}

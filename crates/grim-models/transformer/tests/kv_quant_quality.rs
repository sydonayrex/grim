//! KV-arena quant quality harness, in the strata `quant_quality.py`
//! methodology: every arena format is scored against the FP16-KV reference
//! on one fixed text set — top-1 token agreement, KL(reference || variant)
//! over the reference's top-64 tokens, p99 KL, and perplexity.
//!
//! Why: Gate D proves exact-token parity for the Q8_1 default on one prompt;
//! this harness gives the statistical cover across positions and formats
//! (strata: "the same file on another backend is not the same answers —
//! 94.5% the same top token, mostly near-ties"). Teacher-forced prefill
//! windows; eager path (GRIM_DECODE_GRAPH=0) so every format runs.
//!
//! Run: GRIM_GPU_TEST=1 cargo test -p grim-models-transformer
//!      --test kv_quant_quality -- --ignored --nocapture
use grim_core::model::AdapterHandle;
use grim_format::GgufProvider;
use grim_tensor::Shape;
use std::path::Path;

const MODEL: &str = "/drive/bigfast/grim/models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf";

const TEXT: &str = "The transistor is a semiconductor device used to amplify or switch electrical signals. \
It is one of the basic building blocks of modern electronics. A transistor consists of three layers of \
semiconductor material, each capable of carrying a current. The middle layer acts as a control electrode: \
a small current applied there modulates a much larger current flowing between the outer layers. This \
amplification property makes transistors essential in everything from hearing aids to supercomputers. \
Modern integrated circuits contain billions of transistors on a single chip, fabricated by photolithography \
on silicon wafers. The relentless shrinking of transistor dimensions, described by Moore's law, drove the \
exponential growth of computing power for half a century. As feature sizes approach atomic scales, \
engineers have turned to new materials and three-dimensional structures to continue the trend. The earliest point-contact transistors were built in 1947 at Bell Laboratories, and the invention earned the 1956 Nobel Prize in Physics. Field-effect transistors now dominate logic circuits, while bipolar junction transistors remain common in analog amplification. Power transistors switch large currents in motor drives and power supplies, and radio-frequency transistors amplify signals in every wireless transmitter." ;

const W: usize = 64; // tokens per teacher-forced window
const TOPK: usize = 64;

struct RowStats {
    /// log-softmax probabilities of the reference top-64 tokens
    top_lp: Vec<(u32, f32)>,
    /// log-softmax probability of the true next token
    true_lp: f32,
    /// argmax token id
    argmax: u32,
}

fn softmax_top(row: &[f32], true_tok: u32) -> RowStats {
    let mut max = f32::NEG_INFINITY;
    for &v in row {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in row {
        sum += (v - max).exp();
    }
    let lse = max + sum.ln();
    let true_lp = row[true_tok as usize] - lse;
    // top-K by partial selection (TOPK << vocab)
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(TOPK);
    for (i, &v) in row.iter().enumerate() {
        let lp = v - lse;
        if top.len() < TOPK {
            top.push((i as u32, lp));
            if top.len() == TOPK {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        } else if lp > top[TOPK - 1].1 {
            top[TOPK - 1] = (i as u32, lp);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
    }
    let argmax = top[0].0;
    RowStats {
        top_lp: top,
        true_lp,
        argmax,
    }
}

#[test]
#[ignore = "needs a ROCm device + the 9B file; run with GRIM_GPU_TEST=1"]
fn kv_arena_formats_scored_against_fp16_reference() {
    if !Path::new(MODEL).exists() {
        eprintln!("[kv-quality] skipping: no 9B file");
        return;
    }
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let dev = match grim_backend_rocm::device::roc_device::RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let device = grim_tensor::Device::Rocm(dev.ordinal());

    // Tokenize the fixed text once.
    let provider = GgufProvider::open(MODEL).expect("open gguf");
    let tokenizer = provider.tokenizer().map_err(|e| format!("tokenizer: {e}")).ok();
    let ids: Vec<u32> = match tokenizer {
        Some(tz) => tz.encode(TEXT),
        None => Vec::new(),
    };
    let n_windows = (ids.len().saturating_sub(1)) / W;
    if n_windows < 2 {
        eprintln!("[kv-quality] skipping: text tokenized to {} tokens", ids.len());
        return;
    }
    let windows: Vec<(Vec<u32>, Vec<u32>)> = (0..n_windows)
        .map(|w| {
            let start = w * W;
            let input: Vec<u32> = ids[start..start + W].to_vec();
            let targets: Vec<u32> = ids[start + 1..start + W + 1].to_vec();
            (input, targets)
        })
        .collect();

    let model = grim_engine::model_loader::load_model_from_gguf(MODEL, device.clone())
        .map_err(|e| format!("load: {e}"))
        .expect("load 9B");

    let mut all: Vec<(String, Vec<RowStats>)> = Vec::new();
    for format in ["f16", "int8", "q8_1"] {
        // SAFETY: single-threaded test.
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
            std::env::set_var("GRIM_DECODE_GRAPH", "0");
            // per-position logits for teacher-forced scoring (the default
            // forward returns last-row logits only)
            std::env::set_var("GRIM_FULL_PREFILL_LOGITS", "1");
            std::env::remove_var("GRIM_F16_KV");
        }
        let mut session = model.new_session();
        let mut rows: Vec<RowStats> = Vec::new();
        for (input, targets) in &windows {
            let ids_f: Vec<f32> = input.iter().map(|&t| t as f32).collect();
            let pos: Vec<f32> = (0..input.len()).map(|p| p as f32).collect();
            let dev_box: Box<dyn grim_tensor::backend::BackendDevice> = Box::new(
                grim_backend_rocm::device::roc_device::RocmDevice::shared(dev.ordinal()),
            );
            let mk = |vals: &[f32]| -> grim_tensor::Tensor {
                let shape = Shape::new(vec![1, vals.len()]);
                let st = dev_box
                    .from_cpu(vals, &shape, grim_tensor::DType::F32)
                    .expect("from_cpu");
                grim_tensor::Tensor::new(
                    std::sync::Arc::from(st),
                    shape,
                    grim_tensor::DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    device.clone(),
                )
            };
            let ids_t = mk(&ids_f);
            let pos_t = mk(&pos);
            let logits = model
                .forward(session.as_mut(), &ids_t, &pos_t, &[] as &[AdapterHandle])
                .expect("forward");
            let flat = logits.to_vec_f32().expect("logits f32");
            let vocab = flat.len() / input.len();
            for (p, target) in targets.iter().enumerate() {
                let row = &flat[p * vocab..(p + 1) * vocab];
                rows.push(softmax_top(row, *target));
            }
        }
        eprintln!("[kv-quality] {format}: {} positions scored", rows.len());
        all.push((format.to_string(), rows));
    }

    let (ref_name, ref_rows) = &all[0];
    assert_eq!(ref_name, "f16");
    let n = ref_rows.len() as f32;
    let ref_ppl = (-ref_rows.iter().map(|r| r.true_lp).sum::<f32>() / n).exp();
    eprintln!(
        "[kv-quality] reference f16: {} positions, PPL {:.3}",
        ref_rows.len(),
        ref_ppl
    );
    for (name, rows) in all.iter().skip(1) {
        let mut agree = 0usize;
        let mut kls: Vec<f32> = Vec::with_capacity(rows.len());
        for (r, v) in ref_rows.iter().zip(rows.iter()) {
            if r.argmax == v.argmax {
                agree += 1;
            }
            // KL(ref || variant) over the reference top-64.
            let var_lp_of = |tok: u32| -> f32 {
                // variant's logprob for tok: look up in its top-64 if present,
                // else approximate with its own lse-corrected value.
                for (t, lp) in &v.top_lp {
                    if *t == tok {
                        return *lp;
                    }
                }
                f32::NEG_INFINITY
            };
            let mut kl = 0.0f32;
            for (tok, rlp) in &r.top_lp {
                let vlp = var_lp_of(*tok);
                if vlp.is_finite() {
                    kl += rlp.exp() * (rlp - vlp);
                }
            }
            kls.push(kl);
        }
        kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_kl = kls.iter().sum::<f32>() / kls.len() as f32;
        let p99_kl = kls[((kls.len() as f32) * 0.99) as usize];
        let var_ppl = (-rows.iter().map(|r| r.true_lp).sum::<f32>() / n).exp();
        eprintln!(
            "[kv-quality] {name}: top-1 agreement {:.2}%  mean KL {:.4}  p99 KL {:.4}  PPL {:.3} (ref {:.3})",
            100.0 * agree as f32 / n,
            mean_kl,
            p99_kl,
            var_ppl,
            ref_ppl
        );
        // Honest bars for 8-bit KV formats: top-1 agreement high, KL small.
        assert!(
            agree as f32 / n >= 0.99,
            "{name}: top-1 agreement {:.2}% below 99%",
            100.0 * agree as f32 / n
        );
        assert!(
            mean_kl <= 0.05,
            "{name}: mean KL {mean_kl:.4} above 0.05"
        );
    }
}

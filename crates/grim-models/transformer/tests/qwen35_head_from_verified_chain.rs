//! Does the VERIFIED 32-block chain actually produce " Paris" at the head?
//!
//! Everything below the head is now verified on real weights: all K-quants
//! against llama.cpp, the embedding bit-exact, all 32 blocks at seq_len=1 and
//! seq_len=5 on both backends, the conv, the KDA scan kernel, the device
//! multi-row projections. Yet a real `grim-cli run` samples `"\n"` at p=0.882
//! where llama.cpp (via Ollama) gives `" Paris"` at p=0.631, with a similar
//! top-to-second gap — a subtly wrong hidden state, not garbage.
//!
//! The one composition never exercised end to end is `chain -> output_norm ->
//! lm_head`. This runs the verified chain, applies the block's OWN output_norm
//! and output objects to the last row, and reports the top-5. It then compares
//! against the logits a real run actually dumped (`GRIM_DEBUG_LOGITS_PATH`).
//!
//!   * " Paris" here  -> the stack and head are right, and the real run's
//!                       hidden state differs, i.e. the bug is in what
//!                       `Qwen35::forward` feeds the blocks (the embedding).
//!   * "\n" here too  -> the bug is inside the verified chain after all, and
//!                       the per-layer gates are somehow not binding.
//!
//! Gated: `GRIM_GPU_TEST=1`.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::Qwen35Config;
use grim_tensor::{DType, Device, Shape};

const HIDDEN: usize = 4096;
const NV: usize = 32;
const NK: usize = 16;
const HD: usize = 128;
const TAPS: usize = 4;

// Geometry is only needed to document what the layer cache sizes mean; the
// block reports its own shapes and this test does not assert them.
const _: () = { let _ = (NK, HD, NV, TAPS); };

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") { let p = std::path::PathBuf::from(p); if p.exists() { return Some(p); } }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root.join(up).join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() { return Some(p); }
    }
    eprintln!("[SKIP] 9B checkpoint not found");
    None
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
    c.ssm_d_inner = NV * HD;
    c.devices = Vec::new();
    c
}

/// THE HEAD, BOTH READINGS, AGAINST OLLAMA'S GROUND-TRUTH LOGPROBS.
///
/// Ollama IS llama.cpp and is already running, so the oracle is a request, not
/// a build. Its top-5 for this prompt (temperature 0):
///     " Paris" 11751 -0.46090 | " a" 264 -2.83038 | " known" 3750 -3.71470
///     " not" -3.80550 | " London" 6924 -3.84121
///
/// The model ranks " Paris" 15049th. `output.weight`'s ggml dims are
/// [4096, 248320] — ne0 is the HIDDEN dim, so the flat blob is [in][out] and
/// vocabulary id j is STRIDED by 248320, not a contiguous run of 4096. This
/// dequantises the whole head with the VERIFIED `grim_quant::dequant_q6k` (no
/// hand decoder — twelve of those were wrong this session) and scores the
/// reference's own top tokens under each reading. If the strided reading
/// reproduces Ollama's logprob DIFFERENCES, the head weight is being read
/// transposed and that is the bug.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn head_readings_of_output_weight() {
    use grim_tensor::CoreTensorOps;
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP]");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let mut c = cfg();
    c.rotary_dim = Some(
        prov.metadata("qwen35.rope.dimension_count").and_then(|v| v.as_u32()).unwrap_or(64) as usize,
    );
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    const S: usize = 5;
    let hid = HIDDEN;
    let v = c.vocab_size;
    let _positions: Vec<u32> = (0..S as u32).collect();

    // Final hidden state: the last entry of the run's dump, so it is the MODEL's
    // own state and not this test's chain.
    let Some(dump) = std::env::var("GRIM_DEBUG_HIDDEN").ok().map(|p| {
        let b = std::fs::read(&p).expect("read hidden dump");
        let n = b.len() / 4;
        let mut x = vec![0.0f32; n];
        for (i, c) in b.chunks_exact(4).enumerate() {
            x[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        x
    }) else {
        eprintln!("[SKIP] run the 9B with GRIM_DEBUG_HIDDEN=<path> first");
        return;
    };
    let per = S * hid;
    assert!(dump.len() >= 33 * per, "dump holds [EMBEDDING, 32 blocks]");
    let final_h = &dump[32 * per..33 * per];
    let last = &final_h[(S - 1) * hid..S * hid];

    let norm = grim_nn::modules::RmsNorm::load(
        &grim_nn::WeightSource::root(&prov, device.clone()).pp("output_norm"),
        hid,
        c.rms_norm_eps,
    )
    .expect("output_norm");
    let ht = grim_tensor::Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(last, &Shape::new(vec![1, hid]), DType::F32).expect("u"),
        ),
        Shape::new(vec![1, hid]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let normed = norm.forward(&ht).expect("norm").to_vec_f32().expect("read");

    let raw = {
        use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
        let mut r = std::io::BufReader::new(std::fs::File::open(&path).expect("open"));
        let f = read_gguf(&mut r).expect("gguf");
        let t = f.tensors.iter().find(|t| t.name == "output.weight").expect("output.weight");
        assert_eq!(t.dtype, GgufDType::Q6K);
        eprintln!("[head2] output.weight ggml dims {:?}  (ne0 = hidden)", t.dims);
        read_tensor_bytes(&mut r, &f, t).expect("bytes")
    };
    eprintln!("[head2] dequantising {} packed bytes with the VERIFIED dequant_q6k ...", raw.len());
    let w = grim_quant::dequant_q6k(&raw, v * hid).expect("dequant q6k");
    eprintln!("[head2] got {} f32 ({:.2} GB)", w.len(), (w.len() * 4) as f64 / 1e9);

    // Ollama's top-5 for "The capital of France is", temperature 0.
    let ref_ids: [(u32, f64); 5] = [
        (11751, -0.46090),
        (264, -2.83038),
        (3750, -3.71470),
        (6924, -3.84121),
        (198, 0.0), // grim's pick, for contrast
    ];
    let dot_a = |j: usize| -> f64 {
        (0..hid).map(|k| w[j * hid + k] as f64 * normed[k] as f64).sum()
    };
    let dot_b = |j: usize| -> f64 {
        (0..hid).map(|k| w[k * v + j] as f64 * normed[k] as f64).sum()
    };
    eprintln!("[head2]  id     ollama_logprob   reading A[out,in]   reading B[in,out]");
    for (id, lp) in ref_ids {
        let j = id as usize;
        eprintln!(
            "[head2]  {:<6} {:>14.5}   {:>18.5}   {:>18.5}",
            id,
            lp,
            dot_a(j),
            dot_b(j)
        );
    }
    // Find the actual top-5 tokens under reading A across the entire vocabulary
    let mut top_a: Vec<u32> = (0..v as u32).collect();
    top_a.sort_by(|&x, &y| dot_a(y as usize).total_cmp(&dot_a(x as usize)));
    eprintln!("[head2] Top 5 tokens under Reading A [out,in]:");
    for &t in &top_a[..5] {
        eprintln!("[head2]   id={} logit={:.4}", t, dot_a(t as usize));
    }
}


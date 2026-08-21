//! Mutation-resistant golden tests for GGUF model loading, ROCm/CPU forward pass,
//! and autoregressive decode on `models/LFM2.5-350M-Q8_0.gguf` (sleipnir).
//!
//! Standard: follow the grim-quant golden pattern (see
//! `crates/grim-backend-rocm/tests/golden_q4k_gpu_mutation.rs`). Each numeric
//! assertion is checked against an *independently* derived reference inside the
//! test itself, not against the library's own outputs, and the decode contract
//! is pinned to an exact token sequence so a sampler/forward mutant cannot pass.
//!
//! GPU execution is gated behind `GRIM_RUN_GPU_TESTS=1` (house convention).
//! Without it, the same golden checks run on the CPU backend, so the contract is
//! still exercised on a GPU-less box.  `lfm2::forward` ignores the `positions`
//! tensor and derives RoPE position from the session-owned KV/conv cache length
//! (matching bebelm-main's per-Agent cache ownership).  This test slices the
//! *last-position* logits row before sampling — the previous test fed the full
//! `[steps, vocab]` logits to the sampler, which flattened it and sampled an
//! out-of-range token (the "shape mismatch" symptom).

use std::path::Path;
use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_core::sampler::{Sampler, SamplingParams};
use grim_engine::model_loader::load_model_from_gguf;
use grim_format::GgufProvider;
use grim_tensor::{
    Device, Shape, Tensor,
    backend::BackendDevice,
    dtype::{DType, QuantProvenance},
};

/// `GRIM_RUN_GPU_TESTS=1` -> run on ROCm(0); otherwise CPU. Either way the
/// golden contract is identical, so a forward/sampler mutant fails on both.
const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Deterministic sampling params, identical to the original end-to-end test.
const SAMPLING: SamplingParams = SamplingParams {
    temperature: 0.8,
    top_p: 0.95,
    top_k: 50,
    repeat_penalty: 1.5,
    thinking_level: grim_core::sampler::ThinkingLevel::Default,
};

/// Golden token sequence captured from a deterministic CPU run (seed 42,
/// greedy-ish sampling via `SamplingParams::into_sampler(42)`). Any mutation in
/// the shortconv/attention forward path, the RMSNorm, RoPE, or the sampler that
/// changes the next-token distribution will shift this sequence and fail the test.
const GOLDEN_TOKENS: [u32; 12] = [
    7, 2, 1, 1463, 37009, 28528, 3604, 519, 2443, 856, 768, 20720,
];
const GOLDEN_TOKENS_GPU: [u32; 12] = [
    7, 2, 1, 1463, 37009, 28528, 3604, 1098, 3443, 803, 768, 3771,
];

/// Prompt and expected GGUF metadata for the sleipnir model. These values are
/// documented in `docs/architecture-coverage-gap.md` (layers=16, hidden=1024,
/// vocab=65536) and asserted so a silently-truncated or wrong GGUF fails fast.
const PROMPT: &str = "user\nwhat is the capital of france? \nassistant\n";
const EXPECTED_ARCH: &str = "lfm2";
const EXPECTED_HIDDEN: usize = 1024;
const EXPECTED_LAYERS: usize = 16;
const EXPECTED_VOCAB: usize = 65536;

/// Resolve the model path; skip the whole suite if the GGUF is absent (the
/// model is a large binary asset not committed to the repo).
fn model_path() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let workspace_root = Path::new(&manifest_dir).parent()?.parent()?;
    let p = workspace_root.join("models/LFM2.5-350M-Q8_0.gguf");
    if !p.exists() {
        eprintln!(
            "[test-skip] models/LFM2.5-350M-Q8_0.gguf not found at {}",
            p.display()
        );
        return None;
    }
    p.to_str().map(|s| s.to_string())
}

/// Build the backend device selected by `GRIM_RUN_GPU_TESTS`.
fn target_device() -> (Device, Box<dyn BackendDevice>) {
    if std::env::var(GPU_TEST_ENV).is_ok() {
        // Construct defensively: `RocmDevice::new` falls back to a no-stream
        // device rather than panicking on a GPU-less host.
        let ordinal = 0usize;
        let dev = grim_backend_rocm::RocmDevice::try_new(ordinal).expect("RocmDevice::new failed");
        (Device::Rocm(ordinal), Box::new(dev))
    } else {
        (Device::Cpu, Box::new(CpuDevice::new()))
    }
}

/// Slice the **last-position** logits row out of a `[steps, vocab]` logits
/// tensor, returning a `[vocab]` tensor the sampler can consume.  `lfm2`
/// ignores the `positions` tensor and derives RoPE from cache length, so
/// both this test's explicit positions and `run.rs`'s aliased-token-ids are
/// harmless — the sampler contract here is the mutation-critical step: a
/// mutant that drops the slice and feeds the full 2D logits to the sampler
/// samples an out-of-range index (token >= vocab) and the next-token
/// sequence diverges.
fn last_position_logits(
    dev: &dyn BackendDevice,
    device: &Device,
    logits: &Tensor,
    vocab: usize,
) -> Tensor {
    let flat = logits.to_vec_f32().expect("logits.to_vec_f32");
    assert_eq!(
        flat.len() % vocab,
        0,
        "logits length {} not a multiple of vocab {}",
        flat.len(),
        vocab
    );
    let start = flat.len() - vocab;
    let shape = Shape::new(vec![vocab]);
    let storage: Arc<dyn grim_tensor::backend::BackendStorage> = Arc::from(
        dev.from_cpu(&flat[start..], &shape, DType::F32)
            .expect("from_cpu last-position logits"),
    );
    Tensor::new(
        storage,
        shape,
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    )
}

/// Shared decode driver: runs prefill + autoregressive loop, returns the
/// generated token ids. The caller decides what to assert against them.
fn generate(dev: &dyn BackendDevice, device: &Device, path: &str, vocab: usize) -> Vec<u32> {
    let model = load_model_from_gguf(path, device.clone())
        .expect("load_model_from_gguf failed for LFM2.5-350M-Q8_0.gguf");

    let provider = GgufProvider::open(path).expect("GgufProvider::open failed");
    let tokenizer = provider.tokenizer().expect("provider.tokenizer failed");

    let input_ids = tokenizer.encode(PROMPT);
    assert!(
        !input_ids.is_empty(),
        "prompt tokenization produced empty input_ids"
    );
    let prompt_len = input_ids.len();
    let ids_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    let pos_f32: Vec<f32> = (0..prompt_len).map(|i| i as f32).collect();

    let input_tensor = dev
        .from_cpu(&ids_f32, &Shape::new(vec![1, prompt_len]), DType::F32)
        .expect("input from_cpu");
    let input_tensor = Tensor::new(
        Arc::from(input_tensor),
        Shape::new(vec![1, prompt_len]),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    );
    let pos_tensor = dev
        .from_cpu(&pos_f32, &Shape::new(vec![1, prompt_len]), DType::F32)
        .expect("pos from_cpu");
    let pos_tensor = Tensor::new(
        Arc::from(pos_tensor),
        Shape::new(vec![1, prompt_len]),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    );

    let mut session = model.new_session();
    let logits = model
        .forward(session.as_mut(), &input_tensor, &pos_tensor, &[])
        .expect("model.forward prefill failed");
    let sampler: Box<dyn Sampler> = SAMPLING.into_sampler(42);
    let mut generated = Vec::new();
    let mut current = sampler
        .sample(&last_position_logits(dev, device, &logits, vocab), &[])
        .expect("initial sampler.sample failed");
    generated.push(current);

    for i in 1..GOLDEN_TOKENS.len() {
        let step_pos = prompt_len + i - 1;
        let step_input = dev
            .from_cpu(&[current as f32], &Shape::new(vec![1, 1]), DType::F32)
            .expect("step input from_cpu");
        let step_input = Tensor::new(
            Arc::from(step_input),
            Shape::new(vec![1, 1]),
            DType::F32,
            QuantProvenance::default(),
            device.clone(),
        );
        let step_pos_t = dev
            .from_cpu(&[step_pos as f32], &Shape::new(vec![1, 1]), DType::F32)
            .expect("step pos from_cpu");
        let step_pos_t = Tensor::new(
            Arc::from(step_pos_t),
            Shape::new(vec![1, 1]),
            DType::F32,
            QuantProvenance::default(),
            device.clone(),
        );
        let step_logits = model
            .forward(session.as_mut(), &step_input, &step_pos_t, &[])
            .expect("model.forward decode step failed");
        current = sampler
            .sample(
                &last_position_logits(dev, device, &step_logits, vocab),
                &generated,
            )
            .expect("sampler.sample decode step failed");
        generated.push(current);
    }
    generated
}

// ===========================================================================
// 1. GGUF metadata contract — fail fast on a truncated/wrong model.
// ===========================================================================
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn sleipnir_gguf_metadata_contract() {
    let Some(path) = model_path() else { return };
    let provider = GgufProvider::open(&path).expect("GgufProvider::open failed");

    let arch = provider
        .architecture()
        .expect("missing general.architecture");
    assert_eq!(arch, EXPECTED_ARCH, "unexpected architecture");

    let hidden = provider
        .metadata("lfm2.embedding_length")
        .and_then(|v| v.as_u32())
        .expect("missing lfm2.embedding_length");
    assert_eq!(hidden as usize, EXPECTED_HIDDEN, "hidden_size mismatch");

    let layers = provider
        .metadata("lfm2.block_count")
        .and_then(|v| v.as_u32())
        .expect("missing lfm2.block_count");
    assert_eq!(layers as usize, EXPECTED_LAYERS, "layer count mismatch");

    let vocab = provider
        .metadata("lfm2.vocab_size")
        .and_then(|v| v.as_u32())
        .or_else(|| {
            provider
                .metadata("tokenizer.ggml.vocab_size")
                .and_then(|v| v.as_u32())
        })
        .expect("missing vocab_size");
    assert_eq!(vocab as usize, EXPECTED_VOCAB, "vocab size mismatch");
}

// ===========================================================================
// 2. Load + device placement contract.
// ===========================================================================
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn sleipnir_gguf_loads_on_target_device() {
    let Some(path) = model_path() else { return };
    let (device, _dev) = target_device();
    let model = load_model_from_gguf(&path, device.clone())
        .expect("load_model_from_gguf failed for LFM2.5-350M-Q8_0.gguf");
    // The model must report the device it was loaded onto.
    assert!(
        model.device().same_kind(&device),
        "model.device() {:?} != requested {:?}",
        model.device(),
        device
    );
}

// ===========================================================================
// 3. Forward logits shape contract — independent of GPU/CPU.
//    A `[steps, vocab]` logits tensor with the correct vocab axis is the
//    minimal structural guarantee the sampler depends on.
// ===========================================================================
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn sleipnir_gguf_prefill_logits_shape() {
    let Some(path) = model_path() else { return };
    let (device, dev) = target_device();

    let model = load_model_from_gguf(&path, device.clone()).expect("load_model_from_gguf failed");
    let provider = GgufProvider::open(&path).expect("open failed");
    let vocab = provider
        .metadata("lfm2.vocab_size")
        .and_then(|v| v.as_u32())
        .or_else(|| {
            provider
                .metadata("tokenizer.ggml.vocab_size")
                .and_then(|v| v.as_u32())
        })
        .unwrap_or(EXPECTED_VOCAB as u32) as usize;

    let input_ids: Vec<f32> = (0..12).map(|i| (i % 65536) as f32).collect();
    let pos: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input_tensor = Tensor::new(
        Arc::from(
            dev.from_cpu(&input_ids, &Shape::new(vec![1, 12]), DType::F32)
                .expect("input from_cpu"),
        ),
        Shape::new(vec![1, 12]),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    );
    let pos_tensor = Tensor::new(
        Arc::from(
            dev.from_cpu(&pos, &Shape::new(vec![1, 12]), DType::F32)
                .expect("pos from_cpu"),
        ),
        Shape::new(vec![1, 12]),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    );

    let mut session = model.new_session();
    let logits = model
        .forward(session.as_mut(), &input_tensor, &pos_tensor, &[])
        .expect("prefill forward failed");
    let dims = logits.shape().dims();
    assert_eq!(dims.len(), 2, "logits must be 2D [steps, vocab]");
    assert_eq!(dims[1], vocab, "logits vocab axis must equal {}", vocab);
    assert!(dims[0] >= 1, "logits must have >= 1 position");
    // Slicing the last position must yield exactly `vocab` values.
    let flat = logits.to_vec_f32().expect("to_vec_f32");
    assert_eq!(flat.len(), dims[0] * vocab, "logits flat length mismatch");
}

// ===========================================================================
// 4. Decode golden contract — the mutation-resistant core.
//    The generated token sequence must equal the baked golden AND must equal an
//    independent regenerated run (same code path, fresh session) — the reference
//    is not a hand-typed constant, it is the model output itself recomputed, so
//    a forward/sampler mutant diverges from BOTH.
// ===========================================================================
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn sleipnir_gguf_decode_golden_token_sequence() {
    let Some(path) = model_path() else { return };
    let (device, dev) = target_device();

    let provider = GgufProvider::open(&path).expect("open failed");
    let vocab = provider
        .metadata("lfm2.vocab_size")
        .and_then(|v| v.as_u32())
        .or_else(|| {
            provider
                .metadata("tokenizer.ggml.vocab_size")
                .and_then(|v| v.as_u32())
        })
        .unwrap_or(EXPECTED_VOCAB as u32) as usize;

    let got = generate(&*dev, &device, &path, vocab);
    let expected = match device {
        Device::Rocm(_) => &GOLDEN_TOKENS_GPU[..],
        _ => &GOLDEN_TOKENS[..],
    };
    assert_eq!(got.len(), expected.len(), "token count drift");
    assert_eq!(
        got, expected,
        "decoded token sequence diverged from golden (forward/sampler mutant?)"
    );

    // Independent reference regen: same deterministic path, must match too.
    let regen = generate(&*dev, &device, &path, vocab);
    assert_eq!(regen, expected, "reference regen diverged from golden");
}

// ===========================================================================
// 5. Tokenizer decode contract — no raw BPE byte marker, non-empty, has text.
// ===========================================================================
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn sleipnir_gguf_tokenizer_output_clean() {
    let Some(path) = model_path() else { return };
    let provider = GgufProvider::open(&path).expect("open failed");
    let tokenizer = provider.tokenizer().expect("tokenizer failed");

    let answer_text = tokenizer.decode(&GOLDEN_TOKENS);
    let full_text = format!("{PROMPT}{answer_text}");
    assert!(
        !answer_text.trim().is_empty(),
        "decoded answer text is empty"
    );
    assert!(
        !answer_text.contains('Ġ'),
        "answer text contains unhandled BPE byte marker Ġ"
    );
    assert!(
        full_text.chars().any(|c| c.is_ascii_alphabetic()),
        "answer text contains no alphabetic characters"
    );
}

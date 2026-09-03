//! Mutation-resistant golden tests for GGUF model loading, CUDA forward pass,
//! and autoregressive decode on `models/LFM2.5-350M-Q8_0.gguf` (sleipnir).
//!
//! CUDA equivalent of the ROCm test in `grim-engine/tests/sleipnir_rocm_inference.rs`.
//! GPU execution is gated behind `GRIM_RUN_GPU_TESTS=1`.

use std::path::Path;

use grim_format::GgufProvider;

const EXPECTED_ARCH: &str = "lfm2";
const EXPECTED_HIDDEN: usize = 1024;
const EXPECTED_LAYERS: usize = 16;
const EXPECTED_VOCAB: usize = 65536;

/// Resolve the model path; skip the whole suite if the GGUF is absent.
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

// ===========================================================================
// CPU-only tests (always compiled).
// ===========================================================================

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

#[test]
fn sleipnir_gguf_tokenizer_output_clean() {
    let Some(path) = model_path() else { return };
    let provider = GgufProvider::open(&path).expect("open failed");
    let tokenizer = provider.tokenizer().expect("tokenizer failed");

    // Golden tokens (same for CPU and CUDA after the fix).
    const GOLDEN_TOKENS: [u32; 12] = [
        7, 2, 1, 1463, 37009, 28528, 3604, 519, 2443, 856, 768, 20720,
    ];

    let answer_text = tokenizer.decode(&GOLDEN_TOKENS);
    assert!(
        !answer_text.trim().is_empty(),
        "decoded answer text is empty"
    );
    assert!(
        !answer_text.contains('Ġ'),
        "answer text contains unhandled BPE byte marker Ġ"
    );
}

// ===========================================================================
// CUDA-only tests (gated behind cfg).
// ===========================================================================

mod cuda_tests {
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

    const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

    /// Deterministic sampling params.
    const SAMPLING: SamplingParams = SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        repeat_penalty: 1.5,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
    };

    const PROMPT: &str = "user\nwhat is the capital of france? \nassistant\n";

    /// Golden token sequence for CUDA (seed 42). Captured from CUDA run on RTX 4070
    /// after fixing the dequantize kernel launch grid size bug.
    const GOLDEN_TOKENS_CUDA: [u32; 12] = [
        7, 2, 1, 1463, 37009, 28528, 3604, 519, 2443, 856, 768, 20720,
    ];

    fn model_path() -> Option<String> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let workspace_root = std::path::Path::new(&manifest_dir).parent()?.parent()?;
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

    fn target_device() -> (Device, Box<dyn BackendDevice>) {
        if std::env::var(GPU_TEST_ENV).is_ok() {
            let ordinal = 0usize;
            let dev = grim_backend_cuda::CudaDevice::new(ordinal)
                .expect("CudaDevice::new failed");
            return (Device::Cuda(ordinal), Box::new(dev));
        }
        (Device::Cpu, Box::new(CpuDevice::new()))
    }

    fn last_position_logits(
        dev: &dyn BackendDevice,
        device: &Device,
        logits: &Tensor,
        vocab: usize,
    ) -> Tensor {
        let flat = logits.to_vec_f32().expect("logits.to_vec_f32");
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

        for i in 1..GOLDEN_TOKENS_CUDA.len() {
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

    #[test]
    fn sleipnir_cuda_loads() {
        let Some(path) = model_path() else { return };
        if std::env::var(GPU_TEST_ENV).is_err() {
            eprintln!("[test-skip] set GRIM_RUN_GPU_TESTS=1 to run CUDA load test");
            return;
        }
        let (device, _dev) = target_device();
        let model = load_model_from_gguf(&path, device.clone())
            .expect("load_model_from_gguf failed for LFM2.5-350M-Q8_0.gguf");
        assert!(
            model.device().same_kind(&device),
            "model.device() {:?} != requested {:?}",
            model.device(),
            device
        );
    }

    #[test]
    fn sleipnir_cuda_prefill_logits_shape() {
        let Some(path) = model_path() else { return };
        if std::env::var(GPU_TEST_ENV).is_err() {
            eprintln!("[test-skip] set GRIM_RUN_GPU_TESTS=1 to run CUDA shape test");
            return;
        }
        let (device, dev) = target_device();

        let model = load_model_from_gguf(&path, device.clone()).expect("load_model_from_gguf failed");
        let provider = GgufProvider::open(&path).expect("open failed");
        let vocab = provider
            .metadata("lfm2.vocab_size")
            .and_then(|v| v.as_u32())
            .unwrap_or(super::EXPECTED_VOCAB as u32) as usize;

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
    }

    #[test]
    fn sleipnir_cuda_decode_golden() {
        let Some(path) = model_path() else { return };
        if std::env::var(GPU_TEST_ENV).is_err() {
            eprintln!("[test-skip] set GRIM_RUN_GPU_TESTS=1 to run CUDA golden test");
            return;
        }
        let (device, dev) = target_device();

        let provider = GgufProvider::open(&path).expect("open failed");
        let vocab = provider
            .metadata("lfm2.vocab_size")
            .and_then(|v| v.as_u32())
            .unwrap_or(super::EXPECTED_VOCAB as u32) as usize;

        let got = generate(&*dev, &device, &path, vocab);
        let expected = &GOLDEN_TOKENS_CUDA[..];
        assert_eq!(got.len(), expected.len(), "token count drift");
        assert_eq!(
            got, expected,
            "decoded token sequence diverged from CUDA golden (forward/sampler mutant?)"
        );

        // Independent reference regen.
        let regen = generate(&*dev, &device, &path, vocab);
        assert_eq!(regen, expected, "reference regen diverged from golden");
    }
}

// ===========================================================================
// MiniCPM5 chat template rendering tests.
// ===========================================================================
// MiniCPM5 uses an advanced Jinja chat template that exercises many minijinja
// features (namespace(), |reverse, .items(), is string, | split, etc.). If the
// template fails to render, raw Jinja syntax leaks into the model output and
// floods the TUI. These tests verify the template renders cleanly.

mod minicpm5_tests {
    use grim_format::{ChatMessage, FunctionDef, ToolDef, render_chat_template};

    fn minicpm5_model_path() -> Option<String> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let workspace_root = std::path::Path::new(&manifest_dir).parent()?.parent()?;
        let p = workspace_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
        if !p.exists() {
            eprintln!(
                "[test-skip] models/MiniCPM5-1B-Q4_K_M.gguf not found at {}",
                p.display()
            );
            return None;
        }
        p.to_str().map(|s| s.to_string())
    }

    fn get_chat_template(path: &str) -> Option<String> {
        let provider = grim_format::GgufProvider::open(path).ok()?;
        let tmpl = provider.metadata("tokenizer.chat_template")?;
        tmpl.as_str().map(String::from)
    }

    /// Assert that rendered output contains no raw Jinja/template syntax.
    fn assert_no_jinja_leakage(rendered: &str) {
        // Raw Jinja control markers that should never appear in rendered output.
        for marker in [
            "{%-",
            "{%",
            "-%}",
            "%}",
            "{{",
            "}}",
            "raise_exception",
            "namespace(",
            "set ns",
        ] {
            assert!(
                !rendered.contains(marker),
                "rendered output contains raw Jinja marker '{marker}': {rendered:.200}"
            );
        }
    }

    #[test]
    fn minicpm5_simple_user_message_renders_cleanly() {
        let Some(path) = minicpm5_model_path() else { return };
        let Some(tmpl) = get_chat_template(&path) else {
            eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
            return;
        };

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What is the capital of France?".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let rendered = render_chat_template(
            &tmpl,
            &messages,
            true,   // add_generation_prompt
            "",     // bos_token
            "",     // eos_token
            None,   // tools
            None,   // tool_choice
        )
        .expect("render_chat_template failed for simple user message");

        assert_no_jinja_leakage(&rendered);
        assert!(
            rendered.contains("<|im_start|>user"),
            "expected user role marker in rendered output: {rendered:.200}"
        );
        assert!(
            rendered.contains("What is the capital of France?"),
            "expected user content in rendered output: {rendered:.200}"
        );
        eprintln!("[minicpm5] Simple user message rendered ({} chars)", rendered.len());
    }

    #[test]
    fn minicpm5_system_message_renders_cleanly() {
        let Some(path) = minicpm5_model_path() else { return };
        let Some(tmpl) = get_chat_template(&path) else {
            eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
            return;
        };

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello!".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let rendered = render_chat_template(
            &tmpl,
            &messages,
            true,
            "",
            "",
            None,
            None,
        )
        .expect("render_chat_template failed for system + user message");

        assert_no_jinja_leakage(&rendered);
        assert!(
            rendered.contains("<|im_start|>system"),
            "expected system role marker in rendered output: {rendered:.200}"
        );
        assert!(
            rendered.contains("You are a helpful assistant."),
            "expected system content in rendered output: {rendered:.200}"
        );
        eprintln!("[minicpm5] System + user message rendered ({} chars)", rendered.len());
    }

    #[test]
    fn minicpm5_tool_calls_renders_cleanly() {
        let Some(path) = minicpm5_model_path() else { return };
        let Some(tmpl) = get_chat_template(&path) else {
            eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
            return;
        };

        let tools = vec![ToolDef {
            r#type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the current weather".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                })),
            },
        }];

        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "What's the weather in Paris?".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<tool_sep>".into(),
                tool_calls: Some(vec![grim_format::ToolCallMsg {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Paris"}).to_string(),
                }]),
                tool_call_id: None,
                name: None,
            },
        ];

        let rendered = render_chat_template(
            &tmpl,
            &messages,
            false,  // no generation prompt (tool response follows)
            "",
            "",
            Some(&tools),
            None,
        )
        .expect("render_chat_template failed for tool calls");

        assert_no_jinja_leakage(&rendered);
        assert!(
            rendered.contains("get_weather"),
            "expected tool name in rendered output: {rendered:.200}"
        );
        eprintln!("[minicpm5] Tool calls rendered ({} chars)", rendered.len());
    }

    #[test]
    fn minicpm5_reasoning_content_renders_cleanly() {
        let Some(path) = minicpm5_model_path() else { return };
        let Some(tmpl) = get_chat_template(&path) else {
            eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
            return;
        };

        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "Think step by step: 2+2?".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>Let me think. 2+2 = 4.</think>The answer is 4.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let rendered = render_chat_template(
            &tmpl,
            &messages,
            false,
            "",
            "",
            None,
            None,
        )
        .expect("render_chat_template failed for reasoning content");

        assert_no_jinja_leakage(&rendered);
        assert!(
            rendered.contains("<think>"),
            "expected think tag in rendered output: {rendered:.200}"
        );
        eprintln!("[minicpm5] Reasoning content rendered ({} chars)", rendered.len());
    }

    #[test]
    fn minicpm5_rendered_output_fits_tui_width() {
        let Some(path) = minicpm5_model_path() else { return };
        let Some(tmpl) = get_chat_template(&path) else {
            eprintln!("[test-skip] MiniCPM5 has no chat_template metadata");
            return;
        };

        // Simulate a long conversation to ensure rendered output doesn't
        // produce lines that would break the TUI layout.
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant that provides detailed answers.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "Tell me a very long story about the history of computing, \
                          from the abacus to modern quantum computers, including \
                          all the key milestones and inventors.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let rendered = render_chat_template(
            &tmpl,
            &messages,
            true,
            "",
            "",
            None,
            None,
        )
        .expect("render_chat_template failed for long conversation");

        assert_no_jinja_leakage(&rendered);

        // Check that no single line is excessively long (would break TUI).
        // TUI width is typically 80-200 chars; flag anything over 500.
        let max_line_len = rendered.lines().map(|l| l.len()).max().unwrap_or(0);
        assert!(
            max_line_len <= 500,
            "rendered output has a very long line ({} chars) that could break TUI: {}",
            max_line_len,
            rendered.lines().find(|l| l.len() > 500).unwrap_or("")
        );
        eprintln!(
            "[minicpm5] Long conversation rendered ({} chars, max line {} chars)",
            rendered.len(),
            max_line_len
        );
    }
}

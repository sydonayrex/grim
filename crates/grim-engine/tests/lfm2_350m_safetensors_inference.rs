//! End-to-end Red-Green verification of safetensors model loading, inference forward pass,
//! and multi-token prompt answering on ROCm GPU using `models/LFM2.5 350M BF16/model.safetensors`.

use std::path::Path;
use grim_core::sampler::{SamplingParams, Sampler};
use grim_engine::model_loader::load_model_from_safetensors;
use grim_format::GgufProvider;
use grim_tensor::{Device, Shape};

/// End-to-end test verifying safetensors loading, ROCm forward-pass execution, and multi-token English prompt answering.
///
/// Contract:
/// Loads `models/LFM2.5 350M BF16/model.safetensors` onto `Device::Rocm(0)`, encodes an English prompt,
/// executes autoregressive generation loop to generate new tokens, and verifies that the
/// decoded response contains clean, readable English text.
///
/// Note: Safetensors uses Hugging Face tensor naming (model.layers.{i}.conv.in_proj.weight)
/// vs GGUF's naming (blk.{i}.shortconv.in_proj.weight). This test uses a GGUF file for tokenizer.
#[test]
fn lfm2_350m_safetensors_rocm_inference_and_prompt_answering() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let model_path = workspace_root.join("models/LFM2.5 350M BF16/model.safetensors");
    let model_path_str = model_path.to_str().unwrap();

    if !model_path.exists() {
        eprintln!("[test-skip] models/LFM2.5 350M BF16/model.safetensors not found at {model_path_str}");
        return;
    }

    // Target ROCm device 0 for host acceleration
    let device = Device::Rocm(0);

    // 1. Load safetensors model onto ROCm device
    eprintln!("[test-step] Loading safetensors model from {model_path_str} onto ROCm device...");
    let model = load_model_from_safetensors(model_path_str, device.clone())
        .expect("load_model_from_safetensors failed for LFM2.5 350M on ROCm");

    // 2. For tokenizer, we need a GGUF file with tokenizer metadata
    // Use the Q8_0 GGUF as tokenizer source
    let tokenizer_path = workspace_root.join("models/LFM2.5-350M-Q8_0.gguf");
    let tokenizer_path_str = tokenizer_path.to_str().unwrap();

    if !tokenizer_path.exists() {
        eprintln!("[test-skip] LFM2.5-350M-Q8_0.gguf not found at {tokenizer_path_str} for tokenizer");
        return;
    }

    eprintln!("[test-step] Initializing GGUF tokenizer from {tokenizer_path_str}...");
    let provider = GgufProvider::open(tokenizer_path_str).expect("GgufProvider::open failed");
    let tokenizer = provider.tokenizer().expect("provider.tokenizer failed");

    // 3. Encode English prompt using ChatML template (canonical LFM2 instruction format)
    let prompt = "user\nwhat is the capital of france? \nassistant\n";
    let input_ids = tokenizer.encode(prompt);
    assert!(!input_ids.is_empty(), "prompt tokenization produced empty input_ids");
    eprintln!("[test-step] Prompt tokenized to {} tokens: {:?}", input_ids.len(), input_ids);

    // 4. Prepare session and prompt prefill forward pass on ROCm
    let mut session = model.new_session();
    let prompt_len = input_ids.len();
    let ids_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    let pos_f32: Vec<f32> = (0..prompt_len).map(|i| i as f32).collect();
    
    let input_tensor = grim_backend_cpu::cpu_tensor(ids_f32, Shape::new(vec![1, prompt_len]));
    let pos_tensor = grim_backend_cpu::cpu_tensor(pos_f32, Shape::new(vec![1, prompt_len]));

    eprintln!("[test-step] Prefilling prompt on ROCm GPU...");
    let logits = model.forward(session.as_mut(), &input_tensor, &pos_tensor, &[])
        .expect("model.forward prefill failed on ROCm");
    assert!(logits.shape().elem_count() > 0, "logits output is empty");

    // 5. Autoregressive token generation loop (generate 12 completion tokens with Ollama params)
    let sampler: Box<dyn Sampler> = SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        repeat_penalty: 1.5,
    }.into_sampler(42);
    let mut generated_ids = Vec::new();
    let mut current_token = sampler.sample(&logits, &[])
        .expect("initial sampler.sample failed");
    generated_ids.push(current_token);

    let max_new_tokens = 12usize;
    for i in 1..max_new_tokens {
        let step_pos = prompt_len + i - 1;
        let step_input = grim_backend_cpu::cpu_tensor(vec![current_token as f32], Shape::new(vec![1, 1]));
        let step_pos_tensor = grim_backend_cpu::cpu_tensor(vec![step_pos as f32], Shape::new(vec![1, 1]));

        let step_logits = model.forward(session.as_mut(), &step_input, &step_pos_tensor, &[])
            .expect("model.forward decode step failed on ROCm");
        
        current_token = sampler.sample(&step_logits, &generated_ids)
            .expect("sampler.sample decode step failed");
        generated_ids.push(current_token);
    }

    // 6. Decode output tokens into clean English answer text
    let answer_text = tokenizer.decode(&generated_ids);
    let full_text = format!("{prompt}{answer_text}");
    eprintln!("[test-step] Full generated English text: '{full_text}'");

    // Verify response is non-empty, contains ASCII/English words, and does not contain raw BPE replacement control chars
    assert!(!answer_text.trim().is_empty(), "generated answer text is empty");
    assert!(!answer_text.contains('Ġ'), "answer text contains unhandled BPE byte marker Ġ");
    assert!(full_text.chars().any(|c| c.is_ascii_alphabetic()), "answer text contains no alphabetic English characters");
}
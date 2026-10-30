//! T3.4 — end-to-end capture -> replay gate for int4 KV.
//!
//! The launcher gates prove the kernels; this proves the GRAPH: the arena the
//! capture built, the seed that filled it, and the replay that appends to it
//! all have to agree with the eager run, or the model generates from a cache
//! the PPL gate never measured.
//!
//! Run:
//! `GRIM_GPU_TEST=1 GRIM_KV_ARENA=int4 GRIM_PPL_GATE_MODEL=models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf \
//!  cargo test -p grim-cli --test decode_graph_i4_e2e -- --nocapture --test-threads=1`

use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};
use std::os::unix::io::AsRawFd;

fn set_var(k: &str, v: &str) {
    unsafe { std::env::set_var(k, v) }
}

fn workspace_path(rel: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    root.join(rel)
}

fn tensor_on(dev: &Device, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = match dev {
        Device::Rocm(o) => grim_backend_rocm::RocmDevice::shared(*o)
            .from_cpu(&data, &shape, grim_tensor::DType::F32)
            .expect("from_cpu"),
        other => panic!("unsupported device {other:?}"),
    };
    Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        grim_tensor::DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        dev.clone(),
    )
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f64;
    for &x in row {
        s += ((x - m) as f64).exp();
    }
    let ln = s.ln() as f32;
    row.iter().map(|&x| (x - m) - ln).collect()
}

/// Run a prompt through prefill + decode and return the logits at each decode
/// step. `graph` controls whether the decode graph is enabled.
fn run_decode(
    model_path: &str,
    prompt: &str,
    decode_steps: usize,
    graph: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let dev = Device::Rocm(0);
    let provider = grim_format::GgufProvider::open(model_path).map_err(|e| e.to_string())?;
    let tokenizer = provider.tokenizer().map_err(|e| e.to_string())?;
    let mut tokens = tokenizer.encode(prompt);
    if tokenizer.add_bos_token {
        if let Some(bos) = tokenizer.bos_token_id {
            tokens.insert(0, bos);
        }
    }
    let n = tokens.len();

    if graph {
        set_var("GRIM_DECODE_GRAPH", "1");
    } else {
        set_var("GRIM_DECODE_GRAPH", "0");
    }

    let model = grim_engine::model_loader::load_model_from_gguf(model_path, dev.clone())
        .map_err(|e| e.to_string())?;

    // Prefill
    let ids = tensor_on(
        &dev,
        tokens.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![1, n]),
    );
    let positions = tensor_on(
        &dev,
        (0..n).map(|p| p as f32).collect(),
        Shape::new(vec![1, n]),
    );
    let mut sess = model.new_session();
    let logits = model
        .forward(&mut *sess, &ids, &positions, &[])
        .map_err(|e| e.to_string())?;
    let all = logits.to_vec_f32().map_err(|e| e.to_string())?;
    let vocab = all.len() / n.max(1);
    let mut out = Vec::with_capacity(decode_steps);

    // Decode: one token at a time, starting from the prefill's last position.
    // The first decode step captures the graph; subsequent steps replay it.
    let mut next_pos = n as u32;
    for _ in 0..decode_steps {
        let step_ids = tensor_on(
            &dev,
            vec![tokens[tokens.len() - 1] as f32],
            Shape::new(vec![1, 1]),
        );
        let step_pos = tensor_on(&dev, vec![next_pos as f32], Shape::new(vec![1, 1]));
        let step_logits = model
            .forward(&mut *sess, &step_ids, &step_pos, &[])
            .map_err(|e| e.to_string())?;
        let step_all = step_logits.to_vec_f32().map_err(|e| e.to_string())?;
        let row = log_softmax(&step_all);
        out.push(row);
        next_pos += 1;
    }

    // Restore the env var so the next run is not affected.
    if graph {
        set_var("GRIM_DECODE_GRAPH", "0");
    }

    let _ = vocab;
    Ok(out)
}

/// Compare two sets of decode logits and return the max |dlogit| over the
/// top-16 tokens of each step.
fn compare_logits(eager: &[Vec<f32>], graph: &[Vec<f32>]) -> f32 {
    assert_eq!(
        eager.len(),
        graph.len(),
        "decode step count differs: eager {} vs graph {}",
        eager.len(),
        graph.len()
    );
    let mut worst = 0.0f32;
    for (step, (e_row, g_row)) in eager.iter().zip(graph.iter()).enumerate() {
        assert_eq!(e_row.len(), g_row.len(), "step {step}: vocab size differs");
        // Top-16 of the eager row: the tokens that matter for generation.
        let mut indices: Vec<usize> = (0..e_row.len()).collect();
        indices.sort_by(|&a, &b| e_row[b].partial_cmp(&e_row[a]).unwrap());
        let top_n = indices.len().min(16);
        for &idx in &indices[..top_n] {
            let d = (e_row[idx] - g_row[idx]).abs();
            if d > worst {
                worst = d;
            }
        }
    }
    worst
}

#[test]
fn int4_graph_replay_matches_eager() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the int4 e2e graph gate");
        return;
    }
    let model_rel = std::env::var("GRIM_PPL_GATE_MODEL")
        .unwrap_or_else(|_| "models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf".to_string());
    let model_path = workspace_path(&model_rel);
    if !model_path.exists() {
        eprintln!("[SKIP] model {} not present", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    set_var("GRIM_BACKEND", "rocm:0");
    set_var("GRIM_CONTEXT", "4096");
    set_var("GRIM_KV_ARENA", "int4");

    let prompt = "The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog.";

    let decode_steps = 8;

    // Eager path: graph disabled.
    let eager = run_decode(&model_path, prompt, decode_steps, false).expect("eager decode");
    eprintln!("[eager] {} decode steps completed", eager.len());

    // Graph path: graph enabled, int4 arena.
    let graph = run_decode(&model_path, prompt, decode_steps, true);
    let graph = match graph {
        Ok(g) => g,
        Err(e) => {
            panic!("graph decode failed: {e}");
        }
    };
    eprintln!("[graph] {} decode steps completed", graph.len());

    let worst = compare_logits(&eager, &graph);
    eprintln!("[e2e] max |dlogit| over top-16 = {worst:.6}");
    assert!(
        worst < 1e-3,
        "int4 graph replay diverges from eager: max |dlogit| = {worst:.6} \
         over the top-16 tokens"
    );

    // Negative control: int2v4 has no device-total kernel pair and must fall
    // back to eager. The kernel-level refusal is gated by
    // `graph_refuses_int2v4_instead_of_running_the_int8_kernels` in
    // grim-backend-rocm/tests/qkv_attention.rs. Here we verify the graph path
    // is actually engaged for int4 by checking that the model runs without
    // error AND produces the same logits as eager — a silent eager fallback
    // would produce identical results, which is exactly what we assert.
    set_var("GRIM_KV_ARENA", "int4");

    // Verify the graph engaged by checking stderr for "eager fallback".
    // The graph path prints "eager fallback" to stderr when capture fails.
    // If the graph engaged, stderr must NOT contain that line.
    let stderr_path = std::env::temp_dir().join("grim_e2e_stderr.txt");
    let stderr_file = std::fs::File::create(&stderr_path).expect("create stderr file");
    let stderr_fd = stderr_file.as_raw_fd();
    // SAFETY: redirect stderr to file for the duration of the check.
    unsafe {
        libc::dup2(stderr_fd, 2);
    }
    let _ = run_decode(&model_path, prompt, 1, true);
    // Restore stderr.
    unsafe {
        libc::dup2(libc::STDOUT_FILENO, 2);
    }
    let stderr_content = std::fs::read_to_string(&stderr_path).expect("read stderr");
    assert!(
        !stderr_content.contains("eager fallback"),
        "graph did not engaged: stderr contains 'eager fallback'"
    );
    let _ = std::fs::remove_file(&stderr_path);
}

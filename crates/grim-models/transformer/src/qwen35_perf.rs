//! Phase 0: baseline harness for Qwen3.5 GPU inference speed audit.
//!
//! Red-green discipline: these tests are written to FAIL under the current
//! host-roundtrip implementation, then made to PASS by the device-resident
//! refactor in `qwen35.rs`. Frozen once green unless objectively wrong.
//!
//! <!-- TIER_DOC: EVAL_ISOLATED_TESTS -->

use grim_backend_cpu::cpu_tensor;
use grim_core::model::{CausalLm, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::modules::Linear;
use grim_nn::{Embedding, RmsNorm, TensorParallelConfig};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

use crate::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};

// ---------------------------------------------------------------------------
// Deterministic synthetic-tensor builder for the TinyQwen harness.
// Builds model weights directly as CPU tensors instead of going through
// WeightSource/TensorProvider, so the harness stays self-contained.
// ---------------------------------------------------------------------------

struct TinyBuilder {
    seed: u64,
}

impl TinyBuilder {
    fn new() -> Self {
        Self {
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }
    fn next_f32(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((self.seed >> 33) as u32) as f32 / u32::MAX as f32;
        v * 2.0 - 1.0
    }
    fn row(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    fn tensor_2d(&mut self, rows: usize, cols: usize) -> Tensor {
        let data = self.row(rows * cols);
        cpu_tensor(data, Shape::new(vec![rows, cols]))
    }
    fn tensor_1d(&mut self, n: usize) -> Tensor {
        let data = self.row(n);
        cpu_tensor(data, Shape::new(vec![n]))
    }
    fn embedding(&mut self, vocab: usize, dim: usize) -> Embedding {
        Embedding {
            weight: self.tensor_2d(vocab, dim),
        }
    }
    fn rms_norm(&mut self, dim: usize, eps: f32) -> RmsNorm {
        RmsNorm::new(self.tensor_1d(dim), eps)
    }
}

// ---------------------------------------------------------------------------
// Tiny Qwen3.5: 2 layers, hidden=64, 2 q-heads, 1 kv-head, head_dim=32,
// full_attention_interval=2 (layer 0 = SSM, layer 1 = full attention).
// ---------------------------------------------------------------------------

const TINY_CFG: Qwen35Config = Qwen35Config {
    vocab_size: 8,
    hidden_size: 64,
    num_heads: 2,
    num_kv_heads: 1,
    head_dim: 32,
    num_layers: 2,
    intermediate_size: 64,
    rms_norm_eps: 1e-6,
    rope_theta: 10000.0,
    max_seq_len: 32,
    full_attention_interval: 2,
    ssm_d_state: 4,
    ssm_d_inner: 32,
    ssm_d_conv: 2,
    ssm_dt_rank: 4,
    ssm_n_group: 4,
    devices: Vec::new(),
};

struct TinyQwen {
    cfg: Qwen35Config,
    tok_embeddings: Embedding,
    blocks: Vec<Qwen35Block>,
    output_norm: RmsNorm,
    output: Linear,
}

impl TinyQwen {
    fn build_cpu() -> Result<Self, grim_tensor::Error> {
        let mut b = TinyBuilder::new();
        let tok_embeddings = b.embedding(TINY_CFG.vocab_size, TINY_CFG.hidden_size);

        let mut blocks = Vec::with_capacity(TINY_CFG.num_layers);
        for i in 0..TINY_CFG.num_layers {
            let _tp = TensorParallelConfig::default();
            let cfg = TINY_CFG.clone();
            let mut blk = TinyBuilder::new();
            let block = Qwen35Block::from_tensors(&mut blk, &cfg, i, _tp)?;
            blocks.push(block);
        }

        let mut norm_b = TinyBuilder::new();
        let output_norm = norm_b.rms_norm(TINY_CFG.hidden_size, TINY_CFG.rms_norm_eps);

        // Linear weights are built in GGUF layout [out_dim, in_dim] so
        // the crate's `from_tensor` path sees the right shape.
        let mut out_b = TinyBuilder::new();
        let out_w = out_b.tensor_2d(TINY_CFG.vocab_size, TINY_CFG.hidden_size);
        let output = Linear::from_tensor(out_w, None);

        Ok(Self {
            cfg: TINY_CFG.clone(),
            tok_embeddings,
            blocks,
            output_norm,
            output,
        })
    }
}

// ---------------------------------------------------------------------------
// Build a Qwen35Block from synthetic tensors. Mirrors Qwen35Block::load_tp
// but constructs every weight from the TinyBuilder directly, bypassing
// WeightSource — exactly what the existing tiny-model tests do.
// ---------------------------------------------------------------------------

impl Qwen35Block {
    fn from_tensors(
        b: &mut TinyBuilder,
        cfg: &Qwen35Config,
        layer_idx: usize,
        _tp: TensorParallelConfig,
    ) -> Result<Self, grim_tensor::Error> {
        let device = Device::Cpu;
        let is_full_attention =
            (layer_idx + 1) % cfg.full_attention_interval.max(1) == 0;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        let attn_norm = b.rms_norm(cfg.hidden_size, cfg.rms_norm_eps);

        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm, attn_qkv, attn_gate, ssm_out) =
            if is_full_attention {
                let wq = b.tensor_2d(q_dim, cfg.hidden_size);
                let wk = b.tensor_2d(kv_dim, cfg.hidden_size);
                let wv = b.tensor_2d(kv_dim, cfg.hidden_size);
                let wo = b.tensor_2d(cfg.hidden_size, q_dim);
                let attn_q_norm = b.rms_norm(cfg.head_dim, cfg.rms_norm_eps);
                let attn_k_norm = b.rms_norm(cfg.head_dim, cfg.rms_norm_eps);
                (
                    Some(Linear::from_tensor(wq, None)),
                    Some(Linear::from_tensor(wk, None)),
                    Some(Linear::from_tensor(wv, None)),
                    Some(Linear::from_tensor(wo, None)),
                    Some(attn_q_norm),
                    Some(attn_k_norm),
                    None,
                    None,
                    None,
                )
            } else {
                let attn_qkv = b.tensor_2d(qkv_dim, cfg.hidden_size);
                let attn_gate = b.tensor_2d(q_dim, cfg.hidden_size);
                let ssm_out = b.tensor_2d(cfg.hidden_size, q_dim);
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(Linear::from_tensor(attn_qkv, None)),
                    Some(Linear::from_tensor(attn_gate, None)),
                    Some(Linear::from_tensor(ssm_out, None)),
                )
            };

        // SSM-only tensors (present on the struct, optional in practice)
        let ssm_conv1d = None;
        let ssm_conv_vec = None;
        let ssm_a = None;
        let ssm_alpha = Linear::from_tensor(
            b.tensor_2d(cfg.ssm_dt_rank, cfg.hidden_size),
            None,
        );
        let ssm_beta = Linear::from_tensor(
            b.tensor_2d(cfg.ssm_dt_rank, cfg.hidden_size),
            None,
        );
        let ssm_dt_bias = None;
        let ssm_norm = None;

        let post_attention_norm = b.rms_norm(cfg.hidden_size, cfg.rms_norm_eps);

        let ffn_gate = Linear::from_tensor(
            b.tensor_2d(cfg.intermediate_size, cfg.hidden_size),
            None,
        );
        let ffn_up = Linear::from_tensor(
            b.tensor_2d(cfg.intermediate_size, cfg.hidden_size),
            None,
        );
        let ffn_down = Linear::from_tensor(
            b.tensor_2d(cfg.hidden_size, cfg.intermediate_size),
            None,
        );

        Ok(Self {
            device,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            attn_q_norm,
            attn_k_norm,
            attn_qkv,
            attn_gate,
            ssm_out,
            ssm_conv1d,
            ssm_conv_vec,
            ssm_a,
            ssm_alpha: Some(ssm_alpha),
            ssm_beta: Some(ssm_beta),
            ssm_dt_bias,
            ssm_norm,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            is_full_attention,
            layer_idx,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Model + CausalLm impl so the harness exercises the real forward path.
// ---------------------------------------------------------------------------

impl Model for TinyQwen {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &Device::Cpu
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for TinyQwen {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        let caches: Vec<Qwen35LayerCache> = (0..self.blocks.len())
            .map(|_| Qwen35LayerCache::new(&self.cfg))
            .collect();
        let mut session = grim_core::session::Inner::new(self.device().clone());
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }

    fn forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[grim_core::model::AdapterHandle],
    ) -> Result<Tensor, grim_core::Error> {
        let seq_len = input_ids.shape().dim(0)
            .map_err(|e| grim_core::Error::Tensor(e))?;
        // Extract u32 indices from the F32 input tensor the same way
        // Qwen35::forward does (to_vec_f32 + map).
        let ids_u32: Vec<u32> = input_ids.to_vec_f32()
            .unwrap_or_else(|_| vec![0.0])
            .into_iter()
            .map(|x| x as u32)
            .collect();
        let mut h = self
            .tok_embeddings
            .forward(&ids_u32, seq_len, self.cfg.hidden_size)
            .map_err(|e| grim_core::Error::Tensor(e))?;

        if session.model_state().is_none() {
            let fresh: Vec<Qwen35LayerCache> = (0..self.blocks.len())
                .map(|_| Qwen35LayerCache::new(&self.cfg))
                .collect();
            session.set_model_state(Box::new(fresh));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Qwen35LayerCache>>())
            .expect("TinyQwen forward: session state must be Vec<Qwen35LayerCache>");

        let positions: Vec<u32> = match positions.dtype() {
            d if d == DType::F32 => positions
                .to_vec_f32()
                .unwrap_or_else(|_| vec![0.0])
                .into_iter()
                .map(|x| x as u32)
                .collect(),
            _ => vec![0; seq_len],
        };

        for (i, block) in self.blocks.iter().enumerate() {
            h = block
                .forward(&h, &positions, &mut caches[i])?;
            caches[i].current_pos += positions.len();
        }
        let normed = self
            .output_norm
            .forward(&h)
            .map_err(|e| grim_core::Error::Tensor(e))?;
        let logits = self
            .output
            .forward(&normed)
            .map_err(|e| grim_core::Error::Tensor(e))?;
        session.advance_pos(positions.len());
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// Helper: forward TinyQwen, return output logits as Vec<f32>.
// ---------------------------------------------------------------------------

fn forward_tiny(
    model: &TinyQwen,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Vec<f32>, grim_tensor::Error> {
    let seq_len = input_ids.len();
    let ids = cpu_tensor(
        input_ids.iter().map(|&x| x as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let pos = cpu_tensor(
        positions.iter().map(|&x| x as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let mut sess = model.new_session();
    let logits = model
        .forward(&mut *sess, &ids, &pos, &[])
        .map_err(|e| grim_tensor::Error::Backend(e.to_string()))?;
    Ok(logits.to_vec_f32()?)
}

// ---------------------------------------------------------------------------
// Phase 0 tests
// ---------------------------------------------------------------------------

#[test]
fn phase0_cpu_parity_baseline() {
    let model = TinyQwen::build_cpu().expect("build CPU model");
    let ids = [0, 1, 2, 3];
    let pos = [0, 1, 2, 3];
    let out = forward_tiny(&model, &ids, &pos).expect("forward");
    assert_eq!(out.len(), 4 * TINY_CFG.vocab_size);
    assert!(
        !out.iter().all(|x| x.is_nan()),
        "CPU forward produced NaN — broken weights"
    );
}

#[test]
fn phase0_structural_host_roundtrip_in_forward_body() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qwen35.rs"),
    )
    .expect("qwen35.rs readable");

    // Find Qwen35Block::forward specifically: locate "impl Qwen35Block",
    // then the first "pub fn forward(" after it.
    let impl_pos = src
        .find("impl Qwen35Block")
        .expect("Qwen35Block impl not found");
    let fn_pos = src[impl_pos..]
        .find("pub fn forward(")
        .expect("Qwen35Block::forward not found")
        + impl_pos;

    // Brace-match from the opening { of the function signature to extract
    // exactly the function body.
    let brace = src[fn_pos..]
        .find('{')
        .expect("Qwen35Block::forward has no opening brace");
    let mut depth = 0usize;
    let mut end = fn_pos + brace;
    loop {
        end += 1;
        match src.as_bytes().get(end) {
            Some(b'{') => depth += 1,
            Some(b'}') => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            Some(_) => {}
            None => panic!("unterminated Qwen35Block::forward body"),
        }
    }
    let body = &src[fn_pos + brace + 1..end];

    // Phase 1: no local silu_mul call in forward body
    assert!(
        !body.contains("silu_mul("),
        "Qwen35Block::forward still calls local silu_mul — should use silu_mul_on_device"
    );

    // Phase 2: no add_tensors calls in forward body
    assert!(
        !body.contains("add_tensors("),
        "Qwen35Block::forward still calls add_tensors — should use add_on_device"
    );

    // Phase 3: no host KV cache extend (re-upload every step)
    assert!(
        !body.contains("cache.k_cache.extend_from_slice"),
        "Qwen35Block::forward still extends host k_cache — should use device arena"
    );
    assert!(
        !body.contains("cache.v_cache.extend_from_slice"),
        "Qwen35Block::forward still extends host v_cache — should use device arena"
    );

    // Phase 3: no host-history attention overload
    assert!(
        !body.contains("fused_or_scalar_attention("),
        "Qwen35Block::forward still calls fused_or_scalar_attention — should use arena variant"
    );
}

#[test]
#[ignore = "requires ROCm device; enable when GPU available"]
fn phase0_device_parity_target() {
    let cpu_model = TinyQwen::build_cpu().expect("build CPU");
    let gpu_model = TinyQwen::build_cpu().expect("build GPU");
    let ids = [0, 1, 2, 3];
    let pos = [0, 1, 2, 3];
    let cpu_out = forward_tiny(&cpu_model, &ids, &pos).expect("CPU forward");
    let gpu_out = forward_tiny(&gpu_model, &ids, &pos).expect("GPU forward");
    assert_eq!(cpu_out.len(), gpu_out.len());
    for (i, (&a, &b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        let rel = (a - b).abs() / (a.abs().max(b.abs()).max(1e-6));
        assert!(
            rel < 1e-3,
            "device parity mismatch at output[{i}]: cpu={a} gpu={b} rel={rel}"
        );
    }
}

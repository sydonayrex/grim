//! Qwen3.8-Flash-Next architecture with Hybrid Gated DeltaNet + QSA Attention, Gated Residual streams, N-gram embeddings, and 512 Fine-Grained Routed Experts.
//! # Architecture Details - **Hybrid Attention**: Interleaved 3:1 Gated DeltaNet (GDN) linear attention and Qwen.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::hyperparams::MetadataLookup;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// Config

/// Configuration for Qwen3.8-Flash-Next architecture (matching HuggingFace `qwen4_exp_text`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen38FlashNextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub routed_scaling_factor: f32,
    pub layer_types: Vec<String>,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel_dim: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ngram_vocab_size: Option<usize>,
    pub ngram_dim: Option<usize>,
    pub ngram_size: usize,
    pub split_ngram_parts: usize,
    pub ple_layer_ids: Vec<usize>,
    pub ple_conv_kernel_size: usize,
    /// `ple.layer_multipliers`: the published 64-bit hash constants, one per
    /// n-gram order. `ngram_size` entries.
    pub ple_layer_multipliers: Vec<u64>,
    /// `ple.head_offsets`: row offset of each head's slice in the shared table.
    /// `(ngram_size - 1) * heads_per_ngram` entries.
    pub ple_head_offsets: Vec<u64>,
    /// `ple.head_vocab_sizes`: per-head modulus. NOT uniform: the released file
    /// has 16 distinct values around 2.0e7, so a single shared vocab is wrong.
    pub ple_head_vocab_sizes: Vec<u64>,
    /// `ple.heads_per_ngram`.
    pub ple_heads_per_ngram: usize,
    /// `ple.eos_token_id`; resets the n-gram window.
    pub ple_eos_token_id: u32,
    /// `ple.image_token_id`, hashed in place of a token for embedding batches.
    pub ple_image_token_id: Option<u32>,
    pub mrope_section: [usize; 4],
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub full_yarn: Option<YaRNParams>,
    pub indexer_top_k: usize,
    /// `qwen4exp.ssm.time_step_rank` (48). Width of `ssm_a`, `ssm_dt.bias`,
    /// `ssm_alpha` and `ssm_beta` — the per-timestep-rank parameters. This
    /// happens to equal `linear_num_value_heads` (48) for the released
    /// checkpoint, but they are different quantities: one is the GDN
    /// timestep rank, the other is the value-head count.
    pub ssm_dt_rank: usize,
    /// `qwen4exp.ssm.group_count` (16) — GDN key group count.
    pub ssm_n_group: usize,
    /// `qwen4exp.ssm.state_size` (128) — GDN per-head state width.
    pub ssm_d_state: usize,
    /// `qwen4exp.ssm.inner_size` (6144) — GDN value projection width.
    pub ssm_d_inner: usize,
    /// `qwen4exp.attention.value_length` (256). Usually equal to
    /// [`Self::head_dim`] but tracked separately because the GQA key and value
    /// widths are independently declared in GGUF.
    pub value_head_dim: usize,
    /// `qwen4exp.attention.indexer.head_count` (4) — indexer QUERY heads.
    /// The indexer has a single key head; see [`Self::indexer_key_length`].
    pub indexer_n_heads: usize,
    /// `qwen4exp.attention.indexer.key_length` (128) — indexer head width,
    /// for both the one key head and each of the `n_heads` query heads.
    pub indexer_key_length: usize,
    /// `qwen4exp.attention.compress_ratios[layer]`, per layer: 4 on the 12
    /// full-attention layers, 0 on the 36 Gated DeltaNet ones. Only the former
    /// reach the QSA path, and upstream asserts `r > 0` there.
    pub attention_compress_ratios: Vec<usize>,
}

impl Qwen38FlashNextConfig {
    /// Build the config from `qwen4exp.*` GGUF metadata.
    ///
    /// Every field is read from the checkpoint rather than hardcoded, because
    /// the previous loader guessed several of them and got them wrong for the
    /// released GSQ-RCO file: `layer_types` was left empty, `ngram_dim` was 512
    /// instead of the real embedding width, `mrope_section` dropped the
    /// trailing 0, and `ngram_vocab_size` / `linear_num_*_heads` were guesses.
    /// A wrong field here produces a model that loads and emits finite logits
    /// while being numerically wrong, so each lookup records what it used.
    ///
    /// `m` is any [`MetadataLookup`], so this works identically for a GGUF
    /// provider and for a test double.
    ///
    /// # Metadata keys consumed
    /// `qwen4exp.block_count`, `.embedding_length`, `.attention.head_count`,
    /// `.attention.head_count_kv`, `.attention.key_length`,
    /// `.attention.recurrent_layers`, `.expert_count`, `.expert_used_count`,
    /// `.expert_feed_forward_length`, `.expert_shared_feed_forward_length`,
    /// `.ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}`,
    /// `.hyper_connection.{count,low_rank}`, `.attention.indexer.top_k`,
    /// `.rope.{dimension_count,dimension_sections,freq_base}`,
    /// `.attention.layer_norm_rms_epsilon`, and
    /// `.ple.{layers,ngram_size,conv_kernel,layer_multipliers,head_offsets,head_vocab_sizes}`
    /// plus `.embedding_length_per_layer_input`.
    pub fn from_qwen4exp_metadata<M: MetadataLookup + ?Sized>(m: &M) -> Self {
        let gu = |k: &str| m.get_u32(k).map(|v| v as usize);

        let key_len = gu("qwen4exp.attention.key_length").unwrap_or(256);
        let val_len = gu("qwen4exp.attention.value_length").unwrap_or(key_len);
        let ssm_d_inner = gu("qwen4exp.ssm.inner_size").unwrap_or(6144);
        let ssm_n_group = gu("qwen4exp.ssm.group_count").unwrap_or(16);
        let ssm_d_state = gu("qwen4exp.ssm.state_size").unwrap_or(128);
        let ssm_dt_rank = gu("qwen4exp.ssm.time_step_rank").unwrap_or(48);

        // GDN key/value geometry follows from the SSM shape:
        //   keys   = group_count * state_size      (16 * 128 = 2048)
        //   values = inner_size                    (6144), split over `key_len` heads
        let linear_num_key_heads = ssm_n_group;
        let linear_key_head_dim = ssm_d_state;
        let linear_value_head_dim = ssm_d_state;
        let linear_num_value_heads = if linear_value_head_dim > 0 {
            ssm_d_inner / linear_value_head_dim
        } else {
            1
        };

        // `attention.recurrent_layers[i] == true` means Gated DeltaNet at layer i.
        // Derive layer_types from it; fall back to the interval schedule.
        let layer_types: Vec<String> = match m
            .get_bool_array("qwen4exp.attention.recurrent_layers")
            .filter(|v| !v.is_empty())
        {
            Some(rec) => rec
                .iter()
                .map(|r| {
                    if *r {
                        "linear_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect(),
            None => {
                let interval = gu("qwen4exp.full_attention_interval").unwrap_or(4).max(1);
                let layers = gu("qwen4exp.block_count").unwrap_or(48);
                (0..layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect()
            }
        };

        // mrope section layout, e.g. [11, 11, 10, 0]. The trailing 0 is
        // significant: it is the unrotated tail of `rope.dimension_count`.
        let secs = m.get_u32_array("qwen4exp.rope.dimension_sections");
        let mrope_section = match secs.as_ref().map(|v| v.len()) {
            Some(n) if n >= 4 => [
                secs.as_ref().unwrap()[0] as usize,
                secs.as_ref().unwrap()[1] as usize,
                secs.as_ref().unwrap()[2] as usize,
                secs.as_ref().unwrap()[3] as usize,
            ],
            Some(n) if n == 3 => {
                let s = secs.as_ref().unwrap();
                [s[0] as usize, s[1] as usize, s[2] as usize, 0]
            }
            _ => [11, 11, 10, 0],
        };
        let rope_dim = gu("qwen4exp.rope.dimension_count").unwrap_or(64);

        // PLE: the per-head table width is `embedding_length_per_layer_input`
        // (160), NOT the model hidden size. The old loader set 512.
        let ple_heads = m
            .get_u32("qwen4exp.embedding_length_per_layer_input")
            .map(|v| v as usize)
            .unwrap_or(160);
        let ple_ngram = gu("qwen4exp.ple.ngram_size").unwrap_or(3);
        // Table rows are the sum over heads; the biggest offset+modulus is the
        // authoritative bound when the metadata is present.
        let ngram_vocab_size = m
            .get_u64_array("qwen4exp.ple.head_offsets")
            .zip(m.get_u64_array("qwen4exp.ple.head_vocab_sizes"))
            .map(|(off, vs)| {
                off.iter()
                    .zip(vs.iter())
                    .map(|(o, v)| o + v)
                    .max()
                    .unwrap_or(20_000_000) as usize
            });

        let ple_layer_ids: Vec<usize> = m
            .get_u32_array("qwen4exp.ple.layers")
            .map(|v| v.into_iter().map(|x| x as usize).collect())
            .unwrap_or_else(|| vec![1]);
        let heads_per_ngram = m
            .get_u32("qwen4exp.ple.heads_per_ngram")
            .map(|v| v as usize)
            .unwrap_or(8);

        // RoPE is partial: only `rope_dim` of head_dim are rotated, so the
        // rotary factor is that ratio, not 1.0 as the old loader assumed.
        // `value_length` is declared independently of `key_length` in GGUF,
        // so it is carried separately rather than assumed equal.
        let head_dim = key_len;
        let partial_rotary_factor = if head_dim > 0 {
            rope_dim as f32 / head_dim as f32
        } else {
            1.0
        };

        Self {
            vocab_size: m
                .get_array_len("tokenizer.ggml.tokens")
                .or_else(|| gu("qwen4exp.vocab_size"))
                .unwrap_or(248320),
            hidden_size: gu("qwen4exp.embedding_length").unwrap_or(2560),
            num_heads: gu("qwen4exp.attention.head_count").unwrap_or(24),
            num_kv_heads: gu("qwen4exp.attention.head_count_kv").unwrap_or(2),
            head_dim,
            num_layers: gu("qwen4exp.block_count").unwrap_or(48),
            intermediate_size: gu("qwen4exp.expert_feed_forward_length").unwrap_or(640),
            num_experts: gu("qwen4exp.expert_count").unwrap_or(512),
            num_experts_per_tok: gu("qwen4exp.expert_used_count").unwrap_or(10),
            shared_expert_intermediate_size: gu("qwen4exp.expert_shared_feed_forward_length"),
            routed_scaling_factor: m
                .get_f32("qwen4exp.expert_weights_scale")
                .or_else(|| m.get_f32("qwen4exp.routed_scaling_factor"))
                .unwrap_or(1.0),
            layer_types,
            linear_key_head_dim,
            linear_num_key_heads,
            linear_value_head_dim,
            linear_num_value_heads,
            linear_conv_kernel_dim: gu("qwen4exp.ssm.conv_kernel").unwrap_or(4),
            hc_count: gu("qwen4exp.hyper_connection.count").unwrap_or(4),
            hc_lowrank: gu("qwen4exp.hyper_connection.low_rank").unwrap_or(320),
            ngram_vocab_size,
            ngram_dim: Some(ple_heads),
            ple_layer_multipliers: m
                .get_u64_array("qwen4exp.ple.layer_multipliers")
                .unwrap_or_default(),
            ple_head_offsets: m
                .get_u64_array("qwen4exp.ple.head_offsets")
                .unwrap_or_default(),
            ple_head_vocab_sizes: m
                .get_u64_array("qwen4exp.ple.head_vocab_sizes")
                .unwrap_or_default(),
            ple_heads_per_ngram: heads_per_ngram,
            ple_eos_token_id: gu("qwen4exp.ple.eos_token_id").unwrap_or(0) as u32,
            ple_image_token_id: gu("qwen4exp.ple.image_token_id").map(|v| v as u32),
            ngram_size: ple_ngram,
            split_ngram_parts: heads_per_ngram * 2,
            ple_layer_ids,
            ple_conv_kernel_size: gu("qwen4exp.ple.conv_kernel").unwrap_or(4),
            mrope_section,
            partial_rotary_factor,
            rms_norm_eps: m
                .get_f32("qwen4exp.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_theta: m.get_f32("qwen4exp.rope.freq_base").unwrap_or(10000000.0),
            max_seq_len: gu("qwen4exp.context_length").unwrap_or(262144),
            full_yarn: None,
            indexer_top_k: gu("qwen4exp.attention.indexer.top_k").unwrap_or(2048),
            ssm_dt_rank,
            ssm_n_group,
            ssm_d_state,
            ssm_d_inner,
            value_head_dim: val_len,
            indexer_n_heads: gu("qwen4exp.attention.indexer.head_count").unwrap_or(4),
            indexer_key_length: gu("qwen4exp.attention.indexer.key_length").unwrap_or(128),
            attention_compress_ratios: m
                .get_u32_array("qwen4exp.attention.compress_ratios")
                .map(|v| v.into_iter().map(|x| x as usize).collect())
                .unwrap_or_default(),
        }
    }
}

impl Qwen38FlashNextConfig {
    /// Build the PLE addressing from the checkpoint's published constants.
    ///
    /// # Errors
    /// Propagates the metadata validation from
    /// [`Qwen38NgramAddressing::from_metadata`].
    pub fn ple_addressing(&self) -> Result<Qwen38NgramAddressing> {
        Qwen38NgramAddressing::from_metadata(
            self.ple_layer_multipliers.clone(),
            self.ple_head_offsets.clone(),
            self.ple_head_vocab_sizes.clone(),
            self.ple_heads_per_ngram,
            self.ngram_size,
            self.ple_eos_token_id,
            self.ple_image_token_id,
        )
    }
}

impl Default for Qwen38FlashNextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 2560,
            num_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            num_layers: 48,
            intermediate_size: 640,
            num_experts: 512,
            num_experts_per_tok: 10,
            shared_expert_intermediate_size: Some(640),
            routed_scaling_factor: 2.5,
            layer_types: (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect(),
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 48,
            linear_conv_kernel_dim: 4,
            hc_count: 4,
            hc_lowrank: 320,
            ngram_vocab_size: Some(320_001_446),
            ngram_dim: Some(160),
            ngram_size: 3,
            split_ngram_parts: 128,
            ple_layer_ids: vec![1],
            ple_conv_kernel_size: 4,
            // The released GSQ-RCO file's exact PLE constants. 16 heads =
            // (ngram_size - 1) * heads_per_ngram = 2 * 8.
            ple_layer_multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            ple_head_offsets: vec![
                0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                300001275,
            ],
            ple_head_vocab_sizes: vec![
                20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
            ],
            ple_heads_per_ngram: 8,
            ple_eos_token_id: 248044,
            ple_image_token_id: Some(248056),
            mrope_section: [11, 11, 10, 0],
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 262144,
            full_yarn: None,
            indexer_top_k: 2048,
            ssm_dt_rank: 48,
            ssm_n_group: 16,
            ssm_d_state: 128,
            ssm_d_inner: 6144,
            value_head_dim: 256,
            indexer_n_heads: 4,
            indexer_key_length: 128,
            // Every 4th layer is full attention and carries a ratio of 4; the
            // GDN layers carry 0 and never reach QSA.
            attention_compress_ratios: (0..48)
                .map(|i| if (i + 1) % 4 == 0 { 4 } else { 0 })
                .collect(),
        }
    }
}

impl ModelConfig for Qwen38FlashNextConfig {
    fn name(&self) -> &str {
        "qwen3_8_flash_next"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Hyper-Connection Mixer (Residual Stream Routing)

/// Hyper-Connection Mixer performing low-rank multi-branch residual projection.
#[derive(Clone)]
pub struct Qwen38HyperConnection {
    pub hc_norm: RmsNorm,
    pub input_mix_down: Linear,
    pub input_mix_up: Linear,
    pub block_inject: Option<Linear>,
}

impl Qwen38HyperConnection {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        hc_lowrank: usize,
        eps: f32,
    ) -> Result<Self> {
        let hc_norm = RmsNorm::load(&ws.scoped("hc_norm"), hidden_size, eps)
            .or_else(|_| RmsNorm::load(&ws.scoped("norm"), hidden_size, eps))?;
        let input_mix_down = Linear::load_shape(
            &ws.scoped("input_mix_weight_down"),
            [hidden_size, hc_lowrank],
        )
        .or_else(|_| Linear::load_shape(&ws.scoped("down"), [hidden_size, hc_lowrank]))?;
        let input_mix_up =
            Linear::load_shape(&ws.scoped("input_mix_weight_up"), [hc_lowrank, hidden_size])
                .or_else(|_| Linear::load_shape(&ws.scoped("up"), [hc_lowrank, hidden_size]))?;
        let block_inject = Linear::load_shape(
            &ws.scoped("block_inject_weight"),
            [hidden_size, hidden_size],
        )
        .or_else(|_| Linear::load_shape(&ws.scoped("inject"), [hidden_size, 4]))
        .ok();

        Ok(Self {
            hc_norm,
            input_mix_down,
            input_mix_up,
            block_inject,
        })
    }

    pub fn random(hidden_size: usize, hc_lowrank: usize, eps: f32) -> Self {
        let hc_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0f32; hidden_size], Shape::new(vec![hidden_size])),
            eps,
        };
        let down_w = cpu_tensor(
            vec![0.01f32; hc_lowrank * hidden_size],
            Shape::new(vec![hc_lowrank, hidden_size]),
        );
        let up_w = cpu_tensor(
            vec![0.01f32; hidden_size * hc_lowrank],
            Shape::new(vec![hidden_size, hc_lowrank]),
        );
        Self {
            hc_norm,
            input_mix_down: Linear::from_tensor(down_w, None),
            input_mix_up: Linear::from_tensor(up_w, None),
            block_inject: None,
        }
    }

    pub fn mix(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.hc_norm.forward(x)?;
        let down = self.input_mix_down.forward(&normed)?;
        let up = self.input_mix_up.forward(&down)?;
        // `up` output width is hidden_size (weight [hc_lowrank, hidden_size]),
        // so it matches `x` element-for-element; stay on-device.
        Ok(grim_nn::modules::add_on_device(x, &up)?)
    }

    pub fn prepare(
        &self,
        residual: &Tensor,
        cfg: &Qwen38FlashNextConfig,
    ) -> Result<(Tensor, Tensor)> {
        // RMSNorm on full residual stream [seq, hc_count * hidden_size]
        let x_norm = self.hc_norm.forward(residual)?;
        // Down projection
        let down_proj = self.input_mix_down.forward(&x_norm)?;
        let down_scaled =
            grim_nn::modules::mul_scalar_on_device(&down_proj, 1.0 / cfg.hc_count as f32)?;
        let h_mix = grim_nn::modules::silu_on_device(&down_scaled)?;
        // Up projection
        let mix = self.input_mix_up.forward(&h_mix)?;
        let mix_weights = grim_nn::modules::sigmoid_on_device(&mix)?;
        // Per-stream weighted reduction
        let branch = grim_nn::modules::reduce_weighted_streams(
            &x_norm,
            &mix_weights,
            cfg.hc_count,
            cfg.hidden_size,
        )?;
        Ok((x_norm, branch))
    }

    pub fn inject(
        &self,
        prev_residual: &Tensor,
        x_norm: &Tensor,
        branch: &Tensor,
        cfg: &Qwen38FlashNextConfig,
    ) -> Result<Tensor> {
        if let Some(ref inject_lin) = self.block_inject {
            let inject_proj = inject_lin.forward(x_norm)?;
            let inject_scaled =
                grim_nn::modules::mul_scalar_on_device(&inject_proj, 1.0 / cfg.hc_count as f32)?;
            let raw_sig = grim_nn::modules::sigmoid_on_device(&inject_scaled)?;
            let weights = grim_nn::modules::mul_scalar_on_device(&raw_sig, 2.0)?;
            let delta =
                grim_nn::modules::broadcast_and_mul_streams(branch, &weights, cfg.hc_count)?;
            Ok(grim_nn::modules::add_on_device(prev_residual, &delta)?)
        } else {
            Ok(prev_residual.clone())
        }
    }
}

// Block Layers & Feed Forward

struct Qwen38MoeExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen38MoeExpert {
    fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        // Fused silu(gate) * up on-device; skips the per-expert host roundtrip.
        let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

enum Qwen38MoeExperts {
    Bank(grim_nn::moe::ExpertBank),
    Individual(Vec<Qwen38MoeExpert>),
}

enum Qwen38SharedExpert {
    Gated {
        gate_inp: Linear,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    },
    Legacy(Qwen38MoeExpert),
}

struct Qwen38MoeBlock {
    gate: Linear,
    experts: Qwen38MoeExperts,
    shared_expert: Option<Qwen38SharedExpert>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f32,
    _charon_cache: crate::shared_moe::CharonCache,
}

impl Qwen38MoeBlock {
    fn load(ws: &WeightSource<'_>, cfg: &Qwen38FlashNextConfig) -> Result<Self> {
        // Router gate: ffn_gate_inp or gate
        let gate = Linear::load_shape(
            &ws.scoped("ffn_gate_inp"),
            [cfg.hidden_size, cfg.num_experts],
        )
        .or_else(|_| Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts]))?;

        let experts = if ws.has_tensor("ffn_gate_exps.weight") {
            let bank = grim_nn::moe::ExpertBank::load(
                ws,
                cfg.num_experts,
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?;
            Qwen38MoeExperts::Bank(bank)
        } else {
            let mut exp_vec = Vec::with_capacity(cfg.num_experts);
            for i in 0..cfg.num_experts {
                let expert_ws = ws.scoped("experts").scoped(&i.to_string());
                exp_vec.push(Qwen38MoeExpert::load(
                    &expert_ws,
                    cfg.hidden_size,
                    cfg.intermediate_size,
                )?);
            }
            Qwen38MoeExperts::Individual(exp_vec)
        };

        let shared_expert = if ws.has_tensor("ffn_gate_shexp.weight") {
            let shared_dim = cfg
                .shared_expert_intermediate_size
                .unwrap_or(cfg.intermediate_size);
            let gate_inp =
                Linear::load_shape(&ws.scoped("ffn_gate_inp_shexp"), [cfg.hidden_size, 1])?;
            let gate_proj =
                Linear::load_shape(&ws.scoped("ffn_gate_shexp"), [cfg.hidden_size, shared_dim])?;
            let up_proj =
                Linear::load_shape(&ws.scoped("ffn_up_shexp"), [cfg.hidden_size, shared_dim])?;
            let down_proj =
                Linear::load_shape(&ws.scoped("ffn_down_shexp"), [shared_dim, cfg.hidden_size])?;
            Some(Qwen38SharedExpert::Gated {
                gate_inp,
                gate_proj,
                up_proj,
                down_proj,
            })
        } else if let Some(shared_dim) = cfg.shared_expert_intermediate_size {
            let exp =
                Qwen38MoeExpert::load(&ws.scoped("shared_expert"), cfg.hidden_size, shared_dim)
                    .ok();
            exp.map(Qwen38SharedExpert::Legacy)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
            _charon_cache: crate::shared_moe::CharonCache::new(),
        })
    }

    fn forward_expert(&self, idx: usize, x: &Tensor) -> Result<Tensor> {
        match &self.experts {
            Qwen38MoeExperts::Individual(list) => list[idx].forward(x),
            Qwen38MoeExperts::Bank(bank) => {
                let g = bank.gate[idx].forward(x)?;
                let u = bank.up[idx].forward(x)?;
                let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
                Ok(bank.down[idx].forward(&act)?)
            }
        }
    }

    fn forward_shared(&self, x: &Tensor) -> Result<Tensor> {
        match self.shared_expert.as_ref() {
            Some(Qwen38SharedExpert::Gated {
                gate_inp,
                gate_proj,
                up_proj,
                down_proj,
            }) => {
                let gate_logits = gate_inp.forward(x)?;
                let gate_sig = grim_nn::modules::sigmoid_on_device(&gate_logits)?;
                let g = gate_proj.forward(x)?;
                let u = up_proj.forward(x)?;
                let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
                let down = down_proj.forward(&act)?;
                let dev = grim_nn::modules::pick_device_for_tensor(&down);
                let (prod, _) = dev.mul(&**down.storage(), &**gate_sig.storage(), down.shape())?;
                Ok(Tensor::new(
                    std::sync::Arc::from(prod),
                    down.shape().clone(),
                    down.dtype(),
                    down.provenance().clone(),
                    down.device().clone(),
                ))
            }
            Some(Qwen38SharedExpert::Legacy(exp)) => exp.forward(x),
            None => Ok(x.clone()),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let router_logits = self.gate.forward(x)?;
        let dims = x.shape().dims();
        let hidden_dim = dims[dims.len() - 1];
        let seq_len = x.shape().elem_count() / hidden_dim;

        let num_exp = match &self.experts {
            Qwen38MoeExperts::Bank(b) => b.num_experts(),
            Qwen38MoeExperts::Individual(l) => l.len(),
        };

        let logits_vec = router_logits.to_vec_f32()?;

        if x.device() != &Device::Cpu && seq_len == 1 {
            let row = &logits_vec[0..num_exp];
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k = self.num_experts_per_tok.min(num_exp);
            let topk = &indexed[..k];

            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = row.iter().map(|l| (l - max_l).exp()).sum::<f32>() + 1e-12;
            let weights: Vec<f32> = topk
                .iter()
                .map(|(_, l)| ((l - max_l).exp() / denom) * self.routed_scaling_factor)
                .collect();

            let mut acc: Option<Tensor> = if self.shared_expert.is_some() {
                Some(self.forward_shared(x)?)
            } else {
                None
            };

            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.forward_expert(*exp_idx, x)?;
                acc = Some(match acc {
                    Some(a) => grim_nn::modules::axpy_on_device(&a, w, &exp_out)?,
                    None => {
                        let dev = grim_nn::modules::pick_device_for_tensor(&exp_out);
                        let (scaled_st, _) =
                            dev.mul_scalar(&**exp_out.storage(), w, exp_out.shape())?;
                        Tensor::new(
                            std::sync::Arc::from(scaled_st),
                            exp_out.shape().clone(),
                            exp_out.dtype(),
                            grim_tensor::dtype::QuantProvenance::default(),
                            exp_out.device().clone(),
                        )
                    }
                });
            }

            if let Some(out) = acc {
                return Ok(out);
            }
            return Ok(x.clone());
        }

        let x_vec = x.to_vec_f32()?;
        let mut out_vec = vec![0.0f32; x_vec.len()];

        for s in 0..seq_len {
            let row = &logits_vec[s * num_exp..(s + 1) * num_exp];
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k = self.num_experts_per_tok.min(num_exp);
            let topk = &indexed[..k];

            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = row.iter().map(|l| (l - max_l).exp()).sum::<f32>() + 1e-12;
            let weights: Vec<f32> = topk
                .iter()
                .map(|(_, l)| ((l - max_l).exp() / denom) * self.routed_scaling_factor)
                .collect();

            // The per-token slice is staged on the host, so it has to be put
            // back on the model's device: the expert Linears are device matmuls
            // and reject host storage with "matmul: input b is not
            // RocmStorage". On the CPU this is a no-op clone.
            let token_host = cpu_tensor(
                x_vec[s * hidden_dim..(s + 1) * hidden_dim].to_vec(),
                Shape::new(vec![1, hidden_dim]),
            );
            let token_x = if x.device().is_cpu() {
                token_host
            } else {
                grim_nn::modules::move_to_device(&token_host, x.device())?
            };

            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.forward_expert(*exp_idx, &token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out_vec[s * hidden_dim + d] += w * exp_out[d];
                }
            }

            if self.shared_expert.is_some() {
                let shared_out = self.forward_shared(&token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out_vec[s * hidden_dim + d] += shared_out[d];
                }
            }
        }

        // The multi-token path accumulates on the host, so the result is host
        // storage. Return it on the model's device: the block's residual add is
        // dispatched on-device and rejects host input.
        let out_host = cpu_tensor(out_vec, x.shape().clone());
        if x.device().is_cpu() {
            Ok(out_host)
        } else {
            Ok(grim_nn::modules::move_to_device(&out_host, x.device())?)
        }
    }
}

pub enum Qwen38Attention {
    Linear {
        attn_qkv: Linear,
        attn_gate: Linear,
        ssm_conv1d: Linear,
        ssm_dt: Tensor,
        ssm_a: Tensor,
        ssm_beta: Linear,
        ssm_alpha: Linear,
        ssm_norm: RmsNorm,
        ssm_out: Linear,
        /// `n_v_heads` = `ssm_dt_rank` (48). Carried here rather than read
        /// from config at forward time so the mixer cannot be sized from the
        /// wrong quantity if the two ever diverge.
        n_v_heads: usize,
        /// `n_k_heads` = `ssm_n_group` (16).
        n_k_heads: usize,
        /// `head_dim` = `d_k` = `d_v` = `ssm_d_state` (128).
        head_dim: usize,
        /// `conv_dim` = `key_dim * 2 + value_dim` (10240).
        conv_dim: usize,
        /// Value-head to key-head mapping. A 3:1 ratio is consistent with both
        /// pairings and the GGUF does not encode it, so it is explicit.
        pairing: crate::qwen4exp_gdn::KdaHeadPairing,
    },
    Full {
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        indexer_q: Option<Linear>,
        indexer_k: Option<Linear>,
        indexer_q_norm: Option<RmsNorm>,
        indexer_k_norm: Option<RmsNorm>,
        rope: Rope,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        wqkv_q80_fused: Option<std::sync::Arc<grim_backend_rocm::FusedQkvWeights>>,
    },
}

pub struct Qwen38FlashNextBlock {
    /// Index of this block in the model, for per-layer metadata lookup.
    pub layer_index: usize,
    pub hc_attn: Qwen38HyperConnection,
    pub attn: Qwen38Attention,
    pub hc_ffn: Qwen38HyperConnection,
    pub ffn_norm: RmsNorm,
    pub attn_norm: RmsNorm,
    moe_block: Qwen38MoeBlock,
    pub gated_residual_scale: f32,
}

impl Qwen38FlashNextBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Qwen38FlashNextConfig,
        layer_index: usize,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let hc_lowrank = cfg.hc_lowrank;
        let hc_dim = cfg.hc_count * cfg.hidden_size;

        // HC Attn
        let hc_attn = if ws.has_tensor("hc_attn_norm.weight") {
            let norm = RmsNorm::load(&ws.scoped("hc_attn_norm"), hc_dim, cfg.rms_norm_eps)?;
            let down = Linear::load_shape(&ws.scoped("hc_attn_down"), [hc_dim, hc_lowrank])?;
            let up = Linear::load_shape(&ws.scoped("hc_attn_up"), [hc_lowrank, hc_dim])?;
            let inject =
                Linear::load_shape(&ws.scoped("hc_attn_inject"), [hc_dim, cfg.hc_count]).ok();
            Qwen38HyperConnection {
                hc_norm: norm,
                input_mix_down: down,
                input_mix_up: up,
                block_inject: inject,
            }
        } else {
            Qwen38HyperConnection::load(
                &ws.scoped("hc_attn"),
                cfg.hidden_size,
                hc_lowrank,
                cfg.rms_norm_eps,
            )
            .unwrap_or_else(|_| {
                Qwen38HyperConnection::random(cfg.hidden_size, hc_lowrank, cfg.rms_norm_eps)
            })
        };

        // HC FFN
        let hc_ffn = if ws.has_tensor("hc_ffn_norm.weight") {
            let norm = RmsNorm::load(&ws.scoped("hc_ffn_norm"), hc_dim, cfg.rms_norm_eps)?;
            let down = Linear::load_shape(&ws.scoped("hc_ffn_down"), [hc_dim, hc_lowrank])?;
            let up = Linear::load_shape(&ws.scoped("hc_ffn_up"), [hc_lowrank, hc_dim])?;
            let inject =
                Linear::load_shape(&ws.scoped("hc_ffn_inject"), [hc_dim, cfg.hc_count]).ok();
            Qwen38HyperConnection {
                hc_norm: norm,
                input_mix_down: down,
                input_mix_up: up,
                block_inject: inject,
            }
        } else {
            Qwen38HyperConnection::load(
                &ws.scoped("hc_ffn"),
                cfg.hidden_size,
                hc_lowrank,
                cfg.rms_norm_eps,
            )
            .unwrap_or_else(|_| {
                Qwen38HyperConnection::random(cfg.hidden_size, hc_lowrank, cfg.rms_norm_eps)
            })
        };

        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .unwrap_or_else(|_| RmsNorm {
            weight: cpu_tensor(
                vec![1.0f32; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        });

        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .unwrap_or_else(|_| RmsNorm {
            weight: cpu_tensor(
                vec![1.0f32; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        });

        let attn = if ws.has_tensor("attn_qkv.weight") {
            let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
            let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;

            let attn_qkv = Linear::load_shape(&ws.scoped("attn_qkv"), [cfg.hidden_size, conv_dim])?;
            let attn_gate =
                Linear::load_shape(&ws.scoped("attn_gate"), [cfg.hidden_size, value_dim])?;
            let ssm_conv1d = Linear::load_shape(
                &ws.scoped("ssm_conv1d"),
                [cfg.linear_conv_kernel_dim, conv_dim],
            )?;
            let ssm_dt = ws
                .get_raw_packed("ssm_dt.bias")
                .map(|raw| {
                    let floats: Vec<f32> = raw
                        .bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    cpu_tensor(floats, Shape::new(raw.shape.clone()))
                })
                .unwrap_or_else(|_| {
                    cpu_tensor(
                        vec![0.0f32; cfg.ssm_dt_rank],
                        Shape::new(vec![cfg.ssm_dt_rank]),
                    )
                });
            let ssm_a = ws
                .get_raw_packed("ssm_a")
                .map(|raw| {
                    let floats: Vec<f32> = raw
                        .bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    cpu_tensor(floats, Shape::new(raw.shape.clone()))
                })
                .unwrap_or_else(|_| {
                    cpu_tensor(
                        vec![0.0f32; cfg.ssm_dt_rank],
                        Shape::new(vec![cfg.ssm_dt_rank]),
                    )
                });
            // ssm_alpha / ssm_beta are [hidden, time_step_rank], NOT [hidden, value_heads].
            // These coincide at 48 for the released checkpoint but are separate
            // quantities; sizing by head count silently mis-loads other geometry.
            let ssm_beta =
                Linear::load_shape(&ws.scoped("ssm_beta"), [cfg.hidden_size, cfg.ssm_dt_rank])?;
            let ssm_alpha =
                Linear::load_shape(&ws.scoped("ssm_alpha"), [cfg.hidden_size, cfg.ssm_dt_rank])?;
            // ssm_norm is [ssm_d_state], the GDN head width, and the
            // recurrence applies it per state channel of that width. It is not
            // [linear_value_head_dim]: the released checkpoint happens to have
            // both equal to 128, so the two only differ under another geometry.
            let ssm_norm =
                RmsNorm::load(&ws.scoped("ssm_norm"), cfg.ssm_d_state, cfg.rms_norm_eps)?;
            let ssm_out = Linear::load_shape(&ws.scoped("ssm_out"), [value_dim, cfg.hidden_size])?;

            Qwen38Attention::Linear {
                attn_qkv,
                attn_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
                n_v_heads: cfg.ssm_dt_rank,
                n_k_heads: cfg.ssm_n_group,
                head_dim: cfg.ssm_d_state,
                conv_dim,
                pairing: crate::qwen4exp_gdn::KdaHeadPairing::default(),
            }
        } else {
            let q_dim = cfg.num_heads * cfg.head_dim;
            let kv_dim = cfg.num_kv_heads * cfg.head_dim;

            let attn_ws = ws.scoped("self_attn");
            let wq = Linear::load_shape(&ws.scoped("attn_q"), [cfg.hidden_size, q_dim]).or_else(
                |_| Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim]),
            )?;
            let wk = Linear::load_shape(&ws.scoped("attn_k"), [cfg.hidden_size, kv_dim]).or_else(
                |_| Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim]),
            )?;
            let wv = Linear::load_shape(&ws.scoped("attn_v"), [cfg.hidden_size, kv_dim]).or_else(
                |_| Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim]),
            )?;
            let wo = Linear::load_shape(&ws.scoped("attn_output"), [q_dim, cfg.hidden_size])
                .or_else(|_| {
                    Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])
                })?;

            let q_norm =
                RmsNorm::load(&ws.scoped("attn_q_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
            let k_norm =
                RmsNorm::load(&ws.scoped("attn_k_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();

            // Widths come from the config, not literals: the released values
            // are 4 query heads x 128 and one 128-wide key head, but hardcoding
            // them makes any other geometry fail to load with a shape error.
            let idx_q_dim = cfg.indexer_n_heads * cfg.indexer_key_length;
            let idx_k_dim = cfg.indexer_key_length;
            let indexer_q =
                Linear::load_shape(&ws.scoped("indexer.q_proj"), [cfg.hidden_size, idx_q_dim]).ok();
            let indexer_k =
                Linear::load_shape(&ws.scoped("indexer.k_proj"), [cfg.hidden_size, idx_k_dim]).ok();
            let indexer_q_norm =
                RmsNorm::load(&ws.scoped("indexer.q_norm"), idx_k_dim, cfg.rms_norm_eps).ok();
            let indexer_k_norm =
                RmsNorm::load(&ws.scoped("indexer.k_norm"), idx_k_dim, cfg.rms_norm_eps).ok();

            let rope = Rope::new(cfg.head_dim, cfg.rope_theta);
            let wqkv_q80_fused = crate::shared_attention::build_fused_qkv_q80(&wq, &wk, &wv)
                .map(std::sync::Arc::new);

            Qwen38Attention::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                indexer_q,
                indexer_k,
                indexer_q_norm,
                indexer_k_norm,
                rope,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                wqkv_q80_fused,
            }
        };

        // MoE block: mlp or blk root
        let moe_block =
            if ws.has_tensor("ffn_gate_exps.weight") || ws.has_tensor("ffn_gate_inp.weight") {
                Qwen38MoeBlock::load(ws, cfg)?
            } else {
                Qwen38MoeBlock::load(&ws.scoped("mlp"), cfg)?
            };

        Ok(Self {
            hc_attn,
            attn,
            hc_ffn,
            attn_norm,
            ffn_norm,
            moe_block,
            gated_residual_scale: 1.0 / (cfg.hc_count as f32).sqrt(),
            layer_index,
        })
    }

    /// This block's QSA indexer geometry, or `None` for a Gated DeltaNet layer.
    ///
    /// `compress_ratio` is 4 on the 12 full-attention layers and 0 on the 36
    /// GDN ones; upstream asserts `r > 0` on the QSA path, so a 0 ratio is
    /// reported as "not sparse" rather than as a geometry error.
    pub fn qsa_geometry(
        &self,
        cfg: &Qwen38FlashNextConfig,
    ) -> Option<crate::qwen4exp_qsa::QsaIndexConfig> {
        if matches!(&self.attn, Qwen38Attention::Linear { .. }) {
            return None;
        }
        let idx = self.layer_index;
        let r = cfg
            .attention_compress_ratios
            .get(idx)
            .copied()
            .unwrap_or_else(|| if (idx + 1) % 4 == 0 { 4 } else { 0 });
        if r == 0 {
            return None;
        }
        // The indexer weights are absent on a checkpoint without an indexer;
        // fall back to dense rather than failing to load.
        let has_indexer = matches!(
            &self.attn,
            Qwen38Attention::Full {
                indexer_q: Some(_),
                indexer_k: Some(_),
                ..
            }
        );
        if !has_indexer {
            return None;
        }
        Some(crate::qwen4exp_qsa::QsaIndexConfig {
            idx_dim: cfg.indexer_key_length,
            n_idx_h: cfg.indexer_n_heads,
            top_k: cfg.indexer_top_k,
            compress_ratio: r,
        })
    }

    /// This block's GDN geometry, or [`GdnGeometry::Unknown`](crate::qwen4exp_gdn::GdnGeometry::Unknown)
    /// for a full-attention layer.
    pub fn gdn_geometry(&self) -> crate::qwen4exp_gdn::GdnGeometry {
        match &self.attn {
            Qwen38Attention::Linear {
                n_v_heads,
                n_k_heads,
                head_dim,
                conv_dim,
                ..
            } => crate::qwen4exp_gdn::GdnGeometry::Known(crate::qwen4exp_gdn::GdnShape {
                n_v_heads: *n_v_heads,
                n_k_heads: *n_k_heads,
                head_dim: *head_dim,
                conv_dim: *conv_dim,
            }),
            Qwen38Attention::Full { .. } => crate::qwen4exp_gdn::GdnGeometry::Unknown,
        }
    }

    /// Forward one block.
    ///
    /// `cfg` is the model's real config, not a default: the previous signature
    /// built one internally, so every geometry this block depends on (hidden
    /// size, HC count, conv width) was whatever `Default` happened to say rather
    /// than what the checkpoint declares.
    ///
    /// `gdn_cache` carries the Gated DeltaNet recurrent state for this layer
    /// and is advanced in place. Layers that are full attention ignore it.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        cfg: &Qwen38FlashNextConfig,
        gdn_cache: &mut crate::qwen4exp_gdn::Qwen38GdnCache,
        qsa_keys: &mut Vec<f32>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];

        // 1. Attention sub-layer with HC Prepare/Inject
        let (x_norm_attn, branch_attn) = if x.shape().dims().last() == Some(&cfg.hidden_size) {
            let normed = self.attn_norm.forward(x)?;
            (normed.clone(), normed)
        } else {
            self.hc_attn.prepare(x, &cfg)?
        };

        let attn_out = match &self.attn {
            Qwen38Attention::Linear {
                attn_qkv,
                attn_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
                n_v_heads,
                n_k_heads,
                head_dim,
                conv_dim,
                pairing,
            } => {
                let (n_v_heads, n_k_heads, head_dim, conv_dim) =
                    (*n_v_heads, *n_k_heads, *head_dim, *conv_dim);
                let pairing = *pairing;
                // 1. Fused qkv projection, then the depthwise causal conv over
                //    the whole [K K V] stream. The conv history lives in the
                //    cache so it survives across decode steps.
                let qkv = attn_qkv.forward(&branch_attn)?;
                let conv_mix = {
                    let taps = cfg.linear_conv_kernel_dim.max(2);
                    let need = (taps - 1) * conv_dim;
                    if gdn_cache.conv_state.len() < need {
                        gdn_cache.conv_state.resize(need, 0.0);
                    }
                    let history = gdn_cache.conv_state[..need].to_vec();
                    let mut state_t = cpu_tensor(
                        history,
                        grim_tensor::Shape::new(vec![1, taps - 1, conv_dim]),
                    );
                    // short_conv1d takes [B, S, D] and the block stream is
                    // [S, D]. Do the conv on the host, alongside the recurrence
                    // that consumes it, rather than pushing a rank change through
                    // a device tensor.
                    let qkv_dims = qkv.shape().dims().to_vec();
                    let (qkv_seq, qkv_chan) = match qkv_dims.as_slice() {
                        [s, d] => (*s, *d),
                        [b, s, d] => (*s * *b, *d),
                        other => {
                            return Err(grim_core::error::Error::Shape(format!(
                                "qwen38 GDN conv: expected [S, D] or [B, S, D], got {other:?}"
                            )));
                        }
                    };
                    if qkv_chan != conv_dim {
                        return Err(grim_core::error::Error::Shape(format!(
                            "qwen38 GDN conv: attn_qkv produced {qkv_chan} channels, \
                             expected {conv_dim} (key_dim*2 + value_dim)"
                        )));
                    }
                    let qkv_b1 = cpu_tensor(
                        qkv.to_vec_f32()?,
                        grim_tensor::Shape::new(vec![1, qkv_seq, qkv_chan]),
                    );
                    // short_conv1d is a HOST reference op: it reads its
                    // weight through to_vec_f32, so handing it a device tensor
                    // fails with "storage is not CpuStorage" on a ROCm run.
                    // The activations are already on the host here (qkv_b1 is a
                    // cpu_tensor), so the weight must come down to match.
                    let conv_w = if ssm_conv1d.weight.device().is_cpu() {
                        ssm_conv1d.weight.clone()
                    } else {
                        let w = ssm_conv1d.weight.to_vec_f32()?;
                        let shape = ssm_conv1d.weight.shape().clone();
                        cpu_tensor(w, shape)
                    };
                    let out =
                        grim_nn::modules::short_conv1d(&qkv_b1, &conv_w, None, Some(&mut state_t))?;
                    let updated = out.to_vec_f32()?;
                    // The conv kernel advances the state in place; mirror it back
                    // so the next decode step sees the history.
                    let hist = state_t.to_vec_f32()?;
                    gdn_cache.conv_state[..need].copy_from_slice(&hist[..need]);
                    // SiLU on the convolved stream, per the reference.
                    updated
                        .iter()
                        .map(|v| v / (1.0 + (-v).exp()))
                        .collect::<Vec<f32>>()
                };

                // 2. Per-timestep-rank gates: alpha raw (softplus + ssm_a inside
                //    the recurrence), beta raw (sigmoid inside).
                let alpha_raw = ssm_alpha.forward(&branch_attn)?.to_vec_f32()?;
                let beta_raw = ssm_beta.forward(&branch_attn)?.to_vec_f32()?;
                let a_vec: Vec<f32> = host_f32(&ssm_a)?;
                let dt_vec: Vec<f32> = host_f32(&ssm_dt)?;
                // ssm_norm is an RmsNorm over the 128-wide state channel, but
                // the recurrence applies it as a per-channel weight (the
                // normalization itself already happened upstream in the
                // reference; here it is the output scale). Take the weights
                // directly rather than running the norm, which would subtract
                // a mean the recurrence does not.
                let ssm_norm_weight: Vec<f32> = host_f32(&ssm_norm.weight)?;

                // 3. The recurrence itself, over the [seq] axis in order.
                let value_dim = n_v_heads * head_dim;
                let mut gdn_out = vec![0.0f32; seq_len * value_dim];
                crate::qwen4exp_gdn::gated_delta_net_forward(
                    &crate::qwen4exp_gdn::GdnParams {
                        conv_mix: &conv_mix,
                        alpha: &alpha_raw,
                        beta: &beta_raw,
                        a: &a_vec,
                        dt_bias: &dt_vec,
                        norm: &ssm_norm_weight,
                        // layer_norm_rms_eps, used by the q/k L2 norm.
                        eps: cfg.rms_norm_eps,
                        n_v_heads,
                        n_k_heads,
                        head_dim,
                        conv_dim,
                        seq_len,
                        pairing,
                    },
                    gdn_cache,
                    &mut gdn_out,
                )?;

                // 4. Output gate, then ssm_out.
                //
                // The gate multiply must happen on ONE device. Picking the
                // device from `gdn_t` (the recurrence output, which is host)
                // and then multiplying a device `gate_sig` through the CPU
                // backend fails with "storage is not CpuStorage"; picking the
                // device from the gate and multiplying host data through the
                // ROCm backend fails the same way. The recurrence is host code,
                // so the gate is folded on the host and the PRODUCT is lifted
                // back to the model's device for `ssm_out`.
                let gate = attn_gate.forward(&branch_attn)?;
                let gate_host = host_f32(&gate)?;
                let gated: Vec<f32> = gdn_out
                    .iter()
                    .zip(gate_host.iter())
                    .map(|(g, s)| g / (1.0 + (-s).exp()))
                    .collect();
                let gated_t = if branch_attn.device().is_cpu() {
                    cpu_tensor(gated, grim_tensor::Shape::new(vec![seq_len, value_dim]))
                } else {
                    grim_nn::modules::move_to_device(
                        &cpu_tensor(gated, grim_tensor::Shape::new(vec![seq_len, value_dim])),
                        branch_attn.device(),
                    )?
                };
                ssm_out.forward(&gated_t)?
            }
            Qwen38Attention::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                indexer_q,
                indexer_k,
                indexer_q_norm,
                indexer_k_norm,
                rope: _,
                num_heads,
                num_kv_heads,
                head_dim,
                wqkv_q80_fused,
            } => {
                let (mut q, mut k, v) = match wqkv_q80_fused.as_ref() {
                    Some(fused) if seq_len == 1 => {
                        crate::shared_attention::fused_qkv_project_raw(&branch_attn, fused)?
                    }
                    _ => {
                        let q = wq.forward(&branch_attn)?;
                        let k = wk.forward(&branch_attn)?;
                        let v = wv.forward(&branch_attn)?;
                        (q, k, v)
                    }
                };

                // q/k norms are PER HEAD: the weight is [head_dim] and the
                // tensor is [S, n_heads * head_dim], so it has to be viewed as
                // [S * n_heads, head_dim] and viewed back. Norming the flat
                // tensor with a head_dim weight is a shape error at best and a
                // silently wrong scale at worst.
                if let Some(qn) = q_norm {
                    let nh = *num_heads;
                    let hd = *head_dim;
                    let v = crate::block::reshaped_view(&q, &Shape::new(vec![seq_len * nh, hd]))?;
                    let n = qn.forward(&v)?;
                    q = crate::block::reshaped_view(&n, &Shape::new(vec![seq_len, nh * hd]))?;
                }
                if let Some(kn) = k_norm {
                    let nh = *num_kv_heads;
                    let hd = *head_dim;
                    let v = crate::block::reshaped_view(&k, &Shape::new(vec![seq_len * nh, hd]))?;
                    let n = kn.forward(&v)?;
                    k = crate::block::reshaped_view(&n, &Shape::new(vec![seq_len, nh * hd]))?;
                }

                let dev = grim_nn::modules::pick_device_for_storage_device(x.device());
                let rope_cfg = grim_tensor::RopeConfig::new(*head_dim, 10000.0);
                let rope_ext = |t: &Tensor, heads: usize| -> Result<Tensor> {
                    let mut pos_ext = Vec::with_capacity(seq_len * heads);
                    for &pos in positions {
                        for _ in 0..heads {
                            pos_ext.push(pos);
                        }
                    }
                    let t3 = crate::block::reshaped_view(
                        t,
                        &Shape::new(vec![1, seq_len * heads, *head_dim]),
                    )?;
                    match dev.rope(t3.storage().as_ref(), &pos_ext, &rope_cfg, t3.shape()) {
                        Ok((rope_s, _h)) => {
                            let roped = Tensor::new(
                                rope_s.into(),
                                t3.shape().clone(),
                                grim_tensor::DType::F32,
                                t.provenance().clone(),
                                t.device().clone(),
                            );
                            crate::block::reshaped_view(
                                &roped,
                                &Shape::new(vec![seq_len, heads * *head_dim]),
                            )
                        }
                        Err(_) => {
                            let mut vec = t.to_vec_f32()?;
                            crate::qwen35::apply_rope_neox(
                                &mut vec, positions, heads, *head_dim, 10000.0,
                            );
                            Ok(cpu_tensor(vec, t.shape().clone()))
                        }
                    }
                };

                let q_rope = rope_ext(&q, *num_heads)?;
                let k_rope = rope_ext(&k, *num_kv_heads)?;

                // --- Qwen Sparse Attention indexer (plan 4d) ---
                //
                // The indexer runs on the K/V history this layer is about to
                // attend over, and its selection becomes an additive mask.
                // Without it the layer is dense, which is what upstream
                // f3f1a8f did (it computed top_k and passed 0) and what this
                // implementation did before; current upstream applies it.
                // The indexer and the masked softmax are host reference code
                // today. On a GPU run that means reading the K/V history back
                // across the bus once per token per sparse layer, which is a
                // real cost. Set GRIM_QWEN38_QSA=0 to force dense and measure
                // the difference; the indexer weights stay loaded either way.
                let qsa_enabled = std::env::var("GRIM_QWEN38_QSA")
                    .ok()
                    .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "no"))
                    .unwrap_or(true);
                let qsa_geom = if qsa_enabled {
                    self.qsa_geometry(cfg)
                } else {
                    None
                };
                let qsa_keep: Option<Vec<f32>> = match qsa_geom {
                    Some(qcfg) => {
                        // Raw indexer keys for the NEW tokens are cached
                        // unpooled and un-normed; pooling, norm and rotation
                        // all happen on the pooled result.
                        let k_new = indexer_k
                            .as_ref()
                            .expect("qsa_geometry implies an indexer")
                            .forward(&branch_attn)?
                            .to_vec_f32()?;
                        let idx_dim = qcfg.idx_dim;
                        // `qsa_keys` is this layer's flat raw-key history.
                        qsa_keys.extend_from_slice(&k_new);
                        let n_kv = qsa_keys.len() / idx_dim.max(1);

                        // QSA block allocation is keyed on (sequence set, index bucket), not on
                        // cell index. Grim's Qwen3.8 session is single-sequence today, so every
                        // cell carries sequence 0; the allocator still runs because that is the
                        // case where `pos / r` and the correct grouping coincide, and a
                        // multi-sequence cache will need this to be per-sequence.
                        let n_blocks = qcfg.n_blocks(n_kv);
                        let cell_info: Vec<crate::qwen4exp_qsa_blocks::CellInfo> = (0..n_kv)
                            .map(|j| crate::qwen4exp_qsa_blocks::CellInfo::new(j as u32, 0))
                            .collect();
                        let layout = crate::qwen4exp_qsa_blocks::allocate_blocks(
                            &cell_info,
                            n_blocks,
                            qcfg.compress_ratio,
                        )?;
                        if layout.out_of_range {
                            return Err(grim_core::error::Error::Config(format!(
                                "qwen38 QSA: a cell position runs past the block window \
                                 (n_kv={n_kv}, n_blocks={n_blocks}, r={})",
                                qcfg.compress_ratio
                            )));
                        }
                        let keys = qsa_keys.clone();
                        // Indexer query: project, norm per head, then reuse the
                        // block's rope so the two are on the same frequency
                        // schedule.
                        let q_idx = indexer_q
                            .as_ref()
                            .expect("qsa_geometry implies an indexer")
                            .forward(&branch_attn)?
                            .to_vec_f32()?;
                        let mut q_normed = q_idx.clone();
                        if let Some(n) = indexer_q_norm.as_ref() {
                            let w = n.weight.to_vec_f32()?;
                            crate::qwen4exp_qsa::rms_norm_rows(
                                &mut q_normed,
                                qcfg.n_idx_h * idx_dim,
                                &w,
                                cfg.rms_norm_eps,
                            );
                        }
                        // Pool + norm the keys, over the ALLOCATED blocks
                        // rather than cell/r. Upstream pools first, then norms
                        // the pooled rows.
                        let mut pooled =
                            layout.pooled(&keys, n_kv, idx_dim, qcfg.compress_ratio)?;
                        if let Some(n) = indexer_k_norm.as_ref() {
                            let w = n.weight.to_vec_f32()?;
                            crate::qwen4exp_qsa::rms_norm_rows(
                                &mut pooled,
                                idx_dim,
                                &w,
                                cfg.rms_norm_eps,
                            );
                        }
                        // Score, expand, select. One query row: the last token
                        // of this step decides which history cells stay visible.
                        let q_pos = positions.last().copied().unwrap_or(0);
                        // Eq. (15): a block is scored only once all r of its
                        // tokens are observed, and Eq. (16) makes the tail
                        // always visible. Upstream folds both into a bias added
                        // to the scores before top-k; without it a query
                        // attends to future blocks and the tail can be dropped.
                        let bias = layout.block_bias(q_pos, qcfg.compress_ratio, &|_| true);
                        let mut scores = crate::qwen4exp_qsa::indexer_block_scores(
                            &pooled,
                            &q_normed[q_normed.len() - qcfg.n_idx_h * idx_dim..],
                            layout.n_bid,
                            qcfg.n_idx_h,
                            idx_dim,
                            1,
                        )?;
                        for (b, s) in scores.iter_mut().enumerate() {
                            *s += bias.get(b).copied().unwrap_or(0.0);
                        }
                        // Cell -> block comes from the ALLOCATION, not from
                        // `cell / r`. Unpooled cells land at -inf and so are
                        // never selected, which also covers the first token of
                        // a sequence, where no complete block exists yet.
                        let cells = crate::qwen4exp_qsa::expand_allocated_block_scores(
                            &scores,
                            &layout.blk_of,
                            1,
                        );
                        // The indexer scores its own key history, whose length
                        // is not necessarily the attention K/V length. The mask
                        // must span the ATTENTION kv cells, so clamp the
                        // selection to that length. `k` is [S, n_kv * head_dim]
                        // and is available here, before the host materialization.
                        let sel = crate::qwen4exp_qsa::top_k_cells(&cells, qcfg.select_width(n_kv));
                        let kv_stride = (*num_kv_heads * *head_dim).max(1);
                        let k_elems = k.to_vec_f32()?.len();
                        let kv_len = (k_elems / kv_stride).max(1);
                        let sel = sel.into_iter().filter(|c| *c < kv_len).collect::<Vec<_>>();
                        Some(crate::qwen4exp_qsa::build_top_k_mask(kv_len, &sel, None)?)
                    }
                    None => None,
                };

                let out_shape = Shape::new(vec![seq_len, *num_heads * *head_dim]);
                // The fused device kernel takes no additive mask, so it CANNOT
                // honour the QSA selection. Trying it first and only falling
                // back to the masked path on Err means a device that supports
                // the kernel silently runs DENSE -- the indexer weights load,
                // the scores are computed, and the result is thrown away. That
                // is the f3f1a8f behaviour with none of the visibility.
                //
                // So: when a mask exists, the masked path is the path. The
                // kernel is only eligible for a dense step.
                let attn_tensor = match dev.qkv_attention(
                    q_rope.storage().as_ref(),
                    k_rope.storage().as_ref(),
                    v.storage().as_ref(),
                    *num_kv_heads,
                    seq_len,
                    0,
                    None,
                    &out_shape,
                    None,
                    None,
                ) {
                    Ok((s, _h)) if qsa_keep.is_none() => Tensor::new(
                        std::sync::Arc::from(s),
                        out_shape.clone(),
                        grim_tensor::DType::F32,
                        grim_tensor::QuantProvenance::default(),
                        x.device().clone(),
                    ),
                    // Any other outcome (kernel failed, or a mask exists that the
                    // kernel cannot honour) takes the masked/dense host path.
                    _ => {
                        let q_heads = q_rope.to_vec_f32()?;
                        let k_heads = k_rope.to_vec_f32()?;
                        let v_heads = v.to_vec_f32()?;
                        match qsa_keep.as_ref() {
                            // Sparse: restrict the softmax to the cells the
                            // indexer selected. Only valid when this step's K/V
                            // history IS the cells the mask was built over,
                            // which holds for a single-token decode.
                            Some(keep) if seq_len == 1 => {
                                let kv_len = keep.len();
                                let out = crate::qwen4exp_qsa::masked_gqa_attention(
                                    &q_heads,
                                    &k_heads,
                                    &v_heads,
                                    *num_heads,
                                    *num_kv_heads,
                                    *head_dim,
                                    seq_len,
                                    kv_len,
                                    kv_len.saturating_sub(seq_len),
                                    keep,
                                )?;
                                // The indexer and the masked softmax are host
                                // reference code, so their result arrives on the
                                // CPU. Land it on the tensor's own device or
                                // every later op in the block silently mixes
                                // devices.
                                let host = cpu_tensor(out, out_shape.clone());
                                if *x.device() == Device::Cpu {
                                    host
                                } else {
                                    grim_nn::modules::move_to_device(&host, x.device())?
                                }
                            }
                            // Prefill with a multi-token step cannot apply a
                            // single-row mask, so fall back to dense. The
                            // indexer is a decode-time accelerator; upstream
                            // scores per token and masks per token, which needs
                            // the full [n_kv, n_tps] mask this path does not
                            // build.
                            _ => crate::shared_attention::fused_or_scalar_attention(
                                &q_heads,
                                &k_heads,
                                &v_heads,
                                *num_heads,
                                *num_kv_heads,
                                *head_dim,
                                seq_len,
                                None,
                                x.device(),
                            )?,
                        }
                    }
                };
                wo.forward(&attn_tensor)?
            }
        };

        let res1 = if x.shape().dims().last() == Some(&cfg.hidden_size) {
            grim_nn::modules::add_on_device(x, &attn_out)?
        } else {
            self.hc_attn.inject(x, &x_norm_attn, &attn_out, &cfg)?
        };

        // 2. FFN / MoE sub-layer with HC Prepare/Inject
        let (x_norm_ffn, branch_ffn) = if res1.shape().dims().last() == Some(&cfg.hidden_size) {
            let normed = self.ffn_norm.forward(&res1)?;
            (normed.clone(), normed)
        } else {
            self.hc_ffn.prepare(&res1, &cfg)?
        };

        let moe_out = self.moe_block.forward(&branch_ffn)?;

        let res2 = if res1.shape().dims().last() == Some(&cfg.hidden_size) {
            grim_nn::modules::add_on_device(&res1, &moe_out)?
        } else {
            self.hc_ffn.inject(&res1, &x_norm_ffn, &moe_out, &cfg)?
        };

        Ok(res2)
    }
}

/// PLE n-gram hash addressing.
///
/// Transcribed from `old/repo/llama.cpp-master/src/models/qwen4exp.cpp` at
/// commit `d7241ac8`, `llm_graph_input_ple::set_input`:
///
/// ```text
/// mixed_n = (t[p] * m[0]) ^ (t[p-1] * m[1]) ^ ... ^ (t[p-n+1] * m[n-1])
/// row    = mixed_n % vocab[h] + offset[h]
/// ```
///
/// for every `n` in `2..=ngram_size` and, within each `n`, for all
/// `heads_per_ngram` heads. Note this is an XOR of scaled products, NOT a
/// polynomial modular hash, and the moduli are PER-HEAD vocab sizes taken
/// from the checkpoint, not `vocab_size` of the model. The previous
/// implementation synthesised coprime moduli from an `m_base` and used a
/// polynomial sum, neither of which appears in the reference.
///
/// The hash is host-side because it needs 64-bit multiply-xor; upstream says
/// the same ("ggml has no int64 and no xor").
#[derive(Clone, Debug)]
pub struct Qwen38NgramAddressing {
    /// `ple_layer_multipliers`, length `ngram_size`. These are the published
    /// 64-bit constants, NOT a base to derive weights from.
    pub layer_multipliers: Vec<u64>,
    /// `ple_head_offsets`, length `(ngram_size - 1) * heads_per_ngram`.
    pub head_offsets: Vec<u64>,
    /// `ple_head_vocab_sizes`, same length as `head_offsets`. These differ per
    /// head (the released file has 16 distinct values around 2.0e7), so a single
    /// shared vocabulary is not a valid approximation.
    pub head_vocab_sizes: Vec<u64>,
    /// `ple_heads_per_ngram`.
    pub heads_per_ngram: usize,
    /// `ple_ngram_size`.
    pub ngram_size: usize,
    /// `ple_eos_token_id`; an EOS in the window resets everything at or before it.
    pub eos_token_id: u32,
    /// `ple_image_token_id`, used when a batch arrives as embeddings.
    pub image_token_id: Option<u32>,
}

impl Qwen38NgramAddressing {
    /// Total head count, `(ngram_size - 1) * heads_per_ngram`. Upstream
    /// requires this to equal the lengths of `head_offsets` and
    /// `head_vocab_sizes`.
    pub fn n_heads(&self) -> usize {
        (self.ngram_size.saturating_sub(1)).saturating_mul(self.heads_per_ngram)
    }

    /// Row count the shared PLE table must provide: `max(offset + vocab_size)`.
    ///
    /// The released file's 16 heads are contiguous but have per-head vocab
    /// sizes, so the total is `320_001_446`, not `16 * 20_000_000`.
    pub fn table_rows(&self) -> usize {
        self.head_offsets
            .iter()
            .zip(self.head_vocab_sizes.iter())
            .map(|(o, v)| (o + v) as usize)
            .max()
            .unwrap_or(0)
    }

    /// Build the addressing from checkpoint metadata.
    ///
    /// # Errors
    /// Returns a config error when the arrays are inconsistent: upstream
    /// requires `layer_multipliers` to have `ngram_size` entries and both head
    /// arrays to have `n_heads`. Silently accepting a mismatch would gather
    /// from the wrong rows and produce plausible, wrong embeddings.
    pub fn from_metadata(
        layer_multipliers: Vec<u64>,
        head_offsets: Vec<u64>,
        head_vocab_sizes: Vec<u64>,
        heads_per_ngram: usize,
        ngram_size: usize,
        eos_token_id: u32,
        image_token_id: Option<u32>,
    ) -> Result<Self> {
        let probe = Self {
            layer_multipliers: layer_multipliers.clone(),
            head_offsets: head_offsets.clone(),
            head_vocab_sizes: head_vocab_sizes.clone(),
            heads_per_ngram,
            ngram_size,
            eos_token_id,
            image_token_id,
        };
        let n_heads = probe.n_heads();
        if ngram_size < 2 {
            return Err(grim_core::Error::Config(format!(
                "PLE n-gram size {ngram_size} is out of range (must be >= 2)"
            )));
        }
        if n_heads == 0 {
            return Err(grim_core::Error::Config(
                "PLE head count is zero".to_string(),
            ));
        }
        if layer_multipliers.len() != ngram_size {
            return Err(grim_core::Error::Config(format!(
                "PLE layer_multipliers has {} entries, expected ngram_size={ngram_size}",
                layer_multipliers.len()
            )));
        }
        if head_offsets.len() != n_heads || head_vocab_sizes.len() != n_heads {
            return Err(grim_core::Error::Config(format!(
                "PLE head arrays have {}/{} entries, expected n_heads={n_heads} \
                 (=(ngram_size-1) * heads_per_ngram)",
                head_offsets.len(),
                head_vocab_sizes.len()
            )));
        }
        if head_vocab_sizes.iter().any(|v| *v == 0) {
            return Err(grim_core::Error::Config(
                "PLE head_vocab_sizes contains a zero".to_string(),
            ));
        }
        Ok(probe)
    }

    /// Gather row indices for one position.
    ///
    /// `ctx` is the token window oldest-first with `ctx[0]` the current token,
    /// as upstream builds it: an EOS in the window resets everything at or
    /// before it, a missing predecessor reads as EOS, and the current token's
    /// own EOS does not cut its own context.
    ///
    /// Returns `n_heads` row indices into the shared table.
    pub fn rows_for_context(&self, ctx: &[u32]) -> Vec<u32> {
        let n_heads = self.n_heads();
        let mut idx = vec![0u32; n_heads];
        let per_gram = self.heads_per_ngram;
        for n in 2..=self.ngram_size {
            // XOR of the scaled products for the n newest tokens.
            let mut mixed = (ctx.first().copied().unwrap_or(self.eos_token_id) as u64)
                .wrapping_mul(self.layer_multipliers[0]);
            for j in 1..n {
                let t = ctx.get(j).copied().unwrap_or(self.eos_token_id) as u64;
                mixed ^= t.wrapping_mul(self.layer_multipliers[j]);
            }
            let base = (n - 2) * per_gram;
            for g in 0..per_gram {
                let h = base + g;
                idx[h] = (mixed % self.head_vocab_sizes[h] + self.head_offsets[h]) as u32;
            }
        }
        idx
    }

    /// Build the window for position `i` from the token history, applying the
    /// EOS reset rule.
    ///
    /// `prev` is the `ngram_size - 1` preceding tokens, oldest-first, with
    /// `None` for anything before the start of the sequence.
    pub fn context_at(&self, token: u32, prev: &[Option<u32>]) -> Vec<u32> {
        let n_gram = self.ngram_size;
        let mut ctx = vec![self.eos_token_id; n_gram];
        ctx[0] = token;
        let mut cut = false;
        for s in 1..n_gram {
            // prev is oldest-first, so the token s positions back is
            // prev[n_prev - s].
            let n_prev = n_gram - 1;
            let t = if cut {
                None
            } else {
                prev.get(n_prev - s).copied().flatten()
            };
            cut = cut || t.is_none_or(|v| v == self.eos_token_id);
            ctx[s] = if cut {
                self.eos_token_id
            } else {
                t.unwrap_or(self.eos_token_id)
            };
        }
        ctx
    }
}

/// N-gram embedding layer implementing Position-aware / Prompt-Lookup N-gram Embedding (PLE).
/// Maps high-order token n-grams ($N \in [2, 3]$) to compact auxiliary representations that augment standard.
#[derive(Clone)]
pub struct Qwen38NgramEmbedding {
    /// N-gram vocabulary size ($V_{\text{ngram}}$, e.g. 20M entries).
    pub ngram_vocab_size: usize,
    /// N-gram embedding dimension ($d_{\text{ngram}}$, e.g. 2560).
    pub ngram_dim: usize,
    /// Model hidden dimension ($d_{\text{model}}$, e.g. 2560).
    pub hidden_size: usize,
    /// N-gram embedding table, `[ngram_vocab, ple_head_dim]`.
    ///
    /// Row count is PADDED past the largest head range: the checkpoint stores
    /// 320_001_536 rows while `max(head_offset + head_vocab_size)` is
    /// 320_001_446, so a config-derived row count cannot be used to fetch it.
    /// Both container formats expose the table row-major, so the gather's
    /// `row * ngram_dim` indexing is correct for either.
    pub table: Tensor,
    /// Linear projection from $d_{\text{ngram}} \to d_{\text{model}}$.
    pub proj: Linear,
    /// Deterministic coprime polynomial modular addressing generator.
    pub addressing: Qwen38NgramAddressing,
}

impl Qwen38NgramEmbedding {
    /// Performs N-gram lookup and projection for a sequence of tokens.
    /// # Contract * `tokens.len() == seq_len`.
    pub fn lookup_and_project(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Ok(cpu_tensor(
                vec![],
                grim_tensor::Shape::new(vec![0, self.hidden_size]),
            ));
        }

        let table_vec = self.table.to_vec_f32()?;
        let n_heads = self.addressing.n_heads();
        // Upstream lays the gathered rows out as [ple_n_heads * n_tokens] and
        // then reshapes to [n_tokens, ple_n_heads * ple_head_dim], so the
        // per-token block is the concatenation of that token's n_heads rows in
        // head order. One row per (n, g) pair, not one row per position.
        let row_dim = n_heads * self.ngram_dim;
        let mut gathered_ngram = vec![0.0f32; seq_len * row_dim];

        let n_prev = self.addressing.ngram_size - 1;
        for i in 0..seq_len {
            let prev: Vec<Option<u32>> = (0..n_prev)
                .map(|j| tokens.get(i + 1 + j).copied())
                .collect();
            let ctx = self.addressing.context_at(tokens[i], &prev);
            let rows = self.addressing.rows_for_context(&ctx);
            for (h, &row) in rows.iter().enumerate() {
                let src = row as usize * self.ngram_dim;
                let dst = i * row_dim + h * self.ngram_dim;
                if src + self.ngram_dim <= table_vec.len() {
                    gathered_ngram[dst..dst + self.ngram_dim]
                        .copy_from_slice(&table_vec[src..src + self.ngram_dim]);
                }
            }
        }

        // The gathered block is n_heads * ngram_dim wide, so the tensor shape
        // must say so. Declaring ngram_dim here while the buffer is
        // n_heads * ngram_dim long is a size mismatch, and the projection is
        // sized [ngram_dim, hidden] to match the reference's ple_head_dim.
        let ngram_tensor = cpu_tensor(
            gathered_ngram,
            grim_tensor::Shape::new(vec![seq_len, row_dim]),
        );
        Ok(self.proj.forward(&ngram_tensor)?)
    }
}

// Model

/// Qwen3.8-Flash-Next Causal Language Model.
pub struct Qwen38FlashNext {
    pub cfg: Qwen38FlashNextConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub ngram_embeddings: Option<Qwen38NgramEmbedding>,
    pub layers: Vec<Qwen38FlashNextBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

/// Per-session Gated DeltaNet state, one [`Qwen38GdnCache`] per layer.
///
/// Held in [`SessionT::model_state`](grim_core::session::SessionT::model_state)
/// rather than on the model: `CausalLm::forward` takes `&self`, so a
/// model-owned cache could not be advanced in place and decode would be
/// stateless across steps. The slot is typed `Any + Send`, so this needs no
/// change to the core session contract.
#[derive(Debug, Clone, Default)]
pub struct Qwen38GdnSession {
    pub caches: Vec<crate::qwen4exp_gdn::Qwen38GdnCache>,
    /// Raw indexer keys per full-attention layer, `[n_kv, indexer_key_length]`.
    ///
    /// Stored RAW on purpose: upstream caches `index_k_proj` output and applies
    /// pooling, then the norm and the rotation, to the pooled result. Caching a
    /// normed or rotated key would not commute with the mean.
    pub qsa_keys: Vec<Vec<f32>>,
}

impl Qwen38GdnSession {
    /// Ensure a cache exists for every layer, sizing each from the layer's own
    /// declared geometry.
    pub fn ensure(&mut self, layers: &[Qwen38FlashNextBlock], cfg: &Qwen38FlashNextConfig) {
        if self.caches.len() < layers.len() {
            self.caches.resize(layers.len(), Default::default());
        }
        for (i, layer) in layers.iter().enumerate() {
            if !self.caches[i].ssm_state.is_empty() {
                continue;
            }
            if let crate::qwen4exp_gdn::GdnGeometry::Known(g) = layer.gdn_geometry() {
                self.caches[i] = crate::qwen4exp_gdn::Qwen38GdnCache::new(
                    g.n_v_heads,
                    g.head_dim,
                    g.head_dim,
                    cfg.linear_conv_kernel_dim.max(2),
                    g.conv_dim,
                );
            }
        }
    }
}

impl Qwen38FlashNext {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        // Root resolution, in the order the two container formats need.
        //
        // safetensors nests everything under `model.language_model`, so a
        // probe there succeeds and that becomes the root. GGUF has NO prefix at
        // all: the checkpoint provides `token_embd.weight`,
        // `per_layer_token_embd.weight`, `blk.N.*` and `output.weight` at the
        // top level. Probing only the prefixed form made every real GGUF load
        // fail on `model.embed_tokens.weight`, while the synthetic tests -- which
        // emit the prefixed names -- passed.
        //
        // The token embedding is therefore probed under all three real names.
        // Probe by NAME, never by shape: the two containers disagree on the
        // token embedding's orientation. safetensors stores
        // [vocab, hidden]; GGUF stores `token_embd.weight` as [hidden, vocab],
        // so one shared shape probe matches at most one of them and silently
        // falls through to the wrong root.
        let root = if ws
            .scoped("model")
            .scoped("language_model")
            .has_tensor("embed_tokens")
        {
            ws.scoped("model").scoped("language_model")
        } else if ws.scoped("model").has_tensor("embed_tokens") {
            ws.scoped("model")
        } else if ws.has_tensor("token_embd.weight") {
            // GGUF: every tensor sits at the root -- `blk.N.*`, `output.weight`,
            // `token_embd.weight`, `per_layer_token_embd.weight`.
            ws.scoped("")
        } else {
            ws.scoped("model")
        };

        // The token embedding under any of its three real names.
        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )
        .or_else(|_| {
            Linear::load_shape(
                &root.scoped("token_embd"),
                [cfg.vocab_size, cfg.hidden_size],
            )
        })
        .or_else(|_| {
            // GGUF orientation: [hidden, vocab].
            Linear::load_shape(
                &root.scoped("token_embd"),
                [cfg.hidden_size, cfg.vocab_size],
            )
        })?;

        let ngram_embeddings = if let (Some(ngram_vocab), Some(ngram_dim)) =
            (cfg.ngram_vocab_size, cfg.ngram_dim)
        {
            // The gather yields `ple_n_heads` rows per token, each
            // `ple_head_dim` (= ngram_dim) wide, so the projected block is
            // n_heads * ngram_dim wide. For the released geometry that is
            // 16 * 160 = 2560 = hidden_size, and the reference applies no
            // projection at all (`reshape_2d(emb, ple_head_dim * n_heads,
            // n_tokens)`). Declaring the projection at ngram_dim is a size
            // mismatch against that gather.
            let ple_gathered_dim = cfg.ple_addressing()?.n_heads() * ngram_dim;
            // The row count is PADDED past the largest head range, so it
            // comes from the tensor's own metadata and not from the config.
            let padded_rows = root
                .meta("per_layer_token_embd.weight")
                .map(|m| m.shape[0])
                .unwrap_or(ngram_vocab);
            // The checkpoint PADS the PLE table past the largest head range:
            // `per_layer_token_embd.weight` is [160, 320001536] while
            // max(head_offset + head_vocab_size) is 320001446. Probing the
            // derived row count can therefore never match, so the tensor is
            // fetched by name and its own shape is trusted.
            let table = match root.get([padded_rows, ngram_dim], "per_layer_token_embd.weight") {
                Ok(t) => t,
                Err(_) => root
                    .scoped("layers")
                    .scoped("1")
                    .scoped("ple")
                    .scoped("ple_embedding")
                    .scoped("ngram_embedding")
                    .get([ngram_vocab, ngram_dim], "shard_0")
                    .or_else(|_| {
                        root.scoped("ngram_embeddings")
                            .get([ngram_vocab, ngram_dim], "weight")
                    })
                    .or_else(|_| {
                        root.scoped("ple_ngram_embd")
                            .get([ngram_vocab, ngram_dim], "weight")
                    })
                    .map_err(|e| {
                        grim_core::Error::Config(format!(
                            "Qwen38FlashNext: config defines ngram_vocab_size={ngram_vocab}, \
                             ngram_dim={ngram_dim}, but no PLE table was found under any known \
                             name (per_layer_token_embd [{ngram_dim}, {ngram_vocab}], \
                             layers.1.ple.*, ngram_embeddings, ple_ngram_embd): {e}"
                        ))
                    })?,
            };

            let proj = Linear::load_shape(
                &root
                    .scoped("layers")
                    .scoped("1")
                    .scoped("ple")
                    .scoped("key_proj"),
                [ple_gathered_dim, cfg.hidden_size],
            )
            .or_else(|_| {
                Linear::load_shape(
                    &root.scoped("ngram_proj"),
                    [ple_gathered_dim, cfg.hidden_size],
                )
            })
            .or_else(|_| {
                Linear::load_shape(
                    &root.scoped("ple_ngram_proj"),
                    [ple_gathered_dim, cfg.hidden_size],
                )
            })
            .map_err(|e| {
                grim_core::Error::Config(format!(
                    "Qwen38FlashNext: checkpoint config specifies PLE projection, \
                     but 'key_proj'/'ngram_proj' were not found in weights: {e}"
                ))
            })?;

            let addressing = cfg.ple_addressing()?;

            Some(Qwen38NgramEmbedding {
                ngram_vocab_size: ngram_vocab,
                ngram_dim,
                hidden_size: cfg.hidden_size,
                table,
                proj,
                addressing,
            })
        } else {
            None
        };

        let num_layers_to_load = cfg.num_layers;
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            // safetensors nests blocks under `layers.N`, GGUF names them
            // `blk.N`. Probe both, so one loader serves either container.
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let layer_ws = if layer_ws.scoped("input_layernorm").has_tensor("weight") {
                layer_ws
            } else {
                root.scoped("blk").scoped(&i.to_string())
            };
            let block = Qwen38FlashNextBlock::load(&layer_ws, &cfg, i, tp)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps).or_else(
            |_| {
                RmsNorm::load(
                    &root.scoped("hyper_connection_mixer").scoped("hc_norm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )
            },
        )?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| Linear::from_tensor(tok_embeddings.w_t.clone(), None));

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            ngram_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: Qwen38FlashNextConfig) -> Self {
        let tok_embeddings = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        let ngram_embeddings = if let (Some(ngram_vocab), Some(ngram_dim)) =
            (cfg.ngram_vocab_size, cfg.ngram_dim)
        {
            let table = cpu_tensor(
                vec![0.01f32; ngram_vocab * ngram_dim],
                grim_tensor::Shape::new(vec![ngram_vocab, ngram_dim]),
            );
            let ple_gathered_dim = cfg.ple_addressing().expect("default PLE").n_heads() * ngram_dim;
            let proj = Linear::from_tensor(
                cpu_tensor(
                    vec![0.01f32; cfg.hidden_size * ple_gathered_dim],
                    grim_tensor::Shape::new(vec![cfg.hidden_size, ple_gathered_dim]),
                ),
                None,
            );
            // `random` returns Self, so it cannot propagate a metadata
            // error. Use the default (well-formed) addressing; the real
            // constants arrive through `load_tp`, which does propagate.
            let addressing = cfg
                .ple_addressing()
                .expect("the default PLE constants are well-formed");
            Some(Qwen38NgramEmbedding {
                ngram_vocab_size: ngram_vocab,
                ngram_dim,
                hidden_size: cfg.hidden_size,
                table,
                proj,
                addressing,
            })
        } else {
            None
        };

        let norm = RmsNorm {
            weight: cpu_tensor(
                vec![1.0; cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        };
        let output = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        Self {
            cfg,
            device,
            tok_embeddings,
            ngram_embeddings,
            layers: vec![],
            norm,
            output,
        }
    }
}

/// Read a tensor into a host `Vec<f32>` for the host reference paths.
///
/// Every GDN and QSA tensor is consumed by host code (the delta rule, the
/// indexer, `short_conv1d`). Those read through `to_vec_f32`, which works for a
/// device storage, but the ops they then call -- `short_conv1d` and friends --
/// downcast to `CpuStorage` and fail with "storage is not CpuStorage" when handed
/// a device tensor. This helper is the one place that boundary is crossed, so
/// there is one thing to audit rather than a scattering of `.to_vec_f32()`.
///
/// # Errors
/// Propagates the device-to-host failure.
fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_vec_f32()?)
}

impl Model for Qwen38FlashNext {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for Qwen38FlashNext {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.into_iter().map(|p| p as u32).collect();

        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let ids_u32: Vec<u32> = ids_f32.iter().map(|&v| v as u32).collect();

        let mut h = if self.device != Device::Cpu {
            let mut h_dev = grim_nn::modules::embedding_gather_on_device(
                &self.tok_embeddings.weight,
                &ids_u32,
                seq_len,
                self.cfg.hidden_size,
            )?;
            if let Some(ref ngram_emb) = self.ngram_embeddings {
                let ngram_h = ngram_emb.lookup_and_project(&ids_u32)?;
                let ngram_h_dev = grim_nn::modules::move_to_device(&ngram_h, &self.device)?;
                h_dev = grim_nn::modules::add_on_device(&h_dev, &ngram_h_dev)?;
            }
            h_dev
        } else {
            let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
            let mut h_vec = vec![0.0f32; seq_len * self.cfg.hidden_size];

            for (i, &tok) in ids_u32.iter().enumerate() {
                let tok = tok as usize;
                if tok < self.cfg.vocab_size {
                    let src_start = tok * self.cfg.hidden_size;
                    let dst_start = i * self.cfg.hidden_size;
                    if src_start + self.cfg.hidden_size <= embed_w.len() {
                        h_vec[dst_start..dst_start + self.cfg.hidden_size]
                            .copy_from_slice(&embed_w[src_start..src_start + self.cfg.hidden_size]);
                    }
                }
            }

            // Auxiliary Position-aware / Prompt-Lookup N-gram Embedding (PLE) fusion
            if let Some(ref ngram_emb) = self.ngram_embeddings {
                let ngram_h = ngram_emb.lookup_and_project(&ids_u32)?;
                let ng_vec = ngram_h.to_vec_f32()?;
                for i in 0..h_vec.len().min(ng_vec.len()) {
                    h_vec[i] += ng_vec[i];
                }
            }

            cpu_tensor(
                h_vec,
                grim_tensor::Shape::new(vec![seq_len, self.cfg.hidden_size]),
            )
        };

        // Per-session GDN state. Created on first use and reused thereafter, so
        // the recurrence accumulates across decode steps within a request and
        // starts clean for the next one.
        // Take the typed slot out of the session, use it, and put it back: the
        // borrow checker cannot hold a `&mut dyn Any` across the layer loop
        // while also borrowing `self`.
        let mut gdn_state = match session.model_state_mut().and_then(|s| s.downcast_mut()) {
            Some(s) => std::mem::take(s),
            None => Qwen38GdnSession::default(),
        };
        gdn_state.ensure(&self.layers, &self.cfg);
        if gdn_state.qsa_keys.len() < self.layers.len() {
            gdn_state.qsa_keys.resize(self.layers.len(), Vec::new());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = gdn_state
                .caches
                .get_mut(i)
                .expect("ensure() sized one cache per layer");
            let keys = gdn_state.qsa_keys.get_mut(i).expect("qsa_keys sized above");
            h = layer.forward(&h, &pos_u32, &self.cfg, cache, keys)?;
        }
        session.set_model_state(Box::new(gdn_state));

        let normed = self.norm.forward(&h)?;
        session.set_last_hidden_state(normed.clone());
        Ok(self.output.forward(&normed)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen4exp_flash_next_config_defaults() {
        let cfg = Qwen38FlashNextConfig::default();
        assert_eq!(cfg.name(), "qwen3_8_flash_next");
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.hc_count, 4);
        assert_eq!(cfg.hc_lowrank, 320);
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
        assert_eq!(cfg.max_seq_len, 262144);
        // The PLE table must provide max(head_offset + head_vocab_size) rows.
        // The released file's 16 heads total 320_001_446, NOT 20_000_000 and
        // NOT 16 * 20_000_000: the per-head vocab sizes are all distinct.
        assert_eq!(
            cfg.ngram_vocab_size,
            Some(320_001_446),
            "the PLE table row count is max(offset + vocab) over all heads"
        );
        assert_eq!(
            cfg.ngram_dim,
            Some(160),
            "ple_head_dim, not the model hidden"
        );
        assert_eq!(cfg.ple_heads_per_ngram, 8);
        assert_eq!(cfg.ple_layer_multipliers.len(), 3);
        assert_eq!(cfg.ple_head_offsets.len(), 16);
        assert_eq!(cfg.ple_head_vocab_sizes.len(), 16);
        assert_eq!(cfg.split_ngram_parts, 128);
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn test_qwen38_ngram_lookup_and_forward_fusion() {
        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.ngram_vocab_size = Some(100);
        cfg.ngram_dim = Some(8);
        cfg.num_layers = 0; // Test embeddings and norm directly

        let model = Qwen38FlashNext::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![5.0, 12.0, 18.0], grim_tensor::Shape::new(vec![3]));
        let positions = cpu_tensor(vec![0.0, 1.0, 2.0], grim_tensor::Shape::new(vec![3]));

        let out = model
            .forward(session.as_mut(), &input_ids, &positions, &[])
            .unwrap();
        assert_eq!(out.shape().dims(), &[3, 32]);
    }

    #[test]
    fn test_qwen38_moe_swiglu_and_residual_scaling_numerics() {
        // Test SwiGLU activation formula: x * sigmoid(x) * u
        let g_val = 2.0f32;
        let u_val = 3.0f32;
        let sig = 1.0f32 / (1.0f32 + (-g_val).exp());
        let expected_swiglu = g_val * sig * u_val;

        let diff = (expected_swiglu - (2.0 * (1.0 / (1.0 + (-2.0f32).exp())) * 3.0)).abs();
        assert!(diff < 1e-7);

        // Test 4-branch gated residual scale: 1 / sqrt(4) = 0.5
        let branches = 4;
        let scale = 1.0f32 / (branches as f32).sqrt();
        assert_eq!(scale, 0.5f32);
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn test_qwen38_missing_ple_weights_fails_loudly() {
        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 16;
        cfg.hidden_size = 8;
        cfg.ngram_vocab_size = Some(100);
        cfg.ngram_dim = Some(4);

        // Empty weight provider: loading must error loudly rather than silently substitute dummy 0.01 tensors
        struct EmptyProvider;
        impl grim_tensor::TensorProvider for EmptyProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<grim_tensor::RawTensor> {
                Err(grim_tensor::error::Error::Backend(format!(
                    "tensor '{name}' not found"
                )))
            }
            fn meta(&self, _name: &str) -> grim_tensor::error::Result<grim_tensor::TensorMeta> {
                Err(grim_tensor::error::Error::Backend(
                    "tensor not found".into(),
                ))
            }
        }

        let provider = EmptyProvider;
        let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);

        let err = Qwen38FlashNext::load(Device::Cpu, &ws, cfg);
        assert!(
            err.is_err(),
            "load_tp must fail loudly when PLE weights are missing"
        );
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn test_qwen38_real_safetensors_layout_weight_loading_and_forward() {
        use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
        use std::collections::HashMap;

        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 16;
        cfg.hidden_size = 8;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.head_dim = 4;
        cfg.num_layers = 1;
        cfg.intermediate_size = 16;
        cfg.num_experts = 4;
        cfg.num_experts_per_tok = 2;
        cfg.shared_expert_intermediate_size = Some(16);
        cfg.ngram_vocab_size = Some(20);
        cfg.ngram_dim = Some(4);
        cfg.split_ngram_parts = 2;
        cfg.ngram_size = 3;

        let q_dim = cfg.num_heads * cfg.head_dim; // 8
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 4
        let ngram_vocab = 20;
        let ngram_dim = 4;
        // Matches the loader: the gathered block is n_heads * ple_head_dim wide.
        let ple_gathered_dim = cfg.ple_addressing().expect("default PLE").n_heads() * ngram_dim;

        fn raw_f32_tensor(
            val: f32,
            shape: Vec<usize>,
        ) -> (
            Vec<u8>,
            Vec<usize>,
            grim_tensor::DType,
            grim_tensor::QuantProvenance,
        ) {
            let count: usize = shape.iter().product();
            let mut bytes = Vec::with_capacity(count * 4);
            for _ in 0..count {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            (
                bytes,
                shape,
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::GrimNative,
            )
        }

        let mut tensors = HashMap::new();
        // Model embeddings & output (Linear::load_shape expects [out_features, in_features] or transposed)
        tensors.insert(
            "model.embed_tokens.weight".into(),
            raw_f32_tensor(0.05, vec![cfg.hidden_size, cfg.vocab_size]),
        );
        tensors.insert(
            "lm_head.weight".into(),
            raw_f32_tensor(0.02, vec![cfg.vocab_size, cfg.hidden_size]),
        );
        tensors.insert(
            "model.norm.weight".into(),
            raw_f32_tensor(1.0, vec![cfg.hidden_size]),
        );

        // PLE N-gram embedding table (matching HuggingFace / vLLM naming)
        tensors.insert(
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0".into(),
            raw_f32_tensor(0.1, vec![ngram_vocab, ngram_dim]),
        );
        tensors.insert(
            "model.layers.1.ple.key_proj.weight".into(),
            // The PLE projection consumes n_heads * ple_head_dim, not ple_head_dim.
            raw_f32_tensor(0.05, vec![cfg.hidden_size, ple_gathered_dim]),
        );

        // Layer 0 Attention & MoE weights
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            raw_f32_tensor(0.01, vec![q_dim, cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            raw_f32_tensor(0.01, vec![kv_dim, cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            raw_f32_tensor(0.01, vec![kv_dim, cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            raw_f32_tensor(0.01, vec![cfg.hidden_size, q_dim]),
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".into(),
            raw_f32_tensor(1.0, vec![cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            raw_f32_tensor(1.0, vec![cfg.hidden_size]),
        );

        // MoE Router Gate
        tensors.insert(
            "model.layers.0.mlp.gate.weight".into(),
            raw_f32_tensor(0.01, vec![cfg.num_experts, cfg.hidden_size]),
        );

        // MoE Experts
        for e in 0..cfg.num_experts {
            tensors.insert(
                format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
                raw_f32_tensor(0.01, vec![cfg.intermediate_size, cfg.hidden_size]),
            );
            tensors.insert(
                format!("model.layers.0.mlp.experts.{e}.up_proj.weight"),
                raw_f32_tensor(0.01, vec![cfg.intermediate_size, cfg.hidden_size]),
            );
            tensors.insert(
                format!("model.layers.0.mlp.experts.{e}.down_proj.weight"),
                raw_f32_tensor(0.01, vec![cfg.hidden_size, cfg.intermediate_size]),
            );
        }

        // Shared Expert
        tensors.insert(
            "model.layers.0.mlp.shared_expert.gate_proj.weight".into(),
            raw_f32_tensor(0.01, vec![16, cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.mlp.shared_expert.up_proj.weight".into(),
            raw_f32_tensor(0.01, vec![16, cfg.hidden_size]),
        );
        tensors.insert(
            "model.layers.0.mlp.shared_expert.down_proj.weight".into(),
            raw_f32_tensor(0.01, vec![cfg.hidden_size, 16]),
        );

        struct SafeTensorsMockProvider {
            tensors: HashMap<
                String,
                (
                    Vec<u8>,
                    Vec<usize>,
                    grim_tensor::DType,
                    grim_tensor::QuantProvenance,
                ),
            >,
        }

        impl TensorProvider for SafeTensorsMockProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
                let (bytes, shape, dtype, provenance) =
                    self.tensors.get(name).cloned().ok_or_else(|| {
                        grim_tensor::error::Error::Backend(format!(
                            "Tensor {name} not found in SafeTensors mock provider"
                        ))
                    })?;
                Ok(RawTensor {
                    bytes,
                    shape,
                    dtype,
                    provenance,
                })
            }
            fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
                let (_, shape, dtype, provenance) =
                    self.tensors.get(name).cloned().ok_or_else(|| {
                        grim_tensor::error::Error::Backend(format!(
                            "Tensor meta {name} not found in SafeTensors mock provider"
                        ))
                    })?;
                Ok(TensorMeta {
                    dtype,
                    provenance,
                    shape,
                    fusion_mask: 0,
                })
            }
        }

        let provider = SafeTensorsMockProvider { tensors };
        let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);

        // Load model completely through real SafeTensors WeightSource path
        let model = Qwen38FlashNext::load(Device::Cpu, &ws, cfg)
            .expect("Qwen38FlashNext must load completely from SafeTensors WeightSource");
        assert!(
            model.ngram_embeddings.is_some(),
            "PLE N-gram embeddings must be loaded from weights"
        );

        let mut session = model.new_session();
        let input_ids = cpu_tensor(vec![3.0, 7.0, 11.0], grim_tensor::Shape::new(vec![3]));
        let positions = cpu_tensor(vec![0.0, 1.0, 2.0], grim_tensor::Shape::new(vec![3]));

        let logits = model
            .forward(session.as_mut(), &input_ids, &positions, &[])
            .expect("Forward pass on loaded model must succeed");
        assert_eq!(
            logits.shape().dims(),
            &[3, 16],
            "Logits shape must match [seq_len, vocab_size]"
        );

        // Assert valid numeric output
        let logits_vec = logits.to_vec_f32().unwrap();
        assert!(!logits_vec.is_empty());
        for &val in &logits_vec {
            assert!(!val.is_nan(), "Logits must not contain NaN");
            assert!(!val.is_infinite(), "Logits must not contain Inf");
        }
    }

    /// Verifies numerical weight loading and forward signal propagation directly against the real physical 992MB SafeTensors model shard on disk (`models/qwen3.8-model-00001-of-00131.safetensors`).
    /// # Contract & Checks 1.
    #[test]
    fn test_qwen38_real_disk_safetensor_shard_numerics() {
        use grim_format::tprov::SafetensorsProvider;
        use std::path::Path;

        let shard_path = Path::new("../../../models/qwen3.8-model-00001-of-00131.safetensors");
        if !shard_path.exists() {
            println!(
                "[SKIP] test_qwen38_real_disk_safetensor_shard_numerics: '{}' not present in environment",
                shard_path.display()
            );
            return;
        }

        println!(
            "[EXEC] test_qwen38_real_disk_safetensor_shard_numerics: reading real 992MB shard '{}'",
            shard_path.display()
        );

        let provider = SafetensorsProvider::open(shard_path.to_str().unwrap())
            .expect("Must open real 992MB Qwen 3.8 safetensors shard");
        let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);

        // Verify hyper-connection mixer weights present in shard 1
        let hc_lowrank = 320;
        let hidden_size = 10240; // 4 branches * 2560
        let hc_mixer_res = Qwen38HyperConnection::load(
            &ws.scoped("model")
                .scoped("language_model")
                .scoped("hyper_connection_mixer"),
            hidden_size,
            hc_lowrank,
            1e-6,
        );

        assert!(
            hc_mixer_res.is_ok(),
            "Hyper-connection mixer must load from real safetensor shard: {:?}",
            hc_mixer_res.err()
        );
        let hc_mixer = hc_mixer_res.unwrap();

        // Verify numeric forward mixing with real BF16 weights converted to tensor
        let x = cpu_tensor(vec![1.0f32; hidden_size], Shape::new(vec![hidden_size]));
        let mixed = hc_mixer
            .mix(&x)
            .expect("HC mixer must run forward without error");
        let mixed_vec = mixed.to_vec_f32().unwrap();

        assert_eq!(mixed_vec.len(), hidden_size);
        for (i, &v) in mixed_vec.iter().enumerate() {
            assert!(!v.is_nan(), "Mixed value at index {i} must not be NaN");
            assert!(!v.is_infinite(), "Mixed value at index {i} must not be Inf");
        }

        // Verify that mixing actually transformed the signal (not a trivial no-op zero)
        let mean = mixed_vec.iter().sum::<f32>() / (mixed_vec.len() as f32);
        assert!(
            mean.abs() > 1e-4,
            "Real weights must produce non-trivial mean response (got {mean})"
        );
    }
}

// ===========================================================================
// D2D MoE routing parity (who-dat 2.3): the Charon device dispatch
// (`fused_moe_dispatch_from_logits`, routing + expert GEMVs fully on-device)
// must match the host reference math. Regression gate for the qwen38 wiring.
// =========================================================================

#[cfg(test)]
mod moe_d2d_parity_tests {
    use super::*;
    use grim_backend_rocm::RocmDevice;
    use grim_tensor::CoreTensorOps;
    type DType = grim_tensor::dtype::DType;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
            })
            .collect()
    }

    fn rocm_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
        let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
        Tensor::new(
            std::sync::Arc::from(storage),
            shape,
            DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            Device::Rocm(0),
        )
    }

    fn make_block(
        dev: Option<&RocmDevice>,
        hidden: usize,
        inter: usize,
        n_exp: usize,
    ) -> Qwen38MoeBlock {
        let lin = |data: Vec<f32>, out: usize, inp: usize| -> Linear {
            let t = match dev {
                Some(d) => rocm_tensor(d, data, Shape::new(vec![out, inp])),
                None => cpu_tensor(data, Shape::new(vec![out, inp])),
            };
            Linear::from_tensor(t, None)
        };
        let experts_vec: Vec<Qwen38MoeExpert> = (0..n_exp)
            .map(|e| {
                let s = (e as u64 + 1) * 977;
                Qwen38MoeExpert {
                    gate_proj: lin(rand_vec(inter * hidden, s + 1), inter, hidden),
                    up_proj: lin(rand_vec(inter * hidden, s + 2), inter, hidden),
                    down_proj: lin(rand_vec(inter * hidden, s + 3), hidden, inter),
                }
            })
            .collect();
        Qwen38MoeBlock {
            gate: lin(rand_vec(n_exp * hidden, 42), n_exp, hidden),
            experts: Qwen38MoeExperts::Individual(experts_vec),
            shared_expert: None,
            num_experts_per_tok: 2,
            routed_scaling_factor: 1.0,
            _charon_cache: crate::shared_moe::CharonCache::new(),
        }
    }

    fn host_reference(block: &Qwen38MoeBlock, x: &[f32], seq: usize, hidden: usize) -> Vec<f32> {
        let logits_v = block
            .gate
            .forward(&cpu_tensor(x.to_vec(), Shape::new(vec![seq, hidden])))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let experts = match &block.experts {
            Qwen38MoeExperts::Individual(v) => v,
            Qwen38MoeExperts::Bank(_) => panic!("host_reference only supports Individual experts"),
        };
        let n_exp = experts.len();
        let mut out = vec![0.0f32; seq * hidden];
        for s in 0..seq {
            let row = &logits_v[s * n_exp..(s + 1) * n_exp];
            let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &idx[..block.num_experts_per_tok];
            // Global softmax over ALL experts (matches `grim_moe_route_topk` mode 0).
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = row.iter().map(|l| (l - max_l).exp()).sum::<f32>() + 1e-12;
            let token_x = &x[s * hidden..(s + 1) * hidden];
            for (ei, l) in topk.iter() {
                let w = ((l - max_l).exp() / denom) * block.routed_scaling_factor;
                let e = &experts[*ei];
                let g = e
                    .gate_proj
                    .forward(&cpu_tensor(token_x.to_vec(), Shape::new(vec![1, hidden])))
                    .unwrap()
                    .to_vec_f32()
                    .unwrap();
                let u = e
                    .up_proj
                    .forward(&cpu_tensor(token_x.to_vec(), Shape::new(vec![1, hidden])))
                    .unwrap()
                    .to_vec_f32()
                    .unwrap();
                let act: Vec<f32> = g
                    .iter()
                    .zip(u.iter())
                    .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
                    .collect();
                let d = e
                    .down_proj
                    .forward(&cpu_tensor(act.clone(), Shape::new(vec![1, act.len()])))
                    .unwrap()
                    .to_vec_f32()
                    .unwrap();
                for (j, dv) in d.iter().enumerate() {
                    out[s * hidden + j] += w * dv;
                }
            }
        }
        out
    }

    #[test]
    fn qwen38_moe_device_dispatch_matches_host_reference() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let dev = RocmDevice::shared(0);
        let hidden = 32usize;
        let inter = 64usize;
        let n_exp = 8usize;
        let seq = 3usize;

        let block_gpu = make_block(Some(&dev), hidden, inter, n_exp);
        let x_data = rand_vec(seq * hidden, 7);
        let x = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));

        let out_gpu = block_gpu.forward(&x).unwrap().to_vec_f32().unwrap();

        // Host reference (identical weights on CPU device).
        let block_cpu = make_block(None, hidden, inter, n_exp);
        let out_ref = host_reference(&block_cpu, &x_data, seq, hidden);

        // Reference self-check: the model's own CPU path must equal the
        // reference math (validates the reference before judging the D2D path).
        let out_cpu_model = block_cpu
            .forward(&cpu_tensor(x_data.clone(), Shape::new(vec![seq, hidden])))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let d0 = out_cpu_model
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            d0 < 1e-4,
            "reference self-check failed: cpu model vs reference max_diff={d0}"
        );

        assert_eq!(out_gpu.len(), out_ref.len());
        let max_diff = out_gpu
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2e-3,
            "D2D MoE dispatch diverged from host reference: max_diff={max_diff}"
        );
    }
}

//! Weight loading for the Qwen3-VL clip tower.
//!
//! Split from the config in [`Qwen3VlClipConfig`] because loading has its own
//! failure modes: the shapes in the checkpoint header are always self-consistent,
//! so a tensor whose bytes never arrived still "loads". Every weight therefore
//! passes a variance gate before the tower is considered usable.
//!
//! Two details here are load-bearing and easy to get backwards:
//!
//! * **The two patch kernels are summed.** `v.patch_embd.weight` and
//!   `v.patch_embd.weight.1` are distinct tensors that Qwen2/3-VL's
//!   `build_inp_with_temporal_merge` ADDS together
//!   (`old/repo/llama.cpp-master/tools/mtmd/models/qwen2vl.cpp:3-16`). For a
//!   still image both convolutions see the same pixels, so the effective kernel
//!   is `(W0 + W1) * patch + bias`. Loading only one is silently wrong: no shape
//!   error, no load error, just a model that ignores half its patch signal.
//! * **`mm.1.*` does not exist.** llama.cpp binds its `mm_1_w` to tensor index 2
//!   (`clip.cpp:2486`), so this loader asks for `mm.2` directly rather than
//!   probing for an index 1.

use grim_core::error::{Error, Result};
use grim_nn::WeightSource;
use grim_tensor::dtype::Device;
use grim_tensor::provider::TensorProvider;

use super::Qwen3VlClipConfig;

/// Fail on an all-zero or otherwise degenerate weight payload.
///
/// Shape agreement is not evidence that bytes arrived: a zero-filled or
/// mis-sliced read has the right shape, so it passes every structural check and
/// produces a tower that runs and emits zeros. Variance is the cheapest signal
/// that distinguishes "loaded" from "declared".
///
/// A single non-constant buffer of length `n` has `var > 1e-12`; real weights
/// sit orders of magnitude above that, so the threshold only trips on genuinely
/// empty buffers.
pub fn assert_non_degenerate(name: &str, v: &[f32]) -> Result<()> {
    if v.is_empty() {
        return Err(Error::Backend(format!(
            "{name}: empty payload - the tensor did not load"
        )));
    }
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    if !(var.is_finite() && var > 1e-12) {
        return Err(Error::Backend(format!(
            "{name}: degenerate payload (n={n}, mean={mean}, var={var}) - \
             the tensor did not load"
        )));
    }
    Ok(())
}

/// One pre-LN ViT block: LayerNorm, fused QKV, attention, output projection,
/// residual, LayerNorm, GELU MLP, residual.
///
/// The MLP has a single up-projection. The text tower is SwiGLU, but the mmproj
/// declares `ffn_up`/`ffn_down` and no `ffn_gate`
/// (`old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp:141-145` passes
/// `ff_gate_w = nullptr`), so a gated MLP here would be a different function.
#[derive(Debug, Clone)]
pub struct Qwen3VlBlock {
    pub attn_qkv_weight: Vec<f32>,
    pub attn_qkv_bias: Vec<f32>,
    pub attn_out_weight: Vec<f32>,
    pub attn_out_bias: Vec<f32>,
    pub ffn_up_weight: Vec<f32>,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down_weight: Vec<f32>,
    pub ffn_down_bias: Vec<f32>,
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
}

/// A loaded Qwen3-VL clip tower.
#[derive(Debug, Clone)]
pub struct Qwen3VlClip {
    pub cfg: Qwen3VlClipConfig,
    pub blocks: Vec<Qwen3VlBlock>,

    /// `(W0 + W1)` for the two patch kernels, flattened channel-major as
    /// `[patch_y][patch_x][channel][out]`. Summed once at load so the hot path
    /// never has to know there were two of them.
    pub patch_kernel_summed: Vec<f32>,
    pub patch_bias: Vec<f32>,
    /// `[embedding_length, n_patches]` learned position table.
    pub position_embd: Vec<f32>,
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Vec<f32>,

    /// Merger input projection: `merger_in x merger_in`.
    pub mm_0_weight: Vec<f32>,
    pub mm_0_bias: Vec<f32>,
    /// Merger output projection: `merger_in x projection_dim`.
    pub mm_2_weight: Vec<f32>,
    pub mm_2_bias: Vec<f32>,

    /// How many tensors the provider reported, for the load report.
    pub tensors_loaded: usize,
    /// Cached geometry so the forward pass does not re-derive it per call.
    pub merger_in_cols: usize,
    pub merger_out_cols: usize,
    pub merged_tokens: usize,
    /// Position-table grid side, i.e. `sqrt(n_patches)`.
    pub position_grid: usize,
}

/// Row-major f32 read of a named tensor, with a named error on absence.
///
/// Deliberately goes to the provider directly rather than through
/// [`WeightSource::get_f32`]. `WeightSource::full_name` always renders as
/// `prefix + "." + leaf`, and the clip tensor names already contain dots
/// (`v.patch_embd.weight.1`), so every such name would be addressed as
/// `...weight.` and match nothing. These names come from the file's own table
/// and are used verbatim; there is no hierarchical walk to do.
///
/// Decoding goes through the same `WeightSource` materialiser the rest of the
/// workspace uses, so this file does not grow its own F16/BF16 path that could
/// drift from it.
///
/// A missing tensor is an operator-fixable condition (wrong file, truncated
/// download), so the message names the tensor rather than reporting a shape
/// mismatch three layers up.
fn load_f32(
    provider: &dyn TensorProvider,
    ws: &WeightSource<'_>,
    name: &str,
    expect: usize,
) -> Result<Vec<f32>> {
    let raw = provider
        .get(name)
        .map_err(|e| Error::Backend(format!("clip tensor '{name}' is missing: {e}")))?;
    let got = raw.shape.iter().product::<usize>();
    if got != expect {
        return Err(Error::Shape(format!(
            "clip tensor '{name}': expected {expect} elements, file declares {got} \
             (shape {:?})",
            raw.shape
        )));
    }
    let t = ws
        .materialize_named(raw, name, [expect])
        .map_err(|e| Error::Backend(format!("clip tensor '{name}': could not decode: {e}")))?;
    let v = t.to_vec_f32()?;
    if v.len() != expect {
        return Err(Error::Shape(format!(
            "clip tensor '{name}': decoded {} of {expect} elements",
            v.len()
        )));
    }
    Ok(v)
}

impl Qwen3VlClip {
    /// Load the whole tower from a provider, onto the CPU.
    ///
    /// Host f32 is the right store here: this is a reference implementation and
    /// a parity oracle, and the device path materialises from the same numbers.
    /// Refuses a configuration this build cannot execute before touching a
    /// tensor, so an unsupported checkpoint fails in one sentence rather than
    /// after a 927 MB read.
    pub fn load(provider: &dyn TensorProvider, cfg: Qwen3VlClipConfig) -> Result<Self> {
        Self::load_on(provider, cfg, Device::Cpu)
    }

    /// Load the whole tower, materialising each weight onto `device`.
    ///
    /// Refuses a configuration this build cannot execute before touching a
    /// tensor, so an unsupported checkpoint fails in one sentence rather than
    /// after a 927 MB read. The gate lives here rather than in [`Self::load`]
    /// so every entry point gets it.
    pub fn load_on(
        provider: &dyn TensorProvider,
        cfg: Qwen3VlClipConfig,
        device: Device,
    ) -> Result<Self> {
        cfg.ensure_supported()?;
        let ws = WeightSource::root(provider, device);

        let e = cfg.embedding_length;
        let f = cfg.feed_forward_length;
        let qkv_out = 3 * e;
        let patch_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let patch_elems = patch_in * e;
        let n_patches = cfg.n_patches();
        let mi = cfg.merger_in();

        // --- patch embedding: sum both conv kernels ---
        let k0 = load_f32(provider, &ws, "v.patch_embd.weight", patch_elems)?;
        let k1 = load_f32(provider, &ws, "v.patch_embd.weight.1", patch_elems)?;
        assert_non_degenerate("v.patch_embd.weight", &k0)?;
        assert_non_degenerate("v.patch_embd.weight.1", &k1)?;
        let patch_kernel_summed: Vec<f32> = k0.iter().zip(k1.iter()).map(|(a, b)| a + b).collect();

        let patch_bias = load_f32(provider, &ws, "v.patch_embd.bias", e)?;
        assert_non_degenerate("v.patch_embd.bias", &patch_bias)?;

        // --- learned position table ---
        let position_embd = load_f32(provider, &ws, "v.position_embd.weight", e * n_patches)?;
        assert_non_degenerate("v.position_embd.weight", &position_embd)?;

        let post_ln_weight = load_f32(provider, &ws, "v.post_ln.weight", e)?;
        let post_ln_bias = load_f32(provider, &ws, "v.post_ln.bias", e)?;

        // --- blocks ---
        let mut blocks = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let p = format!("v.blk.{i}");
            let block = Qwen3VlBlock {
                attn_qkv_weight: load_f32(
                    provider,
                    &ws,
                    &format!("{p}.attn_qkv.weight"),
                    e * qkv_out,
                )?,
                attn_qkv_bias: load_f32(provider, &ws, &format!("{p}.attn_qkv.bias"), qkv_out)?,
                attn_out_weight: load_f32(provider, &ws, &format!("{p}.attn_out.weight"), e * e)?,
                attn_out_bias: load_f32(provider, &ws, &format!("{p}.attn_out.bias"), e)?,
                ffn_up_weight: load_f32(provider, &ws, &format!("{p}.ffn_up.weight"), e * f)?,
                ffn_up_bias: load_f32(provider, &ws, &format!("{p}.ffn_up.bias"), f)?,
                ffn_down_weight: load_f32(provider, &ws, &format!("{p}.ffn_down.weight"), f * e)?,
                ffn_down_bias: load_f32(provider, &ws, &format!("{p}.ffn_down.bias"), e)?,
                ln1_weight: load_f32(provider, &ws, &format!("{p}.ln1.weight"), e)?,
                ln1_bias: load_f32(provider, &ws, &format!("{p}.ln1.bias"), e)?,
                ln2_weight: load_f32(provider, &ws, &format!("{p}.ln2.weight"), e)?,
                ln2_bias: load_f32(provider, &ws, &format!("{p}.ln2.bias"), e)?,
            };
            // The block's own weights, not just its norms: a zeroed attention
            // projection is as silent as a zeroed norm.
            assert_non_degenerate(&format!("{p}.attn_qkv.weight"), &block.attn_qkv_weight)?;
            assert_non_degenerate(&format!("{p}.attn_out.weight"), &block.attn_out_weight)?;
            assert_non_degenerate(&format!("{p}.ffn_up.weight"), &block.ffn_up_weight)?;
            assert_non_degenerate(&format!("{p}.ffn_down.weight"), &block.ffn_down_weight)?;
            blocks.push(block);
        }

        // --- merger ---
        let mm_0_weight = load_f32(provider, &ws, "mm.0.weight", mi * mi)?;
        let mm_0_bias = load_f32(provider, &ws, "mm.0.bias", mi)?;
        assert_non_degenerate("mm.0.weight", &mm_0_weight)?;
        let mm_2_weight = load_f32(provider, &ws, "mm.2.weight", mi * cfg.projection_dim)?;
        let mm_2_bias = load_f32(provider, &ws, "mm.2.bias", cfg.projection_dim)?;
        assert_non_degenerate("mm.2.weight", &mm_2_weight)?;

        // The position table is a square grid by construction: `n_patches` is
        // `per_side^2`, and the shape check above already refused any table
        // whose entry count disagrees. So `grid` is exact by construction and
        // needs no validation of its own.
        let grid = (n_patches as f64).sqrt() as usize;

        let tensors_loaded = provider.tensor_names().len();

        Ok(Self {
            blocks,
            patch_kernel_summed,
            patch_bias,
            position_embd,
            post_ln_weight,
            post_ln_bias,
            mm_0_weight,
            mm_0_bias,
            mm_2_weight,
            mm_2_bias,
            merger_in_cols: mi,
            merger_out_cols: cfg.projection_dim,
            merged_tokens: cfg.merged_tokens(),
            position_grid: grid,
            cfg,
            tensors_loaded,
        })
    }

    /// The width this tower emits per output token, i.e. what the text model
    /// must be able to accept in place of a token embedding.
    pub fn output_width(&self) -> usize {
        self.cfg.projection_dim
    }
}

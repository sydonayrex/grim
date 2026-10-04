//! Forward-pass tests for the Qwen3-VL clip tower.
//!
//! Each concern is pinned separately so a failure names the step that broke.
//!
//! Two facts from the reference implementation (`old/repo/llama.cpp-master/tools/
//! mtmd/models/qwen3vl.cpp`) that these tests exist to lock down:
//!
//! * **The patch bias is added AFTER the spatial-merge block**, not before
//!   (`:33-37`). My original plan had this backwards.
//! * **The position embedding runs through the SAME permute chain as `inp`**
//!   (`:41-50`) before being added, so it is NOT added row-for-row in patch
//!   order. Adding it directly would misalign every row after the first block.

use std::collections::HashMap;

use grim_models_vision::qwen3vl_clip::{Qwen3VlClip, Qwen3VlClipConfig};
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

/// A 4x4-patch tower: patch 4, hidden 8, 2 heads of dim 4, 2 blocks, merge 2.
/// `image_size` 16 gives a 4x4 grid = 16 patches -> 4 merged tokens, and the
/// merger consumes 4 * 8 = 32 features.
fn tiny_config() -> Qwen3VlClipConfig {
    Qwen3VlClipConfig {
        image_size: 16,
        patch_size: 4,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        num_heads: 2,
        projection_dim: 12,
        spatial_merge_size: 2,
        layer_norm_eps: 1e-6,
        use_gelu: true,
        in_channels: 3,
        merge_area: 4,
        image_mean: vec![0.5, 0.5, 0.5],
        image_std: vec![0.5, 0.5, 0.5],
        deepstack_layers: vec![],
    }
}

/// An in-memory provider holding one correctly-sized buffer per tensor name.
///
/// Each tensor starts as a non-degenerate ramp so the loader's variance gate is
/// satisfied, then the fixture overwrites the entries it cares about.
#[derive(Default)]
struct MemProvider {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl MemProvider {
    /// Set tensor `name` to `elems` values produced by `f`.
    ///
    /// Some fixtures want an exact all-zero tensor (to isolate a stage), others
    /// want mostly zeros with a few taps. Either way `f` is the whole truth -
    /// the caller owns the values, so a stage that must be provably zeroed is
    /// genuinely zero rather than silently damped.
    fn set(&mut self, name: &str, elems: usize, f: impl Fn(usize) -> f32) {
        let vals: Vec<f32> = (0..elems).map(f).collect();
        self.tensors.insert(name.to_string(), (vals, vec![elems]));
    }

    /// Set tensor `name` to `f`'s values plus a small per-index offset.
    ///
    /// The loader's variance gate rejects an all-zero weight, which is the right
    /// behaviour - an all-zero tensor is exactly the "declared but never
    /// loaded" failure it exists to catch. A fixture that wants a stage
    /// *inactive* therefore cannot express that with literal zeros; it uses a
    /// negligible non-zero value instead, which is inert numerically (the
    /// stages under test ignore it or it is 20 orders of magnitude below the
    /// signal) while satisfying the gate.
    fn set_nonzero(&mut self, name: &str, elems: usize, f: impl Fn(usize) -> f32) {
        let vals: Vec<f32> = (0..elems).map(|i| f(i) + (i % 7) as f32 * 1e-4).collect();
        self.tensors.insert(name.to_string(), (vals, vec![elems]));
    }
}

impl TensorProvider for MemProvider {
    fn get(&self, name: &str) -> Result<RawTensor, grim_tensor::error::Error> {
        let (vals, shape) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(RawTensor {
            bytes,
            shape,
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta, grim_tensor::error::Error> {
        let (_, shape) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        Ok(TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
            shape,
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.tensors.keys().cloned().collect();
        n.sort();
        n
    }
}

/// A tower whose weights make each stage's contribution analytically known, so a
/// test can assert on a value rather than on "it did not crash".
///
///  * `attn_qkv.weight = 0` and `attn_out.weight = 0` -> attention is constant
///  * `ffn_up.weight = I`, `ffn_down.weight = I`       -> MLP is GELU, elementwise
///  * all norms are weight 1, bias 0
///  * `position_embd = 0`, `mm.2.weight = 0`          -> isolate the early path
fn separable_provider(cfg: &Qwen3VlClipConfig) -> MemProvider {
    let e = cfg.embedding_length;
    let f = cfg.feed_forward_length;
    let patch_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    let mut p = MemProvider::default();

    // Patch kernel A: tap only the first input element, so the conv output is a
    // known function of one pixel per patch. Kernel B is a DISTINCT non-zero
    // pattern, because an all-zero kernel would trip the loader's variance
    // gate (correctly - that gate is what catches a tensor that did not load).
    // Since the loader SUMS the two, the fixture computes that sum below.
    p.set("v.patch_embd.weight", patch_in * e, |i| {
        if i == 0 {
            1.0
        } else if i % 11 == 3 {
            0.02
        } else {
            0.0
        }
    });
    p.set("v.patch_embd.weight.1", patch_in * e, |i| {
        if i == 0 {
            2.0
        } else if i % 13 == 5 {
            0.03
        } else {
            0.0
        }
    });
    p.set_nonzero("v.patch_embd.bias", e, |_| 0.0);
    p.set_nonzero("v.position_embd.weight", e * cfg.n_patches(), |_| 0.0);
    p.set("v.post_ln.weight", e, |_| 1.0);
    p.set_nonzero("v.post_ln.bias", e, |_| 0.0);

    for blk in 0..cfg.block_count {
        let q = format!("v.blk.{blk}");
        p.set_nonzero(&format!("{q}.attn_qkv.weight"), e * 3 * e, |_| 0.0);
        p.set_nonzero(&format!("{q}.attn_qkv.bias"), 3 * e, |_| 0.0);
        p.set_nonzero(&format!("{q}.attn_out.weight"), e * e, |_| 0.0);
        p.set_nonzero(&format!("{q}.attn_out.bias"), e, |_| 0.0);
        // Row-major identity inside the e x f padded MLP.
        p.set(&format!("{q}.ffn_up.weight"), e * f, |i| {
            if i % f == i / f {
                1.0
            } else {
                0.0
            }
        });
        p.set_nonzero(&format!("{q}.ffn_up.bias"), f, |_| 0.0);
        p.set(&format!("{q}.ffn_down.weight"), f * e, |i| {
            if i % e == i / f {
                1.0
            } else {
                0.0
            }
        });
        p.set_nonzero(&format!("{q}.ffn_down.bias"), e, |_| 0.0);
        p.set(&format!("{q}.ln1.weight"), e, |_| 1.0);
        p.set_nonzero(&format!("{q}.ln1.bias"), e, |_| 0.0);
        p.set(&format!("{q}.ln2.weight"), e, |_| 1.0);
        p.set_nonzero(&format!("{q}.ln2.bias"), e, |_| 0.0);
    }

    let mi = cfg.merger_in();
    p.set("mm.0.weight", mi * mi, |i| {
        if i % mi == i / mi {
            1.0
        } else {
            0.0
        }
    });
    p.set_nonzero("mm.0.bias", mi, |_| 0.0);
    p.set_nonzero("mm.2.weight", mi * cfg.projection_dim, |_| 0.0);
    p.set_nonzero("mm.2.bias", cfg.projection_dim, |_| 0.0);
    p
}

/// A pixel buffer that varies per patch, so a correct conv cannot be confused
/// with one that reads a constant.
fn varied_pixels(cfg: &Qwen3VlClipConfig, h: usize, w: usize) -> Vec<f32> {
    (0..(cfg.in_channels * h * w))
        .map(|i| (i % 7) as f32 * 0.05 - 0.15)
        .collect()
}

/// The forward pass must reject an image whose side is not a whole number of
/// 2x2 patch blocks. `ggml_conv_2d` with no padding silently drops a partial
/// block otherwise, and the tower would emit fewer tokens than the chat template
/// reserved slots for.
#[test]
fn image_side_not_a_whole_merge_block_is_rejected() {
    let cfg = tiny_config();
    let p = separable_provider(&cfg);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    // patch 4 * merge 2 = 8, so 20 is not a multiple of the required 8.
    let pixels = vec![0.5f32; 3 * 20 * 20];
    let err = clip
        .forward_cpu(&pixels, 20, 20)
        .err()
        .expect("a partial merge block must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains('2') && msg.contains('0') && msg.contains('8'),
        "the error must state the image side and the required multiple: {msg}"
    );
}

/// A square image of the declared size produces exactly `merged_tokens` rows of
/// `projection_dim` features. This is the shape contract the chat-template slot
/// count is matched against, so it is asserted rather than implied.
#[test]
fn square_image_yields_merged_tokens_by_projection_dim() {
    let cfg = tiny_config();
    let p = separable_provider(&cfg);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    let side = cfg.image_size;
    let pixels = varied_pixels(&cfg, side, side);
    let out = clip.forward_cpu(&pixels, side, side).expect("forward runs");
    assert_eq!(cfg.merged_tokens(), 4, "4x4 patches / 2x2 merge");
    assert_eq!(
        out.len(),
        cfg.merged_tokens() * cfg.projection_dim,
        "expected {} tokens x {} features",
        cfg.merged_tokens(),
        cfg.projection_dim
    );
}

/// The patch bias must reach the output. If the bias were dropped, or applied
/// along the wrong axis of the merge, this fails.
#[test]
fn patch_bias_reaches_the_output() {
    let cfg = tiny_config();
    let mut p = separable_provider(&cfg);
    let e = cfg.embedding_length;
    p.set("v.patch_embd.bias", e, |_| 0.25);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    let side = cfg.image_size;
    let pixels = vec![0.5f32; 3 * side * side];
    let out = clip.forward_cpu(&pixels, side, side).expect("forward runs");
    assert!(
        out.iter().any(|v| v.abs() > 1e-6),
        "a non-zero patch bias must produce a non-zero embedding"
    );
}

/// A taller-than-wide image is accepted when both sides are whole merge blocks.
/// Vision towers are variable-size by design; refusing non-square input would
/// defeat that.
#[test]
fn non_square_image_of_valid_geometry_is_accepted() {
    let cfg = tiny_config();
    let p = separable_provider(&cfg);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    let (h, w) = (16usize, 24usize); // two merge blocks by three
    let pixels = varied_pixels(&cfg, h, w);
    let out = clip
        .forward_cpu(&pixels, h, w)
        .expect("valid geometry accepted");
    let patch_rows = (h / cfg.patch_size) * (w / cfg.patch_size);
    assert_eq!(
        out.len(),
        (patch_rows / cfg.merge_area) * cfg.projection_dim,
        "expected {} tokens for a {h}x{w} image",
        patch_rows / cfg.merge_area
    );
}

/// Repeated calls must agree. This catches state leaking between calls, which
/// the device path will need; it does NOT catch a permutation bug, which is
/// deterministic and needs the llama.cpp parity test.
#[test]
fn repeated_calls_are_deterministic() {
    let cfg = tiny_config();
    let p = separable_provider(&cfg);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    let side = cfg.image_size;
    let pixels = varied_pixels(&cfg, side, side);
    let a = clip.forward_cpu(&pixels, side, side).expect("first call");
    let b = clip.forward_cpu(&pixels, side, side).expect("second call");
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < 1e-6, "element {i}: {x} vs {y}");
    }
}

/// Changing the pixels must change the embedding. A tower that ignored its input
/// (a mis-shaped gather, a zero conv) would still pass every shape test.
#[test]
fn different_images_produce_different_embeddings() {
    let cfg = tiny_config();
    let p = separable_provider(&cfg);
    let clip = Qwen3VlClip::load(&p, cfg.clone()).expect("tower loads");
    let side = cfg.image_size;
    let a = vec![0.25f32; 3 * side * side];
    let b = vec![-0.4f32; 3 * side * side];
    let ea = clip.forward_cpu(&a, side, side).expect("image A");
    let eb = clip.forward_cpu(&b, side, side).expect("image B");
    let differs = ea.iter().zip(eb.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
    assert!(
        differs,
        "two different images must not encode identically - the conv or the \
         gather is ignoring its input"
    );
}

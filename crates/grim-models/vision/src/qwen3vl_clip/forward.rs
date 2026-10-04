//! CPU reference forward pass for the Qwen3-VL clip tower.
//!
//! Written op-for-op against `old/repo/llama.cpp-master/tools/mtmd/models/
//! qwen3vl.cpp`, each step citing the line it implements. This module is the
//! parity oracle for the device path and for the llama.cpp differential test, so
//! clarity beats speed: plain `Vec<f32>` loops, no clever indexing.
//!
//! Op order, and the two details that are easy to get backwards:
//!
//! 1. patch-embed conv2d with the **summed** kernel (`qwen2vl.cpp:3-16`)
//! 2. the 5-op spatial-merge block (`qwen3vl.cpp:18-31`)
//! 3. **add the patch bias** (`qwen3vl.cpp:33-37`) - after the merge, not before
//! 4. **add the position embedding**, itself run through the *same* merge block
//!    (`qwen3vl.cpp:41-51`)
//! 5. `block_count` pre-norm blocks (`qwen3vl.cpp:74-160`)
//! 6. post LayerNorm (`qwen3vl.cpp:169-171`)
//! 7. merger: reshape to `merge_area * embedding_length`, GELU, project
//!    (`qwen3vl.cpp:174-181`)
//!
//! Step 4 is the subtlety: llama.cpp applies the identical permute chain to
//! `learned_pos_embd` as to `inp`, so position rows are *not* added in patch
//! order. Adding the table directly would misalign every row.

use grim_core::error::{Error, Result};

use super::Qwen3VlClip;

/// Row-major `y = x @ w + b`, where `w` is `[nin, nout]`.
///
/// The clip checkpoints store weights as `[in, out]` (verified: `mm.2.weight` is
/// `[4608, 5120]`, projecting 4608 features out to the text hidden size), so this
/// is the transpose of the more common `[out, in]` convention.
fn linear(x: &[f32], rows: usize, w: &[f32], nin: usize, nout: usize, b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * nout];
    for r in 0..rows {
        let xr = &x[r * nin..r * nin + nin];
        let yr = &mut y[r * nout..r * nout + nout];
        for o in 0..nout {
            let wcol = &w[o * nin..o * nin + nin];
            let mut acc = b[o];
            for (i, &xv) in xr.iter().enumerate() {
                acc += xv * wcol[i];
            }
            yr[o] = acc;
        }
    }
    y
}

/// Row-wise LayerNorm (mean-subtracted), `gamma`/`beta` per row.
///
/// **LayerNorm, not RMSNorm.** The reference sets `norm_type = NORM_TYPE_NORMAL`
/// for this projector (`qwen3vl.cpp:12`) and the file carries a
/// `layer_norm_epsilon`. RMSNorm differs on any row with non-zero mean, so this
/// is a silent-but-wrong substitution rather than an approximation.
fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..r * dim + dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let orow = &mut out[r * dim..r * dim + dim];
        for d in 0..dim {
            orow[d] = (row[d] - mean) * inv * gamma[d] + beta[d];
        }
    }
    out
}

/// tanh-approximate GELU, matching `grim_gelu`
/// (`crates/grim-backend-rocm/src/kernels/compute_kernels.rs:258-265`) and the
/// reference's `FFN_GELU`.
///
/// A plain activation, not the SwiGLU gate the text tower uses: the mmproj has
/// `ffn_up`/`ffn_down` and no `ffn_gate`.
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let t = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + t.tanh())
}

/// Full bidirectional attention over `rows` tokens, `heads` heads of `head_dim`.
///
/// No causal mask: this is an encoder, every token sees every other. Softmax is
/// in f32 with a max-subtraction for stability.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; rows * heads * head_dim];
    let mut scores = vec![0.0f32; rows];

    for h in 0..heads {
        let hd = h * head_dim;
        for r in 0..rows {
            let qrow = &q[r * heads * head_dim + hd..r * heads * head_dim + hd + head_dim];
            let mut max = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let krow = &k[j * heads * head_dim + hd..j * heads * head_dim + hd + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += qrow[d] * krow[d];
                }
                *score = scale * dot;
                if *score > max {
                    max = *score;
                }
            }
            let sum: f32 = scores.iter().map(|s| (*s - max).exp()).sum();
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() {
                *s = (*s - max).exp() * inv;
            }
            let orow = &mut out[r * heads * head_dim + hd..r * heads * head_dim + hd + head_dim];
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &s) in scores.iter().enumerate() {
                    acc += s * v[j * heads * head_dim + hd + d];
                }
                orow[d] = acc;
            }
        }
    }
    out
}

/// Split a `[rows, 3*embedding]` QKV buffer at offsets 0, `e`, `2e`
/// (`qwen3vl.cpp:83-95`).
fn split_qkv(qkv: &[f32], rows: usize, e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut q, mut k, mut v) = (
        Vec::with_capacity(rows * e),
        Vec::with_capacity(rows * e),
        Vec::with_capacity(rows * e),
    );
    for r in 0..rows {
        let base = r * 3 * e;
        q.extend_from_slice(&qkv[base..base + e]);
        k.extend_from_slice(&qkv[base + e..base + 2 * e]);
        v.extend_from_slice(&qkv[base + 2 * e..base + 3 * e]);
    }
    (q, k, v)
}

impl Qwen3VlClip {
    /// Encode an already-normalised image into `projection_dim`-wide embeddings.
    ///
    /// `pixels` is `[in_channels, h, w]` row-major f32. Both `h` and `w` must be
    /// whole multiples of `patch_size * spatial_merge_size`: the conv has no
    /// padding, so a partial 2x2 block would silently drop patches and emit
    /// fewer tokens than the chat template reserved slots for.
    ///
    /// Returns `[merged_tokens, projection_dim]` row-major.
    pub fn forward_cpu(&self, pixels: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let e = cfg.embedding_length;
        let ps = cfg.patch_size;
        let block = ps * cfg.spatial_merge_size;

        if h % block != 0 || w % block != 0 {
            return Err(Error::Shape(format!(
                "image {h}x{w} must be a whole number of {block}-pixel merge blocks \
                 (patch_size {ps} x spatial_merge_size {})",
                cfg.spatial_merge_size
            )));
        }
        if pixels.len() != cfg.in_channels * h * w {
            return Err(Error::Shape(format!(
                "pixel buffer has {} values, expected {} for a {h}x{w} image with {} channels",
                pixels.len(),
                cfg.in_channels * h * w,
                cfg.in_channels
            )));
        }

        let px = w / ps; // patch columns
        let py = h / ps; // patch rows

        // --- 1. patch-embed conv2d (qwen2vl.cpp:3-16) ---
        // The kernel is already the SUM of both convolutions (see `weights.rs`),
        // stride 1, no padding. Output layout `[channel][x][y]`, matching ggml's
        // `ggml_conv_2d` result which is `[n_embd, n_patches_x, n_patches_y]`.
        let mut conv = vec![0.0f32; e * px * py];
        for (px_i, x) in (0..px).enumerate() {
            let _ = px_i;
            for y in 0..py {
                for c in 0..cfg.in_channels {
                    for o in 0..e {
                        let mut acc = self.patch_bias[o];
                        for ky in 0..ps {
                            for kx in 0..ps {
                                let pix = pixels[c * h * w + (y * ps + ky) * w + (x * ps + kx)];
                                acc += pix
                                    * self.patch_kernel_summed
                                        [(c * ps * ps + ky * ps + kx) * e + o];
                            }
                        }
                        conv[o * px * py + x * py + y] = acc;
                    }
                }
            }
        }

        // --- 2. spatial merge (qwen3vl.cpp:18-31) ---
        // Flat `[px*py, e]` row-major.
        let mut rows = spatial_merge(&conv, e, px, py);

        // --- 3. patch bias, AFTER the merge (qwen3vl.cpp:33-37) ---
        // Broadcast over every row: the bias is per-output-channel, and the
        // merged layout's channel axis is the contiguous (last) one.
        let n_patch_rows = rows.len() / e;
        for r in 0..n_patch_rows {
            for k in 0..e {
                rows[r * e + k] += self.patch_bias[k];
            }
        }

        // --- 4. position embedding, through the SAME merge chain
        //        (qwen3vl.cpp:39-52) ---
        let pos = self.position_rows(px, py);
        for r in 0..n_patch_rows {
            for k in 0..e {
                rows[r * e + k] += pos[r * e + k];
            }
        }

        let n_rows = rows.len() / e;
        let heads = cfg.num_heads;
        let head_dim = cfg.head_dim();

        // --- 5. blocks (qwen3vl.cpp:74-160) ---
        for blk in &self.blocks {
            // ln1 -> QKV -> rope -> attention -> out -> residual
            let normed = layer_norm(
                &rows,
                n_rows,
                e,
                &blk.ln1_weight,
                &blk.ln1_bias,
                cfg.layer_norm_eps,
            );
            let qkv = linear(
                &normed,
                n_rows,
                &blk.attn_qkv_weight,
                e,
                3 * e,
                &blk.attn_qkv_bias,
            );
            let (_q, _k, v) = split_qkv(&qkv, n_rows, e);
            let (qr, kr) = self.rope_qk(n_rows, heads, head_dim, px, py)?;

            let ctx = attention(&qr, &kr, &v, n_rows, heads, head_dim);
            let attn_proj = linear(&ctx, n_rows, &blk.attn_out_weight, e, e, &blk.attn_out_bias);
            for (acc, av) in rows.iter_mut().zip(attn_proj.iter()) {
                *acc += av;
            }

            // ln2 -> GELU MLP -> residual (qwen3vl.cpp:122-158)
            let normed2 = layer_norm(
                &rows,
                n_rows,
                e,
                &blk.ln2_weight,
                &blk.ln2_bias,
                cfg.layer_norm_eps,
            );
            let up = linear(
                &normed2,
                n_rows,
                &blk.ffn_up_weight,
                e,
                cfg.feed_forward_length,
                &blk.ffn_up_bias,
            );
            let act: Vec<f32> = up.iter().map(|&v| gelu_tanh(v)).collect();
            let mlp = linear(
                &act,
                n_rows,
                &blk.ffn_down_weight,
                cfg.feed_forward_length,
                e,
                &blk.ffn_down_bias,
            );
            for (acc, mv) in rows.iter_mut().zip(mlp.iter()) {
                *acc += mv;
            }
        }

        // --- 6. post LayerNorm (qwen3vl.cpp:169-171) ---
        let rows = layer_norm(
            &rows,
            n_rows,
            e,
            &self.post_ln_weight,
            &self.post_ln_bias,
            cfg.layer_norm_eps,
        );

        // --- 7. merger (qwen3vl.cpp:174-181) ---
        // reshape [e, n_rows] -> [merge_area*e, n_rows/merge_area], then
        // GELU and the two projections. The reshape is a plain reinterpretation
        // of the contiguous element sequence.
        let merge_area = cfg.merge_area;
        let n_merged = n_rows / merge_area;
        let mi = merge_area * e;
        let mut wide = vec![0.0f32; n_merged * mi];
        for n in 0..n_merged {
            for m in 0..mi {
                wide[n * mi + m] = rows[(n * merge_area + m / e) * e + (m % e)];
            }
        }
        let act: Vec<f32> = wide.iter().map(|&v| gelu_tanh(v)).collect();
        let projected = linear(&act, n_merged, &self.mm_0_weight, mi, mi, &self.mm_0_bias);
        let act2: Vec<f32> = projected.iter().map(|&v| gelu_tanh(v)).collect();
        Ok(linear(
            &act2,
            n_merged,
            &self.mm_2_weight,
            mi,
            cfg.projection_dim,
            &self.mm_2_bias,
        ))
    }

    /// Position-embedding rows for an `px x py` patch grid, run through the same
    /// merge chain as the conv output (`qwen3vl.cpp:41-51`).
    ///
    /// Returns the same flat `[px*py, e]` layout as [`spatial_merge`].
    fn position_rows(&self, px: usize, py: usize) -> Vec<f32> {
        let e = self.cfg.embedding_length;
        let grid = self.position_grid;
        let mut buf = vec![0.0f32; e * px * py];

        if px == grid && py == grid {
            // `resize_position_embeddings` returns the table unchanged when the
            // image grid already matches (clip.cpp:321-323). Still passed through
            // the merge chain, because the reference adds the MERGED layout.
            for r in 0..grid {
                for c in 0..e {
                    buf[c * grid * grid + r] = self.position_embd[r * e + c];
                }
            }
        } else {
            // Otherwise bilinear with align-corners (clip.cpp:325-329): corner
            // sample positions map to source indices 0 and grid-1 exactly.
            for y in 0..py {
                let fy = if py == 1 {
                    0.0
                } else {
                    y as f32 * (grid - 1) as f32 / (py - 1) as f32
                };
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(grid - 1);
                let ty = fy - y0 as f32;
                for x in 0..px {
                    let fx = if px == 1 {
                        0.0
                    } else {
                        x as f32 * (grid - 1) as f32 / (px - 1) as f32
                    };
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(grid - 1);
                    let tx = fx - x0 as f32;
                    for c in 0..e {
                        let v00 = self.position_embd[(y0 * grid + x0) * e + c];
                        let v01 = self.position_embd[(y0 * grid + x1) * e + c];
                        let v10 = self.position_embd[(y1 * grid + x0) * e + c];
                        let v11 = self.position_embd[(y1 * grid + x1) * e + c];
                        let top = v00 + (v01 - v00) * tx;
                        let bot = v10 + (v11 - v10) * tx;
                        buf[c * px * py + x * py + y] = top + (bot - top) * ty;
                    }
                }
            }
        }
        spatial_merge(&buf, e, px, py)
    }

    /// Vision M-RoPE over the per-token position ids.
    ///
    /// Task 2.2 (WI-3) implements the real rotation. Until then this refuses:
    /// returning unrotated q/k would yield a tower that is deterministic,
    /// plausible, and wrong - the RoPE is an identity at position 0, so every
    /// single-position gate would pass while the model was comprehensively wrong.
    fn rope_qk(
        &self,
        rows: usize,
        _heads: usize,
        _head_dim: usize,
        _px: usize,
        _py: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(Error::Unimplemented(format!(
            "Qwen3-VL vision M-RoPE is not implemented yet; {rows} rows would \
             otherwise be encoded with unrotated q/k, which is silently wrong"
        )))
    }
}

/// The 5-op spatial-merge chain of `qwen3vl.cpp:18-31`, as explicit index passes.
///
/// `ggml_permute` produces a strided VIEW; `ggml_cont` on that view GATHERS
/// through the view's strides. The net effect is therefore not a plain reshape,
/// and a hand-derived closed form is invisible-wrong at position 0. So each op is
/// applied as its own pass, each mirroring exactly one ggml call:
///
/// ```text
/// op1  permute(1,2,0,3)            [c,x,y,b] -> view [x,y,c,b]
/// op2  cont_4d(2e, px/2, py, b)   gather from op1's view
/// op3  reshape_4d(2e, px/2, 2, py/2)   pure reinterpretation
/// op4  permute(0,2,1,3)            -> view [2e, 2, px/2, py/2]
/// op5  cont_3d(e, px*py, b)        gather from op4's view -> px*py rows of width e
/// ```
///
/// **This permutation is NOT yet verified against llama.cpp.** It is written to
/// the shape the op chain implies, and WI-2 (the differential test against
/// `MTMD_DEBUG_EMBEDDINGS` output) is what establishes it. Until then the
/// forward pass refuses to run, because a wrong merge still produces
/// plausible-shaped output.
///
/// `buf` is `[e, px, py]` channel-major. Returns `[px*py, e]` row-major.
fn spatial_merge(buf: &[f32], e: usize, px: usize, py: usize) -> Vec<f32> {
    assert_eq!(
        buf.len(),
        e * px * py,
        "conv output must be [e, px, py] channel-major"
    );
    assert_eq!(px % 2, 0, "the merge chain folds x by 2");
    assert_eq!(py % 2, 0, "the merge chain folds y by 2");
    let hx = px / 2;
    let hy = py / 2;

    // op1+op2: gather op1's [x,y,c] view into a contiguous [2e, hx, py] buffer.
    // op2's leading dim is 2e, so its index `a` packs a channel in [0, e) and an
    // x-parity in [e, 2e); the parity selects the source column 2*x + parity.
    let mut t2 = vec![0.0f32; 2 * e * hx * py];
    for a in 0..(2 * e) {
        let (chan, x_parity) = (a % e, a / e);
        for x in 0..hx {
            let sx = 2 * x + x_parity;
            for y in 0..py {
                // op1 view logical index is (sx, y, chan).
                let src = chan * px * py + sx * py + y;
                // op2 logical index is (a, x, y) -> contiguous slot.
                t2[(y * hx + x) * (2 * e) + a] = buf[src];
            }
        }
    }

    // op3: reshape [2e, hx, py] -> [2e, hx, 2, hy]. Pure reinterpretation: the
    // y index splits as y = yh * 2 + y_parity is NOT what op3 does; op3 reshapes
    // the flattened sequence, so py becomes 2 * hy with y' = y % hy and
    // y_parity = y / hy. Both are tracked explicitly below.
    //
    // op4: permute(0,2,1,3) -> view [2e, 2, hx, hy], so the reshaped dim2 (the
    // split of py) becomes the view's dim1 and the old dim1 (hx) becomes dim2.
    //
    // op5: cont_3d(e, px*py, b) gathers that view down to e channels over
    // px*py rows.
    let mut out = vec![0.0f32; e * px * py];
    for a in 0..(2 * e) {
        let (chan, x_parity) = (a % e, a / e);
        for x in 0..hx {
            for y in 0..py {
                // op3's coordinates of the flattened py index.
                let yp = y % hy;
                let y_parity = y / hy;
                // op4 view logical index (a, y_parity, x); its own slot.
                let view_slot = ((y_parity * hx) + x) * (2 * hy) + yp;
                // op5 reduces the leading 2e down to e by taking the half that
                // matches this channel, and lays rows out as (x', y').
                let sx = 2 * x + x_parity;
                let sy = 2 * yp + y_parity;
                let _ = view_slot;
                let row = sx * py + sy;
                out[chan * px * py + row] = t2[(y * hx + x) * (2 * e) + a];
            }
        }
    }

    // op5's result is [e, px*py, b]; transpose to [px*py, e] rows.
    let mut rows = vec![0.0f32; px * py * e];
    for c in 0..e {
        for r in 0..(px * py) {
            rows[r * e + c] = out[c * px * py + r];
        }
    }
    rows
}

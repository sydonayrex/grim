//! Shared attention entry point for transformer loaders.
//!
//! One canonical function replaces the ~25 per-model scalar CPU attention
//! loops (lfm2.rs-style). It first tries the fused device kernel
//! (`BackendDevice::qkv_attention`, which handles GQA, causal masking and
//! sliding windows on ROCm/CUDA/Metal/Vulkan/CPU), and falls back to the
//! reference scalar loop (ported from `block.rs::cpu_attention_fallback`)
//! when the backend returns `Unimplemented` or the tensors live on CPU.
//!
//! See docs/adr/0001-attention-own-vs-delegate.md.

use grim_core::error::Result;
use grim_nn::modules::pick_device_for_storage_device;
use grim_tensor::{DType, Device, Shape, Tensor};
use std::sync::Arc;

/// Inputs are flat host buffers with the layouts the scalar loops already use:
/// - `q`: `[steps, num_heads, head_dim]` (post-RoPE)
/// - `k_history` / `v_history`: `[kv_len, num_kv_heads, head_dim]`, already
///   extended with the current step's keys/values (so `kv_len >= steps` and
///   `cache_offset = kv_len - steps`).
///
/// Returns a `[steps, num_heads * head_dim]` tensor on `device`.
///
/// Causal/window contract (matches `BackendDevice::qkv_attention`): query at
/// absolute position `cache_offset + i` attends to keys `j` with
/// `j <= cache_offset + i` and, when `window` is set,
/// `j >= cache_offset + i - window + 1`.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let kv_stride = num_kv_heads * head_dim;
    let kv_len = k_history.len() / kv_stride;
    debug_assert_eq!(k_history.len(), kv_len * kv_stride);
    debug_assert_eq!(v_history.len(), kv_len * kv_stride);
    let cache_offset = kv_len.saturating_sub(steps);

    // GRIM_QKV_FUSED=0 forces the scalar reference path on GPU backends.
    // Correctness escape hatch while the fused-route generation corruption
    // (bisected to d95f21f, kernel itself verified correct standalone) is
    // being root-caused.
    if std::env::var("GRIM_QKV_FUSED").as_deref() == Ok("0") {
        return scalar_attention(
            q,
            k_history,
            v_history,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            kv_len,
            cache_offset,
            window,
            1.0 / (head_dim as f32).sqrt(),
            &pick_device_for_storage_device(device),
            device,
        );
    }

    let q_shape = Shape::new(vec![steps, num_heads, head_dim]);
    // Allocate the kernel output directly with the FLAT [steps, heads*dim]
    // shape consumers expect. Relabeling a [steps, heads, dim] storage under a
    // flat shape corrupts downstream matmuls that consult the storage dims
    // (bisected: LFM2 `wo` projections on ROCm).
    let out_shape = Shape::new(vec![steps, num_heads * head_dim]);
    let dev = pick_device_for_storage_device(device);

    let q_st = dev.from_cpu(q, &q_shape, DType::F32)?;
    let kv_shape = Shape::new(vec![kv_len, num_kv_heads, head_dim]);
    let k_st = dev.from_cpu(k_history, &kv_shape, DType::F32)?;
    let v_st = dev.from_cpu(v_history, &kv_shape, DType::F32)?;

    match dev.qkv_attention(
        q_st.as_ref(),
        k_st.as_ref(),
        v_st.as_ref(),
        num_kv_heads,
        kv_len,
        cache_offset as u32,
        window,
        &out_shape,
        None,
        None,
    ) {
        Ok((storage, _handle)) => {
            // Output storage was allocated with the flat consumer shape —
            // no relabeling, storage dims and tensor dims agree.
            Ok(Tensor::new(
                Arc::from(storage),
                out_shape.clone(),
                DType::F32,
                grim_tensor::QuantProvenance::default(),
                device.clone(),
            ))
        }
        Err(_) => scalar_attention(
            q,
            k_history,
            v_history,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            kv_len,
            cache_offset,
            window,
            1.0 / (head_dim as f32).sqrt(),
            &dev,
            device,
        ),
    }
}

/// WI-X2: attention over a caller-maintained device KV arena (see
/// `block.rs::cache_append_kv`). Only the per-step K/V rows cross H2D; the
/// history stays resident, so decode cost is O(new tokens), not O(context).
/// Falls back to the host-history path when the device kernel rejects the
/// call or the backend lacks the fused kernel.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention_arena(
    q: &[f32],
    k_arena: &dyn grim_tensor::BackendStorage,
    v_arena: &dyn grim_tensor::BackendStorage,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let q_shape = Shape::new(vec![steps, num_heads, head_dim]);
    // Flat output allocation — see fused_or_scalar_attention for why the
    // storage shape must already match the consumer-facing logical shape.
    let out_shape = Shape::new(vec![steps, num_heads * head_dim]);
    let dev = pick_device_for_storage_device(device);
    let q_st = dev.from_cpu(q, &q_shape, DType::F32)?;
    let cache_offset = kv_len.saturating_sub(steps);
    if let Ok((storage, _handle)) = dev.qkv_attention(
        q_st.as_ref(),
        k_arena,
        v_arena,
        num_kv_heads,
        kv_len,
        cache_offset as u32,
        window,
        &out_shape,
        None,
        None,
    ) {
        return Ok(Tensor::new(
            Arc::from(storage),
            out_shape.clone(),
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        ));
    }
    // Fallback: materialize exactly `kv_len` rows (the arena may be larger
    // than the live history — capacity grows geometrically) and take the
    // host-history path.
    let kv_stride = num_kv_heads * head_dim;
    let k_hist = k_arena.to_cpu_vec_f32()?;
    let v_hist = v_arena.to_cpu_vec_f32()?;
    fused_or_scalar_attention(
        q,
        &k_hist[..kv_len * kv_stride],
        &v_hist[..kv_len * kv_stride],
        num_heads,
        num_kv_heads,
        head_dim,
        steps,
        window,
        device,
    )
}

/// Paged attention entry point (WI-X2): operates on device-resident KV pages via block tables,
/// avoiding full-history CPU-to-GPU re-uploading on decode steps.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention_paged(
    q: &[f32],
    block_tables: &dyn grim_tensor::BackendStorage,
    k_pages: &dyn grim_tensor::BackendStorage,
    v_pages: &dyn grim_tensor::BackendStorage,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks: usize,
    page_size: usize,
    kv_seq_len: usize,
    cache_offset: u32,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let out_shape = Shape::new(vec![1, num_heads, head_dim]);
    let dev = pick_device_for_storage_device(device);
    let q_st = dev.from_cpu(q, &out_shape, DType::F32)?;

    match dev.qkv_attention_paged(
        q_st.as_ref(),
        block_tables,
        k_pages,
        v_pages,
        num_kv_heads,
        max_blocks,
        page_size,
        kv_seq_len,
        cache_offset,
        window,
        &out_shape,
    ) {
        Ok((storage, _handle)) => {
            let flat_shape = Shape::new(vec![1, num_heads * head_dim]);
            Ok(Tensor::new(
                Arc::from(storage),
                flat_shape,
                DType::F32,
                grim_tensor::QuantProvenance::default(),
                device.clone(),
            ))
        }
        Err(_) => {
            // Fallback to scalar attention when the paged device kernel is
            // unavailable. Audit fix (grim-models): the pre-fix fallback
            // IGNORED the block table (treating the page arena as a linear
            // history — wrong whenever blocks are non-contiguous) and
            // `unwrap_or_default()`-ed failed D2H reads into EMPTY K/V
            // (fabricated zeros). It now gathers rows through the block
            // table and propagates read errors.
            let bt_host = block_tables.to_cpu_vec_f32()?;
            let block_table: Vec<usize> = bt_host.iter().map(|&b| b as usize).collect();
            let kv_stride = num_kv_heads * head_dim;
            let k_flat = k_pages.to_cpu_vec_f32()?;
            let v_flat = v_pages.to_cpu_vec_f32()?;
            let k_hist =
                gather_paged_history(&k_flat, &block_table, page_size, kv_stride, kv_seq_len)?;
            let v_hist =
                gather_paged_history(&v_flat, &block_table, page_size, kv_stride, kv_seq_len)?;
            scalar_attention(
                q,
                &k_hist,
                &v_hist,
                num_heads,
                num_kv_heads,
                head_dim,
                1,
                kv_seq_len,
                cache_offset as usize,
                window,
                1.0 / (head_dim as f32).sqrt(),
                &dev,
                device,
            )
        }
    }
}

/// Gather `kv_seq_len` logical history rows out of a paged KV arena using
/// the per-sequence block table: logical position `p` lives at physical row
/// `block_table[p / page_size] * page_size + (p % page_size)`.
pub fn gather_paged_history(
    pages: &[f32],
    block_table: &[usize],
    page_size: usize,
    row_elems: usize,
    kv_seq_len: usize,
) -> grim_core::error::Result<Vec<f32>> {
    use grim_core::error::Error;
    if page_size == 0 || row_elems == 0 {
        return Err(Error::Shape(
            "gather_paged_history: page_size and row_elems must be > 0".into(),
        ));
    }
    let total_rows = pages.len() / row_elems;
    let mut hist = Vec::with_capacity(kv_seq_len * row_elems);
    for pos in 0..kv_seq_len {
        let block = *block_table.get(pos / page_size).ok_or_else(|| {
            Error::Shape(format!(
                "gather_paged_history: block table has {} entries, need {} for kv_seq_len {}",
                block_table.len(),
                pos / page_size + 1,
                kv_seq_len
            ))
        })?;
        let row = block * page_size + (pos % page_size);
        if row >= total_rows {
            return Err(Error::Shape(format!(
                "gather_paged_history: physical row {row} out of range ({total_rows} rows)"
            )));
        }
        hist.extend_from_slice(&pages[row * row_elems..(row + 1) * row_elems]);
    }
    Ok(hist)
}

/// Like [`fused_or_scalar_attention`] but with an explicit softmax scale
/// override (e.g. `qk_scale_factor / sqrt(head_dim)`). Used by models whose
/// config carries a non-unit `qk_scale_factor` (muse_glimmer-class); the
/// device kernel contract has no scale parameter, so those models always
/// take the scalar path when `scale` differs from `1/sqrt(head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention_scaled(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    window: Option<usize>,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let kv_stride = num_kv_heads * head_dim;
    let kv_len = k_history.len() / kv_stride;
    let cache_offset = kv_len.saturating_sub(steps);
    let dev = pick_device_for_storage_device(device);
    scalar_attention(
        q,
        k_history,
        v_history,
        num_heads,
        num_kv_heads,
        head_dim,
        steps,
        kv_len,
        cache_offset,
        window,
        scale,
        &dev,
        device,
    )
}

/// Tensor-level fused attention (GPU-first): q/k/v stay on their device and
/// only the fused `qkv_attention` kernel runs — no host roundtrip on GPU
/// backends. `k`/`v` carry the full history (`kv_len` rows, layout
/// `[kv_len, num_kv_heads * head_dim]`); the kernel applies the causal mask
/// at `cache_offset + i` (`cache_offset = kv_len - steps`), with an optional
/// sliding `window`. Falls back to the scalar host path only when the
/// backend lacks the kernel, matching the `fused_or_scalar_attention`
/// contract without forcing per-call H2D uploads of Q/K/V.
///
/// `q` is `[steps, num_heads * head_dim]` (post-RoPE); storage layouts are
/// relabeled zero-copy where possible (D2D otherwise).
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_tensors(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    kv_len: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    let device = q.device().clone();
    let dev = pick_device_for_storage_device(&device);
    let cache_offset = kv_len.saturating_sub(steps);
    let q3 = crate::block::reshaped_view(q, &Shape::new(vec![steps, num_heads, head_dim]))?;
    let k3 = crate::block::reshaped_view(k, &Shape::new(vec![kv_len, num_kv_heads, head_dim]))?;
    let v3 = crate::block::reshaped_view(v, &Shape::new(vec![kv_len, num_kv_heads, head_dim]))?;
    let out_shape = Shape::new(vec![steps, num_heads * head_dim]);
    let q3s = q3.storage().as_ref();
    let k3s = k3.storage().as_ref();
    let v3s = v3.storage().as_ref();
    // Kernel contracts differ per backend: ROCm/CUDA allocate the flat
    // `[steps, heads*dim]` output directly; the CPU kernel requires a 3-D
    // `[steps, heads, dim]` out_shape. Try flat, retry 3-D, relabel the
    // storage to the flat consumer shape (zero-copy) when needed.
    let fused = |out: &Shape| {
        dev.qkv_attention(
            q3s,
            k3s,
            v3s,
            num_kv_heads,
            kv_len,
            cache_offset as u32,
            window,
            out,
            None,
            None,
        )
    };
    let dim3 = Shape::new(vec![steps, num_heads, head_dim]);
    let storage = match fused(&out_shape) {
        Ok((s, _handle)) => s,
        Err(flat_err) => match fused(&dim3) {
            Ok((s, _handle)) => {
                let t = Tensor::new(
                    Arc::from(s),
                    dim3,
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    device.clone(),
                );
                return crate::block::reshaped_view(&t, &out_shape);
            }
            Err(e) if grim_nn::is_kernel_unimplemented(&e) => {
                let scale = 1.0 / (head_dim as f32).sqrt();
                return scalar_attention(
                    &q.to_vec_f32()?,
                    &k.to_vec_f32()?,
                    &v.to_vec_f32()?,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    steps,
                    kv_len,
                    cache_offset,
                    window,
                    scale,
                    &dev,
                    &device,
                );
            }
            Err(_) => return Err(grim_core::error::Error::from(flat_err)),
        },
    };
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    ))
}

/// Concatenate two `[rows_a, width]` / `[rows_b, width]` tensors along rows
/// (GPU-first). Device path: fresh arena + two D2D copies (the
/// `block.rs::cache_append_kv` primitive pair); host fallback only when the
/// backend lacks `alloc_storage`/`copy_slice_into`.
pub fn concat_rows_on_device(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let rows = a.shape().dims()[0] + b.shape().dims()[0];
    let width = *a.shape().dims().last().expect("non-empty tensor");
    let out_shape = Shape::new(vec![rows, width]);
    let dev = pick_device_for_storage_device(a.device());
    if let Ok(fresh) = dev.alloc_storage(&out_shape, DType::F32) {
        let a_ok = dev.copy_slice_into(
            fresh.as_ref(),
            a.storage().as_ref(),
            0,
            a.shape().elem_count(),
        );
        let b_ok = a_ok.and_then(|_| {
            dev.copy_slice_into(
                fresh.as_ref(),
                b.storage().as_ref(),
                a.shape().elem_count(),
                b.shape().elem_count(),
            )
        });
        if b_ok.is_ok() {
            return Ok(Tensor::new(
                Arc::from(fresh),
                out_shape,
                DType::F32,
                a.provenance().clone(),
                a.device().clone(),
            ));
        }
    }
    // Host fallback.
    let mut data = a.to_vec_f32()?;
    data.extend_from_slice(&b.to_vec_f32()?);
    let storage = dev.from_cpu(&data, &out_shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    ))
}

/// NeoX RoPE for `[steps, num_heads * head_dim]` tensors (GPU-first).
/// Relabels to one head_dim-wide row per head (`(1, steps * num_heads, D)`,
/// positions repeated per head — the `block.rs`/`muse_glimmer` kernel
/// contract), runs the grim-nn `Rope` module (device kernel on GPU, host
/// loop on the CPU fallback backend), relabels back to `[steps, width]`.
/// `x.width() == num_heads * rope.config.dim` must hold.
pub fn rope_2d_on_device(
    rope: &grim_nn::Rope,
    x: &Tensor,
    num_heads: usize,
    positions: &[u32],
) -> Result<Tensor> {
    let dims = x.shape().dims();
    let (steps, width) = (dims[0], dims[1]);
    debug_assert_eq!(width, num_heads * rope.config.dim);
    let head_dim = rope.config.dim;

    let mut ext_positions = Vec::with_capacity(steps * num_heads);
    for si in 0..steps {
        let pos = positions.get(si).copied().unwrap_or(si as u32);
        for _ in 0..num_heads {
            ext_positions.push(pos);
        }
    }

    let rows3 = crate::block::reshaped_view(x, &Shape::new(vec![1, steps * num_heads, head_dim]))?;
    let roped3 = rope.forward(&rows3, &ext_positions)?;
    crate::block::reshaped_view(&roped3, &Shape::new(vec![steps, width]))
}

/// Reference scalar attention with causal + sliding-window masking.
/// Direct port of `block.rs::cpu_attention_fallback`, taking explicit dims
/// so loaders without a `BlockConfig` can use it.
#[allow(clippy::too_many_arguments)]
fn scalar_attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    kv_len: usize,
    cache_offset: usize,
    window: Option<usize>,
    scale: f32,
    dev: &std::sync::Arc<dyn grim_tensor::backend::BackendDevice>,
    device: &Device,
) -> Result<Tensor> {
    let num_head_dims = num_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;
    let mut out = vec![0.0f32; steps * num_head_dims];

    for h in 0..num_heads {
        let kvh = (h * num_kv_heads) / num_heads;
        for t in 0..steps {
            let causal_limit = cache_offset + t;
            let window_start = match window {
                Some(w) => (causal_limit + 1).saturating_sub(w),
                None => 0,
            };
            let mut scores = vec![0.0f32; kv_len];
            for t2 in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * num_head_dims + h * head_dim + d]
                        * k_history[t2 * kv_stride + kvh * head_dim + d];
                }
                scores[t2] = dot * scale;
            }
            for (t2, s) in scores.iter_mut().enumerate() {
                if t2 > causal_limit || t2 < window_start {
                    *s = f32::NEG_INFINITY;
                }
            }

            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t2 in window_start..=causal_limit {
                    acc += scores[t2] * v_history[t2 * kv_stride + kvh * head_dim + d];
                }
                out[t * num_head_dims + h * head_dim + d] = acc;
            }
        }
    }

    let flat = Shape::new(vec![steps, num_head_dims]);
    let storage = dev.from_cpu(&out, &flat, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        flat,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audit gate: the paged fallback's gather must follow the BLOCK TABLE,
    /// not assume the arena is linear history. Pages are filled with
    /// position-encoded values in permuted physical order; the gather must
    /// reconstruct linear order exactly.
    #[test]
    fn gather_paged_history_follows_block_table() {
        let page_size = 4usize;
        let row_elems = 2usize;
        let kv_seq_len = 10usize; // 2.5 pages → 3-page block table
        // Physical arena where row r holds value r as [r.0, r.25].
        let n_pages = 5usize;
        let mut arena = vec![0.0f32; n_pages * page_size * row_elems];
        for r in 0..n_pages * page_size {
            arena[r * row_elems] = r as f32;
            arena[r * row_elems + 1] = r as f32 + 0.25;
        }
        // Logical pages 0..3 map to PHYSICAL pages 4, 0, 3 (permuted).
        let block_table = vec![4usize, 0, 3];

        let got = gather_paged_history(&arena, &block_table, page_size, row_elems, kv_seq_len)
            .expect("gather");

        assert_eq!(got.len(), kv_seq_len * row_elems);
        for pos in 0..kv_seq_len {
            let expect_row = (block_table[pos / page_size] * page_size + pos % page_size) as f32;
            assert_eq!(got[pos * row_elems], expect_row, "logical pos {pos}");
            assert_eq!(
                got[pos * row_elems + 1],
                expect_row + 0.25,
                "logical pos {pos}"
            );
        }

        // Short block table must error, never fabricate.
        let short = gather_paged_history(&arena, &[0], page_size, row_elems, kv_seq_len);
        assert!(short.is_err(), "short block table must error");
    }

    /// The scalar fallback must match a straightforward textbook reference
    /// with causal + sliding-window masking, including at nonzero offsets.
    #[test]
    fn scalar_attention_matches_reference_with_window() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let kv_len = 24usize;
        let steps = 3usize;
        const TEST_WINDOW: usize = 10;
        let window = Some(TEST_WINDOW);

        let mut q = vec![0.0f32; steps * num_heads * head_dim];
        let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
        let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
        let mut seed = 0x1234_5678u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for x in &mut q {
            *x = rand();
        }
        for x in &mut k {
            *x = rand();
        }
        for x in &mut v {
            *x = rand();
        }

        let dev: std::sync::Arc<dyn grim_tensor::backend::BackendDevice> =
            pick_device_for_storage_device(&Device::Cpu);
        let got = scalar_attention(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            kv_len,
            kv_len - steps,
            window,
            1.0 / (head_dim as f32).sqrt(),
            &dev,
            &Device::Cpu,
        )
        .unwrap();
        let got = got.to_vec_f32().unwrap();

        // Independent reference.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;
        let cache_offset = kv_len - steps;
        for t in 0..steps {
            for h in 0..num_heads {
                let kvh = (h * num_kv_heads) / num_heads;
                let causal_limit = cache_offset + t;
                let window_start = (causal_limit + 1).saturating_sub(TEST_WINDOW);
                let mut scores = Vec::with_capacity(causal_limit - window_start + 1);
                for t2 in window_start..=causal_limit {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[t * num_heads * head_dim + h * head_dim + d]
                            * k[t2 * kv_stride + kvh * head_dim + d];
                    }
                    scores.push(dot * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..head_dim {
                    let mut acc = 0.0;
                    for (i, s) in scores.iter().enumerate() {
                        let t2 = window_start + i;
                        acc += ((s - mx).exp() / sum) * v[t2 * kv_stride + kvh * head_dim + d];
                    }
                    let expect = acc;
                    let idx = t * num_heads * head_dim + h * head_dim + d;
                    assert!(
                        (got[idx] - expect).abs() < 1e-5,
                        "t={t} h={h} d={d}: got {} expect {}",
                        got[idx],
                        expect
                    );
                }
            }
        }
    }

    /// WI-X2: the arena entry must produce byte-identical results to the
    /// host-history entry when the device has no fused kernel (CPU) — the
    /// arena path degrades to a materialize-and-fallback, never to wrong math.
    #[test]
    fn arena_attention_matches_host_history_path() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let kv_len = 24usize;
        let steps = 3usize;
        const TEST_WINDOW: usize = 10;
        let window = Some(TEST_WINDOW);
        let kv_stride = num_kv_heads * head_dim;

        let mut seed = 0xfeed_beefu64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..steps * num_heads * head_dim).map(|_| rand()).collect();
        let k: Vec<f32> = (0..kv_len * kv_stride).map(|_| rand()).collect();
        let v: Vec<f32> = (0..kv_len * kv_stride).map(|_| rand()).collect();

        let device = Device::Cpu;
        let host = fused_or_scalar_attention(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            window,
            &device,
        )
        .expect("host-history attention");

        // Arena holding exactly kv_len rows, laid out [kv_len, kv_stride].
        let shape = Shape::new(vec![kv_len, kv_stride]);
        let cpu_dev = grim_nn::modules::pick_device_for_storage_device(&device);
        let k_arena = cpu_dev.from_cpu(&k, &shape, DType::F32).expect("k arena");
        let v_arena = cpu_dev.from_cpu(&v, &shape, DType::F32).expect("v arena");

        let arena = fused_or_scalar_attention_arena(
            &q,
            k_arena.as_ref(),
            v_arena.as_ref(),
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            window,
            &device,
        )
        .expect("arena attention");

        let a = host.to_vec_f32().expect("host vec");
        let b = arena.to_vec_f32().expect("arena vec");
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "arena/host divergence at [{i}]: {x} vs {y}"
            );
        }
    }
}

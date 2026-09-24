//! attention_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{AttentionOps, BackendStorage, CoreTensorOps, Shape};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl AttentionOps for MetalDevice {
    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if q.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(q.dtype())));
                }

                let q_s = q.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention q is not MetalStorage".into())
                })?;
                let k_s = k_tensor
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention k_tensor is not MetalStorage".into())
                    })?;
                let ks_s = k_scales
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention k_scales is not MetalStorage".into())
                    })?;
                let v_s = v_tensor
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention v_tensor is not MetalStorage".into())
                    })?;
                let vs_s = v_scales
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention v_scales is not MetalStorage".into())
                    })?;

                let q_buf = q_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k_tensor has no GPU buffer".into()))?;
                let ks_buf = ks_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k_scales has no GPU buffer".into()))?;
                let v_buf = v_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v_tensor has no GPU buffer".into()))?;
                let vs_buf = vs_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v_scales has no GPU buffer".into()))?;

                let out_dims = out_shape.dims();
                if out_dims.len() != 3 {
                    return Err(Error::Backend("kv_dequant_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into()));
                }
                let seq_len = out_dims[0];
                let num_heads = out_dims[1];
                let head_dim = out_dims[2];

                let out_storage = self.zeros(out_shape, q.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.kv_dequant_attn);
                encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(ks_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(vs_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 5);

                let num_heads_val = num_heads as i32;
                let num_kv_heads_val = num_kv_heads as i32;
                let head_dim_val = head_dim as i32;
                let seq_len_val = seq_len as i32;
                let kv_seq_len_val = kv_seq_len as i32;
                let cache_offset_val = cache_offset as i32;
                let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
                let quant_bits_val = quant_bits as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &num_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_kv_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                    encoder.setBytes_length_atIndex(
                        &head_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        8,
                    );
                    encoder.setBytes_length_atIndex(
                        &seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        9,
                    );
                    encoder.setBytes_length_atIndex(
                        &kv_seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        10,
                    );
                    encoder.setBytes_length_atIndex(
                        &cache_offset_val as *const i32 as *const std::ffi::c_void,
                        4,
                        11,
                    );
                    encoder.setBytes_length_atIndex(
                        &inv_sqrt_d as *const f32 as *const std::ffi::c_void,
                        4,
                        12,
                    );
                    encoder.setBytes_length_atIndex(
                        &quant_bits_val as *const i32 as *const std::ffi::c_void,
                        4,
                        13,
                    );
                }

                let threads_per_group = MTLSize::new(1, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                Err(Error::Backend(
                    "Metal device inner is None (fallback mode)".into(),
                ))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (
                q,
                k_tensor,
                k_scales,
                v_tensor,
                v_scales,
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                quant_bits,
                out_shape,
            );
            Err(Error::Unimplemented(
                "kv_dequant_attention not supported on non-Apple platform".into(),
            ))
        }
    }

    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
            out_max,
            out_sum,
        )
    }

    #[allow(unused_variables)] // params only used on the cfg-gated Apple path
    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // The Metal `grim_qkv_attention_paged` kernel accepts a `window_lo` + `has_window` argument pair; SWA layers compute the lower bound host-side and the kernel masks below it.
        // No host fallback needed.

        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let dims = out_shape.dims();
            if dims.len() != 3 {
                return Err(Error::from(MetalError::DataMismatch(
                    "paged attention expects [batch, heads, dim]".into(),
                )));
            }
            let q_s = q
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged q is not MetalStorage".into()))?;
            if num_kv_heads == 0 || dims[1] % num_kv_heads != 0 {
                return Err(Error::from(MetalError::DataMismatch(
                    "paged attention requires num_heads divisible by num_kv_heads".into(),
                )));
            }
            let table_s = block_tables
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged block table is not MetalStorage".into()))?;
            let k_s = k_pages
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged k pages is not MetalStorage".into()))?;
            let v_s = v_pages
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged v pages is not MetalStorage".into()))?;
            let q_buf = q_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
            let table_buf = table_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("block table has no GPU buffer".into()))?;
            let k_buf = k_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("k pages has no GPU buffer".into()))?;
            let v_buf = v_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("v pages has no GPU buffer".into()))?;
            let out_storage = self.zeros(out_shape, DType::F32)?;
            let out_buf = out_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .unwrap()
                .buffer
                .as_ref()
                .unwrap();
            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;
            encoder.setComputePipelineState(&inner.pipelines.qkv_paged_attn);
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(table_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
            // SWA: window_lo = max(0, cache_offset - window + 1).
            let abs_first = cache_offset as usize;
            let window_lo_val: i32 = match window {
                Some(w) => abs_first.saturating_sub(w.saturating_sub(1)) as i32,
                None => 0,
            };
            let has_window_val: i32 = if window.is_some() { 1 } else { 0 };
            let vals = [
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                page_size as i32,
                max_blocks as i32,
                kv_seq_len as i32,
                num_kv_heads as i32,
                window_lo_val,
                has_window_val,
            ];
            unsafe {
                for (i, value) in vals.iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        value as *const i32 as *const std::ffi::c_void,
                        4,
                        5 + i,
                    );
                }
            }
            encoder.dispatchThreads(
                MTLSize::new(dims[2] as u64, dims[1] as u64, dims[0] as u64),
                MTLSize::new(32, 1, 1),
            );
            encoder.endEncoding();
            return Ok((
                out_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }
        Err(Error::Unimplemented(
            "Metal paged attention requires Apple Metal GPU support".into(),
        ))
    }

    #[allow(unused_variables)] // params only used on the cfg-gated Apple path
    fn tree_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        tree_parents: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dims = out_shape.dims();
        if dims.len() != 4 {
            return Err(Error::from(MetalError::DataMismatch(
                "tree attention expects [batch, 1+gamma, heads, dim]".into(),
            )));
        }
        if num_kv_heads == 0 || dims[2] % num_kv_heads != 0 {
            return Err(Error::from(MetalError::DataMismatch(
                "tree attention requires num_heads divisible by num_kv_heads".into(),
            )));
        }
        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let q_s = q
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree q is not MetalStorage".into()))?;
            let k_s = k
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree k is not MetalStorage".into()))?;
            let v_s = v
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree v is not MetalStorage".into()))?;
            let p_s = tree_parents
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree parents is not MetalStorage".into()))?;
            let q_buf = q_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
            let k_buf = k_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("k has no GPU buffer".into()))?;
            let v_buf = v_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("v has no GPU buffer".into()))?;
            let p_buf = p_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("parents has no GPU buffer".into()))?;
            let out_storage = self.zeros(out_shape, DType::F32)?;
            let out_buf = out_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .unwrap()
                .buffer
                .as_ref()
                .unwrap();
            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;
            encoder.setComputePipelineState(&inner.pipelines.tree_attn);
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(p_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
            let vals = [
                dims[0] as i32,
                dims[2] as i32,
                kv_seq_len as i32,
                dims[3] as i32,
                (dims[1] - 1) as i32,
                cache_offset as i32,
                num_kv_heads as i32,
            ];
            unsafe {
                for (i, value) in vals.iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        value as *const i32 as *const std::ffi::c_void,
                        4,
                        5 + i,
                    );
                }
            }
            encoder.dispatchThreads(
                MTLSize::new(dims[3] as u64, (dims[1] * dims[2]) as u64, dims[0] as u64),
                MTLSize::new(256, 1, 1),
            );
            encoder.endEncoding();
            return Ok((
                out_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }
        Err(Error::Unimplemented(
            "Metal tree attention requires Apple Metal GPU support".into(),
        ))
    }

    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )
    }

    #[allow(clippy::needless_range_loop)]
    fn mla_absorbed_decode(
        &self,
        q_absorbed: &dyn BackendStorage,
        q_rope: &dyn BackendStorage,
        kv_cache: &dyn BackendStorage,
        _w_uv: Option<&dyn BackendStorage>,
        out: &dyn BackendStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        _v_head_dim: usize,
        seq_len: usize,
        _w_uv_offset_words: usize,
        _w_uv_head_stride_words: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        // CPU fallback / host validation for MLA absorbed decode on Metal
        let q_abs_vec = q_absorbed.to_cpu_vec_f32()?;
        let q_rope_vec = q_rope.to_cpu_vec_f32()?;
        let kv_cache_vec = kv_cache.to_cpu_vec_f32()?;

        let d_total = kv_lora_rank + qk_rope_dim;
        let scale = 1.0f32 / (d_total as f32).sqrt();

        let mut scores = vec![0.0f32; seq_len];
        let mut out_acc = vec![0.0f32; num_heads * kv_lora_rank];

        for h in 0..num_heads {
            let q_abs_head = &q_abs_vec[h * kv_lora_rank..(h + 1) * kv_lora_rank];
            let q_rope_head = &q_rope_vec[h * qk_rope_dim..(h + 1) * qk_rope_dim];

            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let kv_slot = &kv_cache_vec[t * d_total..(t + 1) * d_total];
                let k_abs = &kv_slot[..kv_lora_rank];
                let k_rope = &kv_slot[kv_lora_rank..];

                let mut dot = 0.0f32;
                for i in 0..kv_lora_rank {
                    dot += q_abs_head[i] * k_abs[i];
                }
                for i in 0..qk_rope_dim {
                    dot += q_rope_head[i] * k_rope[i];
                }
                let s = dot * scale;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                let p = (scores[t] - max_score).exp();
                scores[t] = p;
                sum_exp += p;
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

            let out_head = &mut out_acc[h * kv_lora_rank..(h + 1) * kv_lora_rank];
            for t in 0..seq_len {
                let weight = scores[t] * inv_sum;
                let v_latent = &kv_cache_vec[t * d_total..t * d_total + kv_lora_rank];
                for i in 0..kv_lora_rank {
                    out_head[i] += weight * v_latent[i];
                }
            }
        }

        let updated = self.from_cpu(&out_acc, out.shape(), out.dtype())?;
        let _ = updated;
        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }

    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let (q_norm, h1) = self.rms_norm(q_raw, q_norm_w, eps, q_raw.shape())?;
        let (kv_norm, h2) = self.rms_norm(kv_raw, kv_norm_w, eps, kv_raw.shape())?;
        h1.synchronize()?;
        h2.synchronize()?;

        let q_vec = q_norm.to_cpu_vec_f32()?;
        let kv_vec = kv_norm.to_cpu_vec_f32()?;

        let q_shape = q_raw.shape().dims();
        let seq = q_shape[0];
        let q_dim = q_shape.last().copied().unwrap_or(qk_nope_dim + qk_rope_dim);
        let kv_shape = kv_raw.shape().dims();
        let kv_dim = kv_shape.last().copied().unwrap_or(qk_rope_dim + v_dim);

        let mut q_nope_vec = Vec::with_capacity(seq * qk_nope_dim);
        let mut q_rope_vec = Vec::with_capacity(seq * qk_rope_dim);
        let mut k_rope_vec = Vec::with_capacity(seq * qk_rope_dim);
        let mut v_vec = Vec::with_capacity(seq * v_dim);

        for s in 0..seq {
            let q_row = &q_vec[s * q_dim..(s + 1) * q_dim];
            q_nope_vec.extend_from_slice(&q_row[..qk_nope_dim.min(q_row.len())]);
            q_rope_vec.extend_from_slice(&q_row[qk_nope_dim.min(q_row.len())..]);

            let kv_row = &kv_vec[s * kv_dim..(s + 1) * kv_dim];
            k_rope_vec.extend_from_slice(&kv_row[..qk_rope_dim.min(kv_row.len())]);
            v_vec.extend_from_slice(&kv_row[qk_rope_dim.min(kv_row.len())..]);
        }

        let q_nope_storage = self.from_cpu(
            &q_nope_vec,
            &Shape::new(vec![seq, qk_nope_dim]),
            q_raw.dtype(),
        )?;
        let q_rope_storage = self.from_cpu(
            &q_rope_vec,
            &Shape::new(vec![seq, qk_rope_dim]),
            q_raw.dtype(),
        )?;
        let k_rope_storage = self.from_cpu(
            &k_rope_vec,
            &Shape::new(vec![seq, qk_rope_dim]),
            kv_raw.dtype(),
        )?;
        let v_storage = self.from_cpu(&v_vec, &Shape::new(vec![seq, v_dim]), kv_raw.dtype())?;

        Ok((
            q_nope_storage,
            q_rope_storage,
            k_rope_storage,
            v_storage,
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dim = cfg.dim;
        let base = cfg.base;
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rope: input x is not MetalStorage".into())
                })?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;

                let out_storage = self.zeros(out_shape, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let num_tokens = positions.len();
                let num_heads = (out_shape.elem_count() / (num_tokens * dim)) as i32;
                let head_dim = dim as i32;

                // Upload positions to a temporary GPU buffer.
                let pos_data = std::sync::Arc::new(positions.to_vec());
                let pos_buf = inner
                    .device
                    .newBufferWithLength_options(
                        (positions.len() * std::mem::size_of::<u32>()) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate pos buffer for rope".into(),
                        ))
                    })?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pos_data.as_ptr() as *const u8,
                        pos_buf.contents() as *mut u8,
                        positions.len() * std::mem::size_of::<u32>(),
                    );
                }

                let total = out_shape.elem_count();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                let num_tokens_val = num_tokens as i32;
                let num_heads_val = num_heads;
                let head_dim_val = head_dim;
                let base_val = base;

                if !cfg.is_plain() {
                    // Partial-rotary / YaRN: dispatch grim_rope_yarn. The YaRN
                    // ramp + mscale are recomputed inside the kernel.
                    let rotary_dim = cfg.rotary_dim.min(dim) as i32;
                    let rotary_half = (rotary_dim / 2) as usize;
                    let (
                        has_yarn,
                        yarn_factor,
                        yarn_orig_max,
                        yarn_beta_fast,
                        yarn_beta_slow,
                        mscale,
                    ) = match cfg.yarn {
                        Some(y) => (
                            1i32,
                            y.factor,
                            y.original_max_pos as f32,
                            y.beta_fast,
                            y.beta_slow,
                            y.attention_factor,
                        ),
                        None => (0i32, 1.0f32, 8192.0f32, 32.0f32, 1.0f32, 1.0f32),
                    };
                    encoder.setComputePipelineState(&inner.pipelines.rope_yarn);
                    encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&pos_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                    // Scalar buffers 3..14.
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            &num_tokens_val as *const i32 as *const std::ffi::c_void,
                            4,
                            3,
                        );
                        encoder.setBytes_length_atIndex(
                            &num_heads_val as *const i32 as *const std::ffi::c_void,
                            4,
                            4,
                        );
                        encoder.setBytes_length_atIndex(
                            &head_dim_val as *const i32 as *const std::ffi::c_void,
                            4,
                            5,
                        );
                        encoder.setBytes_length_atIndex(
                            &rotary_dim as *const i32 as *const std::ffi::c_void,
                            4,
                            6,
                        );
                        encoder.setBytes_length_atIndex(
                            &has_yarn as *const i32 as *const std::ffi::c_void,
                            4,
                            7,
                        );
                        encoder.setBytes_length_atIndex(
                            &base_val as *const f32 as *const std::ffi::c_void,
                            4,
                            8,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_factor as *const f32 as *const std::ffi::c_void,
                            4,
                            9,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_orig_max as *const f32 as *const std::ffi::c_void,
                            4,
                            10,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_beta_fast as *const f32 as *const std::ffi::c_void,
                            4,
                            11,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_beta_slow as *const f32 as *const std::ffi::c_void,
                            4,
                            12,
                        );
                        encoder.setBytes_length_atIndex(
                            &mscale as *const f32 as *const std::ffi::c_void,
                            4,
                            13,
                        );
                    }
                    // Grid covers max(num_tokens*num_heads*rotary_half, *copy_len).
                    let copy_len = dim - 2 * rotary_half;
                    let total_pairs = (num_tokens
                        * num_heads as usize
                        * rotary_half
                            .max(if copy_len > 0 { copy_len } else { 0 })
                            .max(1)) as u64;
                    let threads_per_group = MTLSize::new(256, 1, 1);
                    let groups = MTLSize::new(((total_pairs + 255) / 256), 1, 1);
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    encoder.endEncoding();
                    return Ok((
                        out_storage,
                        Box::new(MetalHandle {
                            command_buffer: cmd_buffer,
                        }),
                    ));
                }

                // Plain full-rotary RoPE.
                encoder.setComputePipelineState(&inner.pipelines.rope);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&pos_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &num_tokens_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &head_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &base_val as *const f32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                }

                let half_dim = dim / 2;
                let total_pairs = (total / (half_dim * 2)).max(1);
                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total_pairs + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                return Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ));
            }
        }

        // CPU fallback: non-Apple target or Metal unavailable. Plain RoPE only;
        // refuse non-plain configs so the block-level CPU Rope::forward runs.
        if !cfg.is_plain() {
            return Err(Error::Unimplemented(
                "Metal rope: partial-rotary / YaRN not available on non-Apple CPU fallback; falling back to CPU rope module"
                    .into(),
            ));
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);
        let half_dim = dim / 2;

        let mut res = x_vec.clone();
        for (t, &pos) in positions.iter().enumerate() {
            let p = pos as f32;
            for h in 0..num_heads {
                for i in 0..half_dim {
                    let freq = 1.0f32 / base.powf((2 * i) as f32 / dim as f32);
                    let val = p * freq;
                    let cos_v = val.cos();
                    let sin_v = val.sin();

                    let base_idx = (t * num_heads + h) * dim;
                    let idx0 = base_idx + i;
                    let idx1 = base_idx + i + half_dim;

                    let v0 = x_vec[idx0];
                    let v1 = x_vec[idx1];

                    res[idx0] = v0 * cos_v - v1 * sin_v;
                    res[idx1] = v0 * sin_v + v1 * cos_v;
                }
            }
        }

        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn flash_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        _causal: bool,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )?;
        let _ = num_heads;
        let _ = head_dim;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn cross_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q, k, v, num_heads, kv_seq_len, 0, None, out_shape, None, None,
        )?;

        let _ = head_dim;
        let _ = seq_len;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }
}

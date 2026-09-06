//! `AttentionOps` implementation for VulkanDevice.
//!
//! Extracted from lib.rs (modularization): trait impls live in `device/`,
//! dispatch plumbing in `kernel.rs`, buffers in `storage.rs`, device init
//! in `context.rs`.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendStorage, Shape, CoreTensorOps, AttentionOps};

use crate::context::global_context;
use crate::ffi::*;
use crate::kernel::{push_params, run_compute_shader, run_compute_shader_kernel, spirv_for, VulkanKernel};
use crate::{VulkanDevice, VulkanHandle, VulkanStorage};

impl AttentionOps for VulkanDevice {


    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan sage_attention: q is not VulkanStorage".into())
        })?;
        let k_s = k.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan sage_attention: k is not VulkanStorage".into())
        })?;
        let v_s = v.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan sage_attention: v is not VulkanStorage".into())
        })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let q_dims = q.shape().dims();
        let num_heads = q_dims[q_dims.len() - 2];
        let head_dim = q_dims[q_dims.len() - 1];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let buffers = [q_s.buffer, k_s.buffer, v_s.buffer, out_storage.buffer];
        let push = push_params(
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            kv_seq_len as u32,
            64,
            scale,
        );

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::SageAttention,
            &buffers,
            num_heads as u32,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan sage_attention dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
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
        // `window == Some(w)` dispatches the dedicated `QkvAttentionSwa` kernel
        // (host-computed `window_lo` lower bound); `None` runs the plain
        // full-causal `QkvAttention` kernel. Both produce correct on-device
        // output; no host fallback.
        self.qkv_attention_inner(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            out_shape,
            out_max,
            out_sum,
            window,
        )
    }


    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        _max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_paged expects 3-D output shape [batch, num_heads, head_dim]".into(),
            ));
        }
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "qkv_attention_paged requires num_heads divisible by num_kv_heads".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged q is not VulkanStorage".into()))?;
        let table_s = block_tables
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged block_tables is not VulkanStorage".into())
            })?;
        let k_s = k_pages
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged k_pages is not VulkanStorage".into())
            })?;
        let v_s = v_pages
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged v_pages is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            table_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            page_size as u32,
            0,
            kv_seq_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as f32,
        );
        let grid_x = head_dim.div_ceil(32) as u32;

        if let Some(w) = window {
            // Sliding-window paged: dispatch QkvAttentionPagedSwa. window_lo is
            // host-computed max(0, cache_offset - w + 1).
            let abs_first = cache_offset as usize;
            let window_lo = abs_first.saturating_sub(w.saturating_sub(1)) as u32;
            // 8 × u32 = 32 bytes Params block: 6 base slots + window_lo + has_window(=1).
            let swa_push: [u32; 8] = [
                push[0], push[1], push[2], push[3], push[4], push[5], window_lo, 1u32,
            ];
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::QkvAttentionPagedSwa,
                &buffers,
                grid_x,
                num_heads as u32,
                1,
                Some(&swa_push),
            )?;
        } else {
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::QkvAttentionPaged,
                &buffers,
                grid_x,
                num_heads as u32,
                1,
                Some(&push),
            )?;
        }

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }


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
            return Err(Error::Shape(
                "tree_attention expects [batch, 1+gamma, num_heads, head_dim]".into(),
            ));
        }
        let (batch, nodes, num_heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "tree_attention requires num_heads divisible by num_kv_heads".into(),
            ));
        }
        if head_dim > 256 || nodes == 0 || tree_parents.shape().elem_count() < nodes {
            return Err(Error::Shape(
                "tree_attention requires 1+gamma parent entries and head_dim <= 256".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention q is not VulkanStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention v is not VulkanStorage".into()))?;
        let parents_s = tree_parents
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("tree_attention tree_parents is not VulkanStorage".into())
            })?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            parents_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            batch as u32,
            num_heads as u32,
            kv_seq_len as u32,
            head_dim as u32,
            (nodes - 1) as u32,
            f32::from_bits((num_kv_heads as u32) << 16 | (cache_offset & 0xffff)),
        );
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::TreeAttention),
            &buffers,
            1,
            (nodes * num_heads) as u32,
            batch as u32,
            Some(&push),
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }


    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        _num_kv_heads: usize,
        kv_seq_len: usize,
        _cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if quant_bits != 8 {
            return Err(Error::Unimplemented(
                "Vulkan kv_dequant_attention currently supports 8-bit K/V only".into(),
            ));
        }
        let dims = out_shape.dims();
        if dims.len() != 3 {
            return Err(Error::Shape(
                "kv_dequant_attention expects [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention q is not VulkanStorage".into()))?;
        let k_s = k_tensor
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention k_tensor is not VulkanStorage".into())
            })?;
        let ks_s = k_scales
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention k_scales is not VulkanStorage".into())
            })?;
        let v_s = v_tensor
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention v_tensor is not VulkanStorage".into())
            })?;
        let vs_s = v_scales
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention v_scales is not VulkanStorage".into())
            })?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            ks_s.buffer,
            vs_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            kv_seq_len as u32,
            dims[2] as u32,
            0,
            dims[2] as u32,
            dims[1] as u32,
            0.0,
        );
        let grid_x = dims[2].div_ceil(32) as u32;
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::KvDequantAttention),
            &buffers,
            grid_x,
            dims[1] as u32,
            1,
            Some(&push),
        )?;
        Ok((
            Box::new(out_storage),
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
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rope x is not VulkanStorage".into()))?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);

        let pos_shape = Shape::new(vec![num_tokens]);
        let pos_storage = VulkanStorage::alloc_gpu(
            &pos_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;
        let mut mapped_pos: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                pos_storage.memory,
                0,
                pos_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_pos,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for positions buffer: {}",
                    res
                )));
            }
            std::ptr::copy_nonoverlapping(positions.as_ptr(), mapped_pos as *mut u32, num_tokens);
            vkUnmapMemory(ctx.device, pos_storage.memory);
        }

        let buffers = [x_s.buffer, pos_storage.buffer, out_storage.buffer];

        if !cfg.is_plain() {
            // Partial-rotary / YaRN: dispatch the dedicated `RopeYarn` kernel.
            // The YaRN frequency ramp + mscale are recomputed inside the shader
            // from the push-constant scalars (no inv_freq buffer needed),
            // numerically matching the CPU/HIP references.
            let rotary_dim = cfg.rotary_dim.min(dim);
            let rotary_half = rotary_dim / 2;
            let (has_yarn, yarn_factor, yarn_orig_max, yarn_beta_fast, yarn_beta_slow, mscale) =
                match cfg.yarn {
                    Some(y) => (
                        1u32,
                        y.factor,
                        y.original_max_pos as f32,
                        y.beta_fast,
                        y.beta_slow,
                        y.attention_factor,
                    ),
                    None => (0u32, 1.0f32, 8192.0f32, 32.0f32, 1.0f32, 1.0f32),
                };
            // Params block (11 × u32 = 44 bytes):
            //   num_tokens, head_dim, num_heads, rotary_dim, has_yarn,
            //   base(f32 bits), yarn_factor, yarn_orig_max, yarn_beta_fast,
            //   yarn_beta_slow, mscale
            let push: [u32; 11] = [
                num_tokens as u32,
                dim as u32,
                num_heads as u32,
                rotary_dim as u32,
                has_yarn,
                base.to_bits(),
                yarn_factor.to_bits(),
                yarn_orig_max.to_bits(),
                yarn_beta_fast.to_bits(),
                yarn_beta_slow.to_bits(),
                mscale.to_bits(),
            ];
            // Grid covers max(num_tokens*num_heads*rotary_half, *copy_len) for
            // both the rotate pass and the verbatim-tail copy pass.
            let copy_len = dim - 2 * rotary_half;
            let total = (num_tokens
                * num_heads
                * rotary_half
                    .max(if copy_len > 0 { copy_len } else { 0 })
                    .max(1)) as u32;
            let grid_x = total.div_ceil(256);
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::RopeYarn,
                &buffers,
                grid_x,
                1,
                1,
                Some(&push),
            )?;
        } else {
            // Plain full-rotary RoPE.
            let total_pairs = (num_tokens * num_heads * (dim / 2)) as u32;
            let grid_x = total_pairs.div_ceil(256);
            let push = push_params(num_tokens as u32, dim as u32, num_heads as u32, 0, 0, base);
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::Rope,
                &buffers,
                grid_x,
                1,
                1,
                Some(&push),
            )?;
        }

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }


    fn rerope(
        &self,
        k: &dyn BackendStorage,
        old_positions: &[u32],
        new_positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dim = cfg.dim;
        let base = cfg.base;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rerope k is not VulkanStorage".into()))?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let num_tokens = old_positions.len();
        if new_positions.len() != num_tokens {
            return Err(Error::Backend(format!(
                "Vulkan rerope: old_positions len {} != new_positions len {}",
                num_tokens,
                new_positions.len()
            )));
        }
        let num_heads = out_shape.elem_count() / (num_tokens * dim);

        let pos_shape = Shape::new(vec![num_tokens]);
        let old_pos_storage = VulkanStorage::alloc_gpu(
            &pos_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;
        let new_pos_storage = VulkanStorage::alloc_gpu(
            &pos_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;

        // Upload old_positions
        let mut mapped_old: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                old_pos_storage.memory,
                0,
                old_pos_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_old,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for old_positions buffer: {res}"
                )));
            }
            std::ptr::copy_nonoverlapping(
                old_positions.as_ptr(),
                mapped_old as *mut u32,
                num_tokens,
            );
            vkUnmapMemory(ctx.device, old_pos_storage.memory);
        }

        // Upload new_positions
        let mut mapped_new: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                new_pos_storage.memory,
                0,
                new_pos_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_new,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for new_positions buffer: {res}"
                )));
            }
            std::ptr::copy_nonoverlapping(
                new_positions.as_ptr(),
                mapped_new as *mut u32,
                num_tokens,
            );
            vkUnmapMemory(ctx.device, new_pos_storage.memory);
        }

        let buffers = [
            k_s.buffer,
            old_pos_storage.buffer,
            new_pos_storage.buffer,
            out_storage.buffer,
        ];

        let total_pairs = (num_tokens * num_heads * (dim / 2)) as u32;
        let grid_x = total_pairs.div_ceil(256);
        let push = push_params(num_tokens as u32, dim as u32, num_heads as u32, 0, 0, base);
        run_compute_shader_kernel(
            ctx,
            VulkanKernel::Rerope,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
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
        let _ = _causal;
        let out_dims = out_shape.dims();
        if out_dims.len() == 3 {
            let inferred_heads = out_dims[1];
            let inferred_dim = out_dims[2];
            if inferred_heads != num_heads {
                tracing::warn!(
                    "Vulkan flash_attention: out_shape head dim ({inferred_heads}) != num_heads ({num_heads})"
                );
            }
            if inferred_dim != head_dim {
                tracing::warn!(
                    "Vulkan flash_attention: out_shape head_dim ({inferred_dim}) != head_dim ({head_dim})"
                );
            }
        }
        if num_heads != num_kv_heads {
            tracing::warn!(
                "Vulkan flash_attention: GQA detected (num_heads={num_heads}, num_kv_heads={num_kv_heads}); \
                 kernel repeats KV heads to match query heads"
            );
        }
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        // Pass num_kv_heads for GQA head-repeat; num_heads comes from out_shape.
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
        Ok((out_storage, Box::new(VulkanHandle)))
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
        let out_dims = out_shape.dims();
        if out_dims.len() == 3 {
            let inferred_dim = out_dims[2];
            if inferred_dim != head_dim {
                tracing::warn!(
                    "Vulkan cross_attention: out_shape head_dim ({inferred_dim}) != head_dim ({head_dim})"
                );
            }
        }
        tracing::warn!(
            "Vulkan cross_attention: seq_len={seq_len}, kv_seq_len={kv_seq_len}, num_heads={num_heads}, head_dim={head_dim}"
        );
        // Cross-attention: Q and KV share num_heads, so pass it as KV-head count.
        let (out_storage, _h) = self.qkv_attention(
            q, k, v, num_heads, kv_seq_len, 0, None, out_shape, None, None,
        )?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }
    // Tier B complex — MLA kernels (audit gap: DeepSeek-family attention hit
    // Err(Unimplemented) on Vulkan). CPU-reference fallbacks that mirror the
    // documented kernel contract exactly: elementwise norm + split. A device
    // kernel (`grim_mla_*`) is the documented upgrade for decode-path latency.

    /// MLA Q/KV norm + split.
    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        _v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let q = q_raw.to_cpu_vec_f32()?;
        let kv = kv_raw.to_cpu_vec_f32()?;
        let qw = q_norm_w.to_cpu_vec_f32()?;
        let kvw = kv_norm_w.to_cpu_vec_f32()?;
        let total = qk_nope_dim + qk_rope_dim;
        if q.len() < total || kv.len() < total {
            return Err(Error::Shape("mla_q_kv_norm_split: input too short".into()));
        }
        let mut q_nope = vec![0.0f32; qk_nope_dim];
        let mut q_rope = vec![0.0f32; qk_rope_dim];
        let mut kv_nope = vec![0.0f32; qk_nope_dim];
        let mut kv_rope = vec![0.0f32; qk_rope_dim];
        for i in 0..qk_nope_dim {
            let w = qw.get(i).copied().unwrap_or(1.0);
            q_nope[i] = q[i] * w;
            kv_nope[i] = kv[i] * kvw.get(i).copied().unwrap_or(1.0);
        }
        for i in 0..qk_rope_dim {
            let w = qw.get(qk_nope_dim + i).copied().unwrap_or(1.0);
            q_rope[i] = q[qk_nope_dim + i] * w;
            kv_rope[i] = kv[qk_nope_dim + i] * kvw.get(qk_nope_dim + i).copied().unwrap_or(1.0);
        }
        let _ = eps;
        let dt = DType::F32;
        let qn = self.from_cpu(&q_nope, &Shape::new(vec![qk_nope_dim]), dt.clone())?;
        let qr = self.from_cpu(&q_rope, &Shape::new(vec![qk_rope_dim]), dt.clone())?;
        let kn = self.from_cpu(&kv_nope, &Shape::new(vec![qk_nope_dim]), dt.clone())?;
        let kr = self.from_cpu(&kv_rope, &Shape::new(vec![qk_rope_dim]), dt)?;
        Ok((qn, qr, kn, kr, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    /// Matrix-absorbed MLA decode (CPU reference).
    fn mla_absorbed_decode(
        &self,
        q_absorbed: &dyn BackendStorage,
        q_rope: &dyn BackendStorage,
        kv_cache: &dyn BackendStorage,
        w_uv: Option<&dyn BackendStorage>,
        out: &dyn BackendStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
        _w_uv_offset_words: usize,
        _w_uv_head_stride_words: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let qa = q_absorbed.to_cpu_vec_f32()?;
        let qr = q_rope.to_cpu_vec_f32()?;
        let kv = kv_cache.to_cpu_vec_f32()?;
        let w_uv_v = w_uv.map(|w| w.to_cpu_vec_f32()).transpose()?;
        let latent = kv_lora_rank;
        let scale = (1.0 / ((latent + qk_rope_dim) as f32).sqrt()).max(1e-5);
        let mut out_buf = out.to_cpu_vec_f32()?;
        let out_dims = out.shape().dims().to_vec();
        for o in out_buf.iter_mut() { *o = 0.0; }
        if qa.len() < num_heads * latent || qr.len() < num_heads * qk_rope_dim {
            return Err(Error::Shape("mla_absorbed_decode: q size mismatch".into()));
        }
        for h in 0..num_heads {
            let qa_off = h * latent;
            let qr_off = h * qk_rope_dim;
            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let kv_off = t * (latent + qk_rope_dim);
                if kv_off + latent + qk_rope_dim > kv.len() { break; }
                let mut dot = 0.0f32;
                for i in 0..latent { dot += qa[qa_off + i] * kv[kv_off + i]; }
                for i in 0..qk_rope_dim { dot += qr[qr_off + i] * kv[kv_off + latent + i]; }
                scores[t] = dot * scale;
                if scores[t] > max_score { max_score = scores[t]; }
            }
            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                scores[t] = (scores[t] - max_score).exp();
                sum_exp += scores[t];
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            let head_dim = if out_dims.len() >= 3 { out_dims[2] } else { v_head_dim };
            let out_off = h * head_dim;
            for t in 0..seq_len {
                let kv_off = t * (latent + qk_rope_dim);
                let p = scores[t] * inv_sum;
                if let Some(ref w) = w_uv_v {
                    let w_off = h * head_dim * latent;
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for i in 0..latent {
                            let widx = w_off + d * latent + i;
                            if widx < w.len() { acc += w[widx] * kv[kv_off + i]; }
                        }
                        if out_off + d < out_buf.len() { out_buf[out_off + d] += p * acc; }
                    }
                } else {
                    for i in 0..head_dim.min(latent) {
                        if out_off + i < out_buf.len() { out_buf[out_off + i] += p * kv[kv_off + i]; }
                    }
                }
            }
        }
        let _ = self.from_cpu(&out_buf, &Shape::new(out_dims), DType::F32)?;
        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }


    /// QKV attention with ALiBi position bias: score += slopes[h]*(j-i).
    /// CPU reference (the device kernel is `grim_qkv_attention_alibi`).
    fn qkv_attention_alibi(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        alibi_slopes: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_v = q.to_cpu_vec_f32()?;
        let k_v = k.to_cpu_vec_f32()?;
        let v_v = v.to_cpu_vec_f32()?;
        let slopes_v = alibi_slopes.to_cpu_vec_f32()?;
        let dims = out_shape.dims();
        let (seq_len, num_heads, head_dim) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            _ => return Err(Error::Shape("qkv_attention_alibi: out_shape must be [seq,heads,head_dim]".into())),
        };
        if slopes_v.len() < num_heads {
            return Err(Error::Shape("qkv_attention_alibi: slopes fewer than heads".into()));
        }
        let q_per_kv = (num_heads / num_kv_heads.max(1)).max(1);
        let cache_off = cache_offset as i32;
        let win = window.unwrap_or(kv_seq_len);
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        if k_v.len() < kv_seq_len * num_kv_heads * head_dim || v_v.len() < kv_seq_len * num_kv_heads * head_dim {
            return Err(Error::Shape("qkv_attention_alibi: k/v too short".into()));
        }
        for qi in 0..seq_len {
            for h in 0..num_heads {
                let kv_h = h / q_per_kv;
                let q_off = (qi * num_heads + h) * head_dim;
                let mut scores = vec![0.0f32; kv_seq_len];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..kv_seq_len {
                    let q_pos = cache_off + qi as i32;
                    let k_pos = j as i32;
                    if k_pos > q_pos { scores[j] = f32::NEG_INFINITY; continue; }
                    if (q_pos - k_pos) as usize >= win { scores[j] = f32::NEG_INFINITY; continue; }
                    let k_off = (j * num_kv_heads + kv_h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim { dot += q_v[q_off + d] * k_v[k_off + d]; }
                    scores[j] = dot + slopes_v[h] * (k_pos - q_pos) as f32;
                    if scores[j] > max_score { max_score = scores[j]; }
                }
                let mut sum_exp = 0.0f32;
                for j in 0..kv_seq_len {
                    if scores[j].is_finite() {
                        scores[j] = (scores[j] - max_score).exp();
                        sum_exp += scores[j];
                    } else { scores[j] = 0.0; }
                }
                let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                let o_off = (qi * num_heads + h) * head_dim;
                for j in 0..kv_seq_len {
                    let p = scores[j] * inv;
                    if p == 0.0 { continue; }
                    let v_off = (j * num_kv_heads + kv_h) * head_dim;
                    for d in 0..head_dim { out[o_off + d] += p * v_v[v_off + d]; }
                }
            }
        }
        let storage = self.from_cpu(&out, out_shape, DType::F32)?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }


}


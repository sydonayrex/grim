//! Paged/preshtfled attention, chunked extend, state merge, sage + tree attention launchers.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, RocmHandle, arg, as_rocm, dev_ptr, dtype_f32};

impl RocmDevice {
    /// Launch Extend Attention Chunk kernel across context slice [chunk_start, chunk_end).
    pub fn launch_extend_attention_chunk(
        &self,
        q_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        chunk_out_storage: &RocmStorage,
        chunk_lse_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: q has no device ptr".into()))?;
        let k_ptr = k_cache_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: k has no device ptr".into()))?;
        let v_ptr = v_cache_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: v has no device ptr".into()))?;
        let out_ptr = chunk_out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: out has no device ptr".into()))?;
        let lse_ptr = chunk_lse_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: lse has no device ptr".into()))?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut lseptr = lse_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut nkvh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut cstart = chunk_start as i32;
        let mut cend = chunk_end as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let lds_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_extend_attention_chunk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut lseptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut nkvh),
                arg(&mut hd),
                arg(&mut cstart),
                arg(&mut cend),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_bytes,
        )
    }

    /// Launch Log-Sum-Exp Attention State Merging kernel.
    pub fn launch_merge_attn_states(
        &self,
        out_a_storage: &RocmStorage,
        lse_a_storage: &RocmStorage,
        out_b_storage: &RocmStorage,
        lse_b_storage: &RocmStorage,
        out_merged_storage: &RocmStorage,
        lse_merged_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<*mut c_void> {
        let out_a_ptr = out_a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: out_a has no device ptr".into()))?;
        let lse_a_ptr = lse_a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: lse_a has no device ptr".into()))?;
        let out_b_ptr = out_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: out_b has no device ptr".into()))?;
        let lse_b_ptr = lse_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: lse_b has no device ptr".into()))?;
        let out_m_ptr = out_merged_storage.device_ptr.ok_or_else(|| {
            Error::Backend("merge_attn_states: out_merged has no device ptr".into())
        })?;
        let lse_m_ptr = lse_merged_storage.device_ptr.ok_or_else(|| {
            Error::Backend("merge_attn_states: lse_merged has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut a_ptr = out_a_ptr;
        let mut la_ptr = lse_a_ptr;
        let mut b_ptr = out_b_ptr;
        let mut lb_ptr = lse_b_ptr;
        let mut m_ptr = out_m_ptr;
        let mut lm_ptr = lse_m_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;

        self.launch_compute_kernel(
            "grim_merge_attn_states",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_ptr),
                arg(&mut la_ptr),
                arg(&mut b_ptr),
                arg(&mut lb_ptr),
                arg(&mut m_ptr),
                arg(&mut lm_ptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut hd),
            ],
        )
    }

    /// Launch Reshape and Cache into Preshuffled Layout kernel.
    pub fn launch_reshape_and_cache_preshuffled(
        &self,
        key_storage: &RocmStorage,
        value_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        slot_mapping_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<*mut c_void> {
        let k_ptr = key_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("reshape_preshuffled: key has no device ptr".into()))?;
        let v_ptr = value_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("reshape_preshuffled: value has no device ptr".into()))?;
        let kc_ptr = k_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: k_cache has no device ptr".into())
        })?;
        let vc_ptr = v_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: v_cache has no device ptr".into())
        })?;
        let sm_ptr = slot_mapping_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: slot_mapping has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut kcptr = kc_ptr;
        let mut vcptr = vc_ptr;
        let mut smptr = sm_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;

        self.launch_compute_kernel(
            "grim_reshape_and_cache_preshuffled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut kcptr),
                arg(&mut vcptr),
                arg(&mut smptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut bs),
            ],
        )
    }

    /// Launch Preshuffled Paged Attention Decode kernel.
    pub fn launch_preshuffled_paged_attention(
        &self,
        q_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        block_tables_storage: &RocmStorage,
        context_lens_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_seqs: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("preshuffled_paged_attn: q has no device ptr".into()))?;
        let kc_ptr = k_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: k_cache has no device ptr".into())
        })?;
        let vc_ptr = v_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: v_cache has no device ptr".into())
        })?;
        let bt_ptr = block_tables_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: block_tables has no device ptr".into())
        })?;
        let cl_ptr = context_lens_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: context_lens has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: out has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_seqs as u32, num_heads as u32, 1);

        let mut qptr = q_ptr;
        let mut kcptr = kc_ptr;
        let mut vcptr = vc_ptr;
        let mut btptr = bt_ptr;
        let mut clptr = cl_ptr;
        let mut optr = out_ptr;
        let mut nseqs = num_seqs as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut max_b = max_num_blocks_per_seq as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let lds_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_preshuffled_paged_attention",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kcptr),
                arg(&mut vcptr),
                arg(&mut btptr),
                arg(&mut clptr),
                arg(&mut optr),
                arg(&mut nseqs),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut bs),
                arg(&mut max_b),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_bytes,
        )
    }

    /// Fused KV-dequant-attention (WI-R5). [see: `CompressedKvBlock`, `quant_bits`, `k_tensor`, `v_tensor`]
    pub fn kv_dequant_attention_impl(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_format: crate::fusion::KvQuantFormat,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let config = {
            let out_dims = out_shape.dims();
            if out_dims.len() != 3 {
                return Err(Error::Shape(
                    "kv_dequant_attention expects 3-D output shape [seq_len, num_heads, head_dim]"
                        .into(),
                ));
            }
            crate::fusion::KvDequantAttentionConfig {
                enabled: true,
                num_heads: out_dims[1],
                num_kv_heads,
                head_dim: out_dims[2],
                quant_format,
                wavefront_size: self.props.wavefront_size as u32,
            }
        };
        if !config.enabled {
            return Err(Error::Backend(
                "kv_dequant_attention: kernel is gated (KvDequantAttentionConfig.enabled=false)"
                    .into(),
            ));
        }

        if config.num_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
            return Err(Error::Shape(
                "kv_dequant_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "kv_dequant_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            )));
        }
        if config.head_dim > 256 {
            return Err(Error::Shape(format!(
                "kv_dequant_attention supports head_dim <= 256 (got {})",
                config.head_dim
            )));
        }

        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k_tensor)?;
        let ks_s = as_rocm(k_scales)?;
        let v_s = as_rocm(v_tensor)?;
        let vs_s = as_rocm(v_scales)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !ks_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !vs_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "kv_dequant_attention: an input lacks a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];

        // One block per (seq_position, head); block dim 128 for wave32 or 256 for wave64
        let block_dim_x: u32 = if config.wavefront_size == 32 {
            128
        } else {
            256
        };
        let grid_x = (seq_len * config.num_heads) as u32;
        let grid_y = 1u32;
        let shared_mem_bytes = (config.head_dim * 4).min(32768);
        let launch = crate::fusion::HipKernelLaunch {
            grid_dim: HipDim3::new(grid_x, grid_y, 1),
            block_dim: HipDim3::new(block_dim_x, 1, 1),
            shared_mem_bytes,
        };

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;
        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let ks_ptr = dev_ptr(ks_s)?;
        let v_ptr = dev_ptr(v_s)?;
        let vs_ptr = dev_ptr(vs_s)?;

        let num_heads_i = config.num_heads as i32;
        let num_kv_heads_i = config.num_kv_heads as i32;
        let head_dim_i = config.head_dim as i32;
        let seq_len_i = seq_len as i32;
        let kv_seq_len_i = kv_seq_len as i32;
        let cache_offset_i = cache_offset as i32;
        let inv_sqrt_d: f32 = 1.0 / (config.head_dim as f32).sqrt();
        let mut inv_sqrt_d_bits = inv_sqrt_d.to_bits();
        let inv_sqrt_d_ptr = &mut inv_sqrt_d_bits as *mut u32 as *mut f32;
        let inv_sqrt_d_stable = inv_sqrt_d_ptr;
        let quant_bits_i = quant_bits as i32;
        let quant_format_i = config.quant_format.kernel_arg();

        let mut qp = q_ptr;
        let mut kp = k_ptr;
        let mut ksp = ks_ptr;
        let mut vp = v_ptr;
        let mut vsp = vs_ptr;
        let mut op = out_ptr;
        let mut nh = num_heads_i;
        let mut nkv = num_kv_heads_i;
        let mut hd = head_dim_i;
        let mut sl = seq_len_i;
        let mut ksl = kv_seq_len_i;
        let mut co = cache_offset_i;
        let mut isd = inv_sqrt_d;
        let mut qb = quant_bits_i;
        let mut qf = quant_format_i;

        // M4: split-KV FlashDecoding for the quantized-KV path. For
        // single-token decode with a long cache, dequantize K/V to f32 once
        // and take the existing split-KV flash_decode path instead of the
        // single-block fused kernel (same gate as the fp32 qkv_attention
        // FlashDecoding path).
        if seq_len == 1 && kv_seq_len >= self.flash_decode_min_kv() {
            let k_f32 = self.launch_kv_dequant_to_f32(
                k_s,
                ks_s,
                num_kv_heads,
                config.head_dim,
                kv_seq_len,
                quant_bits,
                config.quant_format,
            )?;
            let v_f32 = self.launch_kv_dequant_to_f32(
                v_s,
                vs_s,
                num_kv_heads,
                config.head_dim,
                kv_seq_len,
                quant_bits,
                config.quant_format,
            )?;
            let num_splits = self.flash_decode_split_count(
                q_s,
                &k_f32,
                &v_f32,
                &storage,
                config.num_heads,
                num_kv_heads,
                config.head_dim,
                kv_seq_len,
            );
            let stream = self.launch_flash_decode(
                q_s,
                &k_f32,
                &v_f32,
                &storage,
                config.num_heads,
                num_kv_heads,
                config.head_dim,
                kv_seq_len,
                num_splits,
            )?;
            return Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))));
        }

        let stream = self.launch_compute_kernel(
            "grim_kv_dequant_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qp),
                arg(&mut kp),
                arg(&mut ksp),
                arg(&mut vp),
                arg(&mut vsp),
                arg(&mut op),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut qb),
                arg(&mut qf),
            ],
        )?;

        let _ = (
            qp,
            kp,
            ksp,
            vp,
            vsp,
            op,
            nh,
            nkv,
            hd,
            sl,
            ksl,
            co,
            isd,
            qb,
            inv_sqrt_d_stable,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// M4: dequantize a packed quantized-KV cache into an f32 buffer of shape
    /// [kv_seq_len, num_kv_heads, head_dim] (the layout `launch_flash_decode`
    /// expects). One block per (token, kv_head) row, 256 threads per block.
    fn launch_kv_dequant_to_f32(
        &self,
        tensor: &RocmStorage,
        scales: &RocmStorage,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        quant_bits: u32,
        quant_format: crate::fusion::KvQuantFormat,
    ) -> Result<RocmStorage> {
        let rows = kv_seq_len * num_kv_heads;
        let out = RocmStorage::alloc_gpu(
            &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let t_ptr = dev_ptr(tensor)?;
        let s_ptr = dev_ptr(scales)?;
        let mut o_ptr = dev_ptr(&out)?;

        let mut tp = t_ptr;
        let mut sp = s_ptr;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ksl = kv_seq_len as i32;
        let mut qb = quant_bits as i32;
        let mut qf = quant_format.kernel_arg();

        self.launch_compute_kernel(
            "grim_kv_dequant_to_f32",
            HipDim3::new(rows as u32, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut [
                arg(&mut tp),
                arg(&mut sp),
                arg(&mut o_ptr),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut ksl),
                arg(&mut qb),
                arg(&mut qf),
            ],
        )?;
        Ok(out)
    }

    /// Block-Quantized SageAttention HIP kernel launch.
    pub fn sage_attention_gpu(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid() || !k_s.device_ptr_is_valid() || !v_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sage_attention: inputs lack a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let mut q_ptr = dev_ptr(q_s)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut out_ptr = dev_ptr(&out_storage)?;

        let mut nh_i = num_heads as i32;
        let mut nkv_i = num_kv_heads as i32;
        let mut hd_i = head_dim as i32;
        let mut sl_i = seq_len as i32;
        let mut ksl_i = kv_seq_len as i32;
        let mut sm_scale = 1.0f32 / (head_dim as f32).sqrt();

        let grid = HipDim3::new(num_heads as u32, seq_len.div_ceil(128) as u32, 1);
        let block = HipDim3::new(128, 1, 1);

        self.launch_compute_kernel(
            "grim_sage_attention",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut k_ptr),
                arg(&mut v_ptr),
                arg(&mut out_ptr),
                arg(&mut nh_i),
                arg(&mut nkv_i),
                arg(&mut hd_i),
                arg(&mut sl_i),
                arg(&mut ksl_i),
                arg(&mut sm_scale),
            ],
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Tree-attention wrapper for speculative-decoding verification. [see: `1 + gamma`, `tree_parents`]
    pub fn qkv_attention_paged(
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
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_paged expects 3-D output shape [batch, num_heads, head_dim]".into(),
            ));
        }
        let batch = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let q_s = as_rocm(q)?;
        let bt_s = as_rocm(block_tables)?;
        let k_s = as_rocm(k_pages)?;
        let v_s = as_rocm(v_pages)?;

        if !q_s.device_ptr_is_valid()
            || !bt_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "qkv_attention_paged: inputs lack a valid device pointer".into(),
            ));
        }

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        // WI-X5: autotuner lookup + record side (same treatment as the trait
        // paged path above).
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let paged_key = crate::autotune::KernelKey::paged_attention(
            arch_leak,
            num_heads,
            head_dim,
            kv_seq_len.clamp(1, 1 << 16),
        );
        let mut block_override: Option<u32> = None;
        if let Ok(tuner) = self.autotuner.lock() {
            if let Some(cfg) = tuner.lookup(paged_key) {
                if cfg.block_dim > 0 {
                    block_override = Some(cfg.block_dim);
                }
            }
        }
        if block_override.is_none() {
            block_override = self.autotune_attention_block_dim(
                paged_key,
                self.wavefront_size() as u32 * 4,
                kv_seq_len,
                512,
                |block_x| {
                    crate::launch_paged_attention(
                        self,
                        q_s,
                        bt_s,
                        k_s,
                        v_s,
                        &mut storage,
                        batch as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        max_blocks as u32,
                        page_size as u32,
                        kv_seq_len as u32,
                        cache_offset,
                        0,
                        Some(block_x),
                    )
                },
            );
        }

        crate::launch_paged_attention(
            self,
            q_s,
            bt_s,
            k_s,
            v_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            max_blocks as u32,
            page_size as u32,
            kv_seq_len as u32,
            cache_offset,
            0,
            block_override,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    pub fn tree_attention(
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
        // ─── structural validation ───────────────────────────────────────── [see: `qkv_attention`]
        let out_dims = out_shape.dims();
        if out_dims.len() != 4 {
            return Err(Error::Shape(
                "tree_attention requires 4-D output shape \
                 [batch, 1+gamma, num_heads, head_dim]"
                    .into(),
            ));
        }
        let batch = out_dims[0];
        let one_plus_gamma = out_dims[1];
        let num_heads = out_dims[2];
        let head_dim = out_dims[3];

        if batch == 0 || num_heads == 0 || head_dim == 0 {
            return Err(Error::Shape(
                "tree_attention: zero-sized batch / num_heads / head_dim".into(),
            ));
        }
        if one_plus_gamma == 0 {
            return Err(Error::Shape(
                "tree_attention: 1+gamma must be >= 1 (gamma == 0 still has a root)".into(),
            ));
        }
        // tree_parents must have at least 1+gamma entries.
        if tree_parents.shape().elem_count() < one_plus_gamma {
            return Err(Error::Shape(format!(
                "tree_attention: tree_parents must have >= {} entries (got {})",
                one_plus_gamma,
                tree_parents.shape().elem_count(),
            )));
        }
        // Block dim: 128 threads for Wave32 (gfx1036/RDNA2: 4 Wave32 wavefronts),
        // 256 threads for Wave64 (CDNA: 4 Wave64 wavefronts).
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "tree_attention Phase-3 supports head_dim <= 256 (got {})",
                head_dim
            )));
        }
        // GQA head-count sanity (same rule as `qkv_attention`).
        let gamma = one_plus_gamma - 1;
        if num_kv_heads == 0 || num_kv_heads > num_heads {
            return Err(Error::Shape(format!(
                "tree_attention: num_kv_heads ({}) must be within [1, num_heads] ({})",
                num_kv_heads, num_heads
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "tree_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                num_heads, num_kv_heads
            )));
        }

        // ─── input pointer validation ─────────────────────────────────────
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let tp_s = as_rocm(tree_parents)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !tp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "tree_attention: an input lacks a valid device pointer".into(),
            ));
        }

        // ─── allocate output + launch ──────────────────────────────────
        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let gamma_u32 = gamma as u32;

        // The launcher takes `&dyn BackendStorage` for inputs and [see: `&mut dyn BackendStorage`, `RocmStorage`, `BackendStorage`, `tree_attention`]
        crate::launch_tree_attention(
            self,
            q_s,
            k_s,
            v_s,
            tp_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            gamma_u32,
            kv_seq_len as u32,
            cache_offset,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}

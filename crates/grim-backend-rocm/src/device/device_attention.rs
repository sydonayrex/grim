use std::ffi::c_void;
use std::sync::Mutex;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{AttentionOps, BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, QkvAttentionFusionConfig, QuantMode, RocmHandle, arg, as_rocm, dev_ptr, dtype_f32,
    hipFreeAsync, hipStreamSynchronize, hipSuccess, linear_launch, upload_device_buffer,
};

impl AttentionOps for RocmDevice {
    /// SageAttention dispatch. The GPU entry existed without trait wiring, so callers always hit the
    /// Unimplemented default and fell back to the F32 qkv path; this makes the kernel reachable.
    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.sage_attention_gpu(q, k, v, num_kv_heads, kv_seq_len, out_shape)
    }

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
        let quant_format =
            crate::fusion::KvQuantFormat::from_legacy_quant_bits(quant_bits as u8, true);
        self.kv_dequant_attention_impl(
            q,
            k_tensor,
            k_scales,
            v_tensor,
            v_scales,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            quant_format,
            quant_bits,
            out_shape,
        )
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
        let q_s = as_rocm(q_raw)?;
        let kv_s = as_rocm(kv_raw)?;
        let qw_s = as_rocm(q_norm_w)?;
        let kvw_s = as_rocm(kv_norm_w)?;

        let q_nope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_nope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let q_rope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_rope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let kv_nope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_nope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let kv_rope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_rope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut q_ptr = dev_ptr(q_s)?;
        let mut kv_ptr = dev_ptr(kv_s)?;
        let mut qw_ptr = dev_ptr(qw_s)?;
        let mut kvw_ptr = dev_ptr(kvw_s)?;
        let mut q_nope_ptr = dev_ptr(&q_nope_st)?;
        let mut q_rope_ptr = dev_ptr(&q_rope_st)?;
        let mut kv_nope_ptr = dev_ptr(&kv_nope_st)?;
        let mut kv_rope_ptr = dev_ptr(&kv_rope_st)?;
        let mut nope_i = qk_nope_dim as i32;
        let mut rope_i = qk_rope_dim as i32;
        let mut v_i = v_dim as i32;
        let mut eps_f = eps;

        let total = qk_nope_dim + qk_rope_dim;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mla_q_kv_norm_split",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut kv_ptr),
                arg(&mut qw_ptr),
                arg(&mut kvw_ptr),
                arg(&mut q_nope_ptr),
                arg(&mut q_rope_ptr),
                arg(&mut kv_nope_ptr),
                arg(&mut kv_rope_ptr),
                arg(&mut nope_i),
                arg(&mut rope_i),
                arg(&mut v_i),
                arg(&mut eps_f),
            ],
        )?;

        Ok((
            Box::new(q_nope_st),
            Box::new(q_rope_st),
            Box::new(kv_nope_st),
            Box::new(kv_rope_st),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

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
        w_uv_offset_words: usize,
        w_uv_head_stride_words: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let q_abs = q_absorbed
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mla_absorbed_decode: q_absorbed is not RocmStorage".into())
            })?;
        let q_r = q_rope
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mla_absorbed_decode: q_rope is not RocmStorage".into())
            })?;
        let kv = kv_cache
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mla_absorbed_decode: kv_cache is not RocmStorage".into())
            })?;
        let o = out
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out is not RocmStorage".into()))?;
        let w = w_uv
            .map(|s| {
                s.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                    Error::Backend("mla_absorbed_decode: w_uv is not RocmStorage".into())
                })
            })
            .transpose()?;
        self.launch_mla_absorbed_decode(
            q_abs,
            q_r,
            kv,
            w,
            o,
            num_heads,
            kv_lora_rank,
            qk_rope_dim,
            v_head_dim,
            seq_len,
            w_uv_offset_words,
            w_uv_head_stride_words,
        )?;
        Ok(Box::new(crate::device::handles::RocmHandle::new(Some(
            self.active_stream(),
        ))))
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
        // Compute host-side window_lo per-query position: For full causal attention (window == None), window_lo = 0 for all queries.
        // For sliding-window (window == Some(w)), window_lo = max(0, abs_i - w + 1) is constant.
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                let win_lo = abs_first.saturating_sub(w.saturating_sub(1));
                i32::try_from(win_lo)
                    .map_err(|_| Error::Backend("window_lo exceeds i32::MAX".into()))?
            }
        };
        let config = {
            let out_dims = out_shape.dims();
            let (seq_len, num_heads, head_dim) = if out_dims.len() == 3 {
                (out_dims[0], out_dims[1], out_dims[2])
            } else if out_dims.len() == 2 {
                let seq_len = out_dims[0];
                let hidden_dim = out_dims[1];
                let q_dims = q.shape().dims();
                let head_dim = if q_dims.len() == 3 {
                    q_dims[2]
                } else if q_dims.len() == 2 && num_kv_heads > 0 {
                    q_dims[1] / num_kv_heads
                } else {
                    hidden_dim / num_kv_heads.max(1)
                };
                if head_dim == 0 {
                    return Err(Error::Shape(
                        "qkv_attention head_dim resolved to zero; malformed model dimension".into(),
                    ));
                }
                let num_heads = hidden_dim / head_dim;
                (seq_len, num_heads, head_dim)
            } else {
                return Err(Error::Shape(
                    "qkv_attention expects 2-D [seq_len, hidden_dim] or 3-D [seq_len, num_heads, head_dim] output shape".into(),
                ));
            };
            QkvAttentionFusionConfig {
                enabled: true,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len: seq_len,
                wavefront_size: self.props.wavefront_size as u32,
                quant_mode: QuantMode::Fp32,
            }
        };
        if !config.enabled {
            return Err(Error::Backend(
                "qkv_attention: kernel is gated (QkvAttentionFusionConfig.enabled=false)".into(),
            ));
        }

        // ─── structural validation ──────────────────────────────────────
        if config.num_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
            return Err(Error::Shape(
                "qkv_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "qkv_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            )));
        }
        if config.head_dim > 256 {
            return Err(Error::Shape(format!(
                "qkv_attention Phase 2 supports head_dim <= 256 (got {})",
                config.head_dim
            )));
        }

        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid() || !k_s.device_ptr_is_valid() || !v_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "qkv_attention: inputs lack a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];

        // ─── allocate output + launch ────────────────────────────────────
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;
        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let v_ptr = dev_ptr(v_s)?;

        let mut max_ptr: u64 = 0;
        if let Some(m) = out_max {
            let m_s = as_rocm(m)?;
            max_ptr = dev_ptr(m_s)?;
        }
        let mut sum_ptr: u64 = 0;
        if let Some(s) = out_sum {
            let s_s = as_rocm(s)?;
            sum_ptr = dev_ptr(s_s)?;
        }

        let num_heads_i = config.num_heads as i32;
        let num_kv_heads_i = config.num_kv_heads as i32;
        let head_dim_i = config.head_dim as i32;
        let seq_len_i = seq_len as i32;
        let kv_seq_len_i = kv_seq_len as i32;
        let cache_offset_i = cache_offset as i32;
        let inv_sqrt_d: f32 = 1.0 / (config.head_dim as f32).sqrt();

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut nh = num_heads_i;
        let mut nkv = num_kv_heads_i;
        let mut hd = head_dim_i;
        let mut sl = seq_len_i;
        let mut ksl = kv_seq_len_i;
        let mut co = cache_offset_i;
        let mut isd = inv_sqrt_d;
        let mut wlo = window_lo_i;
        let mut softcap: f32 = self.attn_logit_softcap();
        let mut oproj_ptr: u64 = 0;
        let mut odim: i32 = 0;
        let mut fuseo: i32 = 0;
        let mut alibi_ptr: u64 = 0;
        let mut has_alibi: i32 = 0;

        // Prior RoPE / cache ops were enqueued on the same stream, so stream ordering already guarantees they complete
        // before this kernel reads q/k/v - no host sync needed (each removed sync stalls the whole per-token pipeline).

        // Split-KV FlashDecoding acceleration for long-context single-token decode
        if seq_len == 1
            && kv_seq_len >= 1024
            && window.is_none()
            && out_max.is_none()
            && out_sum.is_none()
            && softcap <= 0.0
        {
            let num_splits = self.flash_decode_split_count(
                q_s,
                k_s,
                v_s,
                &storage,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                kv_seq_len,
            );
            let stream = self.launch_flash_decode(
                q_s,
                k_s,
                v_s,
                &storage,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                kv_seq_len,
                num_splits,
            )?;
            return Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))));
        }

        let mut launch_block_dim = launch.block_dim;
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: "grim_qkv_attention",
            gpu_arch: arch_leak,
            m: config.num_heads,
            n: config.head_dim,
            k: kv_seq_len.clamp(1, 1 << 16),
        };
        let mut tuned_block_dim: Option<u32> = None;
        if let Ok(tuner) = self.autotuner.lock() {
            if let Some(cfg) = tuner.lookup(key) {
                if cfg.block_dim > 0 {
                    launch_block_dim.x = cfg.block_dim;
                    tuned_block_dim = Some(cfg.block_dim);
                }
            }
        }
        if tuned_block_dim.is_none() {
            // WI-X5: record side - on a cache miss (and under the GRIM_ATTENTION_AUTOTUNE gates inside the helper), sweep candidate block dims with real launches and record the winner.
            // The sweep writes the same attention output into `storage`; the real launch below then runs.
            let sweep_winner = self.autotune_attention_block_dim(
                key,
                launch.block_dim.x,
                kv_seq_len,
                512,
                |block_x| {
                    // Fresh arg copies per launch; the outer locals stay
                    // untouched for the real launch below.
                    let mut qptr = q_ptr;
                    let mut kptr = k_ptr;
                    let mut vptr = v_ptr;
                    let mut optr = out_ptr;
                    let mut max_ptr = max_ptr;
                    let mut sum_ptr = sum_ptr;
                    let mut nh = num_heads_i;
                    let mut nkv = num_kv_heads_i;
                    let mut hd = head_dim_i;
                    let mut sl = seq_len_i;
                    let mut ksl = kv_seq_len_i;
                    let mut co = cache_offset_i;
                    let mut isd = inv_sqrt_d;
                    let mut wlo = window_lo_i;
                    let mut softcap = softcap;
                    let mut oproj_ptr: u64 = 0;
                    let mut odim: i32 = 0;
                    let mut fuseo: i32 = 0;
                    let mut alibi_ptr: u64 = 0;
                    let mut has_alibi: i32 = 0;
                    self.launch_compute_kernel(
                        "grim_qkv_attention",
                        launch.grid_dim,
                        HipDim3::new(block_x, 1, 1),
                        &mut [
                            arg(&mut qptr),
                            arg(&mut kptr),
                            arg(&mut vptr),
                            arg(&mut optr),
                            arg(&mut max_ptr),
                            arg(&mut sum_ptr),
                            arg(&mut nh),
                            arg(&mut nkv),
                            arg(&mut hd),
                            arg(&mut sl),
                            arg(&mut ksl),
                            arg(&mut co),
                            arg(&mut isd),
                            arg(&mut wlo),
                            arg(&mut softcap),
                            arg(&mut oproj_ptr),
                            arg(&mut odim),
                            arg(&mut fuseo),
                            arg(&mut alibi_ptr),
                            arg(&mut has_alibi),
                        ],
                    )
                    .map(|_| ())
                },
            );
            if let Some(winner) = sweep_winner {
                launch_block_dim.x = winner;
            }
        }

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch_block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut softcap),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;

        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo, softcap,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );

        // No post-launch sync: the output storage is returned to the caller and any readback (or same-stream reuse of pooled scratch) is ordered by the single active stream.
        // A sync here would serialize the CPU against every attention of every layer of every.

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

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
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let slopes_s = as_rocm(alibi_slopes)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !slopes_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "qkv_attention_alibi: inputs lack a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_alibi: out_shape must be [seq, heads, head_dim]".into(),
            ));
        }
        let (seq_len, num_heads, head_dim) = (out_dims[0], out_dims[1], out_dims[2]);
        if slopes_s.shape().elem_count() < num_heads {
            return Err(Error::Shape(
                "qkv_attention_alibi: alibi_slopes must hold num_heads entries".into(),
            ));
        }

        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => (cache_offset as usize).saturating_sub(w.saturating_sub(1)) as i32,
        };
        let config = QkvAttentionFusionConfig {
            enabled: true,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: seq_len,
            wavefront_size: self.props.wavefront_size as u32,
            quant_mode: QuantMode::Fp32,
        };
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;

        let mut qptr = dev_ptr(q_s)?;
        let mut kptr = dev_ptr(k_s)?;
        let mut vptr = dev_ptr(v_s)?;
        let mut optr = out_ptr;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;
        let mut nh = num_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut ksl = kv_seq_len as i32;
        let mut co = cache_offset as i32;
        let mut isd: f32 = 1.0 / (head_dim as f32).sqrt();
        let mut wlo = window_lo_i;
        let mut softcap: f32 = self.attn_logit_softcap();
        let mut oproj_ptr: u64 = 0;
        let mut odim: i32 = 0;
        let mut fuseo: i32 = 0;
        let mut alibi_ptr = dev_ptr(slopes_s)?;
        let mut has_alibi: i32 = 1;

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut softcap),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo, softcap,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );
        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope: input lacks a valid device pointer".into(),
            ));
        }

        // Partial-rotary / YaRN path: dispatch to grim_rope_yarn which accepts a pre-uploaded inv_freq[]
        // buffer and handles both partial rotary_dim and YaRN magnitude correction entirely on-GPU.
        if !cfg.is_plain() {
            return self.rope_launch_yarn(x_s, positions, cfg, out_shape);
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let d = dim as i32;
        let half = d / 2;
        if positions.len() != s as usize {
            return Err(Error::Shape(
                "rope: positions length must match seq_len".into(),
            ));
        }

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut pos_ptr = upload_device_buffer(self.ordinal, positions)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rope",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
            ],
        )?;

        // pos_ptr is kernel input; release it stream-ordered after the launch
        // (graph-capturable, no host stall).
        unsafe {
            let _ = hipFreeAsync(pos_ptr, stream);
        }

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
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
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let k_s = as_rocm(k)?;
        if !k_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rerope: input lacks a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "Re-RoPE expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let d = dim as i32;
        let half = d / 2;
        if old_positions.len() != s as usize || new_positions.len() != s as usize {
            return Err(Error::Shape(
                "rerope: positions length must match seq_len".into(),
            ));
        }

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut old_pos_ptr = upload_device_buffer(self.ordinal, old_positions)?;
        let mut new_pos_ptr = upload_device_buffer(self.ordinal, new_positions)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rerope",
            grid,
            block,
            &mut [
                arg(&mut k_ptr),
                arg(&mut old_pos_ptr),
                arg(&mut new_pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
            ],
        )?;

        unsafe {
            let _ = hipFreeAsync(old_pos_ptr, stream);
            let _ = hipFreeAsync(new_pos_ptr, stream);
        }

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
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
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_cross_attention(
            q_s,
            k_s,
            v_s,
            &out_storage,
            num_heads,
            head_dim,
            seq_len,
            kv_seq_len,
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SCYTHE-2 WI-5: Paged attention override. [see: `crate::launch_paged_attention`, `grim_qkv_attention_paged`]
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
        // Sliding-window lower bound: None    -> 0 (full causal) Some(w)  -> max(0, cache_offset - (w - 1)).
        // For decode (seq_len==1) this is exact; for prefill it is the per-block conservative lower bound.
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                abs_first.saturating_sub(w.saturating_sub(1)) as i32
            }
        };

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

        let out_dims = out_shape.dims();
        let batch = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        // WI-X5: autotuner lookup + record side for the paged launch.
        // Hot path unchanged on a cache hit; the sweep below only runs under the GRIM_ATTENTION_AUTOTUNE.
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
                        window_lo_i,
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
            window_lo_i,
            block_override,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SCYTHE-2 WI-5: Tree attention override. [see: `crate::launch_tree_attention`, `grim_tree_attention`]
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

        let out_dims = out_shape.dims();
        let batch = out_dims[0];
        let one_plus_gamma = out_dims[1];
        let num_heads = out_dims[2];
        let head_dim = out_dims[3];

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

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
            one_plus_gamma as u32,
            kv_seq_len as u32,
            cache_offset,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}

impl RocmDevice {
    /// GPU-side YaRN / partial-rotary RoPE: computes `inv_freq[]` on the host, uploads it once per call, then dispatches `grim_rope_yarn` entirely on-device.
    /// # Contract - `x_s` must have a valid device pointer (caller checks `device_ptr_is_valid`).
    pub(crate) fn rope_launch_yarn(
        &self,
        x_s: &RocmStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dims = out_shape.dims();
        if dims.len() != 3 || dims[2] != cfg.dim {
            return Err(Error::Shape(format!(
                "rope_launch_yarn: expected [B,S,D={}], got {:?}",
                cfg.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let rotary_dim = cfg.rotary_dim.min(d);
        let rotary_half = rotary_dim / 2;
        let yarn = cfg.yarn;

        if positions.len() != s {
            return Err(Error::Shape(
                "rope_launch_yarn: positions length must match seq_len".into(),
            ));
        }

        // Build the YaRN-ramp-corrected inv_freq[] on the host — O(rotary_half) work,
        // negligible vs kernel launch overhead. This avoids storing per-layer buffers.
        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|i| {
                let freq = 1.0_f32 / cfg.base.powf((2 * i) as f32 / d as f32);
                match yarn {
                    None => freq,
                    Some(y) => {
                        let wavelength = 2.0 * std::f32::consts::PI / freq;
                        let low = y.original_max_pos as f32 / y.beta_slow;
                        let high = y.original_max_pos as f32 / y.beta_fast;
                        if wavelength < high {
                            freq
                        } else if wavelength > low {
                            freq / y.factor
                        } else {
                            let ramp = (y.original_max_pos as f32 / wavelength - y.beta_slow)
                                / (y.beta_fast - y.beta_slow);
                            (1.0 - ramp) * (freq / y.factor) + ramp * freq
                        }
                    }
                }
            })
            .collect();
        let mscale = yarn.map(|y| y.attention_factor).unwrap_or(1.0_f32);

        // Upload positions and inv_freq to device-resident scratch buffers.
        // These are temporary allocations freed after the stream synchronises.
        let mut pos_ptr = upload_device_buffer(self.ordinal, positions)?;
        let mut freq_ptr = upload_device_buffer(self.ordinal, &inv_freq)?;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut b_i = b as i32;
        let mut s_i = s as i32;
        let mut d_i = d as i32;
        let mut rh_i = rotary_half as i32;
        let mut ms_f = mscale;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        // Launch grid covers max(b*s*rotary_half, b*s*copy_len) threads to
        // handle both the rotate pass and the verbatim-copy pass in one kernel launch.
        let copy_len = d - 2 * rotary_half;
        let total = b
            * s
            * rotary_half
                .max(if copy_len > 0 { copy_len } else { 0 })
                .max(1);
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rope_yarn",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut freq_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut rh_i),
                arg(&mut ms_f),
                arg(&mut inter_i),
            ],
        );

        // Free scratch device buffers stream-ordered (after the kernel's
        // reads); graph-capturable and no host stall.
        unsafe {
            let free_stream = stream
                .as_ref()
                .map(|_| self.active_stream())
                .unwrap_or(std::ptr::null_mut());
            let _ = hipFreeAsync(pos_ptr, free_stream);
            let _ = hipFreeAsync(freq_ptr, free_stream);
        }

        let stream = stream?;

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// Item 3: device-base RoPE writing into a CALLER-PROVIDED output buffer.
    /// No allocation inside — required for HIP graph capture (stable pointers
    /// across replays). Same kernel as `rope_dev_base`, different output target.
    pub fn rope_dev_base_into(
        &self,
        q_storage: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        out_storage: &RocmStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        num_heads: usize,
        steps: usize,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let q_s = as_rocm(q_storage)?;
        let pos_s = as_rocm(pos_base_dev)?;
        if !q_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_dev_base_into: input lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rope_dev_base_into expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (num_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "rope_dev_base_into: out_shape middle dim {s} != num_heads({num_heads})*steps({steps})={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;

        let mut out_ptr = dev_ptr(out_storage)?;
        let mut x_ptr = dev_ptr(q_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut heads_i = num_heads as i32;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rope_dev_base",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut heads_i),
            ],
        )?;

        Ok(())
    }

    /// Item 2: device-base RoPE for the decode path. Instead of uploading a
    /// per-layer per-token `positions[]` host vector, the single base position
    /// lives in a device buffer (`pos_base_dev`, one u32) and the kernel derives
    /// each step's position as `base + si` internally. Same fp32 math as `rope`.
    ///
    /// `pos_base_dev` must point to device memory holding one u32 (the absolute
    /// position of step 0); `steps` query positions are rotated by base..base+steps-1.
    /// `q_storage` is the per-head, RoPE-normalized Q/K (already reshaped to
    /// `[1, heads*steps, head_dim]`). Returns the rotated storage.
    pub fn rope_dev_base(
        &self,
        q_storage: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        num_heads: usize,
        steps: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let q_s = as_rocm(q_storage)?;
        let pos_s = as_rocm(pos_base_dev)?;
        if !q_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_dev_base: input lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rope_dev_base expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (num_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "rope_dev_base: out_shape middle dim {s} != num_heads({num_heads})*steps({steps})={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(q_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut heads_i = num_heads as i32;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rope_dev_base",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut heads_i),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SPEED-DOT-OPFUSE (Phase 4a): fused RMSNorm + RoPE for Q/K paths.
    /// Normalizes x across head_dim d with optional norm_weight, then applies RoPE rotation
    /// using positions.
    pub fn rmsnorm_rope(
        &self,
        x_storage: &dyn BackendStorage,
        norm_weight: Option<&dyn BackendStorage>,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let x_s = as_rocm(x_storage)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend("rmsnorm_rope: x lacks valid device ptr".into()));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rmsnorm_rope expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let d = dim as i32;
        let half = d / 2;
        if positions.len() != s as usize {
            return Err(Error::Shape(
                "rmsnorm_rope: positions length must match seq_len".into(),
            ));
        }

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = match norm_weight {
            Some(w) => dev_ptr(as_rocm(w)?)?,
            None => 0u64,
        };
        let mut pos_ptr = upload_device_buffer(self.ordinal, positions)?;
        let mut eps_f = eps;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rmsnorm_rope",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut eps_f),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch LFM2-style fused QKV projection: MXFP4 GEMM (x @ W_qkv) followed by per-head QK-Norm + RoPE (YaRN-aware).
    /// The GEMM result is staged in a scratch buffer (or `out_all` if provided) and consumed.
    pub fn launch_fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_q_storage: &RocmStorage,
        gamma_k_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        q_out_storage: Option<&RocmStorage>,
        k_cache_storage: Option<&RocmStorage>,
        v_cache_storage: Option<&RocmStorage>,
        out_all_storage: Option<&RocmStorage>,
        positions_storage: Option<&RocmStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq_storage: Option<&RocmStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<*mut c_void> {
        let gamma_q_ptr = gamma_q_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_q has no device ptr".into())
        })?;
        let gamma_k_ptr = gamma_k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_k has no device ptr".into())
        })?;

        let n_q = num_q_heads * head_dim;
        let n_k = num_kv_heads * head_dim;
        let n_total = n_q + 2 * n_k;

        // Stage the raw QKV GEMM output. Reuse `out_all` if supplied; otherwise
        // allocate a transient scratch buffer freed after the stream syncs.
        let scratch = if out_all_storage.is_some() {
            None
        } else {
            Some(RocmStorage::alloc_gpu(
                &Shape::from_slice(&[m, n_total]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?)
        };
        let gemm_storage: &RocmStorage = match (out_all_storage, &scratch) {
            (Some(o), _) => o,
            (None, Some(s)) => s,
            (None, None) => unreachable!(),
        };
        let gemm_ptr = gemm_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gemm buffer has no device ptr".into())
        })?;

        // Phase 1: MXFP4 GEMM -> gemm_out (C = x @ W_qkv)
        self.launch_mxfp4_gemm_tiled(
            x_storage,
            w_codes_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: codes ptr".into())
            })?,
            w_exps_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: exps ptr".into())
            })?,
            gemm_storage,
            m,
            n_total,
            k,
        )?;

        // Phase 2: per-head QK-Norm + RoPE -> q_out / k_cache / v_cache
        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let total = m * (num_q_heads + 2 * num_kv_heads);
        let (grid, block) = linear_launch(total);

        let mut gemmptr = gemm_ptr;
        let mut gqptr = gamma_q_ptr;
        let mut gkptr = gamma_k_ptr;
        let mut posptr = positions_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut mm = m as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;

        let stream = self.launch_compute_kernel(
            "grim_qk_norm_rope",
            grid,
            block,
            &mut [
                arg(&mut gemmptr),
                arg(&mut gqptr),
                arg(&mut gkptr),
                arg(&mut posptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
            ],
        )?;

        // The transient scratch buffer returns to the caching allocator when `scratch` drops at scope exit.
        // The previous code additionally hipFree'd the pointer by hand - a double free that also.
        drop(scratch);

        Ok(stream)
    }

    /// Launch FlashDecoding (Split-KV Parallel Attention) across sequence chunks + merge reduction.
    pub fn launch_flash_decode(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        num_splits: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: out has no device ptr".into()))?;

        let num_splits = num_splits.max(1);
        let mid_out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads, head_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_max_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_sum_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut mid_out_ptr = dev_ptr(&mid_out_storage)?;
        let mut mid_max_ptr = dev_ptr(&mid_max_storage)?;
        let mut mid_sum_ptr = dev_ptr(&mid_sum_storage)?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_stage1 = HipDim3::new(num_heads as u32, num_splits as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut nh = num_heads as i32;
        let mut nkvh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut slen = kv_seq_len as i32;
        let mut nsplits = num_splits as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        // Stage 1
        let lds_stage1_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_flash_decode_stage1",
            grid_stage1,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut nh),
                arg(&mut nkvh),
                arg(&mut hd),
                arg(&mut slen),
                arg(&mut nsplits),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_stage1_bytes,
        )?;

        // Stage 2
        let grid_stage2 = HipDim3::new(num_heads as u32, 1, 1);
        let mut optr = out_ptr;
        self.launch_compute_kernel(
            "grim_flash_decode_stage2",
            grid_stage2,
            block_dim,
            &mut [
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut nsplits),
            ],
        )
    }

    /// Launch DeepSeek Multi-Head Latent Attention (MLA) Matrix-Absorbed Decode.
    pub fn launch_mla_absorbed_decode(
        &self,
        q_absorbed: &RocmStorage,
        q_rope: &RocmStorage,
        kv_cache: &RocmStorage,
        w_uv: Option<&RocmStorage>,
        out: &RocmStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
        w_uv_offset_words: usize,
        w_uv_head_stride_words: usize,
    ) -> Result<*mut c_void> {
        let q_abs_ptr = q_absorbed.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_absorbed has no device ptr".into())
        })?;
        let q_rope_ptr = q_rope.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_rope has no device ptr".into())
        })?;
        let kv_ptr = kv_cache.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: kv_cache has no device ptr".into())
        })?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out has no device ptr".into()))?;
        let w_uv_ptr = w_uv.and_then(|s| s.device_ptr).unwrap_or(0);
        let has_w_uv = if w_uv.is_some() { 1i32 } else { 0i32 };

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(num_heads as u32, 1, 1);

        let mut qabsptr = q_abs_ptr;
        let mut qropeptr = q_rope_ptr;
        let mut kvptr = kv_ptr;
        let mut wuvptr = w_uv_ptr;
        let mut optr = out_ptr;
        let mut nh = num_heads as i32;
        let mut lora_r = kv_lora_rank as i32;
        let mut rope_d = qk_rope_dim as i32;
        let mut v_dim = v_head_dim as i32;
        let mut slen = seq_len as i32;
        let mut inv_sqrt = 1.0f32 / ((kv_lora_rank + qk_rope_dim) as f32).sqrt();
        let mut has_w = has_w_uv;
        let mut w_off = w_uv_offset_words as i32;
        let mut w_stride = w_uv_head_stride_words as i32;

        let lds_bytes = 256 * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_mla_absorbed_decode",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qabsptr),
                arg(&mut qropeptr),
                arg(&mut kvptr),
                arg(&mut wuvptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut lora_r),
                arg(&mut rope_d),
                arg(&mut v_dim),
                arg(&mut slen),
                arg(&mut inv_sqrt),
                arg(&mut has_w),
                arg(&mut w_off),
                arg(&mut w_stride),
            ],
            None,
            lds_bytes,
        )
    }

    /// Split-KV count for FlashDecoding: consult the autotuner (persisted in `.autotune_cache/{gpu_target}.json`) keyed by `(num_heads, head_dim, kv_len)`; on miss return the static heuristic.
    /// With `GRIM_ATTENTION_AUTOTUNE=1` (and outside stream capture), a miss instead benchmarks candidate split counts with real.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_decode_split_count(
        &self,
        q_s: &RocmStorage,
        k_s: &RocmStorage,
        v_s: &RocmStorage,
        out: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
    ) -> usize {
        let heuristic = (kv_seq_len / 256).clamp(2, 64);
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: "grim_flash_decode",
            gpu_arch: arch_leak,
            m: num_heads,
            n: head_dim,
            k: kv_seq_len.clamp(1, 1 << 16),
        };
        let Ok(mut tuner) = self.autotuner.lock() else {
            return heuristic;
        };
        if let Some(cfg) = tuner.lookup(key) {
            return cfg.tile_kv.max(1) as usize;
        }
        let tune_enabled = std::env::var("GRIM_ATTENTION_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !tune_enabled || self.active_capture_stream().is_some() {
            return heuristic;
        }

        // Bench candidate splits: real launches on the active stream, timed wall-clock (launch + synchronize).
        // Each candidate runs 3 iterations after 1 warmup; the minimum wins.
        let mut candidates: Vec<usize> = [2usize, 4, 8, 16, 32, 64]
            .into_iter()
            .filter(|&s| s <= kv_seq_len.max(2))
            .collect();
        if !candidates.contains(&heuristic) {
            candidates.push(heuristic);
        }
        let mut best = (heuristic, f64::INFINITY);
        for &splits in &candidates {
            let mut best_ms = f64::INFINITY;
            for iter in 0..4 {
                if let Err(e) = self.launch_flash_decode(
                    q_s,
                    k_s,
                    v_s,
                    out,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_seq_len,
                    splits,
                ) {
                    // A failing candidate (e.g. LDS overflow at high split
                    // counts) is simply not viable; skip it.
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                if iter == 0 {
                    continue; // warmup
                }
                let t = std::time::Instant::now();
                if let Err(e) = self.launch_flash_decode(
                    q_s,
                    k_s,
                    v_s,
                    out,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_seq_len,
                    splits,
                ) {
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                let stream = self.active_stream();
                if unsafe { hipStreamSynchronize(stream) } != hipSuccess {
                    best_ms = f64::INFINITY;
                    break;
                }
                let ms = t.elapsed().as_secs_f64() * 1e3;
                best_ms = best_ms.min(ms);
            }
            if best_ms < best.1 {
                best = (splits, best_ms);
            }
        }
        if best.1.is_finite() {
            let cfg = crate::autotune::AutotuneConfig {
                block_dim: 256,
                tile_kv: best.0 as u32,
                grid_stride: 1,
                cycles_per_invocation: (best.1 * 1e6) as u64,
                spec_gamma: 4,
                spec_acceptance_threshold: 0.6,
                spec_alpha: 0.0,
                split_k: 0,
            };
            let _ = tuner.record(key, cfg);
            let _ = self.save_autotune_cache(std::path::Path::new(&format!(
                ".autotune_cache/{}.json",
                self.gpu_target
            )));
            best.0
        } else {
            heuristic
        }
    }

    /// WI-X5: RECORD side of the attention block-dim autotuner, shared by the dense `grim_qkv_attention` lookup (`qkv_attention`) and the paged `grim_qkv_attention_paged` launch sites.
    /// Callers consult `tuner.lookup(key)` FIRST; this helper runs only on a cache miss so the hot.
    pub(crate) fn autotune_attention_block_dim(
        &self,
        key: crate::autotune::KernelKey,
        fallback_block_dim: u32,
        kv_seq_len: usize,
        min_kv_len: usize,
        mut launch_one: impl FnMut(u32) -> Result<()>,
    ) -> Option<u32> {
        let tune_enabled = std::env::var("GRIM_ATTENTION_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !tune_enabled || self.active_capture_stream().is_some() || kv_seq_len < min_kv_len {
            return None;
        }
        // Double-check under the tuner lock: another thread may have recorded
        // this key between the caller's lookup miss and this sweep.
        if let Ok(tuner) = self.autotuner.lock() {
            if tuner.lookup(key).is_some() {
                return None;
            }
        }
        // At most one sweep attempt per key per process. A failed sweep also
        // lands here — retrying every call would stall the decode loop.
        static SWEPT_KEYS: std::sync::OnceLock<Mutex<std::collections::HashSet<u64>>> =
            std::sync::OnceLock::new();
        let swept = SWEPT_KEYS.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
        let key_hash = seahash::hash(
            format!(
                "{}|{}|{}|{}|{}",
                key.kernel, key.gpu_arch, key.m, key.n, key.k
            )
            .as_bytes(),
        );
        match swept.lock() {
            Ok(mut set) => {
                if !set.insert(key_hash) {
                    return None;
                }
            }
            Err(_) => return None,
        }

        let wf = self.wavefront_size() as u32;
        let candidates: Vec<u32> = [64u32, 128, 256]
            .into_iter()
            .filter(|&d| d % wf == 0 && d / wf <= 8 && d != fallback_block_dim)
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let mut best = (fallback_block_dim, f64::INFINITY);
        for cand in std::iter::once(fallback_block_dim).chain(candidates) {
            let mut best_ms = f64::INFINITY;
            for rep in 0..3 {
                if launch_one(cand).is_err() {
                    // Non-viable candidate (launch/arg error): disqualify.
                    best_ms = f64::INFINITY;
                    break;
                }
                if rep == 0 {
                    // Warmup: drain it so timed reps measure only themselves.
                    let _ = unsafe { hipStreamSynchronize(self.active_stream()) };
                    continue;
                }
                let t0 = std::time::Instant::now();
                if unsafe { hipStreamSynchronize(self.active_stream()) } != hipSuccess {
                    best_ms = f64::INFINITY;
                    break;
                }
                best_ms = best_ms.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            if best_ms < best.1 {
                best = (cand, best_ms);
            }
        }
        if !best.1.is_finite() {
            return None;
        }
        let cfg = crate::autotune::AutotuneConfig {
            block_dim: best.0,
            tile_kv: 64,
            grid_stride: 1,
            cycles_per_invocation: (best.1 * 1e6) as u64,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: 0,
        };
        if let Ok(mut tuner) = self.autotuner.lock() {
            let _ = tuner.record(key, cfg);
        }
        let _ = self.save_autotune_cache(std::path::Path::new(&format!(
            ".autotune_cache/{}.json",
            self.gpu_target
        )));
        Some(best.0)
    }

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

    /// Launch Multimodal 3D Rotary Position Embedding (M-RoPE) for Q and K tensors.
    pub fn launch_mrope_qk(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        positions_storage: &RocmStorage,
        num_tokens: usize,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        section_t: usize,
        section_h: usize,
        section_w: usize,
        rope_theta: f32,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: k has no device ptr".into()))?;
        let pos_ptr = positions_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: positions has no device ptr".into()))?;

        let block_dim = HipDim3::new((rotary_dim / 2) as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, (num_q_heads + num_k_heads) as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut posptr = pos_ptr;
        let mut nt = num_tokens as i32;
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_k_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut st = section_t as i32;
        let mut sh = section_h as i32;
        let mut sw = section_w as i32;
        let mut theta = rope_theta;

        self.launch_compute_kernel(
            "grim_mrope_qk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut posptr),
                arg(&mut nt),
                arg(&mut nqh),
                arg(&mut nkh),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut st),
                arg(&mut sh),
                arg(&mut sw),
                arg(&mut theta),
            ],
        )
    }

    // NOTE: `qkv_attention` was promoted to the `BackendDevice` trait [see: `impl BackendDevice for RocmDevice`]

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

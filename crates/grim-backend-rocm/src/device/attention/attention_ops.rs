//! The `impl AttentionOps for RocmDevice` trait-required block, kept whole.


use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{AttentionOps, BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, QkvAttentionFusionConfig, QuantMode, RocmHandle, arg, as_rocm, dev_ptr, dtype_f32, hipFreeAsync, linear_launch, upload_device_buffer };


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
        let quant_format = match std::env::var("GRIM_KV_QUANT_FORMAT").as_deref() {
            Ok("q4khalf") | Ok("q4k_half") | Ok("3") => crate::fusion::KvQuantFormat::Q4KHalf,
            Ok("q4k") | Ok("2") => crate::fusion::KvQuantFormat::Q4K,
            Ok("q8_0") | Ok("q8") | Ok("1") => crate::fusion::KvQuantFormat::Q8_0,
            Ok("legacy") => crate::fusion::KvQuantFormat::from_legacy_quant_bits(quant_bits as u8, true),
            _ => match quant_bits {
                3 => crate::fusion::KvQuantFormat::Q4KHalf,
                _ => crate::fusion::KvQuantFormat::from_legacy_quant_bits(quant_bits as u8, true),
            },
        };
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
            && kv_seq_len >= self.flash_decode_min_kv()
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

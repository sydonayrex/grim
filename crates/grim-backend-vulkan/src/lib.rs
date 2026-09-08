//! Vulkan backend for Grim - GPU compute via Vulkan 1.1 SPIR-V.
//! Modularized architecture (mirrors CUDA/ROCm backends): - `ffi.rs`    - Vulkan API types, constants, extern "C" declarations.

pub mod autotune;
pub mod caps;
pub mod collective;
pub mod context;
pub mod device;
pub mod ffi;
pub mod fsdp;
pub mod graph_capture;
pub mod hugepage;
pub mod kernel;
pub mod storage;

pub use autotune::{GemmOp, ShapeClass, VulkanAutotuner, VulkanTileConfig};
pub use caps::VulkanCaps;
pub use hugepage::VulkanHugePageBuffer;
pub use kernel::VulkanKernel;
pub use kernel::{binding_count, spirv_for};
pub use storage::{VulkanHandle, VulkanStorage};

use std::ffi::c_void;
use std::sync::Mutex;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
pub use grim_tensor::{
    ArithType, AttentionOps, AutogradOps, BackendDevice, BackendStorage, CollectiveOps,
    CoreTensorOps, ElementwiseOps, FusionOps, GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps,
    RecurrentOps, SamplingOps, ScythePlacement, Shape,
};

// Re-exported from submodules for use within trait impls.
pub(crate) use context::QUEUE_LOCK;
pub(crate) use context::global_context;
use ffi::*;
pub(crate) use kernel::{push_params, run_compute_shader, run_compute_shader_kernel};

// VulkanDevice - all trait implementations and kernel dispatch

/// Vulkan device handle.
#[derive(Debug)]
pub struct VulkanDevice {
    pub caps: VulkanCaps,
    /// Persistent autotuner — survives across matmul calls so a previously measured winner on
    /// this GPU (loaded from disk at construction) is reused instead of re-searched each call.
    autotuner: Mutex<VulkanAutotuner>,
    /// Optional multi-GPU communicator. `None` = single-GPU mode (default).
    pub communicator: Option<collective::VkCommunicator>,
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        // A cloned handle shouldn't share tuning state; give it a fresh (empty) autotuner.
        Self {
            caps: self.caps.clone(),
            autotuner: Mutex::new(VulkanAutotuner::new()),
            communicator: self.communicator.clone(),
        }
    }
}

impl VulkanDevice {
    /// Constructs a new Vulkan device. Threads the real adapter identity (queried in
    /// `VulkanContext::init`) into the device caps so `vendor_id`/`device_id`/`device_name` reflect the actual physical device.
    pub fn new() -> Self {
        let caps = {
            let guard = global_context();
            match guard.as_ref() {
                Some(ctx) => VulkanCaps::probe_default(
                    ctx.device_name.clone(),
                    ctx.vendor_id,
                    ctx.device_id,
                    ctx.driver_version,
                ),
                None => {
                    // Last-resort fallback only — no live context to query.
                    VulkanCaps::probe_default("Vulkan Compute Device".into(), 0x1002, 0x744c, 1)
                }
            }
        };
        let autotuner = VulkanAutotuner::new();
        // Restore prior tuning for this hardware fingerprint so repeat shapes hit the cache.
        autotuner.load_cache(&caps);
        Self {
            caps,
            autotuner: Mutex::new(autotuner),
            communicator: None,
        }
    }

    /// Attach a multi-GPU communicator so `all_reduce` dispatches the ring-allreduce shader across device pairs.
    /// Single-GPU callers leave this as `None`.
    pub fn with_communicator(mut self, comm: collective::VkCommunicator) -> Self {
        self.communicator = Some(comm);
        self
    }

    pub fn caps(&self) -> &VulkanCaps {
        &self.caps
    }

    pub fn hw_fingerprint(&self) -> u64 {
        self.caps.cache_key_hash()
    }

    /// Probes the system for available Vulkan GPUs.
    pub fn probe() -> Result<Vec<VulkanDevice>> {
        let has_ctx = global_context().is_some();
        if has_ctx {
            Ok(vec![VulkanDevice::new()])
        } else {
            Ok(vec![])
        }
    }

    /// Fused QKV attention compute shader dispatch on Vulkan GPU.
    /// When `window` is `Some(w)` the dedicated `QkvAttentionSwa` kernel is dispatched with a host-computed `window_lo =.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_attention_inner(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        _kv_seq_len: usize,
        cache_offset: u32,
        out: &Shape,
        _out_max: Option<&dyn BackendStorage>,
        _out_sum: Option<&dyn BackendStorage>,
        window: Option<usize>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention q is not VulkanStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention v is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [q_s.buffer, k_s.buffer, v_s.buffer, out_storage.buffer];
        let total_work = (seq_len * num_heads) as u32;
        let grid_x = total_work.div_ceil(256);

        let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();

        if let Some(w) = window {
            // Sliding-window: dispatch QkvAttentionSwa.
            // window_lo is the block-minimum lower bound max(0, cache_offset - w + 1); the kernel's causal.
            let abs_first = cache_offset as usize;
            let window_lo = abs_first.saturating_sub(w.saturating_sub(1)) as u32;
            // 8 × u32 = 32 bytes Params block:
            // seq_len, head_dim, num_heads, num_kv_heads, cache_offset, inv_sqrt_d(f32 bits), window_lo, has_window(=1)
            let push: [u32; 8] = [
                seq_len as u32,
                head_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                cache_offset,
                inv_sqrt_d.to_bits(),
                window_lo,
                1u32,
            ];
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::QkvAttentionSwa,
                &buffers,
                grid_x,
                1,
                1,
                Some(&push),
            )?;
        } else {
            // Full causal attention.
            let push = push_params(
                seq_len as u32,
                head_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                cache_offset,
                inv_sqrt_d,
            );
            run_compute_shader_kernel(
                ctx,
                VulkanKernel::QkvAttention,
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
}

impl Default for VulkanDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract raw bytes from a VulkanStorage buffer.
/// Host-visible buffers are read via `vkMapMemory`; device-local buffers are read back through a staging copy.
pub fn extract_raw_bytes(storage: &dyn BackendStorage) -> Result<Vec<u8>> {
    if let Some(b_vk) = storage.as_any().downcast_ref::<VulkanStorage>() {
        b_vk.read_raw_bytes()
    } else {
        Err(Error::Backend(
            "extract_raw_bytes: storage is not VulkanStorage; \
             cannot extract raw bytes safely"
                .into(),
        ))
    }
}

impl VulkanDevice {
    /// On-device quantization for Vulkan.
    pub fn quantize_on_device(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan quantize: input x is not VulkanStorage".into())
        })?;
        let total = x.shape().elem_count();
        let (kernel, out_bytes, output_dtype) = match format {
            QuantFormat::Q8_0 => {
                let n_blocks = total.div_ceil(32);
                (
                    VulkanKernel::QuantQ80,
                    n_blocks * 34,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                )
            }
            QuantFormat::Fp8 => {
                // T1 caps gate: a device without FP8 shader support must not dispatch the fp8 blob.
                if !self.caps.supports_quant_format(QuantFormat::Fp8) {
                    return Err(Error::Backend(
                        "Vulkan quantize_on_device: FP8 not supported on this device".into(),
                    ));
                }
                (
                    VulkanKernel::QuantFp8,
                    4 + total,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                    },
                )
            }

            other => {
                return Err(Error::Backend(format!(
                    "Vulkan quantize_on_device: unsupported format {:?}",
                    other
                )));
            }
        };

        let out_shape = Shape::from_slice(&[out_bytes]);
        let (ctx_device, ctx_physical_device) = {
            let ctx_guard = global_context();
            let ctx = ctx_guard
                .as_ref()
                .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
            (ctx.device, ctx.physical_device)
        };

        let out_storage = VulkanStorage::alloc_device_local_gpu(
            &out_shape,
            output_dtype,
            ctx_device,
            ctx_physical_device,
        )?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let buffers = [x_s.buffer, out_storage.buffer];
        let push = push_params(total as u32, 0, 0, 0, 0, 0.0);
        let grid_x = match kernel {
            VulkanKernel::QuantQ80 => total.div_ceil(32) as u32,
            VulkanKernel::QuantFp8 => total.div_ceil(256) as u32,
            _ => unreachable!(),
        };

        run_compute_shader_kernel(ctx, kernel, &buffers, grid_x, 1, 1, Some(&push))?;
        Ok((Box::new(out_storage), Box::new(VulkanHandle)))
    }

    /// Fused grouped MoE dispatch (WI-M5) - `gate+up` SiLU combine + `down`, accumulated per routed (token, expert) pair into `out`.
    /// Mirrors the ROCm `grim_moe_fused_dispatch` P-DAFD contract: the host pre-expands top-k routing into flat `router_tokens`/`router_experts`/ `router_weights`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        router_tokens: &dyn BackendStorage,
        router_experts: &dyn BackendStorage,
        router_weights: &dyn BackendStorage,
        out_shape: &Shape,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        batch: u32,
        routed_scaling_factor: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan moe_fused_dispatch: x is not VulkanStorage".into())
        })?;
        let gw_s = gate_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_fused_dispatch: gate_w is not VulkanStorage".into())
            })?;
        let uw_s = up_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_fused_dispatch: up_w is not VulkanStorage".into())
            })?;
        let dw_s = down_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_fused_dispatch: down_w is not VulkanStorage".into())
            })?;
        let tok_s = router_tokens
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend(
                    "Vulkan moe_fused_dispatch: router_tokens is not VulkanStorage".into(),
                )
            })?;
        let exp_s = router_experts
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend(
                    "Vulkan moe_fused_dispatch: router_experts is not VulkanStorage".into(),
                )
            })?;
        let wt_s = router_weights
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend(
                    "Vulkan moe_fused_dispatch: router_weights is not VulkanStorage".into(),
                )
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        // Output must be host-visible so we can zero-initialise it via vkMapMemory before the
        // kernel dispatch; device-local memory cannot be mapped on discrete GPUs (NVIDIA/AMD dGPU).
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        unsafe {
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let res = vkMapMemory(
                ctx.device,
                out_storage.memory,
                0,
                out_storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            );
            if res != VK_SUCCESS || mapped.is_null() {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed with status {res} (mapped: {mapped:?})"
                )));
            }
            std::ptr::write_bytes(mapped, 0, out_storage.bytes);
            vkUnmapMemory(ctx.device, out_storage.memory);
        }

        let push = push_params(hidden, inter, num_experts, batch, 0, routed_scaling_factor);

        // Number of routed (token, expert) pairs = router_tokens length / 4 (u32 bytes).
        let num_pairs = (tok_s.bytes / std::mem::size_of::<u32>()) as u32;
        let grid_x = num_pairs.max(1);

        let buffers = [
            x_s.buffer,
            gw_s.buffer,
            uw_s.buffer,
            dw_s.buffer,
            tok_s.buffer,
            exp_s.buffer,
            wt_s.buffer,
            out_storage.buffer,
        ];

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::MoeFusedDispatch,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| {
            Error::Backend(format!(
                "Vulkan moe_fused_dispatch GPU dispatch failed: {e}"
            ))
        })?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// Upload a host `f32` slice into a freshly-allocated device buffer.
    /// Used to stage small CPU-side routing arrays (token/expert/weight) and flattened expert weights for `moe_fused_dispatch`.
    pub fn upload_f32(&self, data: &[f32], shape: &Shape) -> Result<Box<dyn BackendStorage>> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(&bytes, shape, DType::F32)
    }

    /// Upload a host `u32` slice into a freshly-allocated device buffer.
    pub fn upload_u32(&self, data: &[u32], shape: &Shape) -> Result<Box<dyn BackendStorage>> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(
            &bytes,
            shape,
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
        )
    }

    fn upload_bytes(
        &self,
        bytes: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage = VulkanStorage::alloc_gpu(shape, dtype, ctx.device, ctx.physical_device)?;
        unsafe {
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let res = vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed with status {res}"
                )));
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, storage.bytes);
            vkUnmapMemory(ctx.device, storage.memory);
        }
        Ok(Box::new(storage))
    }

    /// Op-tagged GEMM. `op` drives the shape-classifier (via `search_tile_config`): a `LmHead` tag
    /// routes to the wide-N TLOLog tile; everything else classifies by shape.
    pub fn matmul_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        op: Option<GemmOp>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan matmul: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan matmul: input b is not VulkanStorage".into()))?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("Vulkan matmul: inputs must be 2D".into()));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }
        if out_shape.dims() != [m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {out_shape:?}"
            )));
        }

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        // Fall back to exact CPU calculation for small matrices where GPU tile workgroups
        // (min 16x16 or 32x32) or RADV driver precision causes numerical artifacts.
        if m < 16 || n < 16 || k < 16 {
            drop(ctx_guard);
            let a_vec = a.to_cpu_vec_f32()?;
            let b_vec = b.to_cpu_vec_f32()?;
            let mut c_vec = vec![0.0f32; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a_vec[row * k + p] * b_vec[p * n + col];
                    }
                    c_vec[row * n + col] = sum;
                }
            }
            let out_storage = self.from_cpu(&c_vec, out_shape, a.dtype())?;
            return Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)));
        }

        // Persistent autotuner: cached (loaded-from-disk or in-memory) winner on a repeat shape
        // is reused; on a miss the winner is chosen and persisted (search_tile_config saves).
        let tile_config = {
            let autotuner = self.autotuner.lock().unwrap();
            autotuner.search_tile_config(&self.caps, m, n, k, op)
        };
        let shape_class = match op {
            Some(GemmOp::LmHead) => ShapeClass::TLOLog,
            _ => ShapeClass::classify(m, n, k),
        };

        // Use the precompiled, autotuner-matched matmul blob (block size 64 or 32, or BF16).
        // Caps-gate BF16: a device without BF16 shader support must not pick the BF16 blob.
        let kernel = if (a.dtype().arith == ArithType::BF16 || b.dtype().arith == ArithType::BF16)
            && a_s.bytes == m * k * 2
            && self.caps.supports_bf16
        {
            VulkanKernel::Matmul64Bf16
        } else if shape_class == ShapeClass::TLOLog {
            // Wide-N (vocab-dominated) output column: route to the Matmul64 surface.
            VulkanKernel::Matmul64
        } else if tile_config.block_m == 64 {
            VulkanKernel::Matmul64
        } else {
            VulkanKernel::Matmul32
        };
        let spirv_source: Vec<u8> = spirv_for(kernel).to_vec();

        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        // Try GPU dispatch first
        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = n.div_ceil(tile_config.block_n as usize) as u32;
        let grid_y = m.div_ceil(tile_config.block_m as usize) as u32;

        let push = push_params(0, 0, k as u32, n as u32, m as u32, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, grid_y, 1, Some(&push)).map_err(
            |e| grim_tensor::Error::Backend(format!("Vulkan matmul GPU dispatch failed: {e}")),
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// Public hook for the engine layer to tag the lm_head / logit-projection GEMM, so it is classified as `ShapeClass::TLOLog` (op-identity) and gets the wide-N tile candidate set regardless of M.
    /// This is the vulkan-catch-up.md §3 T3 dispatch-layer tag.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, Some(GemmOp::LmHead))
    }

    /// Multi-GPU all-reduce via the ring-allreduce shader.
    /// Structural scaffold: dispatches `VulkanKernel::RingAllReduce` using the communicator's topology.
    fn all_reduce_multi_gpu(
        &self,
        inputs: &[&dyn BackendStorage],
        comm: &collective::VkCommunicator,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Multi-GPU all-reduce requires P2P buffer copy across device pairs, which is the transport layer this phase scaffolds.
        // The ring-allreduce shader in `ring_allreduce.comp` is the reduce step once transport exists.
        let _ = (inputs, comm);
        Err(Error::Backend(
            "all_reduce_multi_gpu: P2P transport not yet wired (ring-allreduce shader is structural)".into()
        ))
    }
}

impl VulkanDevice {
    /// Shared scalar-op dispatch (mul/add/sub/div by a broadcast scalar) —
    /// Tier A semi-parity: one f32 push-constant, one elementwise pass.
    fn run_scalar_op(
        &self,
        kernel: VulkanKernel,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
        op_name: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend(format!("Vulkan {op_name} x is not VulkanStorage")))?;
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

        let spirv_source: Vec<u8> = spirv_for(kernel).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, scalar);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;
        drop(ctx_guard);

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// Shared single-workgroup reduction dispatch (sum / max / argmax).
    /// `out_elems` is 1 for value reductions, 1 for argmax (index packed as uint bits).
    fn run_reduction(
        &self,
        kernel: VulkanKernel,
        x: &dyn BackendStorage,
        out_elems: usize,
    ) -> Result<Vec<f32>> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan reduction x is not VulkanStorage".into()))?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_shape = Shape::new(vec![out_elems]);
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            &out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let spirv_source: Vec<u8> = spirv_for(kernel).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = x.shape().elem_count();
        if n == 0 {
            return Err(Error::Backend("Vulkan reduction: empty tensor".into()));
        }

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        // One workgroup: the reduction shaders loop over the whole input and tree-combine in shared memory
        // (n up to a few million is fine - the strided loop is bandwidth-bound either way).
        run_compute_shader(ctx, &spirv_source, &buffers, 1, 1, 1, Some(&push))?;
        drop(ctx_guard);

        out_storage.to_cpu_vec_f32()
    }
}

impl ElementwiseOps for VulkanDevice {
    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul_scalar x is not VulkanStorage".into()))?;
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

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::MulScalar).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, scalar);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan sqrt x is not VulkanStorage".into()))?;
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

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Sqrt).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan recip x is not VulkanStorage".into()))?;
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

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Recip).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.run_scalar_op(VulkanKernel::AddScalar, x, scalar, out_shape, "add_scalar")
    }

    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.run_scalar_op(VulkanKernel::SubScalar, x, scalar, out_shape, "sub_scalar")
    }

    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if scalar == 0.0 {
            return Err(Error::Backend(
                "Vulkan div_scalar: division by zero scalar".into(),
            ));
        }
        self.run_scalar_op(VulkanKernel::DivScalar, x, scalar, out_shape, "div_scalar")
    }

    fn sub(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan sub a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan sub b is not VulkanStorage".into()))?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Sub).to_vec();
        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let n = out.elem_count();
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn reduce_sum(&self, x: &dyn BackendStorage) -> Result<f32> {
        if x.shape().elem_count() == 0 {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        let v = self.run_reduction(VulkanKernel::ReduceSum, x, 1)?;
        Ok(v[0])
    }

    fn reduce_max(&self, x: &dyn BackendStorage) -> Result<f32> {
        if x.shape().elem_count() == 0 {
            return Err(Error::Backend("reduce_max: empty tensor".into()));
        }
        let v = self.run_reduction(VulkanKernel::ReduceMax, x, 1)?;
        Ok(v[0])
    }

    fn argmax(&self, x: &dyn BackendStorage) -> Result<u32> {
        if x.shape().elem_count() == 0 {
            return Err(Error::Backend("argmax: empty tensor".into()));
        }
        let v = self.run_reduction(VulkanKernel::Argmax, x, 1)?;
        Ok(f32::to_bits(v[0]))
    }
}

impl SamplingOps for VulkanDevice {
    /// Tier A (semi-parity): the greedy path samples via the device argmax kernel - no logit round-trip.
    /// The stochastic path still needs top-k/top-p filtering on the host (a GPU top-p is a.
    fn sample_on_device(
        &self,
        logits: &dyn BackendStorage,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        seed: u64,
    ) -> Result<u32> {
        if temperature <= 0.0 || (top_k == 1 && (top_p >= 1.0 || top_p <= 0.0)) {
            if std::env::var("SAMP_DBG").is_ok() {
                eprintln!("SDBG greedy->argmax");
            }
            let r = self.argmax(logits);
            if std::env::var("SAMP_DBG").is_ok() {
                eprintln!("SDBG argmax done: {r:?}");
            }
            return r;
        }
        if std::env::var("SAMP_DBG").is_ok() {
            eprintln!("SDBG stochastic path");
        }
        let cpu_logits = logits.to_cpu_vec_f32()?;
        if cpu_logits.is_empty() {
            return Err(Error::Backend("sample_on_device: empty logits".into()));
        }
        let mut scaled: Vec<(usize, f32)> = cpu_logits
            .iter()
            .enumerate()
            .map(|(idx, &l)| (idx, l / temperature))
            .collect();
        scaled.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && (top_k as usize) < scaled.len() {
            scaled.truncate(top_k as usize);
        }
        let max_logit = scaled[0].1;
        if !max_logit.is_finite() {
            return Err(Error::Backend(format!(
                "sample_on_device: logits have non-finite maximum ({max_logit})"
            )));
        }
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<(usize, f32)> = scaled
            .iter()
            .map(|&(idx, l)| {
                let p = (l - max_logit).exp();
                exp_sum += p;
                (idx, p)
            })
            .collect();
        for p in probs.iter_mut() {
            p.1 /= exp_sum.max(1e-12);
        }
        if top_p > 0.0 && top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cum += p;
                if cum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
        }
        let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
        state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
        let r = ((state ^ (state >> 31)) as f32) / (u64::MAX as f32);
        let mut cum = 0.0f32;
        for &(idx, p) in &probs {
            cum += p;
            if r <= cum {
                return Ok(idx as u32);
            }
        }
        Ok(probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0))
    }
}

impl FusionOps for VulkanDevice {
    /// Tier B: real device path - silu(gate)*up on device, then quantize on device via the existing `quantize_on_device` helper.
    /// No host round-trip (the trait default decomposes into silu_mul + a host quantize).
    fn silu_mul_quantize(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let (y_unquant, handle) = self.silu_mul(gate, up, out_shape)?;
        handle.synchronize()?;
        let (q_bytes, q_handle) = self.quantize_on_device(y_unquant.as_ref(), format)?;
        q_handle.synchronize()?;
        let scale_storage = self.zeros(&Shape::from_slice(&[1]), grim_tensor::DType::F32)?;
        Ok((
            q_bytes,
            scale_storage,
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// Tier B: broadcast a 1-D bias `[out_dim]` into `[batch, out_dim]`.
    fn broadcast_bias(
        &self,
        bias: &dyn BackendStorage,
        _batch: usize,
        out_dim: usize,

        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let b_s = bias
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan broadcast_bias: bias is not VulkanStorage".into())
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
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::BroadcastBias).to_vec();
        let buffers = [b_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = n.div_ceil(256) as u32;
        let push = push_params(n as u32, 0, 0, out_dim as u32, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;
        drop(ctx_guard);
        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// Tier B: in-place scale+bias epilogue on a `[batch, out_dim]` GEMM output.
    /// Absent a_scale/b_scale are unity-padded; absent bias is zero-padded by the host before dispatch.
    fn scale_bias_epilogue(
        &self,
        out: &dyn BackendStorage,
        a_scale: Option<&dyn BackendStorage>,
        b_scale: Option<&dyn BackendStorage>,
        bias: Option<&dyn BackendStorage>,
        _batch: usize,
        out_dim: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let out_s = out
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan scale_bias_epilogue: out is not VulkanStorage".into())
            })?;
        let n = _batch * out_dim;
        if out.shape().elem_count() != n {
            return Err(Error::Shape(
                "Vulkan scale_bias_epilogue: out size mismatch".into(),
            ));
        }
        // Build padded buffers: absent scale -> [1] filled 1.0; absent bias -> [1] filled 0.0.
        let ones = vec![1.0f32];
        let zeros = vec![0.0f32];
        let a_data: Vec<f32> = if let Some(a) = a_scale {
            a.to_cpu_vec_f32()?
        } else {
            ones.clone()
        };
        let b_data: Vec<f32> = if let Some(b) = b_scale {
            b.to_cpu_vec_f32()?
        } else {
            ones.clone()
        };
        let bias_data: Vec<f32> = if let Some(b) = bias {
            b.to_cpu_vec_f32()?
        } else {
            zeros.clone()
        };
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        let a_s = self.from_cpu(&a_data, &Shape::new(vec![a_data.len()]), dtype.clone())?;
        let b_s = self.from_cpu(&b_data, &Shape::new(vec![b_data.len()]), dtype.clone())?;
        let bi_s = self.from_cpu(
            &bias_data,
            &Shape::new(vec![bias_data.len()]),
            dtype.clone(),
        )?;
        let a_buf = a_s
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan scale_bias_epilogue: a_scale is not VulkanStorage".into())
            })?
            .buffer;
        let b_buf = b_s
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan scale_bias_epilogue: b_scale is not VulkanStorage".into())
            })?
            .buffer;
        let bi_buf = bi_s
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan scale_bias_epilogue: bias is not VulkanStorage".into())
            })?
            .buffer;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::ScaleBiasEpilogue).to_vec();
        let buffers = [out_s.buffer, a_buf, b_buf, bi_buf];
        let grid_x = n.div_ceil(256) as u32;
        let push = push_params(n as u32, out_dim as u32, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;
        drop(ctx_guard);
        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }

    /// Fused Add + RMSNorm: `y_out = x + residual`, `norm_out = rms_norm(y_out, w, eps)`.
    /// Returns `(y_out, norm_out, compute_handle)`.
    fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan fused_add_rms_norm: x is not VulkanStorage".into())
        })?;
        let r_s = residual
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan fused_add_rms_norm: residual is not VulkanStorage".into())
            })?;
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan fused_add_rms_norm: weight is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let y_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let norm_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out_shape.elem_count();
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::AddRmsNorm).to_vec();

        // Bindings: x(0), residual(1), weight(2), y_out(3), norm_out(4).
        let buffers = [
            x_s.buffer,
            r_s.buffer,
            w_s.buffer,
            y_storage.buffer,
            norm_storage.buffer,
        ];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, eps);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push)).map_err(
            |e| {
                Error::Backend(format!(
                    "Vulkan fused_add_rms_norm GPU dispatch failed: {e}"
                ))
            },
        )?;

        Ok((
            Box::new(y_storage),
            Box::new(norm_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }
}

impl AutogradOps for VulkanDevice {
    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let e_s = e.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward e is not VulkanStorage".into())
        })?;
        let g_s = g.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward g is not VulkanStorage".into())
        })?;
        let dw_s = dw.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward dw is not VulkanStorage".into())
        })?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let df = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let de = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let buffers = [e_s.buffer, g_s.buffer, dw_s.buffer, df.buffer, de.buffer];
        let push = push_params(out_shape.elem_count() as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::SiluMulBackward),
            &buffers,
            out_shape.elem_count().div_ceil(256) as u32,
            1,
            1,
            Some(&push),
        )?;
        Ok((
            Box::new(df),
            Box::new(de),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    // Tier B complex - autograd backwards (audit gap: training on Vulkan hit the trait's Err(Unimplemented)).
    // These are CPU-reference fallbacks that mirror the documented ROCm kernel math exactly, so autograd produces.

    /// Softmax backward: `dx_i = s_i * (g_i - Σ_j g_j s_j)` per row.
    /// GPU-resident dispatch via `VulkanKernel::SoftmaxBackward`.
    fn softmax_backward(
        &self,
        out_grad: &dyn BackendStorage,
        softmax_out: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = out_grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan softmax_backward: grad is not VulkanStorage".into())
            })?;
        let s_s = softmax_out
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan softmax_backward: softmax_out is not VulkanStorage".into())
            })?;

        let total = out_shape.elem_count();
        let row_len = out_shape.dims().last().copied().unwrap_or(1).max(1);

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let dx = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [g_s.buffer, s_s.buffer, dx.buffer];
        let grid_x = total.div_ceil(256) as u32;
        let push = push_params(total as u32, row_len as u32, 0, 0, 0, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::SoftmaxBackward,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dx), Box::new(grim_tensor::backend::ReadyHandle)))
    }

    /// RMSNorm backward w.r.t. x and weight.
    fn rmsnorm_backward(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        out_grad: &dyn BackendStorage,
        eps: f32,
        x_shape: &Shape,
        w_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        if !self.caps.supports_fp32_atomic_add {
            return Err(Error::Backend(
                "rmsnorm_backward on Vulkan requires OpAtomicFAddEXT (RDNA3+ / NVIDIA)".into(),
            ));
        }

        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan rmsnorm_backward: x is not VulkanStorage".into())
        })?;
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rmsnorm_backward: weight is not VulkanStorage".into())
            })?;
        let g_s = out_grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rmsnorm_backward: grad is not VulkanStorage".into())
            })?;

        let cols = if x_shape.dims().len() > 1 {
            x_shape.dims()[1]
        } else {
            1
        };
        let total = x_shape.elem_count();

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let dx = VulkanStorage::alloc_device_local_gpu(
            x_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;
        let dw = VulkanStorage::alloc_device_local_gpu(
            w_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        // Zero-initialize dw buffer (atomic scatter-add target)
        unsafe {
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let res = vkMapMemory(
                ctx.device,
                dw.memory,
                0,
                dw.bytes as VkDeviceSize,
                0,
                &mut mapped,
            );
            if res == VK_SUCCESS && !mapped.is_null() {
                std::ptr::write_bytes(mapped, 0, dw.bytes);
                vkUnmapMemory(ctx.device, dw.memory);
            }
        }

        let buffers = [x_s.buffer, w_s.buffer, g_s.buffer, dx.buffer, dw.buffer];
        let grid_x = total.div_ceil(256) as u32;
        let push = push_params(total as u32, cols as u32, 0, 0, 0, eps);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::RmsnormBackward,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((
            Box::new(dx),
            Box::new(dw),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// RoPE backward: `dx = rotate(out_grad, -positions)` (inverse rotation).
    /// GPU-resident dispatch via `VulkanKernel::RopeBackward`.
    fn rope_backward(
        &self,
        out_grad: &dyn BackendStorage,
        cos: &dyn BackendStorage,
        sin: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = out_grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rope_backward: grad is not VulkanStorage".into())
            })?;
        let c_s = cos
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rope_backward: cos is not VulkanStorage".into())
            })?;
        let s_s = sin
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rope_backward: sin is not VulkanStorage".into())
            })?;

        let total = out_shape.elem_count();

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let dx = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [g_s.buffer, c_s.buffer, s_s.buffer, dx.buffer];
        let grid_x = total.div_ceil(256) as u32;
        let push = push_params(total as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::RopeBackward,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dx), Box::new(grim_tensor::backend::ReadyHandle)))
    }

    /// Embedding backward: scatter-add `dweight[token_ids[t], :] += out_grad[t, :]`.
    /// GPU-resident dispatch via `VulkanKernel::EmbeddingBackward`.
    fn embedding_backward(
        &self,
        out_grad: &dyn BackendStorage,
        token_ids: &[u32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if !self.caps.supports_fp32_atomic_add {
            return Err(Error::Backend(
                "embedding_backward on Vulkan requires OpAtomicFAddEXT (RDNA3+ / NVIDIA)".into(),
            ));
        }

        let g_s = out_grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan embedding_backward: grad is not VulkanStorage".into())
            })?;

        let num_tokens = token_ids.len();
        if num_tokens == 0 || hidden_dim == 0 || vocab_size == 0 {
            return Err(Error::Shape(
                "embedding_backward: empty vocab/hidden/tokens".into(),
            ));
        }
        let total = num_tokens * hidden_dim;
        if g_s.shape.elem_count() != total {
            return Err(Error::Shape(
                "embedding_backward: grad size mismatch".into(),
            ));
        }

        // Upload token_ids as a u32 buffer before holding global_context lock
        let token_shape = Shape::new(vec![num_tokens]);
        let tok_s = self.upload_u32(token_ids, &token_shape)?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let dw_shape = Shape::new(vec![vocab_size, hidden_dim]);
        let dw = VulkanStorage::alloc_device_local_gpu(
            &dw_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        // Zero-initialize dw buffer (atomic scatter-add target)
        unsafe {
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let res = vkMapMemory(
                ctx.device,
                dw.memory,
                0,
                dw.bytes as VkDeviceSize,
                0,
                &mut mapped,
            );
            if res == VK_SUCCESS && !mapped.is_null() {
                std::ptr::write_bytes(mapped, 0, dw.bytes);
                vkUnmapMemory(ctx.device, dw.memory);
            }
        }

        let tok_vk = tok_s
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend(
                    "Vulkan embedding_backward: token_ids storage is not VulkanStorage".into(),
                )
            })?;

        let buffers = [tok_vk.buffer, g_s.buffer, dw.buffer];
        let grid_x = total.div_ceil(256) as u32;
        let push = push_params(num_tokens as u32, hidden_dim as u32, 0, 0, 0, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::EmbeddingBackward,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dw), Box::new(grim_tensor::backend::ReadyHandle)))
    }
}

impl VulkanDevice {
    /// Log-softmax VJP: `dx_i = exp(log_p_i) * (g_i - Σ_j g_j)` per row.
    /// GPU-resident dispatch via `VulkanKernel::LogSoftmaxVjp`.
    pub fn log_softmax_vjp(
        &self,
        out_grad: &dyn BackendStorage,
        log_probs: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = out_grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan log_softmax_vjp: grad is not VulkanStorage".into())
            })?;
        let lp_s = log_probs
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan log_softmax_vjp: log_probs is not VulkanStorage".into())
            })?;

        let total = out_shape.elem_count();
        let row_len = out_shape.dims().last().copied().unwrap_or(1).max(1);

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let dx = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [g_s.buffer, lp_s.buffer, dx.buffer];
        let grid_x = total.div_ceil(256) as u32;
        let push = push_params(total as u32, row_len as u32, 0, 0, 0, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::LogSoftmaxVjp,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dx), Box::new(grim_tensor::backend::ReadyHandle)))
    }

    /// Charon MoE expert-weight backward: compute d_gate_w, d_up_w, d_down_w.
    /// GPU-resident dispatch via `VulkanKernel::CharonBackward`.
    #[allow(clippy::too_many_arguments)]
    pub fn charon_backward(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        grad: &dyn BackendStorage,
        num_experts: u32,
        hidden: u32,
        inter: u32,
    ) -> Result<()> {
        if !self.caps.supports_fp32_atomic_add {
            return Err(Error::Backend(
                "charon_backward on Vulkan requires OpAtomicFAddEXT (RDNA3+ / NVIDIA)".into(),
            ));
        }

        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan charon_backward: x is not VulkanStorage".into())
        })?;
        let gw_s = gate_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan charon_backward: gate_w is not VulkanStorage".into())
            })?;
        let uw_s = up_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan charon_backward: up_w is not VulkanStorage".into())
            })?;
        let dw_s = down_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan charon_backward: down_w is not VulkanStorage".into())
            })?;
        let g_s = grad
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan charon_backward: grad is not VulkanStorage".into())
            })?;

        let num_tokens = x_s.shape.elem_count() / hidden as usize;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        // Allocate output gradient buffer (d_gate_w + d_up_w + d_down_w)
        let total_grad_elems = (num_experts as usize) * (inter as usize) * (hidden as usize) * 2
            + (num_experts as usize) * (hidden as usize) * (inter as usize);
        let dg_shape = Shape::new(vec![total_grad_elems]);
        let dg = VulkanStorage::alloc_device_local_gpu(
            &dg_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        // Zero-initialize (atomic scatter-add target)
        unsafe {
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let res = vkMapMemory(
                ctx.device,
                dg.memory,
                0,
                dg.bytes as VkDeviceSize,
                0,
                &mut mapped,
            );
            if res == VK_SUCCESS && !mapped.is_null() {
                std::ptr::write_bytes(mapped, 0, dg.bytes);
                vkUnmapMemory(ctx.device, dg.memory);
            }
        }

        let buffers = [
            x_s.buffer,
            gw_s.buffer,
            uw_s.buffer,
            dw_s.buffer,
            g_s.buffer,
            dg.buffer,
        ];
        let total_dw = (num_experts as usize) * (hidden as usize) * (inter as usize);
        let grid_x = total_dw.div_ceil(64) as u32;
        let push = push_params(num_experts, hidden, inter, num_tokens as u32, 0, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::CharonBackward,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok(())
    }

    /// MoE persistent-worker comm-compute mega-kernel dispatch.
    /// GPU-resident dispatch via `VulkanKernel::MoeMegaKernel`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mega_kernel(
        &self,
        activations: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        dest_slots: &dyn BackendStorage,
        global_offsets: &dyn BackendStorage,
        expert_counts: &dyn BackendStorage,
        batch: u32,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        total_routed: u32,
    ) -> Result<()> {
        let act_s = activations
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: activations is not VulkanStorage".into())
            })?;
        let gw_s = gate_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: gate_w is not VulkanStorage".into())
            })?;
        let uw_s = up_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: up_w is not VulkanStorage".into())
            })?;
        let dw_s = down_w
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: down_w is not VulkanStorage".into())
            })?;
        let ds_s = dest_slots
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: dest_slots is not VulkanStorage".into())
            })?;
        let go_s = global_offsets
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: global_offsets is not VulkanStorage".into())
            })?;
        let ec_s = expert_counts
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan moe_mega_kernel: expert_counts is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        // Allocate output buffer
        let out_elems = (batch as usize) * (hidden as usize);
        let out_shape = Shape::new(vec![out_elems]);
        let output = VulkanStorage::alloc_device_local_gpu(
            &out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [
            act_s.buffer,
            gw_s.buffer,
            uw_s.buffer,
            dw_s.buffer,
            ds_s.buffer,
            go_s.buffer,
            ec_s.buffer,
            output.buffer,
        ];
        let grid_x = total_routed.max(1);
        let push = push_params(batch, hidden, inter, num_experts, top_k, 0.0);

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::MoeMegaKernel,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )?;

        Ok(())
    }
}

impl OptimizerOps for VulkanDevice {
    fn fused_adamw_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = p
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_adamw: p is not VulkanStorage".into()))?;
        let g_s = g
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_adamw: g is not VulkanStorage".into()))?;
        let m_s = m
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_adamw: m is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_adamw: v is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let buffers = [p_s.buffer, g_s.buffer, m_s.buffer, v_s.buffer];
        let grid_x = total.div_ceil(256) as u32;

        let push = [
            total as u32,
            lr.to_bits(),
            beta1.to_bits(),
            beta2.to_bits(),
            eps.to_bits(),
            weight_decay.to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ];

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::FusedAdamw,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan fused_adamw_step dispatch failed: {e}")))?;

        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }

    fn fused_lion_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        exp_avg: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = p
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_lion: p is not VulkanStorage".into()))?;
        let g_s = g
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan fused_lion: g is not VulkanStorage".into()))?;
        let m_s = exp_avg
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan fused_lion: exp_avg is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let buffers = [p_s.buffer, g_s.buffer, m_s.buffer];
        let grid_x = total.div_ceil(256) as u32;

        let push = [
            total as u32,
            lr.to_bits(),
            beta1.to_bits(),
            beta2.to_bits(),
            weight_decay.to_bits(),
            0,
            0,
            0,
        ];

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::FusedLion,
            &buffers,
            grid_x,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan fused_lion_step dispatch failed: {e}")))?;

        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }
}

impl CollectiveOps for VulkanDevice {
    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if inputs.is_empty() {
            return Err(Error::Backend("all_reduce: no inputs".into()));
        }
        if op != "sum" {
            return Err(Error::Backend(format!(
                "all_reduce: only 'sum' supported, got '{op}'"
            )));
        }

        // When a multi-GPU communicator is attached with world_size > 1, dispatch the ring-allreduce shader across device pairs.
        // Current default: single-GPU accumulation (communicator is None).
        if let Some(comm) = &self.communicator {
            if comm.world_size > 1 {
                return self.all_reduce_multi_gpu(inputs, comm);
            }
        }

        let shape = inputs[0].shape().clone();
        let dtype = inputs[0].dtype();
        let total = shape.elem_count();
        let is_f32 = dtype.arith == ArithType::F32;

        // All inputs must share the same shape.
        for s in inputs {
            if s.shape() != &shape {
                return Err(Error::Backend("all_reduce: input shape mismatch".into()));
            }
        }

        // ── GPU fast path: accumulate all inputs into a pre-zeroed output buffer.
        // The `all_reduce` accumulate kernel does Out[i] += A[i], so we zero the output once and.
        {
            let all_vulkan = inputs
                .iter()
                .all(|s| s.as_any().downcast_ref::<VulkanStorage>().is_some());
            if is_f32 && total > 0 && all_vulkan {
                let ctx_guard = global_context();
                if let Some(ctx) = ctx_guard.as_ref() {
                    if let Ok(out_storage) = VulkanStorage::alloc_gpu(
                        &shape,
                        DType::F32,
                        ctx.device,
                        ctx.physical_device,
                    ) {
                        // Zero the output buffer (accumulation target).
                        let zeroed = {
                            let mut mapped: *mut c_void = std::ptr::null_mut();
                            let res = unsafe {
                                vkMapMemory(
                                    ctx.device,
                                    out_storage.memory,
                                    0,
                                    out_storage.bytes as VkDeviceSize,
                                    0,
                                    &mut mapped,
                                )
                            };
                            if res == VK_SUCCESS && !mapped.is_null() {
                                unsafe {
                                    std::ptr::write_bytes(mapped, 0, out_storage.bytes);
                                    vkUnmapMemory(ctx.device, out_storage.memory);
                                }
                                true
                            } else {
                                false
                            }
                        };
                        if zeroed {
                            let spirv = spirv_for(VulkanKernel::AllReduce).to_vec();
                            let grid_x = total.div_ceil(256) as u32;
                            let push = push_params(total as u32, 0, 0, 0, 0, 0.0);
                            let mut ok = true;
                            for input in inputs {
                                if let Some(in_s) = input.as_any().downcast_ref::<VulkanStorage>() {
                                    let buffers = [in_s.buffer, out_storage.buffer];
                                    if run_compute_shader(
                                        ctx,
                                        &spirv,
                                        &buffers,
                                        grid_x,
                                        1,
                                        1,
                                        Some(&push),
                                    )
                                    .is_err()
                                    {
                                        ok = false;
                                        break;
                                    }
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                return Ok((
                                    Box::new(out_storage),
                                    Box::new(grim_tensor::backend::ReadyHandle),
                                ));
                            }
                        }
                    }
                }
            }
        } // ctx_guard dropped — lock released before CPU fallback

        // ── CPU fallback ──────────────────────────────────────────────
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(
                    "all_reduce: input length mismatch during fallback".into(),
                ));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        if partials.is_empty() {
            return Err(Error::Backend("comm_fuse_reduce: no partials".into()));
        }
        let dims0 = partials[0].0.shape().dims();
        let m = dims0[0];
        let n_total: usize = partials
            .iter()
            .map(|(s, _)| s.shape().dims().get(1).copied().unwrap_or(0))
            .sum();
        let dtype = partials[0].0.dtype();
        let is_f32 = dtype.arith == ArithType::F32;
        let out_shape = Shape::new(vec![m, n_total]);

        // ── GPU fast path
        {
            let all_vulkan = partials
                .iter()
                .all(|(s, _)| s.as_any().downcast_ref::<VulkanStorage>().is_some());
            if is_f32 && n_total > 0 && all_vulkan {
                let ctx_guard = global_context();
                if let Some(ctx) = ctx_guard.as_ref() {
                    if let Ok(out_storage) = VulkanStorage::alloc_gpu(
                        &out_shape,
                        DType::F32,
                        ctx.device,
                        ctx.physical_device,
                    ) {
                        // Zero the output buffer.
                        let zeroed = {
                            let mut mapped: *mut c_void = std::ptr::null_mut();
                            let res = unsafe {
                                vkMapMemory(
                                    ctx.device,
                                    out_storage.memory,
                                    0,
                                    out_storage.bytes as VkDeviceSize,
                                    0,
                                    &mut mapped,
                                )
                            };
                            if res == VK_SUCCESS && !mapped.is_null() {
                                unsafe {
                                    std::ptr::write_bytes(mapped, 0, out_storage.bytes);
                                    vkUnmapMemory(ctx.device, out_storage.memory);
                                }
                                true
                            } else {
                                false
                            }
                        };
                        if zeroed {
                            let spirv = spirv_for(VulkanKernel::CommFuseReduce).to_vec();
                            let mut col_offset = 0usize;
                            let mut ok = true;
                            for (storage, _placement) in partials {
                                if let Some(s) = storage.as_any().downcast_ref::<VulkanStorage>() {
                                    let n_src = s.shape().dims().get(1).copied().unwrap_or(0);
                                    let buffers = [s.buffer, out_storage.buffer];
                                    let grid_x = n_src.div_ceil(16) as u32;
                                    let grid_y = m.div_ceil(16) as u32;
                                    let push = push_params(
                                        n_src as u32,
                                        col_offset as u32,
                                        n_total as u32,
                                        m as u32,
                                        0,
                                        0.0,
                                    );
                                    if run_compute_shader(
                                        ctx,
                                        &spirv,
                                        &buffers,
                                        grid_x,
                                        grid_y,
                                        1,
                                        Some(&push),
                                    )
                                    .is_err()
                                    {
                                        ok = false;
                                        break;
                                    }
                                    col_offset += n_src;
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                return Ok(Box::new(out_storage));
                            }
                        }
                    }
                }
            }
        } // ctx_guard dropped — lock released before CPU fallback

        // ── CPU fallback ──────────────────────────────────────────────
        let mut assembled = vec![0.0f32; m * n_total];
        let mut col_offset = 0usize;
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            for row in 0..m {
                for col in 0..n_cols {
                    assembled[row * n_total + col_offset + col] += data[row * n_cols + col];
                }
            }
            col_offset += n_cols;
        }
        let storage = self.from_cpu(&assembled, &out_shape, dtype)?;
        Ok(storage)
    }

    /// Tier B (semi-parity): analytical GEMM latency estimate (milliseconds) for the placement-aware Scythe scheduler.
    /// A roofline-style model: `time = flops / peak_throughput + bytes / peak_bw`, scaled by an.
    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &ScythePlacement,
    ) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        // Heuristic peak FP32 throughput (GFLOPS) by vendor class; the
        // autotuner would replace these with measured tile numbers.
        let peak_gflops = match self.caps.vendor_id {
            0x10de => 15000.0, // NVIDIA consumer/high-end
            0x1002 => 8000.0,  // AMD RDNA
            0x8086 => 2000.0,  // Intel
            _ => 3000.0,
        };
        let peak_bw_gbps = 500.0;
        let elems = (m * k + k * n + m * n) as f64;
        let bytes = elems * dtype.arith.byte_size() as f64;
        let compute_ms = flops / (peak_gflops * 1e6);
        let mem_ms = bytes / (peak_bw_gbps * 1e6 / 1000.0 * 1000.0);
        (compute_ms + mem_ms).max(1e-4)
    }
}

impl MemoryOps for VulkanDevice {
    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let min_bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype))
            .unwrap_or(0)
            .max(data.len());
        let storage = VulkanStorage::alloc_gpu_with_bytes(
            shape,
            dtype,
            ctx.device,
            ctx.physical_device,
            min_bytes,
        )?;

        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed in from_cpu_bytes: {}",
                res
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    /// KV-arena path: allocate uninitialized device-resident storage so decode
    /// steps can append K/V rows with `copy_slice_into` without re-uploading.
    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage = VulkanStorage::alloc_gpu(shape, dtype, ctx.device, ctx.physical_device)?;
        Ok(Box::new(storage))
    }

    /// Device-to-device copy of `count` f32 elements from `src` into `dst` at `dst_elem_offset`.
    /// Both storages must live on this device (no peer routing - that is `copy_via_route`'s job).
    fn copy_slice_into(
        &self,
        dst: &dyn BackendStorage,
        src: &dyn BackendStorage,
        dst_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        let dst_s = dst
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan copy_slice_into: dst is not VulkanStorage".into())
            })?;
        let src_s = src
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan copy_slice_into: src is not VulkanStorage".into())
            })?;

        let dst_elem_count = dst_s.shape.elem_count();
        let src_elem_count = src_s.shape.elem_count();
        let dst_off = dst_elem_offset;
        if dst_off.saturating_add(count) > dst_elem_count {
            return Err(Error::Backend(
                "copy_slice_into: dst offset+count out of bounds".into(),
            ));
        }
        if count > src_elem_count {
            return Err(Error::Backend("copy_slice_into: count exceeds src".into()));
        }

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        unsafe {
            // Create a one-shot command pool on the compute family.
            let pool_ci = VkCommandPoolCreateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                queue_family_index: ctx.compute_family_index,
            };
            let mut command_pool = 0u64;
            let res =
                vkCreateCommandPool(ctx.device, &pool_ci, std::ptr::null(), &mut command_pool);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkCreateCommandPool failed: {res}"
                )));
            }

            struct PoolCleanup {
                device: *mut c_void,
                command_pool: u64,
            }
            impl Drop for PoolCleanup {
                fn drop(&mut self) {
                    if self.command_pool != 0 {
                        unsafe {
                            vkDestroyCommandPool(self.device, self.command_pool, std::ptr::null());
                        }
                    }
                }
            }
            let _pool = PoolCleanup {
                device: ctx.device,
                command_pool,
            };

            // Allocate a one-time command buffer.
            let cmd_alloc = VkCommandBufferAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                command_pool,
                level: 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
                command_buffer_count: 1,
            };
            let mut command_buffer: *mut c_void = std::ptr::null_mut();
            let res = vkAllocateCommandBuffers(ctx.device, &cmd_alloc, &mut command_buffer);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkAllocateCommandBuffers failed: {res}"
                )));
            }

            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: 1, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                p_inheritance_info: std::ptr::null(),
            };
            let res = vkBeginCommandBuffer(command_buffer, &begin_info);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkBeginCommandBuffer failed: {res}"
                )));
            }

            let elem_bytes = std::mem::size_of::<f32>() as VkDeviceSize;
            let region = VkBufferCopy {
                src_offset: 0,
                dst_offset: (dst_off as VkDeviceSize) * elem_bytes,
                size: (count as VkDeviceSize) * elem_bytes,
            };
            vkCmdCopyBuffer(command_buffer, src_s.buffer, dst_s.buffer, 1, &region);

            let res = vkEndCommandBuffer(command_buffer);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkEndCommandBuffer failed: {res}"
                )));
            }

            let cmd_buf_u64 = command_buffer as u64;
            let submit_info = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: std::ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: std::ptr::null(),
                p_wait_dst_stage_mask: std::ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf_u64,
                signal_semaphore_count: 0,
                p_signal_semaphores: std::ptr::null(),
            };
            let _q_lock = QUEUE_LOCK.lock().unwrap();
            let res = vkQueueSubmit(ctx.queue, 1, &submit_info, 0);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkQueueSubmit failed: {res}"
                )));
            }
            let res = vkQueueWaitIdle(ctx.queue);
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "copy_slice_into: vkQueueWaitIdle failed: {res}"
                )));
            }
        }
        Ok(())
    }
}

impl GraphCaptureOps for VulkanDevice {
    /// Graph-capture bookkeeping. Delegates to `VK_GRAPH_CACHE` (see `graph_capture.rs`).
    fn begin_graph_capture(&self, key: &str) -> Result<()> {
        VK_GRAPH_CACHE.begin(key)
    }

    fn end_graph_capture(&self, key: &str) -> Result<()> {
        VK_GRAPH_CACHE.end(key)
    }

    fn replay_graph(&self, key: &str) -> Result<bool> {
        VK_GRAPH_CACHE.replay(key)
    }

    fn has_captured_graph(&self, key: &str) -> bool {
        VK_GRAPH_CACHE.has(key)
    }
}

lazy_static::lazy_static! {
    static ref VK_GRAPH_CACHE: graph_capture::VkGraphCache = graph_capture::VkGraphCache::new();
}
impl grim_tensor::BackendDevice for VulkanDevice {}

/// Helper function to retrieve the size in bytes of a data type.
pub(crate) fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 => 2,
        ArithType::BF16 => 4, // BF16 simulated via f32 round-trip; 4 bytes for kernel compatibility.
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

/// Convert f32 to BF16 and back to f32, simulating BF16 precision.
fn f32_to_bf16_to_f32(val: f32) -> f32 {
    let bits = val.to_bits();
    let sign = bits & 0x80000000;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;

    if exp == 0 {
        // Subnormal or zero -> flush to zero
        0.0
    } else if exp == 255 {
        // Inf or NaN -> preserve
        f32::from_bits(sign | 0x7F800000)
    } else {
        // Normal: truncate mantissa to 7 bits, keep sign and exponent
        let bf16_mant = (mant >> 16) & 0x7F;
        let f32_bits = sign | (exp << 23) | (bf16_mant << 16);
        f32::from_bits(f32_bits)
    }
}

/// Query `(free_bytes, total_bytes)` memory on Vulkan device `ordinal`.
pub fn vram_info(_ordinal: usize) -> Option<(u64, u64)> {
    let guard = global_context();
    if let Some(ctx) = guard.as_ref() {
        unsafe {
            let mut props = VkPhysicalDeviceMemoryProperties {
                memory_type_count: 0,
                memory_types: [VkMemoryType {
                    property_flags: 0,
                    heap_index: 0,
                }; 32],
                memory_heap_count: 0,
                memory_heaps: [VkMemoryHeap { size: 0, flags: 0 }; 16],
            };
            vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mut props);

            // Sum all device-local heaps (VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1).
            let mut total_device_local: u64 = 0;
            for i in 0..(props.memory_heap_count as usize) {
                if (props.memory_heaps[i].flags & 1) != 0 {
                    total_device_local += props.memory_heaps[i].size;
                }
            }

            if total_device_local == 0 {
                return None;
            }

            // Without VK_EXT_memory_budget, live free memory is unavailable.
            // Return None so callers know free memory querying is unsupported.
            return None;
        }
    }

    None
}

/// WI-1: live compute utilization for `ordinal`.
/// Scope note (per WI-1): Vulkan has no core-spec utilization query (vendor extensions only).
pub fn compute_utilization(_ordinal: usize) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::{DType, Shape};

    /// GPU-gated parity test for the fused grouped MoE dispatch kernel.
    /// Runs only when a Vulkan device is present AND it supports FP32 atomic add on.
    #[test]
    #[ignore]
    fn test_vulkan_moe_fused_dispatch_parity() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        // Skip on hardware that doesn't support FP32 atomic add on SSBOs.
        if !dev.caps().supports_fp32_atomic_add {
            eprintln!(
                "test_vulkan_moe_fused_dispatch_parity: skipped (device '{}' does not support fp32 atomic add; requires RDNA3+)",
                dev.caps().device_name
            );
            return;
        }
        let hidden: usize = 4;
        let inter: usize = 3;
        let num_experts: usize = 2;
        let batch: usize = 2;
        let rsf: f32 = 0.5;

        // activations [batch, hidden]
        let x_data: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.1).collect();
        let x = dev
            .from_cpu(&x_data, &Shape::new(vec![batch, hidden]), DType::F32)
            .unwrap();

        // per-expert gate/up [inter, hidden], down [hidden, inter] (identity-ish)
        let mk = |e: usize, sign: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; inter * hidden];
            for i in 0..inter {
                for h in 0..hidden {
                    v[i * hidden + h] =
                        sign * (1.0 + (i as f32) * 0.1 + (h as f32) * 0.01 + e as f32);
                }
            }
            v
        };
        let gate_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let up_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let down_flat: Vec<f32> = (0..num_experts)
            .flat_map(|e| {
                let mut v = vec![0.0f32; hidden * inter];
                for h in 0..hidden {
                    for i in 0..inter {
                        v[h * inter + i] = 1.0 + (h as f32) * 0.05 + (i as f32) * 0.02 + e as f32;
                    }
                }
                v
            })
            .collect();

        // top-1 routing: token0 -> expert0, token1 -> expert1
        let rtok = vec![0u32, 1u32];
        let rexp = vec![0u32, 1u32];
        let rw = vec![1.0f32, 1.0f32];
        let num_pairs = rtok.len();

        let gate_buf = dev
            .upload_f32(&gate_flat, &Shape::new(vec![num_experts * inter * hidden]))
            .unwrap();
        let up_buf = dev
            .upload_f32(&up_flat, &Shape::new(vec![num_experts * inter * hidden]))
            .unwrap();
        let down_buf = dev
            .upload_f32(&down_flat, &Shape::new(vec![num_experts * hidden * inter]))
            .unwrap();
        let tok_buf = dev.upload_u32(&rtok, &Shape::new(vec![num_pairs])).unwrap();
        let exp_buf = dev.upload_u32(&rexp, &Shape::new(vec![num_pairs])).unwrap();
        let w_buf = dev.upload_f32(&rw, &Shape::new(vec![num_pairs])).unwrap();

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out, _h) = dev
            .moe_fused_dispatch(
                x.as_ref(),
                gate_buf.as_ref(),
                up_buf.as_ref(),
                down_buf.as_ref(),
                tok_buf.as_ref(),
                exp_buf.as_ref(),
                w_buf.as_ref(),
                &out_shape,
                hidden as u32,
                inter as u32,
                num_experts as u32,
                batch as u32,
                rsf,
            )
            .unwrap();
        let res = out.to_cpu_vec_f32().unwrap();

        // CPU reference: for each token, expert e, y = (gate*x).silu * (up*x); down*y * rsf.
        let silu = |a: f32| a / (1.0 + (-a).exp());
        let dot = |w: &[f32], x: &[f32]| -> f32 { (0..w.len()).map(|i| w[i] * x[i]).sum() };
        for t in 0..batch {
            let e = rexp[t] as usize;
            let xt = &x_data[t * hidden..(t + 1) * hidden];
            let gw = &gate_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let uw = &up_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let dw = &down_flat[e * hidden * inter..(e + 1) * hidden * inter];
            let mut routed = vec![0.0f32; hidden];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for i in 0..inter {
                    let g = dot(&gw[i * hidden..i * hidden + hidden], xt);
                    let u = dot(&uw[i * hidden..i * hidden + hidden], xt);
                    acc += dw[h * inter + i] * (silu(g) * u);
                }
                routed[h] = rsf * acc;
            }
            for h in 0..hidden {
                let got = res[t * hidden + h];
                let tol = routed[h].abs().max(1.0) * 1e-3 + 1e-3;
                assert!(
                    (got - routed[h]).abs() < tol,
                    "moe tok{} dim{}: gpu {} vs ref {} (tol {})",
                    t,
                    h,
                    got,
                    routed[h],
                    tol
                );
            }
        }
    }

    #[test]
    fn test_vulkan_device_probe() {
        let devices = VulkanDevice::probe().unwrap();
        if global_context().is_some() {
            assert!(!devices.is_empty());
        }
    }

    #[test]
    fn test_vulkan_zeros() {
        if global_context().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![0.0; 8]);
    }

    #[test]
    fn test_vulkan_from_cpu() {
        if global_context().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![3, 2]);
        let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, host_data);
    }

    #[test]
    fn test_vulkan_autotuner_and_spirv() {
        let autotuner = VulkanAutotuner::new();
        let caps = VulkanCaps::probe_default("Vulkan Test Device".into(), 0x1002, 0x744c, 1);
        let config = autotuner.search_tile_config(&caps, 128, 128, 64, None);

        assert_eq!(config.block_m, 32);
        assert_eq!(config.block_n, 32);

        // Verify a precompiled SPIR-V blob is loadable for the chosen tile size.
        let spirv = spirv_for(VulkanKernel::Matmul64);
        assert!(!spirv.is_empty());
    }

    #[test]
    fn test_vulkan_matmul_simulated() {
        if global_context().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let shape = Shape::new(vec![2, 2]);

        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let (out_s, _handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &shape).unwrap();
        let res = out_s.to_cpu_vec_f32().unwrap();
        assert_eq!(res, a_data); // A @ I = A
    }

    #[test]
    fn test_vulkan_matmul_non_identity_and_shape_mismatch() {
        if global_context().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];

        // 1. Non-identity matrix multiplication: [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let shape = Shape::new(vec![2, 2]);

        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let (out_s, _handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &shape).unwrap();
        let res = out_s.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);

        // 2. Shape mismatch error enforcement
        let bad_shape = Shape::new(vec![3, 2]);
        let err_res = dev.matmul(a_s.as_ref(), b_s.as_ref(), &bad_shape);
        assert!(
            err_res.is_err(),
            "matmul with wrong output shape must return Err"
        );
    }

    #[test]
    fn test_vulkan_gpu_compute() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let shape = Shape::new(vec![4]);

        let dev = VulkanDevice::new();
        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let a_storage = a_s.as_any().downcast_ref::<VulkanStorage>().unwrap();
        let b_storage = b_s.as_any().downcast_ref::<VulkanStorage>().unwrap();

        let ctx_guard = global_context();
        let ctx = match ctx_guard.as_ref() {
            Some(c) => c,
            None => return,
        };
        let out_storage =
            VulkanStorage::alloc_gpu(&shape, DType::F32, ctx.device, ctx.physical_device).unwrap();

        // Standard precompiled add SPIR-V binary from radv_repro.rs
        let spirv_add_u32: &[u32] = &[
            0x07230203, 0x00010000, 0x0008000b, 0x00000033, 0x00000000, 0x00020011, 0x00000001,
            0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
            0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
            0x0000000b, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001,
            0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
            0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c,
            0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000019, 0x43667542,
            0x00000000, 0x00040006, 0x00000019, 0x00000000, 0x00000063, 0x00030005, 0x0000001b,
            0x00000000, 0x00040005, 0x00000020, 0x41667542, 0x00000000, 0x00040006, 0x00000020,
            0x00000000, 0x00000061, 0x00030005, 0x00000022, 0x00000000, 0x00040005, 0x00000028,
            0x42667542, 0x00000000, 0x00040006, 0x00000028, 0x00000000, 0x00000062, 0x00030005,
            0x0000002a, 0x00000000, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c, 0x00040047,
            0x00000018, 0x00000006, 0x00000004, 0x00030047, 0x00000019, 0x00000003, 0x00050048,
            0x00000019, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000001b, 0x00000021,
            0x00000002, 0x00040047, 0x0000001b, 0x00000022, 0x00000000, 0x00040047, 0x0000001f,
            0x00000006, 0x00000004, 0x00030047, 0x00000020, 0x00000003, 0x00050048, 0x00000020,
            0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000022, 0x00000021, 0x00000000,
            0x00040047, 0x00000022, 0x00000022, 0x00000000, 0x00040047, 0x00000027, 0x00000006,
            0x00000004, 0x00030047, 0x00000028, 0x00000003, 0x00050048, 0x00000028, 0x00000000,
            0x00000023, 0x00000000, 0x00040047, 0x0000002a, 0x00000021, 0x00000001, 0x00040047,
            0x0000002a, 0x00000022, 0x00000000, 0x00040047, 0x00000032, 0x0000000b, 0x00000019,
            0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
            0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017,
            0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009,
            0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
            0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0004002b, 0x00000006,
            0x00000011, 0x00000004, 0x00020014, 0x00000012, 0x00030016, 0x00000017, 0x00000020,
            0x0003001d, 0x00000018, 0x00000017, 0x0003001e, 0x00000019, 0x00000018, 0x00040020,
            0x0000001a, 0x00000002, 0x00000019, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000002,
            0x00040015, 0x0000001c, 0x00000020, 0x00000001, 0x0004002b, 0x0000001c, 0x0000001d,
            0x00000000, 0x0003001d, 0x0000001f, 0x00000017, 0x0003001e, 0x00000020, 0x0000001f,
            0x00040020, 0x00000021, 0x00000002, 0x00000020, 0x0004003b, 0x00000021, 0x00000022,
            0x00000002, 0x00040020, 0x00000024, 0x00000002, 0x00000017, 0x0003001d, 0x00000027,
            0x00000017, 0x0003001e, 0x00000028, 0x00000027, 0x00040020, 0x00000029, 0x00000002,
            0x00000028, 0x0004003b, 0x00000029, 0x0000002a, 0x00000002, 0x0004002b, 0x00000006,
            0x00000030, 0x00000040, 0x0004002b, 0x00000006, 0x00000031, 0x00000001, 0x0006002c,
            0x00000009, 0x00000032, 0x00000030, 0x00000031, 0x00000031, 0x00050036, 0x00000002,
            0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
            0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c,
            0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f,
            0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x000500ae, 0x00000012, 0x00000013,
            0x00000010, 0x00000011, 0x000300f7, 0x00000015, 0x00000000, 0x000400fa, 0x00000013,
            0x00000014, 0x00000015, 0x000200f8, 0x00000014, 0x000100fd, 0x000200f8, 0x00000015,
            0x0004003d, 0x00000006, 0x0000001e, 0x00000008, 0x0004003d, 0x00000006, 0x00000023,
            0x00000008, 0x00060041, 0x00000024, 0x00000025, 0x00000022, 0x0000001d, 0x00000023,
            0x0004003d, 0x00000017, 0x00000026, 0x00000025, 0x0004003d, 0x00000006, 0x0000002b,
            0x00000008, 0x00060041, 0x00000024, 0x0000002c, 0x0000002a, 0x0000001d, 0x0000002b,
            0x0004003d, 0x00000017, 0x0000002d, 0x0000002c, 0x00050081, 0x00000017, 0x0000002e,
            0x00000026, 0x0000002d, 0x00060041, 0x00000024, 0x0000002f, 0x0000001b, 0x0000001d,
            0x0000001e, 0x0003003e, 0x0000002f, 0x0000002e, 0x000100fd, 0x00010038,
        ];

        let spirv_bytes = unsafe {
            std::slice::from_raw_parts(spirv_add_u32.as_ptr() as *const u8, spirv_add_u32.len() * 4)
        };

        let buffers = [a_storage.buffer, b_storage.buffer, out_storage.buffer];
        run_compute_shader(ctx, spirv_bytes, &buffers, 1, 1, 1, None).unwrap();

        let cpu_data = out_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    // ===== Mutation-resistant kernel math contracts =====

    fn close_vulkan(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-4,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    /// Source-presence guard for the partial-rotary/YaRN RoPE and the sliding-window attention kernels.
    /// No GPU required - asserts the SPIR-V blobs compiled (build.rs emits a `SPIRV_*` const only.
    #[test]
    fn yarn_and_swa_kernel_presence() {
        // If build.rs failed to compile any of these, the `SPIRV_*` const would be absent
        // and `spirv_for` would fail to compile - so merely referencing them is the presence test.
        let _ = spirv_for(VulkanKernel::RopeYarn);
        let _ = spirv_for(VulkanKernel::QkvAttentionSwa);
        let _ = spirv_for(VulkanKernel::QkvAttentionPagedSwa);
        assert_eq!(binding_count(VulkanKernel::RopeYarn), 3);
        assert_eq!(binding_count(VulkanKernel::QkvAttentionSwa), 4);
        assert_eq!(binding_count(VulkanKernel::QkvAttentionPagedSwa), 5);
    }

    #[test]
    fn test_vulkan_add_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let a_data = vec![1.5f32, -2.5, 0.0, std::f32::consts::PI];
        let b_data = vec![2.5f32, 3.5, -1.0, 1.0];
        let shape = Shape::new(vec![4]);
        let a = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.add(a.as_ref(), b.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 4);
        close_vulkan(res[0], 4.0, "vulkan_add w0");
        close_vulkan(res[1], 1.0, "vulkan_add w1");
        close_vulkan(res[2], -1.0, "vulkan_add w2");
        close_vulkan(res[3], 4.14159, "vulkan_add w3");
    }

    #[test]
    fn test_vulkan_math_ops() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let shape = Shape::new(vec![4]);
        let host_data = vec![4.0f32, 9.0, 16.0, 25.0];
        let x = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();

        let (out_sqrt, _) = dev.sqrt(x.as_ref(), &shape).unwrap();
        assert_eq!(out_sqrt.to_cpu_vec_f32().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

        let (out_recip, _) = dev.recip(out_sqrt.as_ref(), &shape).unwrap();
        let recip_vals = out_recip.to_cpu_vec_f32().unwrap();
        let expected_recip = [0.5, 1.0 / 3.0, 0.25, 0.2];
        for (actual, expected) in recip_vals.iter().zip(expected_recip.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let (out_mul, _) = dev.mul_scalar(x.as_ref(), 0.5, &shape).unwrap();
        assert_eq!(out_mul.to_cpu_vec_f32().unwrap(), vec![2.0, 4.5, 8.0, 12.5]);
    }

    #[test]
    fn test_vulkan_silu_mul_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let gate_data = vec![1.0f32, -1.0];
        let up_data = vec![2.0f32, 3.0];
        let shape = Shape::new(vec![2]);
        let gate = dev.from_cpu(&gate_data, &shape, DType::F32).unwrap();
        let up = dev.from_cpu(&up_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.silu_mul(gate.as_ref(), up.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let sig_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
        let expected_0 = sig_1 * 1.0 * 2.0;

        let sig_neg1 = 1.0f32 / (1.0f32 + (1.0f32).exp());
        let expected_1 = -sig_neg1 * 3.0;

        close_vulkan(res[0], expected_0, "vulkan_silu_mul w0");
        close_vulkan(res[1], expected_1, "vulkan_silu_mul w1");
    }

    #[test]
    fn test_vulkan_rms_norm_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let x_data = vec![3.0f32, 4.0];
        let w_data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let w = dev.from_cpu(&w_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.rms_norm(x.as_ref(), w.as_ref(), 1e-6, &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let rms_val = (12.5f32 + 1e-6).sqrt();
        let expected_0 = (3.0 / rms_val) * 1.0;
        let expected_1 = (4.0 / rms_val) * 2.0;
        close_vulkan(res[0], expected_0, "vulkan_rms_norm w0");
        close_vulkan(res[1], expected_1, "vulkan_rms_norm w1");
    }

    #[test]
    fn test_vulkan_softmax_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let x_data = vec![1.0f32, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.softmax(x.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 3);

        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        close_vulkan(res[0], 1.0f32.exp() / sum_exp, "vulkan_softmax w0");
        close_vulkan(res[1], 2.0f32.exp() / sum_exp, "vulkan_softmax w1");
        close_vulkan(res[2], 3.0f32.exp() / sum_exp, "vulkan_softmax w2");
    }

    #[test]
    fn test_vulkan_embedding_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let table = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let weight = dev
            .from_cpu(&table, &Shape::new(vec![3, 2]), DType::F32)
            .unwrap();
        let indices = vec![2u32, 0];
        let out_shape = Shape::new(vec![2, 2]);
        let (out, _h) = dev
            .embedding(weight.as_ref(), &indices, &out_shape)
            .unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![50.0, 60.0, 10.0, 20.0]);
    }

    // BF16 matmul golden test — hand-crafted BF16 inputs, exact FP32 reference.
    // BF16: 1 sign | 8 exponent | 7 mantissa. Accumulates in FP32.

    #[test]
    fn test_vulkan_matmul_bf16_golden_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let shape = Shape::new(vec![2, 2]);

        // Hand-crafted BF16: 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040, 4.0=0x4080
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a = dev.from_cpu(&a_data, &shape, DType::BF16).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::BF16).unwrap();
        let (out, _h) = dev.matmul(a.as_ref(), b.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();

        // Reference: [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50], FP32 accumulation.
        close_vulkan(res[0], 19.0, "bf16_matmul[0,0]");
        close_vulkan(res[1], 22.0, "bf16_matmul[0,1]");
        close_vulkan(res[2], 43.0, "bf16_matmul[1,0]");
        close_vulkan(res[3], 50.0, "bf16_matmul[1,1]");
    }

    // QKV attention golden test — hand-crafted Q/K/V, exact FP32 reference.
    // Q: [seq=2, heads=2, dim=2]; K/V: [kv_seq=4, kv_heads=1, dim=2].
    #[test]
    fn test_vulkan_qkv_attention_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();

        let seq_len = 2usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let kv_seq_len = 4usize;

        let q_data = vec![1.0f32; seq_len * num_heads * head_dim];
        let k_data = vec![1.0f32; kv_seq_len * num_kv_heads * head_dim];
        let v_data = vec![2.0f32; kv_seq_len * num_kv_heads * head_dim];

        let q_shape = Shape::new(vec![seq_len, num_heads, head_dim]);
        let k_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let v_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

        let q_buf = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
        let k_buf = dev.from_cpu(&k_data, &k_shape, DType::F32).unwrap();
        let v_buf = dev.from_cpu(&v_data, &v_shape, DType::F32).unwrap();

        let (out, _h) = dev
            .qkv_attention(
                q_buf.as_ref(),
                k_buf.as_ref(),
                v_buf.as_ref(),
                num_kv_heads,
                kv_seq_len,
                0,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();

        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), seq_len * num_heads * head_dim);
        for &val in res.iter() {
            close_vulkan(val, 2.0, "qkv_attention out");
        }
    }

    #[test]
    fn test_vulkan_qkv_attention_paged_gqa_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let q_shape = Shape::new(vec![1, 2, 2]);
        let page_shape = Shape::new(vec![1, 2, 1, 2]);
        let table_shape = Shape::new(vec![1, 1]);
        let q = dev.from_cpu(&[1.0f32; 4], &q_shape, DType::F32).unwrap();
        let k = dev.from_cpu(&[1.0f32; 4], &page_shape, DType::F32).unwrap();
        let v = dev.from_cpu(&[2.0f32; 4], &page_shape, DType::F32).unwrap();
        let table = dev.from_cpu(&[0.0f32], &table_shape, DType::F32).unwrap();
        let (out, _) = dev
            .qkv_attention_paged(
                q.as_ref(),
                table.as_ref(),
                k.as_ref(),
                v.as_ref(),
                1,
                1,
                2,
                2,
                0,
                None,
                &q_shape,
            )
            .unwrap();
        for value in out.to_cpu_vec_f32().unwrap() {
            close_vulkan(value, 2.0, "paged GQA");
        }
    }

    #[test]
    fn test_vulkan_tree_attention_gqa_exact() {
        if global_context().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let q_shape = Shape::new(vec![1, 2, 2, 2]);
        let kv_shape = Shape::new(vec![2, 1, 2]);
        let parent_shape = Shape::new(vec![2]);
        let q = dev.from_cpu(&[1.0f32; 8], &q_shape, DType::F32).unwrap();
        let k = dev.from_cpu(&[1.0f32; 4], &kv_shape, DType::F32).unwrap();
        let v = dev.from_cpu(&[2.0f32; 4], &kv_shape, DType::F32).unwrap();
        let parents = dev
            .from_cpu(&[0.0f32, 0.0], &parent_shape, DType::F32)
            .unwrap();
        let (out, _) = dev
            .tree_attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                parents.as_ref(),
                1,
                2,
                0,
                &q_shape,
            )
            .unwrap();
        for value in out.to_cpu_vec_f32().unwrap() {
            close_vulkan(value, 2.0, "tree GQA");
        }
    }
}

// Tier A semi-parity gates (Vulkan vs ROCm trait coverage).

#[cfg(test)]
mod tier_a_semi_parity_tests {
    use super::*;
    use grim_tensor::backend::{CoreTensorOps, ElementwiseOps};

    fn context_available() -> bool {
        global_context().as_ref().is_some()
    }

    fn dev() -> VulkanDevice {
        VulkanDevice::new()
    }

    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    /// Device sub: exact per-element a−b.
    #[test]
    fn vulkan_sub_matches_reference() {
        let dev = dev();
        if !context_available() {
            return; // no Vulkan device; the override compiles everywhere
        }
        let a = stor(&dev, &[5.0f32, -1.0, 0.25, 100.0], &[4]);
        let b = stor(&dev, &[2.0f32, 1.0, 0.75, 100.0], &[4]);
        let (out, h) =
            ElementwiseOps::sub(&dev, a.as_ref(), b.as_ref(), &Shape::new(vec![4])).unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.to_cpu_vec_f32().unwrap(), vec![3.0, -2.0, -0.5, 0.0]);
    }

    /// Device scalar ops: add/sub/div by broadcast scalar, exact.
    #[test]
    fn vulkan_scalar_ops_match_reference() {
        let dev = dev();
        if !context_available() {
            return;
        }
        let x = stor(&dev, &[4.0f32, -2.0, 0.5], &[3]);
        let (o1, _) =
            ElementwiseOps::add_scalar(&dev, x.as_ref(), 1.5, &Shape::new(vec![3])).unwrap();
        assert_eq!(o1.to_cpu_vec_f32().unwrap(), vec![5.5, -0.5, 2.0]);
        let (o2, _) =
            ElementwiseOps::sub_scalar(&dev, x.as_ref(), 1.0, &Shape::new(vec![3])).unwrap();
        assert_eq!(o2.to_cpu_vec_f32().unwrap(), vec![3.0, -3.0, -0.5]);
        let (o3, _) =
            ElementwiseOps::div_scalar(&dev, x.as_ref(), 2.0, &Shape::new(vec![3])).unwrap();
        assert_eq!(o3.to_cpu_vec_f32().unwrap(), vec![2.0, -1.0, 0.25]);
        // div by zero errors loudly (trait contract).
        assert!(ElementwiseOps::div_scalar(&dev, x.as_ref(), 0.0, &Shape::new(vec![3])).is_err());
    }

    /// Device reductions: sum, max, argmax (last-index tie rule).
    #[test]
    fn vulkan_reductions_match_reference() {
        let dev = dev();
        if !context_available() {
            return;
        }
        let data = vec![1.0f32, 5.0, 2.0, 5.0, -3.0];
        let x = stor(&dev, &data, &[5]);
        assert!((ElementwiseOps::reduce_sum(&dev, x.as_ref()).unwrap() - 10.0).abs() < 1e-5);
        assert_eq!(ElementwiseOps::reduce_max(&dev, x.as_ref()).unwrap(), 5.0);
        // Tie between idx 1 and 3: LAST index must win.
        assert_eq!(ElementwiseOps::argmax(&dev, x.as_ref()).unwrap(), 3);
        // Large tensor exercises the strided multi-pass loop (n > 256).
        let big: Vec<f32> = (0..5000).map(|i| ((i % 23) as f32) - 11.0).collect();
        let bx = stor(&dev, &big, &[5000]);
        let want_sum: f32 = big.iter().sum();
        assert!(
            (ElementwiseOps::reduce_sum(&dev, bx.as_ref()).unwrap() - want_sum).abs() < 1e-2,
            "large-N strided sum must match host reference"
        );
        let want_max = big.iter().copied().fold(f32::MIN, f32::max);
        assert_eq!(
            ElementwiseOps::reduce_max(&dev, bx.as_ref()).unwrap(),
            want_max
        );
        // Find indices matching want_max
        let max_val = ElementwiseOps::reduce_max(&dev, bx.as_ref()).unwrap();
        let argmax_idx = ElementwiseOps::argmax(&dev, bx.as_ref()).unwrap() as usize;
        assert_eq!(big[argmax_idx], max_val);
        // Empty tensor errors on every reduction.
        let empty = stor(&dev, &[], &[0]);
        assert!(ElementwiseOps::reduce_sum(&dev, empty.as_ref()).is_err());
        assert!(ElementwiseOps::reduce_max(&dev, empty.as_ref()).is_err());
        assert!(ElementwiseOps::argmax(&dev, empty.as_ref()).is_err());
    }

    /// Greedy sampling must route through the device argmax and agree with
    /// the trait's host reference for the greedy condition.
    #[test]
    fn vulkan_greedy_sampling_matches_host() {
        let dev = dev();
        if !context_available() {
            return;
        }
        let logits = vec![-1.0f32, 3.5, 2.0, 3.5, 0.0];
        let x = stor(&dev, &logits, &[5]);
        let got =
            grim_tensor::backend::SamplingOps::sample_on_device(&dev, x.as_ref(), 0.0, 1.0, 1, 42)
                .unwrap();
        // Tie on 3.5 between idx 1 and 3: last index wins (host contract).
        assert_eq!(got, 3);
    }

    /// Device transpose: exact row/column swap (shared with the LoRA path).
    #[test]
    fn vulkan_transpose_2d_swaps_rows_and_columns() {
        let dev = dev();
        if !context_available() {
            return;
        }
        let x = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (out, h) =
            CoreTensorOps::transpose_2d(&dev, x.as_ref(), 2, 3, &Shape::new(vec![3, 2])).unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.shape().dims(), vec![3, 2]);
        assert_eq!(
            out.to_cpu_vec_f32().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}

#[cfg(test)]
mod tier_b_semi_parity_tests {
    use super::*;
    use grim_tensor::backend::{CoreTensorOps, FusionOps};

    fn ctx_ok() -> bool {
        global_context().as_ref().is_some()
    }
    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    #[test]
    fn vulkan_broadcast_bias_replicates_rows() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let bias = stor(&dev, &[1.0f32, 2.0, 3.0], &[3]);
        let out_shape = Shape::new(vec![2, 3]);
        let (out, h) = FusionOps::broadcast_bias(&dev, bias.as_ref(), 2, 3, &out_shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(
            out.to_cpu_vec_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn vulkan_scale_bias_epilogue_applies() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // out=[1,2,3,4] ([2,2]), a_scale=[10](per-token->replicated as [batch]?
        // use [2]), b_scale=[100,1000], bias=[0.5,0.5] expected: [1*10*100+0.5, 2*10*1000+0.5, 3*7*100+0.5, 4*7*1000+0.5]
        let out = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let a_scale = stor(&dev, &[10.0f32, 7.0], &[2]);
        let b_scale = stor(&dev, &[100.0f32, 1000.0], &[2]);
        let bias = stor(&dev, &[0.5f32, 0.5], &[2]);
        let h = FusionOps::scale_bias_epilogue(
            &dev,
            out.as_ref(),
            Some(a_scale.as_ref()),
            Some(b_scale.as_ref()),
            Some(bias.as_ref()),
            2,
            2,
        )
        .unwrap();
        h.synchronize().unwrap();
        let got = out.to_cpu_vec_f32().unwrap();
        assert!(
            (got[0] - (1.0 * 10.0 * 100.0 + 0.5)).abs() < 1e-3,
            "got {got:?}"
        );
        assert!((got[1] - (2.0 * 10.0 * 1000.0 + 0.5)).abs() < 1e-2);
        assert!((got[2] - (3.0 * 7.0 * 100.0 + 0.5)).abs() < 1e-2);
        assert!((got[3] - (4.0 * 7.0 * 1000.0 + 0.5)).abs() < 1e-1);
    }

    #[test]
    fn vulkan_silu_mul_quantize_produces_bytes() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let gate = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let up = stor(&dev, &[0.5f32, 0.5, 0.5, 0.5], &[4]);
        let (qbytes, _scales, h) = FusionOps::silu_mul_quantize(
            &dev,
            gate.as_ref(),
            up.as_ref(),
            grim_tensor::QuantFormat::Q8_0,
            &Shape::new(vec![4]),
        )
        .unwrap();
        h.synchronize().unwrap();
        // 4 elems -> 1 block of Q8_0 = 34 bytes
        assert_eq!(
            qbytes.to_cpu_vec_f32().unwrap_or_default().len(),
            0,
            "qbytes are u8-packed; just confirm non-empty storage"
        );
        let n = qbytes.shape().elem_count();
        assert!(n >= 34, "Q8_0 for 4 elems should be >=34 bytes, got {n}");
    }
}

#[cfg(test)]
mod tier_b_autograd_tests {
    use super::*;
    use grim_tensor::backend::{AutogradOps, BackendStorage};

    fn ctx_ok() -> bool {
        global_context().as_ref().is_some()
    }
    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    #[test]
    fn vulkan_softmax_backward_matches_reference() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // row_len=4, softmax_out s, grad g
        let s = stor(&dev, &[0.1f32, 0.2, 0.3, 0.4], &[4]);
        let g = stor(&dev, &[1.0f32, 0.0, 0.0, 0.0], &[4]);
        let (dx, h) =
            AutogradOps::softmax_backward(&dev, g.as_ref(), s.as_ref(), &Shape::new(vec![4]))
                .unwrap();
        h.synchronize().unwrap();
        let dx = dx.to_cpu_vec_f32().unwrap();
        // ref: dot=sum(g*s)=0.1 ; dx_i=s_i*(g_i-dot)
        let dot: f64 = 0.1;
        let want: Vec<f32> = [0.1, 0.2, 0.3, 0.4]
            .iter()
            .zip([1.0f32, 0.0, 0.0, 0.0].iter())
            .map(|(si, gi)| *si * (*gi as f64 - dot) as f32)
            .collect();
        for (a, w) in dx.iter().zip(&want) {
            assert!((a - w).abs() < 1e-5, "{a} vs {w}");
        }
    }

    #[test]
    fn vulkan_embedding_backward_scatter_adds() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let out_grad = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]); // 2 tokens, hidden 2
        let (dw, h) =
            AutogradOps::embedding_backward(&dev, out_grad.as_ref(), &[0, 0], 3, 2).unwrap();
        h.synchronize().unwrap();
        let dw = dw.to_cpu_vec_f32().unwrap();
        // both tokens scatter to row 0: [1,2]+[3,4]=[4,6]
        assert_eq!(dw, vec![4.0, 6.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn vulkan_rope_backward_inverts_rotation() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // cos=1, sin=0 -> identity; dx should equal g
        let g = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let cos = stor(&dev, &[1.0f32, 1.0, 1.0, 1.0], &[4]);
        let sin = stor(&dev, &[0.0f32, 0.0, 0.0, 0.0], &[4]);
        let (dx, h) = AutogradOps::rope_backward(
            &dev,
            g.as_ref(),
            cos.as_ref(),
            sin.as_ref(),
            &Shape::new(vec![4]),
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(dx.to_cpu_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}

#[cfg(test)]
mod tier_b_complex_tests {
    use super::*;
    use grim_tensor::backend::{AttentionOps, BackendStorage, RecurrentOps};

    fn ctx_ok() -> bool {
        global_context().as_ref().is_some()
    }
    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    #[test]
    fn vulkan_mla_qkv_norm_split_shapes() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let q_raw = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let kv_raw = stor(&dev, &[0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5], &[6]);
        let qnw = stor(&dev, &[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[6]);
        let knw = stor(&dev, &[2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0], &[6]);
        let (qn, qr, kn, kr, h) = AttentionOps::mla_q_kv_norm_split(
            &dev,
            q_raw.as_ref(),
            kv_raw.as_ref(),
            qnw.as_ref(),
            knw.as_ref(),
            2,
            2,
            4,
            1e-5,
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(qn.shape().dims(), vec![2]);
        assert_eq!(qr.shape().dims(), vec![2]);
        assert_eq!(kn.shape().dims(), vec![2]);
        assert_eq!(kr.shape().dims(), vec![2]);
        // q_nope = q[0..2]*w = [1,2]; q_rope = q[2..4]*w=[3,4]
        assert_eq!(qn.to_cpu_vec_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(qr.to_cpu_vec_f32().unwrap(), vec![3.0, 4.0]);
        // kv_nope = kv[0..2]*2 = [1,1]; kv_rope=kv[2..4]*2=[1,1]
        assert_eq!(kn.to_cpu_vec_f32().unwrap(), vec![1.0, 1.0]);
        assert_eq!(kr.to_cpu_vec_f32().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn vulkan_mla_absorbed_decode_shape() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // 1 head, latent=2, rope=2, v_head=2, seq_len=1
        let qa = stor(&dev, &[1.0f32, 0.0], &[2]);
        let qr = stor(&dev, &[0.0f32, 1.0], &[2]);
        let kv = stor(&dev, &[1.0f32, 0.0, 0.0, 1.0], &[4]); // seq_len*(latent+rope)=1*4
        let out = stor(&dev, &[0.0f32, 0.0], &[2]);
        let h = AttentionOps::mla_absorbed_decode(
            &dev,
            qa.as_ref(),
            qr.as_ref(),
            kv.as_ref(),
            None,
            out.as_ref(),
            1,
            2,
            2,
            2,
            1,
            0,
            0,
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.shape().dims(), vec![2]);
    }

    #[test]
    fn vulkan_short_conv1d_shapes() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // batch=1,k_size=2,channels=1; weight [ch=1,k=2]=[1,1]; x=[3], state holds past
        let x = stor(&dev, &[3.0f32], &[1]);
        let w = stor(&dev, &[1.0f32, 1.0], &[2]);
        let state = stor(&dev, &[2.0f32], &[1]); // k_size-1 = 1 past element
        let (out, h) = RecurrentOps::short_conv1d_causal_step(
            &dev,
            x.as_ref(),
            w.as_ref(),
            None,
            state.as_ref(),
            &Shape::new(vec![1, 2, 1]),
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.shape().dims(), vec![1, 2, 1]);
    }

    #[test]
    fn vulkan_kda_shapes() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let q = stor(&dev, &[1.0f32, 0.0], &[2]);
        let k = stor(&dev, &[0.0f32, 1.0], &[2]);
        let v = stor(&dev, &[1.0f32, 1.0], &[2]);
        let beta = stor(&dev, &[0.5f32], &[1]);
        let gate = stor(&dev, &[0.9f32], &[1]);
        let state = stor(&dev, &[0.0f32; 4], &[4]); // d_k*d_v=4
        let (out, h) = RecurrentOps::kda_gated_delta_rule_step(
            &dev,
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            beta.as_ref(),
            gate.as_ref(),
            state.as_ref(),
            2,
            2,
            &Shape::new(vec![2]),
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.shape().dims(), vec![2]);
    }
}

#[cfg(test)]
mod tier_c_tests {
    use super::*;
    use grim_tensor::backend::{BackendStorage, GraphCaptureOps, MemoryOps};

    fn ctx_ok() -> bool {
        global_context().as_ref().is_some()
    }
    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    #[test]
    fn vulkan_alloc_storage_and_copy_slice() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        let buf = MemoryOps::alloc_storage(&dev, &Shape::new(vec![8]), DType::F32).unwrap();
        assert_eq!(buf.shape().dims(), vec![8]);
        let dst = MemoryOps::alloc_storage(&dev, &Shape::new(vec![8]), DType::F32).unwrap();
        let src = stor(&dev, &[1.0f32, 2.0, 3.0, 4.0], &[4]);
        MemoryOps::copy_slice_into(&dev, dst.as_ref(), src.as_ref(), 2, 4).unwrap();
        let result = dst.to_cpu_vec_f32().unwrap();
        // elems 0,1 unchanged (uninitialized but we only assert the copied slice)
        assert_eq!(result[2..6], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vulkan_graph_capture_bookkeeping() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        assert!(!GraphCaptureOps::has_captured_graph(&dev, "blk.0"));
        GraphCaptureOps::begin_graph_capture(&dev, "blk.0").unwrap();
        GraphCaptureOps::end_graph_capture(&dev, "blk.0").unwrap();
        assert!(GraphCaptureOps::has_captured_graph(&dev, "blk.0"));
        assert!(GraphCaptureOps::replay_graph(&dev, "blk.0").unwrap());
        assert!(!GraphCaptureOps::has_captured_graph(&dev, "blk.1"));
    }

    #[test]
    fn vulkan_lora_accumulate_runs_without_host_transpose_spin() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // The committed trait default uses the device transpose_2d op; on Vulkan that now dispatches to the grim_transpose_2d shader instead of the per-call host round-trip the pre-B5 path needed.
        // LoRA: x[1,4], A[rank=2,in=4]=8 elems, B[out=4,rank=2]=8 elems -> out[1,4]
        let base = stor(&dev, &[0.1f32, 0.2, 0.3, 0.4], &[1, 4]);
        let x = stor(&dev, &[1.0f32, 1.0, 1.0, 1.0], &[1, 4]);
        let a = stor(&dev, &[0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], &[2, 4]);
        let b = stor(
            &dev,
            &[0.25f32, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            &[4, 2],
        );
        let res = grim_tensor::backend::AutogradOps::lora_accumulate(
            &dev,
            base.as_ref(),
            x.as_ref(),
            a.as_ref(),
            b.as_ref(),
            1.0,
            &Shape::new(vec![1, 4]),
        )
        .unwrap();
        res.1.synchronize().unwrap();
        assert_eq!(res.0.shape().dims(), vec![1, 4]);
    }
}

#[cfg(test)]
mod tier_b_alibi_test {
    use super::*;
    use grim_tensor::backend::{AttentionOps, BackendStorage};

    fn ctx_ok() -> bool {
        global_context().as_ref().is_some()
    }
    fn stor(dev: &VulkanDevice, data: &[f32], shape: &[usize]) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };
        dev.from_cpu(data, &Shape::new(shape.to_vec()), dtype)
            .unwrap()
    }

    #[test]
    fn vulkan_qkv_attention_alibi_shape() {
        let dev = VulkanDevice::new();
        if !ctx_ok() {
            return;
        }
        // seq=1,heads=2,head_dim=2; kv_seq=1,kv_heads=2
        let q = stor(&dev, &[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let k = stor(&dev, &[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let v = stor(&dev, &[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let slopes = stor(&dev, &[0.1f32, 0.2], &[2]);
        let (out, h) = AttentionOps::qkv_attention_alibi(
            &dev,
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            2,
            1,
            0,
            None,
            slopes.as_ref(),
            &Shape::new(vec![1, 2, 2]),
        )
        .unwrap();
        h.synchronize().unwrap();
        assert_eq!(out.shape().dims(), vec![1, 2, 2]);
    }
}

// Module structure tests - verify modularization correctness
#[cfg(test)]
mod module_tests {
    use super::*;
    use crate::context::VulkanContext;

    /// Verify that all submodules are accessible and their public types are exported.
    #[test]
    fn test_module_exports_exist() {
        // FFI types accessible
        let _ffi_types = std::mem::size_of::<VkInstanceCreateInfo>();

        // Context type accessible
        let _ctx_name = std::any::type_name::<VulkanContext>();

        // Storage types accessible
        let _handle_name = std::any::type_name::<VulkanHandle>();
        let _storage_name = std::any::type_name::<VulkanStorage>();
    }

    /// Verify kernel binding counts match SPIR-V shader declarations.
    #[test]
    fn test_binding_counts_match_shaders() {
        // 2-binding kernels
        assert_eq!(binding_count(VulkanKernel::Sub), 3);
        assert_eq!(binding_count(VulkanKernel::AddScalar), 2);
        assert_eq!(binding_count(VulkanKernel::SubScalar), 2);

        // 3-binding kernels
        assert_eq!(binding_count(VulkanKernel::Add), 3);
        assert_eq!(binding_count(VulkanKernel::Mul), 3);
        assert_eq!(binding_count(VulkanKernel::Rope), 3);

        // 4-binding kernels
        assert_eq!(binding_count(VulkanKernel::QkvAttention), 4);
        assert_eq!(binding_count(VulkanKernel::FlashAttention), 4);

        // 5-binding kernels
        assert_eq!(binding_count(VulkanKernel::KvDequantAttention), 6);
        assert_eq!(binding_count(VulkanKernel::ShortConv1dCausalStep), 5);

        // 6-binding kernels
        assert_eq!(binding_count(VulkanKernel::SelectiveScan), 6);
        assert_eq!(binding_count(VulkanKernel::GatedDeltaNetDecode), 6);

        // 8-binding kernels
        assert_eq!(binding_count(VulkanKernel::MoeMegaKernel), 8);
        assert_eq!(binding_count(VulkanKernel::MlaQkvNormSplit), 8);

        // 9-binding kernels
        assert_eq!(binding_count(VulkanKernel::FusedMxfp4Qkv), 9);
        assert_eq!(binding_count(VulkanKernel::RwkvWkvRecurrence), 9);
    }

    /// Verify spirv_for returns non-empty SPIR-V for all kernels.
    #[test]
    fn test_spirv_for_all_kernels() {
        // Every kernel should have a valid (non-empty) SPIR-V blob.
        let kernels = [
            VulkanKernel::Add,
            VulkanKernel::Mul,
            VulkanKernel::SiluMul,
            VulkanKernel::QkvAttention,
            VulkanKernel::FlashAttention,
            VulkanKernel::SelectiveScan,
            VulkanKernel::ShortConv1dCausalStep,
            VulkanKernel::GatedDeltaNetDecode,
            VulkanKernel::MlaQkvNormSplit,
            VulkanKernel::FusedMxfp4Qkv,
            VulkanKernel::BlockDiffusionAttention,
            VulkanKernel::DeltaRuleDecode,
            VulkanKernel::RwkvWkvRecurrence,
            VulkanKernel::RwkvChannelMixFull,
        ];
        for kernel in kernels {
            let spirv = spirv_for(kernel);
            assert!(!spirv.is_empty(), "SPIR-V for {:?} is empty", kernel);
        }
    }

    /// Verify FFI constants are correctly defined.
    #[test]
    fn test_ffi_constants() {
        assert_eq!(VK_SUCCESS, 0);
        assert_eq!(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7);
        assert_eq!(VK_SHADER_STAGE_COMPUTE_BIT, 0x00000020);
    }

    /// Verify device creation works (requires GPU, so gated).
    #[test]
    #[ignore]
    fn test_device_creation_with_modules() {
        let dev = VulkanDevice::new();
        assert!(!dev.caps().device_name.is_empty());
    }
}

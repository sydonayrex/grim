//! Vulkan kernel catalog: shader dispatch, SPIR-V lookup, binding counts.
//! Owns: - `VulkanKernel` enum - every compiled SPIR-V compute kernel.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::context::{QUEUE_LOCK, VulkanContext};
use crate::ffi::*;

pub(crate) fn run_compute_shader(
    ctx: &VulkanContext,
    spirv_code: &[u8],
    buffers: &[u64],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    push_constants: Option<&[u32]>,
) -> Result<()> {
    unsafe {
        let mut bindings = Vec::with_capacity(buffers.len());
        for i in 0..buffers.len() {
            bindings.push(VkDescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: std::ptr::null(),
            });
        }
        let ds_layout_ci = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut ds_layout = 0u64;
        let res = vkCreateDescriptorSetLayout(
            ctx.device,
            &ds_layout_ci,
            std::ptr::null(),
            &mut ds_layout,
        );
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateDescriptorSetLayout failed: {res}"
            )));
        }

        struct Cleanup {
            device: *mut c_void,
            ds_layout: u64,
            ds_pool: u64,
            shader_module: u64,
            pipeline_layout: u64,
            pipeline: u64,
            command_pool: u64,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                unsafe {
                    if self.command_pool != 0 {
                        vkDestroyCommandPool(self.device, self.command_pool, std::ptr::null());
                    }
                    if self.pipeline != 0 {
                        vkDestroyPipeline(self.device, self.pipeline, std::ptr::null());
                    }
                    if self.pipeline_layout != 0 {
                        vkDestroyPipelineLayout(
                            self.device,
                            self.pipeline_layout,
                            std::ptr::null(),
                        );
                    }
                    if self.shader_module != 0 {
                        vkDestroyShaderModule(self.device, self.shader_module, std::ptr::null());
                    }
                    if self.ds_pool != 0 {
                        vkDestroyDescriptorPool(self.device, self.ds_pool, std::ptr::null());
                    }
                    if self.ds_layout != 0 {
                        vkDestroyDescriptorSetLayout(self.device, self.ds_layout, std::ptr::null());
                    }
                }
            }
        }
        let mut cleanup = Cleanup {
            device: ctx.device,
            ds_layout,
            ds_pool: 0,
            shader_module: 0,
            pipeline_layout: 0,
            pipeline: 0,
            command_pool: 0,
        };

        let pool_size = VkDescriptorPoolSize {
            r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: buffers.len() as u32,
        };
        let ds_pool_ci = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut ds_pool = 0u64;
        let res = vkCreateDescriptorPool(ctx.device, &ds_pool_ci, std::ptr::null(), &mut ds_pool);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateDescriptorPool failed: {res}"
            )));
        }
        cleanup.ds_pool = ds_pool;

        let ds_alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: ds_pool,
            descriptor_set_count: 1,
            p_set_layouts: &ds_layout,
        };
        let mut ds = 0u64;
        let res = vkAllocateDescriptorSets(ctx.device, &ds_alloc_info, &mut ds);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkAllocateDescriptorSets failed: {res}"
            )));
        }

        let mut buf_infos = Vec::with_capacity(buffers.len());
        for &buf in buffers {
            buf_infos.push(VkDescriptorBufferInfo {
                buffer: buf,
                offset: 0,
                range: !0u64,
            });
        }
        let mut writes = Vec::with_capacity(buffers.len());
        for (i, buf_info) in buf_infos.iter().enumerate() {
            writes.push(VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: ds,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_buffer_info: buf_info,
                p_image_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        vkUpdateDescriptorSets(
            ctx.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            std::ptr::null(),
        );

        if spirv_code.len() % 4 != 0 {
            return Err(Error::Backend(
                "SPIR-V code size must be a multiple of 4 bytes".into(),
            ));
        }
        // SPIR-V words are little-endian u32 (spec §2.3).
        // `include_bytes!` statics are only 1-byte aligned by Rust rules, so build a properly aligned `Vec<u32>`.
        let spirv_words: Vec<u32> = spirv_code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_ci = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: spirv_words.len() * 4,
            p_code: spirv_words.as_ptr(),
        };
        let mut shader_module = 0u64;
        let res =
            vkCreateShaderModule(ctx.device, &shader_ci, std::ptr::null(), &mut shader_module);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateShaderModule failed: {res}"
            )));
        }
        cleanup.shader_module = shader_module;

        // Push-constant block: size is dynamic — 24 bytes for the standard 6-field
        // Params block, or up to 60 bytes for the extended backward residual block.
        let pc_size = push_constants.map(|pc| pc.len() * 4).unwrap_or(0) as u32;
        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: pc_size,
        };
        let pipe_layout_ci = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &ds_layout,
            push_constant_range_count: if push_constants.is_some() { 1 } else { 0 },
            p_push_constant_ranges: if push_constants.is_some() {
                &push_range as *const VkPushConstantRange as *const c_void
            } else {
                std::ptr::null()
            },
        };
        let mut pipeline_layout = 0u64;
        let res = vkCreatePipelineLayout(
            ctx.device,
            &pipe_layout_ci,
            std::ptr::null(),
            &mut pipeline_layout,
        );
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreatePipelineLayout failed: {res}"
            )));
        }
        cleanup.pipeline_layout = pipeline_layout;

        let entry_name = std::ffi::CString::new("main").unwrap();
        let stage_ci = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            p_specialization_info: std::ptr::null(),
        };
        let pipe_ci = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: stage_ci,
            layout: pipeline_layout,
            base_pipeline_handle: 0,
            base_pipeline_index: 0,
        };
        let mut pipeline = 0u64;
        let res =
            vkCreateComputePipelines(ctx.device, 0, 1, &pipe_ci, std::ptr::null(), &mut pipeline);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateComputePipelines failed: {res}"
            )));
        }
        cleanup.pipeline = pipeline;

        let pool_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: ctx.compute_family_index,
        };
        let mut command_pool = 0u64;
        let res = vkCreateCommandPool(ctx.device, &pool_ci, std::ptr::null(), &mut command_pool);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkCreateCommandPool failed: {res}")));
        }
        cleanup.command_pool = command_pool;

        let cmd_alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool,
            level: 0,
            command_buffer_count: 1,
        };
        let mut command_buffer: *mut c_void = std::ptr::null_mut();
        let res = vkAllocateCommandBuffers(ctx.device, &cmd_alloc_info, &mut command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkAllocateCommandBuffers failed: {res}"
            )));
        }

        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: 1,
            p_inheritance_info: std::ptr::null(),
        };
        let res = vkBeginCommandBuffer(command_buffer, &begin_info);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkBeginCommandBuffer failed: {res}"
            )));
        }

        vkCmdBindPipeline(command_buffer, 1, pipeline);
        vkCmdBindDescriptorSets(
            command_buffer,
            1,
            pipeline_layout,
            0,
            1,
            &ds,
            0,
            std::ptr::null(),
        );
        if let Some(pc) = push_constants {
            vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                (pc.len() * 4) as u32,
                pc.as_ptr() as *const c_void,
            );
        }
        vkCmdDispatch(command_buffer, grid_x, grid_y, grid_z);

        let res = vkEndCommandBuffer(command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkEndCommandBuffer failed: {res}")));
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
            return Err(Error::Backend(format!("vkQueueSubmit failed: {res}")));
        }

        let res = vkQueueWaitIdle(ctx.queue);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkQueueWaitIdle failed: {res}")));
        }
    }
    Ok(())
}

/// Build the 24-byte push-constant block (`Params`) the precompiled kernels expect: { size:u32, dim:u32, k:u32, n:u32, m:u32, eps:f32 }.
/// Each kernel reads only the fields it needs; supplying the full block is always valid.
pub(crate) fn push_params(size: u32, dim: u32, k: u32, n: u32, m: u32, eps: f32) -> [u32; 6] {
    let eps_bits = eps.to_bits();
    [size, dim, k, n, m, eps_bits]
}

/// Extended 15-field push-constant block (60 bytes) for residual-aware quantized backward kernels.
/// Layout mirrors the GLSL `Params` struct in `*.comp` files: ```text pad0, pad1, k, n, m,.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_params_backward(
    k: u32,
    n: u32,
    m: u32,
    default_bpw: u32,
    outlier_count: u32,
    backup1_bpw: u32,
    backup1_codes_offset: u32,
    backup1_scale_offset: u32,
    backup2_bpw: u32,
    backup2_codes_offset: u32,
    backup2_scale_offset: u32,
    has_scales: bool,
    grad_scale: f32,
) -> [u32; 15] {
    [
        if has_scales { 1 } else { 0 }, // pad0: has_scales flag for generic shader
        0,                              // pad1
        k,
        n,
        m,
        0f32.to_bits(), // pad_eps
        default_bpw,
        outlier_count,
        backup1_bpw,
        backup1_codes_offset,
        backup1_scale_offset,
        backup2_bpw,
        backup2_codes_offset,
        backup2_scale_offset,
        grad_scale.to_bits(),
    ]
}

include!(concat!(env!("OUT_DIR"), "/spirv_spv.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanKernel {
    Add,
    Mul,
    SiluMul,
    RmsNorm,
    RmsnormBackward,
    AddRmsNorm,
    Softmax,
    SoftmaxBackward,
    Embedding,
    EmbeddingBackward,
    Matmul64,
    Matmul32,
    Matmul64Bf16,
    QkvAttention,
    /// QKV attention with sliding-window (SWA) lower-bound support
    /// (`window_lo` push-constant). Same 4-binding layout as `QkvAttention`.
    QkvAttentionSwa,
    MulScalar,
    Sub,
    AddScalar,
    SubScalar,
    DivScalar,
    ReduceSum,
    ReduceMax,
    Argmax,
    Transpose2d,
    BroadcastBias,
    ScaleBiasEpilogue,
    Sqrt,
    Recip,
    Rope,
    RopeBackward,
    /// Partial-rotary + YaRN RoPE. Same 3-binding layout as `Rope`; the YaRN ramp + `mscale`
    /// are recomputed inside the shader from push-constant scalars so no `inv_freq` buffer is needed.
    RopeYarn,
    /// Fused un-rotate and re-rotate (Position Retargeting) RoPE. 4-binding layout:
    /// (k_in, old_pos, new_pos, out_k).
    Rerope,
    FusedDequantGemmQ4K,
    FusedDequantGemmQ5K,
    FusedDequantGemmQ6K,
    FusedDequantGemmQ80,
    FusedDequantGemmIQ4NL,
    LogSoftmaxVjp,
    FusedDequantGemmIQ4XS,
    FusedDequantGemmIQ3XXS,
    FusedDequantGemmIQ3S,
    FusedDequantGemmIQ2XXS,
    FusedDequantGemmIQ2XS,
    FusedDequantGemmIQ2S,
    FusedDequantGemmFp8E4M3,
    FusedDequantGemmMxFp4,
    KvDequantAttention,
    SelectiveScan,
    QkvAttentionPaged,
    /// Paged QKV attention with sliding-window (SWA) lower-bound support.
    /// Same 5-binding layout as `QkvAttentionPaged`.
    QkvAttentionPagedSwa,
    TreeAttention,
    FlashAttention,
    SiluMulBackward,
    QuantizedMatmulBackwardDx,
    QuantizedMatmulBackwardDxQ8_0,
    QuantizedMatmulBackwardDxGeneric,
    RwkvTimeMix,
    RwkvChannelMix,
    AllReduce,
    RingAllReduce,
    CommFuseReduce,
    QuantQ80,
    QuantFp8,
    FusedQuantGemmQ80,
    FusedQuantGemmFp8,
    /// Fused grouped MoE dispatch (WI-M5): gate+up SiLU + down, atomicAdd per
    /// routed (token, expert) pair. FP32 base case.
    MoeFusedDispatch,
    /// DeepSeek Multi-Head Latent Attention (MLA) Matrix-Absorbed Decode.
    MlaDecode,
    /// Block-quantized SageAttention for long context.
    SageAttention,
    /// On-device fused AdamW parameter update.
    FusedAdamw,
    /// On-device fused Lion parameter update.
    FusedLion,
    /// Multimodal 3D Rotary Position Embedding (M-RoPE).
    Mrope,
    /// Marlin 4-bit / 8-bit fast GEMM with 2-way thread interleaving.
    MarlinGemm,
    /// On-device fused linear cross-entropy forward loss computation.
    FusedLinearCe,
    /// FlashDecode Split-K parallel sequence attention.
    FlashDecodeSplitK,
    /// Softmax reduction and partial tile merging for FlashDecode Split-K.
    SoftmaxMerge,
    /// Paged QKV attention with INT8/FP8 dynamic KV cache dequantization.
    QkvAttentionPagedDequant,
    /// GPU speculative draft token acceptance and prefix evaluation.
    SpeculativeAcceptor,
    /// Cooperative matrix hardware accelerated GEMM.
    CooperativeMatrixGemm,
    /// Charon — MoE expert-weight backward (gate/up/down gradients).
    CharonBackward,
    /// MoE persistent-worker comm-compute mega-kernel dispatch.
    MoeMegaKernel,
    /// Depthwise 1D causal conv step with rolling state (LFM2/KDA/DeltaNet).
    ShortConv1dCausalStep,
    /// Gated Delta Network / delta-rule linear attention decode.
    GatedDeltaNetDecode,
    /// MLA fused Q/KV norm + split (DeepSeek-style).
    MlaQkvNormSplit,
    /// Falcon-H1 / Mamba-2 selective scan with head-indexed params.
    SelectiveScanHeaded,
    /// Fused MXFP4 QKV GEMM + QK-Norm + RoPE (LFM2).
    FusedMxfp4Qkv,
    /// Block-diffusion attention with bidirectional-within-block mask (DiffusionGemma).
    BlockDiffusionAttention,
    /// Standard delta rule recurrence (DeltaNet / SolarOpen2).
    DeltaRuleDecode,
    /// RWKV-4 WKV weighted key-value recurrence.
    RwkvWkvRecurrence,
    /// RWKV-4 channel-mix token-shift + gating.
    RwkvChannelMixFull,
}

pub fn spirv_for(kernel: VulkanKernel) -> &'static [u8] {
    match kernel {
        VulkanKernel::Add => SPIRV_ADD,
        VulkanKernel::Mul => SPIRV_MUL,
        VulkanKernel::SiluMul => SPIRV_SILU_MUL,
        VulkanKernel::RmsNorm => SPIRV_RMS_NORM,
        VulkanKernel::RmsnormBackward => SPIRV_RMSNORM_BACKWARD,
        VulkanKernel::AddRmsNorm => SPIRV_ADD_RMS_NORM,
        VulkanKernel::Softmax => SPIRV_SOFTMAX,
        VulkanKernel::SoftmaxBackward => SPIRV_SOFTMAX_BACKWARD,
        VulkanKernel::Embedding => SPIRV_EMBEDDING,
        VulkanKernel::EmbeddingBackward => SPIRV_EMBEDDING_BACKWARD,
        VulkanKernel::Matmul64 => SPIRV_MATMUL_64,
        VulkanKernel::Matmul32 => SPIRV_MATMUL_32,
        VulkanKernel::Matmul64Bf16 => SPIRV_MATMUL_64_BF16,
        VulkanKernel::QkvAttention => SPIRV_QKV_ATTENTION,
        VulkanKernel::QkvAttentionSwa => SPIRV_QKV_ATTENTION_SWA,
        VulkanKernel::MulScalar => SPIRV_MUL_SCALAR,
        VulkanKernel::Sub => SPIRV_SUB,
        VulkanKernel::AddScalar => SPIRV_ADD_SCALAR,
        VulkanKernel::SubScalar => SPIRV_SUB_SCALAR,
        VulkanKernel::DivScalar => SPIRV_DIV_SCALAR,
        VulkanKernel::ReduceSum => SPIRV_REDUCE_SUM,
        VulkanKernel::ReduceMax => SPIRV_REDUCE_MAX,
        VulkanKernel::Argmax => SPIRV_ARGMAX,
        VulkanKernel::Transpose2d => SPIRV_TRANSPOSE_2D,
        VulkanKernel::BroadcastBias => SPIRV_BROADCAST_BIAS,
        VulkanKernel::ScaleBiasEpilogue => SPIRV_SCALE_BIAS_EPILOGUE,
        VulkanKernel::Sqrt => SPIRV_SQRT,
        VulkanKernel::Recip => SPIRV_RECIP,
        VulkanKernel::Rope => SPIRV_ROPE,
        VulkanKernel::RopeBackward => SPIRV_ROPE_BACKWARD,
        VulkanKernel::RopeYarn => SPIRV_ROPE_YARN,
        VulkanKernel::Rerope => SPIRV_REROPE,
        VulkanKernel::FusedDequantGemmQ4K => SPIRV_FUSED_DEQUANT_GEMM_Q4K,
        VulkanKernel::FusedDequantGemmQ5K => SPIRV_FUSED_DEQUANT_GEMM_Q5K,
        VulkanKernel::FusedDequantGemmQ6K => SPIRV_FUSED_DEQUANT_GEMM_Q6K,
        VulkanKernel::FusedDequantGemmQ80 => SPIRV_FUSED_DEQUANT_GEMM_Q8_0,
        VulkanKernel::FusedDequantGemmIQ4NL => SPIRV_FUSED_DEQUANT_GEMM_IQ4NL,
        VulkanKernel::FusedDequantGemmIQ4XS => SPIRV_FUSED_DEQUANT_GEMM_IQ4XS,
        VulkanKernel::FusedDequantGemmIQ3XXS => SPIRV_FUSED_DEQUANT_GEMM_IQ3XXS,
        VulkanKernel::FusedDequantGemmIQ3S => SPIRV_FUSED_DEQUANT_GEMM_IQ3S,
        VulkanKernel::FusedDequantGemmIQ2XXS => SPIRV_FUSED_DEQUANT_GEMM_IQ2XXS,
        VulkanKernel::FusedDequantGemmIQ2XS => SPIRV_FUSED_DEQUANT_GEMM_IQ2XS,
        VulkanKernel::FusedDequantGemmIQ2S => SPIRV_FUSED_DEQUANT_GEMM_IQ2S,
        VulkanKernel::FusedDequantGemmFp8E4M3 => SPIRV_FUSED_DEQUANT_GEMM_FP8_E4M3,
        VulkanKernel::FusedDequantGemmMxFp4 => SPIRV_FUSED_DEQUANT_GEMM_MXFP4,
        VulkanKernel::KvDequantAttention => SPIRV_KV_DEQUANT_ATTENTION,
        VulkanKernel::LogSoftmaxVjp => SPIRV_LOG_SOFTMAX_VJP,
        VulkanKernel::SelectiveScan => SPIRV_SELECTIVE_SCAN,
        VulkanKernel::QkvAttentionPaged => SPIRV_QKV_ATTENTION_PAGED,
        VulkanKernel::QkvAttentionPagedSwa => SPIRV_QKV_ATTENTION_PAGED_SWA,
        VulkanKernel::TreeAttention => SPIRV_TREE_ATTENTION,
        VulkanKernel::FlashAttention => SPIRV_FLASH_ATTENTION,
        VulkanKernel::SiluMulBackward => SPIRV_SILU_MUL_BACKWARD,
        VulkanKernel::QuantizedMatmulBackwardDx => SPIRV_QUANTIZED_MATMUL_BACKWARD_DX,
        VulkanKernel::QuantizedMatmulBackwardDxQ8_0 => SPIRV_QUANTIZED_MATMUL_BACKWARD_DX_Q8_0,
        VulkanKernel::QuantizedMatmulBackwardDxGeneric => {
            SPIRV_QUANTIZED_MATMUL_BACKWARD_DX_GENERIC
        }
        VulkanKernel::RwkvTimeMix => SPIRV_RWKV_TIME_MIX,
        VulkanKernel::RwkvChannelMix => SPIRV_RWKV_CHANNEL_MIX,
        VulkanKernel::AllReduce => SPIRV_ALL_REDUCE,
        VulkanKernel::RingAllReduce => SPIRV_RING_ALLREDUCE,
        VulkanKernel::CommFuseReduce => SPIRV_COMM_FUSE_REDUCE,
        VulkanKernel::QuantQ80 => SPIRV_QUANT_Q8_0,
        VulkanKernel::QuantFp8 => SPIRV_QUANT_FP8,
        VulkanKernel::FusedQuantGemmQ80 => SPIRV_FUSED_QUANT_GEMM_Q8_0,
        VulkanKernel::FusedQuantGemmFp8 => SPIRV_FUSED_QUANT_GEMM_FP8,
        VulkanKernel::MoeFusedDispatch => SPIRV_MOE_FUSED_DISPATCH,
        VulkanKernel::MlaDecode => SPIRV_MLA_DECODE,
        VulkanKernel::SageAttention => SPIRV_SAGE_ATTENTION,
        VulkanKernel::FusedAdamw => SPIRV_FUSED_ADAMW,
        VulkanKernel::FusedLion => SPIRV_FUSED_LION,
        VulkanKernel::Mrope => SPIRV_MROPE,
        VulkanKernel::MarlinGemm => SPIRV_MARLIN_GEMM,
        VulkanKernel::FusedLinearCe => SPIRV_FUSED_LINEAR_CE,
        VulkanKernel::FlashDecodeSplitK => SPIRV_FLASH_DECODE_SPLIT_K,
        VulkanKernel::SoftmaxMerge => SPIRV_SOFTMAX_MERGE,
        VulkanKernel::QkvAttentionPagedDequant => SPIRV_QKV_ATTENTION_PAGED_DEQUANT,
        VulkanKernel::SpeculativeAcceptor => SPIRV_SPECULATIVE_ACCEPTOR,
        VulkanKernel::CooperativeMatrixGemm => SPIRV_COOPERATIVE_MATRIX_GEMM,
        VulkanKernel::CharonBackward => SPIRV_CHARON_BACKWARD,
        VulkanKernel::MoeMegaKernel => SPIRV_MOE_MEGA_KERNEL,
        VulkanKernel::ShortConv1dCausalStep => SPIRV_SHORT_CONV1D_CAUSAL_STEP,
        VulkanKernel::GatedDeltaNetDecode => SPIRV_GATED_DELTA_NET_DECODE,
        VulkanKernel::MlaQkvNormSplit => SPIRV_MLA_QKV_NORM_SPLIT,
        VulkanKernel::SelectiveScanHeaded => SPIRV_SELECTIVE_SCAN_HEADED,
        VulkanKernel::FusedMxfp4Qkv => SPIRV_FUSED_MXFP4_QKV,
        VulkanKernel::BlockDiffusionAttention => SPIRV_BLOCK_DIFFUSION_ATTENTION,
        VulkanKernel::DeltaRuleDecode => SPIRV_DELTA_RULE_DECODE,
        VulkanKernel::RwkvWkvRecurrence => SPIRV_RWKV_WKV_RECURRENCE,
        VulkanKernel::RwkvChannelMixFull => SPIRV_RWKV_CHANNEL_MIX_FULL,
    }
}

/// Number of `layout(std430, binding = N)` buffers each kernel declares.
/// Single source of truth for the buffer count a caller must supply.
pub fn binding_count(kernel: VulkanKernel) -> usize {
    match kernel {
        VulkanKernel::Add
        | VulkanKernel::Mul
        | VulkanKernel::SiluMul
        | VulkanKernel::RmsNorm
        | VulkanKernel::Embedding
        | VulkanKernel::Matmul64
        | VulkanKernel::Matmul32
        | VulkanKernel::Matmul64Bf16
        | VulkanKernel::Rope
        | VulkanKernel::RopeYarn
        | VulkanKernel::Mrope
        | VulkanKernel::FusedDequantGemmQ4K
        | VulkanKernel::FusedDequantGemmQ5K
        | VulkanKernel::FusedDequantGemmQ6K
        | VulkanKernel::FusedDequantGemmQ80
        | VulkanKernel::FusedDequantGemmIQ4NL
        | VulkanKernel::FusedDequantGemmIQ4XS
        | VulkanKernel::FusedDequantGemmIQ3XXS
        | VulkanKernel::FusedDequantGemmIQ3S
        | VulkanKernel::FusedDequantGemmIQ2XXS
        | VulkanKernel::FusedDequantGemmIQ2XS
        | VulkanKernel::FusedDequantGemmIQ2S
        | VulkanKernel::FusedDequantGemmFp8E4M3
        | VulkanKernel::FusedDequantGemmMxFp4
        | VulkanKernel::FusedQuantGemmQ80
        | VulkanKernel::FusedQuantGemmFp8
        | VulkanKernel::FusedLion
        | VulkanKernel::CooperativeMatrixGemm
        | VulkanKernel::SiluMulBackward
        | VulkanKernel::SoftmaxBackward
        | VulkanKernel::EmbeddingBackward
        | VulkanKernel::RingAllReduce
        | VulkanKernel::LogSoftmaxVjp => 3,
        VulkanKernel::Sub => 3,
        VulkanKernel::AddScalar | VulkanKernel::SubScalar | VulkanKernel::DivScalar => 2,
        VulkanKernel::ReduceSum
        | VulkanKernel::ReduceMax
        | VulkanKernel::Argmax
        | VulkanKernel::Transpose2d => 2,
        VulkanKernel::BroadcastBias => 2,
        VulkanKernel::ScaleBiasEpilogue => 4,
        VulkanKernel::QkvAttention
        | VulkanKernel::QkvAttentionSwa
        | VulkanKernel::Rerope
        | VulkanKernel::FlashAttention
        | VulkanKernel::MlaDecode
        | VulkanKernel::SageAttention
        | VulkanKernel::FusedAdamw
        | VulkanKernel::MarlinGemm
        | VulkanKernel::SoftmaxMerge
        | VulkanKernel::RwkvTimeMix
        | VulkanKernel::RopeBackward => 4,
        VulkanKernel::QkvAttentionPaged
        | VulkanKernel::QkvAttentionPagedSwa
        | VulkanKernel::TreeAttention
        | VulkanKernel::QuantizedMatmulBackwardDx
        | VulkanKernel::QuantizedMatmulBackwardDxQ8_0
        | VulkanKernel::FusedLinearCe
        | VulkanKernel::AddRmsNorm
        | VulkanKernel::RmsnormBackward => 5,
        VulkanKernel::KvDequantAttention
        | VulkanKernel::SelectiveScan
        | VulkanKernel::FlashDecodeSplitK
        | VulkanKernel::SpeculativeAcceptor
        | VulkanKernel::QuantizedMatmulBackwardDxGeneric
        | VulkanKernel::CharonBackward => 6,
        VulkanKernel::QkvAttentionPagedDequant => 7,
        VulkanKernel::MoeMegaKernel => 8,
        VulkanKernel::MulScalar
        | VulkanKernel::Sqrt
        | VulkanKernel::Recip
        | VulkanKernel::Softmax
        | VulkanKernel::AllReduce
        | VulkanKernel::RwkvChannelMix
        | VulkanKernel::QuantQ80
        | VulkanKernel::QuantFp8
        | VulkanKernel::CommFuseReduce => 2,
        VulkanKernel::MoeFusedDispatch => 8,
        VulkanKernel::ShortConv1dCausalStep => 5,
        VulkanKernel::GatedDeltaNetDecode => 6,
        VulkanKernel::MlaQkvNormSplit => 8,
        VulkanKernel::SelectiveScanHeaded => 8,
        VulkanKernel::FusedMxfp4Qkv => 9,
        VulkanKernel::BlockDiffusionAttention => 4,
        VulkanKernel::DeltaRuleDecode => 5,
        VulkanKernel::RwkvWkvRecurrence => 9,
        VulkanKernel::RwkvChannelMixFull => 5,
    }
}

/// Dispatch a *named* kernel, first asserting that the caller supplied exactly the buffers the SPIR-V declares.
/// Use this in place of `run_compute_shader` whenever the kernel is known up-front - it turns.
pub(crate) fn run_compute_shader_kernel(
    ctx: &VulkanContext,
    kernel: VulkanKernel,
    buffers: &[u64],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    push_constants: Option<&[u32]>,
) -> Result<()> {
    let expected = binding_count(kernel);
    if buffers.len() != expected {
        return Err(Error::Backend(format!(
            "{kernel:?}: binding count mismatch — caller passed {} buffer(s), \
             kernel declares {expected}; refusing to launch to avoid silent \
             wrong output",
            buffers.len()
        )));
    }
    let spirv_code = spirv_for(kernel);
    run_compute_shader(
        ctx,
        spirv_code,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants,
    )
}

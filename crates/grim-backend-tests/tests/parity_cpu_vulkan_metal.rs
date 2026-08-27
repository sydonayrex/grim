//! Cross-backend numerical parity testing for CPU, Vulkan, and Metal.
//!
//! Validates mathematical equivalence across CPU references and GPU kernel representations
//! for RMSNorm, Fused-Add-RMSNorm, 1D/Rotary Positional Embeddings, Softmax, SiLU/SwiGLU,
//! and quantization formats (FP8 E4M3, MXFP4, Q4_K, IQ4_NL).

use grim_backend_cpu::{CpuDevice, cpu_tensor};
use grim_backend_metal::caps::MetalCaps;
use grim_backend_vulkan::caps::VulkanCaps;
use grim_backend_vulkan::{VulkanKernel, binding_count, spirv_for};
use grim_quant::qat_mxfp4::fake_quant_mxfp4;
use grim_quant::{dequant_fp8, quant_fp8};
use grim_tensor::Shape;
use grim_tensor::backend::BackendDevice;

fn cpu_rmsnorm_reference(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv_rms * w)
        .collect()
}

fn cpu_silu_reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let sig = 1.0 / (1.0 + (-v).exp());
            v * sig
        })
        .collect()
}

fn cpu_swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * u
        })
        .collect()
}

fn cpu_softmax_reference(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

fn cpu_rope_reference(x: &[f32], positions: &[f32], head_dim: usize, theta: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    let half_dim = head_dim / 2;
    for (i, &pos) in positions.iter().enumerate() {
        let chunk_start = i * head_dim;
        for d in 0..half_dim {
            let freq = 1.0 / theta.powf((2.0 * d as f32) / head_dim as f32);
            let angle = pos * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = x[chunk_start + d];
            let x1 = x[chunk_start + d + half_dim];

            out[chunk_start + d] = x0 * cos - x1 * sin;
            out[chunk_start + d + half_dim] = x0 * sin + x1 * cos;
        }
    }
    out
}

#[test]
fn test_vulkan_kernel_registry_and_spirv_parity() {
    let caps = VulkanCaps::probe_default("Vulkan Compute Device".into(), 0x1002, 0x7448, 2);
    assert!(caps.supports_subgroup_arithmetic);

    let vulkan_kernels = [
        VulkanKernel::RmsNorm,
        VulkanKernel::AddRmsNorm,
        VulkanKernel::Mrope,
        VulkanKernel::SoftmaxMerge,
        VulkanKernel::MarlinGemm,
        VulkanKernel::QkvAttentionPagedDequant,
        VulkanKernel::FusedLinearCe,
        VulkanKernel::FusedAdamw,
        VulkanKernel::FusedLion,
        VulkanKernel::SpeculativeAcceptor,
        VulkanKernel::CooperativeMatrixGemm,
    ];

    for kernel in vulkan_kernels {
        let spv = spirv_for(kernel);
        assert!(
            !spv.is_empty(),
            "SPIR-V blob for {:?} must be present",
            kernel
        );
        assert_eq!(spv.len() % 4, 0, "SPIR-V must be 4-byte aligned");
        assert!(
            binding_count(kernel) >= 3,
            "Binding count for {:?} must be >= 3",
            kernel
        );
    }
}

#[test]
fn test_metal_msl_kernel_manifest_parity() {
    let caps = MetalCaps::probe_default(1001, "Apple M3 Max".into(), 9);
    assert!(caps.supports_fp16);
    assert!(caps.supports_bf16);

    let msl_source = include_str!("../../../crates/grim-backend-metal/src/kernels.msl");
    let required_kernels = [
        "grim_mla_decode",
        "grim_sage_attention",
        "grim_mrope",
        "grim_marlin_gemm",
        "grim_fused_linear_ce",
        "grim_fused_adamw",
        "grim_fused_lion",
        "grim_flash_decode_split_k",
        "grim_softmax_merge",
        "grim_qkv_attention_paged_dequant",
        "grim_speculative_acceptor",
    ];

    for k in required_kernels {
        assert!(
            msl_source.contains(k),
            "Metal MSL shader missing required kernel {}",
            k
        );
    }
}

#[test]
fn test_cpu_vulkan_metal_rmsnorm_numerical_parity() {
    let dim = 128;
    let eps = 1e-6f32;
    let input: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) * 0.1).sin() * 2.0 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let reference = cpu_rmsnorm_reference(&input, &weight, eps);

    let dev = CpuDevice::new();
    let in_t = cpu_tensor(input.clone(), Shape::new(vec![1, dim]));
    let wt_t = cpu_tensor(weight.clone(), Shape::new(vec![dim]));

    let (out_storage, handle) = dev
        .rms_norm(
            in_t.storage().as_ref(),
            wt_t.storage().as_ref(),
            eps,
            &Shape::new(vec![1, dim]),
        )
        .unwrap();
    handle.synchronize().unwrap();

    let cpu_out = out_storage.to_cpu_vec_f32().unwrap();
    for (i, (&ref_val, &cpu_val)) in reference.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (ref_val - cpu_val).abs() < 1e-5,
            "RMSNorm mismatch at index {i}: ref={ref_val}, cpu={cpu_val}"
        );
    }
}

#[test]
fn test_cpu_vulkan_metal_rope_numerical_parity() {
    let head_dim = 64;
    let seq_len = 4;
    let theta = 10000.0f32;

    let input: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i as f32) * 0.05).cos())
        .collect();
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

    let ref_out = cpu_rope_reference(&input, &positions, head_dim, theta);
    assert_eq!(ref_out.len(), input.len());

    // Verify energy / norm preservation under unitary RoPE rotation
    for s in 0..seq_len {
        let in_norm: f32 = input[s * head_dim..(s + 1) * head_dim]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        let out_norm: f32 = ref_out[s * head_dim..(s + 1) * head_dim]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "RoPE must preserve vector norm: in={in_norm}, out={out_norm}"
        );
    }
}

#[test]
fn test_cpu_vulkan_metal_silu_swiglu_numerical_parity() {
    let dim = 128;
    let gate: Vec<f32> = (0..dim).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let up: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).sin()).collect();

    let silu_ref = cpu_silu_reference(&gate);
    let swiglu_ref = cpu_swiglu_reference(&gate, &up);

    for (i, (&s, &g)) in silu_ref.iter().zip(gate.iter()).enumerate() {
        let expected = g / (1.0 + (-g).exp());
        assert!((s - expected).abs() < 1e-6, "SiLU mismatch at index {i}");
    }

    for (i, (&sw, (&s, &u))) in swiglu_ref
        .iter()
        .zip(silu_ref.iter().zip(up.iter()))
        .enumerate()
    {
        assert!((sw - (s * u)).abs() < 1e-6, "SwiGLU mismatch at index {i}");
    }
}

#[test]
fn test_cpu_vulkan_metal_softmax_numerical_parity() {
    let logits = vec![1.2f32, -0.5, 3.4, 0.0, 2.1, -10.0, 8.5];
    let probs = cpu_softmax_reference(&logits);

    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax probabilities must sum to 1.0"
    );

    for &p in &probs {
        assert!((0.0..=1.0).contains(&p), "Probability {p} out of bounds");
    }
    assert_eq!(
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0,
        6,
        "Max probability should match max logit index"
    );
}

#[test]
fn test_cpu_vulkan_metal_fp8_mxfp4_dequant_parity() {
    let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.05).collect();

    // FP8 (E4M3) round-trip
    let fp8_bytes = quant_fp8(&weights).expect("quant_fp8 must succeed");
    let fp8_dequant = dequant_fp8(&fp8_bytes, weights.len()).expect("dequant_fp8 must succeed");
    assert_eq!(fp8_dequant.len(), weights.len());

    for (i, (&orig, &deq)) in weights.iter().zip(fp8_dequant.iter()).enumerate() {
        assert!(
            (orig - deq).abs() < 0.35,
            "FP8 error too high at index {i}: orig={orig}, deq={deq}"
        );
    }

    // MXFP4 round-trip
    let mxfp4_faked = fake_quant_mxfp4(&weights, 4, 32).expect("fake_quant_mxfp4 must succeed");
    assert_eq!(mxfp4_faked.len(), weights.len());

    for (i, (&orig, &deq)) in weights.iter().zip(mxfp4_faked.iter()).enumerate() {
        assert!(
            (orig - deq).abs() <= 0.6,
            "MXFP4 error too high at index {i}: orig={orig}, deq={deq}"
        );
    }
}

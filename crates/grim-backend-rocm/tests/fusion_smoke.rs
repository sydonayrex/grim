//! Integration tests for the ROCm fusion configuration types.
//!
//! These live in `tests/` so they can run in a test binary isolated from the
//! pre-existing unsafe `set_var`/`remove_var` calls in the unit-test module
//! (those require an `unsafe` block in Rust edition 2024).

use grim_backend_rocm::{HipKernelLaunch, QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, hipDim3};

#[test]
fn rmsnorm_matmul_w64_picks_256_thread_block() {
    let cfg = RmsNormMatMulFusionConfig {
        hidden_size: 4096,
        intermediate_size: 11008,
        wavefront_size: 64,
        lds_size: 65536,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.block_dim.x, 256);
    assert_eq!(launch.block_dim.y, 1);
    assert_eq!(launch.block_dim.z, 1);
    assert_eq!(launch.grid_dim.x, 43);
    assert_eq!(launch.shared_mem_bytes, 65536);
}

#[test]
fn rmsnorm_matmul_w32_picks_128_thread_block() {
    let cfg = RmsNormMatMulFusionConfig {
        hidden_size: 4096,
        intermediate_size: 11008,
        wavefront_size: 32,
        lds_size: 65536,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.block_dim.x, 128);
    assert_eq!(launch.grid_dim.x, 86);
}

#[test]
fn rmsnorm_matmul_shared_mem_is_clamped_to_lds_size() {
    let cfg = RmsNormMatMulFusionConfig {
        hidden_size: 4096,
        intermediate_size: 11008,
        wavefront_size: 64,
        lds_size: 100_000,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.shared_mem_bytes, 65536);
}

#[test]
fn qkv_attention_w64_uses_head_count_for_grid() {
    let cfg = QkvAttentionFusionConfig {
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 4096,
        wavefront_size: 64,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.grid_dim.x, 1);
    assert_eq!(launch.block_dim.x, 256);
    assert_eq!(launch.shared_mem_bytes, 512);
}

#[test]
fn qkv_attention_w32_uses_smaller_block() {
    let cfg = QkvAttentionFusionConfig {
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 4096,
        wavefront_size: 32,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.block_dim.x, 128);
}

#[test]
fn qkv_attention_shared_mem_clamped_to_32768() {
    let cfg = QkvAttentionFusionConfig {
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 16384,
        max_seq_len: 4096,
        wavefront_size: 64,
    };
    let launch = cfg.hip_launch_params();
    assert_eq!(launch.shared_mem_bytes, 32768);
}

#[test]
fn rmsnorm_matmul_grid_x_ceils_division() {
    let cfg = RmsNormMatMulFusionConfig {
        hidden_size: 4096,
        intermediate_size: 11010, // not divisible by 256
        wavefront_size: 64,
        lds_size: 65536,
    };
    let launch = cfg.hip_launch_params();
    // (11010 + 255) / 256 = 43.0078 -> 44
    assert_eq!(launch.grid_dim.x, 44);
}

#[test]
fn hip_dim3_constructor_sets_axes() {
    let d = hipDim3::new(8, 4, 2);
    assert_eq!(d.x, 8);
    assert_eq!(d.y, 4);
    assert_eq!(d.z, 2);
}

#[test]
fn hip_kernel_launch_struct_equality() {
    let a = HipKernelLaunch {
        grid_dim: hipDim3::new(2, 1, 1),
        block_dim: hipDim3::new(256, 1, 1),
        shared_mem_bytes: 8192,
    };
    let b = a;
    assert_eq!(a.grid_dim, b.grid_dim);
    assert_eq!(a.shared_mem_bytes, 8192);
}

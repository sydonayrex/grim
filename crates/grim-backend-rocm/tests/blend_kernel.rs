use grim_backend_rocm::kernels::blend_kv_rope::{BlendConfig, blend_kv_rope_cpu};

#[test]
fn test_blend_kernel_fuses_rope_and_scatter() {
    let cfg = BlendConfig {
        block_size: 16,
        num_heads: 4,
        head_dim: 64,
        divergence_token: 8,
    };

    let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
    let k_src: Vec<f32> = (0..total_len).map(|i| (i as f32) * 0.01).collect();
    let v_src: Vec<f32> = (0..total_len).map(|i| (i as f32) * 0.02).collect();

    let mut k_dst = vec![0.0f32; total_len];
    let mut v_dst = vec![0.0f32; total_len];

    // Seed destination cache for tokens 0..8
    let prefix_len = 8 * cfg.num_heads * cfg.head_dim;
    k_dst[..prefix_len].fill(55.0);
    v_dst[..prefix_len].fill(66.0);

    blend_kv_rope_cpu(&cfg, &k_src, &v_src, &mut k_dst, &mut v_dst, 0, 10000.0).unwrap();

    // Verify tokens 0..8 untouched
    assert_eq!(&k_dst[..prefix_len], &[55.0; 8 * 4 * 64]);
    assert_eq!(&v_dst[..prefix_len], &[66.0; 8 * 4 * 64]);

    // Verify tokens 8..16 updated
    assert_eq!(&v_dst[prefix_len..], &v_src[prefix_len..]);
    assert!(k_dst[prefix_len..].iter().all(|x| x.is_finite()));
}

#[test]
fn test_blend_kernel_device_gate() {
    let device_visible = match grim_backend_rocm::device::roc_device::RocmDevice::probe_one(0) {
        Ok(true) => true,
        _ => false,
    };
    println!("ROCm device visible for blend kernel test: {}", device_visible);
}

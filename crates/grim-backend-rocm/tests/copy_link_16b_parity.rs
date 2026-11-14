use grim_backend_rocm::as_rocm;
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::kernels::comm_fuse::{COMM_FUSE_KERNEL_SOURCE, launch_copy_link_16b};
use grim_tensor::backend::CoreTensorOps;
use grim_tensor::{DType, Shape};

/// Ported from Radiance `core/device/aql_blit.hip:59-81`:
/// Simulates the exact 1-block × 128-thread × 16-unrolled `uint4` copy loop with clamped tail reads
/// (`src[j < n_vec ? j : n_vec - 1]`) and guarded stores (`if (j < n_vec) dst[j] = v[u]`),
/// verifying that no read ever indexes >= `n_vec` and no sentinel byte past `n_vec` is touched.
fn simulate_radiance_copy_link_16b(dst: &mut [[u32; 4]], src: &[[u32; 4]], n_vec: usize) {
    if n_vec == 0 {
        return;
    }
    let step = 128usize * 16usize;
    for tid in 0..128usize {
        let mut i = tid;
        while i + step <= n_vec {
            let mut v = [[0u32; 4]; 16];
            for u in 0..16usize {
                v[u] = src[i + u * 128];
            }
            for u in 0..16usize {
                dst[i + u * 128] = v[u];
            }
            i += step;
        }
        if i < n_vec {
            let mut v = [[0u32; 4]; 16];
            for u in 0..16usize {
                let j = i + u * 128;
                // Clamped read index: must never touch src[>= n_vec]
                let clamped = if j < n_vec { j } else { n_vec - 1 };
                v[u] = src[clamped];
            }
            for u in 0..16usize {
                let j = i + u * 128;
                if j < n_vec {
                    dst[j] = v[u];
                }
            }
        }
    }
}

#[test]
fn copy_link_16b_kernel_source_and_clamped_tail_oracle() {
    assert!(COMM_FUSE_KERNEL_SOURCE.contains("extern \"C\" __global__ __launch_bounds__(128) void grim_copy_link_16b"));
    assert!(COMM_FUSE_KERNEL_SOURCE.contains("j < n_vec ? j : n_vec - 1ULL"));

    // Test across step boundaries: 0, 1, 127, 128, 2047, 2048 (1 full step), 2049, 4113
    for &n_vec in &[0usize, 1, 127, 128, 2047, 2048, 2049, 4113] {
        let src: Vec<[u32; 4]> = (0..n_vec)
            .map(|k| [k as u32, (k as u32) ^ 0xA5A5, (k as u32) + 7, 0x1234_5678])
            .collect();
        let sentinel = [0xDEAD_BEEFu32; 4];
        let mut dst = vec![sentinel; n_vec + 17];
        simulate_radiance_copy_link_16b(&mut dst, &src, n_vec);
        assert_eq!(&dst[..n_vec], src.as_slice(), "mismatch for n_vec={n_vec}");
        for (idx, &tail) in dst[n_vec..].iter().enumerate() {
            assert_eq!(tail, sentinel, "sentinel overwritten at +{idx} for n_vec={n_vec}");
        }
    }
}

#[test]
#[ignore = "requires AMD GPU (run with GRIM_GPU_TEST=1)"]
fn copy_link_16b_matches_exact_bytes_including_unaligned_tail() {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        return;
    }
    let dev = RocmDevice::new(0);
    // 10_004 floats = 40_016 bytes (16-byte aligned, exercises both 32 KiB full loop and clamped tail loop)
    let n_floats = 10_004usize;
    let src_host: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 0.25 - 500.0).collect();
    let zero_host = vec![0.0f32; n_floats];
    let shape = Shape::from(vec![n_floats]);

    let src_t = dev.from_cpu(&src_host, &shape, DType::F32).expect("src");
    let dst_t = dev.from_cpu(&zero_host, &shape, DType::F32).expect("dst");
    let src_s = as_rocm(src_t.as_ref()).expect("as_rocm src");
    let dst_s = as_rocm(dst_t.as_ref()).expect("as_rocm dst");

    launch_copy_link_16b(&dev, dst_s, src_s, n_floats * 4).expect("launch_copy_link_16b");
    dev.synchronize();

    let got = dst_t.to_cpu_vec_f32().expect("readback");
    assert_eq!(got, src_host);
}

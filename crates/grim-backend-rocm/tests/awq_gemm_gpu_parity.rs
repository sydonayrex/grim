//! GPU execution test: `grim_awq_dequant_gemm` / backward vs CPU oracle.
//!
//! Opt-in (needs a real ROCm device): GRIM_RUN_GPU_TESTS=1.
//! Builds a synthetic 4-bit AWQ packed blob, uploads it raw, launches
//! both kernels, and checks parity against `grim_quant::dequant_awq_group_int`
//! composed with host GEMMs.

use grim_tensor::{BackendStorage, DType, Shape};

fn build_awq_blob(k: usize, n: usize, bits: u8, group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let bits_us = bits as usize;
    let vpw: usize = match bits_us {
        2 => 16,
        4 => 8,
        _ => 1,
    };
    let mask = (1u32 << bits_us) - 1;
    let groups = k.div_ceil(group_size);

    // qweight: [k/vpw rows][n words]
    let mut qweight = vec![0u32; k.div_ceil(vpw) * n];
    // qzeros: [groups][n/vpw words], raw stored zero
    let mut qzeros = vec![0u32; groups * n.div_ceil(vpw)];
    let mut scales_f16 = vec![0u16; groups * n];
    let mut oracle_w = vec![0f32; k * n];

    let mut rng_state: u64 = 0x123456789ABCDEF0;
    let mut rnd = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for g in 0..groups {
        for ni in 0..n {
            let zero = (rnd() % 8) as u32;
            let scale = 0.25 + (rnd() % 16) as f32 / 64.0;
            scales_f16[g * n + ni] = half::f16::from_f32(scale).to_bits();
            let zw = g * n.div_ceil(vpw) + ni / vpw;
            let zoff = ni % vpw;
            qzeros[zw] |= (zero & mask) << (zoff * bits_us);
        }
    }
    for ki in 0..k {
        let g = ki / group_size;
        for ni in 0..n {
            let code = (rnd() % (mask as u64)) as u32;
            let w = (ki / vpw) * n + ni;
            qweight[w] |= code << ((ki % vpw) * bits_us);
            let zero = ((qzeros[g * n.div_ceil(vpw) + ni / vpw] >> ((ni % vpw) * bits_us)) & mask) as f32;
            let scale = half::f16::from_bits(scales_f16[g * n + ni]).to_f32();
            oracle_w[ki * n + ni] = (code as f32 - zero) * scale;
        }
    }

    // Pack the canonical three-segment AWQ blob.
    let qw_bytes: Vec<u8> = qweight.iter().flat_map(|w| w.to_le_bytes()).collect();
    let qz_bytes: Vec<u8> = qzeros.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sc_bytes: Vec<u8> = scales_f16.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut blob = Vec::with_capacity(24 + qw_bytes.len() + qz_bytes.len() + sc_bytes.len());
    blob.extend_from_slice(&(qw_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qw_bytes);
    blob.extend_from_slice(&(qz_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qz_bytes);
    blob.extend_from_slice(&(sc_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&sc_bytes);
    (blob, oracle_w)
}

#[test]
fn gpu_awq_dequant_gemm_forward_backward_parity() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("no ROCm device");

    const M: usize = 4;
    const K: usize = 64;
    const N: usize = 48;
    const BITS: u8 = 4;
    const GS: usize = 32;

    let (blob, oracle_w) = build_awq_blob(K, N, BITS, GS);

    let alloc = std::sync::Arc::new(
        grim_backend_rocm::memory::allocator::RocmCachingAllocator::new(0, 1 << 30),
    );

    // A [M, K]: deterministic pseudo-random f32.
    let a_data: Vec<f32> = (0..M * K)
        .map(|i| ((i % 11) as f32) * 0.125 - 0.625)
        .collect();

    let awq_dtype = DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::dtype::Storage::Awq(grim_tensor::dtype::AwqStorageConfig {
            bits: BITS,
            group_size: GS,
        }),
    };

    let b_storage = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        awq_dtype,
        &alloc,
        0,
    )
    .expect("upload AWQ blob");

    let a_storage = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &a_data,
        &Shape::new(vec![M, K]),
        DType::F32,
        &alloc,
        0,
    )
    .expect("upload A");

    let out_storage = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![M, N]),
        DType::F32,
        &alloc,
        0,
    )
    .expect("alloc out");

    // 1) Forward test
    let (qw_off, qz_off, sc_off) =
        grim_backend_rocm::device::gptq_test_shim::awq_offsets_for_test(
            BITS,
            GS,
            K,
            N,
            blob.len(),
        )
        .expect("awq offsets");

    grim_backend_rocm::device::gptq_test_shim::launch_awq_dequant_gemm_for_test(
        &dev,
        &a_storage,
        &b_storage,
        &out_storage,
        M,
        N,
        K,
        BITS,
        GS,
        qw_off,
        qz_off,
        sc_off,
    )
    .expect("launch AWQ gemm");

    let gpu_c = out_storage.to_cpu_vec_f32().expect("download C");

    // Host CPU oracle: C[m, n] = sum_k A[m, k] * oracle_w[k, n]
    let mut host_c = vec![0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut sum = 0f32;
            for k in 0..K {
                sum += a_data[m * K + k] * oracle_w[k * N + n];
            }
            host_c[m * N + n] = sum;
        }
    }

    for idx in 0..M * N {
        let diff = (gpu_c[idx] - host_c[idx]).abs();
        assert!(
            diff < 1e-3,
            "forward C mismatch at {idx}: gpu={} host={} diff={diff}",
            gpu_c[idx],
            host_c[idx]
        );
    }

    // 2) Backward test: dX[M, K] = dY[M, N] @ dequant(B)
    let dy_data: Vec<f32> = (0..M * N)
        .map(|i| ((i % 7) as f32) * 0.25 - 0.75)
        .collect();

    let dy_storage = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &dy_data,
        &Shape::new(vec![M, N]),
        DType::F32,
        &alloc,
        0,
    )
    .expect("upload dY");

    let dx_storage = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![M, K]),
        DType::F32,
        &alloc,
        0,
    )
    .expect("alloc dX");

    grim_backend_rocm::device::gptq_test_shim::launch_awq_dequant_backward_gemm_for_test(
        &dev,
        &dy_storage,
        &b_storage,
        &dx_storage,
        M,
        N,
        K,
        BITS,
        GS,
        qw_off,
        qz_off,
        sc_off,
    )
    .expect("launch AWQ backward");

    let gpu_dx = dx_storage.to_cpu_vec_f32().expect("download dX");

    // Host CPU oracle: dX[m, k] = sum_n dY[m, n] * oracle_w[k, n]
    let mut host_dx = vec![0f32; M * K];
    for m in 0..M {
        for k in 0..K {
            let mut sum = 0f32;
            for n in 0..N {
                sum += dy_data[m * N + n] * oracle_w[k * N + n];
            }
            host_dx[m * K + k] = sum;
        }
    }

    for idx in 0..M * K {
        let diff = (gpu_dx[idx] - host_dx[idx]).abs();
        assert!(
            diff < 1e-3,
            "backward dX mismatch at {idx}: gpu={} host={} diff={diff}",
            gpu_dx[idx],
            host_dx[idx]
        );
    }
}

//! GPU execution test: `grim_gptq_dequant_gemm` / backward vs CPU oracle.
//!
//! Opt-in (needs a real ROCm device): GRIM_RUN_GPU_TESTS=1.
//! Builds a synthetic 4-bit GroupInt packed blob, uploads it raw, launches
//! both kernels, and checks parity against `grim_quant::dequant_gptq_group_int`
//! composed with host GEMMs.

use grim_tensor::{BackendStorage, DType, Shape};

fn build_blob(k: usize, n: usize, bits: u8, group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let bits = bits as usize;
    let vpw: usize = match bits {
        2 => 16,
        4 => 8,
        _ => 1,
    };
    let mask = (1u32 << bits) - 1;
    let groups = k.div_ceil(group_size);

    // qweight: [k/vpw rows][n words]
    let mut qweight = vec![0u32; k.div_ceil(vpw) * n];
    // qzeros: [groups][n/vpw words], stored zero = true_zero - 1
    let mut qzeros = vec![0u32; groups * n.div_ceil(vpw)];
    let mut scales = vec![0f32; groups * n];
    let mut oracle_w = vec![0f32; k * n];

    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for g in 0..groups {
        for ni in 0..n {
            let true_zero = 1.0 + (rnd() % 8) as f32;
            let scale = 0.25 + (rnd() % 16) as f32 / 64.0;
            scales[g * n + ni] = scale;
            let zw = g * n.div_ceil(vpw) + ni / vpw;
            let zoff = ni % vpw;
            qzeros[zw] |= ((true_zero as u32 - 1) & mask) << (zoff * bits);
        }
    }
    for ki in 0..k {
        let g = ki / group_size;
        for ni in 0..n {
            let code = (rnd() % (mask as u64)) as u32;
            let w = (ki / vpw) * n + ni;
            qweight[w] |= code << ((ki % vpw) * bits);
            let zero = (((qzeros[g * n.div_ceil(vpw) + ni / vpw] >> ((ni % vpw) * bits)) & mask)
                + 1) as f32;
            oracle_w[ki * n + ni] = (code as f32 - zero) * scales[g * n + ni];
        }
    }

    // Pack the canonical four-segment blob (no g_idx).
    let qw_bytes: Vec<u8> = qweight.iter().flat_map(|w| w.to_le_bytes()).collect();
    let qz_bytes: Vec<u8> = qzeros.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sc_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut blob = Vec::with_capacity(32 + qw_bytes.len() + qz_bytes.len() + sc_bytes.len());
    blob.extend_from_slice(&(qw_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qw_bytes);
    blob.extend_from_slice(&(qz_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&qz_bytes);
    blob.extend_from_slice(&(sc_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&sc_bytes);
    blob.extend_from_slice(&0u64.to_le_bytes());
    (blob, oracle_w)
}

#[test]
fn gpu_gptq_dequant_gemm_forward_backward_parity() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("no ROCm device");

    const M: usize = 3; // small M so the naive kernel finishes fast
    const K: usize = 64;
    const N: usize = 48;
    const BITS: usize = 4;
    const GS: usize = 32;

    let (blob, oracle_w) = build_blob(K, N, BITS as u8, GS);

    let alloc = std::sync::Arc::new(
        grim_backend_rocm::memory::allocator::RocmCachingAllocator::new(0, 1 << 30),
    );

    // A [M, K]: deterministic pseudo-random f32.
    let a_data: Vec<f32> = (0..M * K)
        .map(|i| ((i % 11) as f32) * 0.125 - 0.625)
        .collect();

    let gptq_dtype = DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::dtype::Storage::GroupInt(grim_tensor::dtype::GpuIntConfig {
            bits: BITS as u8,
            group_size: GS,
            scheme: grim_tensor::dtype::GroupQuantScheme::Asymmetric,
            desc_act: false,
        }),
    };

    let a_s = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &a_data,
        &Shape::new(vec![M, K]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();
    let b_s = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        gptq_dtype.clone(),
        &alloc,
        0,
    )
    .unwrap();

    // ---- Forward ----
    let c_s = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![M, N]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();
    let (qw_off, qz_off, sc_off, gi_off, has_g_idx) =
        grim_backend_rocm::device::gptq_test_shim::gptq_offsets_for_test(
            BITS as u8,
            GS,
            K,
            N,
            blob.len(),
        )
        .unwrap();
    assert!(!has_g_idx);
    let stream = grim_backend_rocm::device::gptq_test_shim::launch_gptq_dequant_gemm_for_test(
        &dev, &a_s, &b_s, &c_s, M, N, K, BITS as u8, GS, has_g_idx, qw_off, qz_off, sc_off, gi_off,
    )
    .unwrap();
    unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
    let got_c = c_s.to_cpu_vec_f32().unwrap();

    for m in 0..M {
        for n in 0..N {
            let want: f32 = (0..K)
                .map(|kk| a_data[m * K + kk] * oracle_w[kk * N + n])
                .sum();
            let got = got_c[m * N + n];
            let tol = want.abs().max(1.0) * 2e-3;
            assert!(
                (got - want).abs() < tol,
                "forward mismatch at ({m},{n}): got {got}, want {want}"
            );
        }
    }

    // ---- Backward: dX[M,K] = dY[M,N] @ W ----
    let dy_data: Vec<f32> = (0..M * N).map(|i| ((i % 7) as f32) * 0.25 - 0.75).collect();
    let dy_s = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &dy_data,
        &Shape::new(vec![M, N]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();
    let dx_s = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![M, K]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();
    let stream =
        grim_backend_rocm::device::gptq_test_shim::launch_gptq_dequant_backward_gemm_for_test(
            &dev, &dy_s, &b_s, &dx_s, M, N, K, BITS as u8, GS, has_g_idx, qw_off, qz_off, sc_off,
            gi_off,
        )
        .unwrap();
    unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
    let got_dx = dx_s.to_cpu_vec_f32().unwrap();

    for m in 0..M {
        for kk in 0..K {
            let want: f32 = (0..N)
                .map(|n| dy_data[m * N + n] * oracle_w[kk * N + n])
                .sum();
            let got = got_dx[m * K + kk];
            let tol = want.abs().max(1.0) * 2e-3;
            assert!(
                (got - want).abs() < tol,
                "backward mismatch at ({m},{kk}): got {got}, want {want}"
            );
        }
    }
}

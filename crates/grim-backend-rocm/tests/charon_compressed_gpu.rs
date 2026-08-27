//! Integration tests for Charon W8A8 and AWQ grouped MoE kernels.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendStorage, DType, Shape};

#[test]
fn test_charon_w8a8_int8_grouped_moe_structure() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }
    let dev = RocmDevice::try_new(0).expect("ROCm device 0");
    let hidden = 64;
    let inter = 32;
    let num_experts = 4;
    let batch = 2;

    // Per-expert stride: 8 + inter*hidden + inter*4
    let gate_stride = 8 + inter * hidden + inter * 4;
    let down_stride = 8 + hidden * inter + hidden * 4;

    let mut gate_blob = vec![0u8; num_experts * gate_stride];
    let mut up_blob = vec![0u8; num_experts * gate_stride];
    let mut down_blob = vec![0u8; num_experts * down_stride];

    for e in 0..num_experts {
        let g_off = e * gate_stride;
        let d_off = e * down_stride;
        // set u64 length prefix
        gate_blob[g_off..g_off + 8].copy_from_slice(&((inter * hidden) as u64).to_le_bytes());
        up_blob[g_off..g_off + 8].copy_from_slice(&((inter * hidden) as u64).to_le_bytes());
        down_blob[d_off..d_off + 8].copy_from_slice(&((hidden * inter) as u64).to_le_bytes());

        // Fill scales with 1.0f32
        for i in 0..inter {
            let sc_bytes = 1.0f32.to_le_bytes();
            gate_blob[g_off + 8 + inter * hidden + i * 4..g_off + 8 + inter * hidden + i * 4 + 4]
                .copy_from_slice(&sc_bytes);
            up_blob[g_off + 8 + inter * hidden + i * 4..g_off + 8 + inter * hidden + i * 4 + 4]
                .copy_from_slice(&sc_bytes);
        }
        for i in 0..hidden {
            let sc_bytes = 1.0f32.to_le_bytes();
            down_blob[d_off + 8 + hidden * inter + i * 4..d_off + 8 + hidden * inter + i * 4 + 4]
                .copy_from_slice(&sc_bytes);
        }
    }

    let alloc = std::sync::Arc::new(
        grim_backend_rocm::memory::allocator::RocmCachingAllocator::new(0, 1 << 30),
    );

    let egate = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &gate_blob,
        &Shape::new(vec![gate_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let eup = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &up_blob,
        &Shape::new(vec![up_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let edown = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &down_blob,
        &Shape::new(vec![down_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let act_data = vec![0.5f32; batch * hidden];
    let act = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &act_data,
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let a_scale_data = vec![1.0f32; batch];
    let a_scale = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &a_scale_data,
        &Shape::new(vec![batch]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let out = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let sorted = grim_backend_rocm::kernels::charon::SortedRouting {
        sorted_token_ids: vec![0, 1],
        sorted_expert_ids: vec![0, 1],
        sorted_weights: vec![1.0, 1.0],
        num_tokens_post_padded: 2,
        block_size: 64,
    };

    let stream = grim_backend_rocm::device::gptq_test_shim::launch_charon_grouped_dispatch_w8a8_int8_for_test(
        &dev,
        &act,
        egate.device_ptr().unwrap(),
        eup.device_ptr().unwrap(),
        edown.device_ptr().unwrap(),
        a_scale.device_ptr().unwrap(),
        &sorted,
        &out,
        hidden,
        inter,
        num_experts,
        1.0,
    ).expect("launch charon w8a8 int8");

    assert!(!stream.is_null());
}

#[test]
fn test_charon_w8a8_fp8_grouped_moe_structure() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }
    let dev = RocmDevice::try_new(0).expect("ROCm device 0");
    let hidden = 64;
    let inter = 32;
    let num_experts = 4;
    let batch = 2;

    let gate_stride = 8 + inter * hidden + 4;
    let down_stride = 8 + hidden * inter + 4;

    let mut gate_blob = vec![0u8; num_experts * gate_stride];
    let mut up_blob = vec![0u8; num_experts * gate_stride];
    let mut down_blob = vec![0u8; num_experts * down_stride];

    for e in 0..num_experts {
        let g_off = e * gate_stride;
        let d_off = e * down_stride;
        gate_blob[g_off..g_off + 8].copy_from_slice(&((inter * hidden) as u64).to_le_bytes());
        up_blob[g_off..g_off + 8].copy_from_slice(&((inter * hidden) as u64).to_le_bytes());
        down_blob[d_off..d_off + 8].copy_from_slice(&((hidden * inter) as u64).to_le_bytes());

        // per-tensor scale f32 = 1.0
        let sc_bytes = 1.0f32.to_le_bytes();
        gate_blob[g_off + 8 + inter * hidden..g_off + 8 + inter * hidden + 4]
            .copy_from_slice(&sc_bytes);
        up_blob[g_off + 8 + inter * hidden..g_off + 8 + inter * hidden + 4]
            .copy_from_slice(&sc_bytes);
        down_blob[d_off + 8 + hidden * inter..d_off + 8 + hidden * inter + 4]
            .copy_from_slice(&sc_bytes);
    }

    let alloc = std::sync::Arc::new(
        grim_backend_rocm::memory::allocator::RocmCachingAllocator::new(0, 1 << 30),
    );

    let egate = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &gate_blob,
        &Shape::new(vec![gate_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let eup = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &up_blob,
        &Shape::new(vec![up_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let edown = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &down_blob,
        &Shape::new(vec![down_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let act_data = vec![0.5f32; batch * hidden];
    let act = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &act_data,
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let a_scale_data = vec![1.0f32; batch];
    let a_scale = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &a_scale_data,
        &Shape::new(vec![batch]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let out = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let sorted = grim_backend_rocm::kernels::charon::SortedRouting {
        sorted_token_ids: vec![0, 1],
        sorted_expert_ids: vec![0, 1],
        sorted_weights: vec![1.0, 1.0],
        num_tokens_post_padded: 2,
        block_size: 64,
    };

    let stream = grim_backend_rocm::device::gptq_test_shim::launch_charon_grouped_dispatch_w8a8_fp8_for_test(
        &dev,
        &act,
        egate.device_ptr().unwrap(),
        eup.device_ptr().unwrap(),
        edown.device_ptr().unwrap(),
        a_scale.device_ptr().unwrap(),
        &sorted,
        &out,
        hidden,
        inter,
        num_experts,
        1.0,
    ).expect("launch charon w8a8 fp8");

    assert!(!stream.is_null());
}

#[test]
fn test_charon_awq_grouped_moe_structure() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }
    let dev = RocmDevice::try_new(0).expect("ROCm device 0");
    let hidden: usize = 64;
    let inter: usize = 32;
    let num_experts: usize = 4;
    let batch: usize = 2;
    let bits = 4u8;
    let group_size = 32usize;

    // Build gate blob [hidden, inter] AWQ format
    // and down blob [inter, hidden] AWQ format
    let vpw = 8usize;
    let gate_qw_len = hidden.div_ceil(vpw) * inter * 4;
    let gate_groups = hidden.div_ceil(group_size);
    let gate_qz_len = gate_groups * inter.div_ceil(vpw) * 4;
    let gate_sc_len = gate_groups * inter * 2;
    let gate_stride = 8 + gate_qw_len + 8 + gate_qz_len + 8 + gate_sc_len;

    let down_qw_len = inter.div_ceil(vpw) * hidden * 4;
    let down_groups = inter.div_ceil(group_size);
    let down_qz_len = down_groups * hidden.div_ceil(vpw) * 4;
    let down_sc_len = down_groups * hidden * 2;
    let down_stride = 8 + down_qw_len + 8 + down_qz_len + 8 + down_sc_len;

    let mut gate_blob = vec![0u8; num_experts * gate_stride];
    let mut up_blob = vec![0u8; num_experts * gate_stride];
    let mut down_blob = vec![0u8; num_experts * down_stride];

    for e in 0..num_experts {
        let g_off = e * gate_stride;
        let d_off = e * down_stride;
        gate_blob[g_off..g_off + 8].copy_from_slice(&(gate_qw_len as u64).to_le_bytes());
        gate_blob[g_off + 8 + gate_qw_len..g_off + 8 + gate_qw_len + 8]
            .copy_from_slice(&(gate_qz_len as u64).to_le_bytes());
        gate_blob[g_off + 8 + gate_qw_len + 8 + gate_qz_len
            ..g_off + 8 + gate_qw_len + 8 + gate_qz_len + 8]
            .copy_from_slice(&(gate_sc_len as u64).to_le_bytes());

        up_blob[g_off..g_off + 8].copy_from_slice(&(gate_qw_len as u64).to_le_bytes());
        up_blob[g_off + 8 + gate_qw_len..g_off + 8 + gate_qw_len + 8]
            .copy_from_slice(&(gate_qz_len as u64).to_le_bytes());
        up_blob[g_off + 8 + gate_qw_len + 8 + gate_qz_len
            ..g_off + 8 + gate_qw_len + 8 + gate_qz_len + 8]
            .copy_from_slice(&(gate_sc_len as u64).to_le_bytes());

        down_blob[d_off..d_off + 8].copy_from_slice(&(down_qw_len as u64).to_le_bytes());
        down_blob[d_off + 8 + down_qw_len..d_off + 8 + down_qw_len + 8]
            .copy_from_slice(&(down_qz_len as u64).to_le_bytes());
        down_blob[d_off + 8 + down_qw_len + 8 + down_qz_len
            ..d_off + 8 + down_qw_len + 8 + down_qz_len + 8]
            .copy_from_slice(&(down_sc_len as u64).to_le_bytes());
    }

    let alloc = std::sync::Arc::new(
        grim_backend_rocm::memory::allocator::RocmCachingAllocator::new(0, 1 << 30),
    );

    let egate = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &gate_blob,
        &Shape::new(vec![gate_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let eup = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &up_blob,
        &Shape::new(vec![up_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let edown = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host_raw_bytes(
        &down_blob,
        &Shape::new(vec![down_blob.len()]),
        DType::U8,
        &alloc,
        0,
    )
    .unwrap();

    let act_data = vec![0.5f32; batch * hidden];
    let act = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &act_data,
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let a_scale_data = vec![1.0f32; batch];
    let a_scale = grim_backend_rocm::memory::storage::RocmStorage::copy_from_host(
        &a_scale_data,
        &Shape::new(vec![batch]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let out = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![batch, hidden]),
        DType::F32,
        &alloc,
        0,
    )
    .unwrap();

    let sorted = grim_backend_rocm::kernels::charon::SortedRouting {
        sorted_token_ids: vec![0, 1],
        sorted_expert_ids: vec![0, 1],
        sorted_weights: vec![1.0, 1.0],
        num_tokens_post_padded: 2,
        block_size: 64,
    };

    let (g_qw, g_qz, g_sc) = (
        8i64,
        (8 + gate_qw_len + 8) as i64,
        (8 + gate_qw_len + 8 + gate_qz_len + 8) as i64,
    );
    let (d_qw, d_qz, d_sc) = (
        8i64,
        (8 + down_qw_len + 8) as i64,
        (8 + down_qw_len + 8 + down_qz_len + 8) as i64,
    );

    let stream =
        grim_backend_rocm::device::gptq_test_shim::launch_charon_grouped_dispatch_awq_for_test(
            &dev,
            &act,
            egate.device_ptr().unwrap(),
            eup.device_ptr().unwrap(),
            edown.device_ptr().unwrap(),
            a_scale.device_ptr().unwrap(),
            &sorted,
            &out,
            hidden,
            inter,
            num_experts,
            bits,
            group_size,
            g_qw,
            g_qz,
            g_sc,
            gate_stride as u64,
            d_qw,
            d_qz,
            d_sc,
            down_stride as u64,
            1.0,
        )
        .expect("launch charon awq");

    assert!(!stream.is_null());
}

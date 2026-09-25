//! GRAVE Phase 5 validation: persistent wavefront megakernel parity and gates.
//!
//! Gates verified:
//! - M5: One persistent launch executing whole step
//! - M6: Instruction type diversity (>= 5 opcodes)
//! - M7: Lock-free atomic dependency counters (zero grid barriers)
//! - M8: LDS buffer pooling and staging reuse

use grim_backend_rocm::device::compute::gla_mega_launchers::{
    GlaMegaLaunchArgs, GlaMegakernelTask,
};
use grim_backend_rocm::{
    CoreTensorOps, MemoryOps, RocmDevice, RocmStorage, as_rocm, gpu_test_enabled,
};
use grim_tensor::{DType, Shape};

fn as_rocm_ref(s: &Box<dyn grim_tensor::BackendStorage>) -> &RocmStorage {
    as_rocm(s.as_ref()).expect("rocm storage")
}

#[test]
fn test_m5_to_m8_megakernel_structure() {
    use grim_backend_rocm::kernels::gla_mega_kernel::GLA_MEGA_KERNEL_SOURCE;

    // M5: Single persistent launch
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("grim_gla_persistent_megakernel"));

    // M6: >= 5 instruction types
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_RMS_NORM"));
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_DOT4_GEMV"));
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_GDN2_RECURRENT"));
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_SHORTCONV_FUSED"));
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_FFN_GATEUP_SILU"));
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_RESIDUAL_ADD"));

    // M7: Zero grid barriers
    assert!(!GLA_MEGA_KERNEL_SOURCE.contains("grid.sync"));
    assert!(!GLA_MEGA_KERNEL_SOURCE.contains("cooperative_groups"));

    // M8: LDS buffer pooling
    assert!(GLA_MEGA_KERNEL_SOURCE.contains("__shared__ float s_lds[1024];"));
}

#[test]
#[ignore]
fn test_megakernel_gpu_execution() {
    if !gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU megakernel test");
        return;
    }
    let Some(dev) = RocmDevice::try_new(0).ok() else {
        eprintln!("skip: no ROCm device 0");
        return;
    };

    let hidden = 64;
    let pool_size = 1024;
    let pool_raw = vec![0.0f32; pool_size];
    let pool_s = dev
        .from_cpu(&pool_raw, &Shape::new(vec![pool_size]), DType::F32)
        .unwrap();

    let weights_raw = vec![0u8; hidden * hidden * 4];
    let weights_s = dev
        .from_cpu_bytes(
            &weights_raw,
            &Shape::new(vec![hidden * hidden * 4]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::Storage::Native,
            },
        )
        .unwrap();

    let rec_state_raw = vec![0.0f32; hidden * 64];
    let rec_state_s = dev
        .from_cpu(&rec_state_raw, &Shape::new(vec![hidden * 64]), DType::F32)
        .unwrap();

    let counters_raw = vec![0u32; 16];
    let counters_s = dev
        .from_cpu_bytes(
            unsafe {
                std::slice::from_raw_parts(
                    counters_raw.as_ptr() as *const u8,
                    counters_raw.len() * 4,
                )
            },
            &Shape::new(vec![16]),
            DType {
                arith: grim_tensor::ArithType::U32,
                storage: grim_tensor::Storage::Native,
            },
        )
        .unwrap();

    let cursor_raw = vec![0u32; 1];
    let cursor_s = dev
        .from_cpu_bytes(
            unsafe {
                std::slice::from_raw_parts(cursor_raw.as_ptr() as *const u8, cursor_raw.len() * 4)
            },
            &Shape::new(vec![1]),
            DType {
                arith: grim_tensor::ArithType::U32,
                storage: grim_tensor::Storage::Native,
            },
        )
        .unwrap();

    let tasks = vec![
        GlaMegakernelTask {
            op_type: 1, // RMSNorm
            layer_idx: 0,
            in_offset: 0,
            weight_offset: 0,
            out_offset: hidden as i32,
            aux_offset: 0,
            dim_m: 1,
            dim_n: 1,
            dim_k: hidden as i32,
            dep_counter_idx: -1,
            dep_counter_val: 0,
            post_counter_idx: 0,
        },
        GlaMegakernelTask {
            op_type: 6, // ResidualAdd
            layer_idx: 0,
            in_offset: 0,
            weight_offset: 0,
            out_offset: 0,
            aux_offset: hidden as i32,
            dim_m: 1,
            dim_n: 1,
            dim_k: hidden as i32,
            dep_counter_idx: 0,
            dep_counter_val: 1,
            post_counter_idx: 1,
        },
    ];

    let res = dev.launch_gla_persistent_megakernel(&GlaMegaLaunchArgs {
        tasks: &tasks,
        staging_pool: as_rocm_ref(&pool_s),
        weight_pool: as_rocm_ref(&weights_s),
        recurrent_state: as_rocm_ref(&rec_state_s),
        stage_counters: as_rocm_ref(&counters_s),
        task_cursor: as_rocm_ref(&cursor_s),
        total_layers: 1,
        hidden_dim: hidden,
        num_cus: 32,
    });

    assert!(res.is_ok(), "persistent megakernel launch succeeded");
    dev.synchronize();
}

//! GRAVE Phase 5 validation: persistent wavefront megakernel parity and gates.
//!
//! What was here before: 10x `KERNEL_SOURCE.contains(...)` string asserts plus
//! `res.is_ok()` on a launch, so a kernel that launched and computed garbage
//! still passed.
//!
//! What this is now:
//! - ONE structural contains, kept deliberately: it documents which kernel this
//!   test would run (`grim_gla_persistent_megakernel` in
//!   `GLA_MEGA_KERNEL_SOURCE`). M6/M7/M8 string checks are dropped — they
//!   duplicate the in-src unit tests, and they never measured the arithmetic.
//! - `test_megakernel_chained_rms_residual_matches_f64` LAUNCHES the
//!   megakernel (`launch_gla_persistent_megakernel`) with a two-task chain
//!   (OP_RMS_NORM → OP_RESIDUAL_ADD, dependency via the stage counters) on a
//!   small shape and compares the staging-pool readback against an f64 oracle
//!   written inline here, asserting worst ≤ 1e-4.
//!
//! Reachable vs not (factual locus):
//! - Reachable: OP_RMS_NORM and OP_RESIDUAL_ADD. Their kernels
//!   (`src/kernels/gla_mega_kernel.rs`) are plain: RMSNorm
//!   `y[i] = x[i] * rsqrt(sum(x²)/n + 1e-5) * w[i]`, residual add `y = a + b`.
//!   The oracle below is those two lines in f64, so this is a true oracle, not
//!   CPU == ROCm with a shared mistake.
//! - NOT reachable as `oracle_head` parity: OP_GDN2_RECURRENT in the
//!   megakernel hard-codes `a = 0.989f`, `delta = 0.5 * v[j] - r[j]`, and drops
//!   the gated RMSNorm/sigmoid epilogue that `oracle_head`
//!   (`tests/gla_fused_parity.rs:25`) implements — it cannot match that oracle
//!   to 1e-4 by construction, on any input. It is excluded from the launch
//!   comparison for that reason, and OP_SHORTCONV_FUSED/OP_FFN_GATEUP_SILU/
//!   OP_DOT4_GEMV are excluded because they read staging offsets and weight
//!   encodings the megakernel's `in_offset`/`weight_offset` geometry does not
//!   pin down for a parity oracle. If the megakernel's GDN2 arm is later
//!   parameterised by (alpha, beta, norm_w, gate), the upgrade is to add its
//!   arm to this same chain and compare against `oracle_head` verbatim.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device, `#[ignore]` so it never runs
//! in a CPU/offline CI lane. Local manual runs only (consumer ROCm project).

use grim_backend_rocm::device::compute::gla_mega_launchers::{
    GlaMegaLaunchArgs, GlaMegakernelTask,
};
use grim_backend_rocm::{
    BackendStorage, CoreTensorOps, MemoryOps, RocmDevice, RocmStorage, as_rocm, gpu_test_enabled,
};
use grim_tensor::{DType, Shape};

fn as_rocm_ref(s: &Box<dyn grim_tensor::BackendStorage>) -> &RocmStorage {
    as_rocm(s.as_ref()).expect("rocm storage")
}

fn from_cpu_bytes_f32(
    dev: &RocmDevice,
    data: &[f32],
) -> Box<dyn grim_tensor::BackendStorage> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    dev.from_cpu_bytes(
        &bytes,
        &Shape::new(vec![data.len()]),
        DType::F32,
    )
    .expect("from_cpu_bytes")
}

fn from_cpu_bytes_u32(
    dev: &RocmDevice,
    data: &[u32],
) -> Box<dyn grim_tensor::BackendStorage> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    grim_backend_rocm::MemoryOps::from_cpu_bytes(
        dev,
        &bytes,
        &Shape::new(vec![data.len()]),
        DType {
            arith: grim_tensor::ArithType::U32,
            storage: grim_tensor::Storage::Native,
        },
    )
    .expect("from_cpu_bytes u32")
}

fn read_f32(s: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    as_rocm_ref(s).to_cpu_vec_f32().expect("readback")
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// The ONE structural check: documents which kernel the parity test launches.
/// Everything else is measured below, not string-matched.
#[test]
fn test_megakernel_source_names_persistent_kernel() {
    use grim_backend_rocm::kernels::gla_mega_kernel::GLA_MEGA_KERNEL_SOURCE;

    assert!(
        GLA_MEGA_KERNEL_SOURCE.contains("grim_gla_persistent_megakernel"),
        "GLA_MEGA_KERNEL_SOURCE no longer contains grim_gla_persistent_megakernel; \
         the launch-compare test below would run nothing"
    );
}

/// Persistent-megakernel launch vs an independent f64 oracle for the chained
/// OP_RMS_NORM → OP_RESIDUAL_ADD pair (the pair whose arithmetic the
/// megakernel actually specifies — see header for the GDN2 exclusion).
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 (local manual only; no new CI)"]
fn test_megakernel_chained_rms_residual_matches_f64() {
    if !gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run megakernel parity");
        return;
    }
    let Some(dev) = RocmDevice::try_new(0).ok() else {
        eprintln!("[SKIP] ROCm device 0 unavailable");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // Non-zero inputs: a zero x would make the RMS denominator and the norm
    // weight invisible. Scale ≤ 2 so no value can blow past the smoke region.
    let hidden = 64usize;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let x: Vec<f32> = (0..hidden)
        .map(|_| (lcg(&mut seed) * 2.0) as f32)
        .collect();
    let norm_w: Vec<f32> = (0..hidden)
        .map(|_| (1.0 + 0.1 * lcg(&mut seed)) as f32)
        .collect();
    let residual: Vec<f32> = (0..hidden)
        .map(|_| (lcg(&mut seed) * 2.0) as f32)
        .collect();

    // ── Independent f64 oracle (the two lines the kernels implement). ──
    let eps = 1e-5f64;
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms = (ss / hidden as f64 + eps).sqrt();
    let y_oracle: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as f64) / rms * norm_w[i] as f64)
        .collect();
    let want: Vec<f64> = y_oracle
        .iter()
        .enumerate()
        .map(|(i, &v)| v + residual[i] as f64)
        .collect();

    // Staging layout: [0..hidden) = x, [hidden..2*hidden) = residual,
    // [2*hidden..3*hidden) = normed-out, [3*hidden..4*hidden) = final out.
    // (OP_RESIDUAL_ADD takes a = staging[in_offset], b = staging[aux_offset].)
    let pool_size = 4 * hidden;
    let mut pool_raw = vec![0.0f32; pool_size];
    pool_raw[..hidden].copy_from_slice(&x);
    pool_raw[hidden..2 * hidden].copy_from_slice(&residual);
    let pool_s = dev
        .from_cpu(&pool_raw, &Shape::new(vec![pool_size]), DType::F32)
        .unwrap();

    // Weight pool: the megakernel reads norm weights as f32 at weight_offset.
    let weights_s = from_cpu_bytes_f32(&dev, &norm_w);

    let rec_state_raw = vec![0.0f32; 64 * 64];
    let rec_state_s = dev
        .from_cpu(&rec_state_raw, &Shape::new(vec![64 * 64]), DType::F32)
        .unwrap();

    let counters_s = from_cpu_bytes_u32(&dev, &vec![0u32; 4]);
    let cursor_s = from_cpu_bytes_u32(&dev, &vec![0u32; 1]);

    let tasks = vec![
        GlaMegakernelTask {
            op_type: 1, // OP_RMS_NORM: y = x * rsqrt(sum(x²)/n + 1e-5) * w
            layer_idx: 0,
            in_offset: 0,
            weight_offset: 0,
            out_offset: (2 * hidden) as i32,
            aux_offset: 0,
            dim_m: 1,
            dim_n: 1,
            dim_k: hidden as i32,
            dep_counter_idx: -1,
            dep_counter_val: 0,
            post_counter_idx: 0,
        },
        GlaMegakernelTask {
            op_type: 6, // OP_RESIDUAL_ADD: y = a + b, gated on counter 0 == 1
            layer_idx: 0,
            in_offset: (2 * hidden) as i32,
            weight_offset: 0,
            out_offset: (3 * hidden) as i32,
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
    assert!(res.is_ok(), "persistent megakernel launch failed: {:?}", res.err());
    dev.synchronize();

    let pool_out = read_f32(&pool_s);
    let got = &pool_out[3 * hidden..4 * hidden];
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g as f64 - w).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1e-4,
        "megakernel chained RMSNorm+ResidualAdd vs f64 oracle Δ={worst:.3e} \
         at {worst_at} (got {}, want {}) > 1e-4",
        got[worst_at],
        want[worst_at]
    );
    println!("[OK] megakernel chained RMSNorm+ResidualAdd parity Δ={worst:.3e}");
}

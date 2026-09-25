//! GRAVE Phase 4 gate G4: fused GDN-2 kernel vs f64 oracle, batch 1 and 4.
//!
//! Run: `GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test gla_fused_parity`
//! Without `GRIM_GPU_TEST=1` the test skips (same convention as
//! `cross_attention_gpu`). Tolerance: max|Δ| ≤ 1e-4 (f32 kernel vs f64 oracle).

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::compute::gla_launchers::GlaLaunchArgs;
use grim_tensor::{CoreTensorOps, DType, Shape};

const DK: usize = 64;
const DV: usize = 64;
const HEADS: usize = 2;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// Independent oracle: serial GDN-2 step + gated RMSNorm, f64.
/// Written from the math (not copied from `gla.rs`) so agreement is evidence.
#[allow(clippy::too_many_arguments)]
fn oracle_head(
    state: &mut [Vec<f64>],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha: &[f64],
    b: &[f64],
    w: &[f64],
    norm_w: &[f64],
    gate: f64,
    eps: f64,
) -> Vec<f64> {
    for i in 0..DK {
        for j in 0..DV {
            state[i][j] *= alpha[i];
        }
    }
    let mut r = vec![0.0f64; DV];
    for i in 0..DK {
        let e = b[i] * k[i];
        for j in 0..DV {
            r[j] += e * state[i][j];
        }
    }
    for i in 0..DK {
        for j in 0..DV {
            state[i][j] += k[i] * (w[j] * v[j] - r[j]);
        }
    }
    let mut o = vec![0.0f64; DV];
    for j in 0..DV {
        for i in 0..DK {
            o[j] += q[i] * state[i][j];
        }
    }
    let rms = (o.iter().map(|x| x * x).sum::<f64>() / DV as f64 + eps).sqrt();
    let g = 1.0 / (1.0 + (-gate).exp());
    o.iter()
        .enumerate()
        .map(|(j, &x)| x / rms * norm_w[j] * g)
        .collect()
}

fn run_batch(batch: usize) {
    let mut seed = 0x9E3779B97F4A7C15u64 ^ ((batch as u64) << 32);
    let n = batch * HEADS;
    let mut q = vec![0.0f32; n * DK];
    let mut k = vec![0.0f32; n * DK];
    let mut v = vec![0.0f32; n * DV];
    let mut alpha = vec![0.0f32; n * DK];
    let mut b = vec![0.0f32; n * DK];
    let mut w = vec![0.0f32; n * DV];
    let mut norm_w = vec![0.0f32; n * DV];
    let mut gate = vec![0.0f32; n];
    for x in q.iter_mut() {
        *x = lcg(&mut seed) as f32;
    }
    for x in k.iter_mut() {
        *x = lcg(&mut seed) as f32;
    }
    for x in v.iter_mut() {
        *x = lcg(&mut seed) as f32;
    }
    for x in alpha.iter_mut() {
        *x = (0.9 + 0.099 * lcg(&mut seed)) as f32;
    }
    for x in b.iter_mut() {
        *x = (1.0 / (1.0 + (-4.0 * lcg(&mut seed)))) as f32;
    }
    for x in w.iter_mut() {
        *x = (1.0 / (1.0 + (-4.0 * lcg(&mut seed)))) as f32;
    }
    for x in norm_w.iter_mut() {
        *x = (1.0 + 0.1 * lcg(&mut seed)) as f32;
    }
    for x in gate.iter_mut() {
        *x = (lcg(&mut seed) * 2.0) as f32;
    }
    let state = vec![0.0f32; n * DK * DV];

    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    let up = |d: &[f32], dims: &[usize]| {
        dev.from_cpu(d, &Shape::from_slice(dims), DType::F32)
            .expect("from_cpu")
    };
    let q_s = up(&q, &[n, DK]);
    let k_s = up(&k, &[n, DK]);
    let v_s = up(&v, &[n, DV]);
    let a_s = up(&alpha, &[n, DK]);
    let b_s = up(&b, &[n, DK]);
    let w_s = up(&w, &[n, DV]);
    let nw_s = up(&norm_w, &[n, DV]);
    let g_s = up(&gate, &[n]);
    let st_s = up(&state, &[n, DK, DV]);
    let out_s = up(&vec![0.0f32; n * DV], &[n, DV]);

    dev.launch_gla_state_update_output_into(&GlaLaunchArgs {
        q: as_rocm_ref(&q_s),
        k: as_rocm_ref(&k_s),
        v: as_rocm_ref(&v_s),
        alpha: as_rocm_ref(&a_s),
        b_gate: as_rocm_ref(&b_s),
        w_gate: as_rocm_ref(&w_s),
        norm_w: as_rocm_ref(&nw_s),
        out_gate: as_rocm_ref(&g_s),
        state: as_rocm_ref(&st_s),
        out: as_rocm_ref(&out_s),
        heads: HEADS,
        batch,
        dk: DK,
        dv: DV,
        eps: 1e-5,
    })
    .expect("fused GDN-2 launch");
    let gpu = out_s.to_cpu_vec_f32().expect("readback");

    // Oracle per (batch, head) slot with its own zero state.
    let mut worst = 0.0f64;
    for slot in 0..n {
        let mut st = vec![vec![0.0f64; DV]; DK];
        let f = |src: &[f32], d: usize| -> Vec<f64> {
            src[slot * d..(slot + 1) * d]
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        let o = oracle_head(
            &mut st,
            &f(&q, DK),
            &f(&k, DK),
            &f(&v, DV),
            &f(&alpha, DK),
            &f(&b, DK),
            &f(&w, DV),
            &f(&norm_w, DV),
            gate[slot] as f64,
            1e-5,
        );
        for j in 0..DV {
            worst = worst.max((gpu[slot * DV + j] as f64 - o[j]).abs());
        }
    }
    assert!(
        worst <= 1e-4,
        "batch={batch}: fused kernel vs oracle Δ={worst:.3e} > 1e-4"
    );
    println!("[OK] fused GDN-2 parity batch={batch} Δ={worst:.3e}");
}

fn as_rocm_ref(s: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
    grim_backend_rocm::as_rocm(s.as_ref()).expect("rocm storage")
}

#[test]
#[ignore]
fn fused_gdn2_matches_oracle_batch1() {
    if !grim_backend_rocm::gpu_test_enabled() {
        println!("[INFO] Skipped (requires GRIM_GPU_TEST=1)");
        return;
    }
    run_batch(1);
}

#[test]
#[ignore]
fn fused_gdn2_matches_oracle_batch4() {
    if !grim_backend_rocm::gpu_test_enabled() {
        println!("[INFO] Skipped (requires GRIM_GPU_TEST=1)");
        return;
    }
    run_batch(4);
}

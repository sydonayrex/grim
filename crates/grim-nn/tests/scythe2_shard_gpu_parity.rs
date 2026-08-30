//! WI-SB5 gate: two-rank ROCm shard parity against monolithic GEMM.
//!
//! `Scythe2Linear::forward_placed` with a `[0.6, 0.4]` column split and a
//! `[0.5, 0.5]` row split across Rocm(0)/Rocm(1) must match a single-rank
//! monolithic matmul within fp tolerance (max-abs-diff bound — NOT byte
//! parity: per-shard accumulation order differs). Also exercises the
//! WI-SB5 residency cache: the transposed shard operands are built once and
//! reused across forwards; a second forward must agree with the first.
//!
//! Device-gated: `GRIM_GPU_TEST=1`, requires ≥2 visible ROCm devices.
//!
//! Verified on: gfx1201 / gfx1200 (Dual-GPU) and gfx1036 — 2026-08-29.

use grim_nn::scythe2::Scythe2Linear;
use grim_tensor::backend::ScythePlacement;
use grim_tensor::{CoreTensorOps, DType, Device, Tensor};
use std::sync::Arc;

fn gpu_ready() -> bool {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return false;
    }
    match grim_backend_rocm::peer_access::enumerate_devices() {
        Ok(n) if n >= 2 => true,
        other => {
            eprintln!("[skipped: need ≥2 ROCm devices, got {other:?}]");
            false
        }
    }
}

fn make_rocm_tensor(data: &[f32], shape: &[usize], ordinal: usize) -> Tensor {
    let dev = grim_backend_rocm::RocmDevice::try_new(ordinal).expect("try_new");
    let storage = dev
        .from_cpu(data, &grim_tensor::Shape::from_slice(shape), DType::F32)
        .expect("upload");
    Tensor::new(
        Arc::from(storage),
        grim_tensor::Shape::from_slice(shape),
        DType::F32,
        grim_tensor::dtype::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
}

const M: usize = 8;
const K: usize = 128;
const OUT: usize = 256;

fn build_layer() -> Scythe2Linear {
    let w_data: Vec<f32> = (0..OUT * K)
        .map(|i| ((i % 31) as f32) * 0.02 - 0.3)
        .collect();
    // Full weight lives on rank 0; shards are sliced from it host-side and
    // pinned to their rank devices by the residency cache.
    let full_weight = make_rocm_tensor(&w_data, &[OUT, K], 0);
    Scythe2Linear {
        full_weight,
        bias: None,
        layer_id: 77,
        device: Device::Rocm(0),
    }
}

fn make_x() -> Tensor {
    let x_data: Vec<f32> = (0..M * K).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
    make_rocm_tensor(&x_data, &[M, K], 0)
}

fn two_rank_placement(row: bool) -> ScythePlacement {
    let part = if row { vec![0.5, 0.5] } else { vec![0.6, 0.4] };
    ScythePlacement {
        ranks: vec![0, 1],
        partition: part,
        routes: vec![
            grim_tensor::ScytheLink::PeerDirect,
            grim_tensor::ScytheLink::Host,
            grim_tensor::ScytheLink::Host,
            grim_tensor::ScytheLink::PeerDirect,
        ],
    }
}

/// Monolithic reference: single matmul of x against the fully transposed
/// weight on one device.
fn reference(x: &Tensor, w_flat_host: &[f32], m: usize, k: usize, n_out: usize) -> Vec<f32> {
    let mut w_t = vec![0.0f32; k * n_out];
    for ni in 0..n_out {
        for ki in 0..k {
            w_t[ki * n_out + ni] = w_flat_host[ni * k + ki];
        }
    }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("ref dev");
    let a = dev
        .from_cpu(
            &x.storage().to_cpu_vec_f32().expect("x dtoh"),
            &grim_tensor::Shape::from_slice(&[m, k]),
            DType::F32,
        )
        .expect("a");
    let b = dev
        .from_cpu(
            &w_t,
            &grim_tensor::Shape::from_slice(&[k, n_out]),
            DType::F32,
        )
        .expect("b");
    let (out, h) = dev
        .matmul(
            a.as_ref(),
            b.as_ref(),
            &grim_tensor::Shape::from_slice(&[m, n_out]),
        )
        .expect("ref matmul");
    h.synchronize().expect("sync");
    out.to_cpu_vec_f32().expect("ref readback")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn sb5_two_rank_col_and_row_parity_on_rocm() {
    if !gpu_ready() {
        return;
    }
    let w_host: Vec<f32> = (0..OUT * K)
        .map(|i| ((i % 31) as f32) * 0.02 - 0.3)
        .collect();
    let layer = build_layer();
    let x = make_x();

    // Column-parallel [0.6, 0.4].
    let y_col = layer
        .forward_placed(&x, &two_rank_placement(false), false)
        .unwrap();
    let yv = y_col.storage().to_cpu_vec_f32().unwrap();
    let refr = reference(&x, &w_host, M, K, OUT);
    let d_col = max_abs_diff(&yv, &refr);
    println!("[sb5] col-parallel max_abs_diff={d_col:.3e}");
    assert!(d_col < 1e-3, "col-parallel parity broken: {d_col:.3e}");

    // Row-parallel [0.5, 0.5].
    let y_row = layer
        .forward_placed(&x, &two_rank_placement(true), true)
        .unwrap();
    let yvr = y_row.storage().to_cpu_vec_f32().unwrap();
    let d_row = max_abs_diff(&yvr, &refr);
    println!("[sb5] row-parallel max_abs_diff={d_row:.3e}");
    assert!(d_row < 1e-3, "row-parallel parity broken: {d_row:.3e}");

    // Residency-cache reuse: a second forward must reproduce the first.
    let y_col2 = layer
        .forward_placed(&x, &two_rank_placement(false), false)
        .unwrap();
    let d_re = max_abs_diff(&y_col2.storage().to_cpu_vec_f32().unwrap(), &refr);
    println!("[sb5] cached-reuse max_abs_diff={d_re:.3e}");
    assert!(d_re < 1e-3, "cached second pass diverged: {d_re:.3e}");
}

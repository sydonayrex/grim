//! GPU parity tests for the Xing4.0 MHC primitives.
//!
//! Covers `grim_row_scale` (per-token gate scale) and `grim_mhc_gates` (the
//! fused sigmoid + Sinkhorn gate math). Both are new device primitives used by
//! the Xing4.0 hyper-connection, so they are checked against host references
//! rather than trusted.

use grim_backend_rocm::device::compute::layer_elementwise::MhcGateTensors;
use grim_backend_rocm::CoreTensorOps;
use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{ElementwiseOps, Shape};

/// Deterministic pseudo-random projection in [-1, 1] without pulling a dep.
fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// Host reference for Xing4.0's `gates_from_projection`.
fn ref_gates(
    proj: &[f32],
    base: &[f32],
    scale: &[f32],
    seq: usize,
    hc: usize,
    iters: usize,
    eps: f32,
    cmin: f32,
    cmax: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mix = (2 + hc) * hc;
    // stream-major / token-last, matching grim_mhc_gates
    let mut pre = vec![0.0f32; hc * seq];
    let mut post = vec![0.0f32; hc * seq];
    let mut comb = vec![0.0f32; hc * hc * seq];
    for s in 0..seq {
        let row = &proj[s * mix..(s + 1) * mix];
        for h in 0..hc {
            pre[h * seq + s] = 1.0 / (1.0 + (-(row[h] * scale[0] + base[h])).exp());
            post[h * seq + s] =
                2.0 / (1.0 + (-(row[hc + h] * scale[1] + base[hc + h])).exp());
        }
        let off = 2 * hc;
        let mut m = f32::MIN;
        for k in 0..hc * hc {
            let mut v = row[off + k] * scale[2] + base[off + k];
            v = v.clamp(cmin, cmax);
            comb[k * seq + s] = v;
            if v > m {
                m = v;
            }
        }
        for k in 0..hc * hc {
            comb[k * seq + s] = (comb[k * seq + s] - m).exp();
        }
        for _ in 0..iters {
            for r in 0..hc {
                let mut sum = 0.0f32;
                for i in 0..hc {
                    sum += comb[(r * hc + i) * seq + s];
                }
                let d = sum + eps;
                for i in 0..hc {
                    comb[(r * hc + i) * seq + s] /= d;
                }
            }
            for i in 0..hc {
                let mut sum = 0.0f32;
                for r in 0..hc {
                    sum += comb[(r * hc + i) * seq + s];
                }
                let d = sum + eps;
                for r in 0..hc {
                    comb[(r * hc + i) * seq + s] /= d;
                }
            }
        }
    }
    (pre, post, comb)
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
        }
        assert!(d < tol, "{what}[{i}]: {x} vs {y} (delta {d} >= {tol})");
    }
    println!("{what}: max abs delta = {worst:e}");
}

#[test]
#[ignore = "requires a visible ROCm device"]
fn mhc_gates_gpu_matches_host() {
    let dev = RocmDevice::new(0);
    let (seq, hc, iters) = (37usize, 4usize, 20usize);
    let mix = (2 + hc) * hc;
    let (eps, cmin, cmax) = (1e-6f32, -30.0f32, 30.0f32);

    // Spread the projection across the clamp range so the clamp branch and the
    // row-max stabilization are both actually exercised.
    let mut proj = pseudo_random(seq * mix, 7);
    for (i, v) in proj.iter_mut().enumerate() {
        *v = *v * 40.0 + (i % 5) as f32 * 7.0;
    }
    let base = pseudo_random(mix, 11);
    let scale = vec![1.0f32, 1.0f32, 2.0f32];

    let proj_s = dev.from_cpu(&proj, &Shape::new(vec![seq, mix]), DType::F32).unwrap();
    let base_s = dev.from_cpu(&base, &Shape::new(vec![mix]), DType::F32).unwrap();
    let scale_s = dev
        .from_cpu(&scale, &Shape::new(vec![3]), DType::F32)
        .unwrap();

    let MhcGateTensors { pre, post, comb } = dev
        .mhc_gates_into(
            proj_s.as_ref(),
            base_s.as_ref(),
            scale_s.as_ref(),
            seq,
            hc,
            iters,
            eps,
            cmin,
            cmax,
        )
        .unwrap();

    let (rpre, rpost, rcomb) =
        ref_gates(&proj, &base, &scale, seq, hc, iters, eps, cmin, cmax);

    close(
        &pre.to_cpu_vec_f32().unwrap(),
        &rpre,
        1e-5,
        "mhc pre",
    );
    close(
        &post.to_cpu_vec_f32().unwrap(),
        &rpost,
        1e-5,
        "mhc post",
    );
    close(
        &comb.to_cpu_vec_f32().unwrap(),
        &rcomb,
        1e-5,
        "mhc comb",
    );
}

#[test]
#[ignore = "requires a visible ROCm device"]
fn row_scale_gpu_matches_host() {
    let dev = RocmDevice::new(0);
    let (rows, cols) = (13usize, 7usize);
    let x = pseudo_random(rows * cols, 3);
    let s = pseudo_random(rows, 5);

    let x_s = dev
        .from_cpu(&x, &Shape::new(vec![rows, cols]), DType::F32)
        .unwrap();
    let s_s = dev.from_cpu(&s, &Shape::new(vec![rows]), DType::F32).unwrap();

    let (out, handle) = dev
        .row_scale(
            x_s.as_ref(),
            s_s.as_ref(),
            rows,
            cols,
            &Shape::new(vec![rows, cols]),
        )
        .unwrap();
    handle.synchronize().unwrap();

    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            expected[r * cols + c] = x[r * cols + c] * s[r];
        }
    }
    close(&out.to_cpu_vec_f32().unwrap(), &expected, 1e-6, "row_scale");
}

#[test]
#[ignore = "requires a visible ROCm device"]
fn row_copy_primitives_gpu_match_host() {
    let dev = RocmDevice::new(0);
    let (total_rows, cols) = (12usize, 5usize);
    let x = pseudo_random(total_rows * cols, 13);

    let x_s = dev
        .from_cpu(&x, &Shape::new(vec![total_rows, cols]), DType::F32)
        .unwrap();

    // narrow_rows reads rows [3, 9) contiguously.
    let (slice, h1) = dev
        .narrow_rows(x_s.as_ref(), 3, 6, cols, &Shape::new(vec![6, cols]))
        .unwrap();
    h1.synchronize().unwrap();
    let expected_slice: Vec<f32> = x[3 * cols..9 * cols].to_vec();
    close(
        &slice.to_cpu_vec_f32().unwrap(),
        &expected_slice,
        1e-7,
        "narrow_rows",
    );

    // write_rows stores a [2, cols] block into rows [8, 10) of a fresh buffer.
    let patch = pseudo_random(2 * cols, 17);
    let patch_s = dev
        .from_cpu(&patch, &Shape::new(vec![2, cols]), DType::F32)
        .unwrap();
    let mut expected = x.clone();
    expected[8 * cols..10 * cols].copy_from_slice(&patch);

    let mut dst = dev
        .from_cpu(&x, &Shape::new(vec![total_rows, cols]), DType::F32)
        .unwrap();
    let h2 = dev
        .write_rows(dst.as_mut(), 8, patch_s.as_ref(), 2, cols)
        .unwrap();
    h2.synchronize().unwrap();
    close(&dst.to_cpu_vec_f32().unwrap(), &expected, 1e-7, "write_rows");
}

#[test]
#[ignore = "requires a visible ROCm device"]
fn col_copy_primitives_gpu_match_host() {
    let dev = RocmDevice::new(0);
    let (rows, total_cols) = (9usize, 6usize);
    let cols = 3usize;
    let x = pseudo_random(rows * total_cols, 23);
    let x_s = dev
        .from_cpu(&x, &Shape::new(vec![rows, total_cols]), DType::F32)
        .unwrap();

    // narrow_cols: out[r, c] = x[r, 6 + 2 + c]
    let (slice, h1) = dev
        .narrow_cols(
            x_s.as_ref(),
            total_cols,
            2,
            rows,
            cols,
            &Shape::new(vec![rows, cols]),
        )
        .unwrap();
    h1.synchronize().unwrap();
    let mut want = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            want[r * cols + c] = x[r * total_cols + 2 + c];
        }
    }
    close(&slice.to_cpu_vec_f32().unwrap(), &want, 1e-7, "narrow_cols");

    // write_cols: patch into columns [3, 6) of a fresh [rows, total_cols] buffer.
    let patch = pseudo_random(rows * cols, 29);
    let patch_s = dev
        .from_cpu(&patch, &Shape::new(vec![rows, cols]), DType::F32)
        .unwrap();
    let mut expected = x.clone();
    for r in 0..rows {
        for c in 0..cols {
            expected[r * total_cols + 3 + c] = patch[r * cols + c];
        }
    }
    let mut dst = dev
        .from_cpu(&x, &Shape::new(vec![rows, total_cols]), DType::F32)
        .unwrap();
    let h2 = dev
        .write_cols(dst.as_mut(), total_cols, 3, patch_s.as_ref(), rows, cols)
        .unwrap();
    h2.synchronize().unwrap();
    close(&dst.to_cpu_vec_f32().unwrap(), &expected, 1e-7, "write_cols");
}

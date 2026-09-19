//! gemm matmul tests — split from the original monolithic lib_internal_tests.rs.

use super::common::{approx_eq, run_matmul_on_dev};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    const GPU_TEST_ENV: &str = "GRIM_GPU_TEST";

    fn run_matmul_op(
        env_present: bool,
        a: &[f32],
        a_dims: &[usize],
        b: &[f32],
        b_dims: &[usize],
        out_dims: &[usize],
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        Some(run_matmul_on_dev(&dev, a, a_dims, b, b_dims, out_dims))
    }

    fn cpu_matmul(a: &[f32], a_dims: &[usize], b: &[f32], b_dims: &[usize]) -> Vec<f32> {
        let (m, k) = (a_dims[0], a_dims[1]);
        debug_assert_eq!(b_dims[1], k, "cpu_matmul: b must be [N, K]");
        let n = b_dims[0];
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn transpose2d(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = v[r * cols + c];
            }
        }
        out
    }

    #[test]
    fn matmul_batched_matches_loop_of_single_gemms() {
        // Item 6: a batch of same-shape GEMMs collapsed into one
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        for &batch in &[1usize, 3, 5] {
            let m = 8usize;
            let k = 16usize;
            let n = 8usize;
            let mut a_storages: Vec<Box<dyn BackendStorage>> = Vec::new();
            let mut b_storages: Vec<Box<dyn BackendStorage>> = Vec::new();
            for bi in 0..batch {
                let av: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05) + bi as f32).collect();
                // SPEED-ROC-16: b is [N, K] = [n, k].
                let bv: Vec<f32> = (0..n * k)
                    .map(|i| (i as f32 * 0.05) - 0.5 + bi as f32)
                    .collect();
                a_storages.push(
                    dev.from_cpu(&av, &Shape::from_slice(&[m, k]), DType::F32)
                        .unwrap(),
                );
                b_storages.push(
                    dev.from_cpu(&bv, &Shape::from_slice(&[n, k]), DType::F32)
                        .unwrap(),
                );
            }
            let a_refs: Vec<&dyn BackendStorage> = a_storages.iter().map(|s| s.as_ref()).collect();
            let b_refs: Vec<&dyn BackendStorage> = b_storages.iter().map(|s| s.as_ref()).collect();
            let batched = dev
                .matmul_batched(&a_refs, &b_refs, &Shape::from_slice(&[m, n]))
                .unwrap();
            assert_eq!(
                batched.len(),
                batch,
                "batch count mismatch for batch={batch}"
            );
            for bi in 0..batch {
                let (ref_out, _h) = dev
                    .matmul(
                        a_storages[bi].as_ref(),
                        b_storages[bi].as_ref(),
                        &Shape::from_slice(&[m, n]),
                    )
                    .unwrap();
                let ref_vec = ref_out.to_cpu_vec_f32().unwrap();
                let got = batched[bi].to_cpu_vec_f32().unwrap();
                assert_eq!(got.len(), ref_vec.len(), "len mismatch batch {bi}");
                for (i, (g, e)) in got.iter().zip(ref_vec.iter()).enumerate() {
                    assert!(
                        approx_eq(*g, *e, 1e-2),
                        "matmul_batched mismatch batch {bi} [{}/{}]: got {}, loop {}",
                        i / n,
                        i % n,
                        g,
                        e
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_ex_f32_matches_cpu_reference() {
        // Force the gemm_ex (extended-datatype) code path even for FP32 inputs by
        temp_env::with_var("GRIM_GPU_TARGET", Some("gfx90a"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
            let a_dims = [4usize, 8];
            // SPEED-ROC-16: b is [N, K] = [4, 8] (old [K, N] values transposed).
            let b_dims = [4usize, 8];
            let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 + 1.0).collect();
            let b_old: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2) - 3.0).collect();
            let b = transpose2d(&b_old, 8, 4);
            let expected = cpu_matmul(&a, &a_dims, &b, &b_dims);
            let got = run_matmul_op(env, &a, &a_dims, &b, &b_dims, &[4, 4]);
            if let Some(out) = got {
                assert_eq!(out.len(), expected.len());
                for (i, (g, e)) in out.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        approx_eq(*g, *e, 1e-2),
                        "gemm_ex f32 mismatch at [{}/{}]: got {}, expected {}",
                        i / 4,
                        i % 4,
                        g,
                        e
                    );
                }
            }
        });
    }

}

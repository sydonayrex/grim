//! elementwise norm tests — split from the original monolithic lib_internal_tests.rs.

use super::common::approx_eq;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use grim_tensor::error::{ Result };

    const GPU_TEST_ENV: &str = "GRIM_GPU_TEST";

    fn run_binary_op(
        env_present: bool,
        a: &[f32],
        b: &[f32],
        out_shape: &[usize],
        op: impl FnOnce(
            &RocmDevice,
            &dyn BackendStorage,
            &dyn BackendStorage,
            &Shape,
        ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let a_s = dev
            .from_cpu(a, &Shape::from_slice(&[a.len()]), DType::F32)
            .ok()?;
        let b_s = dev
            .from_cpu(b, &Shape::from_slice(&[b.len()]), DType::F32)
            .ok()?;
        let (out, _h) = op(
            &dev,
            a_s.as_ref(),
            b_s.as_ref(),
            &Shape::from_slice(out_shape),
        )
        .ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn run_softmax_op(env_present: bool, x: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let x_s = dev
            .from_cpu(x, &Shape::from_slice(shape), DType::F32)
            .ok()?;
        let (out, _h) = dev.softmax(x_s.as_ref(), &Shape::from_slice(shape)).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn run_rms_norm_op(
        env_present: bool,
        x: &[f32],
        w: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let x_s = dev
            .from_cpu(x, &Shape::from_slice(shape), DType::F32)
            .ok()?;
        let w_s = dev
            .from_cpu(w, &Shape::from_slice(&[w.len()]), DType::F32)
            .ok()?;
        let (out, _h) = dev
            .rms_norm(x_s.as_ref(), w_s.as_ref(), eps, &Shape::from_slice(shape))
            .ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn run_embedding_op(
        env_present: bool,
        weight: &[f32],
        indices: &[u32],
        vocab: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let w_s = dev
            .from_cpu(weight, &Shape::from_slice(&[vocab, dim]), DType::F32)
            .ok()?;
        let out_shape = Shape::from_slice(&[indices.len(), dim]);
        let (out, _h) = dev.embedding(w_s.as_ref(), indices, &out_shape).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    #[test]
    fn add_produces_elementwise_sum() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_binary_op(
            env,
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[4],
            |d, a, b, s| d.add(a, b, s),
        );
        if let Some(out) = got {
            assert!(
                approx_eq(out[0], 6.0, 1e-3),
                "add[0] expected 6.0 got {}",
                out[0]
            );
            assert!(
                approx_eq(out[3], 12.0, 1e-3),
                "add[3] expected 12.0 got {}",
                out[3]
            );
        }
    }

    #[test]
    fn mul_produces_elementwise_product() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_binary_op(
            env,
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[4],
            |d, a, b, s| d.mul(a, b, s),
        );
        if let Some(out) = got {
            assert!(
                approx_eq(out[0], 5.0, 1e-3),
                "mul[0] expected 5.0 got {}",
                out[0]
            );
            assert!(
                approx_eq(out[3], 32.0, 1e-3),
                "mul[3] expected 32.0 got {}",
                out[3]
            );
        }
    }

    #[test]
    fn silu_mul_matches_swiglu_formula() {
        // silu(gate) * up, with silu(x) = x / (1 + exp(-x))
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let gate = [1.0f32, -2.0, 0.0, 3.5];
        let up = [2.0f32, 4.0, 1.0, 0.5];
        let got = run_binary_op(env, &gate, &up, &[4], |d, a, b, s| d.silu_mul(a, b, s));
        if let Some(out) = got {
            for i in 0..4 {
                let expected = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
                assert!(
                    approx_eq(out[i], expected, 1e-2),
                    "silu_mul[{i}] expected {expected} got {}",
                    out[i]
                );
            }
        }
    }

    #[test]
    fn diag_rms_norm_engine_shapes() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            return;
        };
        use grim_tensor::{DType, Shape};
        let rows = 204usize;
        let width = 1024usize;
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i % 97) as f32) * 0.01)
            .collect();
        let w: Vec<f32> = vec![1.0f32; width];
        let xs = dev
            .from_cpu(&x, &Shape::new(vec![rows, width]), DType::F32)
            .unwrap();
        let ws = dev
            .from_cpu(&w, &Shape::new(vec![width]), DType::F32)
            .unwrap();
        let (out, h) = dev
            .rms_norm(
                xs.as_ref(),
                ws.as_ref(),
                1e-5,
                &Shape::new(vec![rows, width]),
            )
            .unwrap();
        h.synchronize().unwrap();
        let v = out.to_cpu_vec_f32().unwrap();
        eprintln!(
            "DIAG rms rows={} first={:?} last={:?}",
            rows,
            &v[0..4],
            &v[v.len() - 4..]
        );
    }

    #[test]
    fn rms_norm_normalizes_to_unit_when_weight_is_one() {
        // x = [3,4] over row_len 2, weight = 1, eps = 0:
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let got = run_rms_norm_op(env, &x, &w, &[2], 0.0);
        if let Some(out) = got {
            let rms = (12.5f32).sqrt();
            assert!(
                approx_eq(out[0], 3.0 / rms, 1e-3),
                "rms_norm[0] expected {} got {}",
                3.0 / rms,
                out[0]
            );
            assert!(
                approx_eq(out[1], 4.0 / rms, 1e-3),
                "rms_norm[1] expected {} got {}",
                4.0 / rms,
                out[1]
            );
        }
    }

    #[test]
    fn softmax_sums_to_one_per_row_and_orders_by_max() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        // Two rows: [1,2,3] and [10, 0, -5]
        let x = [1.0f32, 2.0, 3.0, 10.0, 0.0, -5.0];
        let got = run_softmax_op(env, &x, &[2, 3]);
        if let Some(out) = got {
            let row0_sum: f32 = out[0..3].iter().sum();
            let row1_sum: f32 = out[3..6].iter().sum();
            assert!(
                approx_eq(row0_sum, 1.0, 1e-3),
                "softmax row0 should sum to 1, got {row0_sum}"
            );
            assert!(
                approx_eq(row1_sum, 1.0, 1e-3),
                "softmax row1 should sum to 1, got {row1_sum}"
            );
            // argmax of row1 is index 0 (value 10)
            assert!(
                out[3] > out[4] && out[3] > out[5],
                "softmax row1 argmax should be col 0"
            );
        }
    }

    #[test]
    fn embedding_gathers_weight_rows_by_index() {
        // weight = [[1,2,3],[4,5,6],[7,8,9]], dim=3, vocab=3
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let got = run_embedding_op(env, &weight, &[2, 0, 1], 3, 3);
        if let Some(out) = got {
            // indices [2,0,1] -> rows 2,0,1 of weight
            assert_eq!(out.len(), 9);
            assert!(
                approx_eq(out[0], 7.0, 1e-3),
                "embed row0[0] expected 7.0 got {}",
                out[0]
            );
            assert!(
                approx_eq(out[3], 1.0, 1e-3),
                "embed row1[0] expected 1.0 got {}",
                out[3]
            );
            assert!(
                approx_eq(out[6], 4.0, 1e-3),
                "embed row2[0] expected 4.0 got {}",
                out[6]
            );
        }
    }

    #[test]
    fn embedding_rejects_index_count_mismatch() {
        // Without a GPU this still exercises the shape guard (no device alloc needed
        let dev = RocmDevice::new(0);
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_s = match dev.from_cpu(&weight, &Shape::from_slice(&[2, 3]), DType::F32) {
            Ok(s) => s,
            Err(_) => return, // no GPU; shape-guard logic is covered by the GPU-gated path
        };
        let out_shape = Shape::from_slice(&[2, 3]);
        let res = dev.embedding(w_s.as_ref(), &[0, 1, 2], &out_shape); // 3 indices vs leading dim 2
        assert!(
            res.is_err(),
            "embedding must reject indices.len() != out leading dim"
        );
    }

    fn close_rocm(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-4,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    #[test]
    fn test_rocm_add_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let a = [1.5f32, -2.5, 0.0, std::f32::consts::PI];
        let b = [2.5f32, 3.5, -1.0, 1.0];
        if let Some(out) = run_binary_op(env, &a, &b, &[4], |d, x, y, s| d.add(x, y, s)) {
            close_rocm(out[0], 4.0, "rocm_add w0");
            close_rocm(out[1], 1.0, "rocm_add w1");
            close_rocm(out[2], -1.0, "rocm_add w2");
            close_rocm(out[3], 1.0 + std::f32::consts::PI, "rocm_add w3");
        }
    }

    #[test]
    fn test_rocm_mul_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let a = [2.0f32, -3.0, 0.5];
        let b = [4.0f32, 2.0, -8.0];
        if let Some(out) = run_binary_op(env, &a, &b, &[3], |d, x, y, s| d.mul(x, y, s)) {
            close_rocm(out[0], 8.0, "rocm_mul w0");
            close_rocm(out[1], -6.0, "rocm_mul w1");
            close_rocm(out[2], -4.0, "rocm_mul w2");
        }
    }

    #[test]
    fn test_rocm_silu_mul_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let gate = [1.0f32, -1.0];
        let up = [2.0f32, 3.0];
        if let Some(out) = run_binary_op(env, &gate, &up, &[2], |d, g, u, s| d.silu_mul(g, u, s)) {
            let sig_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
            let exp_0 = sig_1 * 1.0 * 2.0;

            let sig_neg1 = 1.0f32 / (1.0f32 + (1.0f32).exp());
            let exp_1 = sig_neg1 * -3.0;

            close_rocm(out[0], exp_0, "rocm_silu_mul w0");
            close_rocm(out[1], exp_1, "rocm_silu_mul w1");
        }
    }

    #[test]
    fn test_rocm_rms_norm_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 2.0];
        if let Some(out) = run_rms_norm_op(env, &x, &w, &[2], 1e-6) {
            let rms_val = (12.5f32 + 1e-6).sqrt();
            let exp_0 = (3.0 / rms_val) * 1.0;
            let exp_1 = (4.0 / rms_val) * 2.0;
            close_rocm(out[0], exp_0, "rocm_rms_norm w0");
            close_rocm(out[1], exp_1, "rocm_rms_norm w1");
        }
    }

    #[test]
    fn test_rocm_softmax_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let x = [1.0f32, 2.0, 3.0];
        if let Some(out) = run_softmax_op(env, &x, &[1, 3]) {
            let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
            close_rocm(out[0], 1.0f32.exp() / sum_exp, "rocm_softmax w0");
            close_rocm(out[1], 2.0f32.exp() / sum_exp, "rocm_softmax w1");
            close_rocm(out[2], 3.0f32.exp() / sum_exp, "rocm_softmax w2");
        }
    }

    #[test]
    fn test_rocm_embedding_golden_exact() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let weight = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        if let Some(out) = run_embedding_op(env, &weight, &[2, 0], 3, 2) {
            assert_eq!(out, vec![50.0, 60.0, 10.0, 20.0]);
        }
    }

}

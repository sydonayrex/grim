use super::common::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::cuda_device::CudaDevice;
    use grim_tensor::dtype::DType;
    use grim_tensor::{CoreTensorOps, ElementwiseOps, Shape};

    #[test]
    fn test_cuda_device_probe() {
        let Ok(devices) = CudaDevice::probe() else {
            return;
        };
        if let Some(dev) = devices.first() {
            assert_eq!(dev.ordinal, 0);
        }
    }

    #[test]
    fn test_cuda_zeros() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![0.0; 8]);
    }

    #[test]
    fn test_cuda_from_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let shape = Shape::new(vec![3, 2]);
        let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, host_data);
    }

    #[test]
    fn test_cuda_math_ops() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let shape = Shape::new(vec![4]);
        let host_data = vec![4.0f32, 9.0, 16.0, 25.0];
        let x = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();

        let (out_sqrt, _) = dev.sqrt(x.as_ref(), &shape).unwrap();
        assert_eq!(out_sqrt.to_cpu_vec_f32().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

        let (out_recip, _) = dev.recip(out_sqrt.as_ref(), &shape).unwrap();
        assert_eq!(
            out_recip.to_cpu_vec_f32().unwrap(),
            vec![0.5, 1.0 / 3.0, 0.25, 0.2]
        );

        let (out_mul, _) = dev.mul_scalar(x.as_ref(), 0.5, &shape).unwrap();
        assert_eq!(out_mul.to_cpu_vec_f32().unwrap(), vec![2.0, 4.5, 8.0, 12.5]);
    }

    #[test]
    fn test_cuda_matmul() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let a_shape = Shape::new(vec![2, 2]);
        let b_shape = Shape::new(vec![2, 2]);
        let out_shape = Shape::new(vec![2, 2]);

        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

        let (out_storage, handle) = dev
            .matmul(a_storage.as_ref(), b_storage.as_ref(), &out_shape)
            .unwrap();
        handle.synchronize().unwrap();

        let res = out_storage.to_cpu_vec_f32().unwrap();
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cuda_ops() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let shape = Shape::new(vec![4]);

        let a = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        // 1. Add
        let (out_add, h) = dev.add(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(
            out_add.to_cpu_vec_f32().unwrap(),
            vec![6.0, 8.0, 10.0, 12.0]
        );

        // 2. Mul
        let (out_mul, h) = dev.mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(
            out_mul.to_cpu_vec_f32().unwrap(),
            vec![5.0, 12.0, 21.0, 32.0]
        );

        // 3. SiLU Mul
        let (out_silu, h) = dev.silu_mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_silu = out_silu.to_cpu_vec_f32().unwrap();
        let expected_silu0 = (1.0f32 / (1.0f32 + (-1.0f32).exp())) * 5.0f32;
        assert!((res_silu[0] - expected_silu0).abs() < 1e-4);

        // 4. RMS Norm
        let weight_data = vec![1.0, 1.0, 1.0, 1.0];
        let weight = dev.from_cpu(&weight_data, &shape, DType::F32).unwrap();
        let (out_rms, h) = dev
            .rms_norm(a.as_ref(), weight.as_ref(), 1e-5, &shape)
            .unwrap();
        h.synchronize().unwrap();
        let res_rms = out_rms.to_cpu_vec_f32().unwrap();
        // RMS([1,2,3,4]) = sqrt((1+4+9+16)/4) ≈ 2.7386
        let rms_val = 7.5f32.sqrt();
        assert!((res_rms[0] - 1.0 / rms_val).abs() < 1e-4);

        // 5. Softmax
        let (out_sm, h) = dev.softmax(a.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_sm = out_sm.to_cpu_vec_f32().unwrap();
        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp();
        assert!((res_sm[0] - 1.0f32.exp() / sum_exp).abs() < 1e-4);

        // 6. Embedding
        let weight_emb_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let weight_emb = dev
            .from_cpu(&weight_emb_data, &Shape::new(vec![3, 2]), DType::F32)
            .unwrap();
        let indices = vec![2u32, 0u32];
        let out_emb_shape = Shape::new(vec![2, 2]);
        let (out_emb, h) = dev
            .embedding(weight_emb.as_ref(), &indices, &out_emb_shape)
            .unwrap();
        h.synchronize().unwrap();
        let res_emb = out_emb.to_cpu_vec_f32().unwrap();
        assert_eq!(res_emb, vec![50.0, 60.0, 10.0, 20.0]);
    }

    #[test]
    fn test_cuda_matmul_shape_mismatch_returns_error() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a_shape = Shape::new(vec![2, 3]);
        let b_shape = Shape::new(vec![4, 2]); // K mismatch: 3 != 4
        let out_shape = Shape::new(vec![2, 2]);

        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

        let res = dev.matmul(a_storage.as_ref(), b_storage.as_ref(), &out_shape);
        assert!(
            res.is_err(),
            "matmul with mismatched inner dimension K must return Err"
        );
    }

    #[test]
    fn test_cuda_rms_norm_exact() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

        let x_data = vec![3.0f32, 4.0]; // mean(x^2) = (9 + 16)/2 = 12.5
        let weight_data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);

        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let weight = dev.from_cpu(&weight_data, &shape, DType::F32).unwrap();

        let (out_rms, h) = dev
            .rms_norm(x.as_ref(), weight.as_ref(), 1e-6, &shape)
            .unwrap();
        h.synchronize().unwrap();

        let res = out_rms.to_cpu_vec_f32().unwrap();
        let rms_val = (12.5f32 + 1e-6).sqrt();
        let expected_0 = (3.0 / rms_val) * 1.0;
        let expected_1 = (4.0 / rms_val) * 2.0;

        assert!(
            (res[0] - expected_0).abs() < 1e-4,
            "res[0] = {}, want {}",
            res[0],
            expected_0
        );
        assert!(
            (res[1] - expected_1).abs() < 1e-4,
            "res[1] = {}, want {}",
            res[1],
            expected_1
        );
    }

    #[test]
    fn test_cuda_softmax_exact() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

        let x_data = vec![1.0f32, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();

        let (out_sm, h) = dev.softmax(x.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();

        let res = out_sm.to_cpu_vec_f32().unwrap();
        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        let expected_0 = 1.0f32.exp() / sum_exp;
        let expected_1 = 2.0f32.exp() / sum_exp;
        let expected_2 = 3.0f32.exp() / sum_exp;

        assert!((res[0] - expected_0).abs() < 1e-4);
        assert!((res[1] - expected_1).abs() < 1e-4);
        assert!((res[2] - expected_2).abs() < 1e-4);
    }
}

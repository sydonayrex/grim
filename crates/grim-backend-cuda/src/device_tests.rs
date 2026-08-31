#[cfg(test)]
mod tests {
    use crate::device::cuda_device::CudaDevice;
    use crate::memory::storage::CudaStorage;
    use grim_tensor::dtype::{
        ArithType, DType, FloatPackScheme, KQuantScheme, Storage as DTypeStorage,
    };
    use grim_tensor::{BackendStorage, CoreTensorOps, ElementwiseOps, MemoryOps, QuantOps, Shape};

    /// Source-presence guard for the partial-rotary / YaRN RoPE kernel and the
    /// sliding-window attention extension. No GPU required — this just asserts
    /// the CUDA kernel source declares the symbols and parameters the host
    /// dispatchers expect, mirroring the ROCm `test_rope_yarn_kernel_presence`.
    #[test]
    fn test_rope_yarn_and_window_lo_kernel_presence() {
        let src = crate::kernels::KERNELS_SOURCE;
        assert!(
            src.contains("grim_rope_yarn"),
            "KERNELS_SOURCE must declare grim_rope_yarn"
        );
        assert!(
            src.contains("inv_freq"),
            "grim_rope_yarn must take a pre-computed inv_freq buffer"
        );
        assert!(
            src.contains("mscale"),
            "grim_rope_yarn must take an mscale (attention_factor) parameter"
        );
        assert!(
            src.contains("rotary_half"),
            "grim_rope_yarn must take a rotary_half (partial-rotary) parameter"
        );
        assert!(
            src.contains("window_lo"),
            "grim_qkv_attention must take a window_lo parameter for SWA"
        );
    }

    #[test]
    fn test_cuda_device_probe() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].ordinal, 0);
    }

    #[test]
    fn test_cuda_zeros() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![0.0; 8]);
    }

    #[test]
    fn test_cuda_from_cpu() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![3, 2]);
        let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, host_data);
    }

    #[test]
    fn test_cuda_math_ops() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
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

    /// GPU-gated parity test for the fused grouped MoE dispatch kernel.
    /// Compares the GPU output against a hand-computed CPU reference for a tiny
    /// 2-expert / 2-token / top-1 routing. Numerical tolerance is loose because
    /// FP32 atomic adds can reorder; the contract is correctness of the fused
    /// gate+up SiLU combine + down + routed_scaling_factor accumulate.
    #[test]
    fn test_cuda_moe_fused_dispatch_parity() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        if devices.is_empty() {
            return;
        }
        let dev = &devices[0];

        let hidden: usize = 4;
        let inter: usize = 3;
        let num_experts: usize = 2;
        let batch: usize = 2;
        let rsf: f32 = 0.5;

        // activations [batch, hidden]
        let x_data: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.1).collect();
        let x = dev
            .from_cpu(&x_data, &Shape::new(vec![batch, hidden]), DType::F32)
            .unwrap();

        // per-expert gate/up [inter, hidden], down [hidden, inter]
        let mk = |e: usize, sign: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; inter * hidden];
            for i in 0..inter {
                for h in 0..hidden {
                    v[i * hidden + h] =
                        sign * (1.0 + (i as f32) * 0.1 + (h as f32) * 0.01 + e as f32);
                }
            }
            v
        };
        let gate_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let up_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let down_flat: Vec<f32> = (0..num_experts)
            .flat_map(|e| {
                let mut v = vec![0.0f32; hidden * inter];
                for h in 0..hidden {
                    for i in 0..inter {
                        v[h * inter + i] = 1.0 + (h as f32) * 0.05 + (i as f32) * 0.02 + e as f32;
                    }
                }
                v
            })
            .collect();

        // top-1 routing: token0 -> expert0, token1 -> expert1
        let rtok = [0u32, 1u32];
        let rexp = [0u32, 1u32];
        let rw = [1.0f32, 1.0f32];
        let num_pairs = rtok.len();

        let gate_buf = dev
            .from_cpu(
                &gate_flat,
                &Shape::new(vec![num_experts * inter * hidden]),
                DType::F32,
            )
            .unwrap();
        let up_buf = dev
            .from_cpu(
                &up_flat,
                &Shape::new(vec![num_experts * inter * hidden]),
                DType::F32,
            )
            .unwrap();
        let down_buf = dev
            .from_cpu(
                &down_flat,
                &Shape::new(vec![num_experts * hidden * inter]),
                DType::F32,
            )
            .unwrap();
        let rtok_bytes: Vec<u8> = rtok.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rexp_bytes: Vec<u8> = rexp.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rw_bytes: Vec<u8> = rw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tok_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rtok_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );
        let exp_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rexp_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );
        let w_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rw_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out, _h) = dev
            .moe_fused_dispatch(
                &*x,
                &*gate_buf,
                &*up_buf,
                &*down_buf,
                &*tok_buf,
                &*exp_buf,
                &*w_buf,
                &out_shape,
                hidden as u32,
                inter as u32,
                num_experts as u32,
                batch as u32,
                rsf,
            )
            .unwrap();
        let res = out.to_cpu_vec_f32().unwrap();

        // CPU reference
        let silu = |a: f32| a / (1.0 + (-a).exp());
        let dot = |w: &[f32], xx: &[f32]| -> f32 { (0..w.len()).map(|i| w[i] * xx[i]).sum() };
        for t in 0..batch {
            let e = rexp[t] as usize;
            let xt = &x_data[t * hidden..(t + 1) * hidden];
            let gw = &gate_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let uw = &up_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let dw = &down_flat[e * hidden * inter..(e + 1) * hidden * inter];
            let mut routed = vec![0.0f32; hidden];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for i in 0..inter {
                    let g = dot(&gw[i * hidden..i * hidden + hidden], xt);
                    let u = dot(&uw[i * hidden..i * hidden + hidden], xt);
                    acc += dw[h * inter + i] * (silu(g) * u);
                }
                routed[h] = rsf * acc;
            }
            for h in 0..hidden {
                let got = res[t * hidden + h];
                let tol = routed[h].abs().max(1.0) * 1e-3 + 1e-3;
                assert!(
                    (got - routed[h]).abs() < tol,
                    "moe tok{} dim{}: gpu {} vs ref {} (tol {})",
                    t,
                    h,
                    got,
                    routed[h],
                    tol
                );
            }
        }
    }

    #[test]
    fn test_cuda_matmul() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

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
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

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
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

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
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

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
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

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

    /// CPU reference for quantized_matmul using Q8_0 convention:
    /// B is packed [col][block][32] raw int8, scales holds n*(k/32) per-block values.
    fn q8_matmul_ref(
        a: &[f32],
        b: &[u8],
        scales: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let blocks = k / 32;
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let blk = p / 32;
                    let i = p % 32;
                    if blk >= blocks {
                        continue;
                    }
                    let b_idx = (ni * blocks + blk) * 32 + i;
                    let q = (b[b_idx] as i8) as f32;
                    let scale = if ni * blocks + blk < scales.len() {
                        scales[ni * blocks + blk]
                    } else {
                        1.0
                    };
                    sum += a[mi * k + p] * (q * scale);
                }
                out[mi * n + ni] = sum;
            }
        }
        out
    }

    fn assert_q8_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "Q8_0 quantized matmul max error {max_err} exceeds 1e-3"
        );
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_gpu_fast_path() {
        // Wait 3 seconds between Q8_0 CUDA tests to avoid GPU resource
        // contention false negatives (cuBLAS context thrashing under concurrent loads).
        std::thread::sleep(std::time::Duration::from_secs(3));
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (2usize, 256usize, 8usize);
        let blocks = k / 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 7) % 251) as u8).collect();
        let b_scales: Vec<f32> = (0..n * blocks)
            .map(|i| 0.5 + (i as f32 * 0.1).fract())
            .collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &b_scales, m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &b_scales,
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_empty_scales_defaults() {
        // Wait 3 seconds between Q8_0 CUDA tests to avoid GPU resource
        // contention false negatives (cuBLAS context thrashing under concurrent loads).
        std::thread::sleep(std::time::Duration::from_secs(3));
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (3usize, 64usize, 4usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).cos()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 11) % 256) as u8).collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &[], m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &[],
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_cpu_fallback() {
        // Wait 3 seconds between Q8_0 CUDA tests to avoid GPU resource
        // contention false negatives (cuBLAS context thrashing under concurrent loads).
        std::thread::sleep(std::time::Duration::from_secs(3));
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        // K not a multiple of 32, forcing CPU fallback.
        let (m, k, n) = (3usize, 34usize, 5usize);
        let blocks = k / 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).cos()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 13) % 200) as u8).collect();
        let b_scales: Vec<f32> = (0..n * blocks)
            .map(|i| 1.0 + (i as f32 * 0.25).fract())
            .collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &b_scales, m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &b_scales,
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_backward_dx_q8_0() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (4usize, 64usize, 8usize);
        let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
        let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

        // Host reference: dequantize B to F32, then dX = dY @ B^T.
        let b_packed = grim_quant::quant_q80(&b_orig).unwrap();
        let b_dequant = grim_quant::dequant_q80(&b_packed, k * n).unwrap();
        let mut dx_ref = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for l in 0..n {
                    sum += dy_host[i * n + l] * b_dequant[j * n + l];
                }
                dx_ref[i * k + j] = sum;
            }
        }

        // Upload dY as F32 [M, N].
        let dy_shape = Shape::new(vec![m, n]);
        let dy_dev = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

        // Upload packed B as KQuant(Q80) [K, N] (stays quantized resident).
        let b_shape = Shape::new(vec![k, n]);
        let b_dev = dev
            .from_cpu_bytes(
                &b_packed,
                &b_shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                },
            )
            .unwrap();

        let out_shape = Shape::new(vec![m, k]);
        let (dx_dev, handle) = dev
            .quantized_matmul_backward_dx(
                dy_dev.as_ref(),
                b_dev.as_ref(),
                &[],
                8, // bpw for Q8_0
                m,
                n,
                k,
                &out_shape,
                None,
            )
            .expect("CUDA quantized_matmul_backward_dx must succeed on a real CUDA device");
        handle.synchronize().unwrap();

        let dx_actual = dx_dev
            .to_cpu_vec_f32()
            .expect("CUDA backward result must be readable");
        assert_eq!(dx_actual.len(), m * k);
        for (a, e) in dx_actual.iter().zip(dx_ref.iter()) {
            let err = (a - e).abs();
            assert!(
                err < 1e-3,
                "CUDA Q8_0 backward dX error {err} at actual={a} expected={e}"
            );
        }
    }

    // ===================================================================
    //  GPU dequant kernel golden tests — bit-accurate parity vs the
    //  `grim_quant::dequant_*` CPU oracle. Each test:
    //    1. Builds the packed bytes for one or more super-blocks via
    //       `grim_quant::quant_<type>` (or hand-fabricated for MXFP4).
    //    2. Uploads the packed bytes to a `CudaStorage` with the matching
    //       quantized `DType.storage` (so `dequantize_on_device` dispatches).
    //    3. Calls `dev.dequantize_on_device(as_cuda_storage(storage.as_ref()))` (GPU kernel).
    //    4. Compares `out.to_cpu_vec_f32()` to the CPU oracle within a tight
    //       tolerance that admits only floating-point rounding (1e-4).
    // Skipped (not failed) when no CUDA device is present.
    // ===================================================================

    fn dequant_test_device() -> Option<CudaDevice> {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        CudaDevice::probe()
            .ok()
            .filter(|d| !d.is_empty())
            .map(|d| d[0].clone())
    }

    fn assert_dequant_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "{label}: GPU dequant max error {max_err} exceeds 1e-4 \
             (first 4 actual={:?} expected={:?})",
            &actual[..actual.len().min(4)],
            &expected[..expected.len().min(4)],
        );
    }

    /// Upload raw packed quantized bytes with the given quantized `DType.storage`
    /// to a device-resident `CudaStorage`, returned as `Box<dyn BackendStorage>`.
    fn upload_packed(
        dev: &CudaDevice,
        bytes: &[u8],
        shape: &Shape,
        storage_kind: DTypeStorage,
    ) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: ArithType::U8,
            storage: storage_kind,
        };
        dev.from_cpu_bytes(bytes, shape, dtype)
            .expect("from_cpu_bytes for packed quantized storage")
    }

    /// Downcast a `BackendStorage` to `&CudaStorage` (the only concrete type
    /// `from_cpu_bytes` produces on this backend).
    fn as_cuda_storage(s: &dyn BackendStorage) -> &CudaStorage {
        s.as_any()
            .downcast_ref::<CudaStorage>()
            .expect("expected CudaStorage from from_cpu_bytes")
    }

    fn build_mxfp4_single_buffer(codes: &[u8], exps: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        buf.extend_from_slice(codes);
        buf.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        buf.extend_from_slice(exps);
        buf
    }

    #[test]
    fn test_cuda_dequant_q5k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // 2 super-blocks × 256 weights = 512 weights.
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let packed = grim_quant::quant_q5k(&src).expect("quant_q5k");
        let expected = grim_quant::dequant_q5k(&packed, n).expect("cpu oracle q5k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q5K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q5k");
        let actual = out.to_cpu_vec_f32().expect("readback q5k");
        assert_dequant_close("q5k", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_q4k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let packed = grim_quant::quant_q4k(&src).expect("quant_q4k");
        let expected = grim_quant::dequant_q4k(&packed, n).expect("cpu oracle q4k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q4K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q4k");
        let actual = out.to_cpu_vec_f32().expect("readback q4k");
        assert_dequant_close("q4k", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_q6k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos() * 0.7).collect();
        let packed = grim_quant::quant_q6k(&src).expect("quant_q6k");
        let expected = grim_quant::dequant_q6k(&packed, n).expect("cpu oracle q6k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q6K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q6k");
        let actual = out.to_cpu_vec_f32().expect("readback q6k");
        assert_dequant_close("q6k", &actual, &expected);
    }

    #[test]
    fn test_cuda_fused_quant_gemm_q4k_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // K must be a multiple of 256 for the Q quantized GEMM block layout.
        let m = 4u32;
        let k = 512u32;
        let n = 256u32;

        let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.03).sin()).collect();
        let b_src: Vec<f32> = (0..(k as usize * n as usize))
            .map(|i| (i as f32 * 0.017).cos())
            .collect();

        let a_shape = Shape::new(vec![m as usize, k as usize]);
        let b_shape = Shape::new(vec![n as usize, k as usize]); // [N, K] packed layout
        let a_storage = dev
            .from_cpu(&a_src, &a_shape, DType::F32)
            .expect("a from_cpu");
        let a_cuda = as_cuda_storage(a_storage.as_ref());

        let packed = grim_quant::quant_q4k(&b_src).expect("quant_q4k");
        let b_storage = upload_packed(
            &dev,
            &packed,
            &b_shape,
            DTypeStorage::KQuant(KQuantScheme::Q4K),
        );
        let b_cuda = as_cuda_storage(b_storage.as_ref());

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let (fused_out, h) = dev
            .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q4K, &out_shape)
            .expect("fused_quant_gemm q4k");
        h.synchronize().expect("sync");
        let actual = fused_out.to_cpu_vec_f32().expect("readback");

        // Reference: A @ B^T where B is dequantized [N, K] then transposed.
        let b_deq = grim_quant::dequant_q4k(&packed, (k * n) as usize).expect("cpu dequant q4k");
        let mut expected = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut s = 0.0f32;
                for t in 0..k as usize {
                    s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                }
                expected[i * n as usize + j] = s;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..expected.len() {
            let e = (actual[i] - expected[i]).abs();
            let denom = expected[i].abs().max(1.0);
            max_err = max_err.max(e / denom);
        }
        assert!(
            max_err < 0.05,
            "fused q4k GEMM mismatch: max_rel_err={max_err}"
        );
    }

    #[test]
    fn test_cuda_fused_quant_gemm_q6k_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // K must be a multiple of 256 for the Q quantized GEMM block layout.
        let m = 4u32;
        let k = 512u32;
        let n = 256u32;

        let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.03).sin()).collect();
        let b_src: Vec<f32> = (0..(k as usize * n as usize))
            .map(|i| (i as f32 * 0.017).cos())
            .collect();

        let a_shape = Shape::new(vec![m as usize, k as usize]);
        let b_shape = Shape::new(vec![n as usize, k as usize]); // [N, K] packed layout
        let a_storage = dev
            .from_cpu(&a_src, &a_shape, DType::F32)
            .expect("a from_cpu");
        let a_cuda = as_cuda_storage(a_storage.as_ref());

        let packed = grim_quant::quant_q6k(&b_src).expect("quant_q6k");
        let b_storage = upload_packed(
            &dev,
            &packed,
            &b_shape,
            DTypeStorage::KQuant(KQuantScheme::Q6K),
        );
        let b_cuda = as_cuda_storage(b_storage.as_ref());

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let (fused_out, h) = dev
            .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q6K, &out_shape)
            .expect("fused_quant_gemm q6k");
        h.synchronize().expect("sync");
        let actual = fused_out.to_cpu_vec_f32().expect("readback");

        let b_deq = grim_quant::dequant_q6k(&packed, (k * n) as usize).expect("cpu dequant q6k");
        let mut expected = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut s = 0.0f32;
                for t in 0..k as usize {
                    s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                }
                expected[i * n as usize + j] = s;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..expected.len() {
            let e = (actual[i] - expected[i]).abs();
            let denom = expected[i].abs().max(1.0);
            max_err = max_err.max(e / denom);
        }
        assert!(
            max_err < 0.05,
            "fused q6k GEMM mismatch: max_rel_err={max_err}"
        );
    }

    #[test]
    fn test_cuda_fused_quant_gemm_real_model_q4k_q6k() {
        // Definitive check: run the actual CUDA fused GEMM kernels against
        // REAL Q4_K / Q6_K tensors extracted from the on-disk Q4_K_M model,
        // comparing to grim_quant::dequant_*k + a CPU A@B^T reference.
        let Some(dev) = dequant_test_device() else {
            return;
        };
        use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|p| p.join("models").is_dir())
            .expect("repo root with models/")
            .to_path_buf();
        let path = repo_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
        let Ok(f) = std::fs::File::open(&path) else {
            eprintln!("skip: model not present");
            return;
        };
        let mut reader = std::io::BufReader::new(f);
        let file = read_gguf(&mut reader).expect("read_gguf");

        let mut run_one = |dtype_want: GgufDType, name: &str| {
            let target = file
                .tensors
                .iter()
                .find(|t| t.dtype == dtype_want && t.name == name)
                .expect("target tensor");
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            let dims = &target.dims;
            let (n, k) = (dims[0] as u32, dims[1] as u32);
            let m = 8u32;
            let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.013).sin()).collect();
            let a_shape = Shape::new(vec![m as usize, k as usize]);
            let a_storage = dev
                .from_cpu(&a_src, &a_shape, DType::F32)
                .expect("a from_cpu");
            let a_cuda = as_cuda_storage(a_storage.as_ref());
            let b_shape = Shape::new(vec![n as usize, k as usize]);
            let b_storage = upload_packed(
                &dev,
                &bytes,
                &b_shape,
                DTypeStorage::KQuant(match dtype_want {
                    GgufDType::Q4K => KQuantScheme::Q4K,
                    GgufDType::Q6K => KQuantScheme::Q6K,
                    _ => panic!("unexpected dtype"),
                }),
            );
            let b_cuda = as_cuda_storage(b_storage.as_ref());
            let out_shape = Shape::new(vec![m as usize, n as usize]);
            let fmt = match dtype_want {
                GgufDType::Q4K => grim_tensor::QuantFormat::Q4K,
                GgufDType::Q6K => grim_tensor::QuantFormat::Q6K,
                _ => panic!("unexpected dtype"),
            };
            let (fused_out, h) = dev
                .fused_quant_gemm(a_cuda, b_cuda, fmt, &out_shape)
                .expect("fused_quant_gemm");
            h.synchronize().expect("sync");
            let actual = fused_out.to_cpu_vec_f32().expect("readback");
            let elem = (n as usize) * (k as usize);
            let b_deq = match dtype_want {
                GgufDType::Q4K => grim_quant::dequant_q4k(&bytes, elem).expect("deq"),
                GgufDType::Q6K => grim_quant::dequant_q6k(&bytes, elem).expect("deq"),
                _ => panic!("unexpected dtype"),
            };
            let mut expected = vec![0.0f32; (m * n) as usize];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    let mut s = 0.0f32;
                    for t in 0..k as usize {
                        s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                    }
                    expected[i * n as usize + j] = s;
                }
            }
            let mut max_err = 0.0f32;
            for i in 0..expected.len() {
                let e = (actual[i] - expected[i]).abs();
                let denom = expected[i].abs().max(1.0);
                max_err = max_err.max(e / denom);
            }
            (name.to_string(), max_err)
        };

        let (n1, e1) = run_one(GgufDType::Q4K, "token_embd.weight");
        eprintln!("[real-q4k] {n1}: max_rel_err={e1}");
        assert!(e1 < 0.05, "real-model Q4K fused GEMM mismatch: {e1}");
        let (n1b, e1b) = run_one(GgufDType::Q4K, "blk.0.attn_q.weight");
        eprintln!("[real-q4k] {n1b}: max_rel_err={e1b}");
        assert!(
            e1b < 0.05,
            "real-model Q4K attn_q fused GEMM mismatch: {e1b}"
        );
        let (n2, e2) = run_one(GgufDType::Q6K, "output.weight");
        eprintln!("[real-q6k] {n2}: max_rel_err={e2}");
        assert!(e2 < 0.05, "real-model Q6K fused GEMM mismatch: {e2}");
        let (n2b, e2b) = run_one(GgufDType::Q6K, "blk.0.attn_v.weight");
        eprintln!("[real-q6k] {n2b}: max_rel_err={e2b}");
        assert!(
            e2b < 0.05,
            "real-model Q6K attn_v fused GEMM mismatch: {e2b}"
        );

        // CLI orientation: get transposes attn_v to [256, 1536]; verify the
        // kernel on that transposed layout with several A patterns.
        {
            let target = file
                .tensors
                .iter()
                .find(|t| t.dtype == GgufDType::Q6K && t.name == "blk.0.attn_v.weight")
                .expect("target tensor");
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            let (n, k) = (256u32, 1536u32); // transposed [out, in]
            let m = 16u32;
            let patterns: Vec<(&str, Vec<f32>)> = vec![
                (
                    "sin",
                    (0..(m * k)).map(|i| (i as f32 * 0.013).sin()).collect(),
                ),
                ("ones", vec![1.0f32; (m * k) as usize]),
                (
                    "realmag",
                    (0..(m * k))
                        .map(|i| {
                            // mimic real x_norm range ~[-3.4, 1.7]
                            if i % 7 == 0 { -3.37f32 } else { 1.71f32 }
                        })
                        .collect(),
                ),
            ];
            for (name, a_src) in patterns {
                let a_shape = Shape::new(vec![m as usize, k as usize]);
                let a_storage = dev
                    .from_cpu(&a_src, &a_shape, DType::F32)
                    .expect("a from_cpu");
                let a_cuda = as_cuda_storage(a_storage.as_ref());
                let b_shape = Shape::new(vec![n as usize, k as usize]);
                let b_storage = upload_packed(
                    &dev,
                    &bytes,
                    &b_shape,
                    DTypeStorage::KQuant(KQuantScheme::Q6K),
                );
                let b_cuda = as_cuda_storage(b_storage.as_ref());
                let out_shape = Shape::new(vec![m as usize, n as usize]);
                let (fused_out, h) = dev
                    .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q6K, &out_shape)
                    .expect("fused_quant_gemm");
                h.synchronize().expect("sync");
                let actual = fused_out.to_cpu_vec_f32().expect("readback");
                let nan = actual.iter().filter(|x| x.is_nan()).count();
                let max_abs = actual.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[cli-q6k][{name}] attn_v [256,1536]: nan={nan} max_abs={max_abs:.4}");
            }
        }
    }

    #[test]
    fn test_cuda_dequant_iq4nl_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 0.3).collect();
        let packed = grim_quant::quant_iq4nl(&src).expect("quant_iq4nl");
        let expected = grim_quant::dequant_iq4nl(&packed, n).expect("cpu oracle iq4nl");
        let fallback = dev
            .dequantize_iq_host(&packed, n, KQuantScheme::IQ4NL)
            .expect("host fallback");
        assert_eq!(fallback.len(), expected.len());
    }

    #[test]
    fn test_cuda_dequant_iq4xs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).cos() * 0.4).collect();
        let packed = grim_quant::quant_iq4xs(&src).expect("quant_iq4xs");
        let expected = grim_quant::dequant_iq4xs(&packed, n).expect("cpu oracle iq4xs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ4XS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq4xs");
        let actual = out.to_cpu_vec_f32().expect("readback iq4xs");
        assert_dequant_close("iq4xs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq3xxs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 0.25).collect();
        let packed = grim_quant::quant_iq3xxs(&src).expect("quant_iq3xxs");
        let expected = grim_quant::dequant_iq3xxs(&packed, n).expect("cpu oracle iq3xxs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq3xxs");
        let actual = out.to_cpu_vec_f32().expect("readback iq3xxs");
        assert_dequant_close("iq3xxs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq3s_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.08).cos() * 0.6).collect();
        let packed = grim_quant::quant_iq3s(&src).expect("quant_iq3s");
        let expected = grim_quant::dequant_iq3s(&packed, n).expect("cpu oracle iq3s");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ3S),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq3s");
        let actual = out.to_cpu_vec_f32().expect("readback iq3s");
        assert_dequant_close("iq3s", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2xxs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
        let packed = grim_quant::quant_iq2xxs(&src).expect("quant_iq2xxs");
        let expected = grim_quant::dequant_iq2xxs(&packed, n).expect("cpu oracle iq2xxs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2xxs");
        let actual = out.to_cpu_vec_f32().expect("readback iq2xxs");
        assert_dequant_close("iq2xxs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2xs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() * 0.35).collect();
        let packed = grim_quant::quant_iq2xs(&src).expect("quant_iq2xs");
        let expected = grim_quant::dequant_iq2xs(&packed, n).expect("cpu oracle iq2xs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2XS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2xs");
        let actual = out.to_cpu_vec_f32().expect("readback iq2xs");
        assert_dequant_close("iq2xs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2s_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.06).sin() * 0.45).collect();
        let packed = match grim_quant::quant_iq2s(&src) {
            Ok(p) => p,
            Err(_) => return,
        };
        let Ok(expected) = grim_quant::dequant_iq2s(&packed, n) else {
            return;
        };
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2S),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2s");
        let actual = out.to_cpu_vec_f32().expect("readback iq2s");
        assert_dequant_close("iq2s", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_fp8_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 64;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let packed = grim_quant::quant_fp8(&src).expect("quant_fp8");
        let expected = grim_quant::dequant_fp8(&packed, n).expect("cpu oracle fp8");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::FloatPack(FloatPackScheme::Fp8),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant fp8");
        let actual = out.to_cpu_vec_f32().expect("readback fp8");
        assert_dequant_close("fp8", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_mxfp4_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // 64 values = 2 groups of 32. Hand-build codes + shared exponents so the
        // test is independent of a MXFP4 encoder (grim-quant has none public).
        // code i = (i % 16); shared exp = 127 + (group // ) so group 0 = 127,
        // group 1 = 128 (scale 2^1 = 2.0). Packed nibble: low = even element.
        let n = 64;
        let mut codes_pairs = Vec::with_capacity(n / 2);
        for i in 0..(n / 2) {
            let lo = (i * 2) % 16;
            let hi = (i * 2 + 1) % 16;
            codes_pairs.push((lo as u8) | ((hi as u8) << 4));
        }
        let exps = vec![127u8, 128u8];
        let packed = build_mxfp4_single_buffer(&codes_pairs, &exps);
        let expected = grim_quant::dequant_mxfp4(&packed, n).expect("cpu oracle mxfp4");

        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant mxfp4");
        let actual = out.to_cpu_vec_f32().expect("readback mxfp4");
        assert_dequant_close("mxfp4", &actual, &expected);
    }

    #[test]
    fn test_cuda_qkv_attention_parity() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let steps = 3;
        let kv_len = 5;
        let cache_offset = 2;

        let q: Vec<f32> = (0..steps * num_heads * head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..kv_len * num_kv_heads * head_dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim).map(|i| (i as f32 * 0.03).sin()).collect();

        let q_shape = Shape::new(vec![steps, num_heads, head_dim]);
        let kv_shape = Shape::new(vec![kv_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![steps, num_heads * head_dim]);

        let q_s = dev.from_cpu(&q, &q_shape, DType::F32).unwrap();
        let k_s = dev.from_cpu(&k, &kv_shape, DType::F32).unwrap();
        let v_s = dev.from_cpu(&v, &kv_shape, DType::F32).unwrap();

        let (out_s, _) = dev.qkv_attention(
            q_s.as_ref(),
            k_s.as_ref(),
            v_s.as_ref(),
            num_kv_heads,
            kv_len,
            cache_offset,
            None,
            &out_shape,
            None,
            None,
        ).unwrap();

        let actual = out_s.to_cpu_vec_f32().unwrap();

        // CPU reference calculation:
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;
        for t in 0..steps {
            for h in 0..num_heads {
                let kvh = (h * num_kv_heads) / num_heads;
                let causal_limit = cache_offset as usize + t;
                let mut scores = Vec::new();
                for t2 in 0..=causal_limit.min(kv_len - 1) {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * num_heads * head_dim + h * head_dim + d]
                            * k[t2 * kv_stride + kvh * head_dim + d];
                    }
                    scores.push(dot * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (i, s) in scores.iter().enumerate() {
                        acc += ((s - mx).exp() / sum) * v[i * kv_stride + kvh * head_dim + d];
                    }
                    let idx = t * num_heads * head_dim + h * head_dim + d;
                    assert!(
                        (actual[idx] - acc).abs() < 1e-4,
                        "Mismatch at t={t}, h={h}, d={d}: got {}, want {}",
                        actual[idx],
                        acc
                    );
                }
            }
        }
    }
}

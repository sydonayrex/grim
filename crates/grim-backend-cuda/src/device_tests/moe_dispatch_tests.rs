use super::common::*;

#[cfg(test)]
mod tests {
    use crate::memory::storage::CudaStorage;
    use grim_tensor::dtype::{ DType };
    use super::*;
    use grim_tensor::{ CoreTensorOps, Shape };

    #[test]
    fn test_cuda_moe_fused_dispatch_parity() {
        let Some(dev) = dequant_test_device() else {
            return;
        };

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
}

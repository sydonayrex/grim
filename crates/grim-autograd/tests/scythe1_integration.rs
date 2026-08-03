use grim_autograd::{
    AddArgs, MatMulArgs, ScaleArgs, Scythe1Adapter, Scythe1Optimizer, Tape, TensorId,
    cross_entropy_loss, lora_backward, matmul_backward, scale_backward,
};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

/// Synthetic "small problem" for SCYTHE1 integration test:
/// minimize MSE between adapter output and target over 20 steps.
#[test]
fn test_scythe1_integration_loss_decrease() {
    let d_in = 8;
    let d_out = 8;
    let r = 4;
    let mut adapter = Scythe1Adapter::new(d_out, d_in, r, 1.0).unwrap();
    let mut opt = Scythe1Optimizer::new(0.05, 0.05, 0.0, r);

    let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
    let target = cpu_tensor(vec![1.0f32; d_out], Shape::new(vec![1, d_out]));

    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for step in 0..20 {
        let y = adapter.forward(&x).unwrap();
        let y_vec = y.to_vec_f32().unwrap();
        let t_vec = target.to_vec_f32().unwrap();

        let mut loss = 0.0f32;
        for i in 0..d_out {
            let diff = y_vec[i] - t_vec[i];
            loss += diff * diff;
        }
        if step == 0 {
            initial_loss = Some(loss);
        }
        final_loss = loss;

        let x_vec = x.to_vec_f32().unwrap();
        let u_vec = adapter.inner.u.to_vec_f32().unwrap();
        let v_vec = adapter.inner.v.to_vec_f32().unwrap();
        let sig_vec = adapter.inner.sigma.to_vec_f32().unwrap();
        let scale = adapter.inner.scale;

        let mut x_v = vec![0.0f32; r];
        for k in 0..r {
            let mut sum = 0.0f32;
            for i in 0..d_in {
                sum += x_vec[i] * v_vec[i * r + k];
            }
            x_v[k] = sum;
        }

        let mut g_u = vec![0.0f32; d_out * r];
        for j in 0..d_out {
            for k in 0..r {
                g_u[j * r + k] = 2.0 * (y_vec[j] - t_vec[j]) * scale * x_v[k] * sig_vec[k];
            }
        }

        let mut g_sigma = vec![0.0f32; r];
        for k in 0..r {
            let mut sum = 0.0f32;
            for j in 0..d_out {
                sum += 2.0 * (y_vec[j] - t_vec[j]) * scale * x_v[k] * u_vec[j * r + k];
            }
            g_sigma[k] = sum;
        }

        let mut g_v = vec![0.0f32; d_in * r];
        for i in 0..d_in {
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += 2.0 * (y_vec[j] - t_vec[j]) * scale * sig_vec[k] * u_vec[j * r + k];
                }
                g_v[i * r + k] = x_vec[i] * sum;
            }
        }

        adapter.accumulate_fim(&g_u, &g_v, &g_sigma);
        opt.step(
            "layer0",
            &mut adapter.inner.u,
            &mut adapter.inner.v,
            &mut adapter.inner.sigma,
            &g_u,
            &g_v,
            &g_sigma,
        )
        .unwrap();
    }

    let initial_loss = initial_loss.unwrap();
    assert!(
        final_loss <= initial_loss,
        "Loss must decrease after 20 steps: initial {initial_loss}, final {final_loss}"
    );
}

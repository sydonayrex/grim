use grim_backend_cpu::cpu_tensor;
use grim_nn::RmsNorm;
use grim_tensor::Shape;

fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

fn rmsnorm_forward(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| v / rms * weight[i])
        .collect()
}

/// Input x = [0.5, -1.0, 2.0, -0.5], weight = [1.0, 2.0, 0.5, 1.5]
///
/// rms = sqrt((0.25+1+4+0.25)/4 + 1e-5)
///     = sqrt(5.5/4 + 1e-5) = sqrt(1.375) = 1.17260394...
///
/// output_i = x_i / rms * weight_i
///   [0] =  0.5 / 1.17260394 * 1.0   = 0.42640
///   [1] = -1.0 / 1.17260394 * 2.0   = -1.70560
///   [2] =  2.0 / 1.17260394 * 0.5   = 0.85280
///   [3] = -0.5 / 1.17260394 * 1.5   = -0.63960
#[test]
fn golden_bert_layernorm_vs_grim_nn_rmsnorm_exact_values() {
    let x = vec![0.5, -1.0, 2.0, -0.5];
    let w = vec![1.0, 2.0, 0.5, 1.5];
    let eps = 1e-5;
    let d = x.len();

    let expected = rmsnorm_forward(&x, &w, eps);

    let rms = RmsNorm {
        weight: cpu_tensor(w, Shape::new(vec![d])),
        eps,
    };
    let input_t = cpu_tensor(x, Shape::new(vec![1, d]));
    let output_t = rms.forward(&input_t).unwrap();
    let got = output_t.to_vec_f32().unwrap();

    for i in 0..d {
        close(got[i], expected[i], &format!("rmsnorm[{i}]"));
    }
}

#[test]
fn golden_bert_layernorm_rejects_wrong_shape() {
    let x = vec![0.5, -1.0, 2.0, -0.5];
    let w = vec![1.0, 2.0, 0.5, 1.5, 0.0, 0.0, 0.0, 0.0];

    let rms = RmsNorm {
        weight: cpu_tensor(w, Shape::new(vec![8])),
        eps: 1e-5,
    };
    let input_t = cpu_tensor(x, Shape::new(vec![1, 4]));
    let result = rms.forward(&input_t);

    assert!(result.is_err(), "mismatched weight/shape should error");
}

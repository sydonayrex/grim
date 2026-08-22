//! Numerical accuracy and parity tests for LOMO, AdaLomo, CAME, Sophia, and GaLore optimizers.

use grim_autograd::{
    AdaLomo, AdaLomoConfig, Came, CameConfig, GaLoreConfig, GaLoreOptimizer, LoRAInjectionPoint,
    Lomo, LomoConfig, ParamId, Sophia, SophiaConfig, TrainableParam, TrainableParams,
};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

#[test]
fn test_lomo_and_adalomo_numerical_accuracy() {
    let mut params = TrainableParams::new();
    let id0 = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);
    let id1 = ParamId::new(1, 0, LoRAInjectionPoint::VProj, true);

    let t0 = cpu_tensor(vec![2.0, -3.0, 4.0], Shape::new(vec![3]));
    let t1 = cpu_tensor(vec![1.0, 1.0, 1.0], Shape::new(vec![3]));

    params.insert(TrainableParam::new(id0, t0).unwrap());
    params.insert(TrainableParam::new(id1, t1).unwrap());

    let g0 = cpu_tensor(vec![0.1, -0.2, 0.3], Shape::new(vec![3]));
    let g1 = cpu_tensor(vec![0.5, 0.5, 0.5], Shape::new(vec![3]));

    params.get_mut(id0).unwrap().accumulate_grad(&g0).unwrap();
    params.get_mut(id1).unwrap().accumulate_grad(&g1).unwrap();

    // 1. LOMO Step
    let mut lomo = Lomo::new(LomoConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
        clip_grad_norm: None,
    });

    lomo.step(&mut params).unwrap();

    let p0 = params.get(id0).unwrap().data.to_vec_f32().unwrap();
    // delta = 0.1 * [0.1, -0.2, 0.3] -> p0 = [2.0 - 0.01, -3.0 + 0.02, 4.0 - 0.03]
    assert!((p0[0] - 1.99).abs() < 1e-5);
    assert!((p0[1] - (-2.98)).abs() < 1e-5);
    assert!((p0[2] - 3.97).abs() < 1e-5);

    // 2. AdaLomo Step
    let mut adalomo = AdaLomo::new(AdaLomoConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        clip_grad_norm: None,
    });

    params.get_mut(id0).unwrap().accumulate_grad(&g0).unwrap();
    params.get_mut(id1).unwrap().accumulate_grad(&g1).unwrap();

    adalomo.step(&mut params).unwrap();
    let p0_post = params.get(id0).unwrap().data.to_vec_f32().unwrap();
    assert!(
        p0_post[0] < p0[0],
        "p0[0] should continue decreasing with positive gradient"
    );
}

#[test]
fn test_came_factored_matrix_numerical_accuracy() {
    let mut came = Came::new(CameConfig {
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        beta3: 0.9999,
        eps1: 1e-16,
        eps2: 1e-16,
        weight_decay: 0.0,
        clip_threshold: 1.0,
    });

    let mut param = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

    came.update_matrix(id, &mut param, &grad, 2, 4).unwrap();

    assert!(came.matrix_states.contains_key(&id));
    let state = came.matrix_states.get(&id).unwrap();
    assert_eq!(state.exp_avg_sq_row.len(), 2);
    assert_eq!(state.exp_avg_sq_col.len(), 4);

    for (orig, &updated) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .zip(param.iter())
    {
        assert!(
            updated < *orig,
            "parameter should decrease from {orig}, got {updated}"
        );
    }
}

#[test]
fn test_sophia_second_order_clipping_and_damping() {
    let mut sophia = Sophia::new(SophiaConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.99,
        rho: 0.02,
        gamma: 1e-2,
        weight_decay: 0.0,
        hessian_update_interval: 5,
    });

    let mut param = vec![5.0, -5.0];
    let grad = vec![100.0, -100.0];
    let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

    sophia.update_hessian(id, &[10.0, 10.0]).unwrap();
    sophia.update_param(id, &mut param, &grad).unwrap();

    assert!((param[0] - (5.0 - 0.002)).abs() < 1e-4);
    assert!((param[1] - (-5.0 + 0.002)).abs() < 1e-4);
}

#[test]
fn test_galore_subspace_low_rank_projection_accuracy() {
    let mut galore = GaLoreOptimizer::new(GaLoreConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        rank: 2,
        update_proj_gap: 50,
        scale: 1.0,
    });

    let mut param = vec![1.0; 6 * 4];
    let grad = vec![0.2; 6 * 4];
    let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

    galore.update_matrix(id, &mut param, &grad, 6, 4).unwrap();

    assert!(galore.matrix_states.contains_key(&id));
    let state = galore.matrix_states.get(&id).unwrap();
    assert_eq!(state.proj_p.len(), 6 * 2);
    assert_eq!(state.exp_avg.len(), 2 * 4);

    for &val in &param {
        assert!(val < 1.0, "param should be updated downwards, got {val}");
    }
}

//! CPU-only tests for astrum P4 (Q4 KV + fixed Hadamard-256 rotation plumbing/math).
use grim_kvquant::{apply_rotation, fwht_inplace, hadamard_matrix, named_profile, NAMED_PROFILES};

#[test]
fn fwht256_orthonormal_self_inverse() {
    let dim = 256usize;
    let h = hadamard_matrix(dim);
    assert_eq!(h.len(), dim * dim);
    // H is symmetric and orthonormal => H * H == I
    let mut eye = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                sum += h[i * dim + k] * h[k * dim + j];
            }
            eye[i * dim + j] = sum;
        }
    }
    for i in 0..dim {
        for j in 0..dim {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (eye[i * dim + j] - want).abs() < 1e-4,
                "H*H not identity at ({i},{j}): {}",
                eye[i * dim + j]
            );
        }
    }
}

#[test]
fn fwht_inplace_preserves_norm_orthonormal_via_matrix() {
    // apply_rotation with orthonormal hadamard preserves L2 norm
    let dim = 256usize;
    let h = hadamard_matrix(dim);
    let data: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.17).sin()).collect();
    let rot = apply_rotation(&data, &h, dim, 1);
    let norm_in: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_out: f32 = rot.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_in - norm_out).abs() < 1e-4, "norm {norm_in} -> {norm_out}");
}

#[test]
fn fwht_known_butterfly() {
    let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
    fwht_inplace(&mut v);
    // FWHT([1,2,3,4]) with x+y/x-y butterfly = [10,-2,-4,0]? compute stepwise:
    // h=1: (1,2)->(3,-1),(3,4)->(7,-1) => [3,-1,7,-1]
    // h=2: (3,7)->(10,-4),(-1,-1)->(-2,0) => [10,-2,-4,0]
    assert_eq!(v, vec![10.0, -2.0, -4.0, 0.0]);
}

#[test]
fn q4_hadamard_profile_resolves() {
    let p = named_profile("q4_hadamard").expect("q4_hadamard profile");
    assert_eq!(p.key_bits, 4);
    assert_eq!(p.value_bits, 4);
    assert_eq!(p.group_size, 32);
    let p2 = named_profile("q4-hadamard").expect("alias");
    assert_eq!(p2.key_bits, 4);
    assert!(NAMED_PROFILES.contains(&"q4_hadamard"));
}
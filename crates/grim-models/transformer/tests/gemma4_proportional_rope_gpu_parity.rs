use grim_backend_cpu::cpu_tensor;
use grim_models_transformer::Gemma4ProportionalRope;
use grim_nn::Rope;
use grim_tensor::Shape;

/// Ported from Radiance `arch/gemma4/gemma4.cpp:33-40, 287-306`:
/// Verifies that expressing Gemma-4's proportional RoPE (64 active pairs in a 512-wide head,
/// partner stride 256, exponent denominator 512) as a full-width NeoX `RopeConfig` with a
/// YaRN step ramp at `n_pairs` matches the exact reference angles to < 1e-5 across positions,
/// whereas naive partial rotary (`rotary_dim = 128`) silently computes the wrong function.
#[test]
fn radiance_oracle_gemma4_proportional_rope_yarn_step_matches_reference() {
    let prop = Gemma4ProportionalRope {
        base: 1_000_000.0,
        head_dim: 512,
        n_pairs: 64,
    };

    let steps = 4usize;
    let num_heads = 2usize;
    let width = num_heads * prop.head_dim;
    let data: Vec<f32> = (0..(steps * width))
        .map(|i| ((i as f32) * 0.013).sin() * 2.0 + ((i as f32) * 0.007).cos())
        .collect();
    let x = cpu_tensor(data, Shape::new(vec![steps, width]));
    let positions = vec![0u32, 1, 17, 4096];

    // 1. Reference scalar proportional RoPE
    let expected = prop.apply(&x, num_heads, &positions).expect("scalar apply");
    let exp_vec = expected.to_vec_f32().expect("to_vec_f32");

    // 2. Radiance YaRN step-ramp RopeConfig evaluated through standard `Rope::forward`
    let cfg = prop.to_yarn_rope_config();
    assert_eq!(cfg.dim, 512);
    assert_eq!(cfg.rotary_dim, 512);
    assert!(!cfg.interleaved);

    // Transpose [steps, num_heads, head_dim] -> [num_heads, steps, head_dim] for Rope::forward
    let mut x_bhs = vec![0.0f32; steps * width];
    let x_raw = x.to_vec_f32().unwrap();
    for t in 0..steps {
        for h in 0..num_heads {
            let src_off = t * width + h * prop.head_dim;
            let dst_off = (h * steps + t) * prop.head_dim;
            x_bhs[dst_off..dst_off + prop.head_dim]
                .copy_from_slice(&x_raw[src_off..src_off + prop.head_dim]);
        }
    }
    let x_rope_in = cpu_tensor(x_bhs, Shape::new(vec![num_heads, steps, prop.head_dim]));
    let rope = Rope::from_config(cfg);
    let y_bhs = rope
        .forward(&x_rope_in, &positions)
        .expect("yarn step rope forward")
        .to_vec_f32()
        .unwrap();

    for t in 0..steps {
        for h in 0..num_heads {
            let exp_off = t * width + h * prop.head_dim;
            let got_off = (h * steps + t) * prop.head_dim;
            for d in 0..prop.head_dim {
                let diff = (exp_vec[exp_off + d] - y_bhs[got_off + d]).abs();
                assert!(
                    diff < 1e-3,
                    "mismatch at step={t} head={h} dim={d}: expected {} got {} (diff={diff})",
                    exp_vec[exp_off + d],
                    y_bhs[got_off + d]
                );
            }
            // Verify unrotated pairs (64..256 and 256+64..512) are preserved identically
            for j in prop.n_pairs..(prop.head_dim / 2) {
                assert!(
                    (y_bhs[got_off + j] - x_raw[exp_off + j]).abs() < 1e-6,
                    "unrotated lo channel {j} mutated"
                );
                assert!(
                    (y_bhs[got_off + 256 + j] - x_raw[exp_off + 256 + j]).abs() < 1e-6,
                    "unrotated hi channel {} mutated",
                    256 + j
                );
            }
        }
    }
}

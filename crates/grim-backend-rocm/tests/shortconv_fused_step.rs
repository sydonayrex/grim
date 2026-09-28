//! PLAN-kernel-launch-reduction Phase C regression: `grim_short_conv1d_fused_step`
//! (reads b|x|c from the in_proj output, computes bx, convolves with in-place
//! state update, applies the c gate) must match the split sequence it replaces:
//! bx = b*x -> short_conv1d_causal_step -> y = sum * c, including the state ring.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device.

use grim_backend_rocm::{RocmDevice, RocmStorage, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, RecurrentOps, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn f32_tensor(
    dev: &RocmDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).unwrap()
}

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i * 7) % 113) as f32 / 113.0 - 0.5)
        .collect()
}

#[test]
#[ignore]
fn fused_shortconv_step_matches_split() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let channels = 1024usize;
    let ks = 3usize; // l_cache
    let proj: Vec<f32> = rand_f32(3 * channels, 1);
    let weight: Vec<f32> = rand_f32(channels * ks, 2);
    let state: Vec<f32> = rand_f32(channels * (ks - 1), 3);

    let proj_s = f32_tensor(&dev, &proj, &Shape::new(vec![3 * channels]));
    let w_s = f32_tensor(&dev, &weight, &Shape::new(vec![channels, ks]));

    // --- Split reference: bx = b*x; conv step; y = sum*c ---
    let b = &proj[..channels];
    let c = &proj[channels..2 * channels];
    let x = &proj[2 * channels..];
    let bx: Vec<f32> = b.iter().zip(x).map(|(a, b)| a * b).collect();
    let bx_s = f32_tensor(&dev, &bx, &Shape::new(vec![channels]));
    let state_ref = f32_tensor(&dev, &state, &Shape::new(vec![channels * (ks - 1)]));
    let sum_s = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
    dev.short_conv1d_causal_step_into(
        bx_s.as_ref(),
        w_s.as_ref(),
        None,
        state_ref.as_ref(),
        rocm(&sum_s),
    )
    .unwrap();
    let sum = sum_s.to_cpu_vec_f32().unwrap();
    let y_ref: Vec<f32> = sum.iter().zip(c).map(|(s, c)| s * c).collect();
    let state_ref_v = state_ref.to_cpu_vec_f32().unwrap();

    // --- Fused: one launch, in place on a fresh state copy ---
    let state_f = f32_tensor(&dev, &state, &Shape::new(vec![channels * (ks - 1)]));
    let y_f = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
    dev.short_conv1d_fused_step_into(
        proj_s.as_ref(),
        w_s.as_ref(),
        state_f.as_ref(),
        rocm(&y_f),
        1, // batch
        channels,
        ks,
    )
    .unwrap();

    let got_y = y_f.to_cpu_vec_f32().unwrap();
    let got_state = state_f.to_cpu_vec_f32().unwrap();
    for i in 0..channels {
        assert!(
            (got_y[i] - y_ref[i]).abs() <= 1e-5,
            "y[{i}]: {} vs {}",
            got_y[i],
            y_ref[i]
        );
    }
    for i in 0..state_ref_v.len() {
        assert!(
            (got_state[i] - state_ref_v[i]).abs() <= 1e-6,
            "state[{i}]: {} vs {}",
            got_state[i],
            state_ref_v[i]
        );
    }
}

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a RocmStorage {
    as_rocm(s.as_ref()).unwrap()
}

// ── THE PRODUCTION CHAIN: scan (prefill) then step (decode) ────────────────
// Same methodology as the KDA chain gate: every per-kernel conv gate was
// green while eager device-KDA decode diverged after one token. The untested
// shape is the chain — the 5-token prefill runs `grim_short_conv1d_scan`
// against the device ring, then every decode step runs
// `grim_short_conv1d_causal_step` against the ring THE SCAN LEFT. 9B dims:
// channels 8192, taps 4.
/// KDA-fix regression gate: the PRODUCTION conv chain — prefill runs the
/// scan (batch > 1) against the device ring, then every decode step runs the
/// step kernel against the ring the scan left. Two defects hid here, both
/// invisible to per-kernel gates:
/// 1. `short_conv1d_causal_step_into` had NO scan dispatch: for batch > 1 it
///    ran the per-(b, c) step kernel whose ring offset is
///    `(b*channels + c)*(ks-1)`, so threads with b > 0 indexed past the
///    single-stream ring — token 0 exact, token 1+ garbage;
/// 2. the scan's global in-place shift miscompiled on the hipRTC gfx1201
///    build; the kernel now stages the ring in dynamic LDS.
/// The gate exercises the NON-into variant (the production prefill entry)
/// for the prefill leg and the _into variant (batch 1) for the decode leg.
#[test]
fn conv_scan_then_step_chain_matches_step_only() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let channels = 8192usize;
    let ks = 4usize;
    let seq = 6usize;
    let x: Vec<f32> = rand_f32(seq * channels, 11);
    let weight: Vec<f32> = rand_f32(channels * ks, 12);
    let state0: Vec<f32> = rand_f32(channels * (ks - 1), 13);
    let w_s = f32_tensor(&dev, &weight, &Shape::new(vec![channels, ks]));

    // CPU reference: y[t][c] = sum_k ring[t][k]*w[c][k]; ring shifts per token.
    let mut ring = state0.clone();
    let mut cpu_out = vec![0.0f32; seq * channels];
    for t in 0..seq {
        for c in 0..channels {
            let off = c * (ks - 1);
            let mut sum = x[t * channels + c] * weight[c * ks + ks - 1];
            for k in 0..ks - 1 {
                sum += ring[off + k] * weight[c * ks + k];
            }
            cpu_out[t * channels + c] = sum;
        }
        for c in 0..channels {
            let off = c * (ks - 1);
            for k in 0..ks - 2 {
                ring[off + k] = ring[off + k + 1];
            }
            ring[off + ks - 2] = x[t * channels + c];
        }
    }

    // Arm B: six single-token steps (decode-only chain), state held on device.
    let state_b = f32_tensor(&dev, &state0, &Shape::new(vec![channels * (ks - 1)]));
    let mut b = vec![0.0f32; channels];
    for t in 0..seq {
        let x_s = f32_tensor(&dev, &x[t * channels..(t + 1) * channels], &Shape::new(vec![channels]));
        let out_b = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
        dev.synchronize(); // x_s upload must land before the kernel reads it
        dev.short_conv1d_causal_step_into(
            x_s.as_ref(),
            w_s.as_ref(),
            None,
            state_b.as_ref(),
            rocm(&out_b),
        )
        .unwrap();
        dev.synchronize();
        b.copy_from_slice(&out_b.to_cpu_vec_f32().unwrap());
        if t == seq - 1 {
            let worst = b
                .iter()
                .zip(&cpu_out[t * channels..])
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-2, "step-only chain diverges from CPU: {worst}");
        }
    }

    // seq_len bisect: scan with batch=2, does token 0 already diverge?
    {
        for sl in [2usize, 3, 5] {
            let st = f32_tensor(&dev, &state0, &Shape::new(vec![channels * (ks - 1)]));
            let xs = f32_tensor(&dev, &x[..sl * channels], &Shape::new(vec![sl, channels]));
            dev.synchronize(); // uploads must land before the kernel reads them
            let (o_s, _h) = dev
                .short_conv1d_causal_step(xs.as_ref(), w_s.as_ref(), None, st.as_ref(), &Shape::new(vec![sl, channels]))
                .unwrap();
            dev.synchronize();
            let ov = o_s.to_cpu_vec_f32().unwrap();
            let per_t: Vec<String> = (0..sl)
                .map(|t| {
                    format!(
                        "t{t}={:.5}",
                        ov[t * channels..(t + 1) * channels]
                            .iter()
                            .zip(&cpu_out[t * channels..(t + 1) * channels])
                            .map(|(g, w)| (g - w).abs())
                            .fold(0.0f32, f32::max)
                    )
                })
                .collect();
            eprintln!("[conv-chain-bisect] seq_len={sl}: {}", per_t.join(" "));
            if sl == 2 {
                // Ring forensics: CPU ring after 2 tokens vs the scan's ring.
                let mut cpu_ring = state0.clone();
                for t in 0..2 {
                    for c in 0..channels {
                        let off = c * (ks - 1);
                        for k in 0..ks - 2 {
                            cpu_ring[off + k] = cpu_ring[off + k + 1];
                        }
                        cpu_ring[off + ks - 2] = x[t * channels + c];
                    }
                }
                let sv = st.to_cpu_vec_f32().unwrap();
                let mut per_c_worst = 0.0f32;
                let mut worst_c = 0usize;
                for c in 0..channels {
                    let off = c * (ks - 1);
                    let w: f32 = (0..ks - 1)
                        .map(|k| (sv[off + k] - cpu_ring[off + k]).abs())
                        .fold(0.0f32, f32::max);
                    if w > per_c_worst {
                        per_c_worst = w;
                        worst_c = c;
                    }
                }
                eprintln!(
                    "[conv-chain-bisect] ring after 2: worst {per_c_worst:.5} at c={worst_c};                      scan ring c0={:?} cpu ring c0={:?} scan c1={:?} cpu c1={:?}",
                    &sv[0..3],
                    &cpu_ring[0..3],
                    &sv[(ks - 1)..(ks - 1) + 3],
                    &cpu_ring[(ks - 1)..(ks - 1) + 3]
                );
            }
        }
    }

    // Arm A: one 5-token scan (prefill) + 1 step (decode) on a fresh ring.
    let state_a = f32_tensor(&dev, &state0, &Shape::new(vec![channels * (ks - 1)]));
    let x5 = f32_tensor(&dev, &x[..5 * channels], &Shape::new(vec![5, channels]));
    let _out5 = f32_tensor(&dev, &vec![0.0f32; 5 * channels], &Shape::new(vec![5, channels]));
    dev.synchronize(); // uploads must land before the kernel reads them
    let (o5, _h5) = dev
        .short_conv1d_causal_step(x5.as_ref(), w_s.as_ref(), None, state_a.as_ref(), &Shape::new(vec![5, channels]))
        .unwrap();
    dev.synchronize();
    let out5_v = o5.to_cpu_vec_f32().unwrap();
    for t in 0..5 {
        let worst = out5_v[t * channels..(t + 1) * channels]
            .iter()
            .zip(&cpu_out[t * channels..(t + 1) * channels])
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        if t == 0 {
            // Pattern forensics: print, for the first few channels, the scan
            // output, the CPU output, and what the scan WOULD produce under
            // candidate mis-indexings (transposed weight, x-stride error).
            for c in 0..16usize {
                // Is scan[c] equal to cpu[c'] for some small shift?
                let got = out5_v[c];
                let mut shift: Option<i32> = None;
                for (cp, &w) in cpu_out.iter().enumerate().take(32) {
                    if (w - got).abs() < 2e-4 {
                        shift = Some(cp as i32 - c as i32);
                        break;
                    }
                }
                eprintln!("[conv-chain-forensics] c={c}: scan={got:.5} cpu={:.5} match_shift={:?}",
                    cpu_out[c], shift);
            }
        }
        eprintln!("[conv-chain] scan token {t}: worst {worst:.5}");
    }
    let state_a_v = state_a.to_cpu_vec_f32().unwrap();
    let ring_v = f32_tensor(&dev, &ring, &Shape::new(vec![channels * (ks - 1)]))
        .to_cpu_vec_f32()
        .unwrap();
    let ring_worst = state_a_v
        .iter()
        .zip(&ring_v)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[conv-chain] ring after scan vs CPU ring: worst {ring_worst:.5}");
    let x6 = f32_tensor(&dev, &x[5 * channels..], &Shape::new(vec![channels]));
    let _out6 = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
    dev.synchronize(); // x6 upload must land before the kernel reads it
    let (o6, _h6) = dev
        .short_conv1d_causal_step(x6.as_ref(), w_s.as_ref(), None, state_a.as_ref(), &Shape::new(vec![channels]))
        .unwrap();
    dev.synchronize();
    let a = o6.to_cpu_vec_f32().unwrap();
    let worst = a
        .iter()
        .zip(&cpu_out[5 * channels..])
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[conv-chain] scan(5)+step(1) vs CPU: worst {worst:.5}");
    assert!(
        worst < 1e-2,
        "conv scan->step chain diverges from CPU ({worst}) while step-only matches — \
         the scan leaves the ring in a form the decode step misreads"
    );
}

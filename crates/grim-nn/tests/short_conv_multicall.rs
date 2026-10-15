//! Short-conv CPU state-layout regression gates.
//!
//! The conv ring is channel-major flat (`tap k of channel h` at `h*(K-1)+k`)
//! across the GPU kernel, the graph `sc_state` ring, the `short_conv1d` host
//! fallback, and `CpuDevice::short_conv1d_causal_step` alike. These tests pin
//! that contract: multi-call CPU decode must match single-shot, the fallback
//! must preserve the caller's state shape across calls, and the direct
//! `RecurrentOps` impl must agree with the fallback on nonzero state.
//! RUN: cargo test -p grim-nn --test short_conv_multicall
use grim_nn::modules::short_conv1d;
use grim_tensor::{Shape, Tensor};

fn cpu_tensor(v: Vec<f32>, dims: Vec<usize>) -> Tensor {
    grim_backend_cpu::cpu_tensor(v, Shape::new(dims))
}

#[test]
fn cpu_conv_multicall_matches_single_shot() {
    let d = 8usize;
    let k = 4usize;
    let b = 1usize;
    // weight [D, K] row-major
    let wv: Vec<f32> = (0..d * k).map(|i| ((i * 3 % 7) as f32) / 7.0 - 0.5).collect();
    let w = cpu_tensor(wv.clone(), vec![d, k]);
    // 5-token sequence
    let n = 5usize;
    let xv: Vec<f32> = (0..n * d).map(|i| ((i * 5 % 11) as f32) / 11.0 - 0.3).collect();

    // Reference: single s=5 call, no state.
    let x_all = cpu_tensor(xv.clone(), vec![b, n, d]);
    let out_ref = short_conv1d(&x_all, &w, None, None).expect("ref").to_vec_f32().expect("ref vec");

    // Multi-call qwen35-style: flat cache, state tensor rebuilt [1,K-1,D] each call.
    let need = (k - 1) * d;
    let mut cache = vec![0.0f32; need];
    let mut outs: Vec<Vec<f32>> = vec![];
    for t in 0..n {
        let xt = cpu_tensor(xv[t * d..(t + 1) * d].to_vec(), vec![b, 1, d]);
        let mut state_t = cpu_tensor(cache.clone(), vec![1, k - 1, d]);
        let o = short_conv1d(&xt, &w, None, Some(&mut state_t)).expect("step");
        outs.push(o.to_vec_f32().expect("o"));
        let updated = state_t.to_vec_f32().expect("upd");
        assert_eq!(updated.len(), need, "t={t}: state len changed");
        eprintln!("[t={t}] state shape after = {:?}", state_t.shape().dims());
        cache.copy_from_slice(&updated[..need]);
    }
    let mut maxd = 0.0f32;
    for t in 0..n {
        for i in 0..d {
            let e = (outs[t][i] - out_ref[t * d + i]).abs();
            maxd = maxd.max(e);
            if e > 1e-5 {
                eprintln!("[MISMATCH] t={t} i={i}: multi={} ref={}", outs[t][i], out_ref[t * d + i]);
            }
        }
    }
    eprintln!("[cpu-multicall] maxd={maxd:.6}");
    assert!(maxd < 1e-5, "multi-call CPU decode diverges: {maxd}");
}

#[test]
fn cpu_device_impl_agrees_with_fallback() {
    // Direct RecurrentOps call on CpuDevice must read state with the SAME
    // layout as the GPU kernel (channel-major h*S+k) and the modules fallback.
    use grim_tensor::RecurrentOps;
    let dev = grim_backend_cpu::CpuDevice::new();
    let d = 8usize;
    let k = 4usize;
    let wv: Vec<f32> = (0..d * k).map(|i| ((i * 3 % 7) as f32) / 7.0 - 0.5).collect();
    let w = cpu_tensor(wv.clone(), vec![d, k]);
    // Nonzero channel-major state: tap t of channel h holds (h*10+t).
    let mut stv = vec![0.0f32; (k - 1) * d];
    for h in 0..d {
        for t in 0..k - 1 {
            stv[h * (k - 1) + t] = (h * 10 + t) as f32 * 0.01;
        }
    }
    let xv: Vec<f32> = (0..d).map(|i| i as f32 * 0.03 - 0.1).collect();
    let x = cpu_tensor(xv.clone(), vec![1, 1, d]);
    let st = cpu_tensor(stv.clone(), vec![1, k - 1, d]);
    let out_shape = Shape::new(vec![1, 1, d]);
    let (out_s, h) = dev
        .short_conv1d_causal_step(x.storage().as_ref(), w.storage().as_ref(), None, st.storage().as_ref(), &out_shape)
        .expect("cpu step");
    h.synchronize().expect("sync");
    let got = grim_backend_cpu::cpu_tensor(out_s.to_cpu_vec_f32().expect("vec"), out_shape).to_vec_f32().expect("vec2");
    // Fallback reference with identical inputs.
    let mut st2 = cpu_tensor(stv.clone(), vec![1, k - 1, d]);
    let want = short_conv1d(&x, &w, None, Some(&mut st2)).expect("fb").to_vec_f32().expect("fbv");
    let mut maxd = 0.0f32;
    for (g, wn) in got.iter().zip(want.iter()) {
        maxd = maxd.max((g - wn).abs());
    }
    eprintln!("[cpu-impl-vs-fallback] maxd={maxd:.6}");
    assert!(maxd < 1e-6, "CpuDevice impl disagrees with fallback layout: {maxd}");
}

#[test]
fn cpu_conv_persistent_state_multicall() {
    // KdaLayerCache-style: same Tensor reused across calls (not rebuilt).
    let d = 8usize;
    let k = 4usize;
    let wv: Vec<f32> = (0..d * k).map(|i| ((i * 3 % 7) as f32) / 7.0 - 0.5).collect();
    let w = cpu_tensor(wv.clone(), vec![d, k]);
    let n = 5usize;
    let xv: Vec<f32> = (0..n * d).map(|i| ((i * 5 % 11) as f32) / 11.0 - 0.3).collect();
    let x_all = cpu_tensor(xv.clone(), vec![1, n, d]);
    let out_ref = short_conv1d(&x_all, &w, None, None).expect("ref").to_vec_f32().expect("ref vec");

    let mut state = cpu_tensor(vec![0.0f32; (k - 1) * d], vec![1, k - 1, d]);
    let mut maxd = 0.0f32;
    for t in 0..n {
        let xt = cpu_tensor(xv[t * d..(t + 1) * d].to_vec(), vec![1, 1, d]);
        let o = short_conv1d(&xt, &w, None, Some(&mut state)).expect("step");
        let ov = o.to_vec_f32().expect("o");
        for i in 0..d {
            maxd = maxd.max((ov[i] - out_ref[t * d + i]).abs());
        }
        eprintln!("[persist t={t}] state shape = {:?}", state.shape().dims());
    }
    eprintln!("[cpu-persist] maxd={maxd:.6}");
    assert!(maxd < 1e-5, "persistent-state multi-call diverges: {maxd}");
}

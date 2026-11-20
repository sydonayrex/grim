use grim_models_transformer::qwen4exp_gdn::{
    gated_delta_rule_row, gated_delta_rule_wy_chunk_64, gdn_wy_chunk_enabled,
};

#[test]
fn gdn_wy_chunk_matches_sequential_oracle() {
    let c = 16;
    let d = 8;
    let mut k_chunk = vec![0.0f32; c * d];
    let mut v_chunk = vec![0.0f32; c * d];
    let mut beta_chunk = vec![0.0f32; c];
    let mut gate_chunk = vec![0.0f32; c];
    for t in 0..c {
        beta_chunk[t] = 0.2 + 0.03 * (t as f32);
        gate_chunk[t] = -0.05 * ((t % 4 + 1) as f32);
        for i in 0..d {
            k_chunk[t * d + i] = ((t * 7 + i * 3) as f32).sin() * 0.25;
            v_chunk[t * d + i] = ((t * 5 + i * 11) as f32).cos() * 0.5;
        }
    }

    let mut state_seq = vec![0.0f32; d * d];
    let mut out_seq = vec![0.0f32; c * d];
    for t in 0..c {
        let k = &k_chunk[t * d..(t + 1) * d];
        let v = &v_chunk[t * d..(t + 1) * d];
        gated_delta_rule_row(
            k,
            v,
            beta_chunk[t],
            gate_chunk[t],
            &mut state_seq,
            d,
            d,
        );
        for j in 0..d {
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += state_seq[j * d + i] * k[i];
            }
            out_seq[t * d + j] = acc;
        }
    }

    let mut state_wy = vec![0.0f32; d * d];
    let mut out_wy = vec![0.0f32; c * d];
    gated_delta_rule_wy_chunk_64(
        &k_chunk,
        &v_chunk,
        &beta_chunk,
        &gate_chunk,
        &mut state_wy,
        d,
        d,
        &mut out_wy,
    );

    for (a, b) in state_seq.iter().zip(state_wy.iter()) {
        assert!((a - b).abs() < 1e-5, "state mismatch: {a} vs {b}");
    }
    for (a, b) in out_seq.iter().zip(out_wy.iter()) {
        assert!((a - b).abs() < 1e-5, "out mismatch: {a} vs {b}");
    }
}

#[test]
fn gdn_wy_chunk_defaults_off() {
    let prev = std::env::var("GRIM_GDN_WY_CHUNK").ok();
    unsafe { std::env::remove_var("GRIM_GDN_WY_CHUNK") };
    assert!(!gdn_wy_chunk_enabled());
    if let Some(v) = prev {
        unsafe { std::env::set_var("GRIM_GDN_WY_CHUNK", v) };
    }
}

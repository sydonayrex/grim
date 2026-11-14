use grim_memory::DualRecurrentSlot;

/// Ported from Radiance `tests/ref_test.cpp:5253-5287` (`op_gdn_recurrent_update`):
/// Exact single-token Gated DeltaNet recurrent step over state S [V, K]:
/// S <- exp(g) * S + beta * (v - (exp(g) * S) k) k^T, o_raw = S q.
fn gdn_step_in_place(state: &mut [f32], q: &[f32], k_norm: &[f32], v: &[f32], g: f32, beta: f32) -> Vec<f32> {
    let k_dim = k_norm.len();
    let v_dim = v.len();
    assert_eq!(state.len(), v_dim * k_dim);
    let decay = g.exp();
    let mut o_raw = vec![0.0f32; v_dim];
    for e in 0..v_dim {
        let row = &mut state[e * k_dim..(e + 1) * k_dim];
        let mut dot = 0.0f32;
        for d in 0..k_dim {
            row[d] *= decay;
            dot += row[d] * k_norm[d];
        }
        let u = beta * (v[e] - dot);
        let mut out_e = 0.0f32;
        for d in 0..k_dim {
            row[d] += u * k_norm[d];
            out_e += row[d] * q[d];
        }
        o_raw[e] = out_e;
    }
    o_raw
}

#[test]
fn dual_recurrent_slot_matches_radiance_closed_form_and_speculative_schedule_oracle() {
    // 1. Ported from `ref_test.cpp:5253-5287` (`op_gdn_recurrent_update` closed-form paper check):
    // K=1, V=2, A_log=0, a=0 -> g = -ln(2), b=0 -> beta = 0.5, q=[1], k=[1], v=[3, 4], S_0=[0, 0]
    // -> S_1 = [1.5, 2.0], gated RMSNorm o = [1.5, 2.0] / sqrt(3.125) * 0.5.
    let mut slot0 = DualRecurrentSlot::new(2, 4, 3);
    assert_eq!(slot0.conv_window_len(), 6);
    let o_raw = gdn_step_in_place(
        slot0.draft_state_mut(),
        &[1.0],
        &[1.0],
        &[3.0, 4.0],
        -(2.0f32.ln()),
        0.5,
    );
    slot0.commit_draft();
    assert!((slot0.committed_state()[0] - 1.5).abs() < 1e-6);
    assert!((slot0.committed_state()[1] - 2.0).abs() < 1e-6);
    let sc = 1.0f32 / 3.125f32.sqrt();
    assert!((o_raw[0] * sc * 0.5 - 1.5 * sc * 0.5).abs() < 1e-6);
    assert!((o_raw[1] * sc * 0.5 - 2.0 * sc * 0.5).abs() < 1e-6);

    // 2. Ported from `ref_test.cpp:10907-11000` (`gdn_recurrent_anchor_replays_the_accepted_prefix`):
    // 4-step speculative verify schedule with `ql = 3` candidate tokens per step and variable
    // acceptances `nacc_sched = [1, 2, 3, 1]`. Verifies that `DualRecurrentSlot` (rollback rejected
    // draft tail + commit accepted prefix) matches an exact step-by-step ground-truth oracle bit-for-bit,
    // and that a broken schedule holding `nacc = 1` (`differ_one`) strictly diverges.
    let k_dim = 4usize;
    let v_dim = 4usize;
    let ql = 3usize;
    let nacc_sched = [1usize, 2, 3, 1];

    let init_state: Vec<f32> = (0..(v_dim * k_dim)).map(|i| (i as f32 - 7.0) * 0.05).collect();
    let mut dual = DualRecurrentSlot::new(v_dim * k_dim, 4, ql);
    let mut broken_one = DualRecurrentSlot::new(v_dim * k_dim, 4, ql);
    let mut oracle_state = init_state.clone();
    dual.seed_committed(&init_state);
    broken_one.seed_committed(&init_state);

    let mut differ_one = 0usize;
    for (step, &n_acc) in nacc_sched.iter().enumerate() {
        // Speculative step proposes `ql = 3` draft tokens on `draft_state_mut()`
        let mut step_candidates = Vec::with_capacity(ql);
        for tok in 0..ql {
            let base = (step * 17 + tok * 5 + 1) as f32;
            let q = [base * 0.1, -base * 0.05, 0.2, -0.1];
            let k_raw = [0.5f32, -0.5, 0.5, 0.5]; // unit L2 norm
            let v = [base * 0.2, -base * 0.1, base * 0.05, 0.3];
            let g = -0.15f32 * (tok as f32 + 1.0);
            let beta = 0.4f32;
            let _ = gdn_step_in_place(dual.draft_state_mut(), &q, &k_raw, &v, g, beta);
            step_candidates.push((q, k_raw, v, g, beta));
        }

        // If `n_acc < ql`, reject speculative tail via `rollback_draft()` and advance only `0..n_acc`,
        // then `commit_draft()`. When `n_acc == ql`, all draft tokens were accepted -> `commit_draft()` directly!
        if n_acc < ql {
            dual.rollback_draft();
            for &(ref q, ref k_raw, ref v, g, beta) in &step_candidates[..n_acc] {
                let _ = gdn_step_in_place(dual.draft_state_mut(), q, k_raw, v, g, beta);
            }
        }
        dual.commit_draft();

        // Ground-truth sequential oracle advances only the accepted tokens
        for &(ref q, ref k_raw, ref v, g, beta) in &step_candidates[..n_acc] {
            let _ = gdn_step_in_place(&mut oracle_state, q, k_raw, v, g, beta);
        }
        assert_eq!(
            dual.committed_state(),
            oracle_state.as_slice(),
            "DualRecurrentSlot diverged from sequential oracle at step {step}"
        );

        // Sensitivity check (`differ_one` from `ref_test.cpp:10987`): always accepting only 1 token must diverge
        broken_one.rollback_draft();
        let (ref q0, ref k0, ref v0, g0, beta0) = step_candidates[0];
        let _ = gdn_step_in_place(broken_one.draft_state_mut(), q0, k0, v0, g0, beta0);
        broken_one.commit_draft();
        if broken_one.committed_state() != oracle_state.as_slice() {
            differ_one += 1;
        }
    }
    assert!(differ_one > 0, "schedule oracle must be sensitive to acceptance count");
}

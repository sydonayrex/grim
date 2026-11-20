//! Phase 0.1 Parity Gate: compare Qwen35 M-row DecodeGraph replay (M = 2, 4)
//! against M sequential batch=1 eager forwards from identical post-prefill state.

use grim_core::model::{AdapterHandle, CausalLm};
use grim_models_transformer::qwen35::Qwen35;
use grim_nn::WeightSource;

include!("common/qwen35_synth.rs");

#[test]
fn qwen35_m_row_graph_matches_sequential_eager() {
    let Some((device, dev)) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DECODE_GRAPH_M2", "1");
    }

    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    cfg.max_seq_len = 256;
    let vocab = cfg.vocab_size;
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load tiny hybrid qwen35");

    for &m in &[2usize, 4usize] {
        // 1. Sequential eager baseline: prefill 2 tokens [3, 7] at pos [0, 1], then M tokens one-by-one.
        let mut eager_sess = model.new_session();
        let prefill_toks = [3u32, 7u32];
        for (step, &tok) in prefill_toks.iter().enumerate() {
            let ids = dev_tensor(dev.as_ref(), &device, &[tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
            let _ = CausalLm::forward(
                &model,
                eager_sess.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager prefill");
        }
        let step_toks: Vec<u32> = (0..m).map(|i| (11 + i * 3) as u32).collect();
        let mut eager_rows = Vec::with_capacity(m);
        for (i, &tok) in step_toks.iter().enumerate() {
            let ids = dev_tensor(dev.as_ref(), &device, &[tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[(prefill_toks.len() + i) as f32]);
            let out = CausalLm::forward(
                &model,
                eager_sess.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager step")
            .to_vec_f32()
            .expect("eager readback");
            eager_rows.push(out);
        }

        // 1b. One subsequent step after the M-token step to verify state continuity (KV + conv + KDA).
        let follow_tok = (29 + m) as u32;
        let follow_pos = (prefill_toks.len() + m) as u32;
        let eager_follow = {
            let ids = dev_tensor(dev.as_ref(), &device, &[follow_tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[follow_pos as f32]);
            CausalLm::forward(
                &model,
                eager_sess.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager follow step")
            .to_vec_f32()
            .expect("eager follow readback")
        };
        let after_follow_tok = (43 + m) as u32;
        let after_follow_pos = follow_pos + 1;
        let eager_after_follow = {
            let ids = dev_tensor(dev.as_ref(), &device, &[after_follow_tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[after_follow_pos as f32]);
            CausalLm::forward(
                &model,
                eager_sess.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager after_follow step")
            .to_vec_f32()
            .expect("eager after_follow readback")
        };

        // 2. Graph M-row side: prefill 2 tokens [3, 7] into a fresh session, then run_m2_graph_step for `step_toks`.
        let mut graph_sess = model.new_session();
        for (step, &tok) in prefill_toks.iter().enumerate() {
            let ids = dev_tensor(dev.as_ref(), &device, &[tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
            let _ = CausalLm::forward(
                &model,
                graph_sess.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("graph prefill");
        }
        let mut slot: Option<Box<dyn std::any::Any + Send>> = None;
        let graph_out = model
            .run_m2_graph_step(&mut slot, graph_sess.as_mut(), &step_toks, prefill_toks.len() as u32)
            .expect("run_m2_graph_step")
            .expect("Some(tensor)");
        let graph_flat = graph_out.to_vec_f32().expect("graph readback");
        assert_eq!(graph_flat.len(), m * vocab);

        let mut any_fail = false;
        for r in 0..m {
            let g_row = &graph_flat[r * vocab..(r + 1) * vocab];
            let e_row = &eager_rows[r];
            let max_diff = g_row
                .iter()
                .zip(e_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let g_argmax = g_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let e_argmax = e_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            eprintln!(
                "[m-row-parity] M={m} row={r}: max_diff={max_diff:.6} graph_argmax={g_argmax} (g={:.6}, e={:.6}) eager_argmax={e_argmax} (g={:.6}, e={:.6})",
                g_row[g_argmax], e_row[g_argmax], g_row[e_argmax], e_row[e_argmax]
            );
            if g_argmax != e_argmax || max_diff > 1e-2 {
                any_fail = true;
            }
        }
        assert!(!any_fail, "M={m} parity failed");

        // Verify a second M-row replay starting from `follow_pos` matches eager continuity.
        model
            .commit_graph_spec_state(&mut slot)
            .expect("commit_graph_spec_state");
        let follow_toks: Vec<u32> = (0..m).map(|i| follow_tok + (i as u32) * 2).collect();
        let graph_follow_out = model
            .run_m2_graph_step(&mut slot, graph_sess.as_mut(), &follow_toks, follow_pos)
            .expect("run_m2_graph_step follow")
            .expect("Some(tensor) follow")
            .to_vec_f32()
            .expect("graph follow readback");
        let g_follow_row0 = &graph_follow_out[0..vocab];
        let follow_diff = g_follow_row0
            .iter()
            .zip(eager_follow.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let gf_argmax = g_follow_row0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let ef_argmax = eager_follow
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        eprintln!(
            "[m-row-parity] M={m} follow_row=0 (pos={follow_pos}): max_diff={follow_diff:.6} graph_argmax={gf_argmax} eager_argmax={ef_argmax}"
        );
        assert_eq!(gf_argmax, ef_argmax, "M={m} follow step argmax mismatch");
        assert!(follow_diff <= 1e-2, "M={m} follow step max_diff {follow_diff} > 1e-2");

        // Verify zero-GEMV in-place speculative rejection rollback (`rollback_graph_to_accepted`)
        // followed by a subsequent M-token verify step at `follow_pos + 1`!
        model
            .rollback_graph_to_accepted(&mut slot, follow_pos, 1)
            .expect("rollback_graph_to_accepted(1)");
        let after_follow_toks: Vec<u32> = (0..m).map(|i| after_follow_tok + (i as u32) * 2).collect();
        let fast_after_out = model
            .run_m2_graph_step(&mut slot, graph_sess.as_mut(), &after_follow_toks, after_follow_pos)
            .expect("run_m2_graph_step after fast rollback")
            .expect("Some(tensor) after fast rollback")
            .to_vec_f32()
            .expect("after fast rollback readback");
        let fast_after_row0 = &fast_after_out[0..vocab];
        let fast_after_diff = fast_after_row0
            .iter()
            .zip(eager_after_follow.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[m-row-parity] M={m} fast rollback_graph_to_accepted(1) -> next M-step row=0 (pos={after_follow_pos}): max_diff={fast_after_diff:.6}"
        );
        assert!(
            fast_after_diff <= 1e-2,
            "M={m} fast rollback_graph_to_accepted(1) -> next step max_diff {fast_after_diff} > 1e-2"
        );
    }
}

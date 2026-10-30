//! Capture-level KV-arena gates for the Qwen35 decode graph.
//!
//! `graph_i8_append_and_attend_match_eager_f32` (grim-backend-rocm) gates the
//! int8 launcher pair one level BELOW capture: it calls the two launchers on
//! one synthetic row. These tests gate the layer the production path actually
//! runs — `get_or_create_decode_graph` -> seed-free capture -> `forward_replay`
//! — with the arena format resolved through the SAME env -> `KvArenaFormat`
//! resolver the model loader uses.
//!
//! Three gates:
//! 1. `qwen35_i8_graph_replay_matches_eager` — int8 arena: replayed logits
//!    must match the int8 eager path step for step, and the graph pool must
//!    prove engagement (`kv_arena_packed_width() == Some(8)` + sidecars).
//!    Without the engagement proof a test that silently ran f32 twice passes.
//! 2. `qwen35_packed_graph_refusal_int2v4` — int2v4 resolves through the
//!    parser but has NO capturable kernel pair; capture must REFUSE (eager
//!    fallback), never run the 8-bit kernels over 4-codes-per-byte keys.
//! 3. `qwen35_packed_graph_refusal_int4` — same for int4.
//! 4. `qwen35_whitecrow_graph_replay_matches_eager` — WhiteCrow arena at
//!    head_dim 128: the engagement proof is the 67-BLOCK width plus EMPTY
//!    scale sidecars (its scale + zero are inline), and replay must match the
//!    WhiteCrow eager path.
//!
//! Gated on `GRIM_GPU_TEST=1` + ROCm device. All three mutate process-global
//! env, so they serialize on `KV_FORMAT_ENV_LOCK`.

use grim_backend_rocm::BackendStorage;
use grim_core::model::{AdapterHandle, CausalLm};
use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_models_transformer::qwen35::Qwen35;
use grim_nn::WeightSource;

static KV_FORMAT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// The synthetic Qwen3.5 model builder and the GPU probe, SHARED with the
// M7.0 A/B (`ring_vs_launcher_ab.rs`) so there is one builder rather than a
// copy that can drift. Its `use` items come with it.
include!("common/qwen35_synth.rs");

#[test]
fn qwen35_packed_graph_replay_matches_eager() {
    // Both formats with a DEVICE-TOTAL kernel pair get the same capture-level
    // gate: replayed logits must match the same-format eager path step for
    // step. int8 is the production default; int4's pair landed with WS-3B and
    // needs the same evidence at the capture layer, not just kernel level.
    for (format, bits) in [("int8", 8u32), ("int4", 4u32)] {
        let Some((device, dev)) = gpu_device() else {
            return;
        };
        let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
            std::env::remove_var("GRIM_KV_ARENA_ROWS");
        }

        // The model AND the graph pool resolve the arena format from env at
        // construction, so the format must be set before `load`.
        let provider = synth_weights();
        let ws = WeightSource::root(&provider, device.clone());
        let model = Qwen35::load(device.clone(), &ws, tiny_cfg()).expect("load tiny qwen35");

        // ── Eager side: fresh session, one token per step, absolute positions.
        eprintln!("[STAGE] {format}: eager start");
        let mut session = model.new_session();
        let mut eager_logits = Vec::new();
        for (step, tok) in TOKENS.iter().enumerate() {
            let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
            let out = CausalLm::forward(
                &model,
                session.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager forward")
            .to_vec_f32()
            .expect("eager readback");
            eager_logits.push(out);
        }

        // ── Graph side: warmup (executes; state discarded by the pos
        // rewrite), capture (RECORDS only), then replay the SAME token
        // sequence from an equally empty arena. run.rs re-seeds from a
        // session because its capture step ran eagerly in the caller; here
        // the first replay simply replays the capture token, so no session
        // seed is needed and both sides see identical KV history
        // (0, 1, 2 prior rows).
        eprintln!("[STAGE] {format}: eager done, graph pool alloc");
        let mut graph = model
            .get_or_create_decode_graph(16, 1)
            .unwrap_or_else(|e| panic!("{format}: graph pool alloc failed: {e}"));

        // Engagement proof: the pool must hold the recorded width and its
        // scale sidecars. If the width recording regresses to the old dtype
        // conflation, this fails before any numeric comparison can mislead.
        assert_eq!(
            graph.buffers.kv_arena_packed_width(),
            Some(bits),
            "{format}: graph pool must record the {bits}-bit arena width"
        );
        assert!(
            !graph.buffers.k_scale_arena.is_empty() && !graph.buffers.v_scale_arena.is_empty(),
            "{format}: graph pool must allocate scale sidecars before capture"
        );

        eprintln!("[STAGE] {format}: pool ok, warmups");
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        eprintln!("[STAGE] {format}: warmups done, begin capture");
        graph.begin_capture().expect("begin capture");
        model
            .forward_capture(&mut graph, TOKENS[0])
            .expect("recorded capture");
        graph.end_capture().expect("end capture");
        assert!(graph.is_captured, "{format}: capture must succeed");

        eprintln!("[STAGE] {format}: captured, replays");
        let mut graph_logits = Vec::new();
        for tok in TOKENS {
            dev.reset_launch_count();
            model
                .forward_replay(&mut graph, tok)
                .unwrap_or_else(|e| panic!("{format}: graph replay failed: {e}"));
            // Replay must be a single hipGraphLaunch — any eager launch here
            // means the captured topology silently fell back.
            assert_eq!(
                dev.launch_count(),
                0,
                "{format}: replay must enqueue zero eager launches"
            );
            let logits = graph.read_logits_f32().expect("graph logit readback");
            assert!(
                logits.iter().all(|x| x.is_finite()),
                "{format}: replay logits non-finite"
            );
            graph_logits.push(logits);
            graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        }

        // ── Compare. Both sides quantize K/V with the SAME per-(row, kv_head)
        // amax scheme and read identical codes, so the only divergence source
        // is attention reduction order (eager kernel vs graph device-total
        // kernel) amplified by the FFN — the d2d gates measure ~1e-2 for a
        // FULL format change (f32 vs int8); same-format reduction-order noise
        // is strictly smaller. Bar is ~an order of magnitude under the eager
        // format-change gate (3e-1) with printed headroom.
        for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
            let (e, at, abs) = worst_scaled(a, b);
            eprintln!("[{format}-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
            assert!(
                e <= 1e-1,
                "{format} step {step}: graph replay diverges from eager: worst {e:.3e} at {at} \
                 (eager {}, graph {})",
                a[at],
                b[at]
            );
        }

        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
    }
}

/// WhiteCrow (u4 group-128 OSTQuant) is the third format with a capturable
/// kernel pair, and it differs from int8/int4 in two ways the engagement proof
/// has to express rather than assume:
///
/// 1. Its arena records the BLOCK width — [`grim_backend_rocm::KV_ARENA_BITS_WHITECROW`],
///    67 bytes per 128 values — not a code width. Matching it against
///    `Some(4)` would hand the int4 pair an interleaved-block arena (the
///    historical int2v4 mis-serve, one bit wider).
/// 2. Its bf16 scale + u8 zero ride INLINE in each block, so the pool must
///    allocate NO scale sidecars. The int8/int4 gate asserts sidecars exist;
///    asserting that here would be asserting a DEAD buffer — the complementary
///    proof (they are EMPTY) is what shows the allocator took the WhiteCrow
///    arm of the sidecar skip.
///
/// The model is built at head_dim 128 because WhiteCrow's per-128 groups align
/// to head vectors only when `head_dim % 128 == 0`; the shared synth geometry
/// is 64 and would fail the append's loud check rather than test anything.
#[test]
fn qwen35_whitecrow_graph_replay_matches_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "whitecrow");
        std::env::remove_var("GRIM_KV_ARENA_ROWS");
    }

    let provider = synth_weights_hybrid_hd(1, 1, false, 128);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, tiny_cfg_hd(128)).expect("load tiny qwen35 hd128");

    // ── Eager side: fresh session, one token per step.
    let mut session = model.new_session();
    let mut eager_logits = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        let out = CausalLm::forward(
            &model,
            session.as_mut(),
            &ids,
            &pos,
            &[] as &[AdapterHandle],
        )
        .expect("eager forward")
        .to_vec_f32()
        .expect("eager readback");
        eager_logits.push(out);
    }

    // ── Graph pool: prove engagement before any number can mislead.
    eprintln!("[STAGE] whitecrow: eager done, graph pool alloc");
    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .unwrap_or_else(|e| panic!("whitecrow: graph pool alloc failed: {e}"));
    assert_eq!(
        graph.buffers.kv_arena_packed_width(),
        Some(grim_backend_rocm::KV_ARENA_BITS_WHITECROW),
        "whitecrow: pool must record the 67-byte BLOCK width, not a code width"
    );
    assert!(
        graph.buffers.k_scale_arena.is_empty() && graph.buffers.v_scale_arena.is_empty(),
        "whitecrow: scales are inline in the blocks, so the pool must allocate NO sidecars \
         (a non-empty sidecar means the i8/i4 allocator arm ran)"
    );

    // ── Warmup (executes, state discarded), capture (records), replay.
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "whitecrow: capture must succeed");

    let mut graph_logits = Vec::new();
    for tok in TOKENS {
        dev.reset_launch_count();
        model
            .forward_replay(&mut graph, tok)
            .unwrap_or_else(|e| panic!("whitecrow: graph replay failed: {e}"));
        assert_eq!(
            dev.launch_count(),
            0,
            "whitecrow: replay must enqueue zero eager launches"
        );
        let logits = graph.read_logits_f32().expect("graph logit readback");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "whitecrow: replay logits non-finite"
        );
        graph_logits.push(logits);
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    // Same bar as the int8/int4 capture gate: both sides read the SAME WhiteCrow
    // codes, so the divergence is attention reduction order only.
    for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[whitecrow-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "whitecrow step {step}: graph replay diverges from eager: worst {e:.3e} at {at} \
             (eager {}, graph {})",
            a[at],
            b[at]
        );
    }

    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}

/// WhiteCrow graph replay with the token-parallel SPLIT engaged: same
/// contract as `qwen35_whitecrow_graph_replay_matches_eager`, but the graph
/// pool is created at max_ctx 1024, which makes the build allocate the split
/// partial scratch (2 chunks x 512 tokens) and the capture take the
/// `launch_qkv_attention_dev_batch_whitecrow_split` arm. Engagement is
/// asserted on `buffers.wc_split_chunk_len` BEFORE capture — without that
/// proof a serial-arm pass would silently count as split coverage.
#[test]
fn qwen35_whitecrow_graph_split_replay_matches_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "whitecrow");
        std::env::remove_var("GRIM_KV_ARENA_ROWS");
    }

    let provider = synth_weights_hybrid_hd(1, 1, false, 128);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, tiny_cfg_hd(128)).expect("load tiny qwen35 hd128");

    // ── Eager side: fresh session, one token per step.
    let mut session = model.new_session();
    let mut eager_logits = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        let out = CausalLm::forward(
            &model,
            session.as_mut(),
            &ids,
            &pos,
            &[] as &[AdapterHandle],
        )
        .expect("eager forward")
        .to_vec_f32()
        .expect("eager readback");
        eager_logits.push(out);
    }

    // ── Graph pool at 1024 ctx: 2 chunks x 512 tokens -> split scratch.
    eprintln!("[STAGE] whitecrow-split: eager done, graph pool alloc");
    let mut graph = model
        .get_or_create_decode_graph(1024, 1)
        .unwrap_or_else(|e| panic!("whitecrow-split: graph pool alloc failed: {e}"));
    assert_eq!(
        graph.buffers.wc_split_chunk_len,
        Some(512),
        "whitecrow-split: pool must allocate the split partial scratch at ctx 1024"
    );
    assert_eq!(
        graph.buffers.kv_arena_packed_width(),
        Some(grim_backend_rocm::KV_ARENA_BITS_WHITECROW),
        "whitecrow-split: pool must record the 67-byte BLOCK width"
    );

    // ── Warmup (executes, state discarded), capture (records), replay.
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "whitecrow-split: capture must succeed");

    let mut graph_logits = Vec::new();
    for tok in TOKENS {
        dev.reset_launch_count();
        model
            .forward_replay(&mut graph, tok)
            .unwrap_or_else(|e| panic!("whitecrow-split: graph replay failed: {e}"));
        assert_eq!(
            dev.launch_count(),
            0,
            "whitecrow-split: replay must enqueue zero eager launches"
        );
        let logits = graph.read_logits_f32().expect("graph logit readback");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "whitecrow-split: replay logits non-finite"
        );
        graph_logits.push(logits);
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    // Same bar as the serial WhiteCrow capture gate.
    for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[whitecrow-split-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "whitecrow-split step {step}: graph replay diverges from eager: worst {e:.3e} at {at} \
             (eager {}, graph {})",
            a[at],
            b[at]
        );
    }

    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}

/// int2v4 resolves through `parse_kv_arena_env` and has EAGER kernels, but no
/// capturable pair. `get_or_create_decode_graph` must refuse — the historic
/// bug ran the 8-bit launcher pair over a 4-codes-per-byte arena, silently.
#[test]
fn qwen35_packed_graph_refusal_int2v4() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "int2v4");
    }
    let provider = synth_weights();
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, tiny_cfg()).expect("load tiny qwen35");
    let err = model
        .get_or_create_decode_graph(16, 1)
        .expect_err("int2v4 graph capture must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("no device-total append/attend kernel pair"),
        "refusal must name the missing kernel pair, got: {msg}"
    );
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}

/// A6 GATE: a layer-split model must now CAPTURE and REPLAY, one graph per
/// device, with the boundary hidden state handed across by the ordered peer
/// copy (A4). This replaces the old refusal gate: the refusal existed because
/// a cross-device capture faults on replay (`p2p_graph_probe.rs` isolates
/// that); the per-segment capture removes the cross-device capture, not the
/// symptom.
///
/// Fixture: 4 layers at interval = 2 (see GATE FIXTURE NOTE in the plan) —
/// recurrent {0,2}, attention {1,3}, both segments hold both kinds. NO
/// seeding: both sides start from an empty arena exactly like the
/// single-device gate (seeding coverage is A5's split gate).
#[test]
fn qwen35_layer_split_model_replays_like_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits it, and this tiny model
    // fits anywhere — force the split so this gate tests the split path.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid qwen35");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    let ords: Vec<usize> = model
        .blocks
        .iter()
        .map(|b| match b.device {
            Device::Rocm(o) => o,
            _ => usize::MAX,
        })
        .collect();
    assert!(
        ords.iter().any(|&o| o != ords[0]),
        "model did not layer-split; this gate would be vacuous: {ords:?}"
    );
    // Both layer kinds must be present, or the gate is not testing a hybrid.
    let is_attn = |i: usize| (i + 1) % 2 == 0;
    assert!(
        (0..4).any(is_attn) && (0..4).any(|i| !is_attn(i)),
        "the gate model must contain BOTH attention and recurrent layers"
    );

    // Eager reference, one token per step, absolute positions.
    let mut session = model.new_session();
    let mut eager = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        eager.push(
            CausalLm::forward(
                &model,
                session.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager forward")
            .to_vec_f32()
            .expect("readback"),
        );
    }

    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("a layer-split model must now be capturable");
    assert_eq!(graph.segments.len(), 2, "one graph per device");
    assert_eq!(graph.segments[0].layers, 0..2);
    assert_eq!(graph.segments[1].layers, 2..4);
    // The head must land in the LAST segment, or its kernels are in no graph.
    let head_ord = match model.output.weight().device() {
        Device::Rocm(o) => *o,
        _ => usize::MAX,
    };
    assert_eq!(
        head_ord,
        graph.segments.last().unwrap().device,
        "the output head must live on the last segment's device"
    );

    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "split capture must succeed");

    let mut got = Vec::new();
    for tok in TOKENS {
        model.forward_replay(&mut graph, tok).expect("replay");
        got.push(graph.read_logits_f32().expect("readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    for (step, (a, b)) in eager.iter().zip(got.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[split-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "split step {step}: replay diverges from eager: worst {e:.3e} at {at} \
             (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f16");
    }
}

/// A5b GATE: each segment's `pos` replica must LIVE on that segment's device and
/// must receive the replay's start position.
///
/// A6's split-parity gate catches a wrong VALUE; this catches the wrong DEVICE.
/// `pos_dev` is bumped INSIDE the graph (`launch_bump_i32_slots`), so on a split
/// placement a device-1 layer would mutate device-0 memory — peer access covers
/// the seed pairs only, not this path. The replica design puts one pos scalar on
/// each segment's device and writes the replay start to all of them before the
/// launches; this test asserts both halves of that contract directly, so a
/// replica that silently stayed on device 0 fails HERE rather than as a fault
/// (or a silent peer-read) on replay.
#[test]
fn pos_replicas_exist_per_segment_device_and_start_aligned() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits, and this tiny model
    // fits anywhere — force the split so replicas are actually built.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid qwen35");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("a layer-split model must be capturable");
    assert_eq!(graph.segments.len(), 2, "one graph per device");

    // 1. Every segment's replica must live on that segment's device. A replica
    //    still on device 0 while segment 1's graph reads it is the A5b defect.
    for (seg, s) in graph.segments.iter().enumerate() {
        let p = graph.buffers.pos_dev_for(s.device);
        assert_eq!(
            p.device_ordinal(),
            s.device,
            "pos replica {seg} must live on its segment's device {}",
            s.device
        );
    }

    // 2. Writing the replay start position must land on EVERY segment's replica,
    //    each on its own device's stream.
    graph
        .buffers
        .write_pos_all_segments(7)
        .expect("write pos to all segments");
    for (seg, s) in graph.segments.iter().enumerate() {
        RocmDevice::shared(s.device).synchronize();
        let p = graph.buffers.pos_dev_for(s.device);
        assert_eq!(
            p.to_cpu_vec_u32().expect("read pos")[0],
            7,
            "replica {seg} (device {}) did not receive the replay start position",
            s.device
        );
    }

    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };
}

/// A0 BASELINE (external review 2026-10-16, item 1). The captured pos scalar is
/// bumped by EVERY attention layer (decode_graph.rs:1849, per-layer, no reset),
/// so a step with TWO OR MORE attention layers makes layer 2 read pos + steps.
/// Every existing graph gate here uses num_layers = 1 / interval = 1 — exactly
/// one attention layer per step — so this has never been observed. Single
/// device, f32 arena, hybrid 4-layer model, NO seed: pure graph-vs-eager.
/// If this goes red, the single-device graph has latent pos drift and A5b's
/// per-device replica design inherits the same defect.
#[test]
fn qwen35_graph_matches_eager_with_two_attention_layers_per_step() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2; // attention layers 1 and 3: two bumps per step
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load 4-layer hybrid");

    let is_attn = |i: usize| (i + 1) % 2 == 0;
    let n_attn: usize = (0..4).filter(|&i| is_attn(i)).count();
    assert_eq!(
        n_attn, 2,
        "gate model must hold exactly two attention layers"
    );

    // Eager reference.
    let mut session = model.new_session();
    let mut eager_logits = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        let out = CausalLm::forward(
            &model,
            session.as_mut(),
            &ids,
            &pos,
            &[] as &[AdapterHandle],
        )
        .expect("eager forward")
        .to_vec_f32()
        .expect("eager readback");
        eager_logits.push(out);
    }

    // Graph side: warmup, capture, replay — no seed, both sides from empty arena.
    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("graph pool alloc");
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured);

    let mut graph_logits = Vec::new();
    for tok in TOKENS {
        model.forward_replay(&mut graph, tok).expect("graph replay");
        graph_logits.push(graph.read_logits_f32().expect("graph logit readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[a0-two-attn-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "A0 step {step}: graph replay diverges from eager with two attention \
             layers per step: worst {e:.3e} at {at} (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };
}

/// A5 gate: a split model's graph buffers must be seeded per segment — each
/// family (KV, GDL, conv, KDA) copies on the LAYER's device context. Needs
/// two live devices; the seeder code paths themselves are covered per-layer
/// by construction (layer_devices[i] == the eager layer's device).
#[test]
fn qwen35_split_model_seed_families_run_on_their_own_devices() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits it, and this tiny model
    // fits anywhere — force the split so this gate tests the split path.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    // Eager prefill to produce seed sources.
    let mut session = model.new_session();
    let ids = dev_tensor(dev.as_ref(), &device, &[TOKENS[0] as f32]);
    let pos = dev_tensor(dev.as_ref(), &device, &[0.0f32]);
    let _ = CausalLm::forward(
        &model,
        session.as_mut(),
        &ids,
        &pos,
        &[] as &[AdapterHandle],
    )
    .expect("eager prefill");

    let srcs = model
        .eager_kv_seed_sources(session.as_ref(), 1)
        .expect("kv seeds");
    let conv = model
        .eager_conv_device_seed_sources(session.as_ref())
        .expect("conv seeds");
    let kda = model
        .eager_kda_seed_sources(session.as_ref())
        .expect("kda seeds");

    // The production seed path (run.rs macros) must succeed against a split
    // placement: every family copies on the right device context now.
    let mut graph = model.get_or_create_decode_graph(16, 1).expect("graph");
    let seed_dev = grim_backend_rocm::RocmDevice::shared(0);
    graph
        .buffers
        .seed_kv_arena_from_eager(&seed_dev, &srcs)
        .expect("seed kv (split placement)");
    graph
        .buffers
        .seed_gdl_state_from_eager(&seed_dev, &srcs)
        .expect("seed gdl (split placement)");
    graph
        .buffers
        .seed_conv_rings_from_device(&seed_dev, &conv)
        .expect("seed conv (split placement)");
    graph
        .buffers
        .seed_kda_state_from_eager(&seed_dev, &kda)
        .expect("seed kda (split placement)");

    // The trap the plan names (ai-arch-expert §5): a seed that lands NOTHING is
    // SILENT — a zeroed recurrent state produces fluent-looking garbage with no
    // error. So assert the recurrent state actually ARRIVED, not merely that the
    // seed call returned Ok. `kda[i].is_some()` is the recurrent-layer predicate
    // (attention layers allocate no KDA state), and every recurrent layer here
    // must carry signal after the eager prefill seeded it.
    RocmDevice::shared(0).synchronize();
    RocmDevice::shared(1).synchronize();
    let mut checked = 0usize;
    for (i, kda) in graph.buffers.kda.iter().enumerate() {
        let Some(k) = kda else { continue };
        let v = k.state.to_cpu_vec_f32().expect("kda state readback");
        assert!(
            v.iter().any(|x| *x != 0.0),
            "layer {i} KDA recurrent state is all zeros after seeding — the seed was dropped \
             (this is the fluent-garbage failure mode, not an error)"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no recurrent layer exposed a KDA state — the split model is all-attention and this \
         gate would be vacuous"
    );
}

/// D1 GATE: the decode graph must SERVE `GRIM_KV_EVICT_MODE=curdkv` instead of
/// refusing. Today the refusal fires because the host compaction pass rewrites
/// `current_pos`; the device-side replacement keeps `total_dev` on-device.
/// Fixture: 4-layer hybrid (interval 2), f32 arena, cap 8 rows (sinks 2,
/// residual 2), 12 tokens so the run crosses the eviction boundary twice.
/// Both arms run under the SAME env; the eager arm exercises the host
/// compaction, the graph arm the device-side one — replay must track eager.
#[test]
fn qwen35_graph_serves_curdvk_eviction_like_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::set_var("GRIM_KV_ARENA_ROWS", "8");
        std::env::set_var("GRIM_KV_SINKS", "2");
        std::env::set_var("GRIM_KV_RESIDUAL", "2");
        std::env::set_var("GRIM_KV_EVICT_MODE", "curdkv");
    }

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load curdkv model");

    // 12 tokens: cap 8 with sinks 2 / residual 2 means the boundary is crossed
    // at pos 9 (compaction) and again later — the gate needs both arms to
    // survive at least two compactions.
    let tokens: Vec<u32> = (0..12).map(|i| (3 + 7 * i) % VOCAB as u32).collect();

    // Eager arm: host compaction at the boundary.
    let mut session = model.new_session();
    let mut eager = Vec::new();
    for (step, tok) in tokens.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        eager.push(
            CausalLm::forward(
                &model,
                session.as_mut(),
                &ids,
                &pos,
                &[] as &[AdapterHandle],
            )
            .expect("eager curdkv forward")
            .to_vec_f32()
            .expect("eager readback"),
        );
    }

    // Graph arm: warmup 2, capture 1, replay the rest. The FIRST capture must
    // succeed — that is the red part of this gate while the refusal stands.
    // Env hygiene: this gate OWNS the curdkv env, and a panic must not leak it
    // into the rest of the test process (it poisoned every later gate once).
    let graph_res = model.get_or_create_decode_graph(16, 1);
    let mut graph = match graph_res {
        Ok(g) => g,
        Err(e) => {
            unsafe {
                std::env::remove_var("GRIM_KV_ARENA_ROWS");
                std::env::remove_var("GRIM_KV_SINKS");
                std::env::remove_var("GRIM_KV_RESIDUAL");
                std::env::remove_var("GRIM_KV_EVICT_MODE");
            }
            panic!("D1: the graph must not refuse CurDKV eviction anymore: {e}");
        }
    };
    let _ = model.forward_capture(&mut graph, tokens[0]);
    let _ = model.forward_capture(&mut graph, tokens[1]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, tokens[2])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "curdkv capture must succeed");

    let mut got = Vec::new();
    for (step, tok) in tokens.iter().enumerate().skip(2) {
        model
            .forward_replay(&mut graph, *tok)
            .unwrap_or_else(|e| panic!("D1 replay step {step}: {e}"));
        got.push(graph.read_logits_f32().expect("graph logit readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    for (gi, (a, b)) in eager.iter().skip(2).zip(got.iter()).enumerate() {
        let step = gi + 2;
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[d1-curdvk-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "D1 curdkv step {step}: graph diverges from eager host compaction: \\
             worst {e:.3e} at {at} (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::remove_var("GRIM_KV_ARENA_ROWS");
        std::env::remove_var("GRIM_KV_SINKS");
        std::env::remove_var("GRIM_KV_RESIDUAL");
        std::env::remove_var("GRIM_KV_EVICT_MODE");
    }
}

/// PM-2 gate: with `GRIM_EMIT_TASK_GRAPH=1`, capturing the 4-layer hybrid must
/// publish a task-graph IR whose structure mirrors the real walk — embedding
/// node, per-layer (RECURRENT-or-ATTENTION, FFN GEMM) pairs, final NORM, head
/// GEMM — with a derived capability set and a single sink. This is the M2
/// interpreter's input contract.
#[test]
fn qwen35_capture_emits_a_structurally_true_task_graph() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1");
    }

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load emit model");

    let mut graph = model.get_or_create_decode_graph(16, 1).expect("graph");
    assert!(
        graph.buffers.emitted_task_graph.is_none(),
        "no IR before capture"
    );

    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");

    let ir = graph
        .buffers
        .emitted_task_graph
        .as_ref()
        .expect("the capture must publish the emitted task graph");

    // M5 Step 4 keys `forward_replay` on `ring_context.is_some()`, so building
    // the ring here would make this INSPECTION flag silently reroute the decode
    // path — the same class as int2v4 reaching capture through a spelling list.
    // The ring is built only under its own flag.
    assert!(
        graph.ring_context.is_none(),
        "GRIM_EMIT_TASK_GRAPH alone must not build the ring — that would change \
         what executes, under a flag that only asks for the IR"
    );

    // M7.1: the structure is now PER-LAUNCH. The attention blocks name each
    // of their launches as its own node, so the IR describes the walk
    // exactly rather than summarising a whole block with one node. The
    // RECURRENT layers keep their summary node (the ring's OP_RECURRENT arm
    // runs the whole KDA step, which is one launch).
    //
    // The node count is asserted as a COUNT of the attention layers' launches
    // rather than as a literal, because the literal would be a copy of the
    // block's launch list — and a copy is exactly what can drift. What is
    // pinned here is the SHAPE: one embedding gather, then per layer the
    // attention layers' per-launch chain and the recurrent layers' summary
    // node, then the final norm and the head.
    use grim_backend_rocm::task_graph::op;
    let ops: Vec<u32> = ir.nodes.iter().map(|n| n.opcode).collect();
    eprintln!(
        "[struct] the first 12 opcodes: {:?}",
        &ops[..12.min(ops.len())]
    );
    for (i, n) in ir.nodes.iter().take(6).enumerate() {
        eprintln!(
            "[struct] node {i}: opcode={} n_args={} n_deps={} deps={:?} dev={}",
            n.opcode,
            n.n_args,
            n.n_deps,
            &n.deps[..n.n_deps as usize],
            n.device
        );
    }
    eprintln!("[struct] total nodes: {}", ops.len());
    assert_eq!(
        ops[0],
        op::ELEMENTWISE,
        "first node is the embedding gather"
    );

    // The attention layers' launches: the nodes between a layer's start and
    // the next layer's start. The block's FIRST launch is the norm, and its
    // LAST is the output projection, with the ring-executable ops (NORM,
    // GEMM, ELEMENTWISE, Q8_0_DOT4_F32ACT) named where the ring could run
    // them and FALLBACK everywhere else.
    //
    // The count is asserted PER LAYER, and each attention layer must have
    // MORE than one node — that is the whole point of M7.1: a block that
    // summarised its launches as ONE node is what the refinement removes.
    let mut layer_starts = Vec::new();
    {
        // The recurrent layers' summary nodes carry op::RECURRENT, so they
        // mark the layer boundaries unambiguously.
        let mut idx = 1usize; // past the embedding gather
        for _ in 0..4 {
            layer_starts.push(idx);
            // Skip this layer's nodes: the attention layers have many, the
            // recurrent ones exactly one. The next RECURRENT node (or the
            // final NORM) ends the layer.
            while idx < ops.len() && ops[idx] != op::RECURRENT && ops[idx] != op::NORM {
                idx += 1;
            }
            if idx < ops.len() && ops[idx] == op::RECURRENT {
                idx += 1;
            }
        }
    }
    for (li, &start) in layer_starts.iter().enumerate() {
        let layer = li;
        let is_attn = (layer + 1) % 2 == 0;
        if is_attn {
            // The attention layer's nodes run from `start` to the next
            // layer's start.
            let end = layer_starts.get(li + 1).copied().unwrap_or(ops.len());
            let n = end - start;
            assert!(
                n > 1,
                "layer {layer} is an attention layer, so its launches must be \
                 named individually (M7.1) — the IR gives it {n} node(s)"
            );
        }
    }

    // The chain is total: every node except the first depends on its
    // predecessor, so the single sink is the head.
    assert_eq!(ir.sinks, vec![(ir.nodes.len() - 1) as u32]);

    // The IR now names the attention block's launches, so it MUST contain
    // the opcodes the block's own launches use: the norm, the projections,
    // the residual add, and the fused QKV's Q8_0 dot4.
    assert!(
        ops.contains(&op::NORM),
        "the attention block's RMSNorm must be named (M7.1)"
    );
    assert!(
        ops.contains(&op::ELEMENTWISE),
        "the residual add must be named (M7.1)"
    );
    // The fused QKV's Q8_0 dot4 is named only when the model ACTUALLY takes
    // the fused path, which needs all three projections present AND Q8_0
    // (row-exact). The synthetic model here carries f32 weights, so it takes
    // the plain path and there is no dot4 to name. The assertion is therefore
    // conditional on the model having the fused weights — a gate that
    // demanded a node the model never performs would be a gate demanding a
    // fiction.
    let has_fused_qkv = model
        .blocks
        .iter()
        .any(|b| b.is_full_attention && b.wqkv_q80_fused.is_some());
    if has_fused_qkv {
        assert!(
            ops.contains(&op::Q8_0_DOT4_F32ACT),
            "the fused QKV's Q8_0 dot4 must be named (M7.1)"
        );
    }
    // M7.4: the Q8_0 PATH CONTRACT. Every Q8_0 dot4 node carries the path
    // the device resolved, and the IR's answer must be the one the launcher
    // would take — the contract exists so the IR and the launcher read the
    // SAME decision, not each their own.
    {
        use grim_backend_rocm::task_graph::q8_0_path;
        let shared = grim_backend_rocm::RocmDevice::shared(dev.ordinal());
        let mut checked = 0usize;
        for n in &ir.nodes {
            if n.opcode != op::Q8_0_DOT4_F32ACT {
                continue;
            }
            let carried = q8_0_path::from_params(&n.params);
            // The node's own K (params[2]) resolves the contract — not some
            // other launch's geometry.
            let expected = shared.q8_0_path_contract(n.params[2] as usize);
            assert_eq!(
                carried, expected,
                "the IR's Q8_0 path contract ({carried}) disagrees with the \
                 device's ({expected}) for a node with k={} (M7.4)",
                n.params[2]
            );
            checked += 1;
        }
        eprintln!("[struct] the Q8_0 path contract agrees on {checked} node(s)");
    }

    // And the launches the ring CANNOT run are FALLBACK nodes, not nodes
    // wearing an opcode that names a different computation.
    assert!(
        ops.contains(&op::FALLBACK),
        "the launches with no ring arm (the copies, the RoPE, the KV append, \
         the arena attention, the output gate) must be FALLBACK nodes (M7.1)"
    );

    // The chain is total: every node except the first depends on its
    // predecessor, so the single sink is the head.
    assert_eq!(ir.sinks, vec![(ir.nodes.len() - 1) as u32]);

    // Derived caps, single device.
    assert!(!ir.caps.needs_peer, "single-device gate model");
    // M7.1: the attention block's ARENA attention is a FALLBACK node, not an
    // ATTENTION node. The wave's OP_ATTN arm is a flash-attention over
    // (q, k, v, out) with a 256 head_dim limit; the block's attention reads
    // the PACKED KV arena with per-head scales and a ring-bounded live row
    // count. They are different computations, so a node naming the block's
    // launch with the ring's ATTENTION opcode would tell the ring to
    // compute a DIFFERENT answer — which is precisely what the LFM2's
    // FALLBACK nodes exist to prevent.
    //
    // So the gate asserts the OPPOSITE of what it asserted before M7.1: the
    // IR must NOT contain an ATTENTION node for this model, because this
    // model's attention is not the ring's attention.
    assert!(
        !ir.caps.opcodes.contains(&op::ATTENTION),
        "the qwen35 arena attention is not the ring's OP_ATTN arm, so the IR \
         must not claim an ATTENTION node (M7.1)"
    );
    assert!(ir.caps.opcodes.contains(&op::RECURRENT));
    assert!(ir.caps.opcodes.contains(&op::GEMM));
    assert_eq!(ir.caps.devices.len(), 1);

    // RING-READINESS GATE (replaces the pinned "not full" assertion). The
    // hybrid model's RECURRENT nodes are now ring-executable: the wave has the
    // OP_RECURRENT arm, the emitter staged a descriptor for each recurrent layer,
    // and the plan binds them.
    //
    // BOTH halves are asserted, because `ring_ready` alone only says the opcode
    // is claimed — it cannot see whether the plan can actually bind a node. That
    // is the Q8_0_DOT4 failure mode: an advertised opcode with no arm behind it.
    let ready = grim_backend_rocm::task_graph::ring_ready(&ir.caps);
    assert!(
        ready.full,
        "the hybrid's opcodes must all be ring-covered; missing {:?}",
        ready.missing
    );
    assert!(
        ready.missing.is_empty(),
        "nothing should be missing: {:?}",
        ready.missing
    );
    let plan = grim_backend_rocm::kernels::task_ring::TaskRingPlan::from_graph(&ir);
    assert_eq!(
        plan.recurrences().len(),
        2,
        "both recurrent layers must bind their descriptor: {:?}",
        plan.recurrences()
    );
    // The argument table is the model's: every arg handle resolves.
    for n in &ir.nodes {
        for &a in &n.args[..n.n_args as usize] {
            assert!((a as usize) < ir.arg_table.len(), "dangling arg handle {a}");
        }
    }

    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}

/// M6 STEP 4: the ring handoff at a device boundary — IDENTICAL logits to
/// A6's host-issued peer copy, and ONE FEWER host-visible synchronisation per
/// token.
///
/// A6's boundary is a host-issued peer memcpy plus a recorded event plus the
/// consumer's stream waiting on it: three host calls and a synchronisation per
/// boundary per token. M6's handoff replaces them with ONE wave launch (the
/// producer's copy, in the producer's stream ORDER, so it is strictly after
/// the producer's graph) and ONE device-side poll (the consumer's device
/// watches the producer's monotonic tail over peer access). The host
/// synchronisation at the boundary is GONE.
///
/// The gate measures both halves of that claim:
///  * the logits are the same (the handoff copies the same bytes), and
///  * the per-token host-visible synchronisation count is lower — measured by
///    a counter in the A6 path, not asserted from the plan.
#[test]
fn the_ring_handoff_matches_a6_and_removes_a_host_sync_per_token() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };

    // One split hybrid, replayed N tokens, returning the logits and the number
    // of host-visible boundary synchronisations the replay issued.
    let run = |with_handoff: bool| -> (Vec<Vec<f32>>, u64) {
        // The handoff is ON by default now, so the A6 run must turn it OFF
        // explicitly — "unset" no longer means A6.
        unsafe {
            std::env::set_var("GRIM_RING_HANDOFF", if with_handoff { "1" } else { "0" });
        }
        let mut cfg = tiny_cfg();
        cfg.num_layers = 4;
        cfg.full_attention_interval = 2;
        // The planner prefers ONE device when the model fits — force the
        // split, or there is no boundary to hand off.
        unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
        cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
        let provider = synth_weights_hybrid(4, 2);
        let ws = WeightSource::root(&provider, device.clone());
        let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid qwen35");
        unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

        let mut graph = model
            .get_or_create_decode_graph(16, 1)
            .expect("a layer-split model must now be capturable");
        assert_eq!(graph.segments.len(), 2, "one graph per device");
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        graph.begin_capture().expect("begin capture");
        model
            .forward_capture(&mut graph, TOKENS[0])
            .expect("recorded capture");
        graph.end_capture().expect("end capture");
        assert!(graph.is_captured, "split capture must succeed");

        let syncs_before = grim_backend_rocm::peer_access::boundary_host_sync_count();
        let mut logits = Vec::new();
        for tok in TOKENS {
            model.forward_replay(&mut graph, tok).expect("replay");
            logits.push(graph.read_logits_f32().expect("readback"));
            graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        }
        let syncs = grim_backend_rocm::peer_access::boundary_host_sync_count() - syncs_before;
        (logits, syncs)
    };

    let (a6_logits, a6_syncs) = run(false);
    // The CONTROL: a second A6 run. Two A6 runs differ slightly from each
    // other (the f16 arena's reduction order depends on where the buffers
    // land), so without this the gate could not tell "the handoff changed the
    // logits" from "the runs differ anyway".
    let (a6_control_logits, a6_control_syncs) = run(false);
    let (handoff_logits, handoff_syncs) = run(true);
    unsafe { std::env::remove_var("GRIM_RING_HANDOFF") };

    // The handoff must have been TAKEN: the whole point is that the A6 path
    // was not used. Without this, the gate would pass on two A6 runs.
    assert_eq!(
        handoff_syncs, 0,
        "the handoff path must issue NO host-visible boundary synchronisation (got {handoff_syncs})"
    );
    assert_eq!(
        a6_syncs,
        TOKENS.len() as u64,
        "A6 must issue one host-visible boundary synchronisation per token (got {a6_syncs})"
    );
    assert!(
        handoff_syncs < a6_syncs,
        "the handoff must remove a host-visible synchronisation per token (a6 {a6_syncs}, handoff {handoff_syncs})"
    );

    assert_eq!(
        a6_control_syncs, a6_syncs,
        "the two A6 runs must issue the same number of host synchronisations"
    );

    // And the logits must be the same: the handoff copies the same bytes.
    assert_eq!(
        a6_logits.len(),
        handoff_logits.len(),
        "both runs must produce the same number of steps"
    );
    let mut control_worst = 0.0f32;
    for (step, (a, b)) in a6_logits.iter().zip(a6_control_logits.iter()).enumerate() {
        let (e, _, _) = worst_scaled(a, b);
        control_worst = control_worst.max(e);
        let _ = step;
    }
    let mut handoff_worst = 0.0f32;
    for (step, (a, b)) in a6_logits.iter().zip(handoff_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        handoff_worst = handoff_worst.max(e);
        eprintln!("[m6-handoff] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "handoff step {step} diverges from A6: worst {e:.3e} at {at} \
             (A6 {}, handoff {})",
            a[at],
            b[at]
        );
    }
    eprintln!("[m6-handoff] worst A6-vs-A6 {control_worst:.3e}, A6-vs-handoff {handoff_worst:.3e}");
    assert!(
        handoff_worst <= control_worst.max(1e-1),
        "the handoff must not diverge more than two A6 runs do from each other \
         (A6-vs-A6 {control_worst:.3e}, A6-vs-handoff {handoff_worst:.3e})"
    );
}

/// M6: the Qwen3.5 hybrid's IR walk, with the ring's RECURRENT node taking
/// the KDA step — and the walk's logits against the launcher path's.
///
/// The hybrid is the vehicle that matters: it is the only model with RECURRENT
/// layers, so it is the only one whose IR walk has to put the ring's node
/// BETWEEN the projections and the output projection. The ring's
/// `OP_RECURRENT` arm runs the same three device functions the block's KDA step
/// runs (the short conv causal step, the batched gated delta rule and the
/// head-norm gate), and the device waits for it — a kernel spinning on the
/// descriptor's status — before the output projection reads the branch it
/// wrote.
///
/// The bar is BIT-IDENTICAL: the walk runs the same kernels in the same order,
/// so any difference is the ring's dispatch changing the arithmetic.
#[test]
fn the_qwen35_ring_walk_matches_the_launcher_path() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };
    // The IR is only PUBLISHED when the emitter is on, and the ring's plan is
    // derived from that published IR.
    unsafe { std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The hybrid must have BOTH kinds of layer: the recurrent ones are what
    // the ring takes, and the attention ones are what stays on the launchers.
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load hybrid qwen35");
    let n_attn = model.blocks.iter().filter(|b| b.is_full_attention).count();
    let n_recurrent = model.blocks.len() - n_attn;
    assert_eq!(
        (n_attn, n_recurrent),
        (2, 2),
        "the hybrid must have both kinds"
    );

    // ---- Arm A: the launcher path (the captured graph, replayed). ----
    let mut graph = model.get_or_create_decode_graph(16, 1).expect("capturable");
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "capture must succeed");
    // The capture's two EAGER warm-ups advanced the KV arena's layer totals
    // and the recurrent state, so the graph starts its replay with a polluted
    // history. The walk arms reset theirs (`reset_for_replay`), so this arm
    // must reset too — otherwise the two arms compare DIFFERENT histories and
    // the difference looks like an arithmetic defect in the walk.
    graph.buffers.reset_for_replay().expect("reset (launcher)");

    let mut launcher_logits = Vec::new();
    for tok in TOKENS {
        model
            .forward_replay(&mut graph, tok)
            .expect("launcher replay");
        launcher_logits.push(graph.read_logits_f32().expect("readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    // ---- Arm B: the IR walk on its own (no ring) — the M2 oracle. This
    // isolates the walk from the ring's RECURRENT arm: if the walk alone
    // differs from the launcher path, the walk is at fault; if it matches and
    // only the ring arm differs, the arm is. ----
    let mut gp = model
        .get_or_create_decode_graph(16, 1)
        .expect("capturable (walk)");
    let _ = model.forward_capture(&mut gp, TOKENS[0]);
    let _ = model.forward_capture(&mut gp, TOKENS[0]);
    gp.begin_capture().expect("begin capture (walk)");
    model
        .forward_capture(&mut gp, TOKENS[0])
        .expect("recorded capture (walk)");
    gp.end_capture().expect("end capture (walk)");
    gp.buffers.reset_for_replay().expect("reset (walk)");

    let mut walk_logits = Vec::new();
    for tok in TOKENS {
        model
            .forward_interpret_ir(&mut gp, tok)
            .expect("walk interpret");
        walk_logits.push(gp.read_logits_f32().expect("walk logits"));
        gp.buffers.current_pos = gp.buffers.current_pos.wrapping_add(1);
    }

    // ---- Arm C: the IR walk, with the ring's RECURRENT node. ----
    let mut gp_ring = model
        .get_or_create_decode_graph(16, 1)
        .expect("capturable (ring)");
    let _ = model.forward_capture(&mut gp_ring, TOKENS[0]);
    let _ = model.forward_capture(&mut gp_ring, TOKENS[0]);
    gp_ring.begin_capture().expect("begin capture (ring)");
    model
        .forward_capture(&mut gp_ring, TOKENS[0])
        .expect("recorded capture (ring)");
    gp_ring.end_capture().expect("end capture (ring)");
    gp_ring.buffers.reset_for_replay().expect("reset (ring)");

    // The ring must be bound to ARM C's OWN IR. The RECURRENT node's WEIGHT
    // operand is the device address of that graph's `KdaTaskDescriptor`, and
    // the descriptor holds the addresses of THAT graph's q_buf / conv_out /
    // branch / state. A ring bound to another graph's IR would run the KDA
    // step on the WRONG buffers — the walk would then read a branch nobody
    // wrote, and the difference would look like an arithmetic defect in the
    // wave's arm when it is a wiring defect in the gate.
    let ir = gp_ring
        .buffers
        .emitted_task_graph
        .clone()
        .expect("the capture must publish the IR");
    let mut ring = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(
        &grim_backend_rocm::RocmDevice::shared(dev.ordinal()),
        &ir,
    )
    .expect("ring context")
    .expect("the ring must bind the hybrid's nodes");
    assert_eq!(
        ring.plan().recurrences().len(),
        n_recurrent,
        "the ring must bind one RECURRENT node per recurrent layer"
    );

    // M7.1: the DRAIN BUDGET. The per-launch split multiplies the IR's node
    // count, and the whole risk of that is that it multiplies the RING's
    // drains — the corpus's §9c finding ("any device-side chaining scheme
    // that waits on completion at fine granularity serialises, and the drains
    // multiply with the number of transfers per step"), measured on grim's
    // own hardware.
    //
    // So the budget is a GATE, not a note: the ring walk must cost the SAME
    // number of waves and the SAME number of blocking D2H per step as it did
    // before the split. The per-launch nodes are FALLBACK nodes (the ring
    // has no arm for them), so they must not reach the ring at all — and
    // this asserts exactly that, in the only terms that matter.
    let waves_before = grim_backend_rocm::kernels::task_ring::wave_launch_count();
    let d2h_before = grim_backend_rocm::blocking_d2h_count();
    let mut ring_logits = Vec::new();
    for tok in TOKENS {
        model
            .forward_interpret_ir_ring(&mut gp_ring, tok, &mut ring)
            .expect("ring interpret");
        ring_logits.push(gp_ring.read_logits_f32().expect("ring logits"));
        gp_ring.buffers.current_pos = gp_ring.buffers.current_pos.wrapping_add(1);
    }
    let steps = TOKENS.len() as u64;
    let waves = grim_backend_rocm::kernels::task_ring::wave_launch_count() - waves_before;
    let d2h = grim_backend_rocm::blocking_d2h_count() - d2h_before;
    eprintln!(
        "[qwen35-ring] the ring walk's drain budget: {waves} wave(s) and \
         {d2h} blocking D2H over {steps} step(s)"
    );
    // The recurrent layers are the ONLY ring work, so the budget is one wave
    // per recurrent layer per step — no more. The per-launch split must not
    // add any: the FALLBACK nodes never reach the wave.
    assert!(
        waves <= n_recurrent as u64 * steps,
        "the per-launch split multiplied the ring's wave launches: {waves} waves \
         for {n_recurrent} recurrent layer(s) over {steps} step(s) (M7.1's drain \
         budget)"
    );
    // The blocking D2H is the VERIFICATION read-back, once per step (M7.1b
    // deferred it from once per node). The logits readback below is the
    // test's own, and it is one per step too — so the budget is one per
    // step for the ring's verification, and the test's readback adds one
    // more. The gate is on the ring's share, which is what the budget is
    // about: it must not grow with the node count.
    assert!(
        d2h <= 2 * steps,
        "the per-launch split multiplied the ring's blocking D2H reads: {d2h} \
         over {steps} step(s) (M7.1b's deferred verification, M7.1's drain budget)"
    );

    // ---- The comparison: the RING walk against the M2 walk. ----
    //
    // Both arms are WALKS: the same model code, the same kernels, the same
    // order — the only difference is that one dispatches the recurrent layers'
    // KDA step to the ring's RECURRENT node. So the comparison is bit-exact,
    // and any difference is the ring's arm changing the arithmetic.
    //
    // (The walk against the CAPTURED GRAPH is a different comparison, and a
    // looser one: the eager path and the captured path differ by the known
    // 2.4e-5-class amount — the same difference the qwen35 graph-vs-eager
    // gate already accepts. That is the M2 oracle's job, not this gate's.)
    assert_eq!(
        walk_logits.len(),
        ring_logits.len(),
        "both arms must produce the same number of steps"
    );
    for (step, (a, b)) in walk_logits.iter().zip(ring_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[qwen35-ring] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e == 0.0,
            "ring step {step} diverges from the launcher path: worst {e:.3e} at {at} \
             (launcher {}, ring {})",
            a[at],
            b[at]
        );
    }
    eprintln!("[qwen35-ring] the ring's KDA step is bit-identical to the launcher's");

    // And the M2 oracle: the walk against the captured graph, at the bar the
    // qwen35's graph-vs-eager gate already uses.
    for (step, (a, b)) in launcher_logits.iter().zip(walk_logits.iter()).enumerate() {
        let (e, at, _) = worst_scaled(a, b);
        eprintln!("[qwen35-walk] step {step}: the walk vs the graph replay, worst {e:.3e}");
        assert!(
            e <= 1e-4,
            "the walk diverges from the captured graph at step {step}: worst {e:.3e} at {at}",
        );
    }
    unsafe { std::env::remove_var("GRIM_EMIT_TASK_GRAPH") };
}

/// The IQ4_NL coverage row: a model whose weight is IQ4_NL must put Iq4Nl
/// on the IR's QuantFormat axis, and the walk must still reproduce the
/// captured graph.
///
/// The row's REAL-checkpoint vehicles are both unusable here, and the reasons
/// are worth recording:
///
/// - The Xing40's IQ4_NL sibling (`xing4_0-29b-IQ4_NL.gguf`) is 20.1 GB of
///   4.5 bpw weights against this 17.1 GB card. It cannot be resident, and
///   the failure mode without a pre-flight is misleading: the load
///   "succeeds" through the managed-memory fallback and the first kernel
///   launch faults with `vram_free = 0.00`.
/// - The Gemma-4 E4B IQ4_NL checkpoint (`gemma-4-E4B-it-IQ4_NL.gguf`, 5.2
///   GB — it WOULD fit) cannot be LOADED: the tree has no Gemma-3/4 GGUF
///   loader. `general.architecture` is `gemma4`, the loader routes it to the
///   Gemma (v1) type, and that type asks for `blk.N.wq.weight` while the
///   checkpoint carries `blk.N.attn_q.weight` — and beyond the naming, the
///   model is a dense sliding-window attention with per-layer rope dims and
///   logit softcapping, none of which the tree implements.
///
/// So the row is exercised with a SYNTHETIC model whose down-projection
/// weight is genuinely IQ4_NL-PACKED: the format is carried by the storage
/// (which is what the axis reads), the emitter names it, the axis carries
/// it, and the walk still reproduces the captured graph. The format's ROCm
/// matmul arm exists (`KQuantScheme::IQ4NL` in `quantized_matmul`), so the
/// weight is really executed, not just labelled.
#[test]
fn the_iq4nl_row_is_on_the_coverage_axis() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };
    unsafe { std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    let provider = synth_weights_hybrid_with(4, 2, true);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load iq4nl qwen35");

    // ── The captured graph (the reference) and the IR it publishes.
    let mut graph = model.get_or_create_decode_graph(16, 1).expect("capturable");
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    let ir = graph
        .buffers
        .emitted_task_graph
        .clone()
        .expect("the capture must publish the IR");
    let iq4nl = grim_tensor::QuantFormat::Iq4Nl as u32;
    assert!(
        ir.caps.quant_modes.contains(&iq4nl),
        "the IQ4_NL model's axis {:?} does not carry Iq4Nl ({iq4nl}) — the coverage \
         row is still unexercised",
        ir.caps.quant_modes
    );
    eprintln!(
        "[iq4nl-row] the axis is {:?} (Iq4Nl = {iq4nl})",
        ir.caps.quant_modes
    );

    // ── The walk must reproduce the captured graph (the M2 oracle), with the
    // IQ4_NL weight in play.
    let mut gp = model
        .get_or_create_decode_graph(16, 1)
        .expect("capturable (walk)");
    let _ = model.forward_capture(&mut gp, TOKENS[0]);
    let _ = model.forward_capture(&mut gp, TOKENS[0]);
    gp.begin_capture().expect("begin capture (walk)");
    model
        .forward_capture(&mut gp, TOKENS[0])
        .expect("recorded capture (walk)");
    gp.end_capture().expect("end capture (walk)");
    gp.buffers.reset_for_replay().expect("reset (walk)");

    // BOTH arms must start from an empty arena. The capture's two EAGER
    // warm-ups advanced the KV arena's layer totals, so the graph's history
    // carries two rows the walk's graph does not have — and the attention
    // then reads a different number of rows, which shows up at step 1 (the
    // step-0 rows are the same token, so the difference is small there and
    // grows).
    graph.buffers.reset_for_replay().expect("reset (replay)");

    let mut replay_logits = Vec::new();
    for tok in TOKENS {
        model.forward_replay(&mut graph, tok).expect("graph replay");
        replay_logits.push(graph.read_logits_f32().expect("replay logits"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    let mut walk_logits = Vec::new();
    for tok in TOKENS {
        model
            .forward_interpret_ir(&mut gp, tok)
            .expect("walk interpret");
        walk_logits.push(gp.read_logits_f32().expect("walk logits"));
        gp.buffers.current_pos = gp.buffers.current_pos.wrapping_add(1);
    }
    for (step, (a, b)) in replay_logits.iter().zip(walk_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!(
            "[iq4nl-row] step {step}: the walk vs the graph replay, worst {e:.3e} at {at} (abs {abs:.3e})"
        );
        assert!(
            e <= 1e-4,
            "the IQ4_NL model's walk diverges from the captured graph at step {step}: \
             worst {e:.3e} at {at}"
        );
    }
    unsafe { std::env::remove_var("GRIM_EMIT_TASK_GRAPH") };
}

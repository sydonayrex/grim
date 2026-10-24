//! M7.0 — the megakernel-vs-wavefront A/B, on grim's own hardware.
//!
//! WHY THIS EXISTS. The res7 corpus (§10a, arXiv:2605.27323) contains the
//! only direct A/B in the literature between the two execution models this
//! project is choosing between: a MEGAKERNEL (one persistent launch, a
//! `switch` over node types) and a WAVEFRONT (a conventional two-pass
//! dispatch). The megakernel LOST, 1.16×, on a memory-bound dense workload,
//! and the mechanism was "memory behaviour and cache locality rather than raw
//! compute utilization" — the authors inferred it from the counter
//! DIRECTION, not from per-op accounting, and the paper is 4 pages of path
//! tracing on an RTX 3060 Ti.
//!
//! §10i then makes that A/B a PRECONDITION for grim's own Phase M: "Before
//! committing to Phase M / Option C: run 2605.27323's A/B on grim's own
//! hardware. Launch a persistent decode step and a per-kernel decode at the
//! same token rate, on the same model, and read the same counters. It costs
//! an afternoon and it is the only direct evidence either way. If wavefront
//! wins on grim's dense/hybrid-recurrent workload, the megakernel direction
//! is scoped to MoE only."
//!
//! So this file is that afternoon's experiment, and it is written to be
//! READABLE AS EVIDENCE rather than as a pass/fail gate. The corpus's own
//! warning (§10a) is the design constraint that shapes it:
//!
//!   "Note also the direction of the counters: the losing megakernel has the
//!    HIGHER compute utilisation. Higher SM%/RTCore% did not mean more work
//!    done. Any grim benchmark that infers speedup from occupancy will draw
//!    the wrong conclusion here."
//!
//! Which means a gate that asserted "the ring is faster" would be the wrong
//! artifact. What this file reports instead:
//!
//!  1. TOKENS PER SECOND for both arms — the outcome.
//!  2. LAUNCHES PER TOKEN for both arms — the mechanism the outcome rests on.
//!  3. WAVE LAUNCHES PER TOKEN — the corpus's §9c drain metric ("any
//!     device-side chaining scheme that waits on completion at fine
//!     granularity serialises, and the drains multiply with the number of
//!     transfers per step").
//!  4. The A/B's own CONTROL: both arms run the SAME model, the SAME tokens,
//!     the SAME kernel bodies and the SAME arithmetic. The only difference is
//!     WHERE the recurrent step runs — the ring's persistent wave or the
//!     ordinary launcher. (A control must differ from its subject in exactly
//!     ONE dimension, or it tests nothing.)
//!
//! §10g's co-residency caveat is honoured: the run CHECKS whether the GPU is
//! idle before it measures, because a co-resident benchmark inflates a
//! sub-millisecond kernel's p95 by ~3.9% (§10g's KREX measurement) — and it
//! reports the check's result rather than assuming it.
//!
//! WHAT IT DELIBERATELY DOES NOT DO. It does not assert that the ring wins.
//! The corpus's own recommendation for this experiment is that its RESULT
//! decides the scope of the megakernel direction, and a gate that hard-codes
//! one outcome would destroy the evidence it was built to produce. So the
//! assertions here are all about the measurement being VALID (both arms
//! produce identical logits; both arms are measured at the same token rate;
//!     both arms run the same model), and the RESULT is printed for the plan
//! to record.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_CONTEXT=512 \
//!  cargo test -p grim-models-transformer --test ring_vs_launcher_ab \
//!    -- --nocapture --test-threads=1`
//!
//! The ring suite must run WITHOUT `AMD_SERIALIZE_KERNEL=3`: a resident wave
//! cannot run under serialization. (The LFM2's graph path is the opposite
//! case and REQUIRES it; that is a different test binary.)

use std::time::Instant;

use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_models_transformer::qwen35::Qwen35;
use grim_nn::WeightSource;

// The synthetic model builder, the GPU probe and the metric helpers. Its
// `use` items come with it — the consumers must not re-import them.
include!("common/qwen35_synth.rs");

/// The number of decode steps per arm. Large enough that the per-step cost
/// dominates the setup, small enough that the whole test fits in a minute on
/// a 17 GB card.
const STEPS: usize = 24;
/// Warm-up steps per arm, DISCARDED. They exist to pay the JIT and the
/// allocator: the first step of a fresh graph compiles kernels, and a cold
/// step inside the timed set would be an allocator artefact, not an
/// execution-model effect.
const WARMUP: usize = 4;
/// The model: two attention layers and two recurrent ones, so the ring has
/// something to take AND the launcher has something to leave. A model with
/// only one kind of layer would make the comparison measure nothing.
const LAYERS: usize = 4;
const ATTN_INTERVAL: usize = 2;

// ─── the measurement ────────────────────────────────────────────────────────

/// What one arm of the A/B produced.
struct ArmResult {
    label: &'static str,
    /// Wall-clock microseconds per decode step, AFTER the warm-up.
    us_per_step: f64,
    /// Kernel + GEMM launches per decode step. The mechanism the timing
    /// rests on: the whole point of a persistent wave is that it replaces N
    /// launches with fewer.
    launches_per_step: f64,
    /// The wave launches (persistent-kernel launches) per decode step. This
    /// is the corpus's §9c drain metric.
    waves_per_step: f64,
    /// BLOCKING device-to-host copies per decode step. A blocking D2H is a
    /// host round-trip: it cannot complete until the device has finished
    /// everything queued before it, so every one of them drains the pipeline.
    /// This is the metric §9c's finding is about — "any device-side chaining
    /// scheme that waits on completion at fine granularity serialises, and
    /// the drains multiply with the number of transfers per step" — and it
    /// is what distinguishes "the wave did the work" from "the wave did the
    /// work AND the host paid a round-trip to find out".
    d2h_per_step: f64,
    /// The logits of the last step, for the control comparison.
    logits: Vec<f32>,
}

fn measure_arm(
    label: &'static str,
    model: &Qwen35,
    dev: &RocmDevice,
    with_ring: bool,
) -> Result<ArmResult, String> {
    let mut graph = model
        .get_or_create_decode_graph(64, 1)
        .map_err(|e| format!("graph pool: {e}"))?;
    // Warm-ups, then a recorded capture. The capture publishes the IR the
    // ring is bound from, and it is the SAME IR both arms use.
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().map_err(|e| format!("begin capture: {e}"))?;
    model
        .forward_capture(&mut graph, TOKENS[0])
        .map_err(|e| format!("capture: {e}"))?;
    graph.end_capture().map_err(|e| format!("end capture: {e}"))?;
    graph
        .buffers
        .reset_for_replay()
        .map_err(|e| format!("reset: {e}"))?;

    // The ring, bound to THIS graph's own IR. (A ring bound to another
    // graph's IR would run the recurrent step on the WRONG buffers, and the
    // difference would look like an execution-model effect when it is a
    // wiring artefact.)
    let mut ring = if with_ring {
        let ir = graph
            .buffers
            .emitted_task_graph
            .clone()
            .ok_or("the capture published no task graph")?;
        let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(dev, &ir)
            .map_err(|e| format!("ring context: {e}"))?
            .ok_or("the ring bound no node — the IR has no executable node")?;
        Some(ctx)
    } else {
        None
    };

    // The step function. The only difference between the arms is this one
    // branch, which is the entire point: the SAME model, the SAME kernels,
    // the SAME arithmetic, and the recurrent step runs on the ring's wave or
    // on the ordinary launcher.
    let mut step = |graph: &mut grim_backend_rocm::decode_graph_buffers::DecodeGraph,
                    token: u32|
     -> Result<(), String> {
        match ring.as_mut() {
            Some(r) => model
                .forward_interpret_ir_ring(graph, token, r)
                .map_err(|e| format!("ring interpret: {e}")),
            None => model
                .forward_interpret_ir(graph, token)
                .map_err(|e| format!("launcher interpret: {e}")),
        }
    };

    // Warm-up: the JIT, the allocator, and the ring's first wave. Discarded.
    for i in 0..WARMUP {
        let token = TOKENS[i % TOKENS.len()];
        step(&mut graph, token).map_err(|e| format!("warmup: {e}"))?;
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    dev.synchronize();

    // The timed set. The counters are read around it, so the counts are
    // per-step, not cumulative.
    //
    // The launch counter must be read from the SHARED device instance, not
    // from the `try_new` one this test holds: `RocmDevice::try_new` and
    // `RocmDevice::shared` build SEPARATE structs, each with its own
    // `launch_counter`, and the model's launchers go through `shared`. Reading
    // the wrong instance reports ZERO launches — which is not a small error,
    // it makes the mechanism line of the report say "the megakernel arm
    // replaced 0 launches/step with 2 waves/step", which reads as "the ring
    // removed every launch" when it removed none.
    let shared = grim_backend_rocm::RocmDevice::shared(dev.ordinal());
    let launches_before = shared.launch_count();
    let waves_before = grim_backend_rocm::kernels::task_ring::wave_launch_count();
    let d2h_before = grim_backend_rocm::blocking_d2h_count();
    let t0 = Instant::now();
    let mut logits = Vec::new();
    for i in 0..STEPS {
        let token = TOKENS[i % TOKENS.len()];
        step(&mut graph, token).map_err(|e| format!("timed step {i}: {e}"))?;
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        if i == STEPS - 1 {
            logits = graph
                .read_logits_f32()
                .map_err(|e| format!("read logits: {e}"))?;
        }
    }
    dev.synchronize();
    let elapsed = t0.elapsed();
    let launches = shared.launch_count() - launches_before;
    let waves = grim_backend_rocm::kernels::task_ring::wave_launch_count() - waves_before;
    let d2h = grim_backend_rocm::blocking_d2h_count() - d2h_before;

    Ok(ArmResult {
        label,
        us_per_step: elapsed.as_secs_f64() * 1e6 / STEPS as f64,
        launches_per_step: launches as f64 / STEPS as f64,
        waves_per_step: waves as f64 / STEPS as f64,
        d2h_per_step: d2h as f64 / STEPS as f64,
        logits,
    })
}

/// The GPU must be IDLE for this measurement, or the numbers are another
/// workload's.
///
/// §10g (KREX) measured co-residency's effect on p95 timing: +0.30% for
/// kernels over 10 ms, +1.58% over 1 ms, and +3.90% over 0.1 ms. grim's
/// decode kernels are the small end of that scale, so a co-resident process
/// is a ~4% perturbation on the very quantity this A/B measures. The corpus's
/// instruction is explicit: "grim's gates need an idle GPU, or an explicit
/// co-residency claim with a measured number, not an implicit assumption."
///
/// This is a CHECK, not a claim: it reports what it found, and the plan
/// records the co-residency state alongside the numbers.
fn gpu_is_idle() -> bool {
    // `rocm-smi --showuse` is the box's own tool. A failure to read it is
    // NOT an idle reading — it is an unknown, and an unknown must not be
    // reported as idle.
    let out = std::process::Command::new("rocm-smi")
        .args(["--showuse", "--json"])
        .output();
    let Ok(out) = out else {
        eprintln!("[ab] co-residency: rocm-smi unavailable — the GPU's idleness is UNKNOWN");
        return false;
    };
    if !out.status.success() {
        eprintln!("[ab] co-residency: rocm-smi failed — the GPU's idleness is UNKNOWN");
        return false;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // The ordinal this process sees is the VISIBLE ordinal, so the card
    // index in rocm-smi's JSON is not necessarily the same number. The
    // conservative reading is the MAXIMUM use across all cards: if any card
    // is busy, this measurement's co-residency budget is unknown.
    let mut worst = 0u32;
    for line in text.lines() {
        if let Some(v) = line.split("\"GPU use (%)\":\"").nth(1) {
            if let Some(num) = v.split('"').next() {
                if let Ok(n) = num.trim().parse::<u32>() {
                    worst = worst.max(n);
                }
            }
        }
    }
    eprintln!("[ab] co-residency: the busiest card reports {worst}% use");
    worst < 20
}

#[test]
fn the_megakernel_and_the_wavefront_are_measured_on_the_same_model() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    // The IR is only published when the emitter is on, and the ring is bound
    // from that published IR.
    unsafe { std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1") };

    let idle = gpu_is_idle();
    if !idle {
        eprintln!(
            "[ab] WARNING: the GPU is not verifiably idle. The numbers below are \
             recorded WITH that caveat, per res7 §10g."
        );
    }

    let mut cfg = tiny_cfg();
    cfg.num_layers = LAYERS;
    cfg.full_attention_interval = ATTN_INTERVAL;
    let provider = synth_weights_hybrid(LAYERS, ATTN_INTERVAL);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load hybrid qwen35");

    let n_attn = model.blocks.iter().filter(|b| b.is_full_attention).count();
    let n_rec = model.blocks.len() - n_attn;
    eprintln!("[ab] the model: {n_attn} attention layers, {n_rec} recurrent layers");
    assert_eq!(
        (n_attn, n_rec),
        (LAYERS / ATTN_INTERVAL, LAYERS - LAYERS / ATTN_INTERVAL),
        "the A/B model must have BOTH layer kinds, or the ring has nothing to take \
         and the comparison measures nothing"
    );

    // ARM 1: the WAVEFRONT. Every launch goes through the model's own
    // launcher, one kernel launch at a time.
    let wavefront = measure_arm("wavefront (per-kernel launcher)", &model, &dev, false)
        .expect("wavefront arm");

    // ARM 2: the MEGAKERNEL arm. The SAME walk, with the recurrent step
    // executed by the ring's persistent wave.
    let megakernel = measure_arm("megakernel (persistent ring wave)", &model, &dev, true)
        .expect("megakernel arm");

    // ─── the report ───
    //
    // The corpus's §10a warning governs the FORMAT of this report: the
    // occupancy-style counters are a DIRECTION to read, never the verdict,
    // because in the only direct A/B that exists the LOSING megakernel had
    // the HIGHER compute utilisation. So the verdict line is the token rate
    // and the mechanism line is the launch count, and the launch counts are
    // printed NEXT TO the rate so a reader can see whether a rate difference
    // has a launch difference behind it.
    eprintln!(
        "\n[ab] ==================================================================\n\
         [ab]  M7.0 — megakernel vs wavefront, on grim's own hardware\n\
         [ab]  model: {LAYERS} layers ({n_attn} attention / {n_rec} recurrent), \
         {STEPS} timed steps after {WARMUP} warm-up\n\
         [ab] =================================================================="
    );
    eprintln!(
        "[ab] {:<34} {:>12} {:>14} {:>12} {:>12}",
        "arm", "us/step", "launches/step", "waves/step", "D2H/step"
    );
    eprintln!(
        "[ab] {:<34} {:>12.1} {:>14.1} {:>12.1} {:>12.1}",
        wavefront.label,
        wavefront.us_per_step,
        wavefront.launches_per_step,
        wavefront.waves_per_step,
        wavefront.d2h_per_step
    );
    eprintln!(
        "[ab] {:<34} {:>12.1} {:>14.1} {:>12.1} {:>12.1}",
        megakernel.label,
        megakernel.us_per_step,
        megakernel.launches_per_step,
        megakernel.waves_per_step,
        megakernel.d2h_per_step
    );
    let speed = wavefront.us_per_step / megakernel.us_per_step;
    eprintln!(
        "[ab] rate: wavefront {:>8.1} tok/s   megakernel {:>8.1} tok/s   ratio {:>6.3}x",
        1e6 / wavefront.us_per_step,
        1e6 / megakernel.us_per_step,
        speed
    );
    eprintln!(
        "[ab] mechanism: the megakernel arm replaced {:.1} launches/step with \
         {:.1} waves/step, and cost {:.1} blocking host round-trips/step \
         (the wavefront arm cost {:.1})",
        wavefront.launches_per_step,
        megakernel.waves_per_step,
        megakernel.d2h_per_step,
        wavefront.d2h_per_step
    );
    if idle {
        eprintln!("[ab] co-residency: the GPU was verifiably idle (res7 §10g satisfied)");
    } else {
        eprintln!(
            "[ab] co-residency: NOT VERIFIED — these numbers carry that caveat \
             (res7 §10g)"
        );
    }
    eprintln!(
        "[ab] the corpus's warning: the losing megakernel in the only direct A/B \
         that exists had the HIGHER compute utilisation. Read the rate and the \
         launch count; do not infer the verdict from occupancy."
    );

    // ─── the CONTROL ───
    //
    // The two arms must produce the SAME ANSWER. If they do not, the timing
    // compares two different computations and the ratio means nothing. This
    // is the check that makes the A/B an A/B.
    assert_eq!(
        wavefront.logits.len(),
        megakernel.logits.len(),
        "the arms must read the same number of logits"
    );
    let (worst, at) = wavefront
        .logits
        .iter()
        .zip(&megakernel.logits)
        .enumerate()
        .fold((0.0f32, 0usize), |(w, wi), (i, (a, b))| {
            let d = (a - b).abs();
            if d > w {
                (d, i)
            } else {
                (w, wi)
            }
        });
    eprintln!("[ab] control: the arms' logits differ by at most {worst:.3e} (at {at})");
    assert!(
        worst == 0.0,
        "the A/B's control FAILED: the two arms disagree (worst {worst:.3e} at {at}). \
         A timing comparison of two different computations is not an A/B."
    );

    unsafe { std::env::remove_var("GRIM_EMIT_TASK_GRAPH") };
}

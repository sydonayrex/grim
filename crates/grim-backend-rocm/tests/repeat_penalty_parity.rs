//! B1 done-check (PLAN-reduce-d2h-h2d): device repeat-penalty pre-pass
//! (`grim_repeat_penalty_apply`) reproduces the CPU `apply_repeat_penalty`
//! semantics bit-for-bit, and greedy penalty-aware device sampling agrees
//! with the CPU argmax-over-penalized-logits. Env-gated per house rule —
//! run on real hardware with `GRIM_RUN_GPU_TESTS=1`.

use grim_backend_rocm::{
    CoreTensorOps, RocmDevice, Shape, apply_repeat_penalty_on_device, as_rocm, dev_ptr,
    sample_logits_on_device_with_penalty,
};
use grim_tensor::{BackendStorage, DType};

/// Inline CPU oracle mirroring `grim-core/src/sampler.rs::apply_repeat_penalty`
/// (dedup via HashSet; `<0 ? *p : /p`; out-of-range ids skipped).
fn cpu_apply_penalty(logits: &[f32], penalty: f32, history: &[u32]) -> Vec<f32> {
    if penalty <= 1.0 || history.is_empty() {
        return logits.to_vec();
    }
    let mut out = logits.to_vec();
    let mut seen = std::collections::HashSet::with_capacity(history.len().min(1024));
    for &tok in history {
        if !seen.insert(tok) {
            continue;
        }
        let i = tok as usize;
        if i < out.len() {
            if out[i] < 0.0 {
                out[i] *= penalty;
            } else {
                out[i] /= penalty;
            }
        }
    }
    out
}

fn cpu_argmax_first(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        // NaN never wins (matches CPU `partial_cmp` fallback ordering).
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Deterministic pseudo-random logits covering negatives, zeros, infinities
/// and NaN (fixed indices so failures are reproducible).
fn test_logits(vocab: usize) -> Vec<f32> {
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        state = state
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0x1405_7B7E_F767_814F);
        ((state >> 33) as f32) / (u32::MAX as f32) * 20.0 - 10.0
    };
    let mut v: Vec<f32> = (0..vocab).map(|_| next()).collect();
    if vocab > 3 {
        v[1] = f32::NEG_INFINITY;
    }
    if vocab > 5 {
        v[3] = f32::INFINITY;
    }
    if vocab > 7 {
        v[5] = f32::NAN;
    }
    if vocab > 9 {
        v[7] = 0.0;
    }
    v
}

fn gpu_dev() -> Option<RocmDevice> {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return None;
    }
    match RocmDevice::try_new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("ROCm device 0 not available: {e}");
            None
        }
    }
}

#[test]
fn repeat_penalty_prepass_matches_cpu_bit_for_bit() {
    let Some(dev) = gpu_dev() else { return };
    let vocab = 512usize;
    let base = test_logits(vocab);
    let histories: Vec<Vec<u32>> = vec![
        vec![],
        vec![0, 1, 2, 3],
        vec![7, 7, 7, 7],             // all-duplicate history
        vec![0, 511, 3, 0, 511, 5],   // dupes + NaN/inf indices
        vec![999_999, u32::MAX],      // out-of-range ids (must be skipped)
        (0..300u32).map(|i| (i * 7) % 600).collect(), // long, dupes + OOR mix
    ];
    for penalty in [1.0f32, 1.1, 1.5, 2.0] {
        for hist in &histories {
            let shape = Shape::new(vec![vocab]);
            let storage = dev.from_cpu(&base, &shape, DType::F32).expect("upload");
            let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");
            let ptr = dev_ptr(rocm_st).expect("logits ptr");
            // Host-dedup mirrors the launcher contract (unique ids in).
            let mut seen = std::collections::HashSet::new();
            let mut uniq = Vec::new();
            for &t in hist {
                if seen.insert(t) {
                    uniq.push(t);
                }
            }
            apply_repeat_penalty_on_device(&dev, ptr, vocab, &uniq, penalty)
                .expect("penalty pre-pass");
            let got = rocm_st.to_cpu_vec_f32().expect("readback");
            let want = cpu_apply_penalty(&base, penalty, hist);
            assert_eq!(got.len(), want.len(), "len p={penalty} hist={hist:?}");
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits(),
                    "bit mismatch p={penalty} hist={hist:?} [{i}]: got {g} want {w}"
                );
            }
        }
    }
}

#[test]
fn repeat_penalty_greedy_device_matches_cpu_token() {
    let Some(dev) = gpu_dev() else { return };
    let vocab = 512usize;
    let base = test_logits(vocab);
    let histories: Vec<Vec<u32>> = vec![
        vec![],
        vec![10, 20, 30],
        vec![3, 3, 3], // +inf index penalized repeatedly (dedup ⇒ once)
        (0..vocab as u32).step_by(3).collect(),
    ];
    for penalty in [1.0f32, 1.1, 2.0] {
        for hist in &histories {
            let shape = Shape::new(vec![vocab]);
            let storage = dev.from_cpu(&base, &shape, DType::F32).expect("upload");
            let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");
            let got = sample_logits_on_device_with_penalty(
                &dev,
                rocm_st,
                vocab,
                0.0, // greedy
                0,
                1.0,
                0xB1,
                penalty,
                hist,
            )
            .expect("device greedy+penalty")
            .expect("device greedy+penalty returned None");
            let want = cpu_argmax_first(&cpu_apply_penalty(&base, penalty, hist));
            assert_eq!(
                got, want,
                "greedy token mismatch p={penalty} hist-len={}",
                hist.len()
            );
        }
    }
}

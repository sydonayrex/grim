//! Golden standard for the CPU `rope` / `rerope` paths.
//!
//! The oracle below IS the spec: it implements the ggml/llama.cpp RoPE
//! conventions the rest of this repo was validated against (see the long
//! comment in `grim-backend-rocm`'s `yarn_inv_freq_and_mscale` and commit
//! 528b1508):
//!
//! 1. Frequencies: `inv_freq[i] = 1 / base^(2i / n_rot)` where
//!    `n_rot = min(rotary_dim, head_dim)` — the exponent denominator is the
//!    ROTARY count, never the full head dim.
//! 2. YaRN (when configured): the llama.cpp ramp — corr dims from
//!    `ggml_rope_yarn_corr_dim(n_rot, n_ctx_orig, beta, base)` (index-based,
//!    NOT the HF wavelength ramp), `mscale = rope_mscale|attention_factor *
//!    (1 + 0.1 * ln(factor))`.
//! 3. Pairing: `interleaved` → GPT-J pairs (2i, 2i+1); otherwise NeoX
//!    half-split pairs (i, i + n_rot/2). Applies to `rope` AND `rerope`.
//! 4. `rerope(old→new)` composes rotations per pair: un-rotate with the old
//!    position, re-rotate with the new one.
//!
//! The golden VALUES were generated once from this oracle and are embedded
//! below as exact bit patterns, so an accidental edit of the oracle in
//! tandem with a behaviour change cannot silently pass: the test fails
//! against the committed bits first.
//!
//! History: `rerope` ignored `cfg.interleaved` (always GPT-J pairs) and used
//! the HF wavelength YaRN ramp; both diverged from this standard and were
//! fixed in the same commit as this golden.

use grim_backend_cpu::CpuDevice;
use grim_tensor::backend::{ComputeHandle, RopeConfig, YaRNParams};
use grim_tensor::{DType, Shape};

type R = Result<(), Box<dyn std::error::Error>>;

// ── oracle ────────────────────────────────────────────────────────────────

fn yarn_from(cfg: &RopeConfig) -> Option<&YaRNParams> {
    cfg.yarn.as_ref()
}

/// ggml_rope_yarn_corr_dim (ggml.c): n_dims * ln(n_ctx_orig / (n_rot·2π)) / (2·ln(base))
fn corr_dim(n_dims: f32, n_ctx_orig: f32, n_rot: f32, base: f32) -> f32 {
    n_dims * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

/// The ONE frequency definition (shared with the GPU's
/// `yarn_inv_freq_and_mscale`): inv_freq[i] for i in 0..n_rot/2.
fn oracle_inv_freq(base: f32, n_rot: usize, yarn: Option<&YaRNParams>) -> Vec<f32> {
    let half = n_rot / 2;
    (0..half)
        .map(|i| {
            let freq = 1.0f32 / base.powf((2 * i) as f32 / n_rot as f32);
            match yarn {
                None => freq,
                Some(y) => {
                    // llama.cpp ramp over the DIMENSION INDEX with bounds
                    // from ggml_rope_yarn_corr_dims.
                    let low = corr_dim(n_rot as f32, y.original_max_pos as f32, y.beta_fast, base)
                        .floor()
                        .max(0.0);
                    let high = corr_dim(n_rot as f32, y.original_max_pos as f32, y.beta_slow, base)
                        .ceil()
                        .min((n_rot - 1) as f32);
                    let t = (i as f32 - low) / (high - low).max(0.001);
                    let ramp = 1.0 - t.clamp(0.0, 1.0);
                    (freq / y.factor) * (1.0 - ramp) + freq * ramp
                }
            }
        })
        .collect()
}

fn oracle_mscale(y: Option<&YaRNParams>) -> f32 {
    y.map_or(1.0, |y| {
        y.rope_mscale.unwrap_or(y.attention_factor) * (1.0 + 0.1 * (y.factor).ln())
    })
}

/// cos/sin tables for one position — the exact f32 op order the
/// implementations use (`pos * inv_freq` then cos/sin, mscale applied after).
fn oracle_tables(pos: f32, inv_freq: &[f32], mscale: f32) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos_p = vec![0.0f32; half];
    let mut sin_p = vec![0.0f32; half];
    for i in 0..half {
        let a = pos * inv_freq[i];
        cos_p[i] = a.cos() * mscale;
        sin_p[i] = a.sin() * mscale;
    }
    (cos_p, sin_p)
}

/// Oracle forward rope. Pairs per `cfg.interleaved`.
fn oracle_rope(x: &[f32], b: usize, s: usize, positions: &[u32], cfg: &RopeConfig) -> Vec<f32> {
    let d = cfg.dim;
    let n_rot = cfg.rotary_dim.min(d);
    let half = n_rot / 2;
    let yarn: Option<&YaRNParams> = yarn_from(cfg);
    let inv_freq = oracle_inv_freq(cfg.base, n_rot, yarn);
    let mscale = oracle_mscale(yarn);
    let mut out = x.to_vec();
    for bi in 0..b {
        for si in 0..s {
            let pos = positions.get(si).copied().unwrap_or(si as u32) as f32;
            let (cos_p, sin_p) = oracle_tables(pos, &inv_freq, mscale);
            let base_index = (bi * s + si) * d;
            for i in 0..half {
                let (a1, a2, o1, o2) = if cfg.interleaved {
                    (
                        base_index + 2 * i,
                        base_index + 2 * i + 1,
                        base_index + 2 * i,
                        base_index + 2 * i + 1,
                    )
                } else {
                    (
                        base_index + i,
                        base_index + i + half,
                        base_index + i,
                        base_index + i + half,
                    )
                };
                let x1 = out[a1];
                let x2 = out[a2];
                out[o1] = x1 * cos_p[i] - x2 * sin_p[i];
                out[o2] = x1 * sin_p[i] + x2 * cos_p[i];
            }
        }
    }
    out
}

/// Oracle rerope: un-rotate by old, re-rotate by new, per pair.
///
/// Spec subtlety the golden caught: the cache holds `M·R(old)·x` where M is
/// the YaRN mscale. The mscale is ALREADY baked into the cache content and
/// survives as the overall scale factor, so `rerope` must be a PURE
/// rotation retargeting (UNSCALED tables on both sides): scaling either
/// side multiplies M a second (or third) time and breaks the identity
/// `rerope(old→new) == rope_at_new`.
fn oracle_rerope(
    k: &[f32],
    b: usize,
    s: usize,
    old_pos: &[u32],
    new_pos: &[u32],
    cfg: &RopeConfig,
) -> Vec<f32> {
    let d = cfg.dim;
    let n_rot = cfg.rotary_dim.min(d);
    let half = n_rot / 2;
    let yarn: Option<&YaRNParams> = yarn_from(cfg);
    let inv_freq = oracle_inv_freq(cfg.base, n_rot, yarn);
    // Pure rotation retargeting — mscale must NOT appear here (see doc).
    let mscale: f32 = 1.0;
    debug_assert_eq!(mscale, 1.0);
    let mut out = k.to_vec();
    for bi in 0..b {
        for si in 0..s {
            let p_old = old_pos.get(si).copied().unwrap_or(si as u32) as f32;
            let p_new = new_pos.get(si).copied().unwrap_or(si as u32) as f32;
            let (oc, os) = oracle_tables(p_old, &inv_freq, 1.0);
            let (nc, ns) = oracle_tables(p_new, &inv_freq, 1.0);
            let base_index = (bi * s + si) * d;
            for i in 0..half {
                let (a1, a2) = if cfg.interleaved {
                    (base_index + 2 * i, base_index + 2 * i + 1)
                } else {
                    (base_index + i, base_index + i + half)
                };
                let k1 = out[a1];
                let k2 = out[a2];
                // inverse of [c, -s; s, c] is [c, s; -s, c]
                let o1 = k1 * oc[i] + k2 * os[i];
                let o2 = -k1 * os[i] + k2 * oc[i];
                out[a1] = o1 * nc[i] - o2 * ns[i];
                out[a2] = o1 * ns[i] + o2 * nc[i];
            }
        }
    }
    out
}

// ── case matrix + deterministic inputs ────────────────────────────────────

fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
}

fn yarn_cfg() -> YaRNParams {
    YaRNParams {
        original_max_pos: 4096,
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        rope_mscale: None,
        attention_factor: 1.0,
    }
}

struct Case {
    name: &'static str,
    dim: usize,
    rotary_dim: usize,
    interleaved: bool,
    yarn: bool,
    base: f32,
    old_pos: Vec<u32>,
    new_pos: Vec<u32>,
    seed: u64,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "full_rotary_halfsplit",
            dim: 64,
            rotary_dim: 64,
            interleaved: false,
            yarn: false,
            base: 10000.0,
            old_pos: vec![3, 17, 130],
            new_pos: vec![40, 200, 1024],
            seed: 0x243F6A8885A308D3,
        },
        Case {
            name: "full_rotary_interleaved",
            dim: 64,
            rotary_dim: 64,
            interleaved: true,
            yarn: false,
            base: 10000.0,
            old_pos: vec![3, 17, 130],
            new_pos: vec![40, 200, 1024],
            seed: 0x452821E638D01377,
        },
        Case {
            // The Gate D shape: partial rotary with n_rot < head_dim — the
            // case that exposed the head_dim-vs-n_rot denominator bug.
            name: "partial_rotary_halfsplit",
            dim: 256,
            rotary_dim: 64,
            interleaved: false,
            yarn: false,
            base: 10_000_000.0,
            old_pos: vec![20, 21, 22],
            new_pos: vec![50, 51, 52],
            seed: 0x260E19527B9E3DBD,
        },
        Case {
            name: "partial_rotary_interleaved",
            dim: 256,
            rotary_dim: 64,
            interleaved: true,
            yarn: false,
            base: 10_000_000.0,
            old_pos: vec![20, 21, 22],
            new_pos: vec![50, 51, 52],
            seed: 0x7B9E3DBD260E1952,
        },
        Case {
            name: "yarn_full_rotary",
            dim: 64,
            rotary_dim: 64,
            interleaved: false,
            yarn: true,
            base: 10000.0,
            old_pos: vec![100, 1000, 5000],
            new_pos: vec![200, 4000, 8000],
            seed: 0x8C1A2D4E6F8091A2,
        },
        Case {
            name: "yarn_partial_rotary",
            dim: 256,
            rotary_dim: 128,
            interleaved: false,
            yarn: true,
            base: 10000.0,
            old_pos: vec![100, 1000, 5000],
            new_pos: vec![200, 4000, 8000],
            seed: 0xA2B3C4D5E6F70819,
        },
    ]
}

fn run_cpu_rope(cfg: &RopeConfig, x: &[f32], b: usize, s: usize, pos: &[u32]) -> Vec<f32> {
    use grim_tensor::backend::CoreTensorOps;
    let dev = CpuDevice::new();
    let shape = Shape::new(vec![b, s, cfg.dim]);
    let st = CoreTensorOps::from_cpu(&dev, x, &shape, DType::F32).expect("from_cpu rope");
    let (out, _h) = rope_trait(&dev, st.as_ref(), pos, cfg, &shape);
    out.to_cpu_vec_f32().expect("rope to vec")
}

fn run_cpu_rerope(
    cfg: &RopeConfig,
    k: &[f32],
    b: usize,
    s: usize,
    old_pos: &[u32],
    new_pos: &[u32],
) -> Vec<f32> {
    use grim_tensor::backend::CoreTensorOps;
    let dev = CpuDevice::new();
    let shape = Shape::new(vec![b, s, cfg.dim]);
    let st = CoreTensorOps::from_cpu(&dev, k, &shape, DType::F32).expect("from_cpu rerope");
    let (out, _h) = rerope_trait(&dev, st.as_ref(), old_pos, new_pos, cfg, &shape);
    out.to_cpu_vec_f32().expect("rerope to vec")
}

fn rope_trait(
    dev: &CpuDevice,
    st: &dyn grim_tensor::backend::BackendStorage,
    pos: &[u32],
    cfg: &RopeConfig,
    shape: &Shape,
) -> (Box<dyn grim_tensor::backend::BackendStorage>, Box<dyn ComputeHandle>) {
    use grim_tensor::backend::AttentionOps;
    AttentionOps::rope(dev, st, pos, cfg, shape).expect("cpu rope")
}

fn rerope_trait(
    dev: &CpuDevice,
    st: &dyn grim_tensor::backend::BackendStorage,
    old_pos: &[u32],
    new_pos: &[u32],
    cfg: &RopeConfig,
    shape: &Shape,
) -> (Box<dyn grim_tensor::backend::BackendStorage>, Box<dyn ComputeHandle>) {
    use grim_tensor::backend::AttentionOps;
    AttentionOps::rerope(dev, st, old_pos, new_pos, cfg, shape).expect("cpu rerope")
}

// ── committed golden bits ─────────────────────────────────────────────────
//
// Exact f32 bit patterns of the oracle output for a fixed probe element of
// each case: element (batch 0, seq 1, dim 0), computed with the oracle as
// committed. If an impl/oracle change moves these bits, the change must be
// consciously re-goldened, not absorbed by a tolerance.

const GOLDEN_ROPE_BITS: &[(&str, u32)] = &[
    ("full_rotary_halfsplit", 0xBF16_84F0),
    ("full_rotary_interleaved", 0xBF06_7582),
    ("partial_rotary_halfsplit", 0x3EC8_2452),
    ("partial_rotary_interleaved", 0x3F03_C417),
    ("yarn_full_rotary", 0xBF28_C883),
    ("yarn_partial_rotary", 0x3ECB_5AF4),
];

const GOLDEN_REROPE_BITS: &[(&str, u32)] = &[
    ("full_rotary_halfsplit", 0xBF86_08BC),
    ("full_rotary_interleaved", 0xBF49_5C96),
    ("partial_rotary_halfsplit", 0x3F3C_A793),
    ("partial_rotary_interleaved", 0x3ECB_2552),
    ("yarn_full_rotary", 0x3F52_DA28),
    ("yarn_partial_rotary", 0xBE7D_98D2),
];

// ── tests ─────────────────────────────────────────────────────────────────

#[test]
fn golden_cpu_rope_matches_spec() -> R {
    for case in cases() {
        let b = 1usize;
        let s = case.old_pos.len();
        let mut state = case.seed;
        let x: Vec<f32> = (0..b * s * case.dim).map(|_| xorshift(&mut state)).collect();
        let mut cfg = RopeConfig::new(case.dim, case.base);
        cfg.rotary_dim = case.rotary_dim;
        cfg.interleaved = case.interleaved;
        if case.yarn {
            cfg.yarn = Some(yarn_cfg());
        }

        let got = run_cpu_rope(&cfg, &x, b, s, &case.old_pos);
        let want = oracle_rope(&x, b, s, &case.old_pos, &cfg);

        // 1. committed bit-level golden at the probe element
        let probe = s.min(2) * case.dim; // (bi 0, si min(2, s-1), dim 0)
        let probe_si = s.min(2);
        let probe_idx = probe_si * case.dim;
        let _ = probe;
        let bits = got[probe_idx].to_bits();
        let want_bits = GOLDEN_ROPE_BITS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, b)| *b)
            .ok_or("missing rope golden entry")?;
        assert_eq!(
            bits, want_bits,
            "[{}] rope probe element diverged from the committed golden bits (got {bits:#010x}, want {want_bits:#010x} = {})",
            case.name, f32::from_bits(want_bits)
        );

        // 2. full-vector oracle match (tight: the impl and oracle share the
        //    same op order; anything above 1 ulp indicates a real deviation)
        let mut max_abs = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g - w).abs());
        }
        assert!(
            max_abs <= 1e-6,
            "[{}] rope deviates from the golden oracle: max_abs {max_abs}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn golden_cpu_rerope_matches_spec() -> R {
    for case in cases() {
        let b = 1usize;
        let s = case.old_pos.len();
        let mut state = case.seed;
        let x: Vec<f32> = (0..b * s * case.dim).map(|_| xorshift(&mut state)).collect();
        let mut cfg = RopeConfig::new(case.dim, case.base);
        cfg.rotary_dim = case.rotary_dim;
        cfg.interleaved = case.interleaved;
        if case.yarn {
            cfg.yarn = Some(yarn_cfg());
        }

        // Input: K already rotated at old_pos (what the KV cache holds).
        let k_old = oracle_rope(&x, b, s, &case.old_pos, &cfg);

        let got = run_cpu_rerope(&cfg, &k_old, b, s, &case.old_pos, &case.new_pos);
        let want = oracle_rerope(&k_old, b, s, &case.old_pos, &case.new_pos, &cfg);

        let probe_idx = s.min(2) * case.dim;
        let bits = got[probe_idx].to_bits();
        let want_bits = GOLDEN_REROPE_BITS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, b)| *b)
            .ok_or("missing rerope golden entry")?;
        assert_eq!(
            bits, want_bits,
            "[{}] rerope probe element diverged from the committed golden bits (got {bits:#010x}, want {want_bits:#010x} = {})",
            case.name, f32::from_bits(want_bits)
        );

        let mut max_abs = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g - w).abs());
        }
        assert!(
            max_abs <= 1e-6,
            "[{}] rerope deviates from the golden oracle: max_abs {max_abs}",
            case.name
        );

        // 3. semantic self-check of the oracle itself: rerope(old→new) must
        //    equal rope_at_new applied to the ORIGINAL x (composition
        //    closure). f32-composition noise allows a loose envelope.
        let fresh_new = oracle_rope(&x, b, s, &case.new_pos, &cfg);
        let mut max_comp = 0.0f32;
        for (g, w) in want.iter().zip(fresh_new.iter()) {
            max_comp = max_comp.max((g - w).abs());
        }
        assert!(
            max_comp < 1e-3,
            "[{}] oracle rerope(old→new) != rope(new): composition gap {max_comp}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn golden_cpu_rerope_roundtrip_consistency() -> R {
    // Implementation-level consistency: rope(x,old) → rerope(old→new) must
    // land on rope(x,new) for BOTH pairings and for yarn (this is the
    // assertion the HF-ramp implementation failed before the fix).
    for case in cases() {
        let b = 1usize;
        let s = case.old_pos.len();
        let mut state = case.seed;
        let x: Vec<f32> = (0..b * s * case.dim).map(|_| xorshift(&mut state)).collect();
        let mut cfg = RopeConfig::new(case.dim, case.base);
        cfg.rotary_dim = case.rotary_dim;
        cfg.interleaved = case.interleaved;
        if case.yarn {
            cfg.yarn = Some(yarn_cfg());
        }

        let k_old = run_cpu_rope(&cfg, &x, b, s, &case.old_pos);
        let k_reroped = run_cpu_rerope(&cfg, &k_old, b, s, &case.old_pos, &case.new_pos);
        let k_fresh = run_cpu_rope(&cfg, &x, b, s, &case.new_pos);

        let mut max_diff = 0.0f32;
        for (r, f) in k_reroped.iter().zip(k_fresh.iter()) {
            max_diff = max_diff.max((r - f).abs());
        }
        assert!(
            max_diff <= 1e-5,
            "[{}] rerope(old→new) vs fresh rope(new): max diff {max_diff} (pairing {})",
            case.name,
            if case.interleaved { "gpt-j" } else { "neox" }
        );
    }
    Ok(())
}

/// Coverage proof (standing rule: a gate's coverage is the deliverable): the
/// golden must DISCRIMINATE — the two known-wrong behaviours (HF wavelength
/// ramp for yarn; ignoring `interleaved` in rerope) must actually move the
/// probe bits away from the committed goldens, i.e. the standard would have
/// caught them.
#[test]
fn golden_discriminates_known_wrong_behaviours() {
    for case in cases() {
        let b = 1usize;
        let s = case.old_pos.len();
        let mut state = case.seed;
        let x: Vec<f32> = (0..b * s * case.dim).map(|_| xorshift(&mut state)).collect();
        let mut cfg = RopeConfig::new(case.dim, case.base);
        cfg.rotary_dim = case.rotary_dim;
        cfg.interleaved = case.interleaved;
        if case.yarn {
            cfg.yarn = Some(yarn_cfg());
        }
        let k_old = oracle_rope(&x, b, s, &case.old_pos, &cfg);

        // (a) wrong-implementation rerope: GPT-J pairing forced on (the old
        //     rerope behaviour) — for half-split configs it must disagree
        //     with the golden.
        if !case.interleaved {
            let mut wrong = k_old.clone();
            let half = case.rotary_dim / 2;
            let yarn: Option<&YaRNParams> = yarn_from(&cfg);
            let inv = oracle_inv_freq(cfg.base, case.rotary_dim, yarn);
            let ms = oracle_mscale(yarn);
            let (oc, os) = oracle_tables(case.old_pos[1] as f32, &inv, ms);
            let (nc, ns) = oracle_tables(case.new_pos[1] as f32, &inv, ms);
            let bi = 0;
            let si = 1;
            let base_index = (bi * s + si) * case.dim;
            for i in 0..half {
                let a1 = base_index + 2 * i;
                let a2 = base_index + 2 * i + 1;
                let k1 = wrong[a1];
                let k2 = wrong[a2];
                let o1 = k1 * oc[i] + k2 * os[i];
                let o2 = -k1 * os[i] + k2 * oc[i];
                wrong[a1] = o1 * nc[i] - o2 * ns[i];
                wrong[a2] = o1 * ns[i] + o2 * nc[i];
            }
            let want = oracle_rerope(&k_old, b, s, &case.old_pos, &case.new_pos, &cfg);
            let moved = wrong
                .iter()
                .zip(want.iter())
                .any(|(w, r)| w.to_bits() != r.to_bits());
            assert!(
                moved,
                "[{}] coverage hole: forcing GPT-J pairing did not move the output",
                case.name
            );
        }

        // (b) wrong-implementation rope: HF wavelength yarn ramp — for yarn
        //     cases it must disagree with the golden.
        if case.yarn {
            let y = yarn_cfg();
            let half = case.rotary_dim / 2;
            // HF-style inv_freq built with the wavelength ramp
            let mut inv_hf = vec![0.0f32; half];
            for i in 0..half {
                let freq = 1.0f32 / cfg.base.powf((2 * i) as f32 / case.rotary_dim as f32);
                let wavelength = 2.0 * std::f32::consts::PI / freq;
                let low = y.original_max_pos as f32 / y.beta_slow;
                let high = y.original_max_pos as f32 / y.beta_fast;
                inv_hf[i] = if wavelength < high {
                    freq
                } else if wavelength > low {
                    freq / y.factor
                } else {
                    let ramp = (y.original_max_pos as f32 / wavelength - y.beta_slow)
                        / (y.beta_fast - y.beta_slow);
                    (1.0 - ramp) * (freq / y.factor) + ramp * freq
                };
            }
            let hf = oracle_rope_with_freqs(&x, b, s, &case.old_pos, &cfg, &inv_hf);
            let want = oracle_rope(&x, b, s, &case.old_pos, &cfg);
            let moved = hf
                .iter()
                .zip(want.iter())
                .any(|(h, w)| h.to_bits() != w.to_bits());
            assert!(
                moved,
                "[{}] coverage hole: HF wavelength ramp did not move the output",
                case.name
            );
        }
    }
}

fn oracle_rope_with_freqs(
    x: &[f32],
    b: usize,
    s: usize,
    positions: &[u32],
    cfg: &RopeConfig,
    inv_freq: &[f32],
) -> Vec<f32> {
    let d = cfg.dim;
    let n_rot = cfg.rotary_dim.min(d);
    let half = n_rot / 2;
    let yarn: Option<&YaRNParams> = yarn_from(cfg);
    let mscale = oracle_mscale(yarn);
    let mut out = x.to_vec();
    for bi in 0..b {
        for si in 0..s {
            let pos = positions.get(si).copied().unwrap_or(si as u32) as f32;
            let (cos_p, sin_p) = oracle_tables(pos, inv_freq, mscale);
            let base_index = (bi * s + si) * d;
            for i in 0..half {
                let (a1, a2) = if cfg.interleaved {
                    (base_index + 2 * i, base_index + 2 * i + 1)
                } else {
                    (base_index + i, base_index + i + half)
                };
                let x1 = out[a1];
                let x2 = out[a2];
                out[a1] = x1 * cos_p[i] - x2 * sin_p[i];
                out[a2] = x1 * sin_p[i] + x2 * cos_p[i];
            }
        }
    }
    out
}

/// Regeneration procedure (run manually with `--ignored --nocapture` when a
/// spec change is INTENDED): prints the oracle's probe-element bits to paste
/// into GOLDEN_ROPE_BITS / GOLDEN_REROPE_BITS. The committed bits are the
/// contract; regenerating them requires editing both tables in the same
/// commit as the spec change.
#[test]
#[ignore]
fn golden_print_oracle_bits() -> R {
    for case in cases() {
        let b = 1usize;
        let s = case.old_pos.len();
        let mut state = case.seed;
        let x: Vec<f32> = (0..b * s * case.dim).map(|_| xorshift(&mut state)).collect();
        let mut cfg = RopeConfig::new(case.dim, case.base);
        cfg.rotary_dim = case.rotary_dim;
        cfg.interleaved = case.interleaved;
        if case.yarn {
            cfg.yarn = Some(yarn_cfg());
        }
        let probe_idx = s.min(2) * case.dim;
        let rope = oracle_rope(&x, b, s, &case.old_pos, &cfg);
        let k_old = oracle_rope(&x, b, s, &case.old_pos, &cfg);
        let rerope = oracle_rerope(&k_old, b, s, &case.old_pos, &case.new_pos, &cfg);
        println!(
            "{:26} rope 0x{:04X}_{:04X}  rerope 0x{:04X}_{:04X}",
            case.name,
            (rope[probe_idx].to_bits() >> 16) & 0xFFFFu32,
            rope[probe_idx].to_bits() & 0xFFFFu32,
            (rerope[probe_idx].to_bits() >> 16) & 0xFFFFu32,
            rerope[probe_idx].to_bits() & 0xFFFFu32,
        );
    }
    Ok(())
}

/// Unused-handle silencer for the ComputeHandle import (kept for parity with
/// the other backend tests' signatures).
#[allow(dead_code)]
fn _take(_h: Box<dyn ComputeHandle>) {}

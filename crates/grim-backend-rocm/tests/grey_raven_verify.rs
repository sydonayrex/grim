//! E6 verification: the complete GreyRaven layout hypothesis, tested by
//! prediction rather than by reading tea leaves.
//!
//! Hypothesized mappings (each settled by the pairing/field sweeps):
//!
//! - C[lane l][slot s] = Y[8*(l>=16) + 2*s (+0/1)][col l%16]. Lane = column,
//!   slot = row-pair, half = row octave. (B one-hot lights full columns;
//!   A row-0 bytes light lanes 0-15 slot 0 = rows {0,1} all cols.)
//! - A-lane l = weight row (l%16); half h=(l>=16) holds K-groups {4h..4h+3};
//!   positions {2t,2t+1} hold group (4h+t)'s survivors, ranks (0,1) ascending.
//! - sidx pair p (fields 2p,2p+1) = (slot rank0, slot rank1) for K-groups
//!   {p, p+4} JOINTLY (fields 8-15 inert => forced pattern sharing between
//!   K-halves; diagonal (x,x) = reserved, broadcasts on byte 0).
//! - B-chunk m (16 bytes) = column (m%16), k-half (m>=16); byte j = k-slot j.
//!   At sidx (0,0) only slot-0 bytes live (broadcast); splits reroute ranks.
//!
//! Method: craft W (16x32) with KNOWN per-group patterns satisfying
//! pattern[g]==pattern[g+4] (the forced sharing), X (16x16) random, fill
//! fragments EXACTLY per the hypothesis, predict C with dense f32 math on
//! the pruned model, run the raw probe, compare cosine. Cosine ~= 1 closes
//! E6 and greenlights the production kernel; anything else localizes by
//! construction (wrong column => B-chunk rule; wrong rows => A-lane rule;
//! wrong values => pattern/sidx rule).
//!
//! TEMPORARY if it fails (a diagnostic), TO-KEEP if it passes (it becomes
//! the layout golden test pinning the hardware contract).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn up(dev: &RocmDevice, bytes: &[u8]) -> TestResult<Box<dyn grim_tensor::BackendStorage>> {
    MemoryOps::from_cpu_bytes(
        dev,
        bytes,
        &Shape::new(vec![bytes.len()]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("h2d: {e}").into())
}

fn run_mma(
    dev: &RocmDevice,
    a_bytes: &[u8],
    b_bytes: &[u8],
    sidx: u32,
) -> TestResult<Vec<f32>> {
    let a_t = up(dev, a_bytes)?;
    let b_t = up(dev, b_bytes)?;
    let c_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![32 * 8]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("c: {e}"))?;
    fn r(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    dev.launch_grey_raven_probe(r(&a_t), r(&b_t), sidx, r(&c_t))
        .map_err(|e| format!("probe: {e}"))?;
    dev.synchronize();
    Ok(r(&c_t)
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn e4(code: f32) -> u8 {
    grim_quant::f32_to_fp8_e4m3(code)
}
fn de(code: u8) -> f32 {
    grim_quant::fp8_e4m3_to_f32(code)
}

#[test]
fn grey_raven_layout_verification() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    // Craft W: 16 rows x 32 cols. Per K-group g (0..7), pattern {g%3, ...}:
    // use patterns {0,1},{0,2},{1,2} cycling, MIRRORED so pattern[g] ==
    // pattern[g+4] (the forced sharing): patterns[4..8] = patterns[0..4].
    // Values: small exact E4M3 integers, sign alternating by row, magnitude
    // by (r, k) so every survivor is identifiable.
    const PATS: [[usize; 2]; 4] = [[0, 1], [0, 2], [1, 2], [0, 3]];
    let mut w = vec![0.0f32; 16 * 32];
    // mask[r][k] = true iff survivor
    let mut mask = vec![false; 16 * 32];
    for r in 0..16 {
        for g in 0..8 {
            let pat = PATS[g % 4];
            for &slot in &pat {
                let k = 4 * g + slot;
                let v = (((r * 7 + k * 13) % 5) + 1) as f32 * if (r + k) % 2 == 0 { 1.0 } else { -1.0 };
                w[r * 32 + k] = v;
                mask[r * 32 + k] = true;
            }
        }
    }

    // X: 16x16... X^T window is 32x16, i.e. X is 16 rows x 32 cols here
    // (single K-window tile, Kbase=0).
    let mut x = vec![0.0f32; 16 * 32];
    for m in 0..16 {
        for k in 0..32 {
            x[m * 32 + k] = (((m * 3 + k * 5) % 7) + 1) as f32 * 0.25
                * if (m * k) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }

    // Fill A-frag (256 B) per hypothesis: lane l = row l%16, half h = l>=16;
    // positions {2t,2t+1} = group G=4h+t survivors, ranks ascending.
    // A counts survivors per (row, group): exactly the packer order.
    let mut a_frag = vec![0u8; 256];
    for l in 0..32 {
        let r = l % 16;
        let h = usize::from(l >= 16);
        for t in 0..4 {
            let g = 4 * h + t;
            // group g survivors in ascending slot order (packer convention)
            let mut surv: Vec<(usize, f32)> = Vec::new();
            for s in 0..4 {
                let k = 4 * g + s;
                if mask[r * 32 + k] {
                    surv.push((s, w[r * 32 + k]));
                }
            }
            surv.sort_by_key(|(s, _)| *s);
            assert_eq!(surv.len(), 2, "fixture must be exactly 2:4");
            a_frag[l * 8 + 2 * t] = e4(surv[0].1);
            a_frag[l * 8 + 2 * t + 1] = e4(surv[1].1);
        }
    }

    // sidx: pair p = pattern of groups {p, p+4} = (slot rank0, slot rank1).
    // Fixture mirrors patterns so pair p reads PATS[p]. Field pair p is
    // fields {2p, 2p+1} = bits [4p+3:4p]; value = fa | (fb<<2).
    let mut sidx: u32 = 0;
    for p in 0..4 {
        let (s0, s1) = (PATS[p][0] as u32, PATS[p][1] as u32);
        sidx |= (s0 | (s1 << 2)) << (4 * p);
    }

    // Fill B-frag (512 B) per hypothesis: chunk m = column m%16, k-half m>=16;
    // byte j = E4M3(X[m%16][16*(m>=16)+j]).
    let mut b_frag = vec![0u8; 512];
    for m in 0..32 {
        let col = m % 16;
        let kh = if m >= 16 { 16 } else { 0 };
        for j in 0..16 {
            b_frag[m * 16 + j] = e4(x[col * 32 + kh + j]);
        }
    }

    let c = run_mma(&dev, &a_frag, &b_frag, sidx)?;
    // Reference: dense f32 on the pruned model. C[n][m] = sum_k W[n][k] X[m][k]
    // with A-tile = W[16N x 32K] sparse, B-tile = X^T[32K x 16M].
    // Cand-X C layout: flat idx = lane*8+slot -> Y[8*(lane>=16)+slot][lane%16]
    // (lane = column, slot = row-within-octave; NO row pairs -- the tree's
    // "{2s,2s+1}" note is superseded if this passes).
    let mut dot = 0.0f64;
    let (mut gg, mut xx) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let mut worst_i = 0usize;
    for lane in 0..32 {
        for s in 0..8 {
            let i = lane * 8 + s;
            let (n, m) = (8 * usize::from(lane >= 16) + s, lane % 16);
            let mut acc = 0.0f32;
            for k in 0..32 {
                let wv = if mask[n * 32 + k] {
                    de(e4(w[n * 32 + k]))
                } else {
                    0.0
                };
                acc += wv * de(e4(x[m * 32 + k]));
            }
            let (g, x) = (c[i], acc);
            dot += g as f64 * x as f64;
            gg += (g as f64) * (g as f64);
            xx += (x as f64) * (x as f64);
            let ae = (g - x).abs();
            if ae > max_abs {
                max_abs = ae;
                worst_i = i;
            }
        }
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("GREY_VERIFY cosine={cosine:.8} max_abs={max_abs:e} worst_idx={worst_i}");
    assert!(
        cosine >= 0.999,
        "layout hypothesis refuted: cosine {cosine:.6} -- worst flat idx {worst_i}"
    );
    assert!(max_abs <= 1e-3, "max abs err {max_abs:e}");
    Ok(())
}

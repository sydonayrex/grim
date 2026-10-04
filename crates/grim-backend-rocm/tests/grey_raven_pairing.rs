//! E6 joint pairing probe: value-multiplexed A x B one-hot sweep.
//!
//! The A->k mapping and B's layout are COUPLED: pinning either requires the
//! other known, so varying one operand to learn about the other is circular.
//! This probe breaks the circle with VALUES instead of positions:
//!
//!   A[g] = 2^(g mod 8) for ALL g at once (8 distinct powers of two cycling
//!          every 8 offsets; every FP8-exact, so no encoding error),
//!  B[p] = 1.0 at exactly one offset p, 0 elsewhere (512 runs),
//!
//! then C[r][n] = sum of 2^(o mod 8) over the A offsets o that pair with p.
//! Subset sums of powers of two are UNIQUE (binary), so each nonzero C
//! element decodes to the exact residue set of A offsets contracting with p.
//! Combined with the settled row mapping (A lane -> C row = lane mod 16) and
//! C fragment layout, each B byte maps to (rows contracted, C columns) and
//! each A offset maps to a B-byte set -- the complete pairing graph, with no
//! layout hypothesis assumed on either side.
//!
//! Repeated for sidx in {0,1,2,3} (low 2 bits = the only field with a
//! measured effect: output routing to lane class mod 4). If the pairing
//! graph is sidx-invariant, sidx is pure output routing and the 2:4 pattern
//! is positional (fixed by the byte arrangement); if it changes, sidx
//! selects among input patterns and the change IS the encoding.
//!
//! TEMPORARY: this is a measurement instrument, not a gate. It prints the
//! pairing graph for offline analysis and asserts only structural invariants
//! (every B byte pairs with something, values stay within the power-of-two
//! algebra). Delete or convert to a golden test once E6 closes.

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

/// FP8 E4M3 codes for 2^k, k = 0..7. All small integers are exact in E4M3.
fn pow2_codes() -> [u8; 8] {
    let mut out = [0u8; 8];
    for k in 0..8 {
        out[k] = grim_quant::f32_to_fp8_e4m3((1u32 << k) as f32);
    }
    out
}

#[test]
fn grey_raven_joint_pairing_sweep() -> TestResult {    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let p2 = pow2_codes();
    // Sanity: the codes must decode back to exact powers of two, or the
    // whole value algebra collapses.
    for k in 0..8 {
        let v = grim_quant::fp8_e4m3_to_f32(p2[k]);
        assert_eq!(
            v.to_bits(),
            ((1u32 << k) as f32).to_bits(),
            "2^{k} must be FP8-exact"
        );
    }

    // A pattern: byte g holds 2^(g mod 8). 256 distinct-positioned values
    // from an 8-symbol alphabet; row disambiguation comes from the C
    // position (settled row mapping), not the value.
    let mut a = vec![0u8; 256];
    for g in 0..256 {
        a[g] = p2[g % 8];
    }
    let b_one = grim_quant::f32_to_fp8_e4m3(1.0);
    let b_zero = grim_quant::f32_to_fp8_e4m3(0.0);

    for sidx in [0u32, 1, 2, 3] {
        println!("=== sidx={sidx} ===");
        // Per B offset: how many C elements light, and the residue multiset.
        // Compact print: one line per live B offset.
        let mut live_b = 0usize;
        let mut dead_b = 0usize;
        for p in 0..512usize {
            let mut b = vec![b_zero; 512];
            b[p] = b_one;
            let c = run_mma(&dev, &a, &b, sidx)?;
            let mut nz: Vec<(usize, u32)> = Vec::new();
            for (i, &v) in c.iter().enumerate() {
                if v != 0.0 {
                    // Values must be exact integers (sums of powers of two).
                    // A non-integer here means the value algebra broke
                    // (overflow past 2^24, or a non-power crept in).
                    assert_eq!(
                        v,
                        v.round(),
                        "sidx={sidx} p={p} c[{i}]={v}: C must be an exact integer sum of powers of two"
                    );
                    nz.push((i, v as u32));
                }
            }
            if nz.is_empty() {
                dead_b += 1;
            } else {
                live_b += 1;
                // Decode each sum to residue bits for readability.
                let decoded: Vec<(usize, String)> = nz
                    .iter()
                    .map(|(i, v)| {
                        let mut bits = String::new();
                        for k in 0..8 {
                            if v & (1 << k) != 0 {
                                bits.push((b'0' + k as u8) as char);
                            }
                        }
                        (*i, bits)
                    })
                    .collect();
                println!("  p={p:>3} nC={} {decoded:?}", nz.len());
            }
        }
        println!("sidx={sidx}: live B bytes {live_b}/512, dead {dead_b}/512");
    }
    Ok(())
}

/// E6 field-structure sweep: which 2-bit sidx field controls which pair.
///
/// The pairing sweep varied only the low 2 bits and split exactly one residue
/// pair ({01}). The u32 holds 16 two-bit fields; this sweep sets each field
/// to 1 date separately (sidx = 1<<(2f) for f in 0..16) plus the all-split
/// case (sidx = 0x55555555, every field = 01).
///
/// Compact output per sidx: live-B count, and the p%16 -> residues map (which
/// pairs split). If field f controls pair f, sidx=1<<(2f) splits exactly the
/// f-th pair and the field layout is sequential; anything else (shared
/// fields, non-local effects, no-ops) shows here before any kernel is drawn.
#[test]
fn grey_raven_sidx_field_sweep() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let p2 = pow2_codes();
    let mut a = vec![0u8; 256];
    for g in 0..256 {
        a[g] = p2[g % 8];
    }
    let b_one = grim_quant::f32_to_fp8_e4m3(1.0);
    let b_zero = grim_quant::f32_to_fp8_e4m3(0.0);

    let mut sidxs: Vec<u32> = (0..16).map(|f| 1u32 << (2 * f)).collect();
    sidxs.push(0x55555555);
    sidxs.push(0xFFFFFFFF);
    // Interaction probes: do two nonzero fields compose (additive live sets)
    // or collapse (baseline)? 0x5 = fields 0,1; 0x11 = fields 0,2;
    // 0x101 = fields 0,4; 0x55 = fields 0-3 low bits.
    sidxs.extend([0x5, 0x11, 0x101, 0x55]);
    // Full 4-bit characterization of the field pair (f0, f1): sidx 0-15.
    // Hypothesis under test: (f0, f1) = (B-byte index for survivor rank 0,
    // B-byte index for survivor rank 1), with diagonals falling back to
    // byte-0 broadcast. Predicts (2,2)->collapse [sidx=0xA] and (3,3)->
    // collapse [sidx=0xF], and straight/crossed splits elsewhere.
    sidxs.extend([0xAu32, 0xFu32]);
    for sidx in sidxs {
        // p%16 -> set of residue strings seen (summarizes all 512 B bytes in
        // 16 entries: bytes sharing p%16 share C-block structure).
        let mut map: std::collections::BTreeMap<usize, std::collections::BTreeSet<String>> =
            std::collections::BTreeMap::new();
        let mut live = 0usize;
        for p in 0..512usize {
            let mut b = vec![b_zero; 512];
            b[p] = b_one;
            let c = run_mma(&dev, &a, &b, sidx)?;
            let mut residues = std::collections::BTreeSet::new();
            let mut any = false;
            for &v in &c {
                if v != 0.0 {
                    any = true;
                    assert_eq!(v, v.round(), "sidx={sidx:#x} p={p}: non-integer C {v}");
                    let mut bits = String::new();
                    let mut iv = v as u32;
                    let mut k = 0;
                    while iv > 0 {
                        if iv & 1 != 0 {
                            bits.push((b'0' + k) as char);
                        }
                        iv >>= 1;
                        k += 1;
                    }
                    residues.insert(bits);
                }
            }
            if any {
                live += 1;
            }
            map.entry(p % 16).or_default().extend(residues);
        }
        println!("sidx={sidx:#010x} live={live}/512");
        for (r, set) in &map {
            let v: Vec<&String> = set.iter().collect();
            println!("  p%16={r:<2} residues={v:?}");
        }
    }
    Ok(())
}

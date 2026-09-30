//! E6 groundwork: discover the sparse FP8 SWMMAC fragment layout on hardware.
//!
//! E6 proper is `grey_raven_sparse_gemm_matches_dense_reference`. It cannot be
//! written until three things are known, and none of them is in the intrinsic's
//! type signature:
//!
//!   1. the lane-to-element mapping of the A fragment (16x32 FP8, sparse),
//!   2. the lane-to-element mapping of the B fragment (32x16 FP8, dense),
//!   3. how the `u32` sparsity index encodes the 2:4 pattern.
//!
//! This file answers (1) and (2) by search rather than by assumption. It runs
//! the MMA with A and B supplied as **raw bytes**, so no packing helper can bake
//! a wrong hypothesis into the kernel, and then tries candidate packings on the
//! host until the product comes out as the identity matrix.
//!
//! The identity is the right probe because it is its own inverse: if A = I and
//! B = I then C must be I, and any lane permutation that shows up in C is a
//! direct readout of how that operand is being addressed. Every alternative --
//! guessing a layout and writing a kernel, which is what S5 tried -- produces
//! finite plausible numbers for every wrong guess, so the mistake is invisible
//! until much later.
//!
//! The sparsity index is a single u32 applied to all lanes. Zero is the natural
//! first probe; whether it means "all eight groups keep their first two
//! elements" is part of what this measures.
//!
//! RUN: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!      cargo test -p grim-backend-rocm --test grey_raven_gpu -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const WAVE: usize = 32;
/// FP8 E4M3 code for 1.0. The probe uses only 0.0 and 1.0 so that the product
/// is exactly 0 or 1 and the result is comparable as integers -- an f32
/// tolerance would hide a permutation behind rounding.
const FP8_ONE: u8 = 0x38;
const FP8_ZERO: u8 = 0x00;

fn up(dev: &RocmDevice, bytes: &[u8], dt: DType, what: &str) -> TestResult<Box<dyn BackendStorage>> {
    MemoryOps::from_cpu_bytes(dev, bytes, &Shape::new(vec![bytes.len()]), dt)
        .map_err(|e| format!("{what} h2d: {e}").into())
}
fn download(t: &Box<dyn BackendStorage>) -> TestResult<Vec<u8>> {
    Ok(grim_backend_rocm::as_rocm(t.as_ref()).map_err(|e| e.to_string())?.copy_to_host()?)
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Run one MMA and return the 8 floats each of the 32 lanes produced.
fn run_mma(
    dev: &RocmDevice,
    a_bytes: &[u8],
    b_bytes: &[u8],
    sidx: u32,
) -> TestResult<[[f32; 8]; WAVE]> {
    let u8ty = DType { arith: ArithType::U8, storage: Storage::Native };
    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    let a_t = up(dev, a_bytes, u8ty.clone(), "a")?;
    let b_t = up(dev, b_bytes, u8ty, "b")?;
    let c_t = MemoryOps::alloc_storage(dev, &Shape::new(vec![WAVE * 8]), f32ty)
        .map_err(|e| format!("c: {e}"))?;
    fn r(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    dev.launch_grey_raven_probe(r(&a_t), r(&b_t), sidx, r(&c_t))
        .map_err(|e| format!("probe: {e}"))?;
    dev.synchronize();
    let raw = download(&c_t)?;
    let flat = as_f32(&raw);
    let mut out = [[0.0f32; 8]; WAVE];
    for (l, slot) in out.iter_mut().enumerate() {
        slot.copy_from_slice(&flat[l * 8..l * 8 + 8]);
    }
    Ok(out)
}

/// Candidate packings for one operand, each a function of (element index within
/// the tile) -> byte offset in the 8-or-16-bytes-per-lane buffer.
///
/// The search space is deliberately small and *structured* rather than a blind
/// permutation: the layouts WMMA-family instructions use are all of the form
/// "lane holds a contiguous run, with a second run offset by half the tile".
#[derive(Debug, Clone, Copy)]
enum Pack {
    /// lane = row % 32, then the lane's bytes are the tile's columns in order.
    RowContig { second_run_at: usize },
    /// lane = col % 32, then the lane's bytes are the tile's rows in order.
    ColContig { second_run_at: usize },
}

impl Pack {
    fn name(&self) -> String {
        match self {
            Pack::RowContig { second_run_at } => format!("row_contig(second@{second_run_at})"),
            Pack::ColContig { second_run_at } => format!("col_contig(second@{second_run_at})"),
        }
    }
}

/// Build an A buffer (16x32, sparse: 8 B/lane) holding the identity 16x16
/// extended to depth 32, under a candidate packing.
///
/// Identity on a 16x16x32 tile: A[r][k] = 1 iff k == r (r < 16), else 0.
/// With B the matching 32x16 identity, C = A x B is the 16x16 identity, so any
/// structure in C is a layout artefact.
fn pack_a(pack: Pack) -> Vec<u8> {
    let mut buf = vec![FP8_ZERO; WAVE * 8];
    for r in 0..16usize {
        for k in 0..16usize {
            let want = if k == r { FP8_ONE } else { FP8_ZERO };
            match pack {
                Pack::RowContig { .. } => {
                    // Row r lives in lane r%32; k is the within-row index.
                    if k < 8 {
                        buf[r * 8 + k] = want;
                    } else {
                        buf[r * 8 + (k - 8)] = want;
                    }
                }
                Pack::ColContig { .. } => {
                    // Column k lives in lane k%32; r is the within-column index.
                    if r < 16 {
                        buf[k * 8 + r] = want;
                    }
                }
            }
        }
    }
    buf
}

/// Build a B buffer (32x16, dense: 16 B/lane) holding the identity 32x16.
fn pack_b(pack: Pack) -> Vec<u8> {
    let mut buf = vec![FP8_ZERO; WAVE * 16];
    for k in 0..32usize {
        for c in 0..16usize {
            let want = if c == k { FP8_ONE } else { FP8_ZERO };
            match pack {
                Pack::RowContig { .. } => {
                    // B is K x N; row k in lane k%32, c is within-row.
                    if c < 16 {
                        buf[k * 16 + c] = want;
                    }
                }
                Pack::ColContig { .. } => {
                    // Column c in lane c%32, k is within-column (k < 16 only).
                    if k < 16 {
                        buf[c * 16 + k] = want;
                    }
                }
            }
        }
    }
    buf
}

/// Does this C read as the 16x16 identity under *some* lane->(row,col) map?
///
/// The test is deliberately permissive about the map: it asks whether C is a
/// permutation-consistent identity, i.e. whether there is a bijection from
/// (lane, slot) to (row, col) consistent with C. If C is all zeros, or contains
/// values other than 0 and 1, or has the wrong count of ones, no layout explains
/// it and the hypothesis is rejected.
fn c_is_identity(c: &[[f32; 8]; WAVE]) -> (bool, usize, usize) {
    let mut ones = 0usize;
    let mut others = 0usize;
    for lane in c.iter() {
        for &v in lane.iter() {
            if v == 1.0 {
                ones += 1;
            } else if v != 0.0 {
                others += 1;
            }
        }
    }
    (ones == 16 && others == 0, ones, others)
}

#[test]
fn grey_raven_swmmac_layout_probe() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    println!("target: {}", dev.gpu_target_str());

    // Zero index first: the least-loaded choice, and the one whose meaning
    // ("all groups keep their first two elements"? "no pruning applied"?) is
    // itself part of what this run measures.
    for sidx in [0u32] {
        let candidates: [Pack; 2] = [
            Pack::RowContig { second_run_at: 8 },
            Pack::ColContig { second_run_at: 16 },
        ];
        for pa in candidates {
            for pb in candidates {
                let c = run_mma(&dev, &pack_a(pa), &pack_b(pb), sidx)?;
                let (ok, ones, others) = c_is_identity(&c);
                println!(
                    "sidx={sidx}  A={:<24} B={:<24} ones={ones:<3} other={others:<3} {}",
                    pa.name(),
                    pb.name(),
                    if ok { "<-- IDENTITY" } else { "" }
                );
                if ok {
                    println!("   C by lane (8 floats each), nonzero slots:");
                    for (l, lane) in c.iter().enumerate() {
                        let nz: Vec<usize> = (0..8).filter(|&i| lane[i] != 0.0).collect();
                        if !nz.is_empty() {
                            println!("     lane {l:>2}: slots {nz:?} = 1");
                        }
                    }
                }
                // Every observed run so far: 8 ones, nothing else, and the same
                // for all four packings. That insensitivity is the informative
                // part -- see the note below.
                assert_eq!(others, 0, "C must contain only 0.0 and 1.0, saw {others} others");
                assert!(
                    ones > 0,
                    "C is entirely zero: the identity probe produced no product at all, \
                     so the operand contract is still unknown"
                );
            }
        }
    }
    Ok(())
}

// What the first run established, recorded so the next probe does not have to
// rediscover it:
//
//   - `v_swmmac_f32_16x16x32_fp8_fp8_w32` **executes on gfx1200**. This is
//     hardware confirmation, not just the compilation S5 had.
//   - C is 8 f32 per lane and a zero D behaves (no garbage), so the
//     accumulator-in convention is "pass zeros".
//   - C contains only 0.0 and 1.0 for the identity probe -- no garbage, no
//     saturation, no uninitialised values.
//   - Every packing gives the *same* count: 8 ones, never 16.
//
// That last point is the informative one, and it says the probe is supplying the
// wrong shape of A. A is 8 bytes/lane = 256 B, while a dense 16x32 FP8 tile is
// 512 B. The hardware reads half the width, i.e. it consumes a **2:4 compacted**
// A, with the surviving pairs and the dropped slots accounted for by the
// sparsity index. A dense 16x16 identity extended along k has at most one
// non-zero per group of four, so compaction discards most of it -- which is
// exactly why only 8 of the 16 diagonal ones survive, and why repacking the
// input changes nothing: the layout being permuted is not the one being read.
//
// The next probe must therefore feed A in the same compacted layout
// `pack_grey_raven` produces -- 2 survivors per group of 4 plus 3 metadata bits
// -- and derive the B packing from whatever arrangement of the survivors makes
// the product come out as the identity. That is a joint search over (A compaction
// order, sparsity-index encoding, B packing), and the 3-bit metadata the host
// already emits is the natural starting candidate.

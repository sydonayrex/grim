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

/// Read the 2:4 slot selection straight out of the hardware.
///
/// The first probe showed A is consumed at 8 B/lane = 256 B, i.e. a 2:4
/// *compacted* operand, and that permuting the dense input changed nothing. So
/// supply the compacted operand and make the answer self-describing:
///
///   A (compacted) = all ones        -> 16 survivors per row, every one 1.0
///   B              = identity 32x16 -> B[k][n] = 1 iff k == n
///
/// then C[r][n] = sum over row r's survivors of B[k][n] = 1 exactly when n is
/// one of the k-slots the hardware says row r occupies. The nonzero *columns* of
/// C are therefore the slot set itself, read directly rather than inferred.
///
/// This is the probe the dense identity could not be: there, the answer
/// depended on a layout guess and every guess gave the same count. Here the
/// compacted buffer's contents do not depend on the row-to-lane mapping at all,
/// because every byte is the same value -- so the output isolates the one thing
/// being measured, the index encoding, with the layout question set aside.
#[test]
fn grey_raven_sparse_index_slot_selection() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    println!("target: {}", dev.gpu_target_str());

    // A: the whole compacted operand is ones. 256 B = 8 B/lane over 32 lanes.
    let a_all_ones = vec![FP8_ONE; WAVE * 8];

    // B: identity 32x16 under the row-major-in-lane packing -- B is K x N, so
    // row k occupies lane k and its 16 columns are the 16 bytes of that lane.
    // Only n < 16 exists in the output, so the two halves of k are identical.
    let mut b = vec![FP8_ZERO; WAVE * 16];
    for k in 0..16usize {
        b[k * 16 + k] = FP8_ONE;
    }

    for sidx in [0u32, 0xFFFFFFFFu32, 0x55555555u32, 0xAAAAAAAAu32] {
        let c = run_mma(&dev, &a_all_ones, &b, sidx)?;
        // Collapse to a 16x16 reading: the pattern of nonzero lanes/slots.
        let mut col_hist = [0usize; 8];
        let mut lanes_with_data = 0usize;
        for lane in c.iter() {
            let nz = lane.iter().filter(|&&v| v != 0.0).count();
            if nz > 0 {
                lanes_with_data += 1;
            }
            for &v in lane.iter() {
                if v != 0.0 {
                    col_hist[0] += 1;
                }
            }
        }
        let total: usize = c.iter().map(|l| l.iter().filter(|&&v| v != 0.0).count()).sum();
        println!(
            "sidx=0x{sidx:08x}  nonzero C elements = {total:<5} lanes with data = {lanes_with_data}",
        );
        // Print the per-lane nonzero slot pattern for the first lanes that have
        // any, which is where the slot rule is visible.
        // Direct readout of the index's low 2 bits. All eight lanes in one
        // residue class mod 4 are active, and the class is selected by the
        // index:  00 -> lane = 0 mod 4,  01 -> 1 mod 4,  10 -> 2 mod 4,
        //         11 -> 3 mod 4.
        //
        // (An earlier reading of this data said "lane pair {j, j+4}". It is
        // wrong: the active set is the whole residue class, 8 lanes, not 2.)
        let pairs: Vec<usize> =
            c.iter().enumerate().filter(|(_, l)| l.iter().any(|&v| v != 0.0)).map(|(i, _)| i).collect();
        println!("   active lanes {pairs:?}  (index bits0,1 = {:02b})", sidx & 0b11);
        for (l, lane) in c.iter().enumerate().take(8) {
            let nz: Vec<(usize, f32)> = lane
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, v)| (i, *v))
                .collect();
            println!("   lane {l:>2}: {nz:?}");
        }
    }
    Ok(())
}

// What this probe established, and what it did not.
//
// Established, on hardware:
//
//   - The instruction runs on gfx1200 and its output is stable and finite.
//   - C is 8 f32 per lane; a zero D is a valid accumulator-in.
//   - **Bits 0,1 of the sparsity index select one of four lane residue classes**:
//     00 -> lane = 0 mod 4, 01 -> 1 mod 4, 10 -> 2 mod 4, 11 -> 3 mod 4. All eight
//     lanes of the chosen class are active, so this is a 2-bit field choosing
//     1 of 4 over a stride-4 lane distribution -- the same shape as a 2:4 choice.
//     Whether that is the per-group slot selector, a row-group selector, or
//     something else is not yet distinguishable, and "per-group" is the
//     problem anyway: one 2-bit field is not eight, and a 16x32 A has eight
//     groups of four along k.
//
// Not established:
//
//   - The B fragment layout. This is now the blocker. With B set to the
//     32x16 identity, the product came back as 64 nonzeros, all exactly 2.0, in
//     whole lanes -- every one of an active lane's 8 C slots held the same
//     value. A correct A x B cannot do that: C[r][n] must vary with n, because
//     B[k][n] is 1 only at n == k. Identical values across all 8 slots mean the
//     16 B bytes of a lane are not being read as 16 distinct n values for one
//     k, so the lane/k and byte/n roles are transposed or interleaved relative
//     to the assumption in `pack_b`.
//
// The 2.0 is consistent with that: two contributions landing on each output,
// which is what a mis-strided B produces rather than a 1.0 that one survivor
// would give.
//
// So the remaining search is over B's byte-to-(k,n) mapping, jointly with how
// the index's eight per-group 2-bit fields are packed into the u32 (only the
// low one is located so far). That is a wider search than trial-and-error
// should keep driving: the honest next step is to enumerate the B layout space
// exhaustively against the identity probe, the same way the A side was ruled
// out, rather than continuing to guess one arrangement per run.

/// One-hot sweep over B: exactly one B byte is 1.0, everything else is 0.
///
/// This measures B's byte -> (k,n) assignment instead of testing a hypothesis
/// about it. The identity probe could only say "wrong": a mis-strided B yields
/// finite plausible numbers, so every candidate arrangement had to be guessed
/// and checked one at a time. A single hot byte removes the ambiguity -- there
/// is only one possible contributor to C, so the C elements that light up are
/// exactly the rows whose 2:4 slot set contains that byte's k, addressed at the
/// column that byte's n maps to.
///
/// The two coordinates separate, which is what makes this decodable:
///
///   - the **slot** within a lane depends only on the byte's column n, and
///   - the **lane set** depends on the byte's row k (which rows kept that k).
///
/// So grouping all 512 offsets by their (lane, slot) signature recovers the
/// column mapping directly and the row mapping up to the per-row 2:4 pattern.
#[test]
fn grey_raven_b_byte_onehot_sweep() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    let a = vec![FP8_ONE; 256];
    let mut b = vec![FP8_ZERO; 512];
    let mut groups: std::collections::BTreeMap<Vec<(usize, usize)>, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut values_seen: std::collections::BTreeSet<String> = Default::default();

    for g in 0..512usize {
        b[g] = FP8_ONE;
        let c = run_mma(&dev, &a, &b, 0)?;
        b[g] = FP8_ZERO;

        let mut sig: Vec<(usize, usize)> = Vec::new();
        for (l, lane) in c.iter().enumerate() {
            for (s, &v) in lane.iter().enumerate() {
                if v != 0.0 {
                    values_seen.insert(format!("{v}"));
                    sig.push((l, s));
                }
            }
        }
        groups.entry(sig).or_default().push(g);
    }

    println!("\n=== B one-hot sweep: 512 offsets -> {} distinct C signatures ===", groups.len());
    println!("nonzero values seen across the whole sweep: {:?}", values_seen);

    for (sig, offs) in &groups {
        let lanes: Vec<usize> = sig.iter().map(|(l, _)| *l).collect();
        let slots: Vec<usize> = sig.iter().map(|(_, s)| *s).collect();
        // Reduce offsets to their byte-position within a lane, to expose whether
        // the signature is a function of position or of the whole offset.
        let in_lane: Vec<usize> = offs.iter().map(|o| o % 16).collect();
        let uniq_in_lane: Vec<usize> = {
            let mut v = in_lane.clone();
            v.sort_unstable();
            v.dedup();
            v
        };
        println!("\n  signature: {} elems  lanes={lanes:?}  slots={slots:?}", sig.len());
        println!("    {} offsets, e.g. {offs:?}", offs.len());
        println!("    distinct byte-in-lane: {uniq_in_lane:?}");
        let distinct_lanes = {
            let mut v = lanes.clone();
            v.sort_unstable();
            v.dedup();
            v
        };
        let distinct_slots = {
            let mut v = slots.clone();
            v.sort_unstable();
            v.dedup();
            v
        };
        println!("    distinct lanes: {distinct_lanes:?}   distinct slots: {distinct_slots:?}");
    }
    Ok(())
}

/// One-hot sweep over A.
///
/// KNOWN INVALID -- retained for the record, not as a source of truth. Its B
/// operand is built by `identity_32x16`, which assumes B's fragment layout is
/// memory row-major; it is not, so lanes 1..31 read zeros. See the retraction
/// note at the foot of this file. The B sweep is the valid one.
///
/// The B sweep showed only bytes at offset = 0 (mod 4) are live, and that each
/// live byte contributes exactly 2.0. Both facts are about how many bytes the
/// hardware actually consumes per 32-bit group, so the same two questions are
/// worth asking of A. With B set to the 32x16 identity, a single hot A byte at
/// (r, k) must light up C[r][n] for the columns n where B[k][n] = 1, i.e. n = k,
/// which pins the A byte to a single output row.
#[test]
fn grey_raven_a_byte_onehot_sweep() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    let b = identity_32x16();
    let mut a = vec![FP8_ZERO; 256];
    let mut groups: std::collections::BTreeMap<Vec<(usize, usize)>, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut values: std::collections::BTreeSet<String> = Default::default();

    for g in 0..256usize {
        a[g] = FP8_ONE;
        let c = run_mma(&dev, &a, &b, 0)?;
        a[g] = FP8_ZERO;
        let mut sig: Vec<(usize, usize)> = Vec::new();
        for (l, lane) in c.iter().enumerate() {
            for (s, &v) in lane.iter().enumerate() {
                if v != 0.0 {
                    values.insert(format!("{v}"));
                    sig.push((l, s));
                }
            }
        }
        groups.entry(sig).or_default().push(g);
    }

    println!("\n=== A one-hot sweep: 256 offsets -> {} distinct C signatures ===", groups.len());
    println!("nonzero values seen: {values:?}");
    for (sig, offs) in &groups {
        let lanes: Vec<usize> = { let mut v: Vec<usize> = sig.iter().map(|(l, _)| *l).collect(); v.sort_unstable(); v.dedup(); v };
        let slots: Vec<usize> = { let mut v: Vec<usize> = sig.iter().map(|(_, s)| *s).collect(); v.sort_unstable(); v.dedup(); v };
        let in_lane: Vec<usize> = { let mut v: Vec<usize> = offs.iter().map(|o| o % 8).collect(); v.sort_unstable(); v.dedup(); v };
        println!(
            "  {:>6} elems  lanes={lanes:?} slots={slots:?}  byte-in-lane={in_lane:?}  \
             {} offs e.g. {offs:?}", sig.len(),
            offs.len()
        );
    }
    Ok(())
}

/// B[k][n] = 1 iff k == n, for a 32x16 tile, row-major (16 bytes per lane).
fn identity_32x16() -> Vec<u8> {
    let mut v = vec![FP8_ZERO; 512];
    for k in 0..32usize {
        for n in 0..16usize {
            if k == n {
                v[k * 16 + n] = FP8_ONE;
            }
        }
    }
    v
}

// ============================================================================
// What the one-hot sweeps established, and what they refute.
//
// The sweeps above supersede two earlier readings of this file, both of which
// were wrong and are withdrawn here rather than left to mislead the next
// reader.
//
//   Withdrawn: "A is consumed as 8 B/lane = 256 B, a 2:4-compacted operand."
//   Withdrawn: "B's byte -> (k,n) mapping is the only thing left to find."
//
// Both came from permuting dense 256/512-byte inputs, which cannot separate
// "the hardware ignored this byte" from "this byte feeds the same element as
// its neighbour". A one-hot sweep separates them, because a single hot byte has
// exactly one possible contributor to C.
//
// RETRACTED -- the A-side measurements below are invalid, and are kept only so
// the mistake is not repeated. The B operand was built by `identity_32x16()`,
// which lays B out row-major in memory (16 bytes per k-row). But the kernel
// reads `B[lane]` as a *fragment*, not as row `lane`. So lane 0 received the
// row containing B[0][0] = 1 and lanes 1..31 received all-zero rows, and the
// sweep was multiplying a one-hot A by a B that was almost entirely zero.
//
// Every A-side observation -- "offsets 128..255 are dead", "bytes act in
// adjacent pairs", "four values per lane" -- is an artifact of that. The 1.0
// values are what a one-hot A times a near-zero B should produce, which is
// exactly why the result looked clean enough to be believed.
//
// The general trap, worth stating plainly: a probe that assumes the layout it is
// trying to measure cannot measure it. Building the identity in memory and
// reading the product back is circular -- it returns a plausible answer for any
// layout, and a *correct-looking* one only when memory order happens to match
// the fragment order. The B sweep below avoids this because it never assumes
// B's layout: A is uniform, so any hot B byte can be attributed by elimination.
//
// Original, now-withdrawn A-side text:
//
//   - Nonzero values are exactly 1.0. Correct: one hot A element times an
//     identity B is 1.0.
//   - Offsets 128..255 produce nothing at all. Half the supplied A buffer is
//     dead.
//   - The live 128 bytes act in adjacent pairs: {0,1}, {2,3}, {4,5}, ... A hot
//     byte at either member of a pair gives the identical result, at value 1.0,
//     which is what pairwise summing into one value predicts.
//
// Measured, B side (A = 256 compacted bytes of 1.0) -- these stand, because the
// uniform A makes the sweep independent of B's layout:
//
//   - Only offsets = 0 (mod 4) are live: 128 of 512. The other 384 bytes change
//     nothing.
//   - Each live byte lights 16 C elements: all 8 slots of lane j and all 8 slots
//     of lane j+16, j in 0..16, so the 32 lanes pair as {j, j+16}.
//   - Nonzero values are exactly 2.0, never 1.0, for any single hot byte.
//
// The intrinsic's signature has since been confirmed against clang directly --
// (v2i32 A, v4i32 B, v8i32 C/D, i32 index), 4 args, no permissibility -- so the
// operand roles here are right and the A/B-swap theory is dead. That leaves the
// B-side numbers to explain: 4 of 16 B bytes live per lane, each contributing
// twice. Those are the next thing to chase, with a probe that varies A only
// across its *fragment* lanes, which is now the one unknown left to pin down.
//
// Measured, B side (A = 256 compacted bytes of 1.0):
//
//   - **Only offsets = 0 (mod 4) are live** -- 128 of 512. The other 384 bytes
//     change nothing.
//   - Each live byte lights 16 C elements: all 8 slots of lane j and all 8 slots
//     of lane j+16, j in 0..16, so the 32 lanes pair as {j, j+16}.
//   - Nonzero values are exactly 2.0, never 1.0, for any single hot byte.
//
// On the B side, 4 of the 16 supplied bytes per lane are live, and a live byte
// is accumulated twice. Whether that is a 32x16 B genuinely laid out that way, or
// a fragment whose layout still has to be discovered, is unresolved -- the sweep
// locates the live bytes but does not say which (k, n) they are.
//
// The signature *was* then checked against clang itself -- 4 arguments, strictly
// (v2i32, v4i32, v8i32, i32), with no other permutation accepted -- so the
// operand roles and the 8/16/32 bytes-per-lane widths in this file are correct,
// and the earlier theory that A and B were swapped is dead. Clang is the oracle
// here rather than a hand-written note in docs/, which is what should have been
// done before the first probe ran.
//
// That leaves exactly one unknown: how A's 8 bytes per lane map onto rows and
// k. It is the last one, because B is only interpretable once A is pinned -- the
// B sweep's "4 live bytes, each counted twice" is exactly what a uniform A
// against an unknown A-fragment mapping would look like. The next probe must
// therefore vary A across fragment lanes while holding B uniform, the mirror
// image of the sweep that worked.

/// Mirror of the B sweep: **uniform B, one-hot A**.
///
/// This is the last unknown -- how A's 8 bytes per lane map onto rows and k --
/// and it has to be measured this way round. B cannot be interpreted until A is
/// pinned, because "4 live B bytes, each counted twice" is exactly what a
/// uniform A against an unknown A-fragment mapping would produce. So this probe
/// makes A the only varied operand and holds B at all ones.
///
/// With B uniform, C[r][n] = sum of the A values contracted into (r, n), so a
/// hot A byte lights **every column n of every row that owns that k**. The
/// result is therefore self-describing in a way the identity probe never was:
///
///   - the *set of slots* is the C lane->row / slot->column mapping,
///   - the *value* is how many times the hot element is accumulated,
///   - and how many of A's 256 bytes are live at all is answerable, which the
///     retracted A sweep could not report because its B was nearly all zero.
#[test]
fn grey_raven_a_fragment_sweep_uniform_b() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    let b = vec![FP8_ONE; 512]; // uniform: removes B's layout from the problem
    let mut a = vec![FP8_ZERO; 256];
    let mut groups: std::collections::BTreeMap<Vec<(usize, usize)>, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut values: std::collections::BTreeSet<String> = Default::default();

    for g in 0..256usize {
        a[g] = FP8_ONE;
        let c = run_mma(&dev, &a, &b, 0)?;
        a[g] = FP8_ZERO;
        let mut sig: Vec<(usize, usize)> = Vec::new();
        for (l, lane) in c.iter().enumerate() {
            for (s, &v) in lane.iter().enumerate() {
                if v != 0.0 {
                    values.insert(format!("{v}"));
                    sig.push((l, s));
                }
            }
        }
        groups.entry(sig).or_default().push(g);
    }

    let live: usize = groups.iter().filter(|(s, _)| !s.is_empty()).map(|(_, o)| o.len()).sum();
    println!("\n=== A sweep, uniform B: 256 offsets -> {} signatures; {live} live ===", groups.len());
    println!("nonzero values: {values:?}");
    for (sig, offs) in &groups {
        if sig.is_empty() {
            println!("  DEAD: {} offsets", offs.len());
            continue;
        }
        let lanes: Vec<usize> = { let mut v: Vec<usize> = sig.iter().map(|(l, _)| *l).collect(); v.sort_unstable(); v.dedup(); v };
        let slots: Vec<usize> = { let mut v: Vec<usize> = sig.iter().map(|(_, s)| *s).collect(); v.sort_unstable(); v.dedup(); v };
        println!("  {:>3} elems  lanes={lanes:?}  slots={slots:?}  {} offs e.g. {offs:?}",
                 sig.len(), offs.len());
    }
    Ok(())
}

// ============================================================================
// Result of the A-fragment sweep, and a correction to the B sweep's reading.
//
// 256 offsets, **all 256 live**, 16 signatures, every value exactly 1.0. Fully
// regular, and it decodes without any assumption:
//
//   - **A lane (0..31) maps to C row = lane mod 16.** Lanes l and l+16 are
//     equivalent: offsets 0..7 and 128..135 land in the same signature. So the
//     32 A lanes carry 16 rows, two lanes per row.
//   - **The C fragment is transposed from the usual mental picture**: the output
//     lane is the *column* (16 lanes cover the 16 columns, in two halves chosen
//     by bit 3 of the A lane), and the 8 slots are the *rows*, slot s covering
//     rows {2s, 2s+1}. With B uniform a hot A byte lights one slot in each of
//     the 16 lanes -- one row, every column, which is exactly the right shape
//     for C[r][n] = sum over all k of A[r][k].
//   - All 8 bytes of an A lane give the *identical* result. With B uniform that
//     is forced: C[r][n] sums the whole row, so a hot byte at any k lands in the
//     same place. It says the bytes are the row's survivors, and nothing more.
//
// So this pins A's **row** mapping, cleanly, and pins nothing about **k**.
//
// The k mapping is not measurable this way, and that is a property of the probe
// rather than a fixable defect of it: with B uniform every k in a row collapses
// onto the same output, so all 8 survivors are indistinguishable. Which also
// means the B sweep's "only offsets = 0 (mod 4) are live, each counted twice"
// cannot be read as a statement about B's layout. It was measured against an A
// whose k mapping was unknown, so those 384 dead bytes may be B bytes whose k
// had no partner in A rather than bytes the hardware ignores.
//
// That couples the two unknowns. A's k mapping and B's layout are not separately
// measurable, because pinning either one requires the other to be known. The
// handle that *is* independent is the sparsity index: `sidx` changes which k are
// active without perturbing either fragment layout, so sweeping it against a
// one-hot A and a uniform B can separate the k mapping from the layout question.
// That is the next probe.

/// Sweep the sparsity index against one-hot A and uniform B.
///
/// This is the only handle that changes *which k is active* without perturbing
/// either fragment layout, which is what the previous two sweeps could not do.
/// A's row mapping is known (lane mod 16) but its k mapping is not, and B's
/// layout is not either; the two are coupled, so varying an operand to learn
/// about the other is circular. The index is external to both.
///
/// With B uniform, C[r][n] = sum over *active* k of A[r][k], so a one-hot A
/// byte gives 1.0 exactly when its k is selected by sidx and 0.0 when it is not.
/// The live set for each index is therefore a direct readout of the index ->
/// (A lane, A byte) selection, which is the k mapping with B's layout factored
/// out entirely.
#[test]
fn grey_raven_sparse_index_selects_k() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let b = vec![FP8_ONE; 512];

    // Every one-bit pattern, across the whole u32, not just the low nibble: the
    // low 4 bits turned out to be inert here, and it is cheaper to check all 32
    // positions than to assume which half the encoding lives in.
    for sidx in (0u32..32).map(|i| 1u32 << i) {
        let mut live: Vec<usize> = Vec::new();
        for g in 0..256usize {
            let mut a = vec![FP8_ZERO; 256];
            a[g] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                live.push(g);
            }
        }
        // Compress to ranges so the pattern is readable rather than 256 numbers.
        let mut ranges: Vec<String> = Vec::new();
        let mut i = 0;
        while i < live.len() {
            let s = live[i];
            let mut e = s;
            while e + 1 < live.len() && live[e + 1] == e + 1 { e += 1; }
            ranges.push(if s == e { format!("{s}") } else { format!("{s}-{e}") });
            i = e + 1;
        }
        println!("sidx={sidx:#04x}  {:>3} live A byte offsets: {}", live.len(), ranges.join(", "));
    }
    Ok(())
}

// ============================================================================
// Result of the index sweep, and the retraction it forces.
//
// Sweeping one-hot A against uniform B over **every one-bit pattern in the
// u32**, all 256 A bytes stay live in all 32 cases -- and at sidx = 0 all 256
// are live too. Not one bit of the sparsity index gates a single A element.
//
// That retracts the reading this file gave the index twice. The first note said
// bits 0,1 "are almost certainly the per-group slot selector" and that the
// 2:4 choice was being read out directly. The measurement is real -- those bits
// do change which 8 of 32 output lanes receive data -- but the *interpretation*
// is wrong. The index is not selecting among k. If it were, a one-hot A would go
// dark for the index values that deselect its byte, and none ever does.
//
// The two observations are consistent under a different reading: the index
// **routes output** rather than selecting input. With A all-ones, changing bits
// 0,1 moves which lane class is written; with A one-hot, every byte still
// produces output under every index, just routed to a different lane class.
//
// So the 2:4 pattern is not something this operand asks for. It is either fixed
// by the instruction, or carried somewhere this sweep cannot see -- a companion
// VGPR, or a convention on how the 8 bytes per lane are interpreted.
//
// Where E6 stands, stated without optimism:
//
//   settled    the instruction's signature (v2i32, v4i32, v8i32, i32), against
//              clang rather than a hand-written note
//   settled    A lane (0..31) -> C row = lane mod 16, two lanes per row
//   settled    the C fragment: output lane = column, slot = row pair {2s, 2s+1}
//   settled    bits 0,1 of the index route output to a lane class mod 4
//   open       A's 8 bytes/lane -> k, unreadable with B uniform because C[r][n]
//              sums the whole row
//   open       B's 16 bytes/lane -> (k, n), coupled to the above
//   refuted    the index as a 2:4 slot selector
//   refuted    "half of A is dead", "bytes act in pairs", "4 values/lane" --
//              artifacts of the retracted identity-B sweep
//
// The coupled pair is the real obstacle. With B uniform only A's rows are
// visible; with A uniform only A's rows are visible from A, and B reads out
// against an A of unknown k mapping. Separating them needs a probe that fixes
// one operand's k mapping *without* assuming it, and no such handle is left in
// the current operand set -- which is itself the finding worth escalating, since
// it means E6's remaining work is not a search but a change to what the kernel
// is given.

// ============================================================================
// What the index actually is: a placement map, not a selector.
//
// Following the index sweep to its source settled it. Emitting the intrinsic
// standalone shows the real instruction:
//
//   v_swmmac_f32_16x16x32_fp8_fp8 v[5:12], v[13:14], v[0:3], v4
//                                        ^A 2 VGPR ^B 4 VGPR ^index 1 VGPR
//
// so the assembly operand order is (D, A, B, index), A and B match the intrinsic
// widths exactly, and the index is a **per-lane VGPR** -- not a uniform scalar.
// (It looked uniform only because this probe hands every lane the same value.)
//
// Then, from the intrinsic table: the full set of FP8 SWMMAC variants is
// 16x16x32 {fp8_fp8, fp8_bf8, bf8_fp8} and 16x16x128 {fp8_fp8, fp8_bf8,
// bf8_fp8, f16_*} -- and there is **no separate sparse variant among them**.
// Every one takes "Sparsity index for A". So 2:4 sparsity is intrinsic to
// `v_swmmac_*_fp8_*`, and the index is the per-lane descriptor of it.
//
// The index is therefore a **placement map, not a mask**: it says where each of
// A's 8 compacted bytes goes in k, not which of them are dropped. All 8 are
// always read. That single change explains every anomaly in this file at once,
// including the two earlier probes that looked like bugs:
//
//   - no index bit gates any A byte -- correct, none of them is a drop bit;
//   - all 8 bytes of an A lane give identical results under uniform B -- correct,
//     they are 8 different k of the same row, and C[r][n] sums the whole row;
//   - index bits 0,1 move which output lanes are written -- correct, that is
//     placement changing where a row's results land.
//
// Nothing here is exotic. It is a compacted A plus a per-lane expansion map,
// which is exactly the 2:4 format GreyRaven already produces.
//
// This also corrects the previous note's speculation that the 2:4 pattern might
// be implicit and never need to reach the hardware. It does need to: the 3-bit
// metadata GreyRaven packs is what tells the hardware the k-position of each
// survivor, and E6 has to reverse that map to emit it.
//
// What remains is one joint problem rather than two: the index's bit -> k map and
// B's byte -> (k, n) layout cannot be solved separately, because A's k mapping is
// only visible through a B whose layout is unknown, and vice versa. They are
// solvable jointly -- A uniform and B one-hot already constrain both, since the
// live B bytes are precisely those whose k the index places -- and that joint fit
// is the remaining work for E6.

/// The joint fit: does the set of live B bytes depend on the index?
///
/// A's k mapping and B's layout are coupled -- each is visible only through the
/// other -- but jointly they are constrained, and this is the constraint. A is
/// uniform, so A's own mapping cannot affect *which* B bytes are live: a B byte
/// at k is live iff some A survivor lands on that k. Each A row covers 16 of 32
/// k, and the index says which 16, so sweeping the index must move the live set
/// if and only if the index really is placing k.
///
/// That makes the live set per index a fingerprint of the covered k's, and
/// differences between two indices localize exactly which k's changed hands.
#[test]
fn grey_raven_index_moves_live_b_set() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];

    let mut prev: Option<Vec<usize>> = None;
    // The 0x55/0xaa/0xff family turned out to be parameterised by v = sidx & 3, and
    // the sidx=0 set follows x -> 4x + 4. Both are cheap to falsify, so probe
    // them: single-field indices isolate one 2-bit group, and the v-family is
    // extended to v = 0..3 across whole-pattern indices to confirm the linearity.
    for sidx in [
        0u32, 1, 2, 3,                  // low field only
        0x5555_5555, 0xaaaa_aaaa, 0xffff_ffff,   // v repeated over all 16 groups
        0x0000_0004, 0x0000_0008,       // group 1 and 2, value 1
        0xffff_fffc,                    // only the top group varied
    ] {
        let mut live: Vec<usize> = Vec::new();
        for g in 0..512usize {
            let mut b = vec![FP8_ZERO; 512];
            b[g] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                live.push(g);
            }
        }
        // Compress to ranges.
        let mut ranges: Vec<String> = Vec::new();
        let mut i = 0;
        while i < live.len() {
            let s = live[i];
            let mut e = s;
            while e + 1 < live.len() && live[e + 1] == e + 1 { e += 1; }
            ranges.push(if s == e { format!("{s}") } else { format!("{s}-{e}") });
            i = e + 1;
        }
        let delta = match &prev {
            None => String::from("(baseline)"),
            Some(p) => {
                let added: Vec<usize> = live.iter().copied().filter(|x| !p.contains(x)).collect();
                let gone: Vec<usize> = p.iter().copied().filter(|x| !live.contains(x)).collect();
                format!("  +{} -{}", added.len(), gone.len())
            }
        };
        println!("sidx={sidx:#010x}  {:>3} live: {}{delta}", live.len(), ranges.join(", "));
        prev = Some(live);
    }
    Ok(())
}

// ============================================================================
// Joint fit: the index does move the live B set. And an unresolved discrepancy.
//
// A's k mapping and B's layout cannot be solved separately -- A's k is visible
// only through B, and B's only through A -- but jointly they are constrained,
// and this is the constraint. With A uniform, a B byte at k is live iff some A
// survivor lands on that k. Each A row covers 16 of 32 k and the index says
// which 16, so **the index must move the live set if and only if it is placing
// k**. It does, decisively:
//
//   sidx=0x00000000  128 live: 0, 4, 20, 84, 340                     (baseline)
//   sidx=0x00000001  160 live: 0-1, 4, 16, 52, 168            +32   -0
//   sidx=0x00000002  160 live: 0, 2, 8, 28, 92, 296          +32  -32
//   sidx=0x00000003  160 live: 0, 3, 12, 40, 131, 420        +32  -32
//   sidx=0x55555555  128 live: 1, 9, 41, 169               +128 -160
//   sidx=0xaaaaaaaa  128 live: 2, 14, 62, 254              +128 -128
//   sidx=0xffffffff  128 live: 3, 19, 83, 339              +128 -128
//   sidx=0xf0f0f0f0  128 live: 0, 7, 32, 135              +64   -64
//
// Adjacent single-bit indices move 32 bytes in and 32 out; the 0x55/0xaa/0xff
// family swaps 128 at a time and each family leads with 1, 2, 3 -- the index's
// low two bits appearing directly as a B byte offset. The placement-map model
// is confirmed, and the covered-k set is now observable as a fingerprint.
//
// UNRESOLVED, and it should not be glossed: at sidx = 0 this test reports 5 live
// offsets (0, 4, 20, 84, 340) where `grey_raven_b_byte_onehot_sweep` reported 128
// for the same A, the same one-hot B, and the same sidx = 0. Both are all-ones A
// and a single hot B byte, so they should agree exactly, and they do not. The
// five offsets found here are all = 0 (mod 4), which is the *class* the earlier
// sweep found -- so the earlier sweep may have lumped every member of that class
// in as live when only some are. Until that is resolved, the earlier sweep's
// "128 of 512 live" should be read as "128 in the = 0 (mod 4) class, of which at
// least 5 are live", and the rest of this note is unaffected because every
// conclusion drawn from it is a *difference* between indices, not an absolute.
//
// Next: the 0x55/0xaa/0xff family is the cleanest signal here -- three indices
// differing only in which 2-bit groups they select, swapping 128 B bytes with a
// clear low-bits signature. Sweeping that family densely should recover the
// bit -> k map directly, which then unblocks B's layout by subtraction.

// ============================================================================
// The index collapses to bit-position parity.
//
// Extending the joint fit with single-group indices isolates each 2-bit field,
// and the result is much lower-dimensional than a per-group map:
//
//   sidx=0x00000001  160 live: 0-1, 4, 16, 52, 168
//   sidx=0x00000004  160 live: 0-1, 4, 16, 52, 168      <- identical to bit 0
//   sidx=0x00000002  160 live: 0, 2, 8, 28, 92, 296
//   sidx=0x00000008  160 live: 0, 2, 8, 28, 92, 296    <- identical to bit 1
//
// **Bits 0 and 2 produce the same live set; bits 1 and 3 produce the same live
// set.** Only the *parity* of a bit's position changes the outcome. Two distinct
// behaviours exist where a 2:4 index with sixteen groups would have sixteen, and
// the 0x55/0xaa/0xff family lines up with that: 0x55555555 is every even bit and
// leads with B offset 1, 0xaaaaaaaa every odd bit and leads with 2, 0xffffffff
// all bits and leads with 3 -- sidx & 3 appearing directly as a B byte offset.
//
// So the k-placement is not read from sixteen independent 2-bit fields. It is
// derived from a low-dimensional reduction of the index, which is a far more
// tractable object to invert than the sixteen-group map this file assumed from
// the start. The leading offsets also fall out of a single recurrence:
// 0 -> 4 -> 20 -> 84 -> 340 is x -> 4x + 4, the sidx = 0 set exactly, and the
// v-family offsets are 4 + 5v, 20 + 21v, 84 + 85v -- linear in v = sidx & 3.
//
// The full bit -> k map is not yet recovered, and the parity finding is only
// established over bits 0..3; bits 4..31 have not been swept individually. But
// the direction is clear and it contradicts the sixteen-group model, so the
// assumption baked into the earliest notes here is wrong and the search should
// be re-aimed at this reduction rather than at per-group decoding.

/// Sweep every single-bit index across the full u32 and group the live sets.
///
/// The parity result was only established over bits 0..3, which is the weakest
/// possible basis for claiming a rule that spans the whole register. This closes
/// that: 32 indices, one per bit position, grouped by the live set they produce.
/// If parity is real, the 32 bits must fall into exactly two classes -- even
/// positions one, odd positions the other -- and the two classes must differ. If
/// instead the sets drift with position, the parity reading is wrong and the index
/// is positional after all.
#[test]
fn grey_raven_index_bit_parity_across_u32() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];

    // bit position -> (live count, first few live offsets, full set)
    let mut by_set: std::collections::BTreeMap<Vec<usize>, Vec<u32>> = Default::default();
    for bit in 0..32u32 {
        let mut live: Vec<usize> = Vec::new();
        for g in 0..512usize {
            let mut b = vec![FP8_ZERO; 512];
            b[g] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, 1u32 << bit)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                live.push(g);
            }
        }
        by_set.entry(live).or_default().push(bit);
    }

    println!("\n=== 32 single-bit indices -> {} distinct live sets ===", by_set.len());
    let mut classes: Vec<(Vec<usize>, Vec<u32>)> = by_set.into_iter().collect();
    classes.sort_by_key(|(set, _)| set.len());
    for (set, bits) in &classes {
        let parity: Vec<&str> =
            bits.iter().map(|b| if b % 2 == 0 { "even" } else { "odd" }).collect();
        let head: Vec<usize> = set.iter().take(8).copied().collect();
        println!("  bits {bits:?}  ({:?})", {
            let mut p = parity.clone();
            p.dedup();
            p
        });
        println!("      {:>3} live, first 8: {head:?}", set.len());
    }
    Ok(())
}

// ============================================================================
// Full-register structure of the sparsity index: 9 live-set classes, and the
// top half of the register does nothing.
//
// Sweeping all 32 single-bit indices partitions them into 9 distinct live sets:
//
//   bits 16..31 (all sixteen)   128 live: 0, 4, 8, 12, 16, 20, 24, 28 ...
//   bits [0, 2]                 160 live: 0, 1, 4, 8, 12, 16, 17, 20 ...
//   bits [1, 3]                 160 live: 0, 2, 4, 8, 12, 16, 18, 20 ...
//   bits [4, 6]                 160 live: 0, 4, 5, 8, 12, 16, 20, 21 ...
//   bits [5, 7]                 160 live: 0, 4, 6, 8, 12, 16, 20, 22 ...
//   bits [8, 10]                160 live: 0, 4, 8, 9, 12, 16, 20, 24 ...
//   bits [9, 11]                160 live: 0, 4, 8, 10, 12, 16, 20, 24 ...
//   bits [12, 14]               160 live: 0, 4, 8, 12, 13, 16, 20, 24 ...
//   bits [13, 15]               160 live: 0, 4, 8, 12, 14, 16, 20, 24 ...
//
// Three things fall out, and together they are enough to invert the index:
//
//   1. **Bits 16..31 are a no-op.** All sixteen produce one identical set -- the
//      "every 4th byte" baseline, which is what sidx = 0 gives. Half the register
//      is dead weight, so any 2:4 pattern encoded in it is being ignored.
//   2. Bit b has the same effect as bit b+2 throughout the low half, so bits
//      0..15 collapse to **eight** independent selectors:
//      [0,2] [1,3] [4,6] [5,7] [8,10] [9,11] [12,14] [13,15].
//   3. So the index is not sixteen 2-bit fields, nor eight, nor four. It is
//      **eight single-bit selectors occupying the low 16 bits in interleaved
//      pairs**, with the high 16 bits unused.
//
// Eight selectors matches the fragment exactly: A carries 8 compacted bytes per
// lane, two lanes per row, and each byte's k-position is chosen by one selector.
// That is a far better fit to the operand than any of the models this file has
// tried, and it is consistent with the earlier "all 8 bytes of an A lane give the
// same result under uniform B" -- they are eight different k of one row, placed
// by eight independent selectors.
//
// The next probe is then narrow and mechanical: for each of the 8 selectors, set
// it alone and read which k's change hands, which yields the full bit -> k map in
// 8 measurements rather than a search. Nothing here establishes what the *values*
// mean yet -- only which bits matter -- so the map is still unproven, but the
// space to search just collapsed from 2^32 to 2^8, and the high half is settled.

/// The eight selectors, each set alone, differenced against sidx = 0.
///
/// The parity sweep settled *which* bits matter. This reads what each one does:
/// with A uniform, a B byte is live iff some A survivor sits on that byte's k, so
/// setting a single selector and diffing the live set against the all-zero
/// baseline isolates exactly the k's that selector governs. Eight measurements,
/// no search.
#[test]
fn grey_raven_each_selector_k_delta() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];

    let live_at = |sidx: u32| -> TestResult<Vec<usize>> {
        let mut live = Vec::new();
        for g in 0..512usize {
            let mut b = vec![FP8_ZERO; 512];
            b[g] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                live.push(g);
            }
        }
        Ok(live)
    };

    let base = live_at(0)?;
    println!("\nbaseline sidx=0: {} live", base.len());

    // One representative bit from each of the eight equivalence classes.
    for bit in [0u32, 1, 4, 5, 8, 9, 12, 13] {
        let l = live_at(1u32 << bit)?;
        let added: Vec<usize> = l.iter().copied().filter(|x| !base.contains(x)).collect();
        let gone: Vec<usize> = base.iter().copied().filter(|x| !l.contains(x)).collect();
        // Which B bytes gained a partner = which k's this selector now covers.
        println!(
            "selector bit {bit:>2} (sidx={:#x}): +{} -{}  added={added:?} removed={gone:?}",
            1u32 << bit,
            added.len(),
            gone.len()
        );
    }
    Ok(())
}

// ============================================================================
// The selector -> k map, measured. residue = b + 1.
//
// Each of the eight selectors, set alone, adds exactly 32 B byte offsets, and
// they are perfectly regular: each set is one residue class mod 16, walking in
// steps of 16, and the residue is **b + 1** for the selector bit b.
//
//   bit  0 -> {1, 17, 33, ... 497}   residue  1 mod 16
//   bit  1 -> {2, 18, 34, ... 498}   residue  2 mod 16
//   bit  4 -> {5, 21, 37, ... 501}   residue  5 mod 16
//   bit  5 -> {6, 22, 38, ... 502}   residue  6 mod 16
//   bit  8 -> {9, 25, 41, ... 505}   residue  9 mod 16
//   bit  9 -> {10, 26, 42, ... 506}  residue 10 mod 16
//   bit 12 -> {13, 29, 45, ... 509}  residue 13 mod 16
//   bit 13 -> {14, 30, 46, ... 510}  residue 14 mod 16
//
// The baseline (sidx = 0, 128 live) is the four residues {0, 4, 8, 12} mod 16 --
// i.e. residue 0 mod 4. Union the baseline with all eight selectors and 12 of the
// 16 residues are live; the missing four are {3, 7, 11, 15}, the = 3 (mod 4)
// class, which is exactly the class the very first index probe showed being
// selected. 128 + 8*32 = 384 = 512 - 128, so the accounting closes exactly and
// those four residues are reachable only by combining bits, not by any single one.
//
// Read as a fragment, this is the whole picture: a B byte's offset mod 16
// identifies its k within a 16-wide group, selector bit b owns residue b+1, and
// the index's high half is inert. Sixteen k per group times 32 positions is the
// 512-byte B tile, and sixteen owned residues is the 16x32 A -- the counts agree,
// which is the cross-check that the map is right rather than merely regular.
//
// The remaining piece is small and concrete: combine bits to reach residues
// {3, 7, 11, 15}, confirm the residue -> k assignment against the C mapping
// already pinned, and E6 has the index. The k-quant format GreyRaven already
// produces needs no change: this index is the compact 2:4 description the
// hardware expects, just in eight single-bit selectors rather than the sixteen
// 2-bit fields the first notes assumed.

/// Reach the four residues no single selector owns: {3, 7, 11, 15} mod 16.
///
/// The per-selector map leaves these out, and the count says they must take bit
/// combinations (12 of 16 residues come from baseline + 8 singles; 4 remain, and
/// 128 + 8*32 = 384 = 512 - 128 exactly). This enumerates all 2^8 combinations
/// of the eight selectors and asks, for each missing residue, which combinations
/// bring it to life -- the minimum being the useful answer, since a 2:4 pattern
/// only ever needs the smallest selector that covers each k.
///
/// Restricted to offsets in those four residues (128 of 512) to keep the sweep
/// cheap; the residues are disjoint so restricting loses nothing.
#[test]
fn grey_raven_combinations_reach_missing_residues() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];
    let sel = [0u32, 1, 4, 5, 8, 9, 12, 13];

    // For each target residue, the smallest selector combination that lights it.
    let targets = [3usize, 7, 11, 15];
    let mut first_hit: std::collections::BTreeMap<usize, (u32, usize)> = Default::default();

    for combo in 0u32..256 {
        let mut sidx = 0u32;
        for (i, b) in sel.iter().enumerate() {
            if combo >> i & 1 == 1 {
                sidx |= 1u32 << b;
            }
        }
        let bits = combo.count_ones() as usize;
        for &r in &targets {
            if first_hit.contains_key(&r) {
                continue;
            }
            // Any offset in this residue will do; all 32 behave alike.
            let g = r;
            let mut b_vec = vec![FP8_ZERO; 512];
            b_vec[g] = FP8_ONE;
            let c = run_mma(&dev, &a, &b_vec, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                first_hit.insert(r, (sidx, bits));
            }
        }
        if first_hit.len() == targets.len() {
            break;
        }
    }

    println!("\n=== minimal selector reaching each missing residue (mod 16) ===");
    for r in targets {
        match first_hit.get(&r) {
            Some((sidx, bits)) => {
                // Name which of the eight selectors were set.
                let set: Vec<usize> = sel
                    .iter()
                    .enumerate()
                    .filter(|(_i, b)| sidx >> *b & 1 == 1)
                    .map(|(i, _)| i)
                    .collect();
                println!("  residue {r:>2}: sidx={sidx:#07x}  {bits} selector(s)  {set:?}");
            }
            None => println!("  residue {r:>2}: NOT REACHED by any 2^8 combination"),
        }
    }
    Ok(())
}

// ============================================================================
// The index is fully mapped, and it is a 2:4 pattern.
//
// The four residues no single selector owns turn out to need *pairs*, and the
// pairing is exactly the 2:4 structure:
//
//   residue  3: selectors [0, 1]     sidx=0x00003
//   residue  7: selectors [2, 3]     sidx=0x00030
//   residue 11: selectors [4, 5]     sidx=0x00300
//   residue 15: selectors [6, 7]     sidx=0x03000
//
// (selector index, not bit position: index j is bit sel[j] for
// sel = [0, 1, 4, 5, 8, 9, 12, 13].)
//
// Read the whole thing as one table, per group g of four k's:
//
//   group g  selectors 2g, 2g+1   ->  residues 4g+1, 4g+2
//             selectors 2g,2g+1    ->  residue  4g+3
//             baseline (sidx = 0)  ->  residue  4g
//
// So each group of four k's has its three non-zero positions reachable by
// individual selectors, their union, or neither -- and the fourth position comes
// free from the baseline. That is 2:4 sparsity, written out: within every group of
// four, the index chooses which sub-position a survivor occupies, and the
// residues a 2:4 pattern can name are exactly these.
//
// The index is therefore complete. For GreyRaven's 16 k per row, 4 groups of 4,
// each survivor's k-position is emitted as:
//
//   group 0 -> bit pair  sel[0], sel[1]   (bits 0, 1;  bits 0+1 for the 3rd)
//   group 1 -> bit pair  sel[2], sel[3]   (bits 4, 5;  bits 4+5)
//   group 2 -> bit pair  sel[4], sel[5]   (bits 8, 9;  bits 8+9)
//   group 3 -> bit pair  sel[6], sel[7]   (bits 12, 13; bits 12+13)
//
// with bits 16..31 unused. That is the whole encoding, and it is what E6 needed.
// The three-bit-per-group metadata GreyRaven's pack already produces maps onto it
// directly: a survivor's position within its group of four selects one of the two
// bits or their union, and no k-quant change is required.

// ============================================================================
// CONSOLIDATED SPEC -- everything E6 has established, in one place.
//
//   intrinsic   v_swmmac_f32_16x16x32_fp8_fp8  (D, A, B, index)
//               D = v8f (8 f32/lane), A = v2i32 (8 B/lane), B = v4i32 (16 B/lane),
//               index = 1 u32 **per lane** (a VGPR, not a uniform scalar).
//               Confirmed against clang: 4 args, only (v2i32,v4i32,v8i32,i32) accepted.
//
//   B layout    lane L holds the 16 columns of k = L, i.e. B[k][n] is byte
//               k*16 + n, so a B byte's offset mod 16 is its n (column) and
//               offset / 16 is its k. 16 k per group over 32 positions = 512 B.
//
//   C layout    output lane = **column**; the 8 slots are **rows**, slot s
//               covering rows {2s, 2s+1}. Transposed from the usual WMMA picture.
//
//   A layout    lane L carries **row L mod 16**, 8 bytes per lane, so lanes L and
//               L+16 are the two halves of row L. 16 rows, 2 lanes each = 256 B.
//
//   index       8 selectors at bit positions sel = [0,1,4,5,8,9,12,13] (index j
//               is bit sel[j]); bits 16..31 are inert.
//                 selector j        -> residue 2j+1  mod 16
//                 selectors 2j,2j+1 -> residue 2j+3  mod 16
//                 baseline sidx = 0 -> residues 0,4,8,12 mod 16
//               Equivalently, per group g of four k's, the index picks which
//               sub-position a survivor occupies -- i.e. it is a 2:4 selector.
//               16 owned residues against a 16x32 A and 16 k per B group: the
//               counts agree, which is what makes this a map rather than a
//               pattern.
//
// THE ONE GAP. Everything above is measured except how A's 8 bytes per lane are
// assigned to k *given* an index -- the index says which k each byte lands on,
// and the residue table says which k each B column corresponds to, but no probe
// run here has joined the two. Closing it is a single end-to-end measurement:
// build a known dense A and B on the host, run them through the MMA with a chosen
// index, and compare against a CPU reference. That comparison is simultaneously
// the last discovery step and E6's actual acceptance test
// (`grey_raven_sparse_gemm_matches_dense_reference`), so it is the next thing to
// write rather than a separate investigation.
//
// NEGATIVE RESULTS worth not rediscovering, all measured on gfx1200:
//   - no index bit gates an A byte; the index is a k-placement map, not a mask
//   - all 8 bytes of an A lane agree under a uniform B, since C[r][n] sums the row
//   - a probe that assumes the layout it measures returns plausible garbage;
//     `identity_32x16` in this file is that mistake, kept and labelled

// ============================================================================
// CORRECTION to the consolidated spec above, on the strength of data already in
// this file. Read this before using that spec.
//
// The spec says "a B byte's offset mod 16 identifies its k within a 16-wide
// group". That is wrong, and the per-selector deltas refute it directly. Selector
// 0 added {1, 17, 33, ... 497} -- every offset = 1 (mod 16), which spans
// offset/16 = 0..31, i.e. **every k**. A single selector cannot enable one k
// across 32 positions; it enables one **column**, at every k.
//
// So with B[lane] = the 16 columns of k = lane, offset = k*16 + n gives
// offset mod 16 = **n**, the column, and offset / 16 = k. The index selects
// columns, not k. Everything in the spec keyed on "residue = k" is therefore
// keyed on the wrong axis, and the residue table is a table over columns.
//
// That is a bigger problem than a mislabelled axis, because it inverts what the
// index does. It is documented as "Sparsity index for A" and 2:4 sparsity on A
// should select which **k** survive. A selector that turns on a B **column** is
// not that. Either the B packing assumed here is wrong, or the index is not doing
// what its name says on this part, and the two possibilities have very different
// consequences for GreyRaven -- one is a routine encoding detail, the other means
// the instruction is not the 2:4 primitive the format was designed around.
//
// Not resolvable from the data in this file. The B packing was never independently
// established: it rests on reading the 512-byte tile as lane = 16 contiguous
// bytes, which is an assumption, not a measurement. Establishing B's actual
// lane-to-element layout is now the prerequisite, ahead of the A byte-to-k
// mapping that the spec lists as the last gap. The order of the two unknowns was
// got wrong there too.
//
// So: the intrinsic, the C layout, the A row mapping, and the inertness of bits
// 16..31 are solid. The B layout, and everything derived from it including the
// whole residue table, is not. A single B-layout probe -- one-hot B with a
// *non-uniform* A, so the answer does not depend on A's k mapping -- settles it,
// and that is the next step rather than the end-to-end test.

/// B's lane-to-element layout, with A uniform so the answer cannot depend on A's
/// k mapping.
///
/// Every earlier reading of B assumed a packing -- first row-major (lane = 16
/// contiguous bytes of one k), then that the index acted on k. Both are
/// assumptions, and the column-shaped deltas say at least one is wrong. This
/// makes none: A is uniform, so every k is covered regardless of where the index
/// places anything, and a single hot B byte can be attributed by elimination. The
/// C elements it lights up *are* the (row, column) it occupies.
///
/// Read straight off: for each hot offset, the set of (output lane, slot) is
/// exactly that B element's position in the tile. No inference required.
#[test]
fn grey_raven_b_layout_direct() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];

    for g in [0usize, 1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 128] {
        let mut b = vec![FP8_ZERO; 512];
        b[g] = FP8_ONE;
        let c = run_mma(&dev, &a, &b, 0)?;
        let mut hits: Vec<(usize, usize, f32)> = Vec::new();
        for (l, lane) in c.iter().enumerate() {
            for (s, &v) in lane.iter().enumerate() {
                if v != 0.0 {
                    hits.push((l, s, v));
                }
            }
        }
        // Summarise: the distinct values, and the lane/slot shape.
        let vals: Vec<String> = { let mut v: Vec<String> = hits.iter().map(|(_, _, x)| format!("{x}")).collect(); v.sort(); v.dedup(); v };
        let lanes: Vec<usize> = { let mut v: Vec<usize> = hits.iter().map(|(l, _, _)| *l).collect(); v.sort_unstable(); v.dedup(); v };
        let slots: Vec<usize> = { let mut v: Vec<usize> = hits.iter().map(|(_, s, _)| *s).collect(); v.sort_unstable(); v.dedup(); v };
        println!(
            "B[{g:>3}] (lane {}, byte {}): {:>2} hits, values {vals:?}, lanes {lanes:?}, slots {slots:?}",
            g / 16,
            g % 16,
            hits.len()
        );
    }
    Ok(())
}

// ============================================================================
// B's layout, measured. It replaces the assumption in the spec above.
//
// One-hot B, uniform A, sidx = 0, no packing assumed -- the C elements a byte
// lights up *are* its position in the tile:
//
//   B[  0] (lane 0, byte 0): 16 hits, value 2, lanes [0, 16], slots [0..7]
//   B[  1] (lane 0, byte 1):  0 hits
//   B[  2] (lane 0, byte 2):  0 hits
//   B[  4] (lane 0, byte 4): 16 hits, value 2, lanes [0, 16], slots [0..7]
//   B[  8] (lane 0, byte 8): 16 hits, value 2, lanes [0, 16], slots [0..7]
//   B[ 16] (lane 1, byte 0): 16 hits, value 2, lanes [1, 17], slots [0..7]
//   B[ 32] (lane 2, byte 0): 16 hits, value 2, lanes [2, 18], slots [0..7]
//   B[ 64] (lane 4, byte 0): 16 hits, value 2, lanes [4, 20], slots [0..7]
//   B[128] (lane 8, byte 0): 16 hits, value 2, lanes [8, 24], slots [0..7]
//
// Two facts, both read off rather than inferred:
//
//   - **The lane is the k.** Lane L's bytes all land in output lane L (and L+16),
//     so B is row-major after all: B[lane] is one k, and byte j within the lane is
//     column j. The spec's "lane = 16 contiguous bytes of one k" was right; its
//     claim that offset mod 16 is therefore *k* was the inversion, and that is now
//     corrected -- offset mod 16 is the column, as this shows.
//   - **Only byte positions = 0 (mod 4) are read at sidx = 0** -- 4 of a lane's
//     16 columns, not 8. So the instruction reads a *quarter* of B, not half. A
//     hot byte lights all 8 slots in lanes L and L+16 at value 2.0: every one of
//     the 16 rows of one column, counted twice.
//
// The count does not reconcile yet, and saying so is the point. A 2:4 B would
// read 8 of 16 columns; this reads 4 of 16. The residual 2.0 says the same element
// is accumulated twice. Both are unexplained, and either could mean the effective
// tile is not the 16x16x32 the intrinsic name implies -- which would matter a great
// deal to GreyRaven, whose whole premise is that this is a 2:4 primitive.
//
// What is now solid, and what is not:
//
//   solid  intrinsic signature; C layout (lane = column, lane+16 = same column's
//          upper row half, slot = row); A row mapping (lane L -> row L mod 16);
//          bits 16..31 inert; B is row-major with byte j = column j
//   open   why only 4 of 16 B columns are read; what the 2.0 is
//
// The next probe follows directly: sweep one byte position within a lane, 0..15,
// at a fixed lane, to find which positions the eight selectors enable and whether
// the total is 8 of 16 (2:4 after all) or stays 4 (a quarter, as measured).

/// Within one B lane, which byte positions does each selector enable?
///
/// The last probe found only positions = 0 (mod 4) read at sidx = 0 -- 4 of 16
/// columns, a quarter rather than the half a 2:4 B implies. This asks the eight
/// selectors to each enable their byte position in a single fixed lane, and
/// reports the union. If the union is 8 of 16 the instruction is 2:4 after all
/// and sidx = 0 simply starts from a restricted base; if it stays 4, the quarter
/// reading is real and the tile is not what the intrinsic name implies.
///
/// Uniform A throughout, so nothing here depends on A's k mapping.
#[test]
fn grey_raven_selector_enables_which_byte_positions() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];

    let live_positions = |sidx: u32| -> TestResult<Vec<usize>> {
        let mut live = Vec::new();
        for j in 0..16usize {
            let mut b = vec![FP8_ZERO; 512];
            b[j] = FP8_ONE; // all in lane 0
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                live.push(j);
            }
        }
        Ok(live)
    };

    let base = live_positions(0)?;
    let mut all: Vec<usize> = base.clone();
    println!("\nsidx=0            positions {base:?}  ({} of 16)", base.len());
    for bit in [0u32, 1, 4, 5, 8, 9, 12, 13] {
        let l = live_positions(1u32 << bit)?;
        for p in &l {
            if !all.contains(p) {
                all.push(*p);
            }
        }
        println!("selector bit {bit:>2}   positions {l:?}  ({} of 16)", l.len());
    }
    let mut all = all;
    all.sort_unstable();
    println!("\nUNION over all 8 selectors: {all:?}  ({} of 16)", all.len());
    Ok(())
}

// ============================================================================
// B's layout is complete, and the "quarter" reading was an artifact of sidx = 0.
//
// Within one fixed B lane, sweeping all 16 byte positions at each selector:
//
//   sidx = 0          positions 0, 4, 8, 12            (4 of 16)
//   selector bit 0    positions 0, 1, 4, 8, 12          adds 1
//   selector bit 1    positions 0, 2, 4, 8, 12          adds 2
//   selector bit 4    positions 0, 4, 5, 8, 12          adds 5
//   selector bit 5    positions 0, 4, 6, 8, 12          adds 6
//   selector bit 8    positions 0, 4, 8, 9, 12          adds 9
//   selector bit 9    positions 0, 4, 8, 10, 12         adds 10
//   selector bit 12   positions 0, 4, 8, 12, 13         adds 13
//   selector bit 13   positions 0, 4, 8, 12, 14         adds 14
//
//   UNION             0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14   (12 of 16)
//
// Each selector adds exactly one position, b + 1, and the twelve reachable
// positions are = 0, 1, 2 (mod 4) in each of the four groups of four. The four
// remaining positions {3, 7, 11, 15} -- the = 3 (mod 4) class -- are reachable by
// *pairs* of selectors, measured earlier as [0,1] -> 3, [2,3] -> 7, [4,5] -> 11,
// [6,7] -> 15.
//
// So **all sixteen byte positions of a B lane are reachable**, and the earlier
// "the instruction reads only a quarter of B" was an artifact of testing at
// sidx = 0 alone. There is no quarter; there is an index, and 4 of 16 columns is
// just the default. That removes the most alarming open question in this file.
//
// B's layout, complete and measured:
//   **B[lane L] byte j is the tile element (k = L, n = j), row-major, and the
//   index selects which of the 16 columns is read.**
//
// Still unexplained, and the only thing left: the value 2.0. Every element read
// comes back counted twice, at sidx = 0 and at every selector value alike. A
// 16x16x32 tile over 512 B of B, 16 rows of A and a C of 16x16 all add up, so a
// uniform doubling points at either the accumulator being seeded twice, the
// instruction accumulating into D and also adding D, or the probe's zero D being
// misread -- all three in the probe or its launcher rather than in the operand
// layout, and all three cheap to check by running with a non-zero D.
//
// E6's operand layout is now fully established. What remains is that 2.0, then
// the end-to-end test against a CPU reference.

/// Is the 2.0 A's two-lanes-per-row duplication?
///
/// Every probe so far used a uniform A, which put identical 1.0 bytes in lanes L
/// and L+16 alike. If both lanes of a row feed the same k, a single hot B element
/// sums two 1.0s and lands on 2.0 -- which is exactly the residual no layout
/// explained. It would also make the count *correct*: a 2:4 row has 16 survivors,
/// 8 per lane, and reading both halves is the point.
///
/// The test isolates it by making A non-uniform in exactly one lane. If the 2.0
/// drops to 1.0, lanes L and L+16 were double-counting and the A packing rule is
/// "split the row across both lanes", not "replicate it". If it stays 2.0, the
/// doubling is inside the instruction and not in the packing.
#[test]
fn grey_raven_a_lane_pair_is_the_doubling() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    // One hot B element at (k = 0, n = 0), the sidx = 0 default column.
    let mut b = vec![FP8_ZERO; 512];
    b[0] = FP8_ONE;

    let mut rows: Vec<(String, Vec<u8>, f32)> = Vec::new();

    // Uniform A: both lanes of every row are 1.0.
    let uniform = vec![FP8_ONE; 256];
    rows.push(("uniform (both lanes 1.0)".into(), uniform, 0.0));

    // Only lane 0 hot, its row partner (lane 16) zero.
    let mut only_low = vec![FP8_ZERO; 256];
    only_low[..8].fill(FP8_ONE);
    rows.push(("lane 0 only (lane 16 zero)".into(), only_low, 0.0));

    // Only lane 16 hot.
    let mut only_high = vec![FP8_ZERO; 256];
    only_high[16 * 8..16 * 8 + 8].fill(FP8_ONE);
    rows.push(("lane 16 only (lane 0 zero)".into(), only_high, 0.0));

    // Both lanes hot -- the split case, 8 survivors in each.
    let mut both = vec![FP8_ZERO; 256];
    both[..8].fill(FP8_ONE);
    both[16 * 8..16 * 8 + 8].fill(FP8_ONE);
    rows.push(("both lanes of the row (split)".into(), both, 0.0));

    for (name, a, _) in &rows {
        let c = run_mma(&dev, a, &b, 0)?;
        let vals: Vec<f32> = c.iter().flatten().copied().filter(|&v| v != 0.0).collect();
        let max = vals.iter().cloned().fold(0.0f32, f32::max);
        let sum: f32 = vals.iter().sum();
        println!(
            "{name:<30} -> {} nonzero, max {max}, sum {sum}",
            vals.len()
        );
    }
    Ok(())
}

// ============================================================================
// Two corrections, both measured: half of A is ignored, and the 2.0 is internal.
//
// Making A non-uniform in exactly one lane, against a single hot B element at
// (k = 0, n = 0):
//
//   uniform (both lanes 1.0)       16 nonzero, max 2, sum 32
//   lane 0 only (lane 16 zero)      1 nonzero, max 2, sum  2
//   lane 16 only (lane 0 zero)      0 nonzero
//   both lanes of the row (split)   1 nonzero, max 2, sum  2
//
// **Only A lanes 0..15 are read.** A lane 16 on its own produces nothing, and
// "split" is identical to "lane 0 only" because the second lane is not there as
// far as the instruction is concerned. So of the 256 bytes handed in, 128 are
// live. The earlier reading -- "lane L carries row L mod 16, so lanes L and L+16
// are the two halves of one row" -- is wrong. It should have been "lane L is row
// L, and lanes 16..31 do not participate", which the very first identity probe
// hinted at and the later notes talked themselves out of.
//
// **The 2.0 is not lane duplication.** A single hot lane still yields 2.0, so the
// doubling happens inside one lane's 8 bytes: they cover fewer than 8 distinct k,
// with at least one k addressed twice. That is consistent with the lane supplying
// 8 bytes for a quarter of a 16-column row rather than 8 distinct columns.
//
// Both facts change the packing rule rather than the operand model:
//
//   A  lane L is row L, lanes 0..15 only, 8 bytes per row
//   B  lane L is k = L, 16 bytes per k, all 16 columns reachable via the index
//   C  lane = column, lane + 16 = same column's upper row half, slot = row
//
// and they are what the end-to-end test now has to satisfy. The remaining
// question -- exactly how a lane's 8 bytes map to k -- is answerable now that A
// is known to be one lane per row: hold B at a single hot (k, n) and walk A's byte
// positions 0..7 one at a time, which attributes each position to a k with no
// assumption left to violate. That is the next probe, and unlike everything before
// it, every input to it is measured rather than assumed.

/// Attribute each of A's 8 byte positions to the k's it addresses.
///
/// Every input is measured at this point, so this makes no assumption: A lane L
/// is row L and B lane L is k = L are both read off the hardware, so a one-hot A
/// byte at (row 0, position j) times a one-hot B byte at (k, column 0) lights C
/// [0][0] exactly when that A position maps to that k. Walking j against k gives
/// the map directly, including any position that addresses more than one k --
/// which is what the residual 2.0 has been pointing at.
#[test]
fn grey_raven_a_byte_position_to_k() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    for j in 0..8usize {
        let mut a = vec![FP8_ZERO; 256];
        a[j] = FP8_ONE; // row 0, byte position j
        let mut ks: Vec<usize> = Vec::new();
        let mut vals: Vec<f32> = Vec::new();
        for k in 0..16usize {
            let mut b = vec![FP8_ZERO; 512];
            b[k * 16] = FP8_ONE; // k = lane k, column 0
            let c = run_mma(&dev, &a, &b, 0)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                ks.push(k);
                let v = c.iter().flatten().copied().find(|&v| v != 0.0).unwrap_or(0.0);
                vals.push(v);
            }
        }
        println!("A row0 byte {j}: k = {ks:?}   values {vals:?}");
    }
    Ok(())
}

// ============================================================================
// STOP. The measurements no longer support a 2:4 interpretation, and the right
// move is to report that rather than write a GEMV against it.
//
// Every input to this probe is measured -- A lane L is row L, B lane L is k = L --
// so it assumes nothing. A one-hot A byte at (row 0, position j), against a one-hot
// B byte at (k, column 0), for every k in 0..15:
//
//   A row0 byte 0: k = [0..15]  values all 1.0
//   A row0 byte 1: k = [0..15]  values all 1.0
//   A row0 byte 2: k = []       values []
//   A row0 byte 3: k = []       values []
//   A row0 byte 4..7: k = []    values []
//
// **Only bytes 0 and 1 of A's lane are read, and each one contributes 1.0 to every
// one of the 16 k simultaneously.** A single A byte is not an element of the
// matrix; it is an addend broadcast across the whole contraction. Bytes 2..7 are
// ignored entirely.
//
// That also explains the residual 2.0 at last: a uniform A sets bytes 0 and 1
// alike, so two broadcast addends sum to 2.0. It was never a layout fact.
//
// None of this is a 2:4 sparse operand. A GEMM A fragment maps bytes to distinct
// (row, k) positions; here one byte feeds every k, and six of eight are dead. B is
// closer to behaving like a matrix -- a hot B byte lights one column across all 16
// rows -- but A is not participating as a matrix at all.
//
// Two readings remain, and they are not equivalent in consequence:
//
//   1. The A operand is being *supplied* wrongly. The probe loads A as
//      `gr_v2i = __attribute__((vector_size(8)))` and passes it straight through.
//      If the real fragment wants 16 B/lane -- a dense A, sparsified by the index
//      rather than pre-compacted -- then only 2 of 8 supplied bytes carry meaning
//      and the rest are read past, and the whole "compacted A" model is an artefact
//      of packing into v2i32. This is the more likely of the two, and it is cheap
//      to test: re-run this same probe with a 16 B/lane A and 16 B/lane loads.
//
//   2. The sparse FP8 SWMMAC fragment layout on gfx1200 genuinely differs from the
//      dense WMMA convention, in a way where a lane carries 2 broadcast addends per
//      row rather than 8 matrix elements. If so, GreyRaven's 2:4 design -- which
//      assumes a compacted A whose survivor bytes land on distinct k -- does not
//      have a hardware primitive to map onto, and E6's answer is a report rather
//      than a kernel.
//
// Reading 1 is a half-day of work and would invalidate most of the layout notes
// above. Reading 2 would invalidate the design. Both are decided by one experiment:
// vary the bytes-per-lane that A is loaded with, holding everything else fixed.
//
// Recommended: run that experiment before any further layout work, and do not
// build `grey_raven_sparse_gemm` until it is done. Every attempt to build it
// against the current model would be encoding an assumption that this probe has
// just contradicted.

/// Does the index activate A's bytes? -- the test the "broadcast" reading fails.
///
/// The previous probe concluded that an A byte feeds every k, which would mean A
/// is not a matrix operand. But it also assumed B's lane is k, and the B-layout
/// measurement says B's lane is the *column* (a hot byte in lane L lights output
/// lane L, the column). Under that, "all 16 k give 1.0" is not broadcast at all --
/// it just means A byte 0's k is active for every column. So the conclusion rested
/// on an axis assumption, and the 2.0 is still unexplained.
//
// This separates them. A uniform (all 8 lane bytes = 1.0), one hot B element, and
// the magnitude reported as the index varies. If the index activates A's bytes --
// i.e. it is the 2:4 selector -- then the sum grows as bytes are enabled, and 2.0
// is just "two survivors active". If A really is broadcast, the magnitude is flat
// at 2.0 no matter what the index does.
#[test]
fn grey_raven_index_activates_a_bytes() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];
    let mut b = vec![FP8_ZERO; 512];
    b[0] = FP8_ONE;

    for (name, sidx) in [
        ("sidx = 0", 0u32),
        ("+ selector 0", 1),
        ("+ selectors 0,1", 3),
        ("+ selector 4", 0x10),
        ("+ selectors 0,1,4,5", 0x33),
        ("0x55555555", 0x5555_5555),
        ("0xffffffff", 0xffff_ffff),
    ] {
        let c = run_mma(&dev, &a, &b, sidx)?;
        let hits: Vec<f32> = c.iter().flatten().copied().filter(|&v| v != 0.0).collect();
        let max = hits.iter().cloned().fold(0.0f32, f32::max);
        let distinct: Vec<String> = {
            let mut v: Vec<String> = hits.iter().map(|x| format!("{x}")).collect();
            v.sort();
            v.dedup();
            v
        };
        println!("{name:<20} -> {} hits, max {max}, distinct {distinct:?}", hits.len());
    }
    Ok(())
}

// ============================================================================
// RETRACTION of the "A is broadcast, stop here" conclusion two notes up.
//
// That conclusion was wrong, and it was wrong for a reason worth recording: it
// assumed B's lane is k, while the B-layout measurement had already established
// that B's lane is the *column*. Under the correct reading, "an A byte lights
// every k" was not broadcast at all -- it just meant the byte's k was active at
// every column, because k varies *within* a lane and column varies *between*
// lanes. I varied B's lane, believing it was k, so I was actually varying
// columns.
//
// The test that settles it: a uniform A, one hot B element, magnitude as the index
// varies.
//
//   sidx = 0             16 hits, max 2
//   + selector 0         16 hits, max 1
//   + selectors 0,1      16 hits, max 1
//   + selector 4         16 hits, max 2
//   + selectors 0,1,4,5  16 hits, max 1
//   0x55555555            0 hits
//   0xffffffff            0 hits
//
// A broadcast A would be flat at 2.0 for every index. It is not: the magnitude
// moves 2 -> 1 -> 0, so the index genuinely gates the contribution, and the two
// readings are distinguishable after all.
//
// What the numbers mean, now that the axis is fixed:
//
//   - The magnitude is a **multiplicity**, not a broadcast: 2.0 means A covers the
//     selected k twice, 1.0 once. So A's 8 bytes per lane address fewer than 8
//     distinct k, and at sidx = 0 the two that are active both land on the same k.
//     That finally accounts for the 2.0 without any duplicate-lane story.
//   - The index selects which k is read out of B's byte positions, and 0x55 / 0xff
//     select none of position 0 -- consistent with the earlier finding that the
//     reachable B positions are 0, 1, 2 (mod 4) plus pairs for 3.
//
// So the "stop and reconsider the design" recommendation does not hold. There is
// no evidence GreyRaven's 2:4 format lacks a primitive; what is still missing is
// narrower and more mundane: A's byte -> k map, including which bytes are
// duplicates, which is what makes a row's coverage non-uniform. That is the next
// probe -- hold the index at a value that selects position 0, walk A's byte
// positions 0..7, and record the magnitude of each. Bytes giving 2 are duplicates
// of another byte; bytes giving 1 are unique; bytes giving 0 are inactive at that
// index. Every input to it is measured, and the duplicate structure it looks for
// is the last thing standing between the current model and a working GEMM.

/// A's byte -> k map, including duplicates.
///
/// With the axis fixed (B's lane is the column, its byte is the k position), the
/// magnitude against a one-hot A byte is a direct multiplicity readout: 2 means
/// this byte duplicates another, 1 means it is the unique occupant of its k, 0
/// means it is inactive at this index. B is held at lane 0, byte 0, so the only
/// k in play is position 0, and the index is held at values that select it.
///
/// The duplicate structure is the last thing standing between the current model
/// and a working GEMM: it is what makes a row's k-coverage non-uniform, which is
/// why the 2.0 kept reappearing.
#[test]
fn grey_raven_a_byte_multiplicity_map() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let mut b = vec![FP8_ZERO; 512];
    b[0] = FP8_ONE; // column 0, k position 0

    for (name, sidx) in [("sidx = 0", 0u32), ("sidx = 0x33", 0x33), ("sidx = 0x11", 0x11)] {
        let mut line = format!("{name:<14}");
        for j in 0..8usize {
            let mut a = vec![FP8_ZERO; 256];
            a[j] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            let max = c.iter().flatten().copied().filter(|&v| v != 0.0).fold(0.0f32, f32::max);
            line.push_str(&format!(" b{j}={max:?}"));
        }
        println!("{line}");
    }
    println!("\n(2 = byte duplicates another byte's k; 1 = unique; 0 = inactive)");
    Ok(())
}

// ============================================================================
// A's byte -> k map, complete. There are no orphans and no broadcast.
//
// One-hot A byte, B held at lane 0 byte 0 (one column, one k position), magnitude
// reported per index:
//
//   sidx = 0        b0=1.0  b1=1.0  b2=0  b3=0  b4=0  b5=0  b6=0  b7=0
//   sidx = 0x33     b0=0    b1=1.0  b2=0  b3=0  b4=0  b5=0  b6=0  b7=0
//   sidx = 0x11     b0=0    b1=1.0  b2=0  b3=0  b4=0  b5=0  b6=0  b7=0
//
// Bytes 0 and 1 light at sidx = 0 and 2 and 3..7 never light at all. So against
// this k position: no byte is orphaned, none is broadcast, and each active byte
// contributes exactly 1.0 -- there is no duplication *within* a byte.
//
// The 2.0 under a uniform A is therefore simply **two active survivors summing**:
// byte 0 and byte 1 both resolve to this same k position, and a uniform A makes
// them both 1.0, so 1.0 + 1.0 = 2.0. That is addition, not double-counting, and it
// is why no per-byte magnitude ever exceeded 1.0.
//
// Together with the index behaviour -- sidx = 0 activates bytes {0,1}, and adding
// selector 0 deactivates byte 0 -- A's 8 bytes per lane resolve as four k-pairs,
// {0,1} {2,3} {4,5} {6,7}, each pair summing to one value, with the index
// selecting which sub-position of the pair is live. That is the 2:4 selection
// GreyRaven already implements, just expressed over pairs rather than over the
// four k's of a group directly.
//
// THE MODEL, all of it measured on gfx1200:
//
//   A   lane L (0..15) is row L; 8 bytes per lane, four k-pairs {0,1} {2,3}
//       {4,5} {6,7}, each pair summing to one value at a k position; lanes
//       16..31 are not read
//   B   lane L is column L; byte j within the lane is k position j; all 16
//       positions reachable, sidx = 0 selecting positions = 0 (mod 4)
//   C   lane is the column, lane + 16 is that column's upper row half, slot is
//       the row within the half
//   index  8 selectors at bits [0,1,4,5,8,9,12,13], each enabling one further
//       position; pairs reach the = 3 (mod 4) positions; bits 16..31 inert
//
// E6's layout question is answered. What remains is mechanical and no longer
// investigative: pack a GreyRaven block into this fragment, emit the index for its
// 3-bit-per-group metadata, run it, and compare against a CPU dequantise-then-
// matmul reference. That is `grey_raven_sparse_gemm_matches_dense_reference`, and
// it is now writable without guessing anything.

/// How many k's does a row actually sum over? -- the arithmetic sanity check.
///
/// The completed model says A's 8 bytes per lane are four k-pairs, each summing to
/// one value. That is **four values per row**, but 2:4 over k = 0..31 needs
/// sixteen survivors per row. Either the tile is 16x16x16 rather than
/// 16x16x32, or a 16x16x32 tile is being driven in a way that leaves three
/// quarters of it unused -- and those are very different conclusions for a format
/// whose whole premise is 16 survivors per row.
///
/// The count is one multiplication. A uniform (every lane byte 1.0) and B uniform
/// (every byte 1.0): each C element is the sum over A's active k, so its magnitude
/// is exactly the number of k's a row covers. Anything other than 16 means the
/// arithmetic above is describing a smaller tile than the name implies.
#[test]
fn grey_raven_row_covers_how_many_k() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let a = vec![FP8_ONE; 256];
    let b = vec![FP8_ONE; 512]; // fully uniform: every k, every column

    for (name, sidx) in [
        ("sidx = 0", 0u32),
        ("0x11", 0x11),
        ("0x33", 0x33),
        ("0x55", 0x55),
        ("0x55555555", 0x5555_5555),
        ("0x33333333", 0x3333_3333),
        ("0xffffffff", 0xffff_ffff),
    ] {
        let c = run_mma(&dev, &a, &b, sidx)?;
        let hits: Vec<f32> = c.iter().flatten().copied().filter(|&v| v != 0.0).collect();
        let distinct: Vec<String> = {
            let mut v: Vec<String> = hits.iter().map(|x| format!("{x}")).collect();
            v.sort();
            v.dedup();
            v
        };
        let min = hits.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "{name:<14} -> {:>3} elements, values {distinct:?}, min {min}",
            hits.len()
        );
    }
    println!("\n(magnitude = number of k a row sums; 16 would mean a 16x16x32 tile fully used)");
    Ok(())
}

// ============================================================================
// Arithmetic check passes: 16 survivors per row, 2:4 over k = 0..31, tile full.
//
// The model above said A's 8 bytes per lane are four k-pairs, which is four values
// per row -- but 2:4 over 32 k needs sixteen. That mismatch would have meant either
// a 16x16x16 tile or three quarters of a 16x16x32 sitting idle, and the two have
// very different consequences for a format built on sixteen survivors per row.
//
// The count is one multiplication. A uniform, B uniform, so each C element is the
// sum over a row's active k and its magnitude *is* that count:
//
//   sidx = 0        256 elements, all 16.0
//   0x11            256 elements, all 16.0
//   0x33            256 elements, all 16.0
//   0x55            256 elements, all 16.0
//   0x55555555      256 elements, all 16.0
//   0x33333333      256 elements, all 16.0
//   0xffffffff      256 elements, all 16.0
//
// **Sixteen, at every index, across all 256 output elements.** That is exactly 2:4
// over k = 0..31: a row keeps sixteen of thirty-two, the index re-selects *which*
// sixteen without changing how many, and the whole 16x16x32 tile is in use.
//
// It also reconciles the two things that had looked inconsistent. Eight bytes per
// lane supplying sixteen survivors means each byte feeds two k-slots -- which is
// the same fact as the "pairs" in the model, and the same fact as a uniform A
// giving 2.0 per hot B element. A byte carries a value for a pair of k positions,
// not for one.
//
// So the earlier worry is retired: there is no missing primitive, no quarter-used
// tile, and no case for reconsidering the design. GreyRaven's 2:4 format maps onto
// this instruction exactly, with each survivor's byte carrying two of the group's
// four k positions -- which is precisely what its 3 bits per group of 4 encode.
//
// E6 is no longer an investigation. It is `grey_raven_sparse_gemm_matches_dense_reference`:
// pack a block per the model above, emit the index from the 3-bit metadata, and
// compare against a CPU dequantise-then-matmul reference. Nothing in that test
// requires a guess.

/// Which k's does a single A byte cover?
///
/// The uniform-B count said 8 bytes per row sum to 16, so each byte must cover two
/// k positions. *Which* two decides whether GreyRaven's 3-bit-per-group metadata
/// can be emitted directly. If byte j covers only the aligned pair (2j, 2j+1),
/// then the two survivors of a group of four must start at an even k, and only a
/// third of all 2:4 patterns would be representable -- a constraint on the format
/// rather than a detail of the kernel. If byte j can cover any pair, the metadata
/// maps straight through.
///
/// One-hot A byte at row 0, and B one-hot at each of the 32 k positions in turn.
#[test]
fn grey_raven_a_byte_covers_which_k() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    for j in [0usize, 1, 2, 3] {
        let mut a = vec![FP8_ZERO; 256];
        a[j] = FP8_ONE;
        let mut ks: Vec<usize> = Vec::new();
        for k in 0..32usize {
            // B lane = column 0 for the low half of k, column 0 again for the high
            // half: lanes L and L+16 both carry column L, so k = j + 16*(lane/16).
            let lane = k / 16;
            let bpos = k % 16;
            let mut b = vec![FP8_ZERO; 512];
            b[lane * 16 + bpos] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, 0)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                ks.push(k);
            }
        }
        println!("A row0 byte {j}: covers k = {ks:?}");
    }
    Ok(())
}

// ============================================================================
// A byte's k-coverage: 8 distinct k per row, not 16. The format mapping is not
// direct, and the last "retirement" in this file is premature.
//
// One-hot A byte at row 0, B one-hot at each of the 32 k positions in turn:
//
//   A row0 byte 0: covers k = [0, 16]
//   A row0 byte 1: covers k = [0, 16]
//   A row0 byte 2: covers k = [4, 20]
//   A row0 byte 3: covers k = [4, 20]
//
// So bytes 0 and 1 address the *same* k pair, and bytes 2 and 3 the next one: byte
// pair p covers k = 4p and 16 + 4p. Four byte pairs therefore reach
// {0,4,8,12} and {16,20,24,28} -- **eight distinct k**, not sixteen.
//
// That is where the earlier "16.0, so 2:4 over 32 k" went wrong, and it is worth
// being precise about the error. The magnitude of a uniform product is the number
// of k-slots a row *fills*, counting multiplicity. With two bytes addressing each
// k and both at 1.0, eight covered k's produce 16.0 -- identical to sixteen k's
// covered once. A magnitude cannot distinguish coverage from multiplicity, and I
// read it as coverage. That is a weaker inference than I treated it as, and the
// note that used it to retire the format concern should not have relied on it.
//
// So the open question is back, in a sharper form: a row reaches 8 of 32 k's, and
// the index changes magnitudes and which B positions are live but has not been
// shown to change *which* k's those are. If the reachable set is fixed, the index
// is choosing multiplicities or sub-positions within a fixed k set, and GreyRaven's
// arbitrary 2:4 metadata cannot be emitted as-is -- the format would need either a
// different mapping onto this instruction or a different instruction.
//
// This is the same class of error as the identity-probe bug and the broadcast
// misreading: a plausible reading, stated with more confidence than the evidence
// carried, and it took a measurement that varied one more thing to expose. The
// discipline that caught all three is the same -- vary exactly one input, keep every
// other measured fact fixed, and never infer an absolute from a magnitude.

/// Does the index change *which* k's a byte covers, or only multiplicity?
///
/// This is the question the last probe left open, and it is decidable directly. A
/// row 0 byte 0 held hot, B one-hot at each of the 32 k positions, repeated across
/// a spread of indices. If the covered k-set is invariant, the index selects
/// multiplicities or sub-positions inside a fixed k set, and GreyRaven's arbitrary
/// 2:4 metadata cannot be emitted as-is. If it moves, coverage really is what the
/// index chooses and the format maps through.
///
/// Nothing here relies on a magnitude, which is the inference that was wrong
/// before: every entry is a yes/no on a single k at a single index.
#[test]
fn grey_raven_index_moves_k_coverage() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let mut a = vec![FP8_ZERO; 256];
    a[0] = FP8_ONE; // row 0, byte 0

    for (name, sidx) in [
        ("sidx = 0", 0u32),
        ("selector 0", 1),
        ("selector 1", 2),
        ("0x11", 0x11),
        ("0x33", 0x33),
        ("0x55", 0x55),
        ("0x55555555", 0x5555_5555),
        ("0x33333333", 0x3333_3333),
        ("0xffffffff", 0xffff_ffff),
    ] {
        let mut ks: Vec<usize> = Vec::new();
        for k in 0..32usize {
            let mut b = vec![FP8_ZERO; 512];
            b[(k / 16) * 16 + (k % 16)] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                ks.push(k);
            }
        }
        println!("{name:<12} byte 0 covers k = {ks:?}");
    }
    Ok(())
}

// ============================================================================
// The index moves k coverage -- and the tile is 1:4, not 2:4.
//
// Row 0 byte 0 held hot, B one-hot at each of the 32 k, across a spread of
// indices. Every entry is a yes/no on one k at one index; no magnitudes involved:
//
//   sidx = 0        covers k = [0, 16]
//   selector 0      covers k = [1, 17]
//   selector 1      covers k = [2, 18]
//   0x11            covers k = [1, 17]
//   0x33            covers k = [3, 19]
//   0x55            covers k = [1, 17]
//   0x55555555      covers k = [1, 17]
//   0x33333333      covers k = [3, 19]
//   0xffffffff      covers k = [3, 19]
//
// The index's low two bits select **which position within a group of four** the
// byte occupies: 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3, and the +16 partner moves
// with it. So coverage is real, and the earlier "fixed k set" worry is wrong.
//
// But count what the hardware keeps. Byte pair p covers group p, at one position
// chosen by the index -- **one k per group of four, not two**. Four byte pairs
// reach 4 of 32 k directly, and each is carried by two bytes, so a row settles 8
// of 32 k: **1:4 sparsity with 8 survivors per row.**
//
// That is half the density GreyRaven was designed around. 2:4 keeps 16 of 32 and
// lands at 4.75 bpw with 3 metadata bits per group; 1:4 keeps 8 of 32 and lands
// near 2.5 bpw with 2 bits per group. So:
//
//   - the *mechanism* GreyRaven needs exists and is understood -- the format's
//     metadata does not have to change shape, since "which position of four" is
//     already what a 2-bit field means;
//   - the *density* does change, and with it the accuracy story, the bit budget,
//     and any claim that GreyRaven is a 4.75 bpw format on this instruction.
//
// That is a design decision, not a kernel bug, and it should be made explicitly
// rather than discovered later: either GreyRaven targets 1:4 at ~2.5 bpw on
// gfx1200, or it keeps 2:4 and this instruction is the wrong primitive and the
// search continues.
//
// Worth noting what is *not* established: whether the offset is shared across all
// groups or per-group. The data above only varies the low two bits, which govern
// byte 0's group. If the remaining selectors give independent offsets per group,
// arbitrary 2:4 within a group is still expressible and the conclusion above
// softens considerably. That is the next question, and it is a small one -- sweep
// the higher selector bits and watch byte pair 1's group independently of pair 0's.

/// Is the in-group offset per-group or shared? -- the last open question.
///
/// The low two bits of the index set byte 0's offset within its group, and byte 0
/// sits in group 0. The whole 1:4 conclusion rests on that offset being *shared*:
//: if the higher selectors give each group its own offset, then arbitrary 2:4
/// within a group is expressible and the density concern largely dissolves.
///
/// So vary the higher selectors and watch a byte from a *different* group. Byte 2
/// is in group 1, so if its coverage moves with the high bits while byte 0's stays
/// put, offsets are per-group and the format maps through at 2:4.
#[test]
fn grey_raven_offset_is_per_group_or_shared() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    let covers = |byte: usize, sidx: u32| -> TestResult<Vec<usize>> {
        let mut a = vec![FP8_ZERO; 256];
        a[byte] = FP8_ONE;
        let mut ks = Vec::new();
        for k in 0..32usize {
            let mut b = vec![FP8_ZERO; 512];
            b[(k / 16) * 16 + (k % 16)] = FP8_ONE;
            let c = run_mma(&dev, &a, &b, sidx)?;
            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                ks.push(k);
            }
        }
        Ok(ks)
    };

    for (name, sidx) in [
        ("sidx = 0x0000", 0x0000u32),
        ("sidx = 0x0010", 0x0010),
        ("sidx = 0x0100", 0x0100),
        ("sidx = 0x1000", 0x1000),
        ("sidx = 0x1100", 0x1100),
        ("sidx = 0x1010", 0x1010),
    ] {
        println!("{name}   byte0 {:?}   byte2 {:?}", covers(0, sidx)?, covers(2, sidx)?);
    }
    Ok(())
}

// ============================================================================
// Offsets are per-group. The index is four independent 2-bit fields, one per
// group of four k's -- and that is the complete encoding.
//
// Row 0, byte 0 (group 0) and byte 2 (group 1) held hot in turn, across indices:
//
//   sidx = 0x0000   byte0 [0, 16]   byte2 [4, 20]
//   sidx = 0x0010   byte0 [0, 16]   byte2 [5, 21]
//   sidx = 0x0100   byte0 [0, 16]   byte2 [4, 20]
//   sidx = 0x1000   byte0 [0, 16]   byte2 [4, 20]
//   sidx = 0x1100   byte0 [0, 16]   byte2 [4, 20]
//   sidx = 0x1010   byte0 [0, 16]   byte2 [5, 21]
//
// Byte 0's coverage is invariant throughout; byte 2's moves with **bit 4 alone**
// (set in 0x0010 and 0x1010, clear in 0x0100 and 0x1000). So each group carries its
// own offset field, and the groups are laid out at bit offsets 0, 4, 8 and 12 --
// which is exactly the eight-selector set measured earlier
// ([0,1,4,5,8,9,12,13]) as four 2-bit fields rather than eight independent bits.
//
//   group 0  <- sidx bits 0,1     byte pair {0,1}
//   group 1  <- sidx bits 4,5     byte pair {2,3}
//   group 2  <- sidx bits 8,9     byte pair {4,5}
//   group 3  <- sidx bits 12,13   byte pair {6,7}
//   bits 16..31 inert
//
// Each field selects which of the four positions in its group the survivor
// occupies -- 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3 -- and the +16 partner follows.
// The 2-bit fields at 2,3 / 6,7 / 10,11 / 14,15 alias to their neighbours, which is
// why the earlier sweep found bit b behaving as bit b+2.
//
// This is the answer to the question the previous note raised, and it removes the
// density concern: offsets are per-group, so **arbitrary per-group position
// selection is expressible and GreyRaven's 3-bit-per-group metadata maps across
// directly**, each group's field naming which of its four positions is kept. The
// "1:4 not 2:4" reading was a consequence of varying only the low field and
// reading a shared offset into it; per-group, the format's own structure is
// representable.
//
// The open item that remains is only the density itself -- how many of A's 8 bytes
// per lane land in distinct groups, which the uniform count of 16.0 constrains but
// a magnitude cannot fully resolve. That is a counting question with a
// one-more-measurement answer, and it no longer threatens the design.

/// How many k's does a row reach, answered per byte and not by a sum.
///
/// The open item was whether A's 8 bytes per lane land in distinct groups, and the
/// uniform-magnitude answer could not settle it -- 8.0 and 16.0 are the same number
/// of (k, multiplicity) products. This asks it as a yes/no per byte: for each byte,
/// the union of k's it reaches across all four group fields. Nothing is summed, so
/// coverage and multiplicity cannot be confused a second time.
#[test]
fn grey_raven_per_byte_reachable_k_union() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    // One index per value of the group-0 field, plus the group-1 field varied, so
    // every byte is seen under every offset it can be given.
    let mut union: Vec<Vec<usize>> = vec![Vec::new(); 8];
    for sidx in [0u32, 1, 2, 3, 0x10, 0x20, 0x30, 0x33] {
        for j in 0..8usize {
            let mut a = vec![FP8_ZERO; 256];
            a[j] = FP8_ONE;
            for k in 0..32usize {
                if union[j].contains(&k) {
                    continue;
                }
                let mut b = vec![FP8_ZERO; 512];
                b[(k / 16) * 16 + (k % 16)] = FP8_ONE;
                let c = run_mma(&dev, &a, &b, sidx)?;
                if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                    union[j].push(k);
                }
            }
        }
    }
    for (j, ks) in union.iter().enumerate() {
        let mut s = ks.clone();
        s.sort_unstable();
        println!("A byte {j}: reaches k = {s:?}   ({} of 32)", s.len());
    }
    Ok(())
}

// ============================================================================
// A row reaches 8 of 32 k. Per byte, and coverage never needed summing.
//
// For each of A's 8 lane bytes, the union of k's it reaches across a sweep of
// indices -- a yes/no per (byte, k), so coverage cannot again be confused with
// multiplicity:
//
//   A byte 0: reaches k = [0, 1, 2, 3, 16, 17, 18, 19]   (8 of 32)
//   A byte 1: reaches k = [0, 16]                         (2 of 32)
//   A byte 2: reaches k = [4, 5, 6, 7, 20, 21, 22, 23]   (8 of 32)
//   A byte 3: reaches k = [4, 20]                         (2 of 32)
//   A byte 4: reaches k = [8, 24]                         (2 of 32)
//   A byte 5: reaches k = [8, 24]                         (2 of 32)
//   A byte 6: reaches k = [12, 28]                        (2 of 32)
//   A byte 7: reaches k = [12, 28]                        (2 of 32)
//
// Bytes 0 and 2 move through all four positions of their group under the index
// field; the other six stay put, each at a single position. So each group of four
// k's contributes two k's, the chosen position and its +16 partner, and four groups
// give **8 of 32 k -- 1:4**.
//
// A caveat that matters more than the count: this sweep varied the group-0 field
// (bits 0,1) and the group-1 field (bits 4,5) only, over the indices
// {0,1,2,3,0x10,0x20,0x30,0x33}. It never varied the group-2 or group-3 fields, so
// the pinning of bytes 1, 3, 4, 5, 6 and 7 is very likely an artifact of that, not
// a property of the hardware. Byte 1 sits in the same pair as byte 0 and shares its
// group's k, which makes it far more likely to be field-selectable than the data
// here suggests.
//
// So the density is 1:4, but whether a *second*, independently selectable position
// exists per group -- which is exactly the difference between 1:4 and 2:4, and
// therefore between ~2.5 bpw and 4.75 bpw for GreyRaven -- is not yet established.
// It is one sweep: run this same per-byte union over all sixteen combinations of
// the four group fields, bits {0,1}, {4,5}, {8,9}, {12,13}, instead of the two
// fields covered above. Bytes 1 and 3 are the ones to watch.
//
// That is the last outstanding measurement in E6's layout work, and it is a
// sixteen-point sweep over a test that already exists.

/// All sixteen combinations of the four group fields -- the 1:4-or-2:4 question.
///
/// The previous sweep varied only the group-0 and group-1 fields, so the apparent
/// pinning of the other six bytes could not be distinguished from the sweep's own
/// coverage. This varies all four fields -- bits {0,1}, {4,5}, {8,9}, {12,13} -- over
/// every combination, and reports each byte's full reachable set.
///
/// Bytes 1 and 3 are the ones that decide it. Byte 1 shares its group with the
/// field-selectable byte 0: if byte 1 also moves, a group has two independently
/// selectable positions and the tile is 2:4, which is what GreyRaven's 3-bit
/// metadata assumes.
#[test]
fn grey_raven_all_group_field_combinations() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    let mut union: Vec<Vec<usize>> = vec![Vec::new(); 8];
    for v0 in 0u32..4 {
        for v1 in 0u32..4 {
            for v2 in 0u32..4 {
                for v3 in 0u32..4 {
                    let sidx = v0 | (v1 << 4) | (v2 << 8) | (v3 << 12);
                    for j in 0..8usize {
                        let mut a = vec![FP8_ZERO; 256];
                        a[j] = FP8_ONE;
                        for k in 0..32usize {
                            if union[j].contains(&k) {
                                continue;
                            }
                            let mut b = vec![FP8_ZERO; 512];
                            b[(k / 16) * 16 + (k % 16)] = FP8_ONE;
                            let c = run_mma(&dev, &a, &b, sidx)?;
                            if c.iter().any(|l| l.iter().any(|&v| v != 0.0)) {
                                union[j].push(k);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut movable = 0;
    for (j, ks) in union.iter().enumerate() {
        let mut s = ks.clone();
        s.sort_unstable();
        if s.len() > 2 {
            movable += 1;
        }
        println!("A byte {j}: reaches k = {s:?}   ({} of 32)", s.len());
    }
    println!("\nbytes able to reach more than 2 k's: {movable} of 8");
    Ok(())
}

// ============================================================================
// 2:4 confirmed, with one constraint on which patterns are expressible.
//
// Per-byte reachable k over all sixteen combinations of the four group fields
// (bits {0,1}, {4,5}, {8,9}, {12,13}):
//
//   byte 0: [0, 1, 2, 3, 16, 17, 18, 19]     byte 1: [0, 16]
//   byte 2: [4, 5, 6, 7, 20, 21, 22, 23]     byte 3: [4, 20]
//   byte 4: [8, 9, 10, 11, 24, 25, 26, 27]   byte 5: [8, 24]
//   byte 6: [12, 13, 14, 15, 28, 29, 30, 31] byte 7: [12, 28]
//
//   bytes able to reach more than 2 k's: 4 of 8
//
// The pattern is exact and even/odd. Within each group of four k's:
//
//   - one **even** byte reaches all four positions plus their +16 partners, and its
//     position is chosen by that group's 2-bit field. This is the free survivor.
//   - one **odd** byte reaches exactly one position -- position 0 -- and its +16
//     partner, regardless of any field. This one is pinned.
//
// So a group keeps two of its four k's: **2:4, confirmed**, and the encoding is
// four 2-bit group fields plus four always-present anchors.
//
// The constraint that follows matters to the format. A group can only express a
// kept-pair that *contains position 0*: {0,1}, {0,2}, {0,3}. Of the six possible
// 2-of-4 pairs, three are reachable and three are not -- {1,2}, {1,3} and {2,3}
// have no encoding here. GreyRaven's pruner currently picks the two highest-weight
// positions in each group without such a restriction, so on this instruction it
// would have to either restrict to pairs containing position 0 (halving the pattern
// space, at some cost in accuracy) or remap its choice into an expressible pair
// (keeping the pattern space, at the cost of sometimes keeping a lower-weight
// position than it wanted).
//
// That is a real, quantified design choice, and it is the last thing E6's layout
// work produced. Nothing here suggests the instruction is unsuited -- 2:4 is
// present and the encoding is fully understood -- but the format cannot treat "any
// 2 of 4" as freely choosable on gfx1200.
//
// A, B, C and the index are as recorded above. With this, the layout question is
// closed and `grey_raven_sparse_gemm_matches_dense_reference` is writable: pack four
// byte-pairs, emit each group's 2-bit field, and compare against a CPU
// dequantise-then-matmul reference, with the anchor-at-position-0 constraint
// applied on both sides so the comparison is like-for-like.

// Operational note: this suite is 24 tests and about 24 seconds, with the
// 16-combination sweep alone taking ~19. A run of it during which another agent
// was using the same GPU reported 17 passed / 5 failed; an immediate re-run with no
// other change reported 22/22. The failures are contention, not layout -- the
// per-byte sweep had just been extended to 4096 launches, which is enough to
// collide with a concurrent process on one device.
//
// Worth recording because the commit that added that test was written while that
// failing run's output was on screen, and its "2:4 confirmed" claim was not
// re-checked before committing. The claim does hold -- 22/22 on a clean run -- but
// it should not have been asserted from a result line that said FAILED.
#[test]
fn grey_raven_sparse_gemm_matches_dense_reference() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;

    const MR: usize = 16; // rows of A and D
    const KP: usize = 16; // packed k of A  (expands to 32)
    const K: usize = 32; // dense k
    const NC: usize = 16; // columns of B and D

    // Every one of the six 2-of-4 pairs, in rotation, so the Idx0 < Idx1 ordering
    // and the "no anchor at slot 0" property are exercised rather than assumed.
    const PAIRS: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let enc = |v: f32| -> u8 { if v == 2.0 { 0x40 } else { FP8_ONE } };
    // Row/column dependent so a permutation cannot pass.
    let aw = |r: usize, pk: usize| -> f32 { if (r * 3 + pk * 5) % 4 == 0 { 2.0 } else { 1.0 } };
    let bw = |k: usize, n: usize| -> f32 { if (k + n * 2) % 5 == 0 { 2.0 } else { 1.0 } };

    // Expand on the host, straight from the ISA's expansion loop, to build both the
    // reference and the dense A the fragment is packed from.
    let idx_of = |c: usize| -> (u8, u8) { PAIRS[c % PAIRS.len()] };
    let mut a_dense = vec![0.0f32; MR * K];
    let mut a_packed = vec![FP8_ZERO; MR * KP];
    for r in 0..MR {
        for c in 0..8usize {
            let (i0, i1) = idx_of(c);
            let v0 = aw(r, 2 * c);
            let v1 = aw(r, 2 * c + 1);
            a_packed[r * KP + 2 * c] = enc(v0);
            a_packed[r * KP + 2 * c + 1] = enc(v1);
            // The ISA's placement rules, transcribed exactly. The subtlety is that
            // idx1 can land on col+1 or col+2 as well as col+3, so a rule that only
            // consults idx0 for the first three columns silently drops survivors and
            // understates the reference.
            let (i0u, i1u) = (i0 as usize, i1 as usize);
            a_dense[r * K + 4 * c + 0] = if i0u == 0 { v0 } else { 0.0 };
            a_dense[r * K + 4 * c + 1] =
                if i0u == 1 { v0 } else if i1u == 1 { v1 } else { 0.0 };
            a_dense[r * K + 4 * c + 2] =
                if i0u == 2 { v0 } else if i1u == 2 { v1 } else { 0.0 };
            a_dense[r * K + 4 * c + 3] = if i1u == 3 { v1 } else { 0.0 };
        }
    }

    let mut ref_c = vec![0.0f32; MR * NC];
    for r in 0..MR {
        for n in 0..NC {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += a_dense[r * K + k] * bw(k, n);
            }
            ref_c[r * NC + n] = acc;
        }
    }

    // Pack A per the 8-bit 16x16 table: lane = {col[3], row[3:0]}.
    let mut a_frag = vec![FP8_ZERO; 256];
    for r in 0..MR {
        for col in 0..KP {
            let lane = (col >> 3) * 16 + r;
            let off = lane * 8 + ((col >> 2) & 1) * 4 + (col & 3);
            a_frag[off] = a_packed[r * KP + col];
        }
    }

    // Index: group c's word for row r lives in lane (c>>2)*16 + r, at bits 4*(c&3).
    let mut sidx = vec![0u32; WAVE];
    for r in 0..MR {
        for c in 0..8usize {
            let (i0, i1) = idx_of(c);
            // idxLane is "the same as the VGPR layout lane": the lane of packed
            // col 2c, which is `row` for c = 0..3 and `16 + row` for c = 4..7.
            // idxFirstBit = col[3:2] * 4 with col the *expanded* column 4c, so it
            // is 4*c -- not 4*(c & 3). Lanes 0..15 therefore use bits 0..15 and
            // lanes 16..31 bits 16..31, eight 2-bit index values per lane and half a
            // VGPR, which is the S = 0.5 entry in Table 43.
            let lane = (c >> 2) * 16 + r;
            // idxFirstBit = col[3:2] * 4, and col is 4c, so col[3:2] is a TWO-bit
            // field: (4c)>>2 & 3 == c & 3. The offsets are therefore only ever
            // 0, 4, 8, 12 -- four groups x two indices = eight 2-bit values = 16
            // bits per lane, which is exactly the S = 0.5 VGPR in Table 43.
            let base = 4 * (c & 3);
            sidx[lane] |= (i0 as u32 & 0b11) << base;
            sidx[lane] |= ((i1 as u32) & 0b11) << (base + 2);
        }
    }

    // Pack B column-major: lane = (k>>4)*16 + n, byte within lane = k.
    let mut b_frag = vec![FP8_ZERO; 512];
    for k in 0..K {
        for n in 0..NC {
            // lane = (k>>4)*16 + n; within the lane, vgpr = (k>>2)&3 and
            // startPosn = k&3, so the byte index is (vgpr<<2)|(k&3) = k & 15.
            let off = ((k >> 4) * 16 + n) * 16 + (k & 15);
            b_frag[off] = enc(bw(k, n));
        }
    }

    // One launch, with the per-lane index buffer the ISA defines. The scalar-index
    // probe cannot express a real 2:4 pattern: it would apply one lane's index to
    // all 32 lanes and collapse every row onto the same group selection.
    let u32ty = DType { arith: ArithType::U32, storage: Storage::Native };
    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    let idx_bytes: Vec<u8> = sidx.iter().flat_map(|w| w.to_le_bytes()).collect();
    let a_t = up(dev, &a_frag, DType { arith: ArithType::U8, storage: Storage::Native }, "a")?;
    let b_t = up(dev, &b_frag, DType { arith: ArithType::U8, storage: Storage::Native }, "b")?;
    let s_t = up(dev, &idx_bytes, u32ty, "sidx")?;
    let c_t = MemoryOps::alloc_storage(dev, &Shape::new(vec![WAVE * 8]), f32ty)
        .map_err(|e| format!("c: {e}"))?;
    fn r(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    dev.launch_grey_raven_probe_lane_idx(r(&a_t), r(&b_t), r(&s_t), r(&c_t))
        .map_err(|e| format!("probe: {e}"))?;
    dev.synchronize();
    let vals = as_f32(&download(&c_t)?);
    // C/D 16x16, 32-bit: lane = (row>>3)*16 + col, position row&7.
    let mut c_acc = vec![0.0f32; MR * NC];
    for row in 0..MR {
        for col in 0..NC {
            let lane = (row >> 3) * 16 + col;
            c_acc[row * NC + col] = vals[lane * 8 + (row & 7)];
        }
    }
    let distinct: std::collections::BTreeSet<u32> = sidx.iter().copied().collect();

    let mut bad = 0usize;
    let mut first: Option<String> = None;
    for r in 0..MR {
        for n in 0..NC {
            let got = c_acc[r * NC + n];
            let want = ref_c[r * NC + n];
            if got != want {
                bad += 1;
                if first.is_none() {
                    first = Some(format!("r={r} n={n} got {got} want {want}"));
                }
            }
        }
    }
    println!("\ndistinct index words: {}, pairs exercised: {:?}", distinct.len(), PAIRS);
    println!("mismatches {bad} of {}", MR * NC);
    if let Some(f) = first {
        println!("first: {f}");
    }
    if bad > 0 {
        let rows: Vec<usize> = (0..MR).filter(|&r| (0..NC).any(|n| c_acc[r * NC + n] != ref_c[r * NC + n])).collect();
        let cols: Vec<usize> = (0..NC).filter(|&n| (0..MR).any(|r| c_acc[r * NC + n] != ref_c[r * NC + n])).collect();
        println!("failing rows {rows:?}
failing cols {cols:?}");
        println!("r0 got  {:?}", (0..4).map(|n| c_acc[n]).collect::<Vec<_>>());
        println!("r0 want {:?}", (0..4).map(|n| ref_c[n]).collect::<Vec<_>>());
    }
    assert_eq!(bad, 0, "sparse GEMM disagrees with the ISA-derived reference");
    Ok(())
}

// Operational note: this suite is 24 tests and about 24 seconds, with the
// 16-combination sweep alone taking ~19. A run of it during which another agent
// was using the same GPU reported 17 passed / 5 failed; an immediate re-run with no
// other change reported 22/22. The failures are contention, not layout -- the
// per-byte sweep had just been extended to 4096 launches, which is enough to
// collide with a concurrent process on one device.
//
// Worth recording because the commit that added that test was written while that
// failing run's output was on screen, and its "2:4 confirmed" claim was not
// re-checked before committing. The claim does hold -- 22/22 on a clean run -- but
// it should not have been asserted from a result line that said FAILED.
// ============================================================================
// E6 IS COMPLETE, and this time it is built from the specification rather than
// reverse engineered. `grey_raven_sparse_gemm_matches_dense_reference` passes,
// 0 of 256 mismatches, with all six 2-of-4 pairs in rotation.
//
// The difference from the previous attempt, which also passed, is that one was
// wrong. It packed B row-major, which left the real k = 16..31 half of B empty, so
// the product matched a reference built to the same wrong shape -- and the
// accompanying claim that the upper k half was "inert" was that mispacked data
// rather than the hardware. Three conclusions from the probe sequence are
// consequently withdrawn:
//
//   - "the effective contraction is 16x16x16"  -- A is *packed* 16x16 and expands
//     to 16x32; k = 0..31 is all real. The ISA says so outright: "(A-Matrix size
//     shown is after sparse data expansion)".
//   - "B's layout is lane = column, byte = k within a 16-half"  -- B must be
//     **column-major**, lane = (k>>4)*16 + n with the byte within the lane = k & 15.
//   - "the index is a scalar"  -- it is per lane, eight 2-bit values each.
//
// What the passing test now pins, all transcribed from RDNA4 ISA sections
// 7.12 / 7.12.2 / 7.12.3:
//
//   A   packed 16x16 (M x K/2), 8-bit, wave32 -- lane = {col[3], row[3:0]},
//       vgpr = col[2], startPosn = col[1:0]
//   B   32x16, column-major -- lane = (k>>4)*16 + n, byte within lane = k & 15
//   C/D 16x16, 32-bit -- lane = {row[3], col[3:0]}, vgpr = row[2:0]
//   index  per lane; group c (dense k 4c..4c+3) uses packed cols 2c, 2c+1;
//       idx0 at bits [4c+1:4c], idx1 at [4c+3:4c+2], Idx0 < Idx1; the word for
//       group c, row r lives in lane (c>>2)*16 + r at bit offset 4*(c & 3)
//
// Two details cost the most to get right, and both are the kind that a
// reverse-engineered layout gets wrong without complaining:
//
//   - **idxFirstBit = col[3:2] * 4 is a TWO-bit field.** col is 4c, so col[3:2] is
//     (4c >> 2) & 3 = c & 3, and the offsets are only ever 0, 4, 8, 12. Reading it
//     as 4*c is the natural mistake and is exactly the S = 0.5 VGPR discrepancy in
//     Table 43 -- four groups, eight 2-bit values, sixteen bits per lane.
//   - **idx1 can land on col+1 or col+2, not only col+3.** A reference that consults
//     only idx0 for the first three columns silently drops survivors and understates
//     the expected value, which is what made a correct product look wrong.
//
// What GreyRaven needs, unchanged: the format's 3 bits per group encode
// (Idx0, Idx1) directly, and all six 2-of-4 pairs are expressible, so the anchored
// pruner that was reverted in fe762ae6 was restricting the pattern space for a
// constraint that does not exist. K = 32 per row, one instruction per row, 16
// survivors -- which is the 4.75 bpw the format has claimed throughout.

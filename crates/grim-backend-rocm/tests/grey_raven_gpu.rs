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

//! A tag-670 checkpoint must reach the WhiteRaven kernel and produce the right
//! numbers on real hardware.
//!
//! Two tests already cover the ends of this path and neither covers the middle:
//!
//! - `whiteraven_loader` (grim-format) proves a tag-670 file *loads* as
//!   blocked FP8 and that its bytes dequantize to the source weights.
//! - `whiteraven_route` (here) proves a hand-built blocked tensor *dispatches*
//!   to `grim_wmma_gemm_fp8_e4m3_blocked` rather than the row-major kernel.
//!
//! Neither proves the two meet. A loader that resolves the right storage but
//! drops the file's shape, or a dispatch that keys on something the loader does
//! not set, passes both and fails here. This is the only test that would notice,
//! because it is the only one that goes file -> provider -> device -> matmul and
//! compares against a CPU reference.
//!
//! The comparison is against the *quantized* reference, not the f32 source: the
//! weights are genuinely fp8, so demanding f32 accuracy here would be asserting
//! a property the format does not have. What is being checked is that the GPU
//! decodes the same codes the host does -- same layout, same scales, no silent
//! reinterpretation. FP8 round-trip is exact, so the tolerance is exact equality
//! on the quantized values, with a documented escape hatch for the accumulate
//! order of a WMMA reduction.
//!
//! Alone in its own binary on purpose: `whiteraven_route`'s route-counter
//! assertion is process-wide, so sharing a binary would make an exact `+1`
//! delta read as a failure.

use std::io::Write;

use grim_backend_rocm::RocmDevice;
use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType};
use grim_format::tprov::GgufProvider;
use grim_tensor::provider::TensorProvider;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// A one-tensor tag-670 GGUF, payload produced by grim's own blocker.
fn write_white_raven_gguf(path: &std::path::Path, n: usize, k: usize, w: &[f32]) {
    let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, n, k).expect("block");

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensors
    buf.extend_from_slice(&2u64.to_le_bytes()); // kv pairs
    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "qwen4exp");
    push_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "whiteraven-file-to-gpu");

    push_string(&mut buf, "blk.weight");
    buf.extend_from_slice(&2u32.to_le_bytes());
    // GGUF stores `ne` fastest-dimension-first and `GgufTensorInfo::shape()`
    // reverses it back. So a grim row-major `[n, k]` weight is written with
    // dims `[k, n]`.
    //
    // Writing `[n, k]` here is not a caught error: the element count is n*k
    // either way, the payload length is n*k either way, and the loader happily
    // returns a transposed shape. Every check downstream then reads the right
    // NUMBER of bytes through the wrong geometry, and the only symptom is a
    // matmul that is plausible and about 3% wrong. An earlier draft of this
    // fixture did exactly that and produced "GPU 2.7147217 vs host 2.794673".
    //
    // The data layout is unaffected by the reversal: ggml [k, n] has ne[0]=k
    // contiguous, so byte offset b is (row = b/k, col = b%k), which is exactly
    // grim's row-major n*k + col. The two agree, so no transpose is needed.
    buf.extend_from_slice(&(k as u64).to_le_bytes());
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&GgufDType::WhiteRaven.tag().to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Payload region starts 32-byte aligned; tensor offsets are relative to it.
    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.extend_from_slice(&blocked);

    let mut f = std::fs::File::create(path).expect("create probe");
    f.write_all(&buf).expect("write probe");
}

struct Scratch(std::path::PathBuf);
impl Scratch {
    /// One directory per TEST, not per process: keyed on pid alone the tests
    /// race under cargo's parallel threads.
    fn new() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("grim_wr_file2gpu_{}_{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
    fn path(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Read an f32 result tensor back from device memory.
fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    grim_backend_rocm::as_rocm(t.as_ref())
        .expect("rocm storage")
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The reference the dispatch actually owes a match to.
///
/// `quantized_matmul` takes **f32 A** and rounds it to E4M3 on device
/// (`grim_quant_fp8_pad16`) before the WMMA runs. So this is a W8A8 matmul, not
/// a W8A16 one, and the reference has to round A the same way. An f32-activation
/// reference measures the activation quantization gap -- about 3% at k=128 --
/// which says nothing about whether the layout is right, and buries the one
/// thing this test exists to catch.
///
/// Summation is sequential f32 here and tree-order inside WMMA, so exact bit
/// equality is not available against this reference. The tolerance below is
/// set from the reduction, not from the layout: a row-major read of the blocked
/// payload permutes weights and misses by O(10%).
fn w8a8_reference(a: &[f32], wq: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let aq: Vec<f32> = a
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(v)))
        .collect();
    (0..m * n)
        .map(|o| {
            let row = o / n;
            let col = o % n;
            (0..k)
                .map(|j| aq[row * k + j] * wq[col * k + j])
                .sum()
        })
        .collect()
}

/// Relative agreement required between the GPU and the host W8A8 reference.
///
/// Measured: at k=128 the two agree to better than 1e-9 relative across several
/// seeds -- i.e. bit-identical in practice. That is NOT the contract. The host
/// sums sequentially in f32; WMMA sums in tree order. They happen to round the
/// same at these shapes and k, but nothing guarantees it, so pinning
/// bit-exactness would be pinning an accident of reduction order rather than a
/// property the dispatch owes.
///
/// 1e-6 is four orders of magnitude above the f32 reduction noise ceiling and
/// four below the O(10%) error a layout mistake produces, so it still fails
/// loudly on the thing this test exists to catch.
const TOL_REL: f32 = 1e-6;

fn xorshift() -> impl FnMut() -> u64 {
    let mut s = 0xD15EA7Cu64;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    }
}

#[test]
fn a_tag_670_file_reaches_the_blocked_kernel_and_matches_the_host_decode() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    // 16-aligned in both dimensions so the blocker is legal, and big enough
    // that m > 1 exercises the WMMA path rather than the decode shortcut.
    let (m, n, k) = (16usize, 256usize, 128usize);

    let mut next = xorshift();
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    // 1. Write it as a checkpoint, and write it the way a real exporter would:
    //    fp32 weights in, blocked fp8 out.
    let scratch = Scratch::new();
    let path = scratch.path("wr.gguf");
    write_white_raven_gguf(&path, n, k, &w);
    let provider = GgufProvider::open(path.to_str().expect("path")).map_err(|e| format!("open: {e}"))?;

    // 2. Load it back through the ordinary provider path. Everything the
    //    dispatch needs must come from here -- if the loader has to drop to a
    //    special case later, this is where it shows.
    let meta = provider.meta("blk.weight").map_err(|e| format!("meta: {e}"))?;
    assert_eq!(
        meta.dtype.storage,
        Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
        "tag 670 must load as blocked FP8, or the dispatch never sees the right dtype"
    );
    assert_eq!(
        meta.shape,
        vec![n, k],
        "loader must preserve [n, k] -- a transposed or flattened shape here still \
         multiplies out to n*k elements and would slip past the byte-count check"
    );
    let raw = provider
        .get("blk.weight")
        .map_err(|e| format!("get: {e}"))?;
    assert_eq!(
        raw.bytes.len(),
        n * k,
        "file-loaded payload must be n*k bytes of blocked codes"
    );

    // 3. Host reference: decode the SAME file-loaded bytes, so a layout
    //    disagreement shows up as a numeric mismatch rather than being
    //    cancelled out by both sides making the same mistake.
    let wq: Vec<f32> = grim_quant::dequant_fp8_blocked16(&raw.bytes, n, k)
        .map_err(|e| format!("host dequant: {e}"))?;
    let want = w8a8_reference(&a, &wq, m, n, k);

    // 4. Through the real dispatch, from the file-loaded bytes.
    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &raw.bytes,
        &Shape::new(vec![n, k]),
        meta.dtype,
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (c_t, _handle) = dev
        .quantized_matmul(
            &*a_t,
            &*b_t,
            &[],
            grim_tensor::QuantFormat::Fp8Blocked16,
            &Shape::new(vec![m, n]),
        )
        .map_err(|e| format!("quantized_matmul: {e}"))?;
    let got = read_f32(&c_t);

    assert_eq!(got.len(), m * n);
    for (i, (&g, &x)) in got.iter().zip(&want).enumerate() {
        let rel = (g - x).abs() / x.abs().max(1e-6);
        assert!(
            rel <= TOL_REL,
            "element {i}: GPU {g:e} vs host {x:e}, rel {rel:e} (tol {TOL_REL:e}) -- a \
             row-major read of the blocked payload permutes the weights and misses \
             by O(10%), which this bound still catches"
        );
    }
    Ok(())
}

/// Same journey, but a single row -- the decode case, where the payoff claim is
/// "no worse than what it replaces" rather than "faster".
#[test]
fn a_tag_670_file_at_m_equals_1_matches_the_host_decode() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let (m, n, k) = (1usize, 256usize, 128usize);

    let mut next = xorshift();
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    let scratch = Scratch::new();
    let path = scratch.path("wr_m1.gguf");
    write_white_raven_gguf(&path, n, k, &w);
    let provider = GgufProvider::open(path.to_str().expect("path")).map_err(|e| format!("open: {e}"))?;
    let raw = provider.get("blk.weight").map_err(|e| format!("get: {e}"))?;

    let wq: Vec<f32> = grim_quant::dequant_fp8_blocked16(&raw.bytes, n, k)
        .map_err(|e| format!("host dequant: {e}"))?;
    let want = w8a8_reference(&a, &wq, m, n, k);

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &raw.bytes,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (c_t, _handle) = dev
        .quantized_matmul(
            &*a_t,
            &*b_t,
            &[],
            grim_tensor::QuantFormat::Fp8Blocked16,
            &Shape::new(vec![m, n]),
        )
        .map_err(|e| format!("quantized_matmul: {e}"))?;
    let got = read_f32(&c_t);

    for (i, (&g, &x)) in got.iter().zip(&want).enumerate() {
        // See `TOL_REL`: bit-identical in practice at k=128, but reduction order
        // is not a contract, so the bound is set from the f32 noise floor rather
        // than from the measurement.
        let rel = (g - x).abs() / x.abs().max(1e-6);
        assert!(
            rel <= TOL_REL,
            "decode element {i}: GPU {g:e} vs host {x:e}, rel {rel:e} (tol {TOL_REL:e})"
        );
    }
    Ok(())
}
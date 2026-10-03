//! A tag-672 checkpoint must reach the ForestRaven kernel and produce the
//! right numbers on real hardware: file -> provider -> device -> matmul.
//!
//! Two tests already cover the ends and neither covers the middle:
//! `grim_container_whiteraven` proves `--format forestraven` packs a .grim
//! whose payload decodes to the int8 round of the source, and
//! `forest_dispatch` proves hand-built blobs dispatch correctly. Neither
//! proves the two meet -- a loader that drops the shape, or a dispatch that
//! keys on something the loader does not set, passes both and fails here.
//!
//! Alone in its own binary: route counters are process-wide.
//!
//! The comparison is against the W8A8 model with the kernel's exact
//! activation formula (per-32-block absmax, inv-multiply, full-fp32 scale),
//! asserted with scale-immune statistics (cosine + max abs err) because
//! per-element relative error is meaningless on near-zero outputs.

use std::io::Write;

use grim_backend_rocm::RocmDevice;
use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType};
use grim_format::tprov::GrimProvider;
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

/// A one-tensor F32 GGUF: the source a real exporter produces.
fn write_f32_gguf(path: &std::path::Path, n: usize, k: usize, w: &[f32]) {
    let payload: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "qwen4exp");
    push_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "forest-file-to-gpu");

    push_string(&mut buf, "blk.weight");
    buf.extend_from_slice(&2u32.to_le_bytes());
    // GGUF stores ne fastest-first; shape() reverses it back.
    buf.extend_from_slice(&(k as u64).to_le_bytes());
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&GgufDType::F32.tag().to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.extend_from_slice(&payload);

    let mut f = std::fs::File::create(path).expect("create probe");
    f.write_all(&buf).expect("write probe");
}

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("grim_forest_f2g_{}_{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    grim_backend_rocm::as_rocm(t.as_ref())
        .expect("rocm storage")
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn xorshift() -> impl FnMut() -> u64 {
    let mut s = 0xF02E57u64;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    }
}

/// Host W8A8 with the kernel's exact activation formula.
fn w8a8_reference(a: &[f32], codes: &[u8], scales: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..k.div_ceil(32) {
                let base = blk * 32;
                let len = (k - base).min(32);
                let mut amax = 0.0f32;
                for e in 0..len {
                    amax = amax.max(a[row * k + base + e].abs());
                }
                let d_a = amax / 127.0;
                let inv_a = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                let mut iacc = 0i32;
                for e in 0..len {
                    let qa = ((a[row * k + base + e] * inv_a).round().clamp(-128.0, 127.0)) as i8;
                    let qb = codes[col * k + base + e] as i8;
                    iacc += (qa as i32) * (qb as i32);
                }
                acc += (iacc as f32) * d_a * scales[col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn journey_case(m: usize, n: usize, k: usize, name: &str) -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let arch = dev.gcn_arch().to_string();
    if !(arch.starts_with("gfx11") || arch.starts_with("gfx12")) {
        return Ok(());
    }

    let mut next = xorshift();
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    // 1. Checkpoint in, the way a real exporter writes it.
    let scratch = Scratch::new();
    let src = scratch.0.join("src.gguf");
    write_f32_gguf(&src, n, k, &w);

    // 2. Convert with the flag under test.
    let grim_path = scratch.0.join("out.grim");
    grim_format::convert_to_grim(
        src.to_str().unwrap(),
        grim_path.to_str().unwrap(),
        "gfx1200",
        8.0,
        0,
        None,
        None,
        None,
        None,
        Some("forestraven".to_string()),
        None,
        None,
    )
    .map_err(|e| format!("convert: {e}"))?;

    // 3. Load through the ordinary provider path. Everything the dispatch
    //    needs must come from here.
    let provider = GrimProvider::open(grim_path.to_str().expect("path"))
        .map_err(|e| format!("open: {e}"))?;
    let meta = provider.meta("blk.weight").map_err(|e| format!("meta: {e}"))?;
    assert_eq!(
        meta.dtype.storage,
        Storage::Block(grim_tensor::BlockDtype::Int8PerChannel),
        "{name}: tag 672 must load as row-scaled INT8"
    );
    assert_eq!(meta.shape, vec![n, k], "{name}: loader must preserve [n, k]");
    let raw = provider.get("blk.weight").map_err(|e| format!("get: {e}"))?;

    // 4. Host reference from the SAME file-loaded bytes, so a layout
    //    disagreement shows as a numeric mismatch rather than cancelling out.
    let qw_len = u64::from_le_bytes(raw.bytes[0..8].try_into().unwrap()) as usize;
    let codes = &raw.bytes[8..8 + qw_len];
    let sc_off = 8 + qw_len + 8;
    let scales: Vec<f32> = raw.bytes[sc_off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want = w8a8_reference(&a, codes, &scales, m, n, k);

    // 5. Through the real dispatch, from the file-loaded bytes.
    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &raw.bytes,
        &Shape::new(vec![n, k]),
        meta.dtype,
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (c_t, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Int8PerChannel,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();
    let got = read_f32(&c_t);

    let mut dot = 0.0f64;
    let mut gg = 0.0f64;
    let mut xx = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&g, &x) in got.iter().zip(&want) {
        dot += g as f64 * x as f64;
        gg += (g as f64) * (g as f64);
        xx += (x as f64) * (x as f64);
        max_abs = max_abs.max((g - x).abs());
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("FOREST_JOURNEY {name} cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(cosine >= 0.999, "{name}: cosine {cosine:.6}");
    assert!(max_abs <= 1e-4, "{name}: max abs err {max_abs:e}");
    Ok(())
}

#[test]
fn tag_672_file_reaches_the_kernel_at_decode() -> TestResult {
    journey_case(1, 256, 128, "m1")
}

#[test]
fn tag_672_file_reaches_the_kernel_at_prefill() -> TestResult {
    journey_case(16, 256, 128, "m16")
}

//! A tag-673 checkpoint must reach the GreyRaven kernel and produce the right
//! numbers: file -> provider -> device -> matmul.
//!
//! The chain a real user hits: F32 checkpoint --convert--> .grim (HW-tiled,
//! coupled patterns) --GrimProvider--> dispatch --SWMMAC--> Y. The dispatch
//! test proves hand-built blobs; this proves the two ends meet through the
//! ordinary provider path, including shape preservation ([n, k] -- a
//! transposed or flattened shape still multiplies out to n*k elements and
//! would slip past a byte-count check).
//!
//! Alone in its own binary: route counters are process-wide.
//!
//! Reference: HW-decode the file-loaded bytes + device-mirror E4M3
//! activations (see grey_raven_dispatch for why the mirror, not RNE).
//! Asserted with cosine + max-abs (near-zero outputs make per-element
//! relative error meaningless).

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
    push_string(&mut buf, "grey-file-to-gpu");

    push_string(&mut buf, "blk.weight");
    buf.extend_from_slice(&2u32.to_le_bytes());
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
        let p = std::env::temp_dir().join(format!("grim_grey_f2g_{}_{}", std::process::id(), n));
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
    let mut s = 0x6E67u64;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    }
}

fn device_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let abs = v.abs();
    let sign = if v < 0.0 { 0x80 } else { 0x00 };
    let mut abs = abs;
    if abs > 448.0 {
        abs = 448.0;
    }
    if abs < 0.001953125 {
        let mut mant = (abs * 512.0).round() as u32;
        if mant > 7 {
            mant = 7;
        }
        return sign | mant as u8;
    }
    let u = abs.to_bits();
    let e = ((u >> 23) & 0xFF) as i32 - 127;
    let m = u & 0x7FFFFF;
    let mut e_fp8 = e + 7;
    if e_fp8 >= 15 {
        return sign | 0x7E;
    }
    if e_fp8 <= 0 {
        let mut mant = (abs * 512.0).round() as u32;
        if mant > 7 {
            mant = 7;
        }
        return sign | mant as u8;
    }
    let mut mant_bits = (m + 0x40000) >> 20;
    if mant_bits > 7 {
        mant_bits = 0;
        e_fp8 += 1;
        if e_fp8 >= 15 {
            return sign | 0x7E;
        }
    }
    sign | ((e_fp8 as u8) << 3) | mant_bits as u8
}

fn journey_case(m: usize, n: usize, k: usize, name: &str) -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    if !dev.gcn_arch().starts_with("gfx12") {
        return Ok(());
    }

    let mut next = xorshift();
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 65536.0) - 0.5)
        .collect();

    let scratch = Scratch::new();
    let src = scratch.0.join("src.gguf");
    write_f32_gguf(&src, n, k, &w);

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
        Some("greyraven".to_string()),
        None,
        None,
    )
    .map_err(|e| format!("convert: {e}"))?;

    let provider = GrimProvider::open(grim_path.to_str().expect("path"))
        .map_err(|e| format!("open: {e}"))?;
    let meta = provider.meta("blk.weight").map_err(|e| format!("meta: {e}"))?;
    assert_eq!(
        meta.dtype.storage,
        Storage::Block(grim_tensor::BlockDtype::Fp8Sparse24Hw),
        "{name}: tag 673 must load as HW-tiled"
    );
    assert_eq!(meta.shape, vec![n, k], "{name}: loader must preserve [n, k]");
    let raw = provider.get("blk.weight").map_err(|e| format!("get: {e}"))?;

    // Host reference from the SAME file-loaded bytes.
    let wq = grim_quant::grey_raven::dequant_grey_raven_hw(&raw.bytes, n, k)
        .map_err(|e| format!("host dequant: {e}"))?;
    let aq: Vec<f32> = a
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(device_e4m3(v)))
        .collect();
    let want: Vec<f32> = (0..m * n)
        .map(|o| {
            let (row, col) = (o / n, o % n);
            (0..k).map(|j| aq[row * k + j] * wq[col * k + j]).sum()
        })
        .collect();

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
        grim_tensor::QuantFormat::Fp8Sparse24Hw,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();
    let got = read_f32(&c_t);

    let mut dot = 0.0f64;
    let (mut gg, mut xx) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    for (&g, &x) in got.iter().zip(&want) {
        dot += g as f64 * x as f64;
        gg += (g as f64) * (g as f64);
        xx += (x as f64) * (x as f64);
        max_abs = max_abs.max((g - x).abs());
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("GREY_JOURNEY {name} cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(cosine >= 0.9999, "{name}: cosine {cosine:.6}");
    assert!(max_abs <= 1e-3, "{name}: max abs err {max_abs:e}");
    Ok(())
}

#[test]
fn tag_673_file_reaches_the_kernel_at_decode() -> TestResult {
    journey_case(1, 64, 128, "m1")
}

#[test]
fn tag_673_file_reaches_the_kernel_at_prefill() -> TestResult {
    journey_case(16, 64, 128, "m16")
}

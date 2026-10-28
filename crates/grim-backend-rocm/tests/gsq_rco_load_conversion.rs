//! GSQ-RCO convert-on-load: a 2.25 bpw `GsqRco3p5` weight must be encoded into
//! the W4A4OstQuant blob on the HOST at upload, so the packed copy never lands
//! in VRAM (10.4 GB packed + ~18.5 GB lanes ≈ 29 GB for the 27B — it fits
//! neither one card nor two; the lanes alone split cleanly).
//!
//! Three gates:
//! 1. the upload converts (storage retagged `W4A4OstQuant`, sized to the blob);
//! 2. PARITY — the converted-at-load GEMV is bit-identical to the lazy-memo
//!    path on the same packed bytes (the repack is lossless, so identical is
//!    the only correct answer);
//! 3. the gate fallback — a K that is not a multiple of the 128 group stays
//!    `KQuant(GsqRco3p5)` and keeps the lazy path.

use grim_backend_rocm::{CoreTensorOps, MemoryOps, QuantOps, RocmDevice, RocmStorage};
use grim_tensor::Shape;
use grim_tensor::backend::{BackendStorage, ComputeHandle};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, OstQuantConfig};

fn gsq_dtype() -> DType {
    DType {
        arith: ArithType::U8,
        storage: grim_tensor::Storage::KQuant(KQuantScheme::GsqRco3p5),
    }
}

/// GSQ-RCO is 18 B per 64 weights (2.25 bpw); `quantize_gsq_rco_3p5_block` is
/// the production packer, one 64-block at a time.
fn pack_gsq_rco(rows: &[Vec<f32>]) -> Vec<u8> {
    let mut packed = Vec::new();
    for row in rows {
        for blk in row.chunks(64) {
            let mut out = [0u8; 18];
            grim_quant::quantize_gsq_rco_3p5_block(blk, &mut out).expect("pack block");
            packed.extend_from_slice(&out);
        }
    }
    packed
}

fn run_gemv(
    dev: &RocmDevice,
    b: &dyn BackendStorage,
    n: usize,
    k: usize,
    activations: &[f32],
) -> Vec<f32> {
    let a = dev
        .from_cpu(activations, &Shape::from_slice(&[1, k]), DType::F32)
        .expect("activations");
    let out_shape = Shape::from_slice(&[1, n]);
    let (out, _handle): (Box<dyn BackendStorage>, Box<dyn ComputeHandle>) = dev
        .quantized_matmul(
            a.as_ref(),
            b,
            &[],
            grim_tensor::QuantFormat::W4A4OstQuant,
            &out_shape,
        )
        .expect("quantized_matmul");
    out.to_cpu_vec_f32().expect("readback")
}

#[test]
fn gsq_rco_upload_converts_and_matches_the_lazy_path_bit_for_bit() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // k must be a multiple of the 128 group for the W4A4OstQuant arm.
    let (n, k) = (2usize, 256usize);
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|r| {
            (0..k)
                .map(|i| (((r * k + i) % 13) as f32 / 13.0) - 0.5)
                .collect()
        })
        .collect();
    let packed = pack_gsq_rco(&rows);
    let packed_len = packed.len();

    // 1. Convert-on-load: the upload must retag to W4A4OstQuant and size to the
    //    blob (24 B of length prefixes + nibbles + bf16 scales + u8 zeros) —
    //    NOT the packed size. If the packed bytes were resident, bytes() would
    //    report packed_len.
    let shape = Shape::from_slice(&[n, k]);
    let converted = dev
        .from_cpu_bytes(&packed, &shape, gsq_dtype())
        .expect("convert-on-load upload");
    let want_bytes = 24 + (n * k).div_ceil(2) + (n * k / 128) * 2 + (n * k / 128);
    let conv_rocm = converted
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocms torage");
    assert_eq!(
        conv_rocm.bytes(),
        want_bytes,
        "resident allocation must be the blob ({want_bytes} B), not the packed weight ({packed_len} B)"
    );
    assert!(
        matches!(
            converted.dtype().storage,
            grim_tensor::Storage::W4A4OstQuant(OstQuantConfig { group_size: 128 })
        ),
        "storage must be retagged W4A4OstQuant(group 128), got {:?}",
        converted.dtype().storage
    );

    // 2. Parity: the same packed bytes through the OLD path (direct raw upload,
    //    lazy memo at GEMV time) must produce bit-identical output. The repack
    //    is lossless — bit-identical is the only correct answer.
    let activations: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 / 7.0) - 0.35).collect();
    let got_blob = run_gemv(&dev, converted.as_ref(), n, k, &activations)
        .into_iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();

    let lazy = RocmStorage::copy_from_host_raw_bytes(
        &packed,
        &shape,
        gsq_dtype(),
        &dev.allocator_handle(),
        0,
    )
    .expect("packed upload");
    assert_eq!(lazy.dtype().storage, gsq_dtype().storage);
    let got_lazy = run_gemv(&dev, &lazy, n, k, &activations)
        .into_iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();

    assert_eq!(
        got_blob, got_lazy,
        "convert-on-load GEMV must be bit-identical to the lazy-memo GEMV"
    );
}

#[test]
fn gsq_rco_gate_failure_falls_back_to_packed() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // k = 64 is a valid GSQ-RCO width but not a multiple of the 128 group, so
    // the W4A4OstQuant arm would refuse it at serve time — convert-on-load must
    // refuse it too, and the packed storage must survive unchanged.
    let (n, k) = (1usize, 64usize);
    let rows = vec![vec![0.1f32; k]];
    let packed = pack_gsq_rco(&rows);
    let up = dev
        .from_cpu_bytes(&packed, &Shape::from_slice(&[n, k]), gsq_dtype())
        .expect("fallback upload");
    assert_eq!(
        up.dtype().storage,
        gsq_dtype().storage,
        "gate failure must fall back to the packed KQuant upload"
    );
}

#[test]
fn non_gsq_schemes_are_untouched() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let packed = vec![7u8; 256];
    let dtype = DType {
        arith: ArithType::U8,
        storage: grim_tensor::Storage::KQuant(KQuantScheme::Q80),
    };
    let up = dev
        .from_cpu_bytes(&packed, &Shape::from_slice(&[2, 128]), dtype.clone())
        .expect("upload");
    assert_eq!(
        up.dtype().storage,
        dtype.storage,
        "only GsqRco3p5 converts; every other scheme uploads packed as before"
    );
}

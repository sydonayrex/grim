//! WhiteCrow grouped-dispatch parity: `grim_moe_fused_dispatch_whitecrow`
//! against the host dequant-of-blob reference on identical blob bytes.
//!
//! Both sides consume the SAME OSTQuant u4-group128 blobs (f32 math, zero-
//! point algebra d*(sum(a*q) - z*sum(a))), so any delta beyond f32 rounding
//! is a kernel bug — nibble order, scale index, expert stride, routing
//! indexing, SiLU wiring. The distance to the SOURCE f32 weights (u4
//! quantization error) is printed but intentionally not asserted here: it is
//! a property of the format, not of this kernel.
//!
//! RUN: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!   --test charon_whitecrow_grouped_parity

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::Shape;
use grim_tensor::MemoryOps;
use std::sync::Arc;

const HIDDEN: usize = 256; // group-128: hidden % 128 == 0
const INTER: usize = 128; // inter % 128 == 0
const NUM_EXPERTS: usize = 3;
const BATCH: usize = 2;
#[allow(dead_code)]
const TOP_K: usize = 2;

fn u8_storage(dev: &RocmDevice, v: &[u8]) -> Arc<dyn grim_tensor::backend::BackendStorage> {
    Arc::from(
        dev.from_cpu_bytes(
            v,
            &Shape::new(vec![v.len()]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload u8"),
    )
}

fn u32_storage(dev: &RocmDevice, v: &[u32]) -> Arc<dyn grim_tensor::backend::BackendStorage> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    u8_storage(dev, &bytes)
}

fn f32_storage(dev: &RocmDevice, v: &[f32]) -> Arc<dyn grim_tensor::backend::BackendStorage> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    u8_storage(dev, &bytes)
}

/// OSTQuant u4-group128 encode of one [n, k] row-major weight, as host bytes
/// in the WhiteCrow blob segment order (qw u32 words, sc bf16, zr u8).
fn wc_encode(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    grim_quant::quant_ostquant_w4_group128(w, n, k).expect("ostquant encode")
}

fn blob_from(qw: &[u8], sc: &[u8], zr: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(24 + qw.len() + sc.len() + zr.len());
    b.extend_from_slice(&(qw.len() as u64).to_le_bytes());
    b.extend_from_slice(qw);
    b.extend_from_slice(&(sc.len() as u64).to_le_bytes());
    b.extend_from_slice(sc);
    b.extend_from_slice(&(zr.len() as u64).to_le_bytes());
    b.extend_from_slice(zr);
    b
}

/// Host reference: dequantize one expert's blob columns and run the same
/// fused gate|up -> silu -> down math with f32 activations.
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7],
    ])
}

fn bf16(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((hi as u32) << 24) | ((lo as u32) << 16))
}

fn host_reference(
    gate_blob: &[u8],
    up_blob: &[u8],
    down_blob: &[u8],
    a: &[f32],
    tok: usize,
    exp: usize,
    gate_stride: u64,
    down_stride: u64,
    routed_scaling: f32,
    w: f32,
    out: &mut [f32],
) {
    fn seg(blob: &[u8], _n: usize, k: usize) -> (&[u8], &[u8], &[u8], usize) {
        let lq = rd_u64(blob, 0) as usize;
        let qw = &blob[8..8 + lq];
        let sc_off = 8 + lq;
        let ls = rd_u64(blob, sc_off) as usize;
        let sc = &blob[sc_off + 8..sc_off + 8 + ls];
        let zr_off = sc_off + 8 + ls;
        let zr = &blob[zr_off + 8..];
        let n_groups = k / 128;
        (qw, sc, zr, n_groups)
    }
    let mut act = vec![0f32; INTER];
    for proj in 0..2 {
        let blob = &gate_blob[exp as usize * gate_stride as usize
            ..(exp as usize + 1) * gate_stride as usize];
        let src = if proj == 1 { &up_blob[exp as usize * gate_stride as usize
            ..(exp as usize + 1) * gate_stride as usize] } else { blob };
        let (qw, sc, zr, n_groups) = seg(src, INTER, HIDDEN);
        let _ = (sc, zr, qw, n_groups);
        for j in 0..INTER {
            let mut g = 0f32;
            for gg in 0..n_groups {
                let d = bf16(sc[(j * n_groups + gg) * 2], sc[(j * n_groups + gg) * 2 + 1]);
                let z = zr[j * n_groups + gg] as f32;
                let mut sq = 0f32;
                let mut sa = 0f32;
                for wi in 0..16 {
                    let word = u32::from_le_bytes(
                        qw[j * (HIDDEN / 8) + gg * 16 + wi ..][..4].try_into().unwrap(),
                    );
                    for t in 0..8 {
                        let av = a[tok * HIDDEN + gg * 128 + wi * 8 + t];
                        let q = ((word >> (4 * t)) & 0xF) as f32;
                        sq += av * q;
                        sa += av;
                    }
                }
                g += d * (sq - z * sa);
            }
            let _ = (sc, zr, qw, n_groups);
            if proj == 0 {
                act[j] = g / (1.0 + (-g).exp());
            } else {
                act[j] *= g;
            }
        }
    }
    let dblob = &down_blob[exp as usize * down_stride as usize
        ..(exp as usize + 1) * down_stride as usize];
    let (qw, sc, zr, n_groups) = seg(dblob, HIDDEN, INTER);
    for h in 0..HIDDEN {
        let mut acc = 0f32;
        for gg in 0..n_groups {
            let d = bf16(sc[(h * n_groups + gg) * 2], sc[(h * n_groups + gg) * 2 + 1]);
            let z = zr[h * n_groups + gg] as f32;
            let mut sq = 0f32;
            let mut sa = 0f32;
            for wi in 0..16 {
                let word = u32::from_le_bytes(
                    qw[h * (INTER / 8) + gg * 16 + wi ..][..4].try_into().unwrap(),
                );
                for t in 0..8 {
                    let av = act[gg * 128 + wi * 8 + t];
                    let q = ((word >> (4 * t)) & 0xF) as f32;
                    sq += av * q;
                    sa += av;
                }
            }
            acc += d * (sq - z * sa);
        }
        out[tok * HIDDEN + h] += routed_scaling * w * acc;
    }
}

#[test]
#[ignore]
fn whitecrow_grouped_matches_host_reference() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("RocmDevice");

    // Deterministic weights, magnitudes like a real FFN.
    let mk = |n: usize, salt: f32| -> Vec<f32> {
        (0..NUM_EXPERTS * n)
            .map(|i| {
                let s = (i as f64 * 2654435761.0).fract();
                ((s * 2.0 - 1.0) as f32) * 0.08 * salt
            })
            .collect()
    };
    let (gw, uw, dw) = (
        mk(HIDDEN * INTER, 1.0),
        mk(HIDDEN * INTER, 1.3),
        mk(INTER * HIDDEN, 0.8),
    );

    // Encode per expert and concatenate into expert-strided blobs.
    let mut gate_blob = Vec::new();
    let mut up_blob = Vec::new();
    let mut down_blob = Vec::new();
    for e in 0..NUM_EXPERTS {
        let (q, s, z) = wc_encode(&gw[e * HIDDEN * INTER..(e + 1) * HIDDEN * INTER], INTER, HIDDEN);
        gate_blob.extend_from_slice(&blob_from(&q, &s, &z));
        let (q, s, z) = wc_encode(&uw[e * HIDDEN * INTER..(e + 1) * HIDDEN * INTER], INTER, HIDDEN);
        up_blob.extend_from_slice(&blob_from(&q, &s, &z));
        let (q, s, z) = wc_encode(&dw[e * INTER * HIDDEN..(e + 1) * INTER * HIDDEN], HIDDEN, INTER);
        down_blob.extend_from_slice(&blob_from(&q, &s, &z));
    }
    let gate_stride = (gate_blob.len() / NUM_EXPERTS) as u64;
    let down_stride = (down_blob.len() / NUM_EXPERTS) as u64;

    let act: Vec<f32> = (0..BATCH * HIDDEN)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.09)
        .collect();

    // Routing: token 0 -> {0,1}, token 1 -> {2,0}.
    let tokens = vec![0u32, 0, 1, 1];
    let experts = vec![0u32, 1, 2, 0];
    let weights = vec![0.6f32, 0.4, 0.55, 0.45];
    let num_pairs = tokens.len();
    let rsf = 1.25f32;

    let a_dev = f32_storage(&dev, &act);
    let g_dev = u8_storage(&dev, &gate_blob);
    let u_dev = u8_storage(&dev, &up_blob);
    let d_dev = u8_storage(&dev, &down_blob);
    let t_dev = u32_storage(&dev, &tokens);
    let e_dev = u32_storage(&dev, &experts);
    let w_dev = f32_storage(&dev, &weights);
    let out_shape = Shape::new(vec![BATCH, HIDDEN]);
    let out_rocm = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &out_shape,
        DType::F32,
        &dev.allocator_handle(),
        0,
    )
    .expect("out alloc");
    let a_rocm = a_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("a rocm");
    let g_rocm = g_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("g rocm");
    let u_rocm = u_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("u rocm");
    let d_rocm = d_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("d rocm");
    let t_rocm = t_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("t rocm");
    let e_rocm = e_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("e rocm");
    let w_rocm = w_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("w rocm");

    dev.moe_fused_dispatch_whitecrow_grouped_into(
        a_rocm,
        g_rocm,
        u_rocm,
        d_rocm,
        t_rocm,
        e_rocm,
        w_rocm,
        num_pairs,
        &out_rocm,
        HIDDEN,
        INTER,
        rsf,
        gate_stride,
        down_stride,
    )
    .expect("whitecrow grouped dispatch");
    dev.synchronize();
    let got_host = out_rocm.copy_to_host().expect("read out");
    let got: Vec<f32> = got_host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Host reference over the SAME blobs.
    let mut want = vec![0f32; BATCH * HIDDEN];
    for (p, &t) in tokens.iter().enumerate() {
        host_reference(
            &gate_blob,
            &up_blob,
            &down_blob,
            &act,
            t as usize,
            experts[p] as usize,
            gate_stride,
            down_stride,
            rsf,
            weights[p],
            &mut want,
        );
    }

    let mut max_abs = 0f32;
    let mut scale = 1e-6f32;
    for (g, wv) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - wv).abs());
        scale = scale.max(g.abs()).max(wv.abs());
    }
    let rel = max_abs / scale;
    eprintln!(
        "[whitecrow-parity] kernel vs host-over-blobs: max_abs {max_abs:.3e} rel {rel:.3e}"
    );
    assert!(
        rel < 1e-4,
        "grouped WhiteCrow kernel diverges from the host blob reference (rel {rel:.3e})"
    );
}

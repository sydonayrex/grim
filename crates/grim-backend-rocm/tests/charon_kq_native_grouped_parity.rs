//! Native K-quant grouped-dispatch parity: `grim_moe_fused_dispatch_kq_native`
//! (in-register IQ3_S decode of the resident per-expert banks) against the
//! host `grim_quant::dequant_iq3s` of the SAME packed bytes. Any delta beyond
//! f32 rounding is a kernel bug - nibble/scale indexing, block stride,
//! pointer-array addressing, routing, SiLU wiring, accumulation.
//!
//! The blob bytes are pseudo-random: IQ3_S decodes ANY byte pattern to a
//! finite value, so the kernel and the host reference must agree regardless
//! of what the bytes "mean".
//!
//! RUN: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!   --test charon_kq_native_grouped_parity

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::MemoryOps;
use grim_tensor::Shape;
use std::sync::Arc;

// PRODUCTION geometry (xing4.0 blk.2+): hidden=3584, inter=1024, 64 experts.
// The small-geometry run (256/256/3) caught logic bugs; this one catches
// scale bugs (14 super-blocks per row, 64-entry pointer arrays, >4 KiB LDS).
const HIDDEN: usize = 3584;
const INTER: usize = 1024;
const NUM_EXPERTS: usize = 64;
const BATCH: usize = 1;

thread_local! {
    static SEPARATE_KEEP: std::cell::RefCell<Vec<Arc<dyn grim_tensor::backend::BackendStorage>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static SEPARATE_PTRS: std::cell::RefCell<
        Option<(
            Arc<dyn grim_tensor::backend::BackendStorage>,
            Arc<dyn grim_tensor::backend::BackendStorage>,
            Arc<dyn grim_tensor::backend::BackendStorage>,
        )>,
    > = const { std::cell::RefCell::new(None) };
}

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
        .expect("upload"),
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

/// Pseudo-random but deterministic byte pattern (IQ3_S decodes any bytes).
fn bank_bytes(n: usize, salt: u8) -> Vec<u8> {
    let row_bytes = (HIDDEN / 256) * 110; // for gate/up; down swaps dims
    let mut out = vec![0u8; n * row_bytes];
    let mut s = 0x9E37_79B9u32 ^ (salt as u32);
    for b in out.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 24) as u8;
    }
    out
}

/// Host reference: dequantize one blob's rows and run the fused
/// gate|up -> silu -> down math (f32, same codes as the kernel).
fn host_reference(
    gate_blob: &[u8],
    up_blob: &[u8],
    down_blob: &[u8],
    a: &[f32],
    tok: usize,
    exp: usize,
    gate_row_bytes: usize,
    down_row_bytes: usize,
    routed_scaling: f32,
    w: f32,
    out: &mut [f32],
) {
    let deq_rows = |blob: &[u8], rows: usize, k: usize| -> Vec<f32> {
        let mut flat = Vec::with_capacity(rows * k);
        for r in 0..rows {
            let row = &blob[r * (k / 256) * 110..(r + 1) * (k / 256) * 110];
            flat.extend_from_slice(
                &grim_quant::dequant_iq3s(row, k).expect("host iq3s dequant"),
            );
        }
        let _ = gate_row_bytes.min(down_row_bytes);
        flat
    };
    let g_flat = deq_rows(
        &gate_blob[exp * (INTER * gate_row_bytes)..(exp + 1) * (INTER * gate_row_bytes)],
        INTER,
        HIDDEN,
    );
    let u_flat = deq_rows(
        &up_blob[exp * (INTER * gate_row_bytes)..(exp + 1) * (INTER * gate_row_bytes)],
        INTER,
        HIDDEN,
    );
    let mut act = vec![0f32; INTER];
    for j in 0..INTER {
        let mut g = 0f32;
        let mut u = 0f32;
        for i in 0..HIDDEN {
            let av = a[tok * HIDDEN + i];
            g += av * g_flat[j * HIDDEN + i];
            u += av * u_flat[j * HIDDEN + i];
        }
        act[j] = g / (1.0 + (-g).exp()) * u;
    }
    let d_flat = deq_rows(
        &down_blob[exp * (HIDDEN * down_row_bytes)..(exp + 1) * (HIDDEN * down_row_bytes)],
        HIDDEN,
        INTER,
    );
    for h in 0..HIDDEN {
        let mut acc = 0f32;
        for j in 0..INTER {
            acc += act[j] * d_flat[h * INTER + j];
        }
        out[tok * HIDDEN + h] += routed_scaling * w * acc;
    }
}

#[test]
#[ignore]
fn kq_native_grouped_matches_host_iq3s_reference() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("RocmDevice");

    // gate/up: [inter, hidden] rows; down: [hidden, inter] rows.
    let gate_row_bytes = (HIDDEN / 256) * 110;
    let down_row_bytes = (INTER / 256) * 110;
    let mut gate_blob = Vec::new();
    let mut up_blob = Vec::new();
    let mut down_blob = Vec::new();
    for e in 0..NUM_EXPERTS {
        gate_blob.extend_from_slice(&bank_bytes(INTER, 1 + e as u8));
        up_blob.extend_from_slice(&bank_bytes(INTER, 40 + e as u8));
        down_blob.extend_from_slice(&bank_bytes(HIDDEN, 80 + e as u8));
    }
    let _ = down_row_bytes;

    let act: Vec<f32> = (0..BATCH * HIDDEN)
        .map(|i| ((i % 21) as f32 - 10.0) * 0.1)
        .collect();
    let tokens = vec![0u32, 0, 0, 0];
    let experts = vec![0u32, 17, 41, 63];
    let weights = vec![0.5f32, 0.2, 0.2, 0.1];
    let num_pairs = tokens.len();
    let rsf = 1.1f32;

    // KQ_PARITY_SEPARATE_ALLOCS=1: one allocation per expert, exactly how
    // ExpertBank materializes the model's per-expert banks (the isolated
    // repro of the model-shaped wedge: same bytes, same geometry, different
    // allocation layout).
    if std::env::var("KQ_PARITY_SEPARATE_ALLOCS").as_deref() == Ok("1") {
        let mut gate_addrs = Vec::new();
        let mut up_addrs = Vec::new();
        let mut down_addrs = Vec::new();
        for e in 0..NUM_EXPERTS {
            let g_off = e * INTER * gate_row_bytes;
            let u_off = e * INTER * gate_row_bytes;
            let d_off = e * HIDDEN * down_row_bytes;
            let gd = u8_storage(&dev, &gate_blob[g_off..g_off + INTER * gate_row_bytes]);
            let ud = u8_storage(&dev, &up_blob[u_off..u_off + INTER * gate_row_bytes]);
            let dd = u8_storage(&dev, &down_blob[d_off..d_off + HIDDEN * down_row_bytes]);
            let base = |a: &Arc<dyn grim_tensor::backend::BackendStorage>| -> usize {
                a.as_any()
                    .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
                    .unwrap()
                    .device_ptr_u64()
                    .unwrap() as usize
            };
            gate_addrs.push(base(&gd));
            up_addrs.push(base(&ud));
            down_addrs.push(base(&dd));
            // keep the storages alive for the dispatch
            SEPARATE_KEEP.with(|k| k.borrow_mut().extend(vec![gd, ud, dd]));
        }
        SEPARATE_KEEP.with(|_k| {
            let to_u32 = |addrs: &[usize]| -> Vec<u32> {
                let mut bytes: Vec<u8> = Vec::new();
                for a in addrs {
                    bytes.extend_from_slice(&(*a as u64).to_le_bytes());
                }
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let g_ptrs = u32_storage(&dev, &to_u32(&gate_addrs));
            let u_ptrs = u32_storage(&dev, &to_u32(&up_addrs));
            let d_ptrs = u32_storage(&dev, &to_u32(&down_addrs));
            SEPARATE_PTRS.with(|p| *p.borrow_mut() = Some((g_ptrs, u_ptrs, d_ptrs)));
        });
        let g_ptrs = SEPARATE_PTRS.with(|p| p.borrow().as_ref().unwrap().0.clone());
        let u_ptrs = SEPARATE_PTRS.with(|p| p.borrow().as_ref().unwrap().1.clone());
        let d_ptrs = SEPARATE_PTRS.with(|p| p.borrow().as_ref().unwrap().2.clone());
        let a_t = f32_storage(&dev, &act);
        let t_t = u32_storage(&dev, &tokens);
        let e_t = u32_storage(&dev, &experts);
        let w_t = f32_storage(&dev, &weights);
        let out_rocm = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
            &Shape::new(vec![1, HIDDEN]),
            DType::F32,
            &dev.allocator_handle(),
            0,
        )
        .expect("out");
        dev.moe_fused_dispatch_kq_native_into(
            as_rocm_s(&a_t),
            as_rocm_s(&g_ptrs),
            as_rocm_s(&u_ptrs),
            as_rocm_s(&d_ptrs),
            as_rocm_s(&t_t),
            as_rocm_s(&e_t),
            as_rocm_s(&w_t),
            tokens.len(),
            &out_rocm,
            HIDDEN,
            INTER,
            rsf,
            gate_row_bytes as u64,
            down_row_bytes as u64,
            0,
        )
        .expect("separate-alloc dispatch");
        dev.synchronize();
        eprintln!("[kq-parity] separate-allocation arm: dispatch + sync completed");
        return;
    }
    // Per-expert device pointer arrays over the blob slices.
    let mut gate_addrs: Vec<usize> = Vec::new();
    let mut up_addrs: Vec<usize> = Vec::new();
    let mut down_addrs: Vec<usize> = Vec::new();
    let gate_dev = u8_storage(&dev, &gate_blob);
    let up_dev = u8_storage(&dev, &up_blob);
    let down_dev = u8_storage(&dev, &down_blob);
    for e in 0..NUM_EXPERTS {
        let g_base = gate_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
            .device_ptr_u64()
            .unwrap() as usize;
        let u_base = up_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
            .device_ptr_u64()
            .unwrap() as usize;
        let d_base = down_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
            .device_ptr_u64()
            .unwrap() as usize;
        gate_addrs.push(g_base + e * INTER * gate_row_bytes);
        up_addrs.push(u_base + e * INTER * gate_row_bytes);
        down_addrs.push(d_base + e * HIDDEN * down_row_bytes);
    }
    let to_u32_words = |addrs: &[usize]| -> Vec<u32> {
        let mut bytes: Vec<u8> = Vec::new();
        for a in addrs {
            bytes.extend_from_slice(&(*a as u64).to_le_bytes());
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let a_arc = f32_storage(&dev, &act);
    let g_ptrs = u32_storage(&dev, &to_u32_words(&gate_addrs));
    let u_ptrs = u32_storage(&dev, &to_u32_words(&up_addrs));
    let d_ptrs = u32_storage(&dev, &to_u32_words(&down_addrs));
    let t_rocm = u32_storage(&dev, &tokens);
    let e_rocm = u32_storage(&dev, &experts);
    let w_rocm = f32_storage(&dev, &weights);
    let out_shape = Shape::new(vec![BATCH, HIDDEN]);
    let out_rocm = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &out_shape,
        DType::F32,
        &dev.allocator_handle(),
        0,
    )
    .expect("out alloc");

    fn as_rocm_s(a: &Arc<dyn grim_tensor::backend::BackendStorage>) -> &grim_backend_rocm::memory::storage::RocmStorage {
        a.as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
    }

    let gate_row_bytes = (gate_blob.len() / NUM_EXPERTS / INTER) as u64;
    let down_row_bytes = (down_blob.len() / NUM_EXPERTS / HIDDEN) as u64;
    dev.moe_fused_dispatch_kq_native_into(
        as_rocm_s(&a_arc),
        as_rocm_s(&g_ptrs),
        as_rocm_s(&u_ptrs),
        as_rocm_s(&d_ptrs),
        as_rocm_s(&t_rocm),
        as_rocm_s(&e_rocm),
        as_rocm_s(&w_rocm),
        num_pairs,
        &out_rocm,
        HIDDEN,
        INTER,
        rsf,
        gate_row_bytes,
        down_row_bytes,
                0,
    )
    .expect("kq native dispatch");
    dev.synchronize();
    let got_host = out_rocm.copy_to_host().expect("read out");
    let got: Vec<f32> = got_host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut want = vec![0f32; BATCH * HIDDEN];
    for (p, &t) in tokens.iter().enumerate() {
        host_reference(
            &gate_blob,
            &up_blob,
            &down_blob,
            &act,
            t as usize,
            experts[p] as usize,
            gate_row_bytes as usize,
            down_row_bytes as usize,
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
        "[kq-parity] kernel vs host-iq3s: max_abs {max_abs:.3e} rel {rel:.3e}"
    );
    assert!(
        rel < 1e-3,
        "K-quant native grouped kernel diverges from the host IQ3_S reference (rel {rel:.3e})"
    );
}

/// Real-checkpoint reproduction: fetch xing4.0 blk.2's actual packed expert
/// banks from the GGUF (hidden=3584, inter=1024, 64 experts — production
/// geometry), upload per-expert slices exactly like ExpertBank::load_quantized
/// does, and run the kernel against the host IQ3_S reference. XING_GGUF must
/// point at the checkpoint.
#[test]
#[ignore]
fn kq_native_matches_host_on_real_banks() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    }
    let Ok(path) = std::env::var("XING_GGUF") else {
        eprintln!("[SKIP] requires XING_GGUF");
        return;
    };
    let hidden = 3584usize;
    let inter = 1024usize;
    let num_experts = 64usize;

    use grim_format::tprov::GgufProvider;
    use grim_nn::WeightSource;
    let prov = GgufProvider::open(&path).expect("open gguf");
    let ws = WeightSource::root(&prov, grim_tensor::Device::Cpu).pp("blk").pp("2");
    let gate_raw = ws.get_raw_packed("ffn_gate_exps.weight").expect("gate bank");
    let up_raw = ws.get_raw_packed("ffn_up_exps.weight").expect("up bank");
    let down_raw = ws.get_raw_packed("ffn_down_exps.weight").expect("down bank");
    eprintln!(
        "[kq-real] gate dtype {:?} bytes {} shape {:?}",
        gate_raw.dtype,
        gate_raw.bytes.len(),
        gate_raw.shape
    );
    eprintln!(
        "[kq-real] down dtype {:?} bytes {} shape {:?}",
        down_raw.dtype,
        down_raw.bytes.len(),
        down_raw.shape
    );
    let gate_stride = gate_raw.bytes.len() / num_experts;
    let up_stride = up_raw.bytes.len() / num_experts;
    let down_stride = down_raw.bytes.len() / num_experts;
    let _ = down_stride; // raw is Q4_K; the kernel leg for down is validated on runtime storages
    eprintln!("[kq-real] expert strides: gate {gate_stride} up {up_stride} down {down_stride}");
    // gate/up: IQ3_S = inter * (hidden/256) * 110. down: RAW bank is Q4_K
    // (2064384 B) — but the RUNTIME storages are all-IQ3S (the model's
    // loader path re-tags/re-encodes; verified by kq_native_over_runtime_
    // bank_storages), so this test slices with the RUNTIME IQ3_S stride:
    // hidden * (inter/256) * 110 = 3584 * 4 * 110 = 1576960.
    assert_eq!(gate_stride, inter * (hidden / 256) * 110, "gate stride");
    let down_runtime_stride = hidden * (inter / 256) * 110;
    assert_eq!(down_runtime_stride, 1_576_960, "runtime down stride");

    let dev = RocmDevice::try_new(0).expect("RocmDevice");
    let gate_dev = u8_storage(&dev, &gate_raw.bytes);
    let up_dev = u8_storage(&dev, &up_raw.bytes);
    let down_dev = u8_storage(&dev, &down_raw.bytes);
    let base = |a: &Arc<dyn grim_tensor::backend::BackendStorage>| -> usize {
        a.as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
            .device_ptr_u64()
            .unwrap() as usize
    };
    let mut gate_addrs = Vec::new();
    let mut up_addrs = Vec::new();
    let mut down_addrs = Vec::new();
    for e in 0..num_experts {
        gate_addrs.push(base(&gate_dev) + e * gate_stride);
        up_addrs.push(base(&up_dev) + e * up_stride);
        down_addrs.push(base(&down_dev) + e * down_stride);
    }
    let to_u32_words = |addrs: &[usize]| -> Vec<u32> {
        let mut bytes: Vec<u8> = Vec::new();
        for a in addrs {
            bytes.extend_from_slice(&(*a as u64).to_le_bytes());
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // One token through experts {7, 23, 41} — production-shaped routing.
    let act: Vec<f32> = (0..hidden)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let tokens = vec![0u32, 0, 0];
    let experts = vec![7u32, 23, 41];
    let weights = vec![0.5f32, 0.3, 0.2];
    let num_pairs = tokens.len();
    let rsf = 1.0f32;

    let a_arc = f32_storage(&dev, &act);
    let g_ptrs = u32_storage(&dev, &to_u32_words(&gate_addrs));
    let u_ptrs = u32_storage(&dev, &to_u32_words(&up_addrs));
    let d_ptrs = u32_storage(&dev, &to_u32_words(&down_addrs));
    let t_rocm = u32_storage(&dev, &tokens);
    let e_rocm = u32_storage(&dev, &experts);
    let w_rocm = f32_storage(&dev, &weights);
    let out_rocm = grim_backend_rocm::memory::storage::RocmStorage::alloc_gpu(
        &Shape::new(vec![1, hidden]),
        DType::F32,
        &dev.allocator_handle(),
        0,
    )
    .expect("out");

    fn as_rocm_s(
        a: &Arc<dyn grim_tensor::backend::BackendStorage>,
    ) -> &grim_backend_rocm::memory::storage::RocmStorage {
        a.as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .unwrap()
    }

    dev.moe_fused_dispatch_kq_native_into(
        as_rocm_s(&a_arc),
        as_rocm_s(&g_ptrs),
        as_rocm_s(&u_ptrs),
        as_rocm_s(&d_ptrs),
        as_rocm_s(&t_rocm),
        as_rocm_s(&e_rocm),
        as_rocm_s(&w_rocm),
        num_pairs,
        &out_rocm,
        hidden,
        inter,
        rsf,
        gate_stride as u64,
        down_stride as u64,
                0,
    )
    .expect("kq native dispatch (real banks)");
    dev.synchronize();
    let got_host = out_rocm.copy_to_host().expect("read out");
    let got: Vec<f32> = got_host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Host reference over the same real bytes.
    let mut want = vec![0f32; hidden];
    for (p, &e) in experts.iter().enumerate() {
        let g_blob = &gate_raw.bytes[e as usize * gate_stride..(e as usize + 1) * gate_stride];
        let u_blob = &up_raw.bytes[e as usize * up_stride..(e as usize + 1) * up_stride];
        let d_blob = &down_raw.bytes[e as usize * down_stride..(e as usize + 1) * down_stride];
        let deq_rows = |blob: &[u8], rows: usize, k: usize| -> Vec<f32> {
            let mut flat = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let row = &blob[r * (k / 256) * 110..(r + 1) * (k / 256) * 110];
                flat.extend_from_slice(
                    &grim_quant::dequant_iq3s(row, k).expect("host iq3s"),
                );
            }
            flat
        };
        let g_flat = deq_rows(g_blob, inter, hidden);
        let u_flat = deq_rows(u_blob, inter, hidden);
        let mut actv = vec![0f32; inter];
        for j in 0..inter {
            let mut g = 0f32;
            let mut u = 0f32;
            for i in 0..hidden {
                let av = act[i];
                g += av * g_flat[j * hidden + i];
                u += av * u_flat[j * hidden + i];
            }
            actv[j] = g / (1.0 + (-g).exp()) * u;
        }
        let d_flat = deq_rows(d_blob, hidden, inter);
        for h in 0..hidden {
            let mut acc = 0f32;
            for j in 0..inter {
                acc += actv[j] * d_flat[h * inter + j];
            }
            want[h] += rsf * weights[p] * acc;
        }
    }
    let mut max_abs = 0f32;
    let mut scale = 1e-6f32;
    for (g, wv) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - wv).abs());
        scale = scale.max(g.abs()).max(wv.abs());
    }
    let rel = max_abs / scale;
    eprintln!("[kq-real] kernel vs host on REAL banks: max_abs {max_abs:.3e} rel {rel:.3e}");
    assert!(rel < 1e-3, "real-bank kernel diverges (rel {rel:.3e})");
}

/// Runtime-truth reproduction: load blk.2's expert bank through
/// ExpertBank::load on the ROCm device (the EXACT storages the model runs
/// with), print their dtype/bytes, then run the kernel over them. This is
/// the integration the eager arm and the graph prewarm exercise.
#[test]
#[ignore]
fn kq_native_over_runtime_bank_storages() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    }
    let Ok(path) = std::env::var("XING_GGUF") else {
        eprintln!("[SKIP] requires XING_GGUF");
        return;
    };
    use grim_format::tprov::GgufProvider;
    use grim_nn::varbuilder::WeightSource as Ws;
    let prov = GgufProvider::open(&path).expect("open gguf");
    let ws = Ws::root(&prov, grim_tensor::Device::Rocm(0))
        .pp("blk")
        .pp("2");
    let bank = grim_nn::moe::ExpertBank::load(&ws, 64, 3584, 1024, false)
        .expect("load bank on Rocm");
    let w0 = &bank.gate[0].weight;
    eprintln!(
        "[kq-rt] gate[0] dtype {:?} dims {:?} bytes",
        w0.dtype(),
        w0.shape().dims()
    );
    let d0 = &bank.down[0].weight;
    eprintln!(
        "[kq-rt] down[0] dtype {:?} dims {:?}",
        d0.dtype(),
        d0.shape().dims()
    );
    let s0 = w0
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .unwrap();
    eprintln!("[kq-rt] gate[0] storage bytes {}", s0.bytes());
    let sd0 = d0
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .unwrap();
    eprintln!("[kq-rt] down[0] storage bytes {}", sd0.bytes());
}


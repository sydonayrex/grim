//! B6 — ScrubJay fused dequant + GEMV on the GPU, against a CPU oracle.
//!
//! Two things are checked, and they fail for different reasons so they are
//! separate tests:
//!
//!   1. `fused_dequant_gemm_matches_cpu` — the kernel's in-register codebook
//!      decode reproduces the host `quantize_block`/decode exactly. The weight
//!      planes are produced by the *real* quantizer, and the oracle reads those
//!      same planes, so this isolates the kernel rather than re-measuring the
//!      format's error (which is B4's number).
//!
//!   2. `codebook_matches_host_constant` — the table compiled into the HIP
//!      source equals `SCRUB_JAY_CODEBOOK`. This is here because a drifted
//!      table cannot fail loudly: the decode is table-driven, so a wrong table
//!      still returns plausible finite numbers and would only show up as a
//!      mysterious accuracy delta much later.
//!
//! RUN: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!      cargo test -p grim-backend-rocm --test scrub_jay_gpu -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_quant::scrub_jay::{quantize_block, SCRUB_JAY_BLOCK, SCRUB_JAY_CODEBOOKS};
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        (unit - 0.5) * 2.0
    }
}

/// Parse the `GRIM_SCRUB_JAY_CB` initializer back out of the kernel source.
fn kernel_codebook() -> Vec<i8> {
    let src = grim_backend_rocm::kernels::scrub_jay::KERNEL_SOURCE;
    let start = src.find("GRIM_SCRUB_JAY_CB[256] = {").expect("table present in source");
    let body_start = src[start..].find('{').unwrap() + start;
    let body_end = src[body_start..].find('}').unwrap() + body_start;
    // Strip line comments BEFORE splitting on commas. Doing it the other way
    // round loses one entry per row: the text between a row's trailing comma
    // and the next row's first comma is "  // book N\n   0", and cutting at
    // "//" throws away that row's leading value with the comment.
    let body: String = src[body_start + 1..body_end]
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    body.split(',').filter_map(|t| t.trim().parse::<i8>().ok()).collect()
}

#[test]
fn codebook_matches_host_constant() -> TestResult {
    let k = kernel_codebook();
    assert_eq!(
        k.len(),
        SCRUB_JAY_CODEBOOKS * 16,
        "kernel codebook has {} entries, expected {}",
        k.len(),
        SCRUB_JAY_CODEBOOKS * 16
    );
    for (b, book) in grim_quant::scrub_jay::SCRUB_JAY_CODEBOOK.iter().enumerate() {
        for (i, &e) in book.iter().enumerate() {
            assert_eq!(
                k[b * 16 + i],
                e,
                "codebook[{b}][{i}]: kernel has {}, host has {e}",
                k[b * 16 + i]
            );
        }
    }
    Ok(())
}

#[test]
fn fused_dequant_gemm_matches_cpu() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };

    const K: usize = 1024;
    let mut rng = Lcg(0xB6_C0FFEE);
    // Blocks across the whole tensor, so every one of the 16 codebooks gets
    // exercised somewhere in the sweep rather than only the common shapes.
    let pre_scale = 1.0f32 / 14.0;

    for &n in &[64usize, 251, 1024] {
        let a: Vec<f32> = (0..K).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * K).map(|_| rng.next_f32()).collect();

        // --- quantize with the real quantizer -----------------------------
        let n_blocks_per_col = K / SCRUB_JAY_BLOCK;
        let mut sel = vec![0u8; n * n_blocks_per_col];
        let mut idx = vec![0u8; n * K];
        let mut sgn = vec![0u8; n * n_blocks_per_col];
        let mut scl = vec![0f32; n * n_blocks_per_col];
        for col in 0..n {
            for blk in 0..n_blocks_per_col {
                let mut block = [0f32; SCRUB_JAY_BLOCK];
                for (j, slot) in block.iter_mut().enumerate() {
                    *slot = b[col * K + blk * SCRUB_JAY_BLOCK + j];
                }
                let (s, indices, sign_plane, scale) = quantize_block(&block, pre_scale);
                sel[col * n_blocks_per_col + blk] = s;
                sgn[col * n_blocks_per_col + blk] = sign_plane;
                scl[col * n_blocks_per_col + blk] = scale;
                for j in 0..SCRUB_JAY_BLOCK {
                    idx[col * K + blk * SCRUB_JAY_BLOCK + j] = indices[j];
                }
            }
        }

        // --- upload -------------------------------------------------------
        let a_t = up(&dev, f32_bytes(&a), K, DType { arith: ArithType::F32, storage: Storage::Native }, "a")?;
        let sel_t = up(&dev, u8_bytes(&sel), sel.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sel")?;
        let idx_t = up(&dev, u8_bytes(&idx), idx.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "idx")?;
        let sgn_t = up(&dev, u8_bytes(&sgn), sgn.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sgn")?;
        let scl_t = up(&dev, f32_bytes(&scl), scl.len(), DType { arith: ArithType::F32, storage: Storage::Native }, "scl")?;
        let out_t = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![n]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )
        .map_err(|e| format!("out alloc: {e}"))?;

        fn r(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
            grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
        }
        dev.launch_scrub_jay_gemv(r(&a_t), r(&sel_t), r(&idx_t), r(&sgn_t), r(&scl_t), r(&out_t), pre_scale, n, K)
            .map_err(|e| format!("launch: {e}"))?;
        dev.synchronize();
        let got = download_f32(&out_t)?;

        // --- oracle: read the SAME planes, decode on the host --------------
        // Deliberately not the f32 source: that would score the quantizer (B4),
        // not the kernel.
        let cb = grim_quant::scrub_jay::SCRUB_JAY_CODEBOOK;
        let mut worst_rel = 0.0f32;
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..n_blocks_per_col {
                let s = sel[col * n_blocks_per_col + blk] as usize;
                let sign_plane = sgn[col * n_blocks_per_col + blk];
                let scale = scl[col * n_blocks_per_col + blk] * pre_scale;
                for j in 0..SCRUB_JAY_BLOCK {
                    let k = blk * SCRUB_JAY_BLOCK + j;
                    let mut w = cb[s][idx[col * K + k] as usize] as f32;
                    if sign_plane & (1 << j) != 0 {
                        w = -w;
                    }
                    acc += a[k] * (w * scale);
                }
            }
            let want = acc;
            let d = (got[col] - want).abs();
            worst_rel = worst_rel.max(d / want.abs().max(1e-3));
        }

        // The decode is a table lookup and a scale, so the only error source is
        // f32 accumulation order: a 32-lane tree reduce on GPU vs a sequential
        // sum on CPU. 2e-2 is the same f16-accumulation rationale used by
        // wmma_gemm_cpu_gpu_parity.rs and leaves ample headroom.
        assert!(
            worst_rel <= 2e-2,
            "n={n}: worst relative error {worst_rel:.3e} exceeds 2e-2"
        );
        eprintln!("n={n:<5} worst_rel={worst_rel:.3e}");
    }
    Ok(())
}

fn up(
    d: &RocmDevice,
    bytes: &[u8],
    len: usize,
    dt: DType,
    what: &str,
) -> TestResult<Box<dyn BackendStorage>> {
    MemoryOps::from_cpu_bytes(d, bytes, &Shape::new(vec![len]), dt)
        .map_err(|e| format!("{what} h2d: {e}").into())
}
fn u8_bytes(v: &[u8]) -> &[u8] {
    v
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn download_f32(t: &Box<dyn BackendStorage>) -> TestResult<Vec<f32>> {
    let bytes = grim_backend_rocm::as_rocm(t.as_ref()).map_err(|e| e.to_string())?.copy_to_host()?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

//! Compile `grim_tree_pie_gemm` **standalone** and see whether it faults.
//!
//! The dispatched path routes prefill through the aggregate JIT source -- roughly
//! 30k lines concatenating every kernel in the tree. Every hypothesis tried so far
//! (out-of-bounds indexing, wrong grid, stale pointer, undersized activation,
//! the host conversion, a `TP_M_TILE` collision, a stale entry handle) has been
//! eliminated, and the two structural differences between this kernel and the
//! working GEMV are the `acc[TP_M_TILE]` local array and the runtime-bounded
//! `rows` loop over it.
//!
//! `launch_from_source` compiles a caller-supplied string as its own module, so
//! this splits the question in half:
//!
//!   - **works standalone** -> the aggregate source is implicated
//!   - **faults standalone** -> the kernel is at fault, and the suspect is `acc`
//!
//! Only the kernel's own source is compiled here, so "standalone" is literal.

use grim_backend_rocm::RocmDevice;
use grim_quant::tree_pie::TREE_PIE_WORDS_PER_32;
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const WAVE: usize = 32;

/// `grim_fdot2_f32_f16` lives in dot_gemv's source, not TreePie's -- the aggregate
/// JIT source is a concatenation and TreePie's file has always leaned on it. That
/// dependency is itself worth knowing: compiling TreePie standalone fails until the
/// helper is supplied, so "standalone" is only standalone once it is.
///
/// Copied verbatim from `kernels/dot_gemv.rs` so the arithmetic is byte-identical
/// to the dispatched path; a second implementation would make this test measure a
/// different kernel than the one in the tree.
const DOT2_HELPER: &str = r#"
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_fdot2)
__device__ __forceinline__ float grim_fdot2_f32_f16(unsigned a, unsigned b, float c) {
    return __builtin_amdgcn_fdot2((__attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16)a,
                                  (__attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16)b,
                                  c, false);
}
#else
__device__ __forceinline__ float grim_fdot2_f32_f16(unsigned a, unsigned b, float c) {
    float res = c;
    asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(res) : "v"(a), "v"(b));
    return res;
}
#endif
"#;


/// Control: the GEMV, through the *same* standalone harness.
///
/// Without this, a fault from the GEMM says nothing -- it would be equally
/// consistent with a broken harness. The GEMV is known-good through the dispatched
/// aggregate path, so if it also faults here the harness is at fault, not the GEMM.
#[test]
fn tree_pie_gemv_standalone_control() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let n = 64usize;
    let k = 128usize;

    let mut s = 0x243f_6a88_5a30_1234u64;
    let mut next = move || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let w: Vec<f32> = (0..k * n).map(|_| next() * 0.25).collect();
    let x: Vec<f32> = (0..k).map(|_| next() * 0.5).collect();

    let act_bits: Vec<u8> = x.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes().to_vec())
        .collect();
    let act_t = MemoryOps::from_cpu_bytes(dev, &act_bits, &Shape::new(vec![k]),
        DType { arith: ArithType::F16, storage: Storage::Native })
        .map_err(|e| format!("act: {e}"))?;

    let mut packed = vec![0i32; n * (k / 32) * TREE_PIE_WORDS_PER_32];
    for col in 0..n {
        for g in 0..k / 32 {
            let mut block = [0f32; 32];
            for j in 0..32 { block[j] = w[(g * 32 + j) * n + col]; }
            let words = grim_quant::tree_pie::pack_tree_pie_32(&block);
            let base = (col * (k / 32) + g) * TREE_PIE_WORDS_PER_32;
            packed[base..base + TREE_PIE_WORDS_PER_32].copy_from_slice(&words);
        }
    }
    let b_bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let b_t = MemoryOps::from_cpu_bytes(dev, &b_bytes,
        &Shape::new(vec![n * (k / 32) * TREE_PIE_WORDS_PER_32]),
        DType { arith: ArithType::U32, storage: Storage::Native })
        .map_err(|e| format!("b: {e}"))?;
    let out_t = MemoryOps::alloc_storage(dev, &Shape::new(vec![n]),
        DType { arith: ArithType::F32, storage: Storage::Native })
        .map_err(|e| format!("out: {e}"))?;

    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr = |t: &Box<dyn grim_tensor::BackendStorage>| -> *mut std::ffi::c_void {
        raw(t).device_ptr_u64().expect("device ptr") as *mut std::ffi::c_void
    };
    let (a, b, o) = (ptr(&act_t), ptr(&b_t), ptr(&out_t));
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, n as i32, k as i32, o);

    let src = format!("{DOT2_HELPER}\n{}", grim_backend_rocm::kernels::tree_pie::KERNEL_SOURCE);
    dev.launch_from_source(
        &src,
        "grim_tree_pie_gemv",
        grim_backend_rocm::HipDim3::new(n as u32, 1, 1),
        grim_backend_rocm::HipDim3::new(WAVE as u32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("standalone gemv: {e}"))?;
    dev.synchronize();

    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f64> = out.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect();
    let oracle: f64 = (0..n).map(|col| (0..k).map(|kk| x[kk] as f64 * w[kk * n + col] as f64).sum::<f64>()).sum();
    println!("GEMV standalone: {}/{} outputs nonzero, mean-oracle {oracle:.4}",
             got.iter().filter(|v| **v != 0.0).count(), got.len());
    assert!(got.iter().any(|v| *v != 0.0), "GEMV control produced all zeros");
    Ok(())
}

#[test]
fn tree_pie_gemm_standalone_source() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let m = 2usize;
    let n = 64usize;
    let k = 128usize;

    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let w: Vec<f32> = (0..k * n).map(|_| next() * 0.25).collect();
    let x: Vec<f32> = (0..m * k).map(|_| next() * 0.5).collect();

    // Activations as f16, so nothing about this test depends on the dispatch's
    // conversion path -- it is bypassed entirely.
    let act_bits: Vec<u8> = x.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes().to_vec())
        .collect();
    let act_t = MemoryOps::from_cpu_bytes(
        dev,
        &act_bits,
        &Shape::new(vec![m * k]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;

    // B in the kernel's per-column layout.
    let mut packed = vec![0i32; n * (k / 32) * TREE_PIE_WORDS_PER_32];
    for col in 0..n {
        for g in 0..k / 32 {
            let mut block = [0f32; 32];
            for j in 0..32 {
                block[j] = w[(g * 32 + j) * n + col];
            }
            let words = grim_quant::tree_pie::pack_tree_pie_32(&block);
            let base = (col * (k / 32) + g) * TREE_PIE_WORDS_PER_32;
            packed[base..base + TREE_PIE_WORDS_PER_32].copy_from_slice(&words);
        }
    }
    let b_bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let b_t = MemoryOps::from_cpu_bytes(
        dev,
        &b_bytes,
        &Shape::new(vec![n * (k / 32) * TREE_PIE_WORDS_PER_32]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b: {e}"))?;

    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;

    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }

    let ptr = |t: &Box<dyn grim_tensor::BackendStorage>| -> *mut std::ffi::c_void {
        raw(t).device_ptr_u64().expect("device ptr") as *mut std::ffi::c_void
    };
    let (a, b, o) = (ptr(&act_t), ptr(&b_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut o) = (a, b, m as i32, n as i32, k as i32, o);

    let src = format!("{DOT2_HELPER}\n{}", grim_backend_rocm::kernels::tree_pie::KERNEL_SOURCE);
    dev.launch_from_source(
        &src,
        "grim_tree_pie_gemm",
        grim_backend_rocm::HipDim3::new(n as u32, 1, 1),
        grim_backend_rocm::HipDim3::new(WAVE as u32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("standalone launch: {e}"))?;
    dev.synchronize();

    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f64> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();

    let mut oracle = vec![0.0f64; m * n];
    let mut scale = vec![0.0f64; m * n];
    for r in 0..m {
        for col in 0..n {
            for kk in 0..k {
                let t = x[r * k + kk] as f64 * w[kk * n + col] as f64;
                oracle[r * n + col] += t;
                scale[r * n + col] += t.abs();
            }
        }
    }
    let worst = (0..m * n)
        .map(|i| ((got[i] - oracle[i]) / scale[i].max(f64::MIN_POSITIVE)).abs())
        .fold(0.0f64, f64::max);
    let nonzero = got.iter().filter(|v| **v != 0.0).count();

    println!("standalone grim_tree_pie_gemm: m={m} n={n} k={k}");
    println!("  {nonzero} of {} outputs nonzero", m * n);
    println!("  worst relative error {worst:.3e}");
    assert!(nonzero > 0, "standalone GEMM produced all zeros");
    assert!(worst < 0.05, "standalone GEMM disagrees: worst relative error {worst:.3e}");
    Ok(())
}

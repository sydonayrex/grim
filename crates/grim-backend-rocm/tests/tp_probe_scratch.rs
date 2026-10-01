//! Scratch fault-bisect probes for the TreePie prefill fault. DELETE BEFORE LANDING.
//! P0 isolates launch/harness (empty kernel, same 6-arg shape) from kernel body.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const PROBE_SRC: &str = r#"
#if defined(__AMDGCN__) || defined(__HIP__)
extern "C" __global__ void tp_probe_empty(
    const unsigned short* __restrict__ a,
    const int* __restrict__ b,
    int m, int n, int k,
    float* __restrict__ c)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 0) c[0] = 1.0f;
}
// Same module, full-grid write: isolates grid size from body.
extern "C" __global__ void tp_probe_grid64(
    const unsigned short* __restrict__ a,
    const int* __restrict__ b,
    int m, int n, int k,
    float* __restrict__ c)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 2048) c[i] = (float)i;
}
// Byte-identical copy of tp_probe_cwrite, but inside the working module.
extern "C" __global__ void tp_probe_cwrite_clone(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    if (lane == 0) C[col] = (float)col;
}
// Flat-index write, same shape otherwise: separates body pattern from allocs.
extern "C" __global__ void tp_probe_flat64(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 64 * 32) C[(i >> 5)] = (float)(i >> 5);
}
// Single-index writes: is C[1] writable in an F32[64] alloc?
extern "C" __global__ void tp_probe_c1(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 1) C[1] = 41.0f;
}
// P0-shape control in the CURRENT module: C[0]-only write, grid 64.
extern "C" __global__ void tp_probe_c0(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 0) C[0] = 7.0f;
}
// 2x2 discriminator: 6 params (like empty) but C[1] write (like c1).
extern "C" __global__ void tp_probe_c1six(
    const unsigned short* __restrict__ a,
    const int* __restrict__ b,
    int m, int n, int k,
    float* __restrict__ c)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 1) c[1] = 41.0f;
}
// Param-count sweep: 4-param and 7-param kernels, same C[1] write.
extern "C" __global__ void tp_probe_4p(
    const unsigned short* __restrict__ a,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 1) C[1] = 41.0f;
}
extern "C" __global__ void tp_probe_7p(
    const unsigned short* __restrict__ a,
    const int* __restrict__ b,
    int m, int n, int k, int pad1, int pad2,
    float* __restrict__ c)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 1) c[1] = (float)(pad1 + pad2);
}
#endif
"#;

const GEMV_BISECT_SRC: &str = r#"
#if defined(__AMDGCN__) || defined(__HIP__)
// V1: C-write only, full grid. Tests grid + C allocation.
extern "C" __global__ void tp_probe_cwrite(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    if (lane == 0) C[col] = (float)col;
}
#endif
"#;

const GEMV_BISECT_REST: &str = r#"
#if defined(__AMDGCN__) || defined(__HIP__)
// V2: + B_tree row read. Tests B allocation/size.
extern "C" __global__ void tp_probe_bread(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    const int groups = K / 32;
    const int row_words = groups * 5;
    const int* __restrict__ row = B_tree + (long long)col * row_words;
    float s = 0.0f;
    for (int g = lane; g < groups; g += 32) {
        const int* __restrict__ gp = row + (long long)g * 5;
        s += (float)(gp[0] + gp[1] + gp[2] + gp[3] + gp[4]);
    }
    if (lane == 0) C[col] = s;
}
// V3: + act read + fdot2 + shfl (full GEMV body shape, decode bypassed with w01=0x3c003c00).
extern "C" __global__ void tp_probe_fullshape(
    const unsigned short* __restrict__ act_f16,
    const int* __restrict__ B_tree,
    float* __restrict__ C,
    int N, int K)
{
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    const int groups = K / 32;
    const int* __restrict__ row = B_tree + (long long)col * groups * 5;
    const unsigned* act2 = (const unsigned*)act_f16;
    float facc = 0.0f;
    for (int g = lane; g < groups; g += 32) {
        const int* __restrict__ gp = row + (long long)g * 5;
        unsigned payload[4] = { (unsigned)gp[0], (unsigned)gp[1],
                                (unsigned)gp[2], (unsigned)gp[3] };
        unsigned signs = (unsigned)gp[4];
        (void)payload; (void)signs;
        float bacc = 0.0f;
        for (int j = 0; j < 32; j += 2) {
            unsigned w01 = 0x3c003c00u;
            unsigned a01 = act2[(g * 32 + j) >> 1];
            float aw = __uint_as_float(a01 & 0xffff);
            float aw2 = __uint_as_float(a01 >> 16);
            float bw = __uint_as_float(w01 & 0xffff);
            float bw2 = __uint_as_float(w01 >> 16);
            bacc += aw * bw + aw2 * bw2;
        }
        facc += bacc;
    }
    for (int off = 16; off > 0; off >>= 1)
        facc += __shfl_xor(facc, off);
    if (lane == 0) C[col] = facc;
}
#endif
"#;

#[test]
fn tp_probe_empty_kernel_via_harness() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut o) = (a, b, 2i32, 64i32, 128i32, o);
    eprintln!("[probe] a={a:?} b={b:?} o={o:?}");
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_empty",
        grim_backend_rocm::HipDim3::new(2, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("probe launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    eprintln!("[probe] C[0]={v}");
    assert_eq!(v, 1.0, "empty probe must write 1.0");
    Ok(())
}

#[test]
fn tp_probe_grid64() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    // 2048 f32 out; grid 64 x block 32; every thread writes one element.
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![2048usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut o) = (a, b, 2i32, 64i32, 128i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_grid64",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("grid64 launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 2048);
    assert!((got[2047] - 2047.0).abs() < 1e-6, "C[2047]={}", got[2047]);
    Ok(())
}

#[test]
fn tp_probe_cwrite_clone() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    // Same 5-arg launch shape as cwrite, but the entry lives in PROBE_SRC.
    let n = 64usize;
    let k = 128usize;
    let act_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![k]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    let b_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n * (k / 32) * 5]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&b_t), ptr(&out_t));
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, n as i32, k as i32, o);
    // E_b: drop the U32 B alloc, pass null (kernel never reads it).
    let _keep_b_alive = &b_t;
    let mut bnull: *mut std::ffi::c_void = std::ptr::null_mut();
    let _ = &mut b;
    // E_a: 6 args for a 5-param kernel (extra trailing dummy, must be ignored).
    let mut dummy = 0i32;
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_cwrite_clone",
        grim_backend_rocm::HipDim3::new(n as u32, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut bnull),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
            grim_backend_rocm::device::util::arg(&mut dummy),
        ],
    )
    .map_err(|e| format!("clone launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[63] - 63.0).abs() < 1e-6, "C[63]={} want 63", got[63]);
    Ok(())
}

#[test]
fn tp_probe_flat64() -> TestResult {
    // Same module/allocs/args as the faulting clone, flat write pattern.
    let Some(dev) = gpu_device() else { return Ok(()) };
    let n = 64usize;
    let k = 128usize;
    let act_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![k]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    let b_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n * (k / 32) * 5]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&b_t), ptr(&out_t));
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, n as i32, k as i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_flat64",
        grim_backend_rocm::HipDim3::new(n as u32, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("flat64 launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[63] - 63.0).abs() < 1e-6, "C[63]={} want 63", got[63]);
    Ok(())
}

#[test]
fn tp_probe_c1() -> TestResult {
    // C[1]-only write into F32[64]: is anything past C[0] mapped?
    let Some(dev) = gpu_device() else { return Ok(()) };
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, 64i32, 128i32, o);
    let mut dummy = 0i32;
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_c1",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
            grim_backend_rocm::device::util::arg(&mut dummy),
        ],
    )
    .map_err(|e| format!("c1 launch: {e}"))?;
    eprintln!("[c1] launched");
    dev.synchronize();
    eprintln!("[c1] synced");
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    eprintln!("[c1] copied {} bytes", out.len());
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[1] - 41.0).abs() < 1e-6, "C[1]={} want 41", got[1]);
    Ok(())
}

#[test]
fn tp_probe_c0() -> TestResult {
    // P0 shape in the current module: C[0]-only write, grid 64, same allocs.
    let Some(dev) = gpu_device() else { return Ok(()) };
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, 64i32, 128i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_c0",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("c0 launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[0] - 7.0).abs() < 1e-6, "C[0]={} want 7", got[0]);
    Ok(())
}

#[test]
fn tp_probe_empty_grid64() -> TestResult {
    // P0 entry, grid 64, C[64]: separates grid x C-size from body/module.
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut o) = (a, b, 2i32, 64i32, 128i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_empty",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("empty64 launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[0] - 1.0).abs() < 1e-6, "C[0]={} want 1", got[0]);
    Ok(())
}

#[test]
fn tp_probe_c1six() -> TestResult {
    // 6-param kernel writing C[1]: completes the 2x2 (params x index).
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut o) = (a, b, 2i32, 64i32, 128i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_c1six",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("c1six launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[1] - 41.0).abs() < 1e-6, "C[1]={} want 41", got[1]);
    Ok(())
}

#[test]
fn tp_probe_4p() -> TestResult {
    // 4 params (8+8+8+4 = 28B user): does the fault follow small counts?
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut o, mut nn) = (a, b, o, 64i32);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_4p",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut o),
            grim_backend_rocm::device::util::arg(&mut nn),
        ],
    )
    .map_err(|e| format!("4p launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[1] - 41.0).abs() < 1e-6, "C[1]={} want 41", got[1]);
    Ok(())
}

#[test]
fn tp_probe_7p() -> TestResult {
    // 7 params (8+8+4*4+8 = 40B user).
    let Some(dev) = gpu_device() else { return Ok(()) };
    let dev = &dev;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&act_t), ptr(&out_t));
    let (mut a, mut b, mut mm, mut nn, mut kk, mut p1, mut p2, mut o) =
        (a, b, 2i32, 64i32, 128i32, 20i32, 22i32, o);
    dev.launch_from_source(
        PROBE_SRC,
        "tp_probe_7p",
        grim_backend_rocm::HipDim3::new(64, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut mm),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut p1),
            grim_backend_rocm::device::util::arg(&mut p2),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("7p launch: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), 64);
    assert!((got[1] - 42.0).abs() < 1e-6, "C[1]={} want 42", got[1]);
    Ok(())
}

fn launch_bisect(dev: &RocmDevice, src: &str, entry: &str) -> TestResult<Vec<f32>> {
    // Guard-theory probe: what device is current at launch time?
    let mut cur: i32 = -99;
    unsafe { grim_backend_rocm::device::handles::hipGetDevice(&mut cur); }
    eprintln!("[bisect:{entry}] current-device={cur}");
    let n = 64usize;
    let k = 128usize;
    let act_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![64usize]),
        DType { arith: ArithType::F16, storage: Storage::Native },
    )
    .map_err(|e| format!("act: {e}"))?;
    // E_c: add an unused U32 alloc to the WORKING grid64 config.
    let b_extra = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![1280usize]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b_extra: {e}"))?;
    let _keep = &b_extra;
    let b_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![n * (k / 32) * 5]),
        DType { arith: ArithType::U32, storage: Storage::Native },
    )
    .map_err(|e| format!("b: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out: {e}"))?;
    fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ptr =
        |t: &Box<dyn grim_tensor::BackendStorage>| raw(t).device_ptr_u64().expect("ptr") as *mut std::ffi::c_void;
    let (a, b, o) = (ptr(&act_t), ptr(&b_t), ptr(&out_t));
    eprintln!("[bisect:{entry}] a={a:?} b={b:?} o={o:?}");
    let (mut a, mut b, mut nn, mut kk, mut o) = (a, b, n as i32, k as i32, o);
    dev.launch_from_source(
        src,
        entry,
        grim_backend_rocm::HipDim3::new(n as u32, 1, 1),
        grim_backend_rocm::HipDim3::new(32, 1, 1),
        &mut [
            grim_backend_rocm::device::util::arg(&mut a),
            grim_backend_rocm::device::util::arg(&mut b),
            grim_backend_rocm::device::util::arg(&mut nn),
            grim_backend_rocm::device::util::arg(&mut kk),
            grim_backend_rocm::device::util::arg(&mut o),
        ],
    )
    .map_err(|e| format!("{entry}: {e}"))?;
    dev.synchronize();
    let out = raw(&out_t).copy_to_host().map_err(|e| e.to_string())?;
    Ok(out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn tp_probe_cwrite() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let got = launch_bisect(&dev, GEMV_BISECT_SRC, "tp_probe_cwrite")?;
    assert_eq!(got.len(), 64);
    assert!((got[63] - 63.0).abs() < 1e-6, "C[63]={} want 63", got[63]);
    Ok(())
}

#[test]
fn tp_probe_bread() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let got = launch_bisect(&dev, GEMV_BISECT_REST, "tp_probe_bread")?;
    assert_eq!(got.len(), 64);
    Ok(())
}

#[test]
fn tp_probe_fullshape() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()) };
    let got = launch_bisect(&dev, GEMV_BISECT_REST, "tp_probe_fullshape")?;
    assert_eq!(got.len(), 64);
    Ok(())
}

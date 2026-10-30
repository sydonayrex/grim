//! End-to-end HIP gate for grim-hip-sys: hiprtc-compile a `y = a*x + b`
//! kernel, load it, launch it on a stream, and verify the result — exercising
//! the runtime API, hiprtc, module loading, memory, streams, and memcpy
//! through the vendored bindings in BOTH linkage arms (static default and
//! `dynamic`). Ported from cubecl-hip-sys's lib.rs test.
//!
//! Run on the non-reserved card:
//! ```bash
//! HIP_VISIBLE_DEVICES=1 cargo test -p grim-hip-sys -- --ignored
//! HIP_VISIBLE_DEVICES=1 cargo test -p grim-hip-sys --features dynamic -- --ignored
//! ```

#![allow(non_snake_case)]

use grim_hip_sys as ffi;
use std::ffi::CString;
use std::ptr;
use std::time::Instant;

#[test]
#[ignore]
fn hip_launch_kernel_end_to_end() {
    if !ffi::is_available() {
        panic!("HIP runtime unavailable; run on a box with ROCm installed");
    }

    let mut device_count: i32 = 0;
    // SAFETY: `device_count` is a valid pointer to writable storage.
    let status = unsafe { ffi::hipGetDeviceCount(&mut device_count) };
    assert_eq!(status, 0, "hipGetDeviceCount failed");
    assert!(device_count > 0, "no HIP devices visible");

    let source = CString::new(
        r#"
extern "C" __global__ void kernel(float a, float *x, float *b, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = x[tid] * a + b[tid];
  }
}
 "#,
    )
    .expect("Should construct kernel string");
    let func_name = CString::new("kernel").expect("kernel name");

    // Select the device. Under HIP_VISIBLE_DEVICES=1, ordinal 0 is the
    // non-reserved card; GRIM_TEST_GPU overrides for local runs.
    let ordinal: i32 = std::env::var("GRIM_TEST_GPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    unsafe {
        let status = ffi::hipSetDevice(ordinal);
        assert_eq!(status, 0, "hipSetDevice({ordinal}) failed");
    }

    // Step 1: create + compile the program
    let mut program: ffi::hiprtcProgram = ptr::null_mut();
    unsafe {
        let status = ffi::hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
        );
        assert_eq!(status, ffi::hiprtcResult_HIPRTC_SUCCESS, "create program");
    }
    unsafe {
        let status = ffi::hiprtcCompileProgram(program, 0, ptr::null());
        if status != ffi::hiprtcResult_HIPRTC_SUCCESS {
            let mut log_size: usize = 0;
            ffi::hiprtcGetProgramLogSize(program, &mut log_size);
            let mut log = vec![0i8; log_size];
            ffi::hiprtcGetProgramLog(program, log.as_mut_ptr());
            let log_bytes: Vec<u8> = log.iter().map(|b| *b as u8).collect();
            panic!(
                "hiprtcCompileProgram failed: {}",
                String::from_utf8_lossy(&log_bytes)
            );
        }
    }

    // Step 2: load the compiled code
    let mut code_size: usize = 0;
    unsafe {
        assert_eq!(
            ffi::hiprtcGetCodeSize(program, &mut code_size),
            ffi::hiprtcResult_HIPRTC_SUCCESS
        );
    }
    let mut code = vec![0u8; code_size];
    unsafe {
        assert_eq!(
            ffi::hiprtcGetCode(program, code.as_mut_ptr() as *mut _),
            ffi::hiprtcResult_HIPRTC_SUCCESS
        );
        assert_eq!(
            ffi::hiprtcDestroyProgram(&mut program),
            ffi::hiprtcResult_HIPRTC_SUCCESS
        );
    }
    assert!(!code.is_empty(), "compiled code is empty");

    // Step 3: allocate + upload
    let n = 1024usize;
    let a = 2.0f32;
    let x: Vec<f32> = (0..n as i32).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n as i32).map(|i| (n as i32 - i) as f32).collect();
    let mut out = vec![0.0f32; n];
    let mut device_x = ptr::null_mut();
    let mut device_b = ptr::null_mut();
    let mut device_out = ptr::null_mut();
    unsafe {
        assert_eq!(ffi::hipMalloc(&mut device_x, n * 4), 0, "malloc x");
        assert_eq!(ffi::hipMalloc(&mut device_b, n * 4), 0, "malloc b");
        assert_eq!(ffi::hipMalloc(&mut device_out, n * 4), 0, "malloc out");
        assert_eq!(
            ffi::hipMemcpy(
                device_x,
                x.as_ptr() as *const _,
                n * 4,
                ffi::hipMemcpyKind_hipMemcpyHostToDevice
            ),
            0
        );
        assert_eq!(
            ffi::hipMemcpy(
                device_b,
                b.as_ptr() as *const _,
                n * 4,
                ffi::hipMemcpyKind_hipMemcpyHostToDevice
            ),
            0
        );
    }

    // Step 4: module + function
    let mut module: ffi::hipModule_t = ptr::null_mut();
    let mut function: ffi::hipFunction_t = ptr::null_mut();
    unsafe {
        assert_eq!(
            ffi::hipModuleLoadData(&mut module, code.as_ptr() as *const _),
            0,
            "hipModuleLoadData"
        );
        assert_eq!(
            ffi::hipModuleGetFunction(&mut function, module, func_name.as_ptr()),
            0,
            "hipModuleGetFunction"
        );
    }

    // Step 5: launch on an explicitly created stream
    let mut args: [*mut std::os::raw::c_void; 5] = [
        &a as *const _ as *mut _,
        &device_x as *const _ as *mut _,
        &device_b as *const _ as *mut _,
        &device_out as *const _ as *mut _,
        &(n as i32) as *const _ as *mut _,
    ];
    let mut stream = ptr::null_mut();
    let start = Instant::now();
    unsafe {
        assert_eq!(ffi::hipStreamCreate(&mut stream), 0, "hipStreamCreate");
        assert_eq!(
            ffi::hipModuleLaunchKernel(
                function,
                n as u32 / 64,
                1,
                1,
                64,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            0,
            "hipModuleLaunchKernel"
        );
        assert_eq!(ffi::hipDeviceSynchronize(), 0, "hipDeviceSynchronize");
    }
    println!("execution: {}µs", start.elapsed().as_micros());

    // Step 6: download + verify
    unsafe {
        assert_eq!(
            ffi::hipMemcpy(
                out.as_mut_ptr() as *mut _,
                device_out,
                n * 4,
                ffi::hipMemcpyKind_hipMemcpyDeviceToHost
            ),
            0
        );
    }
    for i in 0..n {
        assert_eq!(out[i], a * x[i] + b[i], "mismatch at {i}");
    }

    // Step 7: teardown
    unsafe {
        assert_eq!(ffi::hipStreamDestroy(stream), 0);
        assert_eq!(ffi::hipFree(device_x), 0);
        assert_eq!(ffi::hipFree(device_b), 0);
        assert_eq!(ffi::hipFree(device_out), 0);
        assert_eq!(ffi::hipModuleUnload(module), 0);
    }
}

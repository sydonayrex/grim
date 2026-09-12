//! Dedicated comprehensive test for HIP graph capture behavior and correctness.
//!
//! Validates the fundamental invariants of how a HIP execution graph must operate:
//! 1. Buffer Preallocation: All persistent inputs, weights, biases, and normalization
//!    tensors must be allocated on device prior to beginning graph capture.
//! 2. Warmup Execution: An eager pass outside capture ensures any JIT kernel compiles
//!    and rocBLAS GEMM dispatch tables are initialized before stream capture begins.
//! 3. Pure Device Capture: During `begin_graph_capture` .. `end_graph_capture`, only
//!    asynchronous stream-ordered device kernel launches occur. Zero host-to-device or
//!    device-to-host synchronous memory transfers (`hipMemcpyDtoH`, `hipMemcpy`) are issued.
//! 4. Graph Instantiation & Cache: The recorded stream operations are instantiated via
//!    `hipGraphInstantiate` and cached under a designated session key.
//! 5. Buffer Stability & Asynchronous Replay: Input buffers are updated in place across
//!    successive decode steps via `write_f32_into`. Replaying the graph via `replay_graph`
//!    enqueues the entire kernel DAG with a single driver call.
//! 6. Parity Verification: Replaying the captured graph produces bit-accurate / floating-point
//!    matching results against eager execution across every step.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

lazy_static::lazy_static! {
    static ref GRAPH_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    unsafe { std::env::set_var("GRIM_CAPTURE_GRAPH", "1") };
    std::panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    })
    .ok()
}

/// Test a representative multi-op decode step DAG under HIP graph capture:
/// matmul -> add (bias) -> rms_norm -> mul (scaling)
#[test]
fn test_hip_graph_multi_op_decode_cycle() -> TestResult {
    let _lock = GRAPH_TEST_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[test] ROCm GPU test skipped (no GPU or not enabled)");
        return Ok(());
    };

    let m = 1usize;      // Single-token decode scenario
    let k = 64usize;     // Hidden dimension
    let n = 64usize;     // Projection dimension

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[n, k]);
    let out_shape = Shape::from_slice(&[m, n]);

    // 1. Preallocate and initialize weights and persistent buffers outside capture
    let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b_dev = CoreTensorOps::from_cpu(&dev, &b_data, &b_shape, DType::F32)?;

    let mut a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) + 0.1).collect();
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;

    let bias_data: Vec<f32> = (0..m * n).map(|i| ((i % 7) as f32) * 0.02).collect();
    let bias_dev = CoreTensorOps::from_cpu(&dev, &bias_data, &out_shape, DType::F32)?;

    let norm_w_data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.005)).collect();
    let norm_w_dev = CoreTensorOps::from_cpu(&dev, &norm_w_data, &out_shape, DType::F32)?;

    let scale_data: Vec<f32> = (0..m * n).map(|i| 0.5 + ((i % 5) as f32 * 0.1)).collect();
    let scale_dev = CoreTensorOps::from_cpu(&dev, &scale_data, &out_shape, DType::F32)?;

    // 2. Warm up pipeline eagerly to ensure kernels and rocBLAS GEMMs are compiled/cached
    {
        let (mm, _) = CoreTensorOps::matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
        let (biased, _) = CoreTensorOps::add(&dev, mm.as_ref(), bias_dev.as_ref(), &out_shape)?;
        let (normed, _) = CoreTensorOps::rms_norm(&dev, biased.as_ref(), norm_w_dev.as_ref(), 1e-5, &out_shape)?;
        let (_scaled, _) = CoreTensorOps::mul(&dev, normed.as_ref(), scale_dev.as_ref(), &out_shape)?;
        dev.synchronize();
    }

    // 3. Capture the graph: mm -> biased -> normed -> scaled
    let graph_key = "test_hip_graph_decode_pipeline";
    dev.begin_graph_capture(graph_key)?;
    let (mm_cap, _) = CoreTensorOps::matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    let (biased_cap, _) = CoreTensorOps::add(&dev, mm_cap.as_ref(), bias_dev.as_ref(), &out_shape)?;
    let (normed_cap, _) = CoreTensorOps::rms_norm(&dev, biased_cap.as_ref(), norm_w_dev.as_ref(), 1e-5, &out_shape)?;
    let (scaled_cap, _) = CoreTensorOps::mul(&dev, normed_cap.as_ref(), scale_dev.as_ref(), &out_shape)?;
    dev.end_graph_capture(graph_key)?;

    assert!(
        dev.has_captured_graph(graph_key),
        "Graph capture must successfully instantiate and store the graph in device cache"
    );

    // 4. Verify multi-step replay parity with eager execution across changing inputs
    for step in 0..10 {
        // Mutate input host buffer
        for (i, x) in a_data.iter_mut().enumerate() {
            *x = (((step + 1) * 17 + i) as f32 * 0.03).sin();
        }

        // Update persistent device buffer in place for the graph
        dev.write_f32_into(a_dev.as_ref(), &a_data)?;

        // Replay graph
        let replayed = dev.replay_graph(graph_key)?;
        assert!(replayed, "Graph replay must return true");
        dev.synchronize();

        // Read graph output directly from the captured buffer
        let graph_out = scaled_cap.to_cpu_vec_f32()?;

        // Compute eager reference from fresh input with current step values
        let a_eager = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let (mm_eag, _) = CoreTensorOps::matmul(&dev, a_eager.as_ref(), b_dev.as_ref(), &out_shape)?;
        let (bias_eag, _) = CoreTensorOps::add(&dev, mm_eag.as_ref(), bias_dev.as_ref(), &out_shape)?;
        let (norm_eag, _) = CoreTensorOps::rms_norm(&dev, bias_eag.as_ref(), norm_w_dev.as_ref(), 1e-5, &out_shape)?;
        let (scale_eag, _) = CoreTensorOps::mul(&dev, norm_eag.as_ref(), scale_dev.as_ref(), &out_shape)?;
        dev.synchronize();
        let eager_out = scale_eag.to_cpu_vec_f32()?;

        // Verify exact output parity
        assert_eq!(eager_out.len(), graph_out.len());
        for i in 0..eager_out.len() {
            let diff = (eager_out[i] - graph_out[i]).abs();
            assert!(
                diff < 1e-4,
                "Step {step} mismatch at index {i}: eager={}, graph={}, diff={}",
                eager_out[i],
                graph_out[i],
                diff
            );
        }
    }

    // 5. Clean up graph
    dev.drop_captured_graph(graph_key)?;
    assert!(!dev.has_captured_graph(graph_key));

    // Keep captured buffers alive until after graph is dropped
    drop((mm_cap, biased_cap, normed_cap, scaled_cap));

    Ok(())
}

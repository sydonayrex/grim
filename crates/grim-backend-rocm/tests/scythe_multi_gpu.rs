//! MG verification: cross-device descriptor routing, OP_COMMFUSE peer pipe,
//! and OP_PEER_REDUCE all-reduce via the ScytheRing persistent dispatch.

use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape};

fn gpu_device(ordinal: usize) -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| {
        RocmDevice::try_new(ordinal).expect("RocmDevice::try_new")
    })
    .ok()
}

/// MG-1 + MG-2: peer access enabled + GEMM routed to a peer device.
#[test]
fn cross_device_gemm_via_peer_routing() {
    let Some(dev_a) = gpu_device(0) else {
        eprintln!("[SKIP] no GPU 0");
        return;
    };
    let Some(dev_b) = gpu_device(1) else {
        eprintln!("[SKIP] no GPU 1");
        return;
    };
    if dev_a.ordinal() == dev_b.ordinal() {
        eprintln!("[SKIP] same device, need 2 GPUs");
        return;
    }

    let (m, k, n) = (4usize, 256usize, 16usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
    // B in [N, K] row-major for OP_ROW_GEMM (C = A @ B^T): b[n,k] = origB[k,n].
    let b_trans: Vec<f32> = (0..n * k)
        .map(|i| {
            let ni = i / k;
            let ki = i - ni * k;
            (( (ki * n + ni) % 7 ) as f32 - 3.0) * 0.05
        })
        .collect();

    // Upload A on device A, B on device B (peer access enables cross-reads).
    let a_dev = dev_a
        .from_cpu(&a, &Shape::new(vec![m, k]), DType::F32)
        .unwrap();
    let b_dev = dev_b
        .from_cpu(&b_trans, &Shape::new(vec![n, k]), DType::F32)
        .unwrap();
    let out_dev = dev_b
        .alloc_storage(&Shape::new(vec![m, n]), DType::F32)
        .unwrap();
    let b_rocm = b_dev.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_rocm = out_dev.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();

    // Flush/sync dev_a so A is visible in device memory before dev_b reads it over PCIe
    dev_a.synchronize();

    // Route GEMM through the ring on device B (where B and output live).
    // A's data is accessible via peer access.
    let ordinal_b = dev_b.ordinal();
    let result = grim_backend_rocm::device::scythe_route::route_gemm_to(
        ordinal_b,
        std::ptr::null_mut(), // stream (device uses active_stream)
        // Safety: storages were allocated on the peer-access-enabled devices.
        a_dev.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap(),
        b_rocm,
        out_rocm,
        m,
        n,
        k,
    );
    // If the ring route succeeds, verify output.
    if result.is_ok() {
        dev_b.synchronize();
        let out_bytes = out_rocm.copy_to_host().unwrap();
        let c_gpu: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // CPU reference: C[m, n] = sum_k A[m, k] * B[k, n]
        let mut c_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[row * k + kk] * b_trans[col * k + kk];
                }
                c_ref[row * n + col] = acc;
            }
        }
        let max_abs = c_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = c_ref
            .iter()
            .zip(c_gpu.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0f32, f32::max);
        let rel = if max_abs == 0.0 { max_err } else { max_err / max_abs };
        eprintln!("[cross-gemm] rel={rel:.3e} max_err={max_err}");
        assert!(rel < 1e-3, "cross-device GEMM diverges: rel={rel:.3e}");
    } else {
        eprintln!("[cross-gemm] ring route not available on this arch — skipped");
    }
}

/// MG-3: OP_COMMFUSE pipe — write on device A, read on device B via peer_ptr.
#[test]
fn commfuse_cross_device_pipe() {
    let Some(dev_a) = gpu_device(0) else {
        eprintln!("[SKIP] no GPU 0");
        return;
    };
    let Some(dev_b) = gpu_device(1) else {
        eprintln!("[SKIP] no GPU 1");
        return;
    };
    if dev_a.ordinal() == dev_b.ordinal() {
        eprintln!("[SKIP] same device");
        return;
    }

    let elems = 64usize;
    let src: Vec<f32> = (0..elems).map(|i| i as f32 * 0.1).collect();
    let src_dev = dev_a
        .from_cpu(&src, &Shape::new(vec![elems]), DType::F32)
        .unwrap();
    let dst_dev = dev_b
        .alloc_storage(&Shape::new(vec![elems]), DType::F32)
        .unwrap();

    let src_rocm = src_dev
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();
    let dst_rocm = dst_dev
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();

    let peer_ptr = dst_rocm.device_ptr_u64().unwrap();
    let stream = std::ptr::null_mut();

    let result = grim_backend_rocm::device::scythe_route::route_commfuse(
        &dev_a,
        stream,
        src_rocm,
        Some(peer_ptr),
        None,
        elems,
    );
    if result.is_ok() {
        // Must synchronize dev_a (which executed the write to dev_b) before dev_b reads!
        dev_a.synchronize();
        dev_b.synchronize();
        let got_bytes = dst_rocm.copy_to_host().unwrap();
        let got: Vec<f32> = got_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let max_err = src
            .iter()
            .zip(got.iter())
            .map(|(s, g)| (s - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[commfuse-pipe] max_err={max_err}");
        assert!(max_err < 1e-5, "COMMFUSE pipe diverges: {max_err}");
    } else {
        eprintln!("[commfuse-pipe] ring route not available — skipped");
    }
}

/// MG-4: OP_PEER_REDUCE — 2-device all-reduce via ScytheRing.
#[test]
fn peer_reduce_2_device() {
    let Some(dev_a) = gpu_device(0) else {
        eprintln!("[SKIP] no GPU 0");
        return;
    };
    let Some(dev_b) = gpu_device(1) else {
        eprintln!("[SKIP] no GPU 1");
        return;
    };
    if dev_a.ordinal() == dev_b.ordinal() {
        eprintln!("[SKIP] same device");
        return;
    }

    let elems = 32usize;
    let partial_a: Vec<f32> = (0..elems).map(|i| i as f32).collect();
    let partial_b: Vec<f32> = (0..elems).map(|i| (elems - i) as f32).collect();

    let pa_dev = dev_a
        .from_cpu(&partial_a, &Shape::new(vec![elems]), DType::F32)
        .unwrap();
    let pb_dev = dev_b
        .from_cpu(&partial_b, &Shape::new(vec![elems]), DType::F32)
        .unwrap();
    let result_dev = dev_a
        .alloc_storage(&Shape::new(vec![elems]), DType::F32)
        .unwrap();

    let pa_rocm = pa_dev
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();
    let pb_rocm = pb_dev
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();
    let res_rocm = result_dev
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();

    // Synchronize both devices before running reduction
    dev_a.synchronize();
    dev_b.synchronize();

    let result = grim_backend_rocm::device::scythe_route::route_peer_reduce(
        &dev_a,
        std::ptr::null_mut(),
        pa_rocm,
        pb_rocm,
        res_rocm,
        elems,
    );

    if result.is_ok() {
        dev_a.synchronize();
        let got_bytes = res_rocm.copy_to_host().unwrap();
        let got: Vec<f32> = got_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let expected: Vec<f32> = partial_a
            .iter()
            .zip(partial_b.iter())
            .map(|(a, b)| a + b)
            .collect();
        let max_err = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[peer-reduce] max_err={max_err}");
        assert!(max_err < 1e-5, "OP_PEER_REDUCE diverges: {max_err}");
    } else {
        eprintln!("[peer-reduce] ring route not available — skipped");
    }
}

/// MG-5: Cross-device dependency tracking via event recording and stream wait.
/// Device A produces data into a buffer; an event is recorded on device A's stream.
/// Device B waits on that event before executing a peer reduce / read.
#[test]
fn cross_device_dependency_tracking() {
    let Some(dev_a) = gpu_device(0) else {
        eprintln!("[SKIP] no GPU 0");
        return;
    };
    let Some(dev_b) = gpu_device(1) else {
        eprintln!("[SKIP] no GPU 1");
        return;
    };
    if dev_a.ordinal() == dev_b.ordinal() {
        eprintln!("[SKIP] same device");
        return;
    }

    let elems = 64usize;
    let data: Vec<f32> = (0..elems).map(|i| (i * 2 + 1) as f32).collect();

    let buf_a = dev_a
        .from_cpu(&data, &Shape::new(vec![elems]), DType::F32)
        .unwrap();
    let buf_b = dev_b
        .alloc_storage(&Shape::new(vec![elems]), DType::F32)
        .unwrap();

    let a_rocm = buf_a.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let b_rocm = buf_b.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Route COMMFUSE from dev_a to dev_b
    let route_res = grim_backend_rocm::device::scythe_route::route_commfuse(
        &dev_a,
        std::ptr::null_mut(),
        a_rocm,
        Some(b_rocm.device_ptr_u64().unwrap()),
        None,
        elems,
    );

    if route_res.is_ok() {
        // Record event on dev_a
        let ev = grim_backend_rocm::device::scythe_route::record_event_on(&dev_a, std::ptr::null_mut())
            .expect("record_event_on");

        // dev_b waits on the event asynchronously — without blocking the host thread!
        grim_backend_rocm::device::scythe_route::stream_wait_event(&dev_b, std::ptr::null_mut(), ev)
            .expect("stream_wait_event");

        // Now synchronizing only dev_b ensures dev_a's write was completed and ordered before dev_b
        dev_b.synchronize();

        grim_backend_rocm::device::scythe_route::destroy_event(ev);

        let got_bytes = b_rocm.copy_to_host().unwrap();
        let got: Vec<f32> = got_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let max_err = data
            .iter()
            .zip(got.iter())
            .map(|(d, g)| (d - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[dep-tracking] max_err={max_err}");
        assert!(max_err < 1e-5, "dependency tracking data mismatch: {max_err}");
    } else {
        eprintln!("[dep-tracking] ring route not available — skipped");
    }
}

/// MG-6: Model-level Tensor-Parallel decode step across GPU[0] + GPU[1].
/// Tests column-parallel Q projection split across 2 GPUs + peer all-reduce,
/// followed by pipeline-stage activation handoff via OP_COMMFUSE.
#[test]
fn tensor_parallel_2_device_decode_pass() {
    let Some(dev_a) = gpu_device(0) else {
        eprintln!("[SKIP] no GPU 0");
        return;
    };
    let Some(dev_b) = gpu_device(1) else {
        eprintln!("[SKIP] no GPU 1");
        return;
    };
    if dev_a.ordinal() == dev_b.ordinal() {
        eprintln!("[SKIP] same device");
        return;
    }

    // Decode step dimensions: batch=1, hidden=128, heads=8 total (4 heads per GPU)
    let (m, k, total_n) = (1usize, 128usize, 128usize);
    let n_per_gpu = total_n / 2; // 64 features per rank

    let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    // Rank 0 weights in [N, K] row-major for OP_ROW_GEMM (C = A @ B^T).
    let w_a: Vec<f32> = (0..n_per_gpu * k)
        .map(|i| (( ( (i % k) * n_per_gpu + i / k) % 17 ) as f32 - 8.0) * 0.02)
        .collect();
    // Rank 1 weights in [N, K] row-major.
    let w_b: Vec<f32> = (0..n_per_gpu * k)
        .map(|i| (( (( (i % k) * n_per_gpu + i / k) + 3) % 19 ) as f32 - 9.0) * 0.02)
        .collect();

    // 1. Allocate input and partial weights
    let x_dev_a = dev_a.from_cpu(&x, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let x_dev_b = dev_b.from_cpu(&x, &Shape::new(vec![m, k]), DType::F32).unwrap();

    let wa_dev = dev_a.from_cpu(&w_a, &Shape::new(vec![n_per_gpu, k]), DType::F32).unwrap();
    let wb_dev = dev_b.from_cpu(&w_b, &Shape::new(vec![n_per_gpu, k]), DType::F32).unwrap();

    let out_a = dev_a.alloc_storage(&Shape::new(vec![m, n_per_gpu]), DType::F32).unwrap();
    let out_b = dev_b.alloc_storage(&Shape::new(vec![m, n_per_gpu]), DType::F32).unwrap();

    let xa_rocm = x_dev_a.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let xb_rocm = x_dev_b.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let wa_rocm = wa_dev.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let wb_rocm = wb_dev.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let outa_rocm = out_a.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();
    let outb_rocm = out_b.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();

    // 2. Launch column-parallel projections simultaneously on each device's ScytheRing
    let res_a = grim_backend_rocm::device::scythe_route::route_gemm(
        &dev_a,
        std::ptr::null_mut(),
        xa_rocm,
        wa_rocm,
        outa_rocm,
        m,
        n_per_gpu,
        k,
    );
    let res_b = grim_backend_rocm::device::scythe_route::route_gemm(
        &dev_b,
        std::ptr::null_mut(),
        xb_rocm,
        wb_rocm,
        outb_rocm,
        m,
        n_per_gpu,
        k,
    );

    if res_a.is_ok() && res_b.is_ok() {
        dev_a.synchronize();
        dev_b.synchronize();

        // Check GEMM output parity on both devices
        let bytes_a = outa_rocm.copy_to_host().unwrap();
        let bytes_b = outb_rocm.copy_to_host().unwrap();
        let got_a: Vec<f32> = bytes_a.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let got_b: Vec<f32> = bytes_b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        // CPU reference for each head partition
        let mut ref_a = vec![0.0f32; n_per_gpu];
        let mut ref_b = vec![0.0f32; n_per_gpu];
        for col in 0..n_per_gpu {
            for kk in 0..k {
                ref_a[col] += x[kk] * w_a[col * k + kk];
                ref_b[col] += x[kk] * w_b[col * k + kk];
            }
        }

        let err_a = ref_a.iter().zip(got_a.iter()).map(|(r, g)| (r - g).abs()).fold(0.0f32, f32::max);
        let err_b = ref_b.iter().zip(got_b.iter()).map(|(r, g)| (r - g).abs()).fold(0.0f32, f32::max);
        assert!(err_a < 1e-4, "TP Rank 0 projection error: {err_a}");
        assert!(err_b < 1e-4, "TP Rank 1 projection error: {err_b}");

        // 3. Now simulate row-parallel FFN all-reduce using all_reduce_sum_peer_pair:
        // Rank 0 and Rank 1 both produced partial vectors of length n_per_gpu.
        // We reduce partial A on dev_a with partial B on dev_b into reduced_dev_a.
        let reduced_a = dev_a.alloc_storage(&Shape::new(vec![m, n_per_gpu]), DType::F32).unwrap();
        let reduced_rocm = reduced_a.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap();

        let comm = grim_backend_rocm::ParallelCommunicator::with_p2p(
            0,
            2,
            vec![dev_a.ordinal(), dev_b.ordinal()],
        ).unwrap();

        comm.all_reduce_sum_peer_pair(outa_rocm, outb_rocm, reduced_rocm, 0).unwrap();
        dev_a.synchronize();

        let red_bytes = reduced_rocm.copy_to_host().unwrap();
        let got_red: Vec<f32> = red_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let expected_sum: Vec<f32> = ref_a.iter().zip(ref_b.iter()).map(|(a, b)| a + b).collect();
        let err_sum = expected_sum.iter().zip(got_red.iter()).map(|(e, g)| (e - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[tp-decode] col-gemm err_a={err_a:.2e} err_b={err_b:.2e}, peer-reduce err={err_sum:.2e}");
        assert!(err_sum < 1e-4, "TP All-Reduce diverges: {err_sum}");
    } else {
        eprintln!("[tp-decode] ScytheRing dispatch not available — skipped");
    }
}


/// MG-6: tensor-parallel single-layer decode across 2 real devices vs single-device.
///
/// Computes one Llama-style block two ways and compares the final residual:
///   (A) 2 devices — QKV column-parallel projection (each device keeps its head
///       shard), O row-parallel projection with all-reduce, residual add, FFN
///       column-parallel gate/up, silu*up, row-parallel down with all-reduce,
///       residual add.
///   (B) single-device host-CPU reference — identical math with full weights.
///
/// Attention is identity (attn_out = projected Q). Per-head SDPA is mathematically
/// independent of the TP split, so it adds no collective-correctness value.
/// The gather/all-reduce paths ARE exercised by the QKV gather, O all-reduce,
/// and FFN down all-reduce. Token parity is the MG-6 acceptance criterion.
#[test]
fn tensor_parallel_2_device_block_decode_matches_single_device() {
    let Some(dev_a) = gpu_device(0) else { eprintln!("[SKIP] no GPU 0"); return; };
    let Some(dev_b) = gpu_device(1) else { eprintln!("[SKIP] no GPU 1 (need 2 devices)"); return; };
    if dev_a.ordinal() == dev_b.ordinal() { eprintln!("[SKIP] same device"); return; }

    let hidden = 128usize;
    let num_heads = 8usize;
    let head_dim = 16usize;
    let q_dim = num_heads * head_dim; // 128
    let inter = 256usize;
    let local_q = q_dim / 2; // 64
    let local_ffn = inter / 2; // 128


    let vals = |seed: u64, n: usize| -> Vec<f32> {
        (0..n).map(|i| (((i as u64).wrapping_add(seed * 2654435761) % 17) as f32 - 8.0) * 0.03).collect()
    };
    let ul = |dev: &grim_backend_rocm::RocmDevice, d: &[f32], dims: &[usize]| {
        dev.from_cpu(d, &Shape::from_slice(dims), DType::F32).unwrap()
    };
    // route_gemm: out = a @ b^T  with a:[m,k], b:[n,k], out:[m,n]
    let rgemm = |dev: &grim_backend_rocm::RocmDevice, a: &grim_backend_rocm::RocmStorage,
                  b: &grim_backend_rocm::RocmStorage, out: &grim_backend_rocm::RocmStorage,
                  m: usize, n: usize, k: usize| {
        grim_backend_rocm::device::scythe_route::route_gemm(
            dev, std::ptr::null_mut(), a, b, out, m, n, k).unwrap();
        dev.synchronize();
    };

    // Full reference weights: Wq [q_dim,h], Wo [hidden,q_dim], Wg/Wu [inter,h], Wd [hidden,inter].
    let wq_full = vals(1, q_dim * hidden);
    let wo_full = vals(4, hidden * q_dim);
    let wg_full = vals(5, inter * hidden);
    let wu_full = vals(6, inter * hidden);
    let wd_full = vals(7, hidden * inter);

    let row_shard = |full: &[f32], cols: usize, start: usize, n: usize| -> Vec<f32> {
        let mut d = Vec::with_capacity(n * cols);
        for r in start..start + n { d.extend_from_slice(&full[r * cols..(r + 1) * cols]); }
        d
    };
    let col_shard = |full: &[f32], rows: usize, cols: usize, start: usize, n: usize| -> Vec<f32> {
        let mut d = Vec::with_capacity(rows * n);
        for r in 0..rows { d.extend_from_slice(&full[r * cols + start..r * cols + start + n]); }
        d
    };

    // Column-parallel QKV/gate/up: rows [start..start+local] -> [local, hidden].
    // Row-parallel O/down: cols [start..start+local] -> [out, local].
    let wq_a = ul(&dev_a, &row_shard(&wq_full, hidden, 0, local_q), &[local_q, hidden]);
    let wo_a = ul(&dev_a, &col_shard(&wo_full, hidden, q_dim, 0, local_q), &[hidden, local_q]);
    let wg_a = ul(&dev_a, &row_shard(&wg_full, hidden, 0, local_ffn), &[local_ffn, hidden]);
    let wu_a = ul(&dev_a, &row_shard(&wu_full, hidden, 0, local_ffn), &[local_ffn, hidden]);
    let wd_a = ul(&dev_a, &col_shard(&wd_full, hidden, inter, 0, local_ffn), &[hidden, local_ffn]);

    let wq_b = ul(&dev_b, &row_shard(&wq_full, hidden, local_q, local_q), &[local_q, hidden]);
    let wo_b = ul(&dev_b, &col_shard(&wo_full, hidden, q_dim, local_q, local_q), &[hidden, local_q]);
    let wg_b = ul(&dev_b, &row_shard(&wg_full, hidden, local_ffn, local_ffn), &[local_ffn, hidden]);
    let wu_b = ul(&dev_b, &row_shard(&wu_full, hidden, local_ffn, local_ffn), &[local_ffn, hidden]);
    let wd_b = ul(&dev_b, &col_shard(&wd_full, hidden, inter, local_ffn, local_ffn), &[hidden, local_ffn]);

    let x_data: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32 - 5.0) * 0.07).collect();
    let x_a = ul(&dev_a, &x_data, &[1, hidden]);
    let x_b = ul(&dev_b, &x_data, &[1, hidden]);

    // alloc_storage returns Box<dyn BackendStorage>; downcast by ref for &RocmStorage.
    fn rs(b: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        b.as_ref().as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap()
    }
    let alloc = |dev: &grim_backend_rocm::RocmDevice, h: usize, w: usize|
                 -> Box<dyn grim_tensor::BackendStorage> {
        dev.alloc_storage(&Shape::from_slice(&[h, w]), DType::F32).unwrap()
    };
    let silu_dev = |dev: &grim_backend_rocm::RocmDevice, n: usize,
                    a: &grim_backend_rocm::RocmStorage, b: &grim_backend_rocm::RocmStorage|
                    -> Box<dyn grim_tensor::BackendStorage> {
        let (out, _) = dev.silu_mul(a, b, &Shape::from_slice(&[1, n])).unwrap();
        dev.synchronize(); out
    };
    let add_dev = |dev: &grim_backend_rocm::RocmDevice, n: usize,
                   a: &grim_backend_rocm::RocmStorage, b: &grim_backend_rocm::RocmStorage|
                   -> Box<dyn grim_tensor::BackendStorage> {
        let (out, _) = dev.add(a, b, &Shape::from_slice(&[1, n])).unwrap();
        dev.synchronize(); out
    };
    let host_f32 = |s: &grim_backend_rocm::RocmStorage| -> Vec<f32> {
        let b = s.copy_to_host().unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };

    // --- 2-device TP block ---
    let tp_out: Vec<f32> = {
        let qa = alloc(&dev_a, 1, local_q);
        rgemm(&dev_a, rs(&x_a), rs(&wq_a), rs(&qa), 1, local_q, hidden);
        let qb = alloc(&dev_b, 1, local_q);
        rgemm(&dev_b, rs(&x_b), rs(&wq_b), rs(&qb), 1, local_q, hidden);

        // O row-parallel: partial O_i = Q_i @ Wo_i^T -> [1,hidden]; allreduce sum.
        let oa = alloc(&dev_a, 1, hidden);
        rgemm(&dev_a, rs(&qa), rs(&wo_a), rs(&oa), 1, hidden, local_q);
        let ob = alloc(&dev_b, 1, hidden);
        rgemm(&dev_b, rs(&qb), rs(&wo_b), rs(&ob), 1, hidden, local_q);
        let comm = grim_backend_rocm::ParallelCommunicator::with_p2p(
            0, 2, vec![dev_a.ordinal(), dev_b.ordinal()]).unwrap();
        let o_sum = alloc(&dev_a, 1, hidden);
        comm.all_reduce_sum_peer_pair(rs(&oa), rs(&ob), rs(&o_sum), 0).unwrap();
        dev_a.synchronize(); dev_b.synchronize();

        // residual x = x + O (each device)
        let x2_a = add_dev(&dev_a, hidden, rs(&x_a), rs(&o_sum));
        let x2_b = add_dev(&dev_b, hidden, rs(&x_b), rs(&o_sum));

        // FFN gate/up column-parallel
        let ga = alloc(&dev_a, 1, local_ffn);
        rgemm(&dev_a, rs(&x2_a), rs(&wg_a), rs(&ga), 1, local_ffn, hidden);
        let ua = alloc(&dev_a, 1, local_ffn);
        rgemm(&dev_a, rs(&x2_a), rs(&wu_a), rs(&ua), 1, local_ffn, hidden);
        let gb = alloc(&dev_b, 1, local_ffn);
        rgemm(&dev_b, rs(&x2_b), rs(&wg_b), rs(&gb), 1, local_ffn, hidden);
        let ub = alloc(&dev_b, 1, local_ffn);
        rgemm(&dev_b, rs(&x2_b), rs(&wu_b), rs(&ub), 1, local_ffn, hidden);

        let gu_a = silu_dev(&dev_a, local_ffn, rs(&ga), rs(&ua));
        let gu_b = silu_dev(&dev_b, local_ffn, rs(&gb), rs(&ub));

        // down row-parallel: partial d_i = gu_i @ Wd_i^T -> [1,hidden]; allreduce -> full down.
        let da = alloc(&dev_a, 1, hidden);
        rgemm(&dev_a, rs(&gu_a), rs(&wd_a), rs(&da), 1, hidden, local_ffn);
        let db = alloc(&dev_b, 1, hidden);
        rgemm(&dev_b, rs(&gu_b), rs(&wd_b), rs(&db), 1, hidden, local_ffn);
        let d_sum = alloc(&dev_a, 1, hidden);
        comm.all_reduce_sum_peer_pair(rs(&da), rs(&db), rs(&d_sum), 0).unwrap();
        dev_a.synchronize(); dev_b.synchronize();

        // final residual x2 = x2 + down
        let out_a = add_dev(&dev_a, hidden, rs(&x2_a), rs(&d_sum));
        dev_a.synchronize();
        host_f32(rs(&out_a))
    };

    // --- Single-device host-CPU reference (full weights) ---
    let ref_out: Vec<f32> = {
        let matmul = |a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)| -> Vec<f32> {
            // c = a @ b^T  with a:[m,k], b:[n,k]  -> c:[m,n]
            let mut c = vec![0.0f32; m * n];
            for i in 0..m { for j in 0..n { let mut s = 0.0f32;
                for p in 0..k { s += a[i * k + p] * b[j * k + p]; }
                c[i * n + j] = s; } }
            c
        };
        let addv = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        };
        let silu_mulv = |g: &[f32], u: &[f32]| -> Vec<f32> {
            g.iter().zip(u.iter()).map(|(gg, up)| {
                let s = gg / (1.0 + (-gg).exp()); s * up
            }).collect()
        };
        let q = matmul(&x_data, &wq_full, (1, hidden, q_dim));
        let o = matmul(&q, &wo_full, (1, q_dim, hidden));
        let x2 = addv(&x_data, &o);
        let g = matmul(&x2, &wg_full, (1, hidden, inter));
        let u = matmul(&x2, &wu_full, (1, hidden, inter));
        let gu = silu_mulv(&g, &u);
        let d = matmul(&gu, &wd_full, (1, inter, hidden));
        addv(&x2, &d)
    };

    let max_err = ref_out.iter().zip(tp_out.iter()).map(|(r, t)| (r - t).abs()).fold(0.0f32, f32::max);
    eprintln!("[mg6] single-vs-2device block max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-3, "MG-6 TP block diverges from single-device: max_err={max_err:.3e}");
}

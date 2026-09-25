//! memory cache tests — split from the original monolithic lib_internal_tests.rs.

use super::common::run_matmul_on_dev;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    const GPU_TEST_ENV: &str = "GRIM_GPU_TEST";

    #[test]
    fn caching_allocator_reuses_buffers_across_steps() {
        // After a short warmup of same-shape matmuls, the steady-state loop must
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        let a_dims = [16usize, 32];
        // SPEED-ROC-16: b is [N, K]; same element count reinterpreted (no
        // numeric check here, only allocator reuse).
        let b_dims = [16usize, 32];
        let a: Vec<f32> = (0..16 * 32).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let b: Vec<f32> = (0..32 * 16).map(|i| i as f32 * 0.02).collect();

        // Warmup so the pool fills with the right size classes.
        for _ in 0..3 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[16, 16]);
        }
        let (m1, _f1) = dev.allocator_stats();
        for _ in 0..20 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[16, 16]);
        }
        let (m2, _f2) = dev.allocator_stats();

        // Steady-state: repeated same-shape matmuls reuse pooled buffers, so new
        assert!(
            (m2 - m1) <= 2,
            "hipMalloc calls grew by {} during steady-state loop (expected ~0, proving pool reuse)",
            m2 - m1
        );
    }

    #[test]
    fn empty_cache_releases_pooled_buffers() {
        // empty_cache() must actually hipFree the retained buffers, bounding memory.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        let a_dims = [8usize, 8];
        let b_dims = [8usize, 8];
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        for _ in 0..5 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[8, 8]);
        }
        let (_m_before, f_before) = dev.allocator_stats();
        dev.empty_cache();
        let (_m_after, f_after) = dev.allocator_stats();
        assert!(
            f_after > f_before,
            "empty_cache must release pooled buffers via hipFree (free_count {} -> {})",
            f_before,
            f_after
        );
    }

    #[test]
    fn module_cache_loads_each_kernel_once() {
        // Each unique compute kernel must be hipModuleLoad'd exactly once for the
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        // The device detects its own gfx target from the driver, so kernel [see: `GRIM_GPU_TARGET`]
        let dev = RocmDevice::new(0);

        let x = dev
            .from_cpu(&[1.0f32; 4 * 8], &Shape::from_slice(&[4, 8]), DType::F32)
            .unwrap();
        let w_norm = dev
            .from_cpu(&[1.0f32; 8], &Shape::from_slice(&[8]), DType::F32)
            .unwrap();
        let w_mat = dev
            .from_cpu(
                &vec![1.0f32; 8 * 16],
                &Shape::from_slice(&[8, 16]),
                DType::F32,
            )
            .unwrap();

        // Warmup: load the rmsnorm_matmul module once.
        let (_o, _h) = dev
            .rmsnorm_matmul(
                x.as_ref(),
                w_norm.as_ref(),
                w_mat.as_ref(),
                1e-5,
                &Shape::from_slice(&[4, 16]),
            )
            .unwrap();
        let baseline = dev.module_load_stats();
        assert!(
            baseline >= 1,
            "expected >=1 module loaded, got {}",
            baseline
        );

        // Repeat many times: module load count must NOT increase.
        for _ in 0..20 {
            let (_o, _h) = dev
                .rmsnorm_matmul(
                    x.as_ref(),
                    w_norm.as_ref(),
                    w_mat.as_ref(),
                    1e-5,
                    &Shape::from_slice(&[4, 16]),
                )
                .unwrap();
        }
        assert_eq!(
            dev.module_load_stats(),
            baseline,
            "module cache reloaded rmsnorm_matmul across repeated dispatches"
        );

        // A second distinct kernel (qkv_attention) must load once, then reuse.
        let q = dev
            .from_cpu(
                &vec![1.0f32; 4 * 4 * 64],
                &Shape::from_slice(&[4, 4, 64]),
                DType::F32,
            )
            .unwrap();
        let (_o, _h) = dev
            .qkv_attention(
                q.as_ref(),
                q.as_ref(),
                q.as_ref(),
                2,    // num_kv_heads: real param, not num_heads/4
                4,    // kv_seq_len
                0,    // cache_offset
                None, // window: full causal
                &Shape::from_slice(&[4, 4, 64]),
                None,
                None,
            )
            .unwrap();
        let with_qkv = dev.module_load_stats();
        assert_eq!(
            with_qkv,
            baseline + 1,
            "qkv_attention should load exactly 1 new module"
        );
        for _ in 0..10 {
            let (_o, _h) = dev
                .qkv_attention(
                    q.as_ref(),
                    q.as_ref(),
                    q.as_ref(),
                    2,
                    4,
                    0,
                    None, // window: full causal
                    &Shape::from_slice(&[4, 4, 64]),
                    None,
                    None,
                )
                .unwrap();
        }
        assert_eq!(
            dev.module_load_stats(),
            with_qkv,
            "module cache reloaded qkv_attention across repeated dispatches"
        );
    }

    #[test]
    fn test_module_cache_solution_index_keys() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();

        let dev = RocmDevice::new(0);

        let a = dev
            .from_cpu(&[1.0f32; 16], &Shape::from_slice(&[4, 4]), DType::F16)
            .unwrap();
        let b = dev
            .from_cpu(&[1.0f32; 16], &Shape::from_slice(&[4, 4]), DType::F16)
            .unwrap();
        let out = dev.zeros(&Shape::from_slice(&[4, 4]), DType::F16).unwrap();

        let a_storage = a.as_any().downcast_ref::<RocmStorage>().unwrap();
        let b_storage = b.as_any().downcast_ref::<RocmStorage>().unwrap();
        let out_storage = out.as_any().downcast_ref::<RocmStorage>().unwrap();

        let initial_loads = dev.module_load_stats();

        let mut a_ptr = a_storage.device_ptr.unwrap();
        let mut b_ptr = b_storage.device_ptr.unwrap();
        let mut out_ptr = out_storage.device_ptr.unwrap();
        let mut m = 4i32;
        let mut n = 4i32;
        let mut k = 4i32;
        let mut sa = 4i32;
        let mut sb = 4i32;
        let mut sc = 4i32;

        let grid_dim = HipDim3::new(1, 1, 1);
        let block_dim = HipDim3::new(256, 1, 1);

        dev.launch_compute_kernel_with_solution(
            "grim_decode_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut m),
                arg(&mut n),
                arg(&mut k),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(42),
            0,
        )
        .unwrap();

        let loads_after_sol42 = dev.module_load_stats();
        assert!(loads_after_sol42 > initial_loads);

        dev.launch_compute_kernel_with_solution(
            "grim_decode_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut m),
                arg(&mut n),
                arg(&mut k),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(42),
            0,
        )
        .unwrap();
        assert_eq!(dev.module_load_stats(), loads_after_sol42);

        dev.launch_compute_kernel_with_solution(
            "grim_decode_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut m),
                arg(&mut n),
                arg(&mut k),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(43),
            0,
        )
        .unwrap();
        assert!(dev.module_load_stats() > loads_after_sol42);
    }

    #[test]
    fn embedding_frees_temp_buffer_after_launch() {
        // Regression: embedding allocated a temp idx buffer and freed it right
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        let weight = dev
            .from_cpu(
                &vec![1.0f32; 16 * 8],
                &Shape::from_slice(&[16, 8]),
                DType::F32,
            )
            .unwrap();
        let indices: Vec<u32> = (0..4).collect();
        let out_shape = Shape::from_slice(&[4, 8]);
        let res = dev.embedding(weight.as_ref(), &indices, &out_shape);
        assert!(
            res.is_ok(),
            "embedding must succeed without use-after-free: {:?}",
            res.err()
        );
    }

    #[test]
    fn zeros_uses_hipmemset_not_host_copy() {
        // zeros() must fill the device buffer with zero bytes for every dtype it
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        let shape = Shape::from_slice(&[3, 7, 5]);

        let dtypes = [
            DType::F32,
            DType {
                arith: ArithType::F16,
                storage: DTypeStorage::Native,
            },
            DType::BF16,
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        ];
        for dtype in &dtypes {
            let storage = dev.zeros(&shape, dtype.clone()).unwrap();
            let rs = storage
                .as_any()
                .downcast_ref::<RocmStorage>()
                .expect("RocmStorage");
            assert!(rs.device_ptr_is_valid(), "expected valid ptr for {dtype:?}");
            let nbytes = rs.bytes();
            let mut host = vec![0xABu8; nbytes];
            let res = unsafe {
                hipMemcpy(
                    host.as_mut_ptr() as *mut c_void,
                    rs.device_ptr.unwrap() as *mut c_void,
                    nbytes,
                    HipMemcpyKind::DeviceToHost,
                )
            };
            assert_eq!(res, hipSuccess, "readback failed for {dtype:?}");
            assert!(
                host.iter().all(|&b| b == 0),
                "zeros() left non-zero bytes for {dtype:?}: {:?}",
                &host[..nbytes.min(8)]
            );
        }
    }

    #[test]
    fn host_transfer_pinned_async_matches_sync() {
        // The pinned + async host-transfer path (Item 4) must produce results
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        let shape = Shape::from_slice(&[64, 64]);
        let data: Vec<f32> = (0..shape.elem_count())
            .map(|i| (i as f32) * 0.1 - 5.0)
            .collect();

        // Cold path: pageable Vec + synchronous hipMemcpy.
        let sync_storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
        let sync_out = sync_storage.to_cpu_vec_f32().unwrap();

        // Hot path: pinned buffer + async hipMemcpy.
        let async_storage = dev.copy_from_host_async(&data, &shape, DType::F32).unwrap();
        dev.synchronize();
        let async_out = dev.read_to_host_async(async_storage.as_ref()).unwrap();

        assert_eq!(sync_out.len(), data.len());
        assert_eq!(async_out.len(), data.len());
        for i in 0..data.len() {
            assert!(
                (sync_out[i] - data[i]).abs() < 1e-3,
                "sync round-trip mismatch at {i}: {} vs {}",
                sync_out[i],
                data[i]
            );
            assert!(
                (async_out[i] - data[i]).abs() < 1e-3,
                "pinned-async round-trip mismatch at {i}: {} vs {}",
                async_out[i],
                data[i]
            );
        }

        // Reusable pinned buffer path (decode-loop steady state).
        let mut pinned = RocmPinnedBuffer::<f32>::alloc(data.len()).unwrap();
        let async_storage2 = dev.copy_from_host_async(&data, &shape, DType::F32).unwrap();
        dev.synchronize();
        dev.read_into_pinned(async_storage2.as_ref(), &mut pinned)
            .unwrap();
        assert_eq!(pinned.as_slice(), data.as_slice());

        // Reusable pinned buffer for the upload side too.
        let pinned_in = RocmPinnedBuffer::<f32>::from_slice(&data).unwrap();
        let async_storage3 = dev
            .upload_from_pinned(&pinned_in, &shape, DType::F32)
            .unwrap();
        dev.synchronize();
        let async_out3 = dev.read_to_host_async(async_storage3.as_ref()).unwrap();
        for i in 0..data.len() {
            assert!(
                (async_out3[i] - data[i]).abs() < 1e-3,
                "upload_from_pinned round-trip mismatch at {i}",
            );
        }
    }

    #[test]
    fn host_transfer_pinned_async_benchmark() {
        // Benchmark: per-token host round-trip latency, pageable+sync vs pinned+async.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let _gpu_guard = crate::device::util::gpu_test_lock();
        let dev = RocmDevice::new(0);
        // Logits-sized staging buffer (vocab ~32k floats), typical decode readback.
        let n = 32_768;
        let shape = Shape::from_slice(&[n]);
        let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();

        let iters = 200;
        let warmup = 20;

        // Pageable + synchronous hipMemcpy round trip.
        for _ in 0..warmup {
            let s = dev.from_cpu(&data, &shape, DType::F32).unwrap();
            let _ = s.to_cpu_vec_f32().unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let s = dev.from_cpu(&data, &shape, DType::F32).unwrap();
            let _ = s.to_cpu_vec_f32().unwrap();
        }
        let sync_elapsed = t0.elapsed();

        // Pinned + async hipMemcpy round trip (reusing one pinned buffer for input
        let pinned_in = RocmPinnedBuffer::<f32>::from_slice(&data).unwrap();
        let mut pinned_out = RocmPinnedBuffer::<f32>::alloc(n).unwrap();
        for _ in 0..warmup {
            let s = dev
                .upload_from_pinned(&pinned_in, &shape, DType::F32)
                .unwrap();
            dev.synchronize();
            dev.read_into_pinned(s.as_ref(), &mut pinned_out).unwrap();
        }
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let s = dev
                .upload_from_pinned(&pinned_in, &shape, DType::F32)
                .unwrap();
            dev.synchronize();
            dev.read_into_pinned(s.as_ref(), &mut pinned_out).unwrap();
        }
        let async_elapsed = t1.elapsed();

        let sync_us = sync_elapsed.as_secs_f64() * 1e6 / iters as f64;
        let async_us = async_elapsed.as_secs_f64() * 1e6 / iters as f64;
        println!(
            "[Item 4 benchmark] pageable+sync={:.1} us/round-trip, pinned+async={:.1} us/round-trip ({:.2}x)",
            sync_us,
            async_us,
            sync_us / async_us.max(1e-9)
        );
        // Sanity: pinned+async must not be catastrophically slower (bandwidth
        assert!(
            async_us <= sync_us * 4.0 + 1.0,
            "pinned+async unexpectedly slower: {async_us:.1} vs {sync_us:.1} us"
        );
    }
}

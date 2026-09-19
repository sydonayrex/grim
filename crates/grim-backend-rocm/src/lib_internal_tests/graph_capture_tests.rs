//! graph capture tests — split from the original monolithic lib_internal_tests.rs.

use super::common::approx_eq;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    const GPU_TEST_ENV: &str = "GRIM_GPU_TEST";

    #[test]
    fn graph_capture_session_replays_decode_sequence() {
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let _gpu_guard = crate::device::util::gpu_test_lock();
            let dev = RocmDevice::new(0);

            // Inputs are uploaded eagerly (outside the capture bracket) so the
            let m = 16usize;
            let k = 32usize;
            let n = 16usize;
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05) - 1.0).collect();
            // SPEED-ROC-16: b is [N, K] = [n, k].
            let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.05) + 0.5).collect();
            let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.1)).collect();
            let a_s = dev
                .from_cpu(&a, &Shape::from_slice(&[m, k]), DType::F32)
                .unwrap();
            let b_s = dev
                .from_cpu(&b, &Shape::from_slice(&[n, k]), DType::F32)
                .unwrap();
            let w_s = dev
                .from_cpu(&w, &Shape::from_slice(&[n]), DType::F32)
                .unwrap();
            let out_shape = Shape::from_slice(&[m, n]);
            let eps = 1e-5f32;

            // --- CPU reference (hardware-independent ground truth) ---
            // SPEED-ROC-16: C = A @ B^T with b [N, K].
            let mut c_ref = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0f32;
                    for kk in 0..k {
                        s += a[i * k + kk] * b[j * k + kk];
                    }
                    c_ref[i * n + j] = s;
                }
            }
            let d_ref: Vec<f32> = c_ref.iter().map(|x| x * 2.0).collect();
            let mut e_ref = vec![0f32; m * n];
            for i in 0..m {
                let mut ss = 0f32;
                for j in 0..n {
                    ss += d_ref[i * n + j] * d_ref[i * n + j];
                }
                let rms = (ss / n as f32 + eps).sqrt();
                for j in 0..n {
                    e_ref[i * n + j] = d_ref[i * n + j] * w[j] / rms;
                }
            }

            // --- Capture + replay ---
            let key = "item5_test_seq";
            // First lookup misses -> caller captures this time.
            assert!(!dev.replay_graph(key).unwrap());
            dev.begin_graph_capture(key).unwrap();
            let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out_shape).unwrap();
            let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out_shape).unwrap();
            let (e, _) = dev
                .rms_norm(d.as_ref(), w_s.as_ref(), eps, &out_shape)
                .unwrap();
            dev.end_graph_capture(key).unwrap();
            // Graph is cached; replay fills c/d/e.
            assert!(dev.replay_graph(key).unwrap());
            let replay = e.to_cpu_vec_f32().unwrap();

            assert_eq!(replay.len(), e_ref.len());
            for (i, (rp, eg)) in replay.iter().zip(e_ref.iter()).enumerate() {
                assert!(
                    approx_eq(*rp, *eg, 1e-2),
                    "capture/replay mismatch at [{}][{}]: got {}, cpu ref {}",
                    i / n,
                    i % n,
                    rp,
                    eg
                );
            }
        });
    }

    #[test]
    fn graph_capture_replay_miss_returns_false() {
        // Capturing under one key and then replaying a *different* key must return
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let _gpu_guard = crate::device::util::gpu_test_lock();
            let dev = RocmDevice::new(0);
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
            let a_s = dev
                .from_cpu(&a, &Shape::from_slice(&[2, 2]), DType::F32)
                .unwrap();
            let b_s = dev
                .from_cpu(&b, &Shape::from_slice(&[2, 2]), DType::F32)
                .unwrap();

            dev.begin_graph_capture("A").unwrap();
            let (out_a, _) = dev
                .matmul(a_s.as_ref(), b_s.as_ref(), &Shape::from_slice(&[2, 2]))
                .unwrap();
            dev.end_graph_capture("A").unwrap();

            assert!(dev.replay_graph("A").unwrap(), "key A should be cached");
            assert!(
                !dev.replay_graph("B").unwrap(),
                "key B is a miss -> Ok(false)"
            );
            // Keep the captured output alive until the test ends so the cached graph
            drop(out_a);
        });
    }

    #[test]
    fn graph_capture_session_benchmark() {
        // Capture once, replay N times, and compare wall-clock against N eager
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let _gpu_guard = crate::device::util::gpu_test_lock();
            let dev = RocmDevice::new(0);
            let m = 64usize;
            let k = 128usize;
            let n = 64usize;
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 1.0).collect();
            // SPEED-ROC-16: b is [N, K] = [n, k].
            let b: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.02).collect();
            let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.1)).collect();
            let a_s = dev
                .from_cpu(&a, &Shape::from_slice(&[m, k]), DType::F32)
                .unwrap();
            let b_s = dev
                .from_cpu(&b, &Shape::from_slice(&[n, k]), DType::F32)
                .unwrap();
            let w_s = dev
                .from_cpu(&w, &Shape::from_slice(&[n]), DType::F32)
                .unwrap();
            let out = Shape::from_slice(&[m, n]);
            let eps = 1e-5f32;

            let iters = 100usize;
            let warmup = 10usize;

            for _ in 0..warmup {
                let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
                let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
                let (_e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
                let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
                let (_e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            }
            let eager_elapsed = t0.elapsed();

            let key = "item5_bench_seq";
            assert!(!dev.replay_graph(key).unwrap());
            dev.begin_graph_capture(key).unwrap();
            let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
            let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
            let (e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            dev.end_graph_capture(key).unwrap();
            for _ in 0..warmup {
                dev.replay_graph(key).unwrap();
            }
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                dev.replay_graph(key).unwrap();
            }
            let replay_elapsed = t1.elapsed();
            // The captured graph targets c/d/e; keep them alive across replays.
            drop(c);
            drop(d);
            drop(e);

            let eager_us = eager_elapsed.as_secs_f64() * 1e6 / iters as f64;
            let replay_us = replay_elapsed.as_secs_f64() * 1e6 / iters as f64;
            println!(
                "[Item 5 benchmark] eager={:.1} us/seq, capture+replay={:.1} us/seq ({:.2}x)",
                eager_us,
                replay_us,
                eager_us / replay_us.max(1e-9)
            );
            // Replay must not be catastrophically slower than eager (launch overhead
            assert!(
                replay_us <= eager_us * 3.0 + 1.0,
                "capture+replay unexpectedly slower: {replay_us:.1} vs {eager_us:.1} us"
            );
        });
    }

}

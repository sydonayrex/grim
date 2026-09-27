//! Localisation probe: does `RocmDevice::try_new` complete on this box?
//!
//! Every ROCm test in this crate stalls before its first assertion, including
//! ones recorded as green. `gpu_props` only calls `hipGetDeviceCount` +
//! `probe_host_gpu`, so it never exercises the constructor. This probe does
//! nothing but construct the device, printing before and after, so a stall
//! localises to the constructor rather than to any kernel.
//!
//! `GRIM_PROBE_STAGE=n` stops after stage n, to bisect inside the constructor.

use grim_backend_rocm::{MemoryOps, RocmDevice};
use std::sync::Arc;
use std::time::Instant;

static ORD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn stage(n: u32) -> bool {
    let limit = std::env::var("GRIM_PROBE_STAGE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX);
    if n > limit {
        eprintln!("[probe] stopping after stage {n} (GRIM_PROBE_STAGE={limit})");
        std::process::exit(0);
    }
    true
}

fn main() {
    let ord: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    ORD.store(ord, std::sync::atomic::Ordering::SeqCst);
    eprintln!("[probe] constructing ordinal {ord}");
    let t0 = Instant::now();
    eprintln!("[probe] start");

    let dev = match std::panic::catch_unwind(|| {
        Arc::new(
            RocmDevice::try_new(ORD.load(std::sync::atomic::Ordering::SeqCst))
                .expect("try_new(ORD)"),
        )
    }) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[probe] try_new(0) PANICKED after {:?}", t0.elapsed());
            std::process::exit(2);
        }
    };
    eprintln!("[probe] try_new(0) OK after {:?}", t0.elapsed());

    if !stage(1) {
        return;
    }

    // First allocation: exercises the allocator and, on the first kernel
    // launch, the JIT/HSACO path.
    let shp = grim_tensor::Shape::new(vec![64]);
    let alloc = dev
        .alloc_storage(&shp, grim_tensor::DType::F32)
        .expect("alloc_storage");
    eprintln!("[probe] alloc_storage OK after {:?}", t0.elapsed());

    if !stage(2) {
        return;
    }

    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let uploaded =
        grim_tensor::CoreTensorOps::from_cpu(dev.as_ref(), &data, &shp, grim_tensor::DType::F32)
            .expect("from_cpu");
    eprintln!("[probe] from_cpu OK after {:?}", t0.elapsed());

    if !stage(3) {
        return;
    }

    let read = uploaded.to_cpu_vec_f32().expect("to_cpu_vec_f32");
    eprintln!(
        "[probe] round trip OK after {:?} (sum {})",
        t0.elapsed(),
        read.iter().sum::<f32>()
    );
    let _ = alloc;
    eprintln!("[probe] all stages passed");
}

use grim_backend_rocm::RocmDevice;
fn main() {
    let dev = RocmDevice::shared(0);
    let (f0, t0) = grim_backend_rocm::vram_info(0);
    // mimic the load: 7722 blocks, avg 1.6 MB, exact sizes
    let mut blocks: Vec<std::ptr::NonNull<std::ffi::c_void>> = Vec::new();
    let mut seed = 12345u64;
    let mut allocated = 0usize;
    for i in 0..7722 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let size = ((seed >> 33) as usize % (3 << 20)) + 4096; // 4 KB .. 3 MB
        match dev.allocator_handle().alloc(size) {
            Ok(p) => { blocks.push(std::ptr::NonNull::new(p).unwrap()); allocated += size; }
            Err(e) => { println!("alloc failed at {i}: {e}"); break; }
        }
    }
    let (f1, _) = grim_backend_rocm::vram_info(0);
    println!("allocated {:.2} GiB in {} blocks; vram used delta = {:.2} GiB",
        allocated as f64/(1<<30) as f64, blocks.len(), (f0-f1) as f64/(1<<30) as f64);
    let _ = t0;
}

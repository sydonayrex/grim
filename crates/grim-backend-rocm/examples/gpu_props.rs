//! Capture the HIP-visible topology used by the SCYTHE-2 syd-beasty gates.
use grim_backend_rocm::{hipGetDeviceCount, probe_host_gpu};

fn main() {
    let mut count = 0;
    let status = unsafe { hipGetDeviceCount(&mut count) };
    if status != 0 {
        eprintln!("hipGetDeviceCount failed: {status}");
        return;
    }
    println!("{{\"devices\":[");
    for ordinal in 0..count {
        if ordinal != 0 {
            print!(",");
        }
        match probe_host_gpu(ordinal as usize) {
            Ok(c) => print!(
                "{{\"ordinal\":{ordinal},\"gcnArchName\":\"{}\",\"wavefront\":{},\"lds_bytes\":{}}}",
                c.gcn, c.wavefront_size, c.lds_size_bytes
            ),
            Err(e) => print!("{{\"ordinal\":{ordinal},\"error\":\"{e}\"}}"),
        }
    }
    println!("\n]}}");
}

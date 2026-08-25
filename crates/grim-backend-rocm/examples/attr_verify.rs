//! One-off: verify hipDeviceGetAttribute numbering on this ROCm.
use grim_backend_rocm::device::handles::{hipDeviceGetAttribute, hipDeviceSynchronize};

fn q(attr: i32, ordinal: i32) -> i32 {
    let mut v: i32 = 0;
    unsafe { hipDeviceGetAttribute(&mut v, attr, ordinal) };
    v
}

fn main() {
    for ord in 0..2 {
        println!(
            "GPU[{ord}] attr#24(ManagedMemory per real map)={} attr#87(WarpSize?)={} \
                  attr#5(ClockRate?)={}kHz attr#63(MultiProcCount?)={} attr#56(MaxThreadsPerBlock?)={} \
                  attr#74(MaxSharedMemPerBlock?)={}",
            q(24, ord),
            q(87, ord),
            q(5, ord),
            q(63, ord),
            q(56, ord),
            q(74, ord)
        );
    }
    unsafe { hipDeviceSynchronize() };
}

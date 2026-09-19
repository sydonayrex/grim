//! Module root: the `AttentionOps` trait impl plus bare-impl launchers grouped
//! by kernel family (RoPE, flash decode, paged attention).





mod attention_ops;
mod flash_decode;
mod paged_attention;
mod rope;

#[allow(unused_imports)] // flat public API surface: `device::attention::<method>`
pub use attention_ops::*;
#[allow(unused_imports)]
pub use flash_decode::*;
#[allow(unused_imports)]
pub use paged_attention::*;
#[allow(unused_imports)]
pub use rope::*;

#[cfg(test)]
mod attention_dispatch_selection_tests {
    use crate::device::roc_device::RocmDevice;
    use crate::{as_rocm};
    use grim_tensor::Shape;
    use grim_tensor::backend::BackendStorage;

    fn gpu_device() -> Option<RocmDevice> {
        if !crate::gpu_test_enabled() {
            eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
            return None;
        }
        std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
    }

    #[test]
    fn flash_decode_min_kv_env_override() {
        let Some(dev) = gpu_device() else { return };
        // SAFETY: tests run single-threaded w.r.t. this env var (no other test reads it concurrently).
        unsafe { std::env::set_var("GRIM_FLASH_DECODE_MIN_KV", "77") };
        assert_eq!(dev.flash_decode_min_kv(), 77);
        unsafe { std::env::remove_var("GRIM_FLASH_DECODE_MIN_KV") };
    }

    #[test]
    fn flash_decode_min_kv_arch_default() {
        let Some(dev) = gpu_device() else { return };
        let expected = if dev.is_rdna34 { 256 } else { 512 };
        assert_eq!(dev.flash_decode_min_kv(), expected);
    }

    #[test]
    fn flash_decode_split_count_static_heuristic() {
        use grim_tensor::{CoreTensorOps, DType};
        let Some(dev) = gpu_device() else { return };
        // GRIM_ATTENTION_AUTOTUNE unset -> pure heuristic; the storages are
        // never dereferenced on this path, so 1-element buffers suffice.
        let mk = || -> Box<dyn BackendStorage> {
            CoreTensorOps::from_cpu(&dev, &[0.0f32], &Shape::new(vec![1]), DType::F32)
                .expect("from_cpu")
        };
        let (qb, kb, vb, ob) = (mk(), mk(), mk(), mk());
        let q = as_rocm(qb.as_ref()).expect("rocm storage");
        let k = as_rocm(kb.as_ref()).expect("rocm storage");
        let v = as_rocm(vb.as_ref()).expect("rocm storage");
        let o = as_rocm(ob.as_ref()).expect("rocm storage");
        let split = |kv: usize| {
            dev.flash_decode_split_count(q, k, v, o, 8, 8, 64, kv)
        };
        assert_eq!(split(100), 2, "kv_len=100 -> clamp(0,2,64)=2");
        assert_eq!(split(1024), 4, "kv_len=1024 -> 1024/256 = 4");
        assert_eq!(split(1 << 20), 64, "kv_len=1M -> clamp saturates at 64");
    }

    #[test]
    fn autotune_attention_block_dim_gated_off() {
        let Some(dev) = gpu_device() else { return };
        let key = crate::autotune::KernelKey {
            kernel: "grim_flash_decode",
            gpu_arch: "gfxTest",
            m: 8,
            n: 64,
            k: 8192,
        };
        let mut called = 0;
        let r = dev.autotune_attention_block_dim(key, 128, 8192, 4096, |_| {
            called += 1;
            Ok(())
        });
        assert!(r.is_none(), "autotune disabled -> no sweep");
        assert_eq!(called, 0, "launch closure must not run when gated off");
    }

    #[test]
    fn autotune_attention_block_dim_kv_below_min() {
        let Some(dev) = gpu_device() else { return };
        let key = crate::autotune::KernelKey {
            kernel: "grim_flash_decode",
            gpu_arch: "gfxTest",
            m: 8,
            n: 64,
            k: 128,
        };
        let mut called = 0;
        let r = dev.autotune_attention_block_dim(key, 128, 128, 4096, |_| {
            called += 1;
            Ok(())
        });
        assert!(r.is_none(), "kv below min_kv_len -> no sweep");
        assert_eq!(called, 0);
    }
}


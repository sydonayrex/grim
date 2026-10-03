//! grim-sonnet F6 / F8 / F9 / F11 — native accelerator capability gates. [see: `rust-ffi`, `rust-gpu-discipline`]

use crate::quantization::{GcnArch, QuantMode, arch_capability, gcn_arch};

// Reuse the crate's real HIP FFI rather than redeclaring it. `detect_gpu_arch` [see: `hipGetDeviceProperties`, `gcnArchName`]
use crate::device::util::detect_gpu_arch;
use crate::hipGetDeviceCount;

// F6 - MFMA availability

/// Whether the arch has native **MFMA** matrix cores for a given arithmetic [see: `cubecl`, `hip/arch.rs`, `is_mfma_capable()`, `gfx1200+`]
pub fn mfma_supported(arch: GcnArch, mode: QuantMode) -> bool {
    // grim targets RDNA and UDNA. MFMA is the CDNA matrix core; on RDNA the
    // path is WMMA/rocWMMA. UDNA carries the MFMA capability, so it keeps the
    // check even though it also covers gfx1200/1201.
    if !matches!(arch, GcnArch::UDNA) {
        return false; // RDNA has no MFMA matrix cores.
    }
    // On UDNA, fp8 MFMA only where fp8 is native; fp16/bf16/fp32 always.
    arch_capability(arch).supports(mode)
}

/// Runtime variant: detect the arch from the actual device and classify MFMA. [see: `detect_gpu_arch`, `hipGetDeviceProperties`]
pub fn mfma_supported_on_device(device: i32, mode: QuantMode) -> bool {
    mfma_supported(gcn_arch(&detect_gpu_arch(device)), mode)
}

/// Dispatch gate for an MFMA-backed GEMM. Returns the resolved mode or `Err`. [see: `resolve_quant_mode`, `__builtin_amdgcn_mfma_*`]
pub fn mfma_dispatch(arch: &str, requested: QuantMode) -> Result<QuantMode, &'static str> {
    let a = gcn_arch(arch);
    if mfma_supported(a, requested) {
        Ok(requested)
    } else if !matches!(a, GcnArch::UDNA) {
        Err("no MFMA matrix cores on RDNA; use WMMA/rocWMMA (GFX11+) or JIT HIP grim_* kernels")
    } else {
        match requested {
            QuantMode::Fp8Native => {
                Err("no native fp8 MFMA on this arch; downshift via resolve_quant_mode")
            }
            _ => Err("requested MFMA mode unavailable; fall back to fp32 path"),
        }
    }
}

// WMMA availability (WI-G)

/// Whether the arch has native **WMMA** matrix cores for a given arithmetic mode.
pub fn wmma_supported(arch: GcnArch, mode: QuantMode) -> bool {
    let is_wmma_capable = matches!(arch, GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::UDNA);
    if !is_wmma_capable {
        return false; // Older RDNA lacks WMMA.
    }
    arch_capability(arch).supports(mode)
}

/// Runtime variant: detect the arch from the actual device and classify WMMA.
pub fn wmma_supported_on_device(device: i32, mode: QuantMode) -> bool {
    wmma_supported(gcn_arch(&detect_gpu_arch(device)), mode)
}

/// Dispatch gate for a WMMA-backed GEMM. Returns the resolved mode or `Err`.
pub fn wmma_dispatch(arch: &str, requested: QuantMode) -> Result<QuantMode, &'static str> {
    let a = gcn_arch(arch);
    if wmma_supported(a, requested) {
        Ok(requested)
    } else if !matches!(a, GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::UDNA) {
        Err("no WMMA matrix cores on this architecture; older RDNA uses JIT HIP")
    } else {
        match requested {
            QuantMode::Fp8Native => {
                Err("no native fp8 WMMA on this RDNA arch; downshift via resolve_quant_mode")
            }
            _ => Err("requested WMMA mode unavailable; fall back to fp32 path"),
        }
    }
}

// F8 - Composable Kernel (CK) dispatch

/// CK (Composable Kernel) is AMD's generic GEMM/attention library. The [see: `ck_tile`, `-DCK_TILE_USE_WMMA`]
pub fn ck_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::UDNA
    )
}

/// Dispatch gate: CK is usable on any modern RDNA/UDNA part. Returns `Ok` for [see: `Err`, `grim_*`]
pub fn ck_dispatch(arch: &str) -> Result<(), &'static str> {
    if ck_supported(gcn_arch(arch)) {
        Ok(())
    } else {
        Err("Composable Kernel unavailable on this GCN arch; use JIT HIP grim_* kernels")
    }
}

// F9 - MIOpen convolution/depthwise kernels

/// MIOpen provides conv/depthwise kernels. It is available (library present +
pub fn miopen_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::UDNA
    )
}

/// Dispatch gate for a MIOpen convolution forward call. [see: `miopen_probe`, `accel_ffi`, `libloading`, `.so`]
pub fn miopen_conv_dispatch(arch: &str) -> Result<(), &'static str> {
    if !miopen_supported(gcn_arch(arch)) {
        return Err("MIOpen conv unavailable on this arch; use a direct JIT HIP conv kernel");
    }
    if crate::device::accel_ffi::miopen_probe().is_err() {
        return Err("MIOpen library not loadable at runtime; cannot dispatch conv");
    }
    Ok(())
}

// F11 - RCCL multi-GPU collectives

/// RCCL (ROCm Collective Communications Library) implements NCCL-style [see: `ncclAllReduce`, `ncclBroadcast`]
pub fn rccl_device_count() -> Result<usize, i32> {
    let mut count: i32 = 0;
    // SAFETY: `count` is a local with a stable address; hipGetDeviceCount writes [see: `count`]
    let status = unsafe { hipGetDeviceCount(&mut count as *mut i32) };
    if status == 0 {
        Ok(count.max(0) as usize)
    } else {
        Err(status)
    }
}

/// Classify whether RCCL collectives are usable given a device count.
pub fn rccl_supported(device_count: usize) -> bool {
    device_count > 1
}

/// Dispatch gate for an RCCL collective. `world_size` is the number of ranks.
pub fn rccl_collective_dispatch(world_size: usize) -> Result<(), &'static str> {
    if rccl_supported(world_size) {
        Ok(())
    } else {
        Err("RCCL collective requires world_size > 1; single-GPU host has no peers to reduce over")
    }
}

#[cfg(test)]
mod self_tests {
    use super::*;

    // F6 — MFMA is a UDNA capability here; CDNA is not a grim target.
    #[test]
    fn f6_mfma_udna_only() {
        // RDNA1/2/3 has no MFMA matrix cores.
        for arch in ["gfx1036", "gfx1100", "gfx1102"] {
            assert!(
                !mfma_supported(gcn_arch(arch), QuantMode::F16),
                "MFMA must be unsupported on RDNA {arch}"
            );
            assert!(mfma_dispatch(arch, QuantMode::F16).is_err());
        }
        // RDNA4 also has no MFMA — it uses WMMA, like the rest of RDNA.
        assert!(!mfma_supported(gcn_arch("gfx1201"), QuantMode::F16));

        // UDNA carries the MFMA path, including native fp8.
        assert!(mfma_supported(gcn_arch("gfx1300"), QuantMode::F16));
        assert!(mfma_supported(gcn_arch("gfx1300"), QuantMode::Fp8Native));
        assert!(mfma_dispatch("gfx1300", QuantMode::Fp8Native).is_ok());

        // CDNA parses to `Other`, so MFMA must be refused rather than
        // silently admitted off a stale capability table.
        for arch in ["gfx906", "gfx908", "gfx942", "gfx950"] {
            assert_eq!(gcn_arch(arch), GcnArch::Other, "{arch} must be Other");
            assert!(
                !mfma_supported(gcn_arch(arch), QuantMode::F16),
                "MFMA must be unsupported on unsupported {arch}"
            );
            assert!(mfma_dispatch(arch, QuantMode::F16).is_err());
        }
    }

    // F8 — CK valid on RDNA (WMMA) + UDNA.
    #[test]
    fn f8_ck_on_rdna_and_udna() {
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx1300"] {
            assert!(ck_dispatch(arch).is_ok(), "CK must be allowed on {arch}");
        }
        for arch in ["gfx908", "gfx942", "gfx950"] {
            assert!(
                ck_dispatch(arch).is_err(),
                "CK must be refused on unsupported {arch}"
            );
        }
    }

    // F9 — MIOpen on RDNA + UDNA.
    #[test]
    fn f9_miopen_on_rdna_and_udna() {
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx1300"] {
            assert!(
                miopen_supported(gcn_arch(arch)),
                "MIOpen policy must cover {arch}"
            );
        }
        let _ = miopen_conv_dispatch("gfx1036");
        assert!(miopen_conv_dispatch("gfx900").is_err());
    }

    // F11 — RCCL only with >1 device.
    #[test]
    fn f11_rccl_requires_multi_device() {
        for n in [0usize, 1] {
            assert!(
                rccl_collective_dispatch(n).is_err(),
                "RCCL must reject world_size={n}"
            );
        }
        for n in [2usize, 4, 8] {
            assert!(
                rccl_collective_dispatch(n).is_ok(),
                "RCCL must allow world_size={n}"
            );
        }
    }
}

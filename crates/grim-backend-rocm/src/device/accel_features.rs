//! grim-sonnet F6 / F8 / F9 / F11 — native accelerator capability gates. [see: `rust-ffi`, `rust-gpu-discipline`]

use crate::quantization::{GcnArch, QuantMode, arch_capability, gcn_arch};

// Reuse the crate's real HIP FFI rather than redeclaring it. `detect_gpu_arch` [see: `hipGetDeviceProperties`, `gcnArchName`]
use crate::device::util::detect_gpu_arch;
use crate::hipGetDeviceCount;

// ---------------------------------------------------------------------------
// F6 — MFMA availability
// ---------------------------------------------------------------------------

/// Whether the arch has native **MFMA** matrix cores for a given arithmetic [see: `cubecl`, `hip/arch.rs`, `is_mfma_capable()`, `gfx1200+`]
pub fn mfma_supported(arch: GcnArch, mode: QuantMode) -> bool {
    let is_cdna = matches!(arch, GcnArch::CDNA2 | GcnArch::CDNA3);
    if !is_cdna {
        return false; // RDNA has no MFMA matrix cores.
    }
    // Inside CDNA, fp8 MFMA only where fp8 is native; fp16/bf16/fp32 always.
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
    } else if !matches!(a, GcnArch::CDNA2 | GcnArch::CDNA3) {
        Err("no MFMA matrix cores on RDNA; use WMMA/rocWMMA (GFX11+) or JIT HIP grim_* kernels")
    } else {
        match requested {
            QuantMode::Fp8Native => {
                Err("no native fp8 MFMA on this CDNA arch; downshift via resolve_quant_mode")
            }
            _ => Err("requested MFMA mode unavailable; fall back to fp32 path"),
        }
    }
}

// ---------------------------------------------------------------------------
// WMMA availability (WI-G)
// ---------------------------------------------------------------------------

/// Whether the arch has native **WMMA** matrix cores for a given arithmetic mode.
pub fn wmma_supported(arch: GcnArch, mode: QuantMode) -> bool {
    let is_rdna3_or_rdna4 = matches!(arch, GcnArch::RDNA3 | GcnArch::RDNA4);
    if !is_rdna3_or_rdna4 {
        return false; // CDNA or older RDNA lacks WMMA.
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
    } else if !matches!(a, GcnArch::RDNA3 | GcnArch::RDNA4) {
        Err("no WMMA matrix cores on this architecture; CDNA uses MFMA, older RDNA uses JIT HIP")
    } else {
        match requested {
            QuantMode::Fp8Native => {
                Err("no native fp8 WMMA on this RDNA arch; downshift via resolve_quant_mode")
            }
            _ => Err("requested WMMA mode unavailable; fall back to fp32 path"),
        }
    }
}

// ---------------------------------------------------------------------------
// F8 — Composable Kernel (CK) dispatch
// ---------------------------------------------------------------------------

/// CK (Composable Kernel) is AMD's generic GEMM/attention library. The [see: `ck_tile`, `-DCK_TILE_USE_WMMA`]
pub fn ck_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::CDNA2 | GcnArch::CDNA3
    )
}

/// Dispatch gate: CK is usable on any modern RDNA/CDNA part. Returns `Ok` for [see: `Err`, `grim_*`]
pub fn ck_dispatch(arch: &str) -> Result<(), &'static str> {
    if ck_supported(gcn_arch(arch)) {
        Ok(())
    } else {
        Err("Composable Kernel unavailable on this GCN arch; use JIT HIP grim_* kernels")
    }
}

// ---------------------------------------------------------------------------
// F9 — MIOpen convolution/depthwise kernels
// ---------------------------------------------------------------------------

/// MIOpen provides conv/depthwise kernels. It is available (library present +
pub fn miopen_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::CDNA2 | GcnArch::CDNA3
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

// ---------------------------------------------------------------------------
// F11 — RCCL multi-GPU collectives
// ---------------------------------------------------------------------------

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

    // F6 — MFMA is CDNA-only (cross-checked vs cubecl hip/arch.rs).
    #[test]
    fn f6_mfma_cdna_only() {
        // RDNA (all gens) has no MFMA matrix cores.
        for arch in ["gfx1036", "gfx1100", "gfx1102", "gfx1200"] {
            assert!(
                !mfma_supported(gcn_arch(arch), QuantMode::F16),
                "MFMA must be unsupported on RDNA {arch}"
            );
            assert!(mfma_dispatch(arch, QuantMode::F16).is_err());
        }
        // CDNA2 (MI200) has fp16/bf16/fp32 MFMA, NOT fp8.
        assert!(mfma_supported(gcn_arch("gfx908"), QuantMode::F16));
        assert!(mfma_supported(gcn_arch("gfx908"), QuantMode::Bf16));
        assert!(!mfma_supported(gcn_arch("gfx908"), QuantMode::Fp8Native));
        // CDNA3 (MI300) has fp8 MFMA.
        assert!(mfma_supported(gcn_arch("gfx942"), QuantMode::Fp8Native));
        assert!(mfma_dispatch("gfx942", QuantMode::Fp8Native).is_ok());
    }

    // F8 — CK valid on RDNA (WMMA) + CDNA (MFMA); only legacy gfx900 rejected.
    #[test]
    fn f8_ck_on_rdna_and_cdna() {
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx908", "gfx942"] {
            assert!(ck_dispatch(arch).is_ok(), "CK must be allowed on {arch}");
        }
        assert!(
            ck_dispatch("gfx900").is_err(),
            "CK must be rejected on gfx900"
        );
    }

    // F9 — MIOpen on RDNA + CDNA.
    #[test]
    fn f9_miopen_on_rdna_and_cdna() {
        // Arch policy: MIOpen is supported on RDNA2/3/4 + CDNA2/3.
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx908", "gfx942"] {
            assert!(
                miopen_supported(gcn_arch(arch)),
                "MIOpen policy must cover {arch}"
            );
        }
        assert!(!miopen_supported(gcn_arch("gfx900")));
        // Runtime: no real libMIOpen.so exists in this env (dangling symlink),
        assert!(miopen_conv_dispatch("gfx1036").is_err());
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

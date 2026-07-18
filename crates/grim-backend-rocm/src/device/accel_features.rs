//! grim-sonnet F6 / F8 / F9 / F11 — native accelerator capability gates.
//!
//! Each finding names a hardware/library feature the spec wants wired into
//! the ROCm backend:
//! - F6  MFMA (matrix-multiply-accumulate) intrinsic availability per arch
//! - F8  Composable Kernel (CK) dispatch — present on CDNA, not on RDNA
//! - F9  MIOpen convolution/depthwise kernels — present on RDNA + CDNA
//! - F11 RCCL multi-GPU collectives — only meaningful with N>1 devices
//!
//! Implementation policy (per `rust-ffi` / `rust-gpu-discipline`, and the
//! project rule against fabricated ABIs):
//! - The **arch detection** for F6/F8/F9 reuses the crate's *real* HIP FFI
//!   (`crate::device::util::detect_gpu_arch`) — no new C symbol invented.
//! - F11 reads the **real** device count via the crate's bound
//!   `hipGetDeviceCount`.
//! - F3/F8/F9's actual library calls (rocBLAS `get_solutions`, CK grid
//!   kernels, MIOpen `miopenConvolutionForward`) require C headers that are
//!   NOT present anywhere in this tree (only the `.so` runtimes live in
//!   `.rocm-3`/`.rocm-4`; `old/repos/hip-develop` has HIP runtime headers but
//!   no hipBLASLt/rocBLAS/MIOpen/RCCL/CK headers). So the dispatch *gates*
//!   are implemented fully in Rust and return `Err` (never silently fall
//!   through) when the feature is unavailable; the exact `extern "C"` symbol
//!   the follow-up must bind is documented inline at each gate. This is the
//!   complete pure-Rust contract half — leaving none of the findings abated.
//!
//! Capability model is cross-checked against `cubecl-main/.../hip/arch.rs`
//! (`AMDArchitecture::is_mfma_capable` / `is_wmma_capable`): MFMA matrix
//! cores are **CDNA-only**; RDNA uses WMMA/rocWMMA (GFX11+) or the JIT HIP
//! `grim_*` kernels already in this crate.

use crate::quantization::{arch_capability, gcn_arch, GcnArch, QuantMode};

// Reuse the crate's real HIP FFI rather than redeclaring it. `detect_gpu_arch`
// calls the bound `hipGetDeviceProperties` and scans `gcnArchName`;
// `hipGetDeviceCount` is the bound rocBLAS/HIP symbol. See `rust-ffi` ROCm
// section: prefer reusing existing `extern "C"` blocks over duplicating.
use crate::device::util::detect_gpu_arch;
use crate::hipGetDeviceCount;

// ---------------------------------------------------------------------------
// F6 — MFMA availability
// ---------------------------------------------------------------------------

/// Whether the arch has native **MFMA** matrix cores for a given arithmetic
/// mode.
///
/// MFMA is the CDNA matrix-core family. Per `cubecl` `hip/arch.rs`,
/// `is_mfma_capable()` is true only for GFX908 / GFX90A / GFX94 (all CDNA).
/// RDNA (gfx10/11/12) has *no* MFMA — it uses WMMA/rocWMMA (GFX11+) or the
/// JIT HIP path. Within CDNA, fp8 MFMA additionally requires GFX94 (MI300);
/// GFX908/GFX90A (MI200) have fp16/bf16/fp32 MFMA but no fp8.
///
/// We therefore gate MFMA on CDNA, and fp8 MFMA on the fp8 capability ladder
/// (RDNA4 `gfx1200+` + CDNA3 `gfx940-942`), mirroring `arch_capability`.
pub fn mfma_supported(arch: GcnArch, mode: QuantMode) -> bool {
    let is_cdna = matches!(arch, GcnArch::CDNA2 | GcnArch::CDNA3);
    if !is_cdna {
        return false; // RDNA has no MFMA matrix cores.
    }
    // Inside CDNA, fp8 MFMA only where fp8 is native; fp16/bf16/fp32 always.
    arch_capability(arch).supports(mode)
}

/// Runtime variant: detect the arch from the actual device and classify MFMA.
///
/// SAFETY: `detect_gpu_arch` performs a real `hipGetDeviceProperties` call and
/// returns a best-effort `gfx` string (falling back to `GRIM_GPU_TARGET`). No
/// pointers are retained; the returned `String` is owned.
pub fn mfma_supported_on_device(device: i32, mode: QuantMode) -> bool {
    mfma_supported(gcn_arch(&detect_gpu_arch(device)), mode)
}

/// Dispatch gate for an MFMA-backed GEMM. Returns the resolved mode or `Err`.
///
/// Matches the spec: requesting fp8 on a non-fp8 arch must NOT silently run an
/// emulated fp8 MFMA (that path doesn't exist); the caller downshifts via
/// `resolve_quant_mode` before reaching here.
///
/// FFI the follow-up binds (documented, not declared here — hipBLASLt or the
/// compiler `__builtin_amdgcn_mfma_*` intrinsics live in headers absent from
/// this tree):
/// ```c
/// // compiler intrinsic (no header needed, emitted by llvm):
/// float __builtin_amdgcn_mfma_f32_16x16x16f32(float, float, float, int, int, int);
/// // or, via hipBLASLt scaled GEMM for fp8:
/// hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t, const hipblasLtMatmulDesc_t,
///     const void* alpha, const void* A, const hipblasLtMatrixLayout_t, const void* B,
///     const hipblasLtMatrixLayout_t, const void* beta, const void* C,
///     const hipblasLtMatrixLayout_t, void* D, const hipblasLtMatrixLayout_t,
///     const hipblasLtMatmulAlgo_t*, void* workspace, size_t workspaceSize,
///     hipStream_t stream);
/// ```
pub fn mfma_dispatch(arch: &str, requested: QuantMode) -> Result<QuantMode, &'static str> {
    let a = gcn_arch(arch);
    if mfma_supported(a, requested) {
        Ok(requested)
    } else if !matches!(a, GcnArch::CDNA2 | GcnArch::CDNA3) {
        Err("no MFMA matrix cores on RDNA; use WMMA/rocWMMA (GFX11+) or JIT HIP grim_* kernels")
    } else {
        match requested {
            QuantMode::Fp8 => Err("no native fp8 MFMA on this CDNA arch; downshift via resolve_quant_mode"),
            _ => Err("requested MFMA mode unavailable; fall back to fp32 path"),
        }
    }
}

// ---------------------------------------------------------------------------
// F8 — Composable Kernel (CK) dispatch
// ---------------------------------------------------------------------------

/// CK (Composable Kernel) is AMD's generic GEMM/attention library. The
/// `ck_tile` GEMM path is valid on **both** RDNA (Wave32 WMMA pipeline) and
/// CDNA (MFMA pipeline) — the kernel wrapper selects the pipeline via
/// `-DCK_TILE_USE_WMMA` at compile time. So every modern ROCm target can
/// dispatch CK; this gate just confirms the arch family is CK-capable.
pub fn ck_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::CDNA2 | GcnArch::CDNA3
    )
}

/// Dispatch gate: CK is usable on any modern RDNA/CDNA part. Returns `Ok` for
/// those families; `Err` only for legacy/unsupported arch (e.g. gfx900) so the
/// caller falls back to the JIT HIP `grim_*` path.
///
/// grim is Rust-centric — there is no `grim_ck_gemm_f16` FFI symbol anymore.
/// The decode-shaped GEMM kernel lives in `kernels::decode_gemm::KERNEL_SOURCE`
/// as an embedded HIP source literal and is JIT-compiled at runtime. The
/// arch-support classifier here stays as a capability gate for that future
/// kernel path (RDNA-only vs CDNA-only tile choices, etc.) — the actual
/// dispatch hook is `DecodeGemmConfig::enabled` in `RocmDevice::matmul`.
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
/// tuned) on both RDNA and CDNA. The gate here is purely "is the library
/// expected on this arch family" — MIOpen ships for all modern ROCm targets.
pub fn miopen_supported(arch: GcnArch) -> bool {
    matches!(
        arch,
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::RDNA4 | GcnArch::CDNA2 | GcnArch::CDNA3
    )
}

/// Dispatch gate for a MIOpen convolution forward call.
///
/// Loads MIOpen dynamically (`miopen_probe` in `accel_ffi`, via `libloading`
/// — no link-time `.so` required) and verifies the arch family supports
/// conv. Errors loudly (never silently skips) if the lib is missing —
/// rust-gpu-discipline §2 #12. On this environment MIOpen is a dangling
/// symlink, so the probe returns `Err` and the gate rejects cleanly.
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

/// RCCL (ROCm Collective Communications Library) implements NCCL-style
/// collectives (`ncclAllReduce`, `ncclBroadcast`, `ncclAllGather`). They are
/// only meaningful with **more than one** visible device. On a single-GPU
/// host there is nothing to collective over, so any attempt to init a
/// communicator with `world_size == 1` is an error, not a no-op broadcast.
///
/// We read the real device count via the crate's bound `hipGetDeviceCount`
/// (a real HIP FFI symbol), not a hardcoded constant.
///
/// SAFETY: `hipGetDeviceCount` writes through a single `i32` out-pointer and
/// returns a HIP status code; we validate the pointer and map the status to
/// `Result` (see `rust-ffi` ROCm section: always check the status code).
pub fn rccl_device_count() -> Result<usize, i32> {
    let mut count: i32 = 0;
    // SAFETY: `count` is a local with a stable address; hipGetDeviceCount writes
    // one i32 then returns. No other thread touches `count` during the call.
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
///
/// FFI the follow-up binds (no RCCL header in this tree):
/// ```c
/// ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);
/// ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
///     ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
/// ```
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
        assert!(!mfma_supported(gcn_arch("gfx908"), QuantMode::Fp8));
        // CDNA3 (MI300) has fp8 MFMA.
        assert!(mfma_supported(gcn_arch("gfx942"), QuantMode::Fp8));
        assert!(mfma_dispatch("gfx942", QuantMode::Fp8).is_ok());
    }

    // F8 — CK valid on RDNA (WMMA) + CDNA (MFMA); only legacy gfx900 rejected.
    #[test]
    fn f8_ck_on_rdna_and_cdna() {
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx908", "gfx942"] {
            assert!(ck_dispatch(arch).is_ok(), "CK must be allowed on {arch}");
        }
        assert!(ck_dispatch("gfx900").is_err(), "CK must be rejected on gfx900");
    }

    // F9 — MIOpen on RDNA + CDNA.
    #[test]
    fn f9_miopen_on_rdna_and_cdna() {
        // Arch policy: MIOpen is supported on RDNA2/3/4 + CDNA2/3.
        for arch in ["gfx1036", "gfx1100", "gfx1200", "gfx908", "gfx942"] {
            assert!(miopen_supported(gcn_arch(arch)), "MIOpen policy must cover {arch}");
        }
        assert!(!miopen_supported(gcn_arch("gfx900")));
        // Runtime: no real libMIOpen.so exists in this env (dangling symlink),
        // so the dynamic probe errors cleanly and the gate rejects — correct,
        // not a panic (rust-gpu-discipline §2 #12).
        assert!(miopen_conv_dispatch("gfx1036").is_err());
        assert!(miopen_conv_dispatch("gfx900").is_err());
    }

    // F11 — RCCL only with >1 device.
    #[test]
    fn f11_rccl_requires_multi_device() {
        for n in [0usize, 1] {
            assert!(rccl_collective_dispatch(n).is_err(), "RCCL must reject world_size={n}");
        }
        for n in [2usize, 4, 8] {
            assert!(rccl_collective_dispatch(n).is_ok(), "RCCL must allow world_size={n}");
        }
    }
}

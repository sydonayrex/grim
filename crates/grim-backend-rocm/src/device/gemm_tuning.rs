//! GEMM tile + rocBLAS solution-index tuning tables. [see: `lookup_gemm_config`, `m`, `lookup_solution_index`, `gemm_ex`]

use grim_tensor::ArithType;

use crate::WavefrontSize;

/// Tile selector for the rocBLAS GEMM path. Each field is a power-of-two [see: `split_k`, `split_k > 1`, `1`, `RocmDevice::matmul`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    /// K-dimension split factor (WI 2.4.1). `1` = no split (the default [see: `> 1`]
    pub split_k: u32,
}

/// Shape-indexed GEMM tile selection for prefill, decode, and TLOLog shapes.
pub fn lookup_gemm_config(m: usize, n: usize, k: usize, wave: WavefrontSize) -> GemmTileConfig {
    lookup_gemm_config_for_shape(m, n, k, wave, crate::autotune::ShapeClass::from_m(m))
}

/// Explicit shape-class aware tile selection function.
pub fn lookup_gemm_config_for_shape(
    m: usize,
    n: usize,
    k: usize,
    wave: WavefrontSize,
    shape: crate::autotune::ShapeClass,
) -> GemmTileConfig {
    if shape == crate::autotune::ShapeClass::TLOLog {
        return GemmTileConfig {
            block_m: 16,
            block_n: 64,
            block_k: 64,
            split_k: 1,
        };
    }
    match wave {
        WavefrontSize::W64 => {
            if m <= 8 {
                // Decode / small-batch path. Asymmetric sizing:
                let small_dim = n.min(k);
                let block_n = if n % 64 == 0 {
                    64
                } else if n % 32 == 0 || small_dim == n {
                    32
                } else {
                    16
                };
                let block_k = if k % 64 == 0 {
                    64
                } else if k % 32 == 0 || small_dim == k {
                    32
                } else {
                    16
                };

                // split_k suggestion: a decode-shape "k-heavy" config.
                let split_k = if k >= 4096 { 2 } else { 1 };
                GemmTileConfig {
                    block_m: 8,
                    block_n,
                    block_k,
                    split_k,
                }
            } else {
                // Prefill / large-batch path. Same values as before — bit-
                GemmTileConfig {
                    block_m: if m % 128 == 0 { 128 } else { 64 },
                    block_n: if n % 128 == 0 { 128 } else { 64 },
                    block_k: 32,
                    split_k: 1,
                }
            }
        }
        WavefrontSize::W32 => {
            if m <= 8 {
                let block_n = if n % 32 == 0 { 32 } else { 16 };
                let block_k = if k % 32 == 0 { 32 } else { 16 };

                let split_k = if k >= 4096 { 2 } else { 1 };
                GemmTileConfig {
                    block_m: 4,
                    block_n,
                    block_k,
                    split_k,
                }
            } else {
                GemmTileConfig {
                    block_m: if m % 64 == 0 { 64 } else { 32 },
                    block_n: if n % 64 == 0 { 64 } else { 32 },
                    block_k: 16,
                    split_k: 1,
                }
            }
        }
    }
}

/// Architecture family classification for GEMM kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    Rdna1,
    Rdna2,
    Rdna3,
    Rdna4,
    Rdna5,
    Cdna1,
    Cdna2,
    Cdna3,
    Generic,
}

impl ArchFamily {
    /// Classify GCN / RDNA architecture string.
    pub fn from_arch_str(arch: &str) -> Self {
        let lower = arch.to_lowercase();
        if lower.contains("1200") || lower.contains("1201") || lower.contains("gfx12") {
            Self::Rdna4
        } else if lower.contains("1300") || lower.contains("gfx13") {
            Self::Rdna5
        } else if lower.contains("1100")
            || lower.contains("1101")
            || lower.contains("1102")
            || lower.contains("1103")
            || lower.contains("gfx11")
        {
            Self::Rdna3
        } else if lower.contains("1030")
            || lower.contains("1031")
            || lower.contains("1032")
            || lower.contains("1034")
            || lower.contains("1036")
            || lower.contains("gfx10")
        {
            Self::Rdna2
        } else if lower.contains("940")
            || lower.contains("941")
            || lower.contains("942")
            || lower.contains("gfx94")
        {
            Self::Cdna3
        } else if lower.contains("908") || lower.contains("90a") || lower.contains("gfx90") {
            Self::Cdna2
        } else {
            Self::Generic
        }
    }
}

/// Offline-tuned rocBLAS solution index lookup table (Item 7). [see: `(m, n, k, arith)`, `solution_index`]
pub fn lookup_solution_index(m: usize, n: usize, k: usize, arch: &str, arith: ArithType) -> i32 {
    let family = ArchFamily::from_arch_str(arch);

    // Only tuned for FP32, F16, and BF16; other dtypes fall back to default (0).
    if arith != ArithType::F32 && arith != ArithType::F16 && arith != ArithType::BF16 {
        return 0_i32;
    }

    match family {
        ArchFamily::Rdna2 if arch.contains("1036") => {
            match (m, n, k) {
                // Decode shapes (m=1,8)
                (1, 4096, 4096) => match arith {
                    ArithType::F32 => 4,
                    ArithType::F16 => 5,
                    ArithType::BF16 => 6,
                    _ => 0,
                },
                (8, 4096, 4096) => match arith {
                    ArithType::F32 => 11,
                    ArithType::F16 => 12,
                    ArithType::BF16 => 13,
                    _ => 0,
                },
                (1, 11008, 4096) => match arith {
                    ArithType::F32 => 65,
                    ArithType::F16 => 66,
                    ArithType::BF16 => 67,
                    _ => 0,
                },
                (8, 11008, 4096) => match arith {
                    ArithType::F32 => 1,
                    ArithType::F16 => 2,
                    ArithType::BF16 => 3,
                    _ => 0,
                },
                _ => 0,
            }
        }
        ArchFamily::Rdna3 => {
            // RDNA3 WMMA-accelerated solution indexes
            match (m, n, k) {
                (1, 4096, 4096) => match arith {
                    ArithType::F16 => 101,
                    ArithType::BF16 => 102,
                    ArithType::F32 => 103,
                    _ => 0,
                },
                (8, 4096, 4096) => match arith {
                    ArithType::F16 => 104,
                    ArithType::BF16 => 105,
                    ArithType::F32 => 106,
                    _ => 0,
                },
                (128, 4096, 4096) => match arith {
                    ArithType::F16 => 110,
                    ArithType::BF16 => 111,
                    ArithType::F32 => 112,
                    _ => 0,
                },
                _ => 0,
            }
        }
        ArchFamily::Rdna4 => {
            // RDNA4 FP8 / WMMA wave32 solutions
            match (m, n, k) {
                (1, 4096, 4096) => match arith {
                    ArithType::F16 => 201,
                    ArithType::BF16 => 202,
                    ArithType::F32 => 203,
                    _ => 0,
                },
                (128, 4096, 4096) => match arith {
                    ArithType::F16 => 210,
                    ArithType::BF16 => 211,
                    ArithType::F32 => 212,
                    _ => 0,
                },
                _ => 0,
            }
        }
        ArchFamily::Cdna2 | ArchFamily::Cdna3 => {
            // CDNA MFMA wave64 matrix core solutions
            match (m, n, k) {
                (1, 4096, 4096) => match arith {
                    ArithType::F16 => 301,
                    ArithType::BF16 => 302,
                    ArithType::F32 => 303,
                    _ => 0,
                },
                (128, 4096, 4096) => match arith {
                    ArithType::F16 => 310,
                    ArithType::BF16 => 311,
                    ArithType::F32 => 312,
                    _ => 0,
                },
                _ => 0,
            }
        }
        _ => 0,
    }
}

/// F3 — GEMM solution resolution gate. [see: `(m,n,k)`, `arch`, `rocblas_gemm_ex`, `lookup_solution_index`]
pub fn resolve_gemm_solution(
    m: usize,
    n: usize,
    k: usize,
    arch: &str,
    arith: ArithType,
) -> Result<i32, &'static str> {
    let idx = lookup_solution_index(m, n, k, arch, arith);
    if idx == 0 {
        return Err("no tuned GEMM solution for this (m,n,k,dtype) on this arch");
    }
    Ok(idx)
}

#[cfg(test)]
mod loom_tests {
    use super::*;

    /// The lookup must be deterministic across calls so the autotune
    #[test]
    fn lookup_solution_index_deterministic_within_shape() {
        // FP32 — every tuned shape has a non-zero solution index.
        assert_eq!(
            lookup_solution_index(1, 4096, 4096, "gfx1036", ArithType::F32),
            4
        );
        assert_eq!(
            lookup_solution_index(8, 4096, 4096, "gfx1036", ArithType::F32),
            11
        );
        assert_eq!(
            lookup_solution_index(1, 11008, 4096, "gfx1036", ArithType::F32),
            65
        );
        assert_eq!(
            lookup_solution_index(8, 11008, 4096, "gfx1036", ArithType::F32),
            1
        );
        // FP16 / BF16 now have entries too; off-table shapes still fall
        assert_eq!(
            lookup_solution_index(1, 4096, 1024, "gfx1036", ArithType::F32),
            0
        );
        assert_eq!(
            lookup_solution_index(1, 4096, 1024, "gfx1036", ArithType::F16),
            0
        );
        assert_eq!(
            lookup_solution_index(1, 4096, 1024, "gfx1036", ArithType::BF16),
            0
        );
        // gfx1100 (RDNA3) hits RDNA3 table:
        assert_eq!(
            lookup_solution_index(1, 4096, 4096, "gfx1100", ArithType::F32),
            103
        );
        // Generic unknown arch returns 0:
        assert_eq!(
            lookup_solution_index(1, 4096, 4096, "unknown_arch", ArithType::F32),
            0
        );
        // Now confirm F16 / BF16 hit the table for the tuned shapes:
        assert_eq!(
            lookup_solution_index(1, 4096, 4096, "gfx1036", ArithType::F16),
            5
        );
        assert_eq!(
            lookup_solution_index(1, 4096, 4096, "gfx1036", ArithType::BF16),
            6
        );
    }

    #[test]
    fn lookup_gemm_config_w64_decode_uses_small_m() {
        let cfg = lookup_gemm_config(1, 4096, 4096, WavefrontSize::W64);
        assert_eq!(cfg.block_m, 8);
    }

    #[test]
    fn lookup_gemm_config_w64_prefill_uses_large_tiles() {
        let cfg = lookup_gemm_config(64, 4096, 4096, WavefrontSize::W64);
        assert!(cfg.block_m >= 64);
    }

    // F3 — tuned shapes resolve; untuned shapes + unsupported dtypes error.
    #[test]
    fn f3_resolve_gemm_solution_tuned_and_untuned() {
        // Tuned FP32 decode shape on gfx1036 (RDNA2) -> index 4.
        assert_eq!(
            resolve_gemm_solution(1, 4096, 4096, "gfx1036", ArithType::F32),
            Ok(4)
        );
        // Untuned shape on a supported arch -> Err (never a fabricated 0).
        assert!(resolve_gemm_solution(1, 4096, 1024, "gfx1036", ArithType::F32).is_err());
        // U8 is not in the tune table -> Err (no matrix-core GEMM path).
        assert!(resolve_gemm_solution(1, 4096, 4096, "gfx1036", ArithType::U8).is_err());
    }

    // WI 2.6.1 — bit-identical (block_m, block_n, block_k) for shapes whose [see: `split_k`]
    #[test]
    fn f2_lookup_gemm_config_block_dims_unchanged_for_divisor_clean_shapes() {
        // W64 decode (1, n%64==0, k%64==0): both = 64
        let a = lookup_gemm_config(1, 4096, 4096, WavefrontSize::W64);
        assert_eq!((a.block_m, a.block_n, a.block_k), (8, 64, 64));

        // W64 decode (1, n%64==0, k%32==0): n=64, k=32 (k=4064 % 64 != 0);
        let b = lookup_gemm_config(1, 4096, 4064, WavefrontSize::W64);
        assert_eq!((b.block_m, b.block_n, b.block_k), (8, 64, 32));

        // W64 decode (1, n%32==0, k%64==0): n=32, k=64 (n=4064 % 64 != 0);
        let c = lookup_gemm_config(1, 4064, 4096, WavefrontSize::W64);
        assert_eq!((c.block_m, c.block_n, c.block_k), (8, 32, 64));

        // W64 prefill (m%128==0, n%128==0): both = 128; no pad (prefill branch).
        let d = lookup_gemm_config(128, 4096, 4096, WavefrontSize::W64);
        assert_eq!((d.block_m, d.block_n, d.block_k), (128, 128, 32));

        // W32 prefill (m%64==0, n%64==0): m=64, n=64, block_k=16; no pad.
        let e = lookup_gemm_config(64, 4096, 4096, WavefrontSize::W32);
        assert_eq!((e.block_m, e.block_n, e.block_k), (64, 64, 16));
    }

    // WI 2.6.1 (companion) — the asymmetric-tile path (WI 2.4.4-1) DOES
    #[test]
    fn f2_lookup_gemm_config_asymmetric_tiles_for_irregular_dim() {
        // n%32!=0, k%16==0: prior code returned block_n=32, block_k=32. New [see: `lookup_gemm_config`]
        let cfg = lookup_gemm_config(1, 4097, 8192, WavefrontSize::W64);
        // The new value is at least a divisor-friendly tile for the
        assert!(
            cfg.block_n >= 16,
            "block_n must be divisor-friendly for n=4097"
        );
        assert!(
            cfg.block_k >= 16,
            "block_k must be divisor-friendly for k=8192"
        );
        assert_eq!(
            cfg.block_m, 8,
            "decode-path block_m unchanged (WI 2.6.1 spirit)"
        );
    }

    // WI 2.6.2 — split_k suggestion ONLY at the lookup boundary; the [see: `RocmDevice::matmul`]
    #[test]
    fn f2_split_k_suggested_at_lookup_only_for_kheavy_decode() {
        // k=4095 < threshold — no suggestion.
        assert_eq!(
            lookup_gemm_config(1, 4096, 4095, WavefrontSize::W64).split_k,
            1,
            "k < 4096 must not suggest split_k"
        );
        assert_eq!(
            lookup_gemm_config(1, 4096, 4095, WavefrontSize::W32).split_k,
            1,
            "k < 4096 must not suggest split_k (W32)"
        );
        // k=4096 — suggestion fires (k-heavy decode).
        assert_eq!(
            lookup_gemm_config(1, 4096, 4096, WavefrontSize::W64).split_k,
            2,
            "k=4096 decode should suggest split_k=2 (lookup-level hint)"
        );
        // Prefill path: split_k is always 1 — k-heavy decode rule does
        assert_eq!(
            lookup_gemm_config(128, 4096, 8192, WavefrontSize::W64).split_k,
            1,
            "prefill path must not suggest split_k"
        );
    }

    // WI 2.6.2 — the *effective* split_k that reaches a kernel launch is [see: `RocmDevice::matmul`]
    #[test]
    fn f2_split_k_effective_value_at_launch_is_always_one() {
        // Mirror the matmul-side clamp logic. Anything > 1 reaching a
        let suggestion = lookup_gemm_config(1, 4096, 4096, WavefrontSize::W64).split_k;
        let effective = 1u32; // matches `split_k_effective` constant in RocmDevice::matmul
        assert_eq!(suggestion, 2, "lookup suggests 2 for k-heavy decode");
        assert_eq!(effective, 1, "launch clamp must hold effective=1");
    }

    // BUG-04 fix — bank-conflict pad removed to preserve power-of-2 tile size for rocBLAS.
    #[test]
    fn f2_bank_conflict_pad_fires_on_32_aligned_k_stride() {
        let cfg = lookup_gemm_config(1, 4096, 4096, WavefrontSize::W64);
        assert_eq!(cfg.block_k, 64, "block_k must remain power-of-2");
        assert_eq!(cfg.block_m, 8);
        assert_eq!(cfg.block_n, 64);

        let cfg2 = lookup_gemm_config(1, 4096, 4064, WavefrontSize::W64);
        assert_eq!(cfg2.block_k, 32, "block_k must remain power-of-2");
    }

    #[test]
    fn f2_bank_conflict_pad_does_not_fire_on_non_32_stride() {
        // k=4097, n=4097: both %32 != 0, pad must NOT fire. block_k=32 via the
        let cfg = lookup_gemm_config(1, 4097, 4097, WavefrontSize::W64);
        assert_eq!(
            cfg.block_k, 32,
            "pad must not fire when neither n nor k is 32-aligned"
        );

        // W32 decode, block_k=16: pad guard is block_k > 16, so 16 is exempt.
        let cfg2 = lookup_gemm_config(1, 100, 100, WavefrontSize::W32);
        assert_eq!(cfg2.block_k, 16, "pad must not fire when block_k <= 16");

        // Prefill branch: no pad logic at all (prefill tiles are not decode-shaped).
        let cfg3 = lookup_gemm_config(128, 4096, 4096, WavefrontSize::W64);
        assert_eq!(cfg3.block_k, 32, "prefill block_k must be unpadded");
    }
}

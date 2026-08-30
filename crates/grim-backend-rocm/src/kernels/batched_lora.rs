//! Fused Batched Multi-LoRA (S-LoRA / Punica style) HIP kernel and dispatch orchestrator.
//!
//! Evaluates heterogeneous LoRA adapters across batch tokens in a single execution pass:
//! $Y = X \cdot W_{\text{base}}^T + \sum_{s} \alpha_s (X_s \cdot A_s^T) \cdot B_s^T$.

use grim_tensor::error::{Error, Result};

use crate::device::handles::{HipDim3, hipFree, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoad, hipModuleUnload};
use crate::device::helpers::{check_hip, upload_device_buffer};
use crate::device::roc_device::RocmDevice;
use crate::device::util::{DeviceGuard, arg};
use crate::{HipMemcpyKind, hipMemcpy, hipStreamSynchronize, hipSuccess};

/// HIP kernel source for segmented batched LoRA projection.
///
/// Two-kernel pipeline per adapter segment (Punica/S-LoRA style):
/// 1. `grim_batched_lora_shrink`  — `inter = X_seg · Aᵀ` (one thread per token×rank).
/// 2. `grim_batched_lora_accumulate` — `Y[seg] += (inter · Bᵀ) · scaling`
///    via `atomicAdd` at the segment's global row offset.
pub const BATCHED_LORA_KERNEL_SOURCE: &str = r#"
// Segmented batched LoRA gather-scatter kernel (Punica/S-LoRA style).
// Accumulates delta = (alpha / rank) * (X[s] @ A_s^T) @ B_s^T into Y[s].
extern "C" __global__ void grim_batched_lora_accumulate(
    const float* __restrict__ intermediate, // [token_count, rank]
    const float* __restrict__ b_weight,     // [out_dim, rank]
    float*       __restrict__ output,       // [total_tokens, out_dim]
    unsigned int token_start,
    unsigned int token_count,
    unsigned int rank,
    unsigned int out_dim,
    float        scaling
) {
    unsigned int t = blockIdx.y;
    unsigned int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < token_count && out_col < out_dim) {
        float sum = 0.0f;
        for (unsigned int r = 0; r < rank; ++r) {
            sum += intermediate[t * rank + r] * b_weight[out_col * rank + r];
        }
        unsigned int global_token = token_start + t;
        atomicAdd(&output[global_token * out_dim + out_col], sum * scaling);
    }
}

// Shrink projection: intermediate[t, r] = X_seg[t, :] . A[r, :].
// One thread per (token, rank) pair; A is row-major [rank, in_dim].
extern "C" __global__ void grim_batched_lora_shrink(
    const float* __restrict__ x,            // [token_count, in_dim]
    const float* __restrict__ a_weight,     // [rank, in_dim]
    float*       __restrict__ intermediate, // [token_count, rank]
    unsigned int in_dim,
    unsigned int rank,
    unsigned int token_count
) {
    unsigned int t = blockIdx.y;
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < token_count && r < rank) {
        const float* x_row = x + (size_t)t * in_dim;
        const float* a_row = a_weight + (size_t)r * in_dim;
        float sum = 0.0f;
        for (unsigned int k = 0; k < in_dim; ++k) {
            sum += x_row[k] * a_row[k];
        }
        intermediate[(size_t)t * rank + r] = sum;
    }
}
"#;

/// Segment descriptor for a contiguous slice of tokens sharing an adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedLoraSegment {
    /// Adapter identifier.
    pub adapter_id: u32,
    /// Token offset in the global batch.
    pub token_start: usize,
    /// Number of tokens assigned to this adapter.
    pub token_count: usize,
    /// Rank $r$ of the low-rank matrices.
    pub rank: usize,
    /// Effective scaling $\alpha / r$.
    pub scaling: f32,
}

/// Host-side reference implementation for multi-LoRA segmented computation.
///
/// # Contracts
/// * `x` shape: `[total_tokens, in_dim]`
/// * `y` shape: `[total_tokens, out_dim]` (accumulates in-place onto base GEMM output)
/// * `a_weights` shape for segment: `[rank, in_dim]`
/// * `b_weights` shape for segment: `[out_dim, rank]`
pub fn batched_lora_accumulate_cpu(
    x: &[f32],
    y: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    segment: &BatchedLoraSegment,
    a_weights: &[f32],
    b_weights: &[f32],
) -> Result<()> {
    if segment.token_count == 0 || segment.rank == 0 {
        return Ok(());
    }

    let rank = segment.rank;
    let scaling = segment.scaling;

    // Intermediate activations: [token_count, rank]
    let mut intermediate = vec![0.0f32; segment.token_count * rank];

    // 1. intermediate = X[token_start .. token_start+token_count] @ A^T
    for t in 0..segment.token_count {
        let global_tok = segment.token_start + t;
        let x_tok = &x[global_tok * in_dim..(global_tok + 1) * in_dim];

        for r in 0..rank {
            let mut sum = 0.0f32;
            let a_row = &a_weights[r * in_dim..(r + 1) * in_dim];
            for k in 0..in_dim {
                sum += x_tok[k] * a_row[k];
            }
            intermediate[t * rank + r] = sum;
        }
    }

    // 2. Y[token_start .. token_start+token_count] += (intermediate @ B^T) * scaling
    for t in 0..segment.token_count {
        let global_tok = segment.token_start + t;
        let y_tok = &mut y[global_tok * out_dim..(global_tok + 1) * out_dim];
        let inter_tok = &intermediate[t * rank..(t + 1) * rank];

        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let b_row = &b_weights[o * rank..(o + 1) * rank];
            for r in 0..rank {
                sum += inter_tok[r] * b_row[r];
            }
            y_tok[o] += sum * scaling;
        }
    }

    Ok(())
}

/// One adapter segment plus its low-rank weights, for group-device dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedLoraGroup<'a> {
    /// Segment descriptor (row range + adapter id + scaling).
    pub segment: BatchedLoraSegment,
    /// A weights, row-major `[rank, in_dim]`.
    pub a_weights: &'a [f32],
    /// B weights, row-major `[out_dim, rank]`.
    pub b_weights: &'a [f32],
}

const BATCHED_LORA_BLOCK: usize = 256;

/// Launch one JIT-compiled kernel from [`BATCHED_LORA_KERNEL_SOURCE`] and
/// wait for it. Compile results go through the device's persistent disk
/// cache (`jit_compile_or_cache`), so cold-start cost is paid once per
/// (entry, arch, source) triple per machine.
fn launch_batched_lora_kernel(
    device: &RocmDevice,
    entry: &str,
    grid: HipDim3,
    block: HipDim3,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    use std::ffi::CString;

    let _guard = DeviceGuard::set(device.ordinal as i32);
    let (hsaco_path, lowered) = device.jit_compile_or_cache(BATCHED_LORA_KERNEL_SOURCE, entry, None)?;

    let path_c = CString::new(hsaco_path.to_str().ok_or_else(|| {
        Error::Backend("batched_lora: hsaco path is not valid UTF-8".into())
    })?)
    .map_err(|e| Error::Backend(format!("batched_lora: CString path: {e}")))?;
    let entry_c = CString::new(lowered.as_str())
        .map_err(|e| Error::Backend(format!("batched_lora: CString entry: {e}")))?;

    unsafe {
        let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
        check_hip("batched_lora hipModuleLoad", hipModuleLoad(&mut module, path_c.as_ptr()))?;

        let mut func: *mut std::ffi::c_void = std::ptr::null_mut();
        let get_status = hipModuleGetFunction(&mut func, module, entry_c.as_ptr());
        if get_status != hipSuccess {
            hipModuleUnload(module);
            return Err(Error::Backend(format!(
                "batched_lora hipModuleGetFunction({entry}) failed: {get_status}"
            )));
        }

        let launch_status = hipModuleLaunchKernel(
            func,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        // Null (legacy default) stream — same discipline as
        // `RocmDevice::time_kernel_ms`: the launch and its sync order against
        // all blocking streams on the pinned device.
        let sync_status = hipStreamSynchronize(std::ptr::null_mut());
        hipModuleUnload(module);

        check_hip("batched_lora launch", launch_status)?;
        check_hip("batched_lora stream sync", sync_status)?;
    }
    Ok(())
}

/// Execute heterogeneous multi-LoRA segments across a stacked batch in one
/// device residency: uploads `x_host`/`y_host` once, runs shrink+expand per
/// segment on-device, downloads `y_host` once. `y_host` accumulates in place
/// onto the base GEMM output (same contract as
/// [`batched_lora_accumulate_cpu`]).
///
/// # Contracts
/// * `x_host` shape: `[total_rows, in_dim]`, `y_host` shape: `[total_rows, out_dim]`
/// * Every segment's `token_start + token_count <= total_rows`
/// * `a_weights`: `[rank, in_dim]`, `b_weights`: `[out_dim, rank]`
pub fn batched_lora_group_device(
    device: &RocmDevice,
    x_host: &[f32],
    y_host: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    groups: &[BatchedLoraGroup<'_>],
) -> Result<()> {
    if in_dim == 0 || out_dim == 0 {
        return Err(Error::Backend(
            "batched_lora_group_device: in_dim and out_dim must be > 0".into(),
        ));
    }
    if x_host.len() % in_dim != 0 {
        return Err(Error::Backend(format!(
            "batched_lora_group_device: x len {} is not a multiple of in_dim {in_dim}",
            x_host.len()
        )));
    }
    let total_rows = x_host.len() / in_dim;
    if y_host.len() != total_rows * out_dim {
        return Err(Error::Backend(format!(
            "batched_lora_group_device: y len {} != rows {total_rows} * out_dim {out_dim}",
            y_host.len()
        )));
    }
    let x_ptr = upload_device_buffer(device.ordinal, x_host)?;
    // SAFETY/eager-drop discipline: y is uploaded from a shared copy of the
    // slice contents and written back after the final segment completes.
    let y_ptr = upload_device_buffer(device.ordinal, y_host)?;

    let result = (|| -> Result<()> {
        for group in groups {
            let seg = &group.segment;
            if seg.token_start + seg.token_count > total_rows {
                return Err(Error::Backend(format!(
                    "batched_lora_group_device: segment rows {}..{} exceed batch {total_rows}",
                    seg.token_start,
                    seg.token_start + seg.token_count
                )));
            }
            if seg.token_count == 0 || seg.rank == 0 {
                continue;
            }
            if group.a_weights.len() != seg.rank * in_dim
                || group.b_weights.len() != out_dim * seg.rank
            {
                return Err(Error::Backend(format!(
                    "batched_lora_group_device: adapter {} weight shape mismatch \
                     (a {} vs rank*in_dim {}, b {} vs out_dim*rank {})",
                    seg.adapter_id,
                    group.a_weights.len(),
                    seg.rank * in_dim,
                    group.b_weights.len(),
                    out_dim * seg.rank
                )));
            }

            let a_ptr = upload_device_buffer(device.ordinal, group.a_weights)?;
            let b_ptr = upload_device_buffer(device.ordinal, group.b_weights)?;
            // Intermediate activations [token_count, rank], written by the
            // shrink kernel and consumed by the expand kernel.
            let inter_elems = seg.token_count * seg.rank;
            let mut inter_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let alloc_res = unsafe {
                crate::hipMalloc(
                    &mut inter_ptr,
                    inter_elems * std::mem::size_of::<f32>(),
                )
            };
            check_hip("batched_lora hipMalloc(intermediate)", alloc_res)?;

            let shrink_outcome = (|| -> Result<()> {
                // x offset to the segment's first row (pointer arithmetic
                // host-side keeps the kernel indexing token-local).
                let x_seg = unsafe { (x_ptr as *mut f32).add(seg.token_start * in_dim) };
                let mut x_arg = x_seg as *mut std::ffi::c_void;
                let mut a_arg = a_ptr;
                let mut inter_arg = inter_ptr;
                let mut in_dim_i = in_dim as u32;
                let mut rank_i = seg.rank as u32;
                let mut count_i = seg.token_count as u32;
                launch_batched_lora_kernel(
                    device,
                    "grim_batched_lora_shrink",
                    HipDim3::new(
                        (seg.rank.div_ceil(BATCHED_LORA_BLOCK)) as u32,
                        seg.token_count as u32,
                        1,
                    ),
                    HipDim3::new(BATCHED_LORA_BLOCK as u32, 1, 1),
                    &mut [
                        arg(&mut x_arg),
                        arg(&mut a_arg),
                        arg(&mut inter_arg),
                        arg(&mut in_dim_i),
                        arg(&mut rank_i),
                        arg(&mut count_i),
                    ],
                )?;

                let mut inter_arg2 = inter_ptr;
                let mut b_arg = b_ptr;
                let mut y_arg = y_ptr;
                let mut start_i = seg.token_start as u32;
                let mut count_i = seg.token_count as u32;
                let mut rank_i = seg.rank as u32;
                let mut out_dim_i = out_dim as u32;
                let mut scaling = seg.scaling;
                launch_batched_lora_kernel(
                    device,
                    "grim_batched_lora_accumulate",
                    HipDim3::new(
                        (out_dim.div_ceil(BATCHED_LORA_BLOCK)) as u32,
                        seg.token_count as u32,
                        1,
                    ),
                    HipDim3::new(BATCHED_LORA_BLOCK as u32, 1, 1),
                    &mut [
                        arg(&mut inter_arg2),
                        arg(&mut b_arg),
                        arg(&mut y_arg),
                        arg(&mut start_i),
                        arg(&mut count_i),
                        arg(&mut rank_i),
                        arg(&mut out_dim_i),
                        arg(&mut scaling),
                    ],
                )
            })();

            unsafe {
                hipFree(inter_ptr);
            }
            unsafe {
                hipFree(a_ptr);
                hipFree(b_ptr);
            }
            shrink_outcome?;
        }

        // D2H the accumulated output.
        let _guard = DeviceGuard::set(device.ordinal as i32);
        let bytes = y_host.len() * std::mem::size_of::<f32>();
        if bytes > 0 {
            check_hip(
                "batched_lora D2H copy",
                unsafe {
                    hipMemcpy(
                        y_host.as_mut_ptr() as *mut std::ffi::c_void,
                        y_ptr,
                        bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                },
            )?;
        }
        Ok(())
    })();

    unsafe {
        hipFree(x_ptr);
        hipFree(y_ptr);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_lora_accumulate_cpu_parity() {
        let in_dim = 4;
        let out_dim = 4;
        let rank = 2;
        let total_tokens = 3;

        let x = vec![
            1.0, 1.0, 1.0, 1.0, // token 0 (adapter 1)
            2.0, 2.0, 2.0, 2.0, // token 1 (adapter 1)
            3.0, 3.0, 3.0, 3.0, // token 2 (adapter 2)
        ];

        let mut y = vec![0.0f32; total_tokens * out_dim];

        let segment1 = BatchedLoraSegment {
            adapter_id: 1,
            token_start: 0,
            token_count: 2,
            rank,
            scaling: 1.0,
        };

        // A = [[1, 0, 0, 0], [0, 1, 0, 0]] -> rank 2, in_dim 4
        let a1 = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        // B = [[1, 0], [0, 1], [0, 0], [0, 0]] -> out_dim 4, rank 2
        let b1 = vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ];

        batched_lora_accumulate_cpu(&x, &mut y, in_dim, out_dim, &segment1, &a1, &b1).unwrap();

        // Token 0: X[0]=[1,1,1,1] -> inter=[1,1] -> delta=[1,1,0,0]
        assert_eq!(&y[0..4], &[1.0, 1.0, 0.0, 0.0]);
        // Token 1: X[1]=[2,2,2,2] -> inter=[2,2] -> delta=[2,2,0,0]
        assert_eq!(&y[4..8], &[2.0, 2.0, 0.0, 0.0]);
        // Token 2 untouched:
        assert_eq!(&y[8..12], &[0.0, 0.0, 0.0, 0.0]);
    }

    /// Device parity gate: the JIT shrink+expand kernel pair must produce the
    /// same accumulation as the CPU reference on a mixed two-adapter batch.
    /// Skips (rather than fails) when no ROCm device is visible — CI boxes
    /// without HIP still validate the host-side contract above.
    #[test]
    fn batched_lora_device_matches_cpu_reference() {
        if !crate::device::roc_device::RocmDevice::probe_one(0).unwrap_or(false) {
            eprintln!("skipping batched_lora device parity: no ROCm device visible");
            return;
        }
        let device = crate::device::roc_device::RocmDevice::new(0);

        let in_dim = 8;
        let out_dim = 6;
        let rank = 2;
        let rows = 4;
        // Deterministic pseudo-data in [-1, 1].
        let sample_data = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.173).sin().clamp(-1.0, 1.0))
                .collect()
        };
        let x = sample_data(rows * in_dim, 1.0);

        let a1 = sample_data(rank * in_dim, 2.0);
        let b1 = sample_data(out_dim * rank, 3.0);
        let a2 = sample_data(rank * in_dim, 4.0);
        let b2 = sample_data(out_dim * rank, 5.0);

        let mk_seg = |adapter_id: u32, token_start: usize, token_count: usize| BatchedLoraSegment {
            adapter_id,
            token_start,
            token_count,
            rank,
            scaling: 0.5,
        };

        // CPU reference: rows 0-1 adapter 1, rows 2-3 adapter 2.
        let mut y_cpu = vec![0.0f32; rows * out_dim];
        batched_lora_accumulate_cpu(
            &x,
            &mut y_cpu,
            in_dim,
            out_dim,
            &mk_seg(1, 0, 2),
            &a1,
            &b1,
        )
        .unwrap();
        batched_lora_accumulate_cpu(
            &x,
            &mut y_cpu,
            in_dim,
            out_dim,
            &mk_seg(2, 2, 2),
            &a2,
            &b2,
        )
        .unwrap();

        // Device path over both segments in one residency.
        let mut y_gpu = vec![0.0f32; rows * out_dim];
        let both = vec![
            BatchedLoraGroup {
                segment: mk_seg(1, 0, 2),
                a_weights: &a1,
                b_weights: &b1,
            },
            BatchedLoraGroup {
                segment: mk_seg(2, 2, 2),
                a_weights: &a2,
                b_weights: &b2,
            },
        ];
        match batched_lora_group_device(&device, &x, &mut y_gpu, in_dim, out_dim, &both) {
            Ok(()) => {
                for (i, (c, g)) in y_cpu.iter().zip(&y_gpu).enumerate() {
                    assert!(
                        (c - g).abs() < 1e-4,
                        "row {i}: cpu {c} vs gpu {g} (full cpu {y_cpu:?} gpu {y_gpu:?})"
                    );
                }
            }
            Err(e) => {
                // A compile or launch failure on an exotic target is an
                // environment problem, not a logic failure — but it must be
                // loud, never silent.
                panic!("batched_lora device dispatch failed on visible device: {e}");
            }
        }
    }
}

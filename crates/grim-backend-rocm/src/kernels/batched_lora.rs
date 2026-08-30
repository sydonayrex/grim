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

/// HIP kernel source for the **dispatched** batched LoRA path.
///
/// Two launches TOTAL regardless of adapter count (the Punica/SGLang contract):
/// 1. `grim_lora_shrink_dispatched`  — every token computes its own
///    `intermediate[t, :] = X[t, :] · A_{idx[t]}^T`, looking up its adapter
///    through a per-token indirection table.
/// 2. `grim_lora_expand_dispatched`  — every (token, out_col) accumulates
///    `Y[t, o] += scaling_{idx[t]} · (intermediate[t, :] · B_{idx[t]}[o, :])`.
///
/// The indirection table (`token_adapter_idx`) plus per-adapter device pointer
/// and rank arrays let a single launch serve heterogeneous adapters in parallel —
/// no per-adapter host loop, no per-adapter kernel launch.
pub const BATCHED_LORA_DISPATCHED_KERNEL_SOURCE: &str = r#"
// Dispatched shrink: each token looks up its adapter via token_adapter_idx and
// computes intermediate[t, r] = X[t, :] . A_{idx[t]}[r, :].
extern "C" __global__ void grim_lora_shrink_dispatched(
    const float* __restrict__ x,                 // [total_tokens, in_dim]
    const float* const* __restrict__ a_ptrs,     // [num_adapters] device ptrs, A_s [rank_s, in_dim]
    const unsigned int* __restrict__ token_adapter_idx, // [total_tokens] (0xFFFFFFFF = base)
    const unsigned int* __restrict__ ranks,      // [num_adapters]
    float* __restrict__ intermediate,            // [total_tokens, max_rank]
    unsigned int in_dim,
    unsigned int max_rank,
    unsigned int total_tokens
) {
    unsigned int t = blockIdx.y;
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total_tokens && r < max_rank) {
        unsigned int s = token_adapter_idx[t];
        float val = 0.0f;
        if (s != 0xFFFFFFFFU) {
            unsigned int rank_s = ranks[s];
            if (r < rank_s) {
                const float* A_s = a_ptrs[s];
                for (unsigned int k = 0; k < in_dim; ++k) {
                    val += x[t * in_dim + k] * A_s[r * in_dim + k];
                }
            }
        }
        intermediate[t * max_rank + r] = val;
    }
}

// Dispatched expand: each (token, out_col) accumulates its adapter's delta via
// atomicAdd. Tokens sharing an adapter run in the same launch in parallel.
extern "C" __global__ void grim_lora_expand_dispatched(
    const float* __restrict__ intermediate,      // [total_tokens, max_rank]
    const float* const* __restrict__ b_ptrs,     // [num_adapters] device ptrs, B_s [out_dim, rank_s]
    const unsigned int* __restrict__ token_adapter_idx, // [total_tokens]
    const unsigned int* __restrict__ ranks,      // [num_adapters]
    const float* __restrict__ scalings,          // [num_adapters]
    float* __restrict__ output,                  // [total_tokens, out_dim]
    unsigned int out_dim,
    unsigned int max_rank,
    unsigned int total_tokens
) {
    unsigned int t = blockIdx.y;
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total_tokens && o < out_dim) {
        unsigned int s = token_adapter_idx[t];
        if (s != 0xFFFFFFFFU) {
            unsigned int rank_s = ranks[s];
            const float* B_s = b_ptrs[s];
            float sum = 0.0f;
            for (unsigned int r = 0; r < rank_s; ++r) {
                sum += intermediate[t * max_rank + r] * B_s[o * rank_s + r];
            }
            atomicAdd(&output[t * out_dim + o], sum * scalings[s]);
        }
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

/// One adapter's weights plus scaling, for the dispatched path. The dispatched
/// kernel looks the adapter up per-token via an indirection table, so weights
/// live in per-adapter device allocations addressed through pointer arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchedLoraAdapter {
    /// A weights, row-major `[rank, in_dim]`.
    pub a_weights: Vec<f32>,
    /// B weights, row-major `[out_dim, rank]`.
    pub b_weights: Vec<f32>,
    /// Low-rank dimension $r$.
    pub rank: usize,
    /// Effective scaling $\alpha / r$.
    pub scaling: f32,
}

/// Dispatched batched multi-LoRA (Punica/SGLang style): two kernel launches
/// TOTAL, regardless of how many adapters are active.
///
/// # How it is dispatched (not grouped)
/// The old [`batched_lora_group_device`] loops over adapters on the host and
/// issues a shrink+expand pair per adapter — 2N launches, serialized. This
/// function issues **one** shrink launch and **one** expand launch over the
/// whole batch; each thread picks its adapter from `token_adapter_idx` and
/// gathers that adapter's A/B pointers and rank from device-side pointer arrays.
/// Tokens using different adapters run in the same launch, in parallel.
///
/// # Contracts
/// * `x_host`: `[total_tokens, in_dim]`, `y_host`: `[total_tokens, out_dim]`
///   (base GEMM output; deltas accumulate in place)
/// * `token_adapter_idx[t]` is the index into `adapters` for token `t`, or
///   `u32::MAX` for the base model (no delta).
/// * Every adapter's `a_weights` is `[rank, in_dim]`, `b_weights` is
///   `[out_dim, rank]`.
pub fn batched_lora_dispatched_device(
    device: &RocmDevice,
    x_host: &[f32],
    y_host: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    token_adapter_idx: &[u32],
    adapters: &[DispatchedLoraAdapter],
) -> Result<()> {
    if in_dim == 0 || out_dim == 0 {
        return Err(Error::Backend(
            "batched_lora_dispatched_device: in_dim and out_dim must be > 0".into(),
        ));
    }
    if x_host.len() % in_dim != 0 {
        return Err(Error::Backend(format!(
            "batched_lora_dispatched_device: x len {} is not a multiple of in_dim {in_dim}",
            x_host.len()
        )));
    }
    let total_tokens = x_host.len() / in_dim;
    if y_host.len() != total_tokens * out_dim {
        return Err(Error::Backend(format!(
            "batched_lora_dispatched_device: y len {} != tokens {total_tokens} * out_dim {out_dim}",
            y_host.len()
        )));
    }
    if token_adapter_idx.len() != total_tokens {
        return Err(Error::Backend(format!(
            "batched_lora_dispatched_device: token_adapter_idx len {} != tokens {total_tokens}",
            token_adapter_idx.len()
       )));
    }
    if adapters.is_empty() {
        return Ok(());
    }
    let max_rank = adapters.iter().map(|a| a.rank).max().unwrap_or(0);
    if max_rank == 0 {
        return Ok(());
    }
    for (i, a) in adapters.iter().enumerate() {
        if a.a_weights.len() != a.rank * in_dim || a.b_weights.len() != out_dim * a.rank {
            return Err(Error::Backend(format!(
                "batched_lora_dispatched_device: adapter {} weight shape mismatch \
                 (a {} vs rank*in_dim {}, b {} vs out_dim*rank {})",
                i,
                a.a_weights.len(),
                a.rank * in_dim,
                a.b_weights.len(),
                out_dim * a.rank
            )));
        }
    }

    let x_ptr = upload_device_buffer(device.ordinal, x_host)?;
    // y is uploaded from the caller's contents and written back after both
    // kernels complete.
    let y_ptr = upload_device_buffer(device.ordinal, y_host)?;

    // Upload each adapter's A and B once, keep their device pointers for the
    // per-adapter pointer arrays the kernels index into.
    let result = (|| -> Result<()> {
        let mut a_bufs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(adapters.len());
        let mut b_bufs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(adapters.len());
        for a in adapters {
            a_bufs.push(upload_device_buffer(device.ordinal, &a.a_weights)?);
            b_bufs.push(upload_device_buffer(device.ordinal, &a.b_weights)?);
        }

        // Host arrays of device pointers + per-adapter rank/scaling, uploaded to
        // device memory so the kernels can gather per-adapter state.
        let a_ptrs_host: Vec<*const f32> = a_bufs.iter().map(|p| *p as *const f32).collect();
        let b_ptrs_host: Vec<*const f32> = b_bufs.iter().map(|p| *p as *const f32).collect();
        let ranks_host: Vec<u32> = adapters.iter().map(|a| a.rank as u32).collect();
        let scalings_host: Vec<f32> = adapters.iter().map(|a| a.scaling).collect();

        let a_ptrs_dev = upload_device_buffer(device.ordinal, &a_ptrs_host)?;
        let b_ptrs_dev = upload_device_buffer(device.ordinal, &b_ptrs_host)?;
        let ranks_dev = upload_device_buffer(device.ordinal, &ranks_host)?;
        let scalings_dev = upload_device_buffer(device.ordinal, &scalings_host)?;
        let idx_dev = upload_device_buffer(device.ordinal, token_adapter_idx)?;

        // Intermediate activations [total_tokens, max_rank].
        let inter_elems = total_tokens * max_rank;
        let mut inter_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let alloc_res =
            unsafe { crate::hipMalloc(&mut inter_ptr, inter_elems * std::mem::size_of::<f32>()) };
        check_hip("dispatched batched_lora hipMalloc(intermediate)", alloc_res)?;

        let launch_outcome = (|| -> Result<()> {
            // Shrink: every token computes its own intermediate in parallel.
            let mut x_arg = x_ptr;
            let mut a_arg = a_ptrs_dev;
            let mut idx_arg = idx_dev;
            let mut ranks_arg = ranks_dev;
            let mut inter_arg = inter_ptr;
            let mut in_dim_i = in_dim as u32;
            let mut max_rank_i = max_rank as u32;
            let mut tokens_i = total_tokens as u32;
            launch_batched_lora_kernel(
                device,
                BATCHED_LORA_DISPATCHED_KERNEL_SOURCE,
                "grim_lora_shrink_dispatched",
                HipDim3::new(max_rank.div_ceil(BATCHED_LORA_BLOCK) as u32, total_tokens as u32, 1),
                HipDim3::new(BATCHED_LORA_BLOCK as u32, 1, 1),
                &mut [
                    arg(&mut x_arg),
                    arg(&mut a_arg),
                    arg(&mut idx_arg),
                    arg(&mut ranks_arg),
                    arg(&mut inter_arg),
                    arg(&mut in_dim_i),
                    arg(&mut max_rank_i),
                    arg(&mut tokens_i),
                ],
            )?;

            // Expand: every (token, out_col) accumulates its adapter's delta.
            let mut inter_arg2 = inter_ptr;
            let mut b_arg = b_ptrs_dev;
            let mut idx_arg2 = idx_dev;
            let mut ranks_arg2 = ranks_dev;
            let mut scalings_arg = scalings_dev;
            let mut y_arg = y_ptr;
            let mut out_dim_i = out_dim as u32;
            let mut max_rank_i = max_rank as u32;
            let mut tokens_i = total_tokens as u32;
            launch_batched_lora_kernel(
                device,
                BATCHED_LORA_DISPATCHED_KERNEL_SOURCE,
                "grim_lora_expand_dispatched",
                HipDim3::new(out_dim.div_ceil(BATCHED_LORA_BLOCK) as u32, total_tokens as u32, 1),
                HipDim3::new(BATCHED_LORA_BLOCK as u32, 1, 1),
                &mut [
                    arg(&mut inter_arg2),
                    arg(&mut b_arg),
                    arg(&mut idx_arg2),
                    arg(&mut ranks_arg2),
                    arg(&mut scalings_arg),
                    arg(&mut y_arg),
                    arg(&mut out_dim_i),
                    arg(&mut max_rank_i),
                    arg(&mut tokens_i),
                ],
            )
        })();

        unsafe {
            hipFree(inter_ptr);
        }
        for buf in a_bufs.into_iter().chain(b_bufs) {
            unsafe {
                hipFree(buf);
            }
        }
        unsafe {
            hipFree(a_ptrs_dev);
            hipFree(b_ptrs_dev);
            hipFree(ranks_dev);
            hipFree(scalings_dev);
            hipFree(idx_dev);
        }
        launch_outcome?;

        // D2H the accumulated output.
        let _guard = DeviceGuard::set(device.ordinal as i32);
        let bytes = y_host.len() * std::mem::size_of::<f32>();
        if bytes > 0 {
            check_hip(
                "dispatched batched_lora D2H copy",
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

/// CPU reference for the dispatched path: mirrors
/// [`batched_lora_dispatched_device`] without the GPU, applying each token's
/// adapter via the same indirection table. Used to validate the GPU kernel and
/// as the portable fallback.
pub fn batched_lora_dispatched_cpu(
    x: &[f32],
    y: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    token_adapter_idx: &[u32],
    adapters: &[DispatchedLoraAdapter],
) -> Result<()> {
    let total_tokens = x.len() / in_dim;
    if y.len() != total_tokens * out_dim || token_adapter_idx.len() != total_tokens {
        return Err(Error::Backend(
            "batched_lora_dispatched_cpu: shape mismatch".into(),
        ));
    }
    let max_rank = adapters.iter().map(|a| a.rank).max().unwrap_or(0);
    let mut intermediate = vec![0.0f32; total_tokens * max_rank];
    for (t, &s) in token_adapter_idx.iter().enumerate() {
        if s as usize >= adapters.len() {
            continue;
        }
        let a = &adapters[s as usize];
        if a.rank == 0 {
            continue;
        }
        let rank = a.rank;
        let x_tok = &x[t * in_dim..(t + 1) * in_dim];
        let inter_tok = &mut intermediate[t * max_rank..t * max_rank + rank];
        for r in 0..rank {
            let a_row = &a.a_weights[r * in_dim..(r + 1) * in_dim];
            let mut sum = 0.0f32;
            for k in 0..in_dim {
                sum += x_tok[k] * a_row[k];
            }
            inter_tok[r] = sum;
        }
        let y_tok = &mut y[t * out_dim..(t + 1) * out_dim];
        let scaling = a.scaling;
        for o in 0..out_dim {
            let b_row = &a.b_weights[o * rank..(o + 1) * rank];
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += inter_tok[r] * b_row[r];
            }
            y_tok[o] += sum * scaling;
        }
    }
    Ok(())
}
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

/// Launch one JIT-compiled kernel from `source` (either the segmented or the
/// dispatched batched-LoRA source) and wait for it. Compile results go through
/// the device's persistent disk cache (`jit_compile_or_cache`), so cold-start
/// cost is paid once per (entry, arch, source) triple per machine.
fn launch_batched_lora_kernel(
    device: &RocmDevice,
    source: &str,
    entry: &str,
    grid: HipDim3,
    block: HipDim3,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    use std::ffi::CString;

    let _guard = DeviceGuard::set(device.ordinal as i32);
    let (hsaco_path, lowered) = device.jit_compile_or_cache(source, entry, None)?;

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
                    BATCHED_LORA_KERNEL_SOURCE,
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
                    BATCHED_LORA_KERNEL_SOURCE,
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

    /// The dispatched path must match the path the engine actually used before
    /// (grouped per-segment launches) — they are mathematically identical, so a
    /// regression in either shows up here. Skips without a ROCm device.
    #[test]
    fn batched_lora_dispatched_matches_grouped() {
        let device = match crate::device::roc_device::RocmDevice::probe_one(0) {
            Ok(true) => crate::device::roc_device::RocmDevice::new(0),
            _ => {
                eprintln!("skipping dispatched-vs-grouped parity: no ROCm device visible");
                return;
            }
        };

        let in_dim = 8;
        let out_dim = 6;
        let rank = 2;
        let rows = 5;
        // Per-row adapter assignment: base, adapter0, adapter0, base, adapter1.
        let token_adapter_idx: Vec<u32> = vec![u32::MAX, 0, 0, u32::MAX, 1];

        let sample = |n: usize, seed: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + seed) * 0.137).sin().clamp(-1.0, 1.0)).collect()
        };

        // Two distinct adapters with distinct weights.
        let a0 = sample(rank * in_dim, 2.0);
        let b0 = sample(out_dim * rank, 3.0);
        let a1 = sample(rank * in_dim, 5.0);
        let b1 = sample(out_dim * rank, 7.0);

        let x = sample(rows * in_dim, 11.0);

        let dispatched_adapters = vec![
            DispatchedLoraAdapter {
                a_weights: a0.clone(),
                b_weights: b0.clone(),
                rank,
                scaling: 0.5,
            },
            DispatchedLoraAdapter {
                a_weights: a1.clone(),
                b_weights: b1.clone(),
                rank,
                scaling: 1.5,
            },
        ];

        // Grouped reference (per-segment launches): adapter0 on rows 1-2,
        // adapter1 on row 4. Matches the dispatched path's math exactly.
        let mut y_grouped = vec![0.0f32; rows * out_dim];
        let x_cpu = x.clone();
        // Segments: rows 1-2 = adapter0, row 4 = adapter1.
        batched_lora_accumulate_cpu(
            &x_cpu,
            &mut y_grouped,
            in_dim,
            out_dim,
            &BatchedLoraSegment {
                adapter_id: 0,
                token_start: 1,
                token_count: 2,
                rank,
                scaling: 0.5,
            },
            &a0,
            &b0,
        )
        .unwrap();
        batched_lora_accumulate_cpu(
            &x_cpu,
            &mut y_grouped,
            in_dim,
            out_dim,
            &BatchedLoraSegment {
                adapter_id: 1,
                token_start: 4,
                token_count: 1,
                rank,
                scaling: 1.5,
            },
            &a1,
            &b1,
        )
        .unwrap();

        // Dispatched (two launches total).
        let mut y_dispatched = vec![0.0f32; rows * out_dim];
        batched_lora_dispatched_device(
            &device,
            &x,
            &mut y_dispatched,
            in_dim,
            out_dim,
            &token_adapter_idx,
            &dispatched_adapters,
        )
        .expect("dispatched dispatch must succeed");

        for (i, (g, d)) in y_grouped.iter().zip(&y_dispatched).enumerate() {
            assert!(
                (g - d).abs() < 1e-4,
                "row {i}: grouped {g} vs dispatched {d} (grouped {y_grouped:?} dispatched {y_dispatched:?})"
            );
        }
    }
}

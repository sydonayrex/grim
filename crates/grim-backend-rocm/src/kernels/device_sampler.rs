//! WI-X3: GPU-native stochastic sampling kernel for ROCm.
//! [`DEVICE_SAMPLER_KERNEL_SOURCE`] defines `grim_sample_logits_stochastic`: a single-block HIP kernel that applies temperature scaling, top-k and top-p filtering.

/// HIP C++ source for `grim_sample_logits_stochastic`. [see: `compute_kernel_source`, `launch_compute_kernel`]
pub const DEVICE_SAMPLER_KERNEL_SOURCE: &str = r#"
// WI-X3: GPU Stochastic Logits Sampler (temperature + top-k + top-p + Gumbel) Grid: (1, 1)   - one block samples one token from one logits row.
// Block: (256, 1) - threads stride across the vocab; all reductions are block-wide tree reductions.
#define GRIM_SAMPLER_BLOCK 256

__device__ unsigned int grim_sampler_hash(unsigned int x) {
    // splitmix32-style finalizer: full avalanche per call, so each
    // (seed, position, thread, chunk) tuple draws an independent uniform.
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ float grim_sampler_uniform(unsigned int seed, int position, int tid, int step) {
    unsigned int h = grim_sampler_hash(
        seed ^ ((unsigned int)position * 0x9e3779b9u)
             ^ ((unsigned int)tid * 0x85ebca6bu)
             ^ ((unsigned int)step * 0xc2b2ae35u));
    // Map 24 bits to the OPEN interval (0, 1): +0.5 keeps every draw strictly
    // inside so logf(u) and logf(-logf(u)) never see an exact zero.
    return ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

extern "C" __global__ void grim_sample_logits_stochastic(
    const float* __restrict__ logits,        // [vocab_size] (one row)
    unsigned int* __restrict__ out_token,    // [1]
    int vocab_size,
    float temperature,                       // <= 0 -> exact greedy argmax
    int top_k,                               // 0 = disabled
    float top_p,                             // >= 1.0 = disabled
    unsigned int seed,
    int position                             // decode step, mixes the RNG stream
) {
    const int tid = threadIdx.x;
    const int block = blockDim.x;

    __shared__ float s_val[GRIM_SAMPLER_BLOCK];
    __shared__ int   s_idx[GRIM_SAMPLER_BLOCK];

    if (temperature <= 0.0f) {
        // Greedy shortcut: T->0 collapses softmax to a point mass, which the Gumbel-max trick reproduces only
        // in the infinite-noise limit - so take the exact argmax instead of dividing by zero.
        float local_max = -1e30f;
        int local_idx = -1;
        for (int v = tid; v < vocab_size; v += block) {
            if (logits[v] > local_max) {
                local_max = logits[v];
                local_idx = v;
            }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();
        for (int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out_token[0] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
        }
        return;
    }

    const float inv_t = 1.0f / temperature;

    // ---- pass 1: max & min of the scaled logits ---------------------------
    float lmax = -1e30f;
    float lmin = 1e30f;
    for (int v = tid; v < vocab_size; v += block) {
        const float s = logits[v] * inv_t;
        if (s > lmax) lmax = s;
        if (s < lmin) lmin = s;
    }
    s_val[tid] = lmax;
    __syncthreads();
    for (int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_val[tid] = fmaxf(s_val[tid], s_val[tid + stride]);
        }
        __syncthreads();
    }
    const float s_max = s_val[0];
    __syncthreads();
    s_val[tid] = lmin;
    __syncthreads();
    for (int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_val[tid] = fminf(s_val[tid], s_val[tid + stride]);
        }
        __syncthreads();
    }
    const float s_min = s_val[0];
    __syncthreads();

    // ---- pass 2: top-k threshold (count-bisection over scaled-logit value).
    // Invariants: count(s >= lo) >= top_k ; count(s >= hi) <= top_k.
    float t_k = -1e30f; // -1e30 sentinel == "keep everything" when disabled
    if (top_k > 0 && top_k < vocab_size) {
        float lo = s_min;
        float hi = s_max;
        for (int it = 0; it < 24; ++it) {
            const float mid = 0.5f * (lo + hi);
            if (!(mid > lo) || !(mid < hi)) break; // float resolution exhausted
            float cnt = 0.0f;
            for (int v = tid; v < vocab_size; v += block) {
                cnt += (logits[v] * inv_t >= mid) ? 1.0f : 0.0f;
            }
            s_val[tid] = cnt;
            __syncthreads();
            for (int stride = block / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    s_val[tid] += s_val[tid + stride];
                }
                __syncthreads();
            }
            if (s_val[0] >= (float)top_k) {
                lo = mid;
            } else {
                hi = mid;
            }
            __syncthreads(); // s_val is rewritten next iteration
        }
        t_k = lo;
    }

    // ---- pass 3: top-p threshold (mass-bisection over surviving set).
    // mass(x) = sum_{unmasked, s >= x} p(s), non-increasing in x.
    float t_p = -1e30f; // sentinel == "disabled"
    if (top_p < 1.0f) {
        float local_z = 0.0f;
        for (int v = tid; v < vocab_size; v += block) {
            const float s = logits[v] * inv_t;
            if (s >= t_k) {
                local_z += __expf(s - s_max);
            }
        }
        s_val[tid] = local_z;
        __syncthreads();
        for (int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_val[tid] += s_val[tid + stride];
            }
            __syncthreads();
        }
        const float inv_z = 1.0f / fmaxf(s_val[0], 1e-30f);
        __syncthreads();
        if (inv_z >= top_p) {
            // The most-likely token alone already carries >= top_p of the
            // surviving mass: keep only the argmax, skip the sweep.
            t_p = s_max;
        } else {
            // Invariants: mass(lo) >= top_p (starts at 1.0 for lo = s_min);
            // mass(hi) < top_p (starts at p_max = inv_z for hi = s_max).
            float lo = s_min;
            float hi = s_max;
            for (int it = 0; it < 24; ++it) {
                const float mid = 0.5f * (lo + hi);
                if (!(mid > lo) || !(mid < hi)) break;
                float m = 0.0f;
                for (int v = tid; v < vocab_size; v += block) {
                    const float s = logits[v] * inv_t;
                    if (s >= mid && s >= t_k) {
                        m += __expf(s - s_max);
                    }
                }
                s_val[tid] = m;
                __syncthreads();
                for (int stride = block / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_val[tid] += s_val[tid + stride];
                    }
                    __syncthreads();
                }
                if (s_val[0] * inv_z >= top_p) {
                    lo = mid;
                } else {
                    hi = mid;
                }
                __syncthreads();
            }
            t_p = lo;
        }
    }

    // ---- pass 4: Gumbel-max multinomial draw over the filtered support ---- key_i = s_i + g_i with g ~ Gumbel(0,1) = -log(-log(u)).
    // argmax key is a sample from softmax(s) restricted to the unmasked tokens - no cumsum.
    float best_key = -1e30f;
    int best_v = -1;
    int n = 0;
    for (int v = tid; v < vocab_size; v += block, ++n) {
        const float s = logits[v] * inv_t;
        if (s < t_k || s < t_p) continue;
        const float u = grim_sampler_uniform(seed, position, tid, n);
        const float key = s - logf(-logf(u));
        if (key > best_key) {
            best_key = key;
            best_v = v;
        }
    }
    s_val[tid] = best_key;
    s_idx[tid] = best_v;
    __syncthreads();
    for (int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_token[0] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
    }
}
"#;

use std::ffi::c_void;

use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::{ArithType, Error, Shape};

use crate::{
    HipDim3, HipMemcpyKind, RocmDevice, RocmStorage, arg, check_hip, dev_ptr, hipMemcpyAsync,
    hipStreamSynchronize,
};
use grim_tensor::error::Result;

use crate::device::util::DeviceGuard;

/// Block size the kernel is compiled/launched with (must match
/// `GRIM_SAMPLER_BLOCK` in [`DEVICE_SAMPLER_KERNEL_SOURCE`]).
const SAMPLER_BLOCK: u32 = 256;

/// Largest vocabulary accepted by the device sampler.
/// Beyond this the LDS / register budget of the single-block design degrades and callers should.
pub const MAX_DEVICE_SAMPLER_VOCAB: usize = 1 << 18; // 262144

/// Shared implementation behind [`sample_logits_on_device`] /
/// [`sample_logits_on_device_at`]. Returns the sampled token id.
fn sample_impl(
    device: &RocmDevice,
    logits_ptr: u64,
    vocab: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    seed: u32,
    position: u32,
) -> Result<u32> {
    let mut buf_guard = match device.sampler_out_buf.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if buf_guard.is_none() {
        *buf_guard = Some(RocmStorage::alloc_gpu(
            &Shape::new(vec![1usize]),
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
            &device.allocator,
            device.ordinal,
        )?);
    }
    let out_storage = buf_guard.as_ref().unwrap();
    let out_ptr = dev_ptr(out_storage)?;

    let mut logits_arg = logits_ptr;
    let mut out_arg = out_ptr;
    let mut vocab_i = vocab as i32;
    let mut temp = temperature;
    let mut topk = top_k;
    let mut topp = top_p;
    let mut seed_u = seed;
    let mut pos = position as i32;

    // Pin the thread to the owning device for the launch + D2H copy so the
    // async copy lands in the right HIP context on multi-GPU boxes.
    let _dev_guard = DeviceGuard::set(device.ordinal as i32);

    // Single block over a single logits row.
    let stream = device.launch_compute_kernel(
        "grim_sample_logits_stochastic",
        HipDim3::new(1, 1, 1),
        HipDim3::new(SAMPLER_BLOCK, 1, 1),
        &mut [
            arg(&mut logits_arg),
            arg(&mut out_arg),
            arg(&mut vocab_i),
            arg(&mut temp),
            arg(&mut topk),
            arg(&mut topp),
            arg(&mut seed_u),
            arg(&mut pos),
        ],
    )?;

    // D2H ONLY the 4-byte token id, ordered on the launch stream.
    let mut host_token: u32 = 0;
    check_hip("grim_sample_logits_stochastic D2H", unsafe {
        hipMemcpyAsync(
            &mut host_token as *mut u32 as *mut c_void,
            out_arg as *mut c_void,
            4,
            HipMemcpyKind::DeviceToHost,
            stream,
        )
    })?;
    check_hip("grim_sample_logits_stochastic sync", unsafe {
        hipStreamSynchronize(stream)
    })?;

    Ok(host_token)
}

/// Validate shape/vocab and return the device pointer, or `None` when the
/// caller must fall back to CPU sampling (`Ok(None)` contract).
fn validate_input(logits: &RocmStorage, vocab: usize, temperature: f32, top_p: f32) -> Option<u64> {
    if vocab == 0 || vocab > MAX_DEVICE_SAMPLER_VOCAB {
        return None;
    }
    // The engine's logits table can be wider than the model vocab; callers
    // slice to the LAST `vocab` entries host-side, so require the tail to fit.
    if logits.bytes() < vocab * std::mem::size_of::<f32>() {
        return None;
    }
    if !temperature.is_finite() || !top_p.is_finite() || temperature < 0.0 {
        return None;
    }
    // Offset to the LAST `vocab` entries: the engine's logits table may be wider than the
    // model vocab (65536-wide), and host-side CPU sampling slices the tail - mirror that exactly on device.
    let tail_offset = logits.bytes() - vocab * std::mem::size_of::<f32>();
    logits
        .device_ptr_u64()
        .filter(|_| logits.device_ptr_is_valid())
        .map(|base| base + tail_offset as u64)
}

/// WI-X3: sample one token from a logits row entirely on the GPU.
/// Applies `temperature`, `top_k` (0 = disabled) and `top_p` (>= 1.0 = disabled) filtering plus multinomial.
pub fn sample_logits_on_device(
    device: &RocmDevice,
    logits: &RocmStorage,
    vocab: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    seed: u64,
) -> Result<Option<u32>> {
    let Some(ptr) = validate_input(logits, vocab, temperature, top_p) else {
        return Ok(None);
    };
    Ok(Some(sample_impl(
        device,
        ptr,
        vocab,
        temperature,
        top_k,
        top_p,
        seed as u32,
        (seed >> 32) as u32,
    )?))
}

/// WI-X3: same as [`sample_logits_on_device`] with the RNG stream position
/// passed explicitly instead of packed into the high 32 bits of `seed`.
pub fn sample_logits_on_device_at(
    device: &RocmDevice,
    logits: &RocmStorage,
    vocab: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    seed: u64,
    position: u32,
) -> Result<Option<u32>> {
    let Some(ptr) = validate_input(logits, vocab, temperature, top_p) else {
        return Ok(None);
    };
    Ok(Some(sample_impl(
        device,
        ptr,
        vocab,
        temperature,
        top_k,
        top_p,
        (seed & 0xffff_ffff) as u32,
        position,
    )?))
}

// Keep `Error` in scope for future error-path extensions; silence unused warn.
#[allow(unused)]
fn _error_type_witness(_: Error) {}

// ── Double-buffered pinned logits fallback (WI-X3) ─────────────────────────
//
// When the device sampler cannot absorb the request's sampling semantics
// (constrained grammar, GRIM_CPU_SAMPLER=1), the CPU still needs the logits.
// The naive path calls `to_cpu_vec_f32()` which allocates a Vec<f32> per step
// and does a blocking synchronous D2H copy.
//
// PinnedLogitsBuf replaces that: it holds two pinned (page-locked) host
// buffers sized for the maximum vocabulary.  Each call to
// `read_logits_to_pinned` issues an async DMA into slot[ping], synchronises
// on that copy's stream, and returns a &[f32] into pinned memory — zero heap
// allocation on the hot path.  The caller pings between slots so the GPU can
// overlap the next kernel with the prior copy (future: overlap with stream).

use crate::memory::pinned::RocmPinnedBuffer;

/// Double-buffered page-locked host buffer for vocab-sized logit readback.
/// Allocate once at session start; reuse across every decode step.
pub struct PinnedLogitsBuf {
    bufs: [RocmPinnedBuffer<f32>; 2],
    ping: usize,
}

impl PinnedLogitsBuf {
    /// Allocate two pinned host buffers each capable of holding `max_vocab` f32 elements.
    pub fn alloc(max_vocab: usize) -> Result<Self> {
        Ok(Self {
            bufs: [
                RocmPinnedBuffer::alloc(max_vocab)?,
                RocmPinnedBuffer::alloc(max_vocab)?,
            ],
            ping: 0,
        })
    }

    /// D2H copy the last `vocab` elements of `logits` into the active pinned slot.
    /// Synchronises on the copy before returning so the slice is CPU-readable.
    /// Advances the ping index for the next call.
    ///
    /// Returns a slice into pinned memory — valid until the next call to this fn.
    pub fn read_logits_to_pinned(
        &mut self,
        device: &RocmDevice,
        logits: &RocmStorage,
        vocab: usize,
    ) -> Result<&[f32]> {
        let slot = self.ping;
        self.ping ^= 1;

        // Validate: need at least vocab * 4 bytes on device.
        let needed_bytes = vocab * std::mem::size_of::<f32>();
        if logits.bytes() < needed_bytes {
            return Err(grim_tensor::error::Error::Backend(format!(
                "PinnedLogitsBuf::read: logits too small ({} bytes) for vocab {vocab}",
                logits.bytes()
            )));
        }
        if self.bufs[slot].len() < vocab {
            return Err(grim_tensor::error::Error::Backend(format!(
                "PinnedLogitsBuf::read: pinned slot too small ({} elems) for vocab {vocab}",
                self.bufs[slot].len()
            )));
        }

        // Offset into the tail: the engine's logit table can be wider than vocab.
        let tail_offset = logits.bytes() - needed_bytes;
        let src_ptr = {
            let base = logits
                .device_ptr_u64()
                .ok_or_else(|| {
                    grim_tensor::error::Error::Backend(
                        "PinnedLogitsBuf::read: logits has no device ptr".into(),
                    )
                })?;
            (base + tail_offset as u64) as *const c_void
        };
        let dst_ptr = self.bufs[slot].as_mut_ptr() as *mut c_void;

        // Pin thread to owning device so the DMA lands in the right HIP context.
        let _guard = DeviceGuard::set(device.ordinal as i32);
        let stream = device.active_stream();

        check_hip("PinnedLogitsBuf hipMemcpyAsync D2H", unsafe {
            hipMemcpyAsync(dst_ptr, src_ptr, needed_bytes, HipMemcpyKind::DeviceToHost, stream)
        })?;
        check_hip("PinnedLogitsBuf hipStreamSynchronize", unsafe {
            hipStreamSynchronize(stream)
        })?;

        Ok(&self.bufs[slot].as_slice()[..vocab])
    }
}


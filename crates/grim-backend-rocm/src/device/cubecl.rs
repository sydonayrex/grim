//! cubecl (pure-Rust, HIP) backend — feature-gated port of the C++ hipRTC kernels. [see: `feature = "cubecl"`, `device::roc_device`]

#![cfg(feature = "cubecl")]

use cubecl::hip::{AmdDevice, HipRuntime};
use cubecl::prelude::*;
use cubecl_hip_sys::{hipDeviceProp_tR0600, hipGetDevicePropertiesR0600};
use std::sync::OnceLock;

pub type Client = ComputeClient<HipRuntime>;

static CLIENT: OnceLock<Client> = OnceLock::new();

/// Process-wide client. Warms every kernel kind once so real launches are 2nd+. [see: `HipRuntime::client()`, `HipContext::compile_kernel`]
pub fn client() -> &'static Client {
    CLIENT.get_or_init(|| {
        // RDNA2-only workaround gate. Detect arch via a raw hipRTC FFI [see: `hipGetDevicePropertiesR0600`, `HipRuntime::client()`]
        let rdna2 = is_rdna2(0);
        if rdna2 {
            eprintln!("cubecl client: detected RDNA2 (gfx10xx) — enabling warmup-retry + compilation cache workaround");
            configure_compilation_cache();
        }
        let c = HipRuntime::client(&AmdDevice::new(0));
        // The cubecl-hip first-launch cold fault (all-zeros / GPU page-fault
        if rdna2 {
            let mut attempts = 0;
            loop {
                warmup_every_kernel(&c);
                if warmup_verify(&c) {
                    break;
                }
                attempts += 1;
                if attempts >= WARMUP_MAX_RETRIES {
                    eprintln!(
                        "cubecl warmup: {attempts} retries still produced a bad \
                         first-launch readback; proceeding anyway (gfx1036 cold fault)"
                    );
                    break;
                }
            }
        }
        c
    })
}

/// Returns true if `device` is an RDNA2 part (gfx10xx), the only family [see: `gcnArchName`, `hipGetDevicePropertiesR0600`]
fn is_rdna2(device: u32) -> bool {
    let mut prop: hipDeviceProp_tR0600 = unsafe { std::mem::zeroed() };
    let status = unsafe { hipGetDevicePropertiesR0600(&mut prop, device as _) };
    if status != 0 {
        // Can't detect -> assume RDNA2 (apply the workaround) rather than
        return true;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(prop.gcnArchName.as_ptr()) };
    let name = name.to_string_lossy();
    let family = name.split(':').next().unwrap_or("");
    family.starts_with("gfx10")
}

/// Max times we re-run the full warmup before giving up on the cold fault.
const WARMUP_MAX_RETRIES: u32 = 3;

/// Launch every kernel kind once. Each helper ends in a blocking `read_one`,
fn warmup_every_kernel(c: &Client) {
    let warm0 = [0.0f32; 1];
    let warm1 = [1.0f32; 1];
    let _ = add(&c, &warm0, &warm1);
    let _ = mul(&c, &warm0, &warm1);
    let _ = silu_mul(&c, &warm0, &warm1);
    let _ = embedding(&c, &warm1, &[0i32], 1);
    let _ = rms_norm(&c, &warm1, 1, 1);
    let _ = softmax(&c, &warm1, 1, 1);
    let tiny_k = [1.0f32; 1];
    let tiny_bt = [0.0f32, 1.0f32];
    let tiny_tp = [0.0f32];
    let _ = qkv_attention(&c, &warm1, &tiny_k, &tiny_k, 1, 1, 1, 1, 1, 0);
    let _ = paged_attention(&c, &warm1, &tiny_k, &tiny_k, &tiny_bt, 1, 1, 1, 1, 1, 1, 0);
    let _ = tree_attention(&c, &warm1, &tiny_k, &tiny_k, &tiny_tp, 1, 1, 1, 1, 1, 0);
    let _ = gptq_correction(&c, &warm1, &warm1, &warm1, 0.5f32, 1, 1, 1);
    // Final hard sync: forces the stream to drain so the just-warmed modules
    let _ = c.sync();
}

/// Verify the warmup actually executed (not the first-launch all-zeros fault). [see: `add([0],[1])`, `[1]`]
fn warmup_verify(c: &Client) -> bool {
    let out = add(c, &[0.0f32; 1], &[1.0f32; 1]);
    out.first().copied() == Some(1.0f32)
}

/// Enable cubecl's disk-backed kernel compilation cache. [see: `CubeClRuntimeConfig::set`, `HipRuntime::client()`]
fn configure_compilation_cache() {
    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    let mut cfg = CubeClRuntimeConfig::default();
    cfg.compilation.cache = Some(CacheConfig::Global);
    CubeClRuntimeConfig::set(cfg);
}

fn launch_elems(len: usize) -> (CubeCount, CubeDim) {
    let cubes = ((len as u32) + 255) / 256;
    (CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(256))
}

// ---------------- elementwise ----------------

#[cube(launch)]
fn add_kernel(output: &mut Array<f32>, a: &Array<f32>, b: &Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

pub fn add(client: &Client, a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let ah = client.create_from_slice(f32::as_bytes(a));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let oh = client.empty(len * 4);
    let (count, dim) = launch_elems(len);
    add_kernel::launch::<HipRuntime>(
        client,
        count.clone(),
        dim.clone(),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(ah.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(bh.clone(), len) },
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

#[cube(launch)]
fn mul_kernel(output: &mut Array<f32>, a: &Array<f32>, b: &Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

pub fn mul(client: &Client, a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let ah = client.create_from_slice(f32::as_bytes(a));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let oh = client.empty(len * 4);
    let (count, dim) = launch_elems(len);
    mul_kernel::launch::<HipRuntime>(
        client,
        count.clone(),
        dim.clone(),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(ah.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(bh.clone(), len) },
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

/// silu(x) = x * sigmoid(x); out = silu(x) * gate.
#[cube(launch)]
fn silu_mul_kernel(output: &mut Array<f32>, x: &Array<f32>, gate: &Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        let v = x[ABSOLUTE_POS];
        output[ABSOLUTE_POS] = v / (1.0f32 + (-v).exp()) * gate[ABSOLUTE_POS];
    }
}

pub fn silu_mul(client: &Client, x: &[f32], gate: &[f32]) -> Vec<f32> {
    let len = x.len();
    let xh = client.create_from_slice(f32::as_bytes(x));
    let gh = client.create_from_slice(f32::as_bytes(gate));
    let oh = client.empty(len * 4);
    let (count, dim) = launch_elems(len);
    silu_mul_kernel::launch::<HipRuntime>(
        client,
        count.clone(),
        dim.clone(),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(xh.clone(), len) },
        unsafe { ArrayArg::from_raw_parts(gh.clone(), len) },
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ---------------- embedding gather ----------------

#[cube(launch)]
fn embedding_kernel(
    output: &mut Array<f32>,
    weight: &Array<f32>,
    indices: &Array<f32>,
    dim: usize,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let row = indices[i / dim] as usize;
        let col = i % dim;
        output[i] = weight[row * dim + col];
    }
}

pub fn embedding(client: &Client, weight: &[f32], indices: &[i32], dim: usize) -> Vec<f32> {
    let out_len = indices.len() * dim;
    let idx_f: Vec<f32> = indices.iter().map(|&r| r as f32).collect();
    let wh = client.create_from_slice(f32::as_bytes(weight));
    let ih = client.create_from_slice(f32::as_bytes(&idx_f));
    let oh = client.empty(out_len * 4);
    let (count, d) = launch_elems(out_len);
    embedding_kernel::launch::<HipRuntime>(
        client,
        count.clone(),
        d.clone(),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(wh.clone(), weight.len()) },
        unsafe { ArrayArg::from_raw_parts(ih.clone(), indices.len()) },
        dim,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ---------------- row reductions (dim <= 64, single wavefront) ----------------

#[cube(launch)]
fn rms_norm_kernel(output: &mut Array<f32>, input: &Array<f32>, rows: usize, dim: usize) {
    let row = ABSOLUTE_POS / 64usize;
    let col = ABSOLUTE_POS % 64usize;
    let valid = row < rows && col < dim;
    let base = row * dim;
    let cclamp = if col < dim { col } else { dim - 1usize };
    let x = input[base + cclamp];
    let ss = if valid { x * x } else { 0.0f32.into() };
    let sum_sq = plane_sum(ss);
    let rstd = 1.0f32 / (sum_sq / dim as f32 + 1e-5f32).sqrt();
    if valid {
        output[row * dim + col] = input[base + col] * rstd;
    }
}

pub fn rms_norm(client: &Client, input: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let out_len = rows * dim;
    let ih = client.create_from_slice(f32::as_bytes(input));
    let oh = client.empty(out_len * 4);
    rms_norm_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(rows as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(ih.clone(), out_len) },
        rows,
        dim,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

#[cube(launch)]
fn softmax_kernel(output: &mut Array<f32>, input: &Array<f32>, rows: usize, dim: usize) {
    let row = ABSOLUTE_POS / 64usize;
    let col = ABSOLUTE_POS % 64usize;
    let valid = row < rows && col < dim;
    let base = row * dim;
    let cclamp = if col < dim { col } else { dim - 1usize };
    let mut m = input[base + cclamp];
    if valid {
        let mut c: usize = col + 64usize;
        while c < dim {
            let val = input[base + c];
            if val > m {
                m = val;
            }
            c += 64usize;
        }
    }
    let row_max = plane_max(m);
    let mut s = 0.0f32;
    if valid {
        let mut c2: usize = col;
        while c2 < dim {
            s += (input[base + c2] - row_max).exp();
            c2 += 64usize;
        }
    }
    let row_sum = plane_sum(s);
    if valid {
        output[row * dim + col] = (input[base + col] - row_max).exp() / row_sum;
    }
}

pub fn softmax(client: &Client, input: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let out_len = rows * dim;
    let ih = client.create_from_slice(f32::as_bytes(input));
    let oh = client.empty(out_len * 4);
    softmax_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(rows as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(ih.clone(), out_len) },
        rows,
        dim,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ======================== Phase 3: causal GQA attention (head_dim up to 256) ======================== [see: `tid`]

#[cube(launch)]
fn qkv_attention_kernel(
    output: &mut Array<f32>,
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _seq_len: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) {
    let block = ABSOLUTE_POS / 64usize;
    let lane = ABSOLUTE_POS % 64usize;
    let i = block / num_heads;
    let h = block % num_heads;
    let q_per_kv = num_heads / num_kv_heads;
    let kv_head = h / q_per_kv;
    let abs_i = cache_offset + i;
    let q_base = (i * num_heads + h) * head_dim;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let mut m = f32::new(-1.0e30);
    let mut l = 0.0f32;
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;

    let chunks = (head_dim + 63usize) / 64usize;
    for j in 0..kv_seq_len {
        if j <= abs_i {
            let kv_base = (j * num_kv_heads + kv_head) * head_dim;
            let mut dot = 0.0f32;
            for c in 0..chunks {
                let dim = c * 64usize + lane;
                if dim < head_dim {
                    let term = q[q_base + dim] * k[kv_base + dim];
                    dot = dot + plane_sum(term);
                }
            }
            let s = dot * inv_sqrt_d;

            if s > m {
                let corr = (m - s).exp();
                a0 = a0 * corr;
                a1 = a1 * corr;
                a2 = a2 * corr;
                a3 = a3 * corr;
                l = l * corr;
                m = s;
                let w = 1.0f32;
                for c in 0..chunks {
                    let dim = c * 64usize + lane;
                    if dim < head_dim {
                        let add = w * v[kv_base + dim];
                        if c == 0 {
                            a0 = a0 + add;
                        } else if c == 1 {
                            a1 = a1 + add;
                        } else if c == 2 {
                            a2 = a2 + add;
                        } else {
                            a3 = a3 + add;
                        }
                    }
                }
                l = l + w;
            } else {
                let w = (s - m).exp();
                for c in 0..chunks {
                    let dim = c * 64usize + lane;
                    if dim < head_dim {
                        let add = w * v[kv_base + dim];
                        if c == 0 {
                            a0 = a0 + add;
                        } else if c == 1 {
                            a1 = a1 + add;
                        } else if c == 2 {
                            a2 = a2 + add;
                        } else {
                            a3 = a3 + add;
                        }
                    }
                }
                l = l + w;
            }
        }
    }

    for c in 0..chunks {
        let dim = c * 64usize + lane;
        if dim < head_dim {
            let val = if c == 0 {
                a0
            } else if c == 1 {
                a1
            } else if c == 2 {
                a2
            } else {
                a3
            };
            output[(i * num_heads + h) * head_dim + dim] = val / l;
        }
    }
}

pub fn qkv_attention(
    client: &Client,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) -> Vec<f32> {
    let out_len = seq_len * num_heads * head_dim;
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let oh = client.empty(out_len * 4);
    let blocks = seq_len * num_heads;
    qkv_attention_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(blocks as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(qh.clone(), q.len()) },
        unsafe { ArrayArg::from_raw_parts(kh.clone(), k.len()) },
        unsafe { ArrayArg::from_raw_parts(vh.clone(), v.len()) },
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len,
        cache_offset,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ---------------- paged QKV attention (same online softmax; paged K/V) ---------------- [see: `block_tables`]
#[cube(launch)]
fn paged_attention_kernel(
    output: &mut Array<f32>,
    q: &Array<f32>,
    k_pages: &Array<f32>,
    v_pages: &Array<f32>,
    block_tables: &Array<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks: usize,
    page_size: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) {
    let block = ABSOLUTE_POS / 64usize;
    let lane = ABSOLUTE_POS % 64usize;
    let batch = block / num_heads;
    let h = block % num_heads;
    let q_per_kv = num_heads / num_kv_heads;
    let kv_head = h / q_per_kv;
    let q_base = (batch * num_heads + h) * head_dim;
    let abs_i = cache_offset;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    let chunks = (head_dim + 63usize) / 64usize;

    let mut m = f32::new(-1.0e30);
    let mut l = 0.0f32;
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;

    let num_blocks = (kv_seq_len + page_size - 1usize) / page_size;
    let mut b = 0usize;
    while b < num_blocks {
        let ebase = (batch * max_blocks + b) * 2usize;
        let block_id = block_tables[ebase] as usize;
        let psize = block_tables[ebase + 1usize] as usize;
        let mut t = 0usize;
        while t < psize {
            let j = b * page_size + t;
            if j <= abs_i && j < kv_seq_len {
                let phys = block_id * page_size + t;
                let kv_base = (phys * num_kv_heads + kv_head) * head_dim;
                let mut dot = 0.0f32;
                for c in 0..chunks {
                    let dim = c * 64usize + lane;
                    if dim < head_dim {
                        dot = dot + plane_sum(q[q_base + dim] * k_pages[kv_base + dim]);
                    }
                }
                let s = dot * inv_sqrt_d;
                if s > m {
                    let corr = (m - s).exp();
                    a0 = a0 * corr;
                    a1 = a1 * corr;
                    a2 = a2 * corr;
                    a3 = a3 * corr;
                    l = l * corr;
                    m = s;
                    let w = 1.0f32;
                    for c in 0..chunks {
                        let dim = c * 64usize + lane;
                        if dim < head_dim {
                            let add = w * v_pages[kv_base + dim];
                            if c == 0 {
                                a0 = a0 + add;
                            } else if c == 1 {
                                a1 = a1 + add;
                            } else if c == 2 {
                                a2 = a2 + add;
                            } else {
                                a3 = a3 + add;
                            }
                        }
                    }
                    l = l + w;
                } else {
                    let w = (s - m).exp();
                    for c in 0..chunks {
                        let dim = c * 64usize + lane;
                        if dim < head_dim {
                            let add = w * v_pages[kv_base + dim];
                            if c == 0 {
                                a0 = a0 + add;
                            } else if c == 1 {
                                a1 = a1 + add;
                            } else if c == 2 {
                                a2 = a2 + add;
                            } else {
                                a3 = a3 + add;
                            }
                        }
                    }
                    l = l + w;
                }
            }
            t += 1;
        }
        b += 1;
    }

    for c in 0..chunks {
        let dim = c * 64usize + lane;
        if dim < head_dim {
            let val = if c == 0 {
                a0
            } else if c == 1 {
                a1
            } else if c == 2 {
                a2
            } else {
                a3
            };
            output[q_base + dim] = val / l;
        }
    }
}

pub fn paged_attention(
    client: &Client,
    q: &[f32],
    k_pages: &[f32],
    v_pages: &[f32],
    block_tables: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks: usize,
    page_size: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) -> Vec<f32> {
    let out_len = num_heads * head_dim; // [batch=1, num_heads, head_dim]
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k_pages));
    let vh = client.create_from_slice(f32::as_bytes(v_pages));
    let bh = client.create_from_slice(f32::as_bytes(block_tables));
    let oh = client.empty(out_len * 4);
    let blocks = 1usize * num_heads;
    paged_attention_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(blocks as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(qh.clone(), q.len()) },
        unsafe { ArrayArg::from_raw_parts(kh.clone(), k_pages.len()) },
        unsafe { ArrayArg::from_raw_parts(vh.clone(), v_pages.len()) },
        unsafe { ArrayArg::from_raw_parts(bh.clone(), block_tables.len()) },
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks,
        page_size,
        kv_seq_len,
        cache_offset,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ---------------- tree attention (same online softmax; tree-structured mask) ---------------- [see: `tree_parents`]
#[cube(launch)]
fn tree_attention_kernel(
    output: &mut Array<f32>,
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    tree_parents: &Array<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _gamma: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) {
    let block = ABSOLUTE_POS / 64usize;
    let lane = ABSOLUTE_POS % 64usize;
    let i = block / num_heads;
    let h = block % num_heads;
    let q_per_kv = num_heads / num_kv_heads;
    let kv_head = h / q_per_kv;
    let q_base = (i * num_heads + h) * head_dim;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    let chunks = (head_dim + 63usize) / 64usize;

    let mut m = f32::new(-1.0e30);
    let mut l = 0.0f32;
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;

    let mut j = 0usize;
    while j < kv_seq_len {
        let mut attend = j < cache_offset;
        if !attend {
            let tree_node = j - cache_offset;
            if tree_node <= i {
                let mut curr = i;
                while curr > 0usize {
                    curr = tree_parents[curr] as usize;
                    if curr == tree_node {
                        attend = true;
                        break;
                    }
                }
            }
        }
        if attend {
            let kv_base = (j * num_kv_heads + kv_head) * head_dim;
            let mut dot = 0.0f32;
            for c in 0..chunks {
                let dim = c * 64usize + lane;
                if dim < head_dim {
                    dot = dot + plane_sum(q[q_base + dim] * k[kv_base + dim]);
                }
            }
            let s = dot * inv_sqrt_d;
            if s > m {
                let corr = (m - s).exp();
                a0 = a0 * corr;
                a1 = a1 * corr;
                a2 = a2 * corr;
                a3 = a3 * corr;
                l = l * corr;
                m = s;
                let w = 1.0f32;
                for c in 0..chunks {
                    let dim = c * 64usize + lane;
                    if dim < head_dim {
                        let add = w * v[kv_base + dim];
                        if c == 0 {
                            a0 = a0 + add;
                        } else if c == 1 {
                            a1 = a1 + add;
                        } else if c == 2 {
                            a2 = a2 + add;
                        } else {
                            a3 = a3 + add;
                        }
                    }
                }
                l = l + w;
            } else {
                let w = (s - m).exp();
                for c in 0..chunks {
                    let dim = c * 64usize + lane;
                    if dim < head_dim {
                        let add = w * v[kv_base + dim];
                        if c == 0 {
                            a0 = a0 + add;
                        } else if c == 1 {
                            a1 = a1 + add;
                        } else if c == 2 {
                            a2 = a2 + add;
                        } else {
                            a3 = a3 + add;
                        }
                    }
                }
                l = l + w;
            }
        }
        j += 1;
    }

    for c in 0..chunks {
        let dim = c * 64usize + lane;
        if dim < head_dim {
            let val = if c == 0 {
                a0
            } else if c == 1 {
                a1
            } else if c == 2 {
                a2
            } else {
                a3
            };
            output[q_base + dim] = val / l;
        }
    }
}

pub fn tree_attention(
    client: &Client,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tree_parents: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gamma: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) -> Vec<f32> {
    let out_len = (1usize + gamma) * num_heads * head_dim;
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let ph = client.create_from_slice(f32::as_bytes(tree_parents));
    let oh = client.empty(out_len * 4);
    let blocks = (1usize + gamma) * num_heads;
    tree_attention_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(blocks as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(oh.clone(), out_len) },
        unsafe { ArrayArg::from_raw_parts(qh.clone(), q.len()) },
        unsafe { ArrayArg::from_raw_parts(kh.clone(), k.len()) },
        unsafe { ArrayArg::from_raw_parts(vh.clone(), v.len()) },
        unsafe { ArrayArg::from_raw_parts(ph.clone(), tree_parents.len()) },
        num_heads,
        num_kv_heads,
        head_dim,
        gamma,
        kv_seq_len,
        cache_offset,
    );
    f32::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

// ---------------- GPTQ correction (Phase 4) ----------------

/// GPTQ diagonal-preconditioned error correction, element-wise: [see: `w_corr = w_approx + α * (w_orig - w_approx) / h_diag[group]`]
#[cube(launch)]
fn gptq_correction_kernel(
    weight_approx: &mut Array<f32>,
    weight_orig: &Array<f32>,
    h_diag: &Array<f32>,
    correction_rate: f32,
    group_size: usize,
    rows: usize,
    cols: usize,
) {
    let flat = ABSOLUTE_POS;
    let n = rows * cols;
    if flat < n {
        let col = flat % cols;
        let group_idx = col / group_size;
        let h = h_diag[group_idx];
        let orig = weight_orig[flat];
        let approx = weight_approx[flat];
        let residual = orig - approx;
        let mut corrected = approx + correction_rate * (residual / h);
        corrected = corrected.max(-65504.0f32);
        corrected = corrected.min(65504.0f32);
        weight_approx[flat] = corrected;
    }
}

pub fn gptq_correction(
    client: &Client,
    weight_approx: &[f32],
    weight_orig: &[f32],
    h_diag: &[f32],
    correction_rate: f32,
    group_size: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let n = rows * cols;
    let ah = client.create_from_slice(f32::as_bytes(weight_approx));
    let oh = client.create_from_slice(f32::as_bytes(weight_orig));
    let hh = client.create_from_slice(f32::as_bytes(h_diag));
    gptq_correction_kernel::launch::<HipRuntime>(
        client,
        CubeCount::Static(((n as u32) + 255) / 256, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts(ah.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(oh.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(hh.clone(), h_diag.len()) },
        correction_rate,
        group_size,
        rows,
        cols,
    );
    f32::from_bytes(&client.read_one(ah).unwrap()).to_vec()
}

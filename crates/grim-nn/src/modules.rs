//! Module-style building blocks: linear, embedding, RMSNorm, RoPE.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::{BackendDevice, Device, DType, Tensor};

use crate::varbuilder::WeightSource;

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;

/// Pick the `BackendDevice` that matches the storage location of `x` so
/// arithmetic ops dispatch to GPU kernels when the tensor lives on a GPU.
/// Falls back to CPU if the requested backend is unavailable in this build.
pub fn pick_device_for_tensor(x: &Tensor) -> Box<dyn BackendDevice> {
    match x.device() {
        Device::Cpu => Box::new(CpuDevice::new()),
        #[cfg(feature = "cuda-mem")]
        Device::Cuda(ordinal) => Box::new(CudaDevice::new(*ordinal)),
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(ordinal) => Box::new(RocmDevice::new(*ordinal)),
        _ => Box::new(CpuDevice::new()),
    }
}

// ---------- Linear ----------

/// Linear: `y = x @ W^T [+ b]` with weight `(out, in)`, optional bias `(out,)`.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Load a Linear layer.
    ///
    /// GGUF stores matrix weights as `[out_dim, in_dim]` (rows = output units,
    /// columns = input units). This matches llama.cpp's convention: `y = x @ W^T`,
    /// so `Linear::forward` transposes the stored weight before matmul.
    pub fn load(ws: &WeightSource<'_>, in_dim: usize, out_dim: usize, has_bias: bool) -> Result<Self> {
        let weight = ws.get([out_dim, in_dim], "weight")?;
        let bias = if has_bias {
            Some(ws.get([out_dim], "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn from_tensor(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = pick_device_for_tensor(x);
        let in_dim = x.shape().dims().last().copied().unwrap_or(0);
        // Weight is GGUF-native: dim(0) = out_dim, dim(1) = in_dim.
        let out_dim = self.weight.shape().dim(0)?;
        let batch = x.shape().elem_count() / in_dim;

        let w_t = transpose_last_two(&self.weight)?;
        let a_storage = x.storage().as_ref();
        let b_storage = w_t.storage().as_ref();
        let (out_s, h) = BackendDevice::matmul(&*dev, a_storage, b_storage, &Shape::new(vec![batch, out_dim]))?;
        h.synchronize()?;
        let mat_out = Tensor::new(
            Arc::from(out_s),
            Shape::new(vec![batch, out_dim]),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        if let Some(b) = &self.bias {
            let broadcast_b = broadcast_bias(b, batch, out_dim)?;
            let (s, hh) = BackendDevice::add(
                &*dev,
                mat_out.storage().as_ref(),
                broadcast_b.storage().as_ref(),
                mat_out.shape(),
            )?;
            hh.synchronize()?;
            return Ok(Tensor::new(
                Arc::from(s),
                mat_out.shape().clone(),
                DType::F32,
                mat_out.provenance().clone(),
                mat_out.device().clone(),
            ));
        }
        Ok(mat_out)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

fn transpose_last_two(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::Shape("transpose_last_two: only 2-D".into()));
    }
    let (a, b) = (dims[0], dims[1]);
    let src = t.to_vec_f32()?;
    let mut out = vec![0.0f32; a * b];
    for i in 0..a {
        for j in 0..b {
            out[j * a + i] = src[i * b + j];
        }
    }
    let new_shape = Shape::new(vec![b, a]);
    if t.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        // Re-upload transposed weights back to the source device so the
        // downstream matmul sees matching CUDA/CUDA storages.
        let dev = pick_device_for_tensor(t);
        let storage = dev.from_cpu(&out, &new_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            new_shape,
            DType::F32,
            t.provenance().clone(),
            t.device().clone(),
        ))
    }
}

fn broadcast_bias(b: &Tensor, batch: usize, out_dim: usize) -> Result<Tensor> {
    let b_vec = b.to_vec_f32()?;
    let mut out = Vec::with_capacity(batch * out_dim);
    for _ in 0..batch {
        out.extend_from_slice(&b_vec);
    }
    if out.len() != batch * out_dim {
        return Err(Error::Shape("broadcast_bias: size mismatch".into()));
    }
    let new_shape = Shape::new(vec![batch, out_dim]);
    if b.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        let dev = pick_device_for_tensor(b);
        let storage = dev.from_cpu(&out, &new_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            new_shape,
            DType::F32,
            b.provenance().clone(),
            b.device().clone(),
        ))
    }
}

// ---------- RMSNorm ----------

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = pick_device_for_tensor(x);
        let dim = x.shape().dims().last().copied().unwrap_or(0);
        let batch = x.shape().elem_count() / dim;
        let out_shape = Shape::new(vec![batch, dim]);
        let (s, h) = BackendDevice::rms_norm(
            &*dev,
            x.storage().as_ref(),
            self.weight.storage().as_ref(),
            self.eps,
            &out_shape,
        )?;
        h.synchronize()?;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
    }
}

// ---------- Embedding ----------

#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn load(ws: &WeightSource<'_>, vocab: usize, dim: usize) -> Result<Self> {
        let weight = match ws.get([vocab, dim], "weight") {
            Ok(t) => t,
            Err(_) => {
                let raw = ws.get([dim, vocab], "weight")?;
                let raw_vec = raw.to_vec_f32()?;
                let mut out = vec![0.0f32; vocab * dim];
                for i in 0..dim {
                    for j in 0..vocab {
                        out[j * dim + i] = raw_vec[i * vocab + j];
                    }
                }
                grim_backend_cpu::cpu_tensor(out, Shape::new(vec![vocab, dim]))
            }
        };
        Ok(Self { weight })
    }

    pub fn forward(&self, indices: &[u32], seq_len: usize, dim: usize) -> Result<Tensor> {
        let dev = pick_device_for_tensor(&self.weight);
        let out_shape = Shape::new(vec![seq_len, dim]);
        let (s, h) = BackendDevice::embedding(&*dev, self.weight.storage().as_ref(), indices, &out_shape)?;
        h.synchronize()?;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            self.weight.provenance().clone(),
            self.weight.device().clone(),
        ))
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// ---------- RoPE ----------

/// Rotary positional embedding — apply RoPE to `(B, S, D)` query/key.
#[derive(Debug, Clone, Copy)]
pub struct Rope {
    pub dim: usize,
    pub base: f32,
}

impl Rope {
    pub fn new(dim: usize, base: f32) -> Self {
        Self { dim, base }
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != self.dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                self.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let half = d / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.base.powf((2 * i) as f32 / d as f32))
            .collect();
        let mut src = x.to_vec_f32()?;
        for bi in 0..b {
            for si in 0..s {
                let pos = positions[si] as f32;
                let base_index = (bi * s + si) * d;
                let mut cos_p = vec![0.0f32; half];
                let mut sin_p = vec![0.0f32; half];
                for i in 0..half {
                    let a = pos * inv_freq[i];
                    cos_p[i] = a.cos();
                    sin_p[i] = a.sin();
                }
                for i in 0..half {
                    let xi = base_index + i;
                    let xj = base_index + i + half;
                    let a = src[xi];
                    let bv = src[xj];
                    src[xi] = a * cos_p[i] - bv * sin_p[i];
                    src[xj] = a * sin_p[i] + bv * cos_p[i];
                }
            }
        }
        let out_shape = Shape::new(vec![b, s, d]);
        if x.device().is_cpu() {
            Ok(grim_backend_cpu::cpu_tensor(src, out_shape))
        } else {
            let dev = pick_device_for_tensor(x);
            let storage = dev.from_cpu(&src, &out_shape, DType::F32)?;
            Ok(Tensor::new(
                Arc::from(storage),
                out_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            ))
        }
    }
}

//! Metal compatibility backend for Grim.
//!
//! Provides the `MetalDevice` and `MetalStorage` structs implementing the `BackendDevice`
//! and `BackendStorage` traits from `grim-tensor`, enabling Metal device target support (MSL).
//! Implements a robust Unified Memory Architecture (UMA) zero-copy FFI and CPU-fallback execution
//! pipeline to ensure full capability compatibility on all supported targets.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{DType, QuantProvenance, Storage as DTypeStorage};
#[cfg(target_vendor = "apple")]
use grim_tensor::dtype::ArithType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

use grim_backend_cpu::{CpuDevice, CpuStorage};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
struct MetalPipelines {
    add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu_mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qkv_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalHandle {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalHandle;

impl ComputeHandle for MetalHandle {
    /// Blocks the host thread until Metal operations tracked by this handle complete.
    fn synchronize(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    /// Checks if the Metal operations tracked by this handle have finished.
    fn is_ready(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandBufferStatus;
            self.command_buffer.status() == MTLCommandBufferStatus::Completed
        }
        #[cfg(not(target_vendor = "apple"))]
        true
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalStorage {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalStorage {
    data: std::sync::Mutex<Vec<f32>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

impl BackendStorage for MetalStorage {
    /// Gets the data type of the storage.
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Gets the quantization provenance of the storage.
    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    /// Gets the shape of the storage.
    fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies the GPU device buffer content back to host memory as an F32 vector.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        #[cfg(target_vendor = "apple")]
        {
            let contents = self.buffer.contents() as *const f32;
            if contents.is_null() {
                return Err(Error::Backend("Metal buffer contents is null".into()));
            }
            let mut out = vec![0.0f32; self.shape.elem_count()];
            unsafe {
                std::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len());
            }
            Ok(out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let data = self.data.lock().unwrap();
            Ok(data.clone())
        }
    }

    /// Returns `self` as `Any` to allow internal downcasting in the backend.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: std::sync::Arc<MetalPipelines>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
}

impl MetalDevice {
    /// Constructs a new device reference for the given ordinal.
    ///
    /// # Panics
    /// Panics if the default device creation fails. Use `try_new` for graceful error propagation.
    pub fn new(ordinal: usize) -> Self {
        Self::try_new(ordinal).unwrap_or_else(|e| {
            panic!("[MetalDevice::new] Failed to initialize Metal device: {:?}", e)
        })
    }

    /// Fallible constructor propagating Metal FFI initialization errors.
    pub fn try_new(ordinal: usize) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCreateSystemDefaultDevice;
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| Error::Backend("No default Metal device found".into()))?;
            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| Error::Backend("Failed to create MTLCommandQueue".into()))?;

            let msl_source = include_str!("kernels.msl");
            let library = device
                .newLibraryWithSource_options_error(&objc2::ns_string!(msl_source), None)
                .map_err(|e| Error::Backend(format!("MSL shader compilation failed: {:?}", e)))?;

            let get_pipeline = |name: &str| -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
                let function = library
                    .newFunctionWithName(&objc2::ns_string!(name))
                    .ok_or_else(|| Error::Backend(format!("MSL function {} not found", name)))?;
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Backend(format!("Failed to create pipeline for {}: {:?}", name, e)))
            };

            let pipelines = std::sync::Arc::new(MetalPipelines {
                add: get_pipeline("grim_add")?,
                mul: get_pipeline("grim_mul")?,
                silu_mul: get_pipeline("grim_silu_mul")?,
                rms_norm: get_pipeline("grim_rms_norm")?,
                softmax: get_pipeline("grim_softmax")?,
                embedding: get_pipeline("grim_embedding")?,
                matmul: get_pipeline("grim_matmul")?,
                qkv_attn: get_pipeline("grim_qkv_attention")?,
            });

            Ok(Self {
                ordinal,
                device,
                command_queue,
                pipelines,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self { ordinal })
        }
    }

    /// Returns the ordinal of this device.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Probes the system for available Metal GPUs.
    pub fn probe() -> Result<Vec<MetalDevice>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Ok(dev) = MetalDevice::try_new(0) {
                Ok(vec![dev])
            } else {
                Ok(vec![])
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok(vec![])
    }
}

impl BackendDevice for MetalDevice {
    /// Allocates a zero-initialized tensor buffer on the Metal device.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLResourceOptions;
            let bytes = shape.elem_count() * dtype_byte_size(&dtype)?;
            let buffer = self
                .device
                .newBufferWithLength_options(bytes as u64, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| Error::Backend("Failed to allocate Metal buffer".into()))?;

            let contents = buffer.contents();
            if !contents.is_null() {
                unsafe {
                    std::ptr::write_bytes(contents, 0, bytes);
                }
            }

            Ok(Box::new(MetalStorage {
                buffer,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let elem_count = shape.elem_count();
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(vec![0.0f32; elem_count]),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    /// Performs matrix multiplication on the Metal device.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            let a_s = a.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal matmul: input a is not MetalStorage".into())
            })?;
            let b_s = b.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal matmul: input b is not MetalStorage".into())
            })?;

            let a_dims = a.shape().dims();
            let b_dims = b.shape().dims();
            if a_dims.len() != 2 || b_dims.len() != 2 {
                return Err(Error::Shape("Metal matmul expects 2-D inputs".into()));
            }
            let (m, k) = (a_dims[0], a_dims[1]);
            let (k2, n) = (b_dims[0], b_dims[1]);
            if k != k2 {
                return Err(Error::ShapeMismatch {
                    expected: a_dims.to_vec(),
                    got: b_dims.to_vec(),
                });
            }

            let dtype_out = DType {
                arith: grim_tensor::dtype::ArithType::F32,
                storage: DTypeStorage::Native,
            };
            let out_storage = self.zeros(out, dtype_out.clone())?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

            let cmd_buffer = self
                .command_queue
                .commandBuffer()
                .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
            let encoder = cmd_buffer
                .computeCommandEncoder()
                .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&self.pipelines.matmul);
            encoder.setBuffer_offset_atIndex(Some(&a_s.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&b_s.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 2);

            let m_val = m as i32;
            let n_val = n as i32;
            let k_val = k as i32;
            unsafe {
                encoder.setBytes_length_atIndex(
                    &m_val as *const i32 as *const std::ffi::c_void,
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    &n_val as *const i32 as *const std::ffi::c_void,
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    &k_val as *const i32 as *const std::ffi::c_void,
                    4,
                    5,
                );
            }

            let threads_per_group = MTLSize::new(16, 16, 1);
            let groups = MTLSize::new(
                ((n + 15) / 16) as u64,
                ((m + 15) / 16) as u64,
                1,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
            encoder.endEncoding();
            cmd_buffer.commit();

            Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.matmul(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise addition on the Metal device.
    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            self.run_elementwise(&self.pipelines.add, a, b, out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.add(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise multiplication on the Metal device.
    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            self.run_elementwise(&self.pipelines.mul, a, b, out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.mul(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise SiLU-multiplication (SwiGLU gate) on the Metal device.
    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            self.run_elementwise(&self.pipelines.silu_mul, gate, up, out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
                cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
            })
        }
    }

    /// Performs RMS Normalization on the Metal device.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal rms_norm: input x is not MetalStorage".into())
            })?;
            let w_s = w.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal rms_norm: input w is not MetalStorage".into())
            })?;

            let out_storage = self.zeros(out, x.dtype())?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

            let total = out.elem_count();
            let row_len = x.shape().dims().last().copied().unwrap_or(1) as i32;

            let cmd_buffer = self
                .command_queue
                .commandBuffer()
                .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
            let encoder = cmd_buffer
                .computeCommandEncoder()
                .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&self.pipelines.rms_norm);
            encoder.setBuffer_offset_atIndex(Some(&x_s.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&w_s.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 2);

            let row_len_val = row_len;
            let eps_val = eps;
            let total_val = total as i32;

            unsafe {
                encoder.setBytes_length_atIndex(
                    &row_len_val as *const i32 as *const std::ffi::c_void,
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    &eps_val as *const f32 as *const std::ffi::c_void,
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    &total_val as *const i32 as *const std::ffi::c_void,
                    4,
                    5,
                );
            }

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
            encoder.endEncoding();
            cmd_buffer.commit();

            Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
                cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
            })
        }
    }

    /// Performs Softmax along the last dimension on the Metal device.
    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal softmax: input x is not MetalStorage".into())
            })?;

            let out_storage = self.zeros(out, x.dtype())?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

            let total = out.elem_count();
            let last_dim = out.dims().last().copied().unwrap_or(1) as i32;

            let cmd_buffer = self
                .command_queue
                .commandBuffer()
                .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
            let encoder = cmd_buffer
                .computeCommandEncoder()
                .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&self.pipelines.softmax);
            encoder.setBuffer_offset_atIndex(Some(&x_s.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 1);

            let last_dim_val = last_dim;
            let total_val = total as i32;

            unsafe {
                encoder.setBytes_length_atIndex(
                    &last_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    2,
                );
                encoder.setBytes_length_atIndex(
                    &total_val as *const i32 as *const std::ffi::c_void,
                    4,
                    3,
                );
            }

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
            encoder.endEncoding();
            cmd_buffer.commit();

            Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let x_vec = x.to_cpu_vec_f32()?;
            let cpu_dev = CpuDevice::new();
            let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
            let x_storage = x_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                Error::Backend("Failed to downcast input x to CpuStorage".into())
            })?;
            let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
            Ok((out_metal, handle))
        }
    }

    /// Performs embedding lookup on the Metal device.
    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            let w_s = weight.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal embedding: weight is not MetalStorage".into())
            })?;

            // Create a temporary buffer for indices and copy them
            let indices_bytes = indices.len() * 4;
            let indices_buffer = self
                .device
                .newBufferWithLength_options(
                    indices_bytes as u64,
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| Error::Backend("Failed to allocate indices buffer".into()))?;
            let indices_contents = indices_buffer.contents() as *mut u32;
            if !indices_contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_contents, indices.len());
                }
            }

            let out_storage = self.zeros(out, weight.dtype())?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

            let embedding_dim = out.dims().last().copied().unwrap_or(1) as i32;
            let num_indices = indices.len() as i32;
            let total = out.elem_count();

            let cmd_buffer = self
                .command_queue
                .commandBuffer()
                .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
            let encoder = cmd_buffer
                .computeCommandEncoder()
                .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&self.pipelines.embedding);
            encoder.setBuffer_offset_atIndex(Some(&w_s.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&indices_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 2);

            unsafe {
                encoder.setBytes_length_atIndex(
                    &embedding_dim as *const i32 as *const std::ffi::c_void,
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    &num_indices as *const i32 as *const std::ffi::c_void,
                    4,
                    4,
                );
            }

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
            encoder.endEncoding();
            cmd_buffer.commit();

            Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let w_vec = weight.to_cpu_vec_f32()?;
            let cpu_dev = CpuDevice::new();
            let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
            let w_storage = w_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                Error::Backend("Failed to downcast weight to CpuStorage".into())
            })?;
            let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
            Ok((out_metal, handle))
        }
    }

    /// Copies a slice of F32 values from host memory to the device storage.
    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLResourceOptions;
            let bytes = shape.elem_count() * dtype_byte_size(&dtype)?;
            let buffer = self
                .device
                .newBufferWithLength_options(bytes as u64, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| Error::Backend("Failed to allocate Metal buffer".into()))?;

            let contents = buffer.contents() as *mut f32;
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len());
                }
            }

            Ok(Box::new(MetalStorage {
                buffer,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(data.to_vec()),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    /// Provide hints about memory usage/advice patterns to the device/system.
    fn advise(&self, _storage: &dyn BackendStorage, _advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        Ok(())
    }
}

impl MetalDevice {
    /// Fused QKV attention matching ROCm / CUDA signatures.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        #[cfg(target_vendor = "apple")]
        {
            let q_s = q.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention q is not MetalStorage".into())
            })?;
            let k_s = k.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention k is not MetalStorage".into())
            })?;
            let v_s = v.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention v is not MetalStorage".into())
            })?;

            let max_s = match out_max {
                Some(m) => Some(m.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("qkv_attention out_max is not MetalStorage".into())
                })?),
                None => None,
            };
            let sum_s = match out_sum {
                Some(s) => Some(s.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("qkv_attention out_sum is not MetalStorage".into())
                })?),
                None => None,
            };

            let out_storage = self.zeros(out, DType::F32)?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

            let cmd_buffer = self
                .command_queue
                .commandBuffer()
                .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
            let encoder = cmd_buffer
                .computeCommandEncoder()
                .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&self.pipelines.qkv_attn);
            encoder.setBuffer_offset_atIndex(Some(&q_s.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&k_s.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&v_s.buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(max_s.map(|m| &m.buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(sum_s.map(|s| &s.buffer), 0, 5);

            let num_heads_val = num_heads as i32;
            let num_kv_heads_val = num_kv_heads as i32;
            let head_dim_val = head_dim as i32;
            let seq_len_val = seq_len as i32;
            let kv_seq_len_val = kv_seq_len as i32;
            let cache_offset_val = cache_offset as i32;
            let inv_sqrt_d_val = 1.0 / (head_dim as f32).sqrt();

            unsafe {
                encoder.setBytes_length_atIndex(
                    &num_heads_val as *const i32 as *const std::ffi::c_void,
                    4,
                    6,
                );
                encoder.setBytes_length_atIndex(
                    &num_kv_heads_val as *const i32 as *const std::ffi::c_void,
                    4,
                    7,
                );
                encoder.setBytes_length_atIndex(
                    &head_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    8,
                );
                encoder.setBytes_length_atIndex(
                    &seq_len_val as *const i32 as *const std::ffi::c_void,
                    4,
                    9,
                );
                encoder.setBytes_length_atIndex(
                    &kv_seq_len_val as *const i32 as *const std::ffi::c_void,
                    4,
                    10,
                );
                encoder.setBytes_length_atIndex(
                    &cache_offset_val as *const i32 as *const std::ffi::c_void,
                    4,
                    11,
                );
                encoder.setBytes_length_atIndex(
                    &inv_sqrt_d_val as *const f32 as *const std::ffi::c_void,
                    4,
                    12,
                );
            }

            let threads_per_group = MTLSize::new(16, 16, 1);
            let groups = MTLSize::new(
                ((seq_len + 15) / 16) as u64,
                ((num_heads + 15) / 16) as u64,
                1,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
            encoder.endEncoding();
            cmd_buffer.commit();

            Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = out_max;
            let _ = out_sum;
            // Simple host-fallback simulation for unit tests
            let q_vec = q.to_cpu_vec_f32()?;
            let k_vec = k.to_cpu_vec_f32()?;
            let v_vec = v.to_cpu_vec_f32()?;

            let mut out_vec = vec![0.0f32; out.elem_count()];
            let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

            for i in 0..seq_len {
                for h in 0..num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let kv_head = h / q_per_kv;
                    let q_offset = (i * num_heads + h) * head_dim;
                    let abs_i = cache_offset as usize + i;
                    let range_len = if abs_i < kv_seq_len { abs_i + 1 } else { kv_seq_len };

                    let mut running_max = -1e30_f32;
                    let mut running_sum = 0.0_f32;

                    let mut scores = vec![0.0f32; range_len];
                    for j in 0..range_len {
                        let mut score = 0.0_f32;
                        for d in 0..head_dim {
                            score += q_vec[q_offset + d] * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        score *= inv_sqrt_d;
                        scores[j] = score;
                        if score > running_max {
                            running_max = score;
                        }
                    }

                    for j in 0..range_len {
                        running_sum += (scores[j] - running_max).exp();
                    }

                    for d in 0..head_dim {
                        let mut acc = 0.0_f32;
                        for j in 0..range_len {
                            let weight = (scores[j] - running_max).exp() / (if running_sum > 0.0_f32 { running_sum } else { 1.0_f32 });
                            acc += weight * v_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        out_vec[q_offset + d] = acc;
                    }
                }
            }

            let out_storage = self.from_cpu(&out_vec, out, DType::F32)?;
            Ok((out_storage, Box::new(MetalHandle)))
        }
    }

    #[cfg(target_vendor = "apple")]
    fn run_elementwise(
        &self,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input a is not MetalStorage".into())
        })?;
        let b_s = b.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input b is not MetalStorage".into())
        })?;

        let out_storage = self.zeros(out, a.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| Error::Backend("Failed to create command buffer".into()))?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Error::Backend("Failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer_offset_atIndex(Some(&a_s.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&b_s.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&out_s.buffer), 0, 2);

        let total_val = total as i32;
        unsafe {
            encoder.setBytes_length_atIndex(
                &total_val as *const i32 as *const std::ffi::c_void,
                4,
                3,
            );
        }

        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
        cmd_buffer.commit();

        Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
    }
}

/// Run binary operations on the CPU fallback pipeline.
#[cfg(not(target_vendor = "apple"))]
fn run_fallback_binary<F>(
    device: &MetalDevice,
    a: &dyn BackendStorage,
    b: &dyn BackendStorage,
    out: &Shape,
    op: F,
) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>
where
    F: FnOnce(&CpuDevice, &CpuStorage, &CpuStorage, &Shape) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
{
    let a_vec = a.to_cpu_vec_f32()?;
    let b_vec = b.to_cpu_vec_f32()?;

    let cpu_dev = CpuDevice::new();
    let a_cpu = cpu_dev.from_cpu(&a_vec, a.shape(), a.dtype())?;
    let b_cpu = cpu_dev.from_cpu(&b_vec, b.shape(), b.dtype())?;

    let a_storage = a_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input a to CpuStorage".into())
    })?;
    let b_storage = b_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input b to CpuStorage".into())
    })?;

    let (res_storage, handle) = op(&cpu_dev, a_storage, b_storage, out)?;

    let res_vec = res_storage.to_cpu_vec_f32()?;
    let out_metal = device.from_cpu(&res_vec, out, a.dtype())?;

    Ok((out_metal, handle))
}

/// Helper function to retrieve the size in bytes of a data type.
#[allow(dead_code)]
fn dtype_byte_size(dtype: &DType) -> Result<usize> {
    #[cfg(target_vendor = "apple")]
    {
        match dtype.arith {
            ArithType::F32 | ArithType::U32 => Ok(4),
            ArithType::F16 | ArithType::BF16 => Ok(2),
            ArithType::I64 => Ok(8),
            ArithType::U8 => Ok(1),
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = dtype;
        Ok(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_probe() {
        let devices = MetalDevice::probe().unwrap();
        // Since we are running tests on non-apple (Linux), probe should return an empty list.
        #[cfg(not(target_vendor = "apple"))]
        assert!(devices.is_empty());
        #[cfg(target_vendor = "apple")]
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_metal_zeros() {
        let dev = MetalDevice::new(0);
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 4]);
        let vec = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(vec, vec![0.0f32; 8]);
    }

    #[test]
    fn test_metal_matmul() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![2, 2]);
        let (out, handle) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_metal_add() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[3.0, 4.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let (out, handle) = dev.add(a.as_ref(), b.as_ref(), &Shape::new(vec![2])).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![4.0, 6.0]);
    }

    #[test]
    fn test_metal_qkv_attention() {
        let dev = MetalDevice::new(0);
        let q = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let k = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let v = dev.from_cpu(&[2.0, 3.0, 4.0, 5.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![1, 2, 2]);
        let (out, handle) = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), 2, 1, 0, &out_shape, None, None).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![2.0, 3.0, 4.0, 5.0]);
    }
}

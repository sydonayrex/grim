//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! GPTQ / AWQ / Marlin / BitNet / OSTQuant / WNA16 weight-only GEMM launchers.

use std::ffi::c_void;

use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::pinned::RocmPinnedBuffer;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, HipMemcpyKind, arg, check_hip, hipMemcpyAsync, hipStreamSynchronize};

impl RocmDevice {
    pub(crate) fn launch_dequant_wna16(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_weights: usize,
        n_bit: i32,
        num_blocks: i32,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 dequant: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 dequant: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((num_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("wna16 dequant: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = num_weights as i32;
        let mut n_bit_v = n_bit;
        let mut nb = num_blocks;
        self.launch_compute_kernel(
            "grim_dequant_wna16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut packed),
                arg(&mut out),
                arg(&mut n_w),
                arg(&mut n_bit_v),
                arg(&mut nb),
            ],
        )
    }

    /// Public dequant service (quant workstream): WNA16 packed blob → F32 weights, decoded on-device.
    /// Layout contract mirrors `Storage::WNA16`: [u32 n_bit][u32 num_blocks][codes][f16 block scales][f32 tensor scale], 256-weight blocks, MSB-first codes.
    pub fn dequant_wna16_blob_to_f32(
        &self,
        blob: &dyn BackendStorage,
        num_weights: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let packed_rocm = blob
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("wna16 blob not rocm".into()))?;
        let (n_bit_raw, num_blocks_raw) = Self::wna16_read_params(packed_rocm, self.ordinal)?;
        let n_bit = n_bit_raw as u8;
        let num_blocks = num_blocks_raw as usize;
        self.dequant_wna16_to_f32(packed_rocm, num_weights, n_bit, num_blocks)
    }

    pub fn dequant_wna16_to_f32(
        &self,
        packed: &RocmStorage,
        num_weights: usize,
        n_bit: u8,
        num_blocks: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let out_shape = Shape::from_slice(&[num_weights]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let _h =
            self.launch_dequant_wna16(packed, &out, num_weights, n_bit as i32, num_blocks as i32)?;
        let stream = self.active_stream();
        check_hip("hipStreamSynchronize(wna16 dequant)", unsafe {
            crate::device::handles::hipStreamSynchronize(stream)
        })?;
        Ok(Box::new(out))
    }

    pub(crate) fn launch_dequant_embedding_wna16_int(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        total_elements: usize,
        n_bit: i32,
        embedding_dim: i32,
        tensor_scale: f32,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage.device_ptr.ok_or_else(|| {
            Error::Backend("emb_wna16_int dequant: packed has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("emb_wna16_int dequant: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((total_elements as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("emb_wna16_int dequant: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_el = total_elements as i32;
        let mut n_bit_v = n_bit;
        let mut dim = embedding_dim;
        let mut ts = tensor_scale;
        self.launch_compute_kernel(
            "grim_dequant_embedding_wna16_int",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut packed),
                arg(&mut out),
                arg(&mut n_el),
                arg(&mut n_bit_v),
                arg(&mut dim),
                arg(&mut ts),
            ],
        )
    }

    /// Public dequant service (quant workstream): EmbeddingWNA16Int packed blob → F32 embedding table, decoded on-device.
    /// Layout contract mirrors `Storage::EmbeddingWNA16Int`: [u32 n_bit][u32 embedding_dim][u32 num_rows][codes MSB-first].
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_embedding_wna16_int_to_f32(
        &self,
        packed: &RocmStorage,
        total_elements: usize,
        n_bit: u8,
        embedding_dim: usize,
        tensor_scale: f32,
    ) -> Result<Box<dyn BackendStorage>> {
        let out_shape = Shape::from_slice(&[total_elements]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let _h = self.launch_dequant_embedding_wna16_int(
            packed,
            &out,
            total_elements,
            n_bit as i32,
            embedding_dim as i32,
            tensor_scale,
        )?;
        let stream = self.active_stream();
        check_hip("hipStreamSynchronize(emb wna16 dequant)", unsafe {
            crate::device::handles::hipStreamSynchronize(stream)
        })?;
        Ok(Box::new(out))
    }

    /// Launch the GPTQ/EfficientQAT GroupInt fused dequant-GEMM (forward).
    /// `b_storage` holds the documented length-prefixed four-segment packed layout (`GpuIntConfig`); `qw/qz/sc/gi` are byte offsets of each.
    pub(crate) fn launch_gptq_dequant_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("gptq gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("gptq gemm: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq gemm: unsupported bit width {bits}"
                )));
            }
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut has_gi = if has_g_idx { 1 } else { 0 };
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;
        let mut gi = gi_off;

        self.launch_compute_kernel(
            "grim_gptq_dequant_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_gi),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
                arg(&mut gi),
            ],
        )
    }

    /// Launch the GPTQ/EfficientQAT GroupInt fused dequant-GEMM (backward).
    /// Computes `dX[M, K] = dY[M, N] @ dequant(B)` from the same packed blob as [`Self::launch_gptq_dequant_gemm`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_gptq_dequant_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: b has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("gptq backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("gptq backward: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq backward: unsupported bit width {bits}"
                )));
            }
        };

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut has_gi = if has_g_idx { 1 } else { 0 };
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;
        let mut gi = gi_off;

        self.launch_compute_kernel(
            "grim_gptq_dequant_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_gi),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
                arg(&mut gi),
            ],
        )
    }

    /// Compute the length-prefixed GroupInt segment offsets for a packed weight blob of `blob_bytes` bytes.
    /// Returns `(qw_off, qz_off, sc_off, gi_off, has_g_idx)`.
    pub(crate) fn gptq_segment_offsets(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64, i64, bool)> {
        const _: usize = 32; // four interleaved u64 length prefixes total
        let vpw: usize = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq gemm: unsupported bit width {bits}"
                )));
            }
        };
        let qw_len = k.div_ceil(vpw) * n * 4;
        let groups = k.div_ceil(group_size);
        let qz_len = groups * n.div_ceil(vpw) * 4;
        let sc_len = groups * n * 4;

        // Each segment is preceded by ITS OWN u64 length prefix: [u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales][u64 gi_len][g_idx] so data starts
        // are 8 / (8+qw+8) / (8+qw+8+qz+8) / (+8+sc), and the blob ends with the (possibly empty-length) g_idx prefix.
        let qz_data = 8 + qw_len + 8;
        let sc_data = qz_data + qz_len + 8;
        let gi_data = sc_data + sc_len + 8;

        let no_gi_total = gi_data; // empty g_idx segment: just the zeroed u64
        let gi_total_u32 = gi_data + k * 4;
        let gi_total_u64 = gi_data + k * 8;

        let has_g_idx = if blob_bytes == no_gi_total {
            false
        } else if blob_bytes == gi_total_u32 {
            true
        } else if blob_bytes == gi_total_u64 {
            return Err(Error::Backend(
                "gptq gemm: 64-bit g_idx entries not supported by the fused kernel".into(),
            ));
        } else {
            return Err(Error::Backend(format!(
                "gptq gemm: packed blob size {blob_bytes} matches no valid \
                 GroupInt layout for bits={bits} group_size={group_size} k={k} n={n} \
                 (expected {no_gi_total}, {gi_total_u32}, or {gi_total_u64})"
            )));
        };

        Ok((8, qz_data as i64, sc_data as i64, gi_data as i64, has_g_idx))
    }

    /// Launch the AWQ fused dequant-GEMM (forward).
    /// Computes `C[M, N] = A[M, K] @ dequant(B)^T` where B is packed in the native.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_awq_dequant_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: A has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: B has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("awq gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("awq gemm: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq gemm: unsupported bit width {bits}"
                )));
            }
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut outptr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;

        self.launch_compute_kernel(
            "grim_awq_dequant_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut outptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
            ],
        )
    }

    /// Launch the AWQ fused dequant-GEMM (backward dX).
    /// Computes `dX[M, K] = dY[M, N] @ dequant(B)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_awq_dequant_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: b has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("awq backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("awq backward: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq backward: unsupported bit width {bits}"
                )));
            }
        };

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;

        self.launch_compute_kernel(
            "grim_awq_dequant_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
            ],
        )
    }

    /// Compute the length-prefixed AWQ segment offsets for a packed
    /// weight blob of `blob_bytes` bytes. Returns `(qw_off, qz_off, sc_off)`.
    pub fn awq_segment_offsets(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64)> {
        let vpw: usize = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq gemm: unsupported bit width {bits}"
                )));
            }
        };
        let qw_len = k.div_ceil(vpw) * n * 4;
        let groups = k.div_ceil(group_size);
        let qz_len = groups * n.div_ceil(vpw) * 4;
        let sc_len = groups * n * 2; // f16 scales = 2 bytes each

        // Layout: [u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales (f16)]
        let qz_data = 8 + qw_len + 8;
        let sc_data = qz_data + qz_len + 8;
        let total_expected = sc_data + sc_len;

        if blob_bytes != total_expected {
            return Err(Error::Backend(format!(
                "awq gemm: packed blob size {blob_bytes} does not match expected {total_expected} \
                 for bits={bits} group_size={group_size} k={k} n={n}"
            )));
        }

        Ok((8, qz_data as i64, sc_data as i64))
    }

    /// Compute the length-prefixed OSTQuant segment offsets for a packed
    /// weight blob of `blob_bytes` bytes. Returns `(qw_off, sc_off, zr_off)`.
    pub fn ostquant_segment_offsets(
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64)> {
        let words_per_col = k / 8;
        let qw_len = n * words_per_col * 4;
        let n_groups = k.div_ceil(group_size);
        let sc_len = n * n_groups * 2; // bf16
        let zr_len = n * n_groups; // u8

        // Layout: [u64 qw_len][qweight][u64 sc_len][scales][u64 zr_len][zeros]
        let sc_data = 8 + qw_len + 8;
        let zr_data = sc_data + sc_len + 8;
        let total_expected = zr_data + zr_len;

        if blob_bytes != total_expected {
            return Err(Error::Backend(format!(
                "ostquant gemv: packed blob size {blob_bytes} does not match expected {total_expected} \
                 for group_size={group_size} k={k} n={n}"
            )));
        }

        Ok((8, sc_data as i64, zr_data as i64))
    }

    /// Launch OSTQuant W4A4 GEMV directly over the resident blob:
    /// `[u64 qw_len][qweight][u64 sc_len][scales][u64 zr_len][zeros]`
    pub fn launch_w4a4_ostquant_gemv_blob(
        &self,
        a_storage: &RocmStorage,
        blob_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        if k % 128 != 0 {
            return Err(Error::Backend(format!(
                "launch_w4a4_ostquant_gemv_blob: K={k} must be divisible by 128"
            )));
        }
        let (qw_off, sc_off, zr_off) =
            Self::ostquant_segment_offsets(group_size, k, n, blob_storage.bytes())?;

        let blob_ptr = blob_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("ostquant gemv: blob has no device ptr".into()))?;

        let n_groups = k / 128;
        let codes_words = m * n_groups * 16;
        let scales_elems = m * n_groups;
        let sums_elems = m * n_groups;

        let codes_bytes = codes_words * std::mem::size_of::<u32>();
        let scales_bytes = scales_elems * std::mem::size_of::<f32>();
        let sums_bytes = sums_elems * std::mem::size_of::<i32>();

        let mut codes_guard = self
            .act_u4_codes_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_codes = match codes_guard.as_ref() {
            Some(s) => s.bytes < codes_bytes,
            None => true,
        };
        if need_codes {
            *codes_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![codes_words]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let mut scales_guard = self
            .act_u4_scales_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_scales = match scales_guard.as_ref() {
            Some(s) => s.bytes < scales_bytes,
            None => true,
        };
        if need_scales {
            *scales_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![scales_elems]),
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let mut sums_guard = self
            .act_u4_sums_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_sums = match sums_guard.as_ref() {
            Some(s) => s.bytes < sums_bytes,
            None => true,
        };
        if need_sums {
            *sums_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![sums_elems]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let act_codes = codes_guard.as_ref().unwrap();
        let act_scales = scales_guard.as_ref().unwrap();
        let act_sums = sums_guard.as_ref().unwrap();

        // Quantize activations to u4 groups
        self.launch_quantize_u4_group128(a_storage, act_codes, act_scales, act_sums, m, k)?;

        // Pointers into blob
        let a_codes_ptr = act_codes.device_ptr.unwrap();
        let a_scales_ptr = act_scales.device_ptr.unwrap();
        let a_sums_ptr = act_sums.device_ptr.unwrap();
        let b_qw_ptr = (blob_ptr as usize + qw_off as usize) as *mut c_void;
        let b_sc_ptr = (blob_ptr as usize + sc_off as usize) as *mut c_void;
        let b_zr_ptr = (blob_ptr as usize + zr_off as usize) as *mut c_void;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("ostquant gemv: out has no device ptr".into()))?;

        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);

        let mut a_c = a_codes_ptr;
        let mut a_s = a_scales_ptr;
        let mut a_sum = a_sums_ptr;
        let mut b_qw = b_qw_ptr;
        let mut b_sc = b_sc_ptr;
        let mut b_zr = b_zr_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_dot8_w4a4_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_c),
                arg(&mut a_s),
                arg(&mut a_sum),
                arg(&mut b_qw),
                arg(&mut b_sc),
                arg(&mut b_zr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Audit-wiring (quant workstream): W4A16 blobs are a SINGLE packed segment pair - `[codes (N*K/8 u32)][scales (N*groups f32)]` per the `Storage::W4A16` layout contract -
    /// so the dense dispatch path needs a launcher that derives the scales pointer from the same device buffer instead of requiring two separate storages.
    pub fn launch_marlin_gemm_w4a16_blob(
        &self,
        a_storage: &RocmStorage,
        blob_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        if k % 8 != 0 {
            return Err(Error::Backend(format!(
                "marlin_w4a16: K={k} must be divisible by 8"
            )));
        }
        let codes_bytes = n * (k / 8) * std::mem::size_of::<u32>();
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: a has no device ptr".into()))?;
        let blob_ptr = blob_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: blob has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: out has no device ptr".into()))?;
        if blob_storage.bytes()
            < codes_bytes + n * (k.div_ceil(group_size)) * std::mem::size_of::<f32>()
        {
            return Err(Error::Backend(
                "marlin_w4a16: blob smaller than codes+scales segments".into(),
            ));
        }

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        // Kernel contract: A row-major [M, K] f32; C [M, N] f32.
        let mut aptr = a_ptr;
        let mut bptr = blob_ptr;
        let mut sptr = (blob_ptr as usize + codes_bytes) as *mut c_void;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut gs = group_size as i32;

        self.launch_compute_kernel(
            "grim_marlin_gemm_w4a16_f32",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut gs),
            ],
        )
    }

    /// Phase 4.5b: Launch OSTQuant W4A4 GEMV using native sudot8 on RDNA4 gfx1200/gfx1201.
    /// Handles activation quantization into scratch buffers followed by grim_dot8_w4a4_gemv.
    pub fn launch_w4a4_ostquant_gemv(
        &self,
        a_storage: &RocmStorage,
        b_qweight: &RocmStorage,
        b_scales: &RocmStorage,
        b_zeros: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        if k % 128 != 0 {
            return Err(Error::Backend(format!(
                "launch_w4a4_ostquant_gemv: K={k} must be divisible by 128"
            )));
        }
        let n_groups = k / 128;
        let codes_words = m * n_groups * 16;
        let scales_elems = m * n_groups;
        let sums_elems = m * n_groups;

        // Ensure scratch buffers are allocated
        let codes_bytes = codes_words * std::mem::size_of::<u32>();
        let scales_bytes = scales_elems * std::mem::size_of::<f32>();
        let sums_bytes = sums_elems * std::mem::size_of::<i32>();

        let mut codes_guard = self
            .act_u4_codes_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_codes = match codes_guard.as_ref() {
            Some(s) => s.bytes < codes_bytes,
            None => true,
        };
        if need_codes {
            *codes_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![codes_words]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let mut scales_guard = self
            .act_u4_scales_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_scales = match scales_guard.as_ref() {
            Some(s) => s.bytes < scales_bytes,
            None => true,
        };
        if need_scales {
            *scales_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![scales_elems]),
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let mut sums_guard = self
            .act_u4_sums_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_sums = match sums_guard.as_ref() {
            Some(s) => s.bytes < sums_bytes,
            None => true,
        };
        if need_sums {
            *sums_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![sums_elems]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }

        let act_codes = codes_guard.as_ref().unwrap();
        let act_scales = scales_guard.as_ref().unwrap();
        let act_sums = sums_guard.as_ref().unwrap();

        // Quantize activations to u4 groups
        self.launch_quantize_u4_group128(a_storage, act_codes, act_scales, act_sums, m, k)?;

        // Launch sudot8 W4A4 GEMV
        self.launch_dot8_w4a4_gemv(
            act_codes,
            act_scales,
            act_sums,
            b_qweight,
            b_scales,
            b_zeros,
            out_storage,
            m,
            n,
            k,
        )
    }

    /// Read the WNA16 blob header ([u32 n_bit][u32 num_blocks]) via pinned D2H.
    pub(crate) fn wna16_read_header(blob: &RocmStorage, ordinal: usize) -> Result<[u8; 8]> {
        let dev = RocmDevice::try_new(ordinal)?;
        let _g = crate::device::util::DeviceGuard::set(ordinal as i32);
        let mut pinned = RocmPinnedBuffer::<u8>::alloc_on(ordinal, 8)?;
        let ptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 header: no device ptr".into()))?;
        check_hip("wna16 header D2H", unsafe {
            hipMemcpyAsync(
                pinned.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                8,
                HipMemcpyKind::DeviceToHost,
                dev.active_stream(),
            )
        })?;
        check_hip("wna16 header sync", unsafe {
            hipStreamSynchronize(dev.active_stream())
        })?;
        let mut out = [0u8; 8];
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(pinned.as_ptr(), 8) });
        Ok(out)
    }

    /// Safely unpacks the `(n_bit, num_blocks)` tuple from a WNA16 blob header with error propagation.
    pub(crate) fn wna16_read_params(blob: &RocmStorage, ordinal: usize) -> Result<(u32, u32)> {
        let hdr = Self::wna16_read_header(blob, ordinal)?;
        let n_bit_bytes: [u8; 4] = hdr[0..4]
            .try_into()
            .map_err(|e| Error::Backend(format!("wna16 header n_bit slice error: {e}")))?;
        let blocks_bytes: [u8; 4] = hdr[4..8]
            .try_into()
            .map_err(|e| Error::Backend(format!("wna16 header num_blocks slice error: {e}")))?;
        let n_bit = u32::from_le_bytes(n_bit_bytes);
        let num_blocks = u32::from_le_bytes(blocks_bytes);
        Ok((n_bit, num_blocks))
    }

    /// Public materialization service (quant workstream): dequantize a Marlin W4A16 packed expert blob to row-major F32 [k_dim?
    /// no -] Returns Dᵀ flattened ([k_dim, n_rows] where C = A @ Dᵀ was computed.
    pub fn dequant_w4a16_blob_to_f32(
        &self,
        blob: &RocmStorage,
        n_rows: usize,
        k_dim: usize,
        group_size: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        use grim_tensor::ArithType;
        let id: Vec<f32> = (0..k_dim)
            .flat_map(|i| (0..k_dim).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
            .collect();
        let a = self.from_cpu(&id, &Shape::new(vec![k_dim, k_dim]), DType::F32)?;
        let out_shape = Shape::new(vec![k_dim, n_rows]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("identity not rocm".into()))?;
        let _h = self
            .launch_marlin_gemm_w4a16_blob(a_rocm, blob, &out, k_dim, n_rows, k_dim, group_size)?;
        Ok(Box::new(out))
    }

    /// Public materialization service (quant workstream): run the GPTQ forward-dequant GEMM with an
    /// identity activation so C = D, the full row-major [n_out, k_in] dequantized weight.
    #[allow(clippy::too_many_arguments)]
    pub fn gptq_dequant_identity_to_f32(
        &self,
        blob: &RocmStorage,
        n_out: usize,
        k_in: usize,
        bits: u8,
        group_size: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        use grim_tensor::ArithType;
        let mut vpw: i32 = match bits {
            2 => 16,
            3 => 32,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq dequant identity: unsupported bit width {bits}"
                )));
            }
        };
        let (qw, qz, sc, gi, has_g_idx) =
            Self::gptq_segment_offsets(bits, group_size, k_in, n_out, blob.bytes())?;
        let mut has_i = if has_g_idx { 1 } else { 0 };
        // C = A @ D with A = I[K=k_in] gives C = D row-major [k_in, n_out]
        // (the CALLER transposes to weight layout [n_out, k_in]).
        let id: Vec<f32> = (0..k_in)
            .flat_map(|i| (0..k_in).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
            .collect();
        let a = self.from_cpu(&id, &Shape::new(vec![k_in, k_in]), DType::F32)?;
        let out_shape = Shape::new(vec![k_in, n_out]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("identity not rocm".into()))?;
        let mut aptr = a_rocm.device_ptr_checked()? as *mut c_void;
        let mut bptr = blob.device_ptr_checked()? as *mut c_void;
        let mut optr = out.device_ptr_checked()? as *mut c_void;
        let mut m_i = k_in as i32;
        let mut n_i = n_out as i32;
        let mut k_i = k_in as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;

        let mut qw_i = qw;
        let mut qz_i = qz;
        let mut sc_i = sc;
        let mut gi_i = gi;
        let grid_x = (n_out * n_out).div_ceil(256) as u32;
        self.launch_compute_kernel(
            "grim_gptq_dequant_gemm",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut m_i),
                arg(&mut n_i),
                arg(&mut k_i),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_i),
                arg(&mut qw_i),
                arg(&mut qz_i),
                arg(&mut sc_i),
                arg(&mut gi_i),
            ],
        )?;
        Ok(Box::new(out))
    }

    /// Launch Marlin-style Interleaved W4A16 GEMM.
    pub fn launch_marlin_gemm_w4a16(
        &self,
        a_storage: &RocmStorage,
        b_w4_storage: &RocmStorage,
        scales_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: a has no device ptr".into()))?;
        let b_ptr = b_w4_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: b has no device ptr".into()))?;
        let scales_ptr = scales_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: scales has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sptr = scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut gs = group_size as i32;

        // Select kernel based on scales dtype (what the kernel actually reads),
        // not output dtype. Out can be F16 or F32 regardless of scale precision.
        let kernel_name = match scales_storage.dtype.arith {
            grim_tensor::ArithType::F16 => "grim_marlin_gemm_w4a16",
            _ => "grim_marlin_gemm_w4a16_f32",
        };

        self.launch_compute_kernel(
            kernel_name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut gs),
            ],
        )
    }

    /// Launch BitNet b1.58 Ternary GEMM (W1.58A8).
    pub fn launch_bitnet_gemm_w158a8(
        &self,
        a_storage: &RocmStorage,
        b_ternary_storage: &RocmStorage,
        scale_b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        scale_a: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: a has no device ptr".into()))?;
        let b_ptr = b_ternary_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: b has no device ptr".into()))?;
        let scale_b_ptr = scale_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: scale_b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sbptr = scale_b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut sa = scale_a;

        self.launch_compute_kernel(
            "grim_bitnet_gemm_w158a8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sbptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
            ],
        )
    }
}

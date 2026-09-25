//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! Host-side dequantization wrappers (device->host reads and pure-CPU mirrors).

use grim_tensor::Shape;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;

impl RocmDevice {
    /// Launch the JIT compiled Q8_0 dequantization kernel.  Reads packed [see: `packed_storage`, `n_weights`, `out_storage`, `materialize()`]
    pub fn dequantize_q8_0(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK8_0: usize = 32;
        // Q8_0 stores weights as packed bytes: each block is 34 bytes (2-byte [see: `n_blocks * 32`]
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / (QK8_0 + 2);
        let n_weights = n_blocks * QK8_0;
        let f32_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q8_0(packed, &f32_storage, n_blocks)?;
        Ok(f32_storage)
    }

    /// Dequantize Q8_0 packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q8_0(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize Q4_K packed bytes to F32 on the GPU.
    /// [see: `block_q4_K`] `packed` must hold `n_blocks` × 144-byte super-blocks; `out` must hold `n_blocks` × 256.
    pub fn dequantize_q4k(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK4_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / BLOCK_BYTES;
        let n_weights = n_blocks * QK4_K;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q4k(packed, &out_storage, n_blocks)?;
        Ok(out_storage)
    }

    /// Dequantize Q4_K packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q4k(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Run any standalone IQ dequant kernel against `bytes` and return `elem_count` f32 values.
    fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        block_bytes: usize,
        kernel: &str,
    ) -> Result<Vec<f32>> {
        const QK: usize = 256;
        let n_blocks = bytes.len() / block_bytes;
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_blocks * QK]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        match kernel {
            "grim_dequant_iq2xxs" => {
                self.launch_dequant_iq2xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2xs" => {
                self.launch_dequant_iq2xs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2s" => {
                self.launch_dequant_iq2s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3xxs" => {
                self.launch_dequant_iq3xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3s" => {
                self.launch_dequant_iq3s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4nl" => {
                self.launch_dequant_iq4nl(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4xs" => {
                self.launch_dequant_iq4xs(&packed, &out_storage, n_blocks)?;
            }
            other => {
                return Err(Error::Backend(format!(
                    "dequantize_iq_host: unknown kernel {other}"
                )));
            }
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize IQ2_XXS packed bytes via the ROCm kernel. 66 bytes / 256-elem super-block.
    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 66, "grim_dequant_iq2xxs")
    }

    /// Dequantize IQ2_XS packed bytes via the ROCm kernel. 74 bytes / 256-elem super-block.
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 74, "grim_dequant_iq2xs")
    }

    /// Dequantize IQ2_S packed bytes via the ROCm kernel. 82 bytes / 256-elem super-block.
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 82, "grim_dequant_iq2s")
    }

    /// Dequantize IQ3_XXS packed bytes via the ROCm kernel. 96 bytes / 256-elem super-block.
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 96, "grim_dequant_iq3xxs")
    }

    /// Dequantize IQ3_S packed bytes via the ROCm kernel. 110 bytes / 256-elem super-block.
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 110, "grim_dequant_iq3s")
    }

    /// Dequantize IQ4_NL packed bytes via the ROCm kernel. 170 bytes / 256-elem super-block.
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 170, "grim_dequant_iq4nl")
    }

    /// Dequantize IQ4_XS packed bytes via the ROCm kernel. 178 bytes / 256-elem super-block.
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 178, "grim_dequant_iq4xs")
    }

    /// Dequantize packed FP8 bytes (4-byte f32 LE scale header, then one E4M3 code per element).
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let scale = if bytes.len() >= 4 {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            1.0
        };
        let payload = if bytes.len() >= 4 { &bytes[4..] } else { bytes };
        let packed = RocmStorage::copy_from_host_raw_bytes(
            payload,
            &Shape::new(vec![payload.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_fp8(&packed, &out_storage, elem_count)?;
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        for v in values.iter_mut() {
            *v *= scale;
        }
        Ok(values)
    }

    /// Helper to split an MXFP single-buffer (length-prefixed codes/exps segments) into two device buffers.
    /// Reuses the same framing as `grim_quant::dequant_mxfp4`/`dequant_mxfp8`.
    pub(crate) fn split_dequant_mxfp(
        &self,
        bytes: &[u8],
        elem_count: usize,
        kernel: &str,
    ) -> Result<Vec<f32>> {
        let mut cursor = 0usize;
        let read_segment = |buf: &[u8], cur: &mut usize| -> Result<Vec<u8>> {
            let len = u64::from_le_bytes(
                buf[*cur..*cur + 8]
                    .try_into()
                    .map_err(|_| Error::Backend("mxfp: bad length prefix".into()))?,
            ) as usize;
            *cur += 8;
            let seg = buf[*cur..*cur + len].to_vec();
            *cur += len;
            Ok(seg)
        };
        let codes = read_segment(bytes, &mut cursor)?;
        let exps = read_segment(bytes, &mut cursor)?;

        let codes_storage = RocmStorage::copy_from_host_raw_bytes(
            &codes,
            &Shape::new(vec![codes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let exps_storage = RocmStorage::copy_from_host_raw_bytes(
            &exps,
            &Shape::new(vec![exps.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;

        if kernel.contains("mxfp4") {
            self.launch_dequant_mxfp4(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        } else {
            self.launch_dequant_mxfp8(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize an MXFP4 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp4")
    }

    /// Dequantize an MXFP8 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp8")
    }

    /// Dequantize NVFP4 interleaved packed bytes (1 E8M0 scale byte + 8 codes per 16 weights) to f32.
    pub fn dequantize_nvfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_nvfp4(&packed, &out_storage, elem_count)?;
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }
}

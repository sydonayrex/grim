//! FreeToken Weight (FTW) bank layout and fast bootstrap loading.
//!
//! Conventional checkpoints store expert weights as individual fragmented tensors per layer,
//! which requires extensive tensor discovery, dictionary lookups, and scatter allocations at startup.
//! FTW normalizes MoE weights into contiguous expert banks whose leading dimension is the
//! flattened `(layer_idx * num_experts + expert_idx)` index.
//!
//! This layout allows zero-repack direct I/O loading straight into host DRAM banks and unified
//! single-descriptor DMA transfers over PCIe.

use std::collections::HashMap;

use grim_tensor::error::{Error, Result};

/// Quantization format of the expert banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtwQuantFormat {
    /// Dense BF16 weights. Banks: `["gate_up", "down"]`.
    Bf16,
    /// Block-scaled FP8 weights (128x128 tiles). Banks: `["gate_up", "gate_up_scale", "down", "down_scale"]`.
    Fp8Block,
    /// MXFP4 (Microscaling FP4). Banks: `["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales"]`.
    MxFp4,
    /// NVFP4 (Nvidia FP4). Banks: `["gate_up_packed", "gate_up_scale", "down_packed", "down_scale"]`.
    NvFp4,
    /// GGUF Q4_0 block quantization. Banks: `["gate_up", "down"]`.
    Q4_0,
}

impl FtwQuantFormat {
    /// Return the required bank schema names in canonical registration order.
    pub fn bank_names(&self) -> &'static [&'static str] {
        match self {
            Self::Bf16 | Self::Q4_0 => &["gate_up", "down"],
            Self::Fp8Block => &["gate_up", "gate_up_scale", "down", "down_scale"],
            Self::MxFp4 => &[
                "gate_up_blocks",
                "gate_up_scales",
                "down_blocks",
                "down_scales",
            ],
            Self::NvFp4 => &[
                "gate_up_packed",
                "gate_up_scale",
                "down_packed",
                "down_scale",
            ],
        }
    }
}

/// Metadata header describing an FTW file or host bank layout.
#[derive(Debug, Clone)]
pub struct FtwHeader {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Total routed experts per MoE layer.
    pub num_experts: usize,
    /// Hidden dimension of the model.
    pub hidden_dim: usize,
    /// Intermediate / feed-forward dimension per expert.
    pub inter_dim: usize,
    /// Quantization format.
    pub quant_format: FtwQuantFormat,
    /// Byte size per expert row for each named bank.
    pub bank_row_bytes: HashMap<String, usize>,
}

impl FtwHeader {
    /// Create a new FTW header and calculate canonical bank byte dimensions.
    ///
    /// # Contract
    /// Dimensions must be non-zero.
    pub fn new(
        num_layers: usize,
        num_experts: usize,
        hidden_dim: usize,
        inter_dim: usize,
        quant_format: FtwQuantFormat,
    ) -> Self {
        let mut bank_row_bytes = HashMap::new();
        match quant_format {
            FtwQuantFormat::Bf16 => {
                // gate_up: [2 * inter_dim, hidden_dim] in bf16 (2 bytes)
                bank_row_bytes.insert("gate_up".to_string(), 2 * inter_dim * hidden_dim * 2);
                // down: [hidden_dim, inter_dim] in bf16
                bank_row_bytes.insert("down".to_string(), hidden_dim * inter_dim * 2);
            }
            FtwQuantFormat::Fp8Block => {
                // gate_up: 1 byte per element
                bank_row_bytes.insert("gate_up".to_string(), 2 * inter_dim * hidden_dim);
                // gate_up_scale: 1 scale per 128x128 block in bf16
                let num_blocks_gu = (2 * inter_dim).div_ceil(128) * hidden_dim.div_ceil(128);
                bank_row_bytes.insert("gate_up_scale".to_string(), num_blocks_gu * 2);
                // down: 1 byte per element
                bank_row_bytes.insert("down".to_string(), hidden_dim * inter_dim);
                let num_blocks_down = hidden_dim.div_ceil(128) * inter_dim.div_ceil(128);
                bank_row_bytes.insert("down_scale".to_string(), num_blocks_down * 2);
            }
            FtwQuantFormat::MxFp4 | FtwQuantFormat::NvFp4 => {
                // 4-bit packed weights: 0.5 bytes per element
                bank_row_bytes.insert(
                    quant_format.bank_names()[0].to_string(),
                    inter_dim * hidden_dim,
                );
                bank_row_bytes.insert(
                    quant_format.bank_names()[1].to_string(),
                    (2 * inter_dim * hidden_dim) / 32,
                );
                bank_row_bytes.insert(
                    quant_format.bank_names()[2].to_string(),
                    (hidden_dim * inter_dim) / 2,
                );
                bank_row_bytes.insert(
                    quant_format.bank_names()[3].to_string(),
                    (hidden_dim * inter_dim) / 32,
                );
            }
            FtwQuantFormat::Q4_0 => {
                // GGUF Q4_0: 18 bytes per 32 elements
                let gu_blocks = (2 * inter_dim * hidden_dim).div_ceil(32);
                bank_row_bytes.insert("gate_up".to_string(), gu_blocks * 18);
                let down_blocks = (hidden_dim * inter_dim).div_ceil(32);
                bank_row_bytes.insert("down".to_string(), down_blocks * 18);
            }
        }

        Self {
            num_layers,
            num_experts,
            hidden_dim,
            inter_dim,
            quant_format,
            bank_row_bytes,
        }
    }

    /// Calculate the total number of flattened expert rows across the model ($L \times E$).
    pub fn total_expert_rows(&self) -> usize {
        self.num_layers * self.num_experts
    }

    /// Calculate the byte offset for an expert `(layer_idx, expert_idx)` within a named bank.
    ///
    /// # Contract
    /// `layer_idx < num_layers` and `expert_idx < num_experts`.
    pub fn expert_byte_offset(
        &self,
        bank_name: &str,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<usize> {
        if layer_idx >= self.num_layers || expert_idx >= self.num_experts {
            return Err(Error::Backend(format!(
                "FtwHeader: expert ({layer_idx}, {expert_idx}) out of range (layers: {}, experts: {})",
                self.num_layers, self.num_experts
            )));
        }
        let row_bytes = self
            .bank_row_bytes
            .get(bank_name)
            .ok_or_else(|| Error::Backend(format!("FtwHeader: unknown bank name '{bank_name}'")))?;
        let flat_idx = layer_idx * self.num_experts + expert_idx;
        Ok(flat_idx * row_bytes)
    }
}

/// Contiguous host DRAM bank holding flat expert rows with post-load page pinning.
pub struct FtwHostBank {
    /// Bank identifier name (e.g., `"gate_up"`, `"down"`).
    pub name: String,
    /// Host buffer memory.
    pub data: Vec<u8>,
    /// Whether host memory has been locked/pinned against paging.
    pub is_pinned: bool,
}

impl FtwHostBank {
    /// Allocate an exact-sized unpinned host bank buffer.
    pub fn allocate(name: &str, total_bytes: usize) -> Self {
        Self {
            name: name.to_string(),
            data: vec![0u8; total_bytes],
            is_pinned: false,
        }
    }

    /// Pin the populated memory pages using `libc::mlock` to accelerate PCIe DMA transfers.
    ///
    /// # Safety & FFI Contract
    /// Complies with `rust-ffi-grim`: non-null buffer pointer and bounded size check.
    /// Gracefully ignores failure if OS user resource limits (RLIMIT_MEMLOCK) deny locking.
    pub fn pin_memory(&mut self) -> bool {
        if self.data.is_empty() || self.is_pinned {
            return self.is_pinned;
        }

        #[cfg(target_os = "linux")]
        {
            let ptr = self.data.as_mut_ptr() as *mut libc::c_void;
            let len = self.data.len();
            if !ptr.is_null() {
                let ret = unsafe { libc::mlock(ptr, len) };
                if ret == 0 {
                    self.is_pinned = true;
                    return true;
                }
            }
        }

        false
    }
}

/// Direct I/O fast bootstrap loader for FTW format.
pub struct FtwDirectLoader {
    pub header: FtwHeader,
    pub banks: HashMap<String, FtwHostBank>,
}

impl FtwDirectLoader {
    /// Initialize loader and pre-allocate contiguous host banks.
    pub fn new(header: FtwHeader) -> Self {
        let mut banks = HashMap::new();
        let total_rows = header.total_expert_rows();

        for (name, &row_bytes) in &header.bank_row_bytes {
            let total_bytes = total_rows * row_bytes;
            banks.insert(name.clone(), FtwHostBank::allocate(name, total_bytes));
        }

        Self { header, banks }
    }

    /// Load bank data directly from an I/O source into pre-allocated memory.
    pub fn load_bank<R: std::io::Read>(&mut self, name: &str, mut reader: R) -> Result<usize> {
        let bank = self.banks.get_mut(name).ok_or_else(|| {
            Error::Backend(format!("FtwDirectLoader: bank '{name}' not configured"))
        })?;

        reader.read_exact(&mut bank.data).map_err(|e| {
            Error::Backend(format!(
                "FtwDirectLoader: failed to read bank '{name}': {e}"
            ))
        })?;

        Ok(bank.data.len())
    }

    /// Post-load pin all populated host banks.
    ///
    /// # Contract
    /// Called after all weights are written to disk layout to avoid initial zero-fault stalls.
    pub fn pin_all_banks(&mut self) -> usize {
        let mut pinned_count = 0;
        for bank in self.banks.values_mut() {
            if bank.pin_memory() {
                pinned_count += 1;
            }
        }
        pinned_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ftw_header_bf16_offsets() {
        let header = FtwHeader::new(32, 64, 4096, 14336, FtwQuantFormat::Bf16);
        assert_eq!(header.total_expert_rows(), 2048);

        // Check offset for layer 1, expert 0 -> flat index 64
        let offset = header.expert_byte_offset("gate_up", 1, 0).unwrap();
        let row_bytes = *header.bank_row_bytes.get("gate_up").unwrap();
        assert_eq!(offset, 64 * row_bytes);

        // Check out of range
        assert!(header.expert_byte_offset("gate_up", 32, 0).is_err());
        assert!(header.expert_byte_offset("gate_up", 0, 64).is_err());
    }

    #[test]
    fn test_ftw_bank_schemas() {
        let fmt = FtwQuantFormat::NvFp4;
        assert_eq!(fmt.bank_names().len(), 4);
        assert_eq!(fmt.bank_names()[0], "gate_up_packed");
    }

    #[test]
    fn test_ftw_direct_loader_and_pinning() {
        let header = FtwHeader::new(2, 2, 8, 16, FtwQuantFormat::Bf16);
        let mut loader = FtwDirectLoader::new(header);

        let gu_bytes = loader.banks.get("gate_up").unwrap().data.len();
        let fake_data = vec![42u8; gu_bytes];

        let loaded = loader.load_bank("gate_up", &fake_data[..]).unwrap();
        assert_eq!(loaded, gu_bytes);
        assert_eq!(loader.banks.get("gate_up").unwrap().data[0], 42);

        // Test pinning invocation
        let _ = loader.pin_all_banks();
    }
}

//! Apple Neural Engine (ANE) Module.
//!
//! Handles generation of Model Intermediate Language (MIL) sub-graphs and compilation
//! of mega-kernels for ANE hardware execution.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use grim_tensor::error::{Error, Result};

/// Represents a compiled Neural Engine program.
pub struct AneProgram {
    pub path: String,
}

/// Generates MIL specifications for fused operations to run on the ANE.
pub struct AneGraphBuilder {
    name: String,
    nodes: Vec<String>,
}

impl AneGraphBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
        }
    }

    pub fn push_matmul(&mut self, out: &str, a: &str, b: &str) {
        self.nodes.push(format!("    %{} = mil.matmul(a=%{}, b=%{})", out, a, b));
    }

    pub fn push_silu(&mut self, out: &str, input: &str) {
        self.nodes.push(format!("    %{} = mil.silu(x=%{})", out, input));
    }

    pub fn push_mul(&mut self, out: &str, a: &str, b: &str) {
        self.nodes.push(format!("    %{} = mil.mul(x=%{}, y=%{})", out, a, b));
    }

    /// Serialises the model graph to MIL format for compiler consumption.
    pub fn compile_mil(&self, path: &Path) -> Result<AneProgram> {
        let mut file = File::create(path).map_err(|e| Error::Backend(e.to_string()))?;
        writeln!(file, "program {} {{", self.name).map_err(|e| Error::Backend(e.to_string()))?;
        for node in &self.nodes {
            writeln!(file, "{}", node).map_err(|e| Error::Backend(e.to_string()))?;
        }
        writeln!(file, "}}").map_err(|e| Error::Backend(e.to_string()))?;

        Ok(AneProgram {
            path: path.to_string_lossy().into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_fused_ffn_mil_generation() {
        let dir = tempdir().unwrap();
        let mil_file = dir.path().join("ffn.mil");
        let mut builder = AneGraphBuilder::new("fused_ffn");
        builder.push_matmul("h_gate", "x", "w_gate");
        builder.push_silu("h_silu", "h_gate");
        builder.compile_mil(&mil_file).unwrap();
        assert!(mil_file.exists());
    }
}
